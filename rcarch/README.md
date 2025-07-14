# RCA: Resumable Compressed Archive

This library implements a file format for storing named blobs with resumable appending and efficient compression across similar blobs.  
The format is called RCA (Resumable Compressed Archive) and  it's designed to be robust even after crashes during blob writes.

---

## Features

- Supports reopening and appending to existing archives with a fresh compressor state, allowing safe continuation after crashes.
- Uses Zstandard (zstd) compression for blobs.
- Maintains the compressor state across blobs, improving compression efficiency when blobs name or content is similar.

---

## Classes in this module

- `Writer` appends blobs to an RCA archive
- `Reader` iterates over blobs in an RCA archive
- `Chunker` writes the outer chunk layer format
- `Unchunker` reads the outer chunk layer format
- `ZstdCompressor` compresses blobs using zstd
- `ZstdDecompressor` decompresses zstd-compressed blob streams

---

# File format

The format has two layers:

1. Outer layer (Chunk Layer):  
   Stores sequential chunks with fixed-size metadata. Used for safe appends and integrity checking.

2. Inner layer (Blob Layer):  
   Encapsulated inside chunk payloads. Stores compressed blobs with optional reset points.

---

## Outer Layer Format (Chunk Layer)

The file is a sequence of *chunks*, each chunk written as:
  - First, a *size field*, a big-endian unsigned integer.  
    - The number of bits of this field doubles each time it appears (16 bits → 32 bits → 64 bits …), enabling exponential file growth.
    - Maximum allowed value: `1 << (number of bits - 1)`.  
      Larger values are reserved for future use.
      If a chunk is shorter than that, then it's the last chunk, anything after is garbage.
    - It represents the number of bytes of the chunk, including the size field bytes.
    - Size = `0` means the chunk is incomplete (write was interrupted), allowing later appends to safely overwrite it.  Readers should treat 0 size chunks as EOF.
  - Then, a fixed-size *metadata* field, whose semantics are opaque to the outer layer.  The inner layer uses it to store a checksum.
    Only the last metadata of a chunk with a non-zero size field is relevant, previous metadata fields must be ignored.
  - Finally, the chunk *payload bytes*.

The payload of the last chunk may be followed by trailing garbage if an append was interrupted.  Readers should ignore that garbage, and writers should overwrite it.

---

## Example 1: Chunk layer file with three valid chunks

```
+---------------------------------------------------+
| Chunk 1 (0x800 bytes)                             |
| Size (16 bits) = \x80\x00                         |
| Metadata (irrelevant)                             |
| Payload                                           |
+---------------------------------------------------+
| Chunk 2 (0x80000000 bytes)                        |
| Size (32 bits) = \x80\x00\x00\x00                 |
| Metadata (irrelevant)                             |
| Payload                                           |
+---------------------------------------------------+
| Chunk 3 (0xf3 bytes)                              |
| Size (64 bits) = \x00\x00\x00\x00\x00\x00\x00\xf3 |
| Metadata (relevant)                               |
| Payload                                           |
+---------------------------------------------------+

```

---

## Example 2: Chunk layer file with an interrupted append

This example assumes that chunk 1 was written until it reached 0x8000 bytes, then a new append of a large amount of data, requiring to create two new chunks, was started, but interrupted mid-way.

Such a write at a chunk boundary first creates a new chunk header with a zero size, then writes payload bytes until this second chunk is full, then creates the third chunk header with its correct size, and adds the payload.

The example assumes the write is interrupted at this point.  If it weren't, it would finally replace the second chunk zero size with the correct value (0x80000000).
```
+--------------------------------------------------------------------+
| Chunk 1 (0x8000 bytes)                                             |
| Size (16 bits) = \x80\x00                                          |
| Metadata (relevant)                                                |
| Payload                                                            |
+--------------------------------------------------------------------+
| Chunk 2 (0x80000000 bytes), all garbage                            |
| Size (32 bits) = \x00\x00\x00\x00                                  |
| Metadata (irrelevant)                                              |
| Payload                                                            |
+--------------------------------------------------------------------+
| Chunk 3 (0x5678 bytes), all garbage                                |
| Size (64 bits) = \x00\x00\x00\x00\x00\x00\x56\x78                  |
| Metadata (irrelevant)                                              |
| Payload                                                            |
+--------------------------------------------------------------------+
```

---

## Inner Layer Format (Blob Layer)

The inner layer is written as payload of the outer layer, storing compressed blobs and optional reset points.  It uses the outer layer with a 16 bits initial size field, and the metadata is a 64 bits blake2s hash.

The data is a sequence of *blocks*, each starting with a little-endian varint, where each byte MSB (bit 7) is set to zero only for the last last byte. The LSB (bit 0) of the varint determines the block type:  `0` = blob block, `1` = control block.

Blob blocks are just zstd compressed data, with no checksums nor flush.  Their size is the value of the varint >> 1.  The compressed data is the name of the blob as a 0 terminated utf-8 string, followed by the blob content.

For control blocks, bits 1-5 of the varint represent the block type, and its remaining bits are a payload size in bytes.  Readers should skip control blocks whose type is unknown to them.

### Reset blocks (block type 0)

Reset blocks signal a reset of the hashing and of the zstd compression state.

The payload of reset blocks starts with a 64 bits blake2s hash, and any additional payload bytes should be ignored.

The hash covers all inner layer bytes except hash bytes, from the start of the file, or from the previous reset block's varint if there is such a block, up to but not including the varint of the current reset block.

The outer layer metadata acts as a substitute for a hash that would otherwise have to be stored in a final trailing reset block.  It's the hash of the inner layer data except hash bytes since the last reset block (or from the start of the file if there are no resets) up to the end of the inner bytes.

---

### Example 1: Inner layer file with no reset blocks

This example shows the inner layer data for three blobs names foo bar and baz.

```

+-------------------------------------------------------------------------------------+
| Varint: \x42 = 0x42 = 66                                                            |
| (66 >> 1 = 33 bytes payload Zstd("foo\x00..."), type=blob)                          |
+-------------------------------------------------------------------------------------+
| Varint: \xf2\x04 = (0xf2 & 0x7f) + (0x04 << 7) = 626                                |
| (626 >> 1 = 313 bytes payload Zstd("bar\x00..."), type=blob)                        |
+-------------------------------------------------------------------------------------+
| Varint: \x84\xa3\x06 = (0x84 & 0x7f) + ((0xa3 & 0x7f) << 7) + (0x06 << 14) = 102788 |
| (102788 >> 51394 bytes payload Zstd("baz\x00..."), type=blob)                       |
+-------------------------------------------------------------------------------------+
```

The hash of all these inner layer bytes is stored as outer layer metadata.

---

### Example 2: Inner layer file with two reset blocks

```
+-----------------------------------------------------------------------------------------------------+
| Varint: \x42                                                                                        |
| (blob, 0x42 >> 1 bytes payload)                                                                     |
| Zstd("foo\0...")                                                                                    |
+-----------------------------------------------------------------------------------------------------+
| Varint: \x3a                                                                                        |
| (blob, 0x3a >> 1 bytes payload)                                                                     |
| Zstd("bar\0...")                                                                                    |
+-----------------------------------------------------------------------------------------------------+
| Varint: \x81\x02 = 0x0101                                                                           |
| (control, type = (0x0101 & ((1 << 6) - 1)) = 0, payload size = 0x0101 >> 5 = 8                      |
| Payload: 8-byte BLAKE2s hash of the two previous blob blocks                                        |
+-----------------------------------------------------------------------------------------------------+
| Varint: \x86\x40                                                                                    |
| (blob, 0x2006 >> 1 bytes payload)                                                                   |
| Zstd("baz\0...")                                                                                    |
+-----------------------------------------------------------------------------------------------------+
| Varint: \x81\x02 = 0x0101                                                                           |
| (control, type = (0x0101 & ((1 << 6) - 1)) = 0, payload size = 0x0101 >> 5 = 8                      |
| Payload: 8-byte BLAKE2s hash of the previous control block except its hash bytes and blob block baz |
+-----------------------------------------------------------------------------------------------------+
| Varint: \x6a                                                                                        |
| (blob, 0x6a >> 1 bytes payload)                                                                     |
| Zstd("foobar\0...")                                                                                 |
+-----------------------------------------------------------------------------------------------------+
```

The final outer layer metadata hashes everything from the varint of the last reset block to the end, except the hash bytes of the reset block.
