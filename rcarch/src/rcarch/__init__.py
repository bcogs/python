from concurrent import futures
import hashlib
import io
import os
import socket
import subprocess


def open_or_create(file_path):
    "open a file in binary mode, creating it if needed, with file pointer at offset 0"
    return os.fdopen(os.open(file_path, os.O_RDWR | os.O_CREAT, 0o666), "r+b")


class _ZstdStreamRunner(object):
    "base class enabling its children to run and use the external zstd_stream executable"

    def __init__(self, command):
        self._bound_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bound_sock.bind(("localhost", 0))
        self._bound_sock.listen(1)
        self._bound_sock_addr = self._bound_sock.getsockname()
        fd = self._bound_sock.fileno()
        os.set_inheritable(fd, True)
        # yes, the bound socket is passed on the stdout file descriptor, the
        # child knows it and won't write to it
        self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=fd, pass_fds=(fd,))

    def send_message_and_read_reply(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self._bound_sock_addr)
            with s.makefile("rb") as s_as_file:
                with futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(s_as_file.read)
                    try:
                        s.sendall(data)
                    finally:
                        s.shutdown(socket.SHUT_WR)
                    return future.result()

    def stop(self):
        self._proc.stdin.close()
        self._bound_sock.close()
        self._proc.wait()


class ZstdCompressor(object):
    "compressor for series of bytes() in zstd format that never resets the compression state"

    def __init__(self):
        "constructor returning a context manager"
        path = os.environ.get("RCARCH_ZSTD_STREAM")
        if path:
            self._runner = _ZstdStreamRunner((path,))
        else:
            import zstandard

            self._compressed = io.BytesIO()
            self._writer = zstandard.ZstdCompressor(level=3, write_checksum=False).stream_writer(self._compressed)
            self._FLUSH_BLOCK = zstandard.FLUSH_BLOCK

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def compress(self, data: bytes) -> bytes:
        "compress a bytes() and return all the bytes needed to decompress it"
        if hasattr(self, "_runner"):
            return self._runner.send_message_and_read_reply(data)
        self._writer.write(data)
        self._writer.flush(self._FLUSH_BLOCK)
        data = self._compressed.getvalue()
        self._compressed.seek(0)
        self._compressed.truncate(0)
        return data

    def close(self):
        "free resources, this must be called when not using the class as a context manager"
        self._runner.stop() if hasattr(self, "_runner") else self._writer.close()


class ZstdDecompressor(object):
    "decompressor for series of byte()s written by ZstdCompressor"

    def __init__(self):
        "constructor returning a context manager"
        path = os.environ.get("RCARCH_ZSTD_STREAM")
        if path:
            self._runner = _ZstdStreamRunner((path, "--decompress"))
        else:
            from zstandard import ZstdDecompressor

            self._reader = ZstdDecompressor().decompressobj()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def decompress(self, data: bytes) -> bytes:
        return (
            self._runner.send_message_and_read_reply(data)
            if hasattr(self, "_runner")
            else self._reader.decompress(data)
        )

    def close(self):
        "free resources, this must be called when not using the class as a context manager"
        self._runner.stop() if hasattr(self, "_runner") else None


def _read(file, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        b = file.read(size - len(buf))
        if not b:
            break
        buf.extend(b)
    return bytes(buf)


def _write(file, data: bytes) -> int:
    wrote = 0
    while wrote < len(data):
        n = file.write(data[wrote:])
        if not n:
            raise IOError("write returned " + str(n))
        wrote += n
    return wrote


def flush_file(file):
    "call file.flush() and os.fsync(file.fileno())"
    file.flush()
    os.fsync(file.fileno())


class _ChunkHandler(object):
    "base class for Chunker and Unchunker"

    def __init__(self, file, meta_len, first_size_bits):
        self._meta_len, self._file = meta_len, file
        self._size_bits = first_size_bits
        if self._max_len() <= self._size_len() + meta_len:
            raise ValueError(
                "first_size_bits is too low to accomodate the first header with its metadata and first data bytes"
            )

    def _max_len(self):  # max size of chunk header + its payload
        return 1 << (self._size_bits - 1)

    def _size_len(self) -> int:
        return (self._size_bits + 7) // 8


class CorruptArchive(Exception):
    pass


class Chunker(_ChunkHandler):
    """
    Append-only writer for outer file format.

    This class manages appending data and writing metadata to outer layer files.
    The metadata is usually a checksum, but it can be anything the user of the
    class wants, the only constraint is to keep its size constant.

    Attributes:
        meta (bytes | None): The metadata provided with the last append.

    Constructor Args:
        file: A seekable file-like object opened in read+write mode, with the
              file pointer currently at the beginning of the file.
        meta_len: Size in bytes of the metadata provided when appending chunks.
        first_size_bits: Number of bits used for the initial size field.
                         Must be large enough to represent the size of the first
                         chunk header plus one byte of payload:
                         1 << (first_size_bits - 1) must be greater than
                         (first_size_bits + 7) // 8 + meta_len
    """

    def __init__(self, file, meta_len, first_size_bits):
        if not file.seekable():
            raise ValueError("Chunker requires a seekable file")
        super().__init__(file, meta_len, first_size_bits)
        self.meta = None
        self._offset = 0
        while True:
            self._header_offset = self._offset
            # read a chunk header: a big endian integer plus the fixed size metadata
            size_len = self._size_len()  # size of the integer
            n = size_len + meta_len  # size of the header
            d = _read(file, n)
            if len(d) < n:  # eof
                if d:  # last header was partially written, it's garbage
                    self._file.seek(self._offset)
                return
            self.meta = d[size_len:]
            size = int.from_bytes(d[:size_len], "big")  # size of the chunk, including header
            if size < len(d):
                if size:
                    raise CorruptArchive("invalid chunk size " + str(size))
                # the last append didn't fully complete
            self._offset = self._file.seek(size - len(d), os.SEEK_CUR)
            if size < self._max_len():  # it's the last chunk
                return
            self._size_bits *= 2

    def append(self, data: bytes, meta: bytes, flush=flush_file):
        """append arbitrary data and replace the metadata

        Args:
          data: the payload bytes to append
          metadata: the value of the metadata to set if this payload is written
                    successfully
          flush: a function that receives the file in argument and flushes it to disk (the default calls file.flush() and os.fsync(file.fileno()), or None for no flushing
        """
        if len(meta) != self._meta_len:
            raise ValueError("metadata has invalid len " + str(len(meta)))
        max_len = self._max_len()

        first_header_offset = self._header_offset
        if first_header_offset >= self._offset:
            self._header_offset = self._offset
            self._offset += _write(self._file, self._make_header(0, meta))
        chunk_len = self._offset - first_header_offset  # current chunk len
        chunk_remaining = max_len - chunk_len
        if len(data) <= chunk_remaining:
            first_header = self._make_header(chunk_len + len(data), meta)
        else:
            self._offset += _write(self._file, data[:chunk_remaining])
            first_header, data = self._make_header(max_len, meta), data[chunk_remaining:]
            while True:
                self._size_bits *= 2
                max_len, chunk_len = self._max_len(), (self._size_bits + 7) // 8 + len(meta) + len(data)
                if chunk_len <= max_len:
                    self._header_offset = self._offset
                    self._offset += _write(self._file, self._make_header(chunk_len, meta))
                    break
                chunk_remaining = max_len - _write(self._file, self._make_header(max_len, meta))
                _write(self._file, data[:chunk_remaining])
                self._offset += max_len
                data = data[chunk_remaining:]
        self._offset += _write(self._file, data)
        if flush:
            flush(self._file)
        self._file.seek(first_header_offset)
        _write(self._file, first_header)
        if flush:
            flush(self._file)
        self._file.seek(self._offset)
        self.meta = meta
        return

    def _make_header(self, size, meta: bytes) -> bytes:
        return size.to_bytes(self._size_len(), "big") + meta


class Unchunker(_ChunkHandler):
    """
    Sequential reader for the outer file format written by the Chunker class.

    Attributes (mustn't be accessed before the end of the data is reached!):
        meta (bytes)|None: The latest read metadata.
        trailing_garbage (bool):
            This attribute is set only at the end of the iteration.  It
            indicates whether there's garbage after the last chunk, which
            typically happens if an append didn't fully complete.

    Constructor Args:
        file: A readable file-like object, pointing to the beginning of the file.
        meta_len: Size in bytes of the metadata.
        first_size_bits: Number of bits used for the initial size field.
    """

    def __init__(self, file, meta_len, first_size_bits):
        super().__init__(file, meta_len, first_size_bits)
        self._last, self.meta, self._remaining = False, None, 0

    def read(self, size: int) -> bytes:
        "read size bytes, or less if and only if eof is reached before"
        out = bytearray()
        while size > 0:
            if self._remaining <= 0:  # we're at a chunk boundary
                if hasattr(self, "trailing_garbage"):
                    break  # already reached the end
                if self._last:  # the current chunk is the last one
                    self.trailing_garbage = bool(self._file.read(1))
                    break
                if not self._read_header():
                    break
            n = min(size, self._remaining)
            b = _read(self._file, n)
            if not b:
                raise CorruptArchive("unexpected EOF while reading chunk payload")
            out.extend(b)
            size -= len(b)
            self._remaining -= len(b)
        return bytes(out)

    def _read_header(self) -> bool:  # header = big endian integer + optional payload + fixed size metadata
        size_len = self._size_len()  # size of the integer
        n = size_len + self._meta_len  # size of the header
        d = _read(self._file, n)
        if len(d) < n:  # eof or incomplete header
            self.trailing_garbage = bool(d)
            return False
        size = int.from_bytes(d[:size_len], "big")  # size of the chunk, including header
        if size < len(d):
            if not size:  # last chunk wasn't fully written or is empty
                self.trailing_garbage = True
                return False
            raise CorruptArchive("invalid chunk size " + str(size))
        self._last = size < self._max_len()
        self.meta, self._remaining = d[size_len:], size - len(d)
        self._size_bits *= 2
        if not self._remaining:
            self.trailing_garbage = bool(self._file.read(1))
            return False
        return True


_FIRST_BITS = 16

_CONTROL_BLOCK_FLAG = 1
_RESET = 0
_CONTROL_BLOCK_LOW_BITS = 6


_HASH_LEN = 8


# to debug hashing issues
# class h(object):
#
#  def __init__(self):
#      print("new hash")
#      self.h = hashlib.blake2s(digest_size=_HASH_LEN)
#
#  def update(self, d):
#      self.h.update(d)
#      print("update %r -> %s" % (d, self.h.digest().hex()))
#
#  def digest(self):
#      return self.h.digest()


def _new_hash():
    # return h()  # to debug hashing issues
    return hashlib.blake2s(digest_size=_HASH_LEN)


class Writer(object):
    "append functionality for rca files"

    def __init__(self, file):
        "constructor returning a context manager"
        self._chunker, self._hash = Chunker(file, _HASH_LEN, _FIRST_BITS), _new_hash()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def append(self, blob_name: str, blob: bytes, flush=flush_file):
        """append a blob fully

        The file temporarily has trailing garbage during a call to append, so
        it's nicer to defer signals or other sources of interruption during the
        call.
        If the file was well formed before the call and the call doesn't return
        an exception, then the file is always well formed after the call.
        Trailing garbage isn't an issue, but avoidinng it is nicer.

        Args:
          blob_name: the name of the blob
          blob: the content of the blob
          flush: a function that receives the file in argument and flushes it to disk (the default calls file.flush() and os.fsync(file.fileno()), or None for no flushing

        Raises:
          CorruptArchive in certain cases of file corruption
        """
        prepend = b""
        if not hasattr(self, "_compressor"):
            self._compressor = ZstdCompressor()
            if self._chunker.meta:  # the file already contains data
                prepend = self._make_control_varint(_RESET, len(self._chunker.meta)) + self._chunker.meta
                self._hash.update(prepend)
        compressed = self._compressor.compress(blob_name.encode("utf-8") + b"\x00" + blob)
        n = self._make_varint(len(compressed) << 1)
        self._hash.update(n)
        self._hash.update(compressed)
        self._chunker.append(prepend + n + compressed, self._hash.digest(), flush=flush)

    def close(self):
        "free resources, this must be called when not using the class as a context manager"
        if hasattr(self, "_compressor"):
            self._compressor.close()

    def _make_control_varint(self, block_type: int, payload_len: int) -> bytes:
        return self._make_varint((payload_len << _CONTROL_BLOCK_LOW_BITS) | (block_type << 1) | _CONTROL_BLOCK_FLAG)

    def _make_varint(self, i: int) -> bytes:
        buf = bytearray()
        while i >= 0x80:
            buf.append((i & 0x7F) | 0x80)
            i >>= 7
        buf.append(i)
        return bytes(buf)


class Reader(object):
    "iterator (and context manager) that reads an rca file"

    def __init__(self, file):  # file should be open in "rb" mode
        "constructor returning a context manager"
        self._unchunker = Unchunker(file, _HASH_LEN, _FIRST_BITS)

    def __iter__(self):
        "yield a (blob_name: str, blob_content: bytes) pair"
        self._reset()
        digest = None
        while True:
            # read the varint
            d = self._unchunker.read(1)
            if len(d) < 1:  # clean eof
                self._handle_legitimate_eof()
                return
            i = d[0]
            if i & _CONTROL_BLOCK_FLAG:
                block_type = (i & ((1 << _CONTROL_BLOCK_LOW_BITS) - 1)) // 2
                if block_type == _RESET:
                    digest = self._hash.digest()
                    self._reset()
            self._hash.update(d)
            n, shift = (i & 0x7F), 7
            while i & 0x80:
                d = self._unchunker.read(1)
                if len(d) < 1:
                    raise CorruptArchive("unexpected EOF while reading varint")
                self._hash.update(d)
                i = d[0]
                n |= (i & 0x7F) << shift
                shift += 7
            if n & _CONTROL_BLOCK_FLAG:
                self._process_control_block(block_type, n, digest)
            else:
                yield self._read_blob(n >> 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        "free resources, this must be called when not using the class as a context manager"
        if hasattr(self, "_decompressor"):
            self._decompressor.close()

    def _handle_legitimate_eof(self):
        if self._unchunker.meta and self._unchunker.meta != self._hash.digest():
            raise CorruptArchive("final checksum mismatch")
        self.trailing_garbage = self._unchunker.trailing_garbage

    def _process_control_block(self, block_type: int, n: int, digest):
        payload_size = n >> _CONTROL_BLOCK_LOW_BITS
        if payload_size:
            payload = self._unchunker.read(payload_size)
            if len(payload) >= payload_size:
                self._hash.update(payload)
            else:
                raise CorruptArchive("unexpected EOF while reading control block payload")
        if block_type == _RESET:
            if len(payload) < _HASH_LEN:
                raise CorruptArchive("reset block payload is shorter than hash len")
            if payload[:_HASH_LEN] != digest:
                raise CorruptArchive(
                    "checksum mismatch"
                    if len(payload) >= _HASH_LEN
                    else "unexpected EOF while reading reset block checksum"
                )

    def _read_blob(self, size: int) -> tuple[str, bytes]:
        compressed = self._unchunker.read(size)
        if len(compressed) < size:
            raise CorruptArchive("unexpected EOF when reading compressed blob")
        self._hash.update(compressed)
        decompressed = self._decompressor.decompress(compressed)
        try:
            name, blob = decompressed.split(b"\x00", 1)
        except ValueError:
            raise CorruptArchive("no \\x00 found in blob name")
        return name.decode("utf-8"), blob

    def _reset(self):
        if hasattr(self, "_decompressor"):
            self._decompressor.close()
        self._decompressor, self._hash = ZstdDecompressor(), _new_hash()
