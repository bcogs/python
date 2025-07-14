import io
import os
import random
import tempfile
import unittest

try:
    import rcarch
except ModuleNotFoundError:
    import __init__ as rcarch


class BytesIO(io.BytesIO):
    def __init__(self):
        self.os_fsyncs = 0


def flush_bytesio(bytesio):
    bytesio.os_fsyncs += 1


class ChunkLayerTest(unittest.TestCase):
    _DUMMY_META = b"test1meta"
    _DUMMY_META2 = b"meta2test"
    _META_LEN = len(_DUMMY_META)

    def _write_chunks(self, parts, first_size_bits, file=None, meta_len=_META_LEN):
        if not file:
            file = BytesIO()
        else:
            file.seek(0)
        n = file.os_fsyncs
        c = rcarch.Chunker(file, meta_len, first_size_bits)
        for data, meta in parts:
            c.append(data, meta, flush=flush_bytesio)
            n += 2
            self.assertEqual(n, file.os_fsyncs)
        if parts:
            self.assertEqual(meta, c.meta)
        else:
            self.assertIsNone(c.meta)
        return file

    def _read_all(self, f, first_size_bits, read_size=10, meta_len=_META_LEN):
        f.seek(0)
        u = rcarch.Unchunker(f, meta_len, first_size_bits)
        result = bytearray()
        has_trailing_garbage_attr = hasattr(u, "trailing_garbage")
        while True:
            d = u.read(read_size)
            if d:
                self.assertFalse(has_trailing_garbage_attr)
            else:
                meta, trailing_garbage = u.meta, u.trailing_garbage
                self.assertEqual(b"", u.read(1))
                self.assertEqual(meta, u.meta)
                self.assertEqual(trailing_garbage, u.trailing_garbage)
                return bytes(result), u.meta, u.trailing_garbage
            result.extend(d)
            has_trailing_garbage_attr = hasattr(u, "trailing_garbage")

    def test_zero_len_initial_write(self):
        f = self._write_chunks([(b"", self._DUMMY_META)], 16)
        self.assertEqual((b"", self._DUMMY_META, False), self._read_all(f, 16))
        self._write_chunks([(b"foo", self._DUMMY_META2)], 16, file=f)
        self.assertEqual((b"foo", self._DUMMY_META2, False), self._read_all(f, 16))

    def test_zero_len_non_initial_write(self):
        f = self._write_chunks([(b"foo", self._DUMMY_META2), (b"", self._DUMMY_META)], 16)
        self.assertEqual((b"foo", self._DUMMY_META, False), self._read_all(f, 16))
        self._write_chunks([(b"bar", self._DUMMY_META2)], 16, file=f)
        self.assertEqual((b"foobar", self._DUMMY_META2, False), self._read_all(f, 16))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            rcarch.Unchunker(BytesIO(), 8, 4)
        with self.assertRaises(ValueError):
            rcarch.Chunker(BytesIO(), 8, 4)

        class NonSeekable(BytesIO):
            def seekable(self):
                return False

        with self.assertRaises(ValueError):
            rcarch.Chunker(NonSeekable(), 8, 24)

    def test_empty_file(self):
        self.assertEqual((b"", None, False), self._read_all(BytesIO(), 24))
        f = self._write_chunks([], 8)
        f.seek(0, os.SEEK_END)
        self.assertEqual(0, f.tell())

    def test_single_byte_chunk(self):
        f = self._write_chunks([(b"A", self._DUMMY_META)], 24)
        self.assertEqual((b"A", self._DUMMY_META, False), self._read_all(f, 24))

    def test_two_appends(self):
        parts = [(b"abc", self._DUMMY_META), (b"defg", self._DUMMY_META2)]
        f = self._write_chunks(parts, 16)
        self.assertEqual((b"abcdefg", self._DUMMY_META2, False), self._read_all(f, 16))

    def test_multiple_chunks(self):
        parts, n, file_len, chunks = [], 5, 0, 0
        while chunks < 4:
            parts.append(
                (
                    bytes([b"abcdefghijklmnopqrstuvwxyz"[len(parts)]]) * n,
                    self._DUMMY_META if len(parts) % 2 else self._DUMMY_META2,
                )
            )
            f = self._write_chunks(parts, 5)
            if f.tell() > file_len + n:
                chunks += 1
            file_len = f.tell()
            self.assertEqual((b"".join(d for d, _ in parts), parts[-1][1], False), self._read_all(f, 5, read_size=1000))
            n *= 2

    def test_interrupted_appends(self, single_write=True):
        SIZES = (5, 10, 20)
        for last_size_bits in SIZES:
            d, header_offs, chunks = b"", 0, []
            for size_bits in SIZES:
                if size_bits > last_size_bits:
                    break
                chunk_size = 1 << (size_bits - 1)  # max valid chunk size
                header_offs += chunk_size
                chunk = b"A" * (
                    chunk_size
                    - (size_bits + 7) // 8  # len of the chunk size field
                    - self._META_LEN
                )
                d += chunk
                chunks.append((chunk, self._DUMMY_META))
            f = self._write_chunks([(d, self._DUMMY_META)] if single_write else chunks, 5)
            # those very important assertions verify that the offset of the
            # header is aligned on a multiple of 1 << (first_size_bits - 1),
            # so the header will be fully contained in a single disk sector
            self.assertEqual(header_offs, f.tell())
            self.assertEqual(0, header_offs % (1 << 4))
            f.seek(0)
            rcarch.Chunker(f, self._META_LEN, 5).append(b"B", self._DUMMY_META2, flush=flush_bytesio)
            db = d + b"B"
            self.assertEqual((db, self._DUMMY_META2, False), self._read_all(f, 5))
            self.assertGreater(f.tell(), header_offs + 1)  # verify it created a new chunk
            # change the header to set the size to 0, to simulate an interrupted append
            size_len = (size_bits * 2 + 7) // 8
            f.seek(header_offs)
            assert f.write(b"\x00" * size_len) == size_len
            self.assertEqual((d, self._DUMMY_META, True), self._read_all(f, 5))
            f.seek(0)
            rcarch.Chunker(f, self._META_LEN, 5).append(b"B", self._DUMMY_META2, flush=flush_bytesio)
            self.assertEqual((db, self._DUMMY_META2, False), self._read_all(f, 5))
            # set the header size to 0 again and truncate, to simulate an append interrupted before the payload is written
            f.seek(header_offs)
            assert f.write(b"\x00" * size_len) == size_len
            f.truncate(header_offs + size_len + len(self._DUMMY_META))
            self.assertEqual((d, self._DUMMY_META, True), self._read_all(f, 5))
            f.seek(0)
            rcarch.Chunker(f, self._META_LEN, 5).append(b"B", self._DUMMY_META2, flush=flush_bytesio)
            self.assertEqual((db, self._DUMMY_META2, False), self._read_all(f, 5))
            # check what happens if a header could be only partially appended
            size = f.tell()
            for n in range(2, size - header_offs):
                f.truncate(size - n)
                self.assertEqual((d, self._DUMMY_META, True), self._read_all(f, 5))
                f.seek(0)
                rcarch.Chunker(f, self._META_LEN, 5).append(b"B", self._DUMMY_META2, flush=flush_bytesio)
                self.assertEqual((db, self._DUMMY_META2, False), self._read_all(f, 5))
                self.assertEqual(size, f.tell())
        if single_write:
            self.test_interrupted_appends(single_write=False)

    def test_resume_appends(self):
        expected, n, f = b"", 5, BytesIO()
        for i in range(10):
            d = bytes([b"abcdefghijklmnopqrstuvwxyz"[i]]) * n
            meta = self._DUMMY_META if i % 2 else self._DUMMY_META2
            f.seek(0)
            c = rcarch.Chunker(f, self._META_LEN, 5)
            c.append(d, meta, flush=flush_bytesio)
            self.assertEqual(meta, c.meta)
            expected += d
            n *= 2
        self.assertEqual((expected, meta, False), self._read_all(f, 5))
        f.write(b"garbage")
        self.assertEqual((expected, meta, True), self._read_all(f, 5))

    def test_exactly_full_chunk(self):
        for first_size_bits in (8, 12, 16, 20, 24):
            chunk_len = 1 << (first_size_bits - 1)
            payload_len = chunk_len - ((first_size_bits + 7) // 8 + self._META_LEN)
            payload = b"A" * payload_len
            f = self._write_chunks([(payload, self._DUMMY_META)], first_size_bits)
            self.assertEqual((payload, self._DUMMY_META, False), self._read_all(f, first_size_bits, read_size=1000))
            offset = f.tell()
            self._write_chunks([(b"B", self._DUMMY_META2)], first_size_bits, file=f)
            self.assertGreater(f.tell(), offset + 1)  # otherwise, it wasn't an exactly full chunk

    def test_trailing_garbage(self):
        f = self._write_chunks([(b"hello", self._DUMMY_META)], 7)
        f.write(b"TRAILINGJUNK")
        f.seek(0)
        self.assertEqual((b"hello", self._DUMMY_META, True), self._read_all(f, 7))

    def test_zero_length_first_header(self):
        f = BytesIO()
        # simulate interrupted append by writing a header advertising 0-length
        header = (0).to_bytes(1, "big") + self._DUMMY_META
        rcarch._write(f, header)
        self.assertEqual((b"", None, True), self._read_all(f, 8))

    def test_zero_length_second_header(self):
        expected = b"A" * 110
        f = self._write_chunks([(expected, self._DUMMY_META)], 8)
        # append 1 byte chunks until a new header is written
        while True:
            offset = f.tell()
            self._write_chunks([(b"B", self._DUMMY_META)], 8, file=f)
            if f.tell() > offset + 1:
                break
            expected += b"B"
        self.assertEqual(offset + 2 + self._META_LEN + 1, f.tell())
        f.truncate(offset)
        # simulate interrupted append by writing a header advertising 0-length
        f.write(b"\x00\x00" + self._DUMMY_META2)
        self.assertEqual((expected, self._DUMMY_META, True), self._read_all(f, 8))

    def test_corrupt_header_size_too_small(self):
        for i in range(1, 3 + self._META_LEN):
            f = BytesIO()
            bad_size = (i).to_bytes(3, "big")
            f.write(bad_size + self._DUMMY_META)
            f.seek(0)
            with self.assertRaises(rcarch.CorruptArchive) as cm:
                rcarch.Unchunker(f, self._META_LEN, 24).read(1)
            self.assertIn("size " + str(i), str(cm.exception))

    def test_append_spanning_two_chunks_with_reasonable_first_size(self):
        first_size_bits = 5
        for first in (False, True):
            for third in (False, True):
                for x in (0.5, 1, 1.5, 2, 4):
                    parts = [(b"1", self._DUMMY_META)] if first else []
                    second = b"2" * (1 << int(x * first_size_bits))
                    parts.append((second, self._DUMMY_META))
                    if third:
                        parts.append((b"3", self._DUMMY_META))
                    f = self._write_chunks(parts, first_size_bits)
                    self.assertEqual(
                        (b"".join(d for d, _ in parts), self._DUMMY_META, False),
                        self._read_all(f, first_size_bits, read_size=50),
                    )

    def test_one_byte_appends_with_multiple_chunks(self):
        # this test is important, it checks for bugs around chunk boundaries
        # use very small metadata and bits sizes, as it grows fast
        META1, META2 = b"m", b"n"
        f = self._write_chunks([(b"A", META1)], 3, meta_len=len(META1))
        self.assertEqual((b"A", META1, False), self._read_all(f, 3, meta_len=len(META1)))
        for i in range(2, 2**12 + 2**6 + 2**3 + 10):
            meta = META1 if i % 2 else META2
            self._write_chunks([(b"A", meta)], 3, file=f, meta_len=len(meta))
            self.assertEqual((b"A" * i, meta, False), self._read_all(f, 3, meta_len=len(meta), read_size=100))
            self.assertGreater(f.tell(), i)
            self.assertLess(f.tell(), i + 30)
        f.write(b"garbage")
        self.assertEqual((b"A" * i, meta, True), self._read_all(f, 3, meta_len=len(meta), read_size=100))

    def test_appends_spanning_multiple_chunks(self):
        # use very small metadata and bits sizes, as it grows fast
        META = b"m"
        for z in (b"", b"Z"):
            for n in (0, 2**3, 2**6 + 2**3, 2**12 + 2**6 + 2**3):
                n += 5
                f = self._write_chunks([(z, META)], 3, meta_len=len(META)) if z else None
                f = self._write_chunks([(b"A" * n, META)], 3, meta_len=len(META), file=f)
                d, meta, garbage = self._read_all(f, 3, meta_len=len(META), read_size=100)
                self.assertEqual((z + b"A" * n, META, False), (d, meta, garbage))
                self._write_chunks([(b"B", META)], 3, meta_len=len(META), file=f)
                self.assertEqual(
                    (z + b"A" * n + b"B", META, False), self._read_all(f, 3, meta_len=len(META), read_size=1001)
                )
                self.assertGreater(f.tell(), n)
                self.assertLess(f.tell(), n + 30)
                f.write(b"garbage")
                self.assertEqual(
                    (z + b"A" * n + b"B", META, True), self._read_all(f, 3, meta_len=len(META), read_size=1001)
                )

    def test_appends_with_only_metadata(self):
        f = self._write_chunks([(b"", self._DUMMY_META)], 5)
        self.assertEqual((b"", self._DUMMY_META, False), self._read_all(f, 5))
        self._write_chunks([(b"", self._DUMMY_META2)], 5, file=f)
        self.assertEqual((b"", self._DUMMY_META2, False), self._read_all(f, 5))
        self._write_chunks([(b"A", self._DUMMY_META)], 5, file=f)
        d = b"A"
        self.assertEqual((d, self._DUMMY_META, False), self._read_all(f, 5))
        for n in range(1 << 5):
            self._write_chunks([(b"", self._DUMMY_META2)], 5, file=f)
            self.assertEqual((d, self._DUMMY_META2, False), self._read_all(f, 5))
            self._write_chunks([(b"B", self._DUMMY_META)], 5, file=f)
            d += b"B"
            self.assertEqual((d, self._DUMMY_META, False), self._read_all(f, 5))


class TestCaseWithTempFile(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            self.path = tf.name

    def tearDown(self):
        os.remove(self.path)


class RcarchTest(TestCaseWithTempFile):
    def _read_all(self):
        with open(self.path, "rb") as f:
            with rcarch.Reader(f) as r:
                result = list(r)
                return result, r.trailing_garbage

    def _append(self, blobs, extra_bytes=b""):
        with open(self.path, "r+b") as f:
            with rcarch.Writer(f) as w:
                for name, data in blobs:
                    w.append(name, data)
        if extra_bytes:
            with open(self.path, "ab") as f:
                f.write(extra_bytes)

    def _append_and_readback(self, blobs, extra_bytes=b""):
        self._append(blobs, extra_bytes=extra_bytes)
        return self._read_all()

    def test_empty_file(self):
        with open(self.path, "wb") as f:
            pass
        with open(self.path, "rb") as f:
            self.assertEqual(([], False), self._read_all())
        with open(self.path, "r+b") as f:
            with rcarch.Writer(f) as w:
                w.append("name", b"foo")
        self.assertEqual(([("name", b"foo")], False), self._read_all())

    def test_empty_append(self):
        self.assertEqual(([], False), self._append_and_readback([]))
        self.assertEqual(([("name", b"foo")], False), self._append_and_readback([("name", b"foo")]))
        self.assertEqual(([("name", b"foo")], False), self._append_and_readback([]))
        self.assertEqual(([("name", b"foo"), ("name2", b"bar")], False), self._append_and_readback([("name2", b"bar")]))

    def test_no_data_with_junk(self):
        with open(self.path, "r+b") as f:
            rcarch.Chunker(f, rcarch._HASH_LEN, rcarch._FIRST_BITS).append(b"", rcarch._new_hash().digest())
        self.assertEqual(
            ([], True),
            self._append_and_readback(
                [], extra_bytes=b"loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong junk"
            ),
        )
        self.assertEqual(([("name", b"foo")], True), self._append_and_readback([("name", b"foo")]))

    def test_zero_appends_with_garbage(self):
        self.assertEqual(([], True), self._append_and_readback([], extra_bytes=b"junk"))

    def test_one_blob(self):
        blobs = [("hello.txt", b"world")]
        self.assertEqual((blobs, False), self._append_and_readback(blobs))

    def test_one_blob_with_garbage(self):
        blobs = [("hello.txt", b"world")]
        self.assertEqual((blobs, True), self._append_and_readback(blobs, extra_bytes=b"junk"))

    def test_two_blobs(self):
        blobs = [("a", b"x"), ("b", b"y")]
        self.assertEqual((blobs, False), self._append_and_readback(blobs))

    def test_two_blobs_with_garbage(self):
        blobs = [("a", b"x"), ("b", b"y")]
        self.assertEqual((blobs, True), self._append_and_readback(blobs, extra_bytes=b"junk"))

    def test_resume_append(self):
        blobs = [("first", b"1"), ("second", b"2"), ("third", b"3")]
        for blob in blobs:
            with open(self.path, "r+b") as f:
                with rcarch.Writer(f) as w:
                    w.append(*blob)
        self.assertEqual((blobs, False), self._read_all())

    def test_empty_blob(self):
        for blobs in [
            # empty blob payload
            [("1", b"")],
            [("1", b""), ("2", b"foo")],
            [("1", b""), ("2", b"foo"), ("3", b"")],
            # empty blob name
            [("", b"foo")],
            [("", b"foo"), ("2", b"bar")],
            [("1", b"foo"), ("", b"bar")],
            [("1", b"foo"), ("", b"bar"), ("3", b"baz")],
            # empty blob name and payload
            [("", b"")],
            [("", b""), ("2", b"bar")],
            [("1", b"foo"), ("", b"")],
            [("1", b"foo"), ("", b""), ("3", b"baz")],
        ]:
            for extra_bytes in (b"", b"garbage"):
                os.truncate(self.path, 0)
                self.assertEqual((blobs, bool(extra_bytes)), self._append_and_readback(blobs, extra_bytes=extra_bytes))
                if len(blobs) <= 1:
                    continue
                os.truncate(self.path, 0)
                blobs = blobs[1:]
                self.assertEqual((blobs, bool(extra_bytes)), self._append_and_readback(blobs, extra_bytes=extra_bytes))

    def test_corrupt_intermediate_checksum(self):
        for resets in range(1, 5):
            for reset in range(0, resets):
                os.truncate(self.path, 0)
                for i in range(0, resets + 1):
                    if i - 1 == reset:
                        offset = os.stat(self.path).st_size
                    self._append_and_readback([(str(i), str(0x41 + i).encode("utf-8"))])
                # corrupt the reset checksum
                with open(self.path, "r+b") as f:
                    f.seek(offset)
                    expected = bytearray()
                    n = rcarch._CONTROL_BLOCK_FLAG | (rcarch._HASH_LEN << rcarch._CONTROL_BLOCK_LOW_BITS)
                    while n >= 0x80:
                        expected.append((n & 0x7F) | 0x80)
                        n >>= 7
                    expected.append(n)
                    varint = f.read(len(expected))
                    self.assertEqual(bytes(expected), varint)
                    f.write(b"Z" * rcarch._HASH_LEN)
                with self.assertRaises(rcarch.CorruptArchive) as cm:
                    self._read_all()
                self.assertIn("checksum", str(cm.exception))
                self._append([("last", b"Z")])
                with self.assertRaises(rcarch.CorruptArchive) as cm:
                    self._read_all()
                self.assertIn("checksum", str(cm.exception))

    def test_corrupt_last_checksum(self):
        for resets in range(3):
            os.truncate(self.path, 0)
            for reset in range(resets):
                self._append_and_readback([(str(reset), str(0x41 + reset).encode("utf-8"))])
            self._append_and_readback([("before_last", b"X")])
            with open(self.path, "r+b") as f:
                rcarch.Chunker(f, rcarch._HASH_LEN, rcarch._FIRST_BITS).append(b"", b"Z" * rcarch._HASH_LEN)
            with self.assertRaises(rcarch.CorruptArchive) as cm:
                self._read_all()
            self.assertIn("checksum", str(cm.exception))
            self._append([("last", b"Y")])
            with self.assertRaises(rcarch.CorruptArchive) as cm:
                self._read_all()
            self.assertIn("checksum", str(cm.exception))

    def test_long_blobs(self):
        r = random.Random(12345)
        all_blobs = [
            ("1", r.randbytes(1000)),
            ("2", r.randbytes(1000 * 1000)),
            ("3", r.randbytes(10 * 1000 * 1000)),
        ]
        for i in range(1, len(all_blobs) + 1):
            blobs = all_blobs[:i]
            self.assertEqual((blobs, False), self._append_and_readback(blobs))
            os.truncate(self.path, 0)


if __name__ == "__main__":
    unittest.main()
