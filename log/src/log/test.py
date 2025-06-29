# to run the tests:   pushd .. && python3 -m unittest log.test; popd
import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest

try:
    import log
except ModuleNotFoundError:
    import __init__ as log


class test_file_sink(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_log_to_file(self):
        for file_level in (-2, log.DBG, log.WARN):
            fpath = os.path.join(self.test_dir, "file_level%d" % file_level)
            with log.file_sink(file_path=fpath, stderr_level=log.FATAL + 1, file_level=file_level) as sink:
                for level in range(-4, log.FATAL):
                    sink.log(level, "foo %d", level)
            with open(fpath) as f:
                nlines = 0
                for line in f:
                    nlines += 1
                    assert line.endswith("\n")
                    c, i = re.search(r"^(.)\d\d\d\d \d\d:\d\d:\d\d foo (-?\d+)$", line).groups()
                    self.assertGreaterEqual(int(i), file_level)
                self.assertEqual(log.FATAL - file_level, nlines)

    def test_log_only_to_stderr(self):
        sink = log.file_sink(stderr_level=log.INFO, file_path="")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            sink.log(log.DBG, "debug")
            sink.log(log.INFO, "info %d", 6)
            sink.log(log.TELL, "tell %s %d", "foo", 42)
            sink.log(log.WARN, "warn %(foo)s", foo="bar")
            sink.log(log.ERR, "invalid *args %s", "foo", "bar")
        stderr = buf.getvalue()
        # print('------------------------------------------------------\n' + stderr)
        self.assertTrue(
            re.search(
                "\n".join(
                    (
                        r"^I.*info 6",
                        r"^T.*tell foo 42",
                        r"^W.*warn bar",
                        r"^E.*format.*invalid \*args.*foo.*bar.*log/test\.py.*line \d+",
                    )
                ),
                stderr,
                re.MULTILINE | re.DOTALL,
            )
        )

    def log_to_both_file_and_stderr(self):
        fpath = os.path.join(self.test_dir, "foo")
        with log.file_sink(stderr_level=log.INFO, file_path=fpath) as sink:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                sink.log(log.WARN, "warning")
            self.assertIn("warning", buf.getvalue())
        with open(fpath) as f:
            self.assertIn("warning", f.read())

    def test_it_does_not_go_bananas_when_logging_binary_rubbish(self):
        fpath = os.path.join(self.test_dir, "foo")
        with log.file_sink(stderr_level=log.INFO, file_path=fpath) as sink:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                sink.log(log.ERR, "binary rubbish \xff")
            self.assertIn("rubbish", buf.getvalue())
        with open(fpath) as f:
            self.assertIn("rubbish", f.read())

    def test_various_log_args(self):
        fpath = os.path.join(self.test_dir, "foo")
        with log.file_sink(file_path=fpath, stderr_level=log.FATAL + 1) as sink:
            sink.log(log.DBG, "valid *args %s %d", "foo", -5)
            sink.log(log.INFO, "valid **kwargs %(foo)s %(bar)d", foo="foo", bar=3)
            sink.log(log.TELL, "valid nothing")
            sink.log(log.WARN, "invalid *args %s %s %s", "foo")
            sink.log(log.WARN, "invalid **kwargs %(foo)s", bar="baz")
            sink.log(log.FATAL, "invalid empty *args %s %d")
            sink.log(log.FATAL, "invalid empty **kwargs %(foo)s")
            sink.log(log.FATAL + 1, "invalid level")
        with open(fpath) as f:
            s = f.read()
        # print('------------------------------------------------------\n' + s)
        self.assertTrue(
            re.search(
                "\n".join(
                    (
                        r"^D\d\d\d\d \d\d:\d\d:\d\d valid \*args foo -5",
                        r"^I\d\d\d\d \d\d:\d\d:\d\d valid \*\*kwargs foo 3",
                        r"^T\d\d\d\d \d\d:\d\d:\d\d valid nothing",
                        r"^E\d\d\d\d \d\d:\d\d:\d\d log format error.*invalid \*args %s %s %s.*foo.*log/test\.py.*line \d+.*",
                        r"^E\d\d\d\d \d\d:\d\d:\d\d log format error.*invalid \*\*kwargs.*foo.*bar.*baz.*log/test\.py.*line \d+.*",
                        r"^F\d\d\d\d \d\d:\d\d:\d\d log format error.*invalid empty \*args %s %d.*log/test\.py.*line \d+.*",
                        r"^F\d\d\d\d \d\d:\d\d:\d\d log format error.*invalid empty \*\*kwargs.*foo.*log/test\.py.*line \d+.*",
                        r"^F\d\d\d\d \d\d:\d\d:\d\d .*invalid log level \d+.*invalid level.*log/test\.py.*line \d+.*",
                    )
                ),
                s,
                re.MULTILINE | re.DOTALL,
            )
        )


class test_ram_sink(unittest.TestCase):
    def test_valid_args(self):
        sink = log.ram_sink()
        sink.log(log.INFO, "info")
        sink.log(log.TELL, "tell")
        sink.log(log.WARN, "warn %d", 5)
        sink.log(log.ERR, "err %(foo)s", foo="bar")
        self.assertEqual(
            [
                (log.INFO, "info"),
                (log.TELL, "tell"),
                (log.WARN, "warn 5"),
                (log.ERR, "err bar"),
            ],
            sink.logs,
        )

    def test_invalid_args(self):
        sink = log.ram_sink()
        sink.log(log.DBG, "dbg", "foo")
        self.assertRegex(sink.logs[0][1], "dbg.*foo")
        sink.log(log.INFO, "info %(foo)s")
        self.assertRegex(sink.logs[1][1], "info.*foo")


class test_make(unittest.TestCase):
    def test_valid(self):
        self.assertEqual((log.INFO, "foo"), log.make(log.INFO, "foo"))
        self.assertEqual((log.WARN, "foo bar 4"), log.make(log.WARN, "foo %s %d", "bar", 4))
        self.assertEqual((-5, "foo bar 4"), log.make(-5, "foo %(foo)s %(bar)d", foo="bar", bar=4))

    def test_invalid_args(self):
        logger, m = log.make(log.INFO, "invalid args %s", "foo", 42)
        self.assertEqual(log.ERR, logger)
        self.assertIn("log format error", m)
        self.assertIn("invalid args", m)
        self.assertIn("foo", m)
        self.assertIn("42", m)

    def test_invalid_kwargs(self):
        logger, m = log.make(log.INFO, "invalid kwargs %(a)s %(b)s", a="foo", z=42)
        self.assertEqual(log.ERR, logger)
        self.assertIn("log format error", m)
        self.assertIn("invalid kwargs", m)
        self.assertIn("foo", m)
        self.assertIn("42", m)

    def test_invalid_level(self):
        for level in range(log.FATAL + 1, log.FATAL + 3):
            logger, m = log.make(level, "foo %(bar)s", bar="baz")
            self.assertEqual(log.FATAL, logger)
            self.assertIn("invalid log level %d" % level, m)
            self.assertIn("foo baz", m)

    def test_invalid_everything(self):
        logger, m = log.make(log.FATAL + 1, "foo", "bar")
        self.assertEqual(log.FATAL, logger)
        self.assertIn("foo", m)
        self.assertIn("bar", m)


class test_logger(unittest.TestCase):
    def test_die(self):
        with self.assertRaises(SystemExit):
            log.logger(log.ram_sink()).die()

    def test_log_functions(self):
        rl = log.logger(log.ram_sink())
        for x in (
            ("log", log.WARN, log.WARN),
            ("v", 3, -3),
            ("dbg", log.DBG),
            ("info", log.INFO),
            ("tell", log.TELL),
            ("warn", log.WARN),
            ("err", log.ERR),
            ("fatal", log.FATAL),
        ):
            method = getattr(rl, x[0])
            args = list(x[1 : len(x) - 1])
            level = x[len(x) - 1]
            if level < log.FATAL:
                method(*(args + ["foo"]))
            else:
                with self.assertRaises(SystemExit):
                    method(*(args + ["foo"]))
            self.assertEqual(level, rl.sink.logs[len(rl.sink.logs) - 1][0])
            self.assertEqual("foo", rl.sink.logs[len(rl.sink.logs) - 1][1])
            if level < log.FATAL:
                method(*(args + ["bar %(a)d %(b)s"]), a=2, b="blah")
            else:
                with self.assertRaises(SystemExit):
                    method(*(args + ["bar %(a)d %(b)s"]), a=2, b="blah")
            self.assertEqual(level, rl.sink.logs[len(rl.sink.logs) - 1][0])
            self.assertEqual("bar 2 blah", rl.sink.logs[len(rl.sink.logs) - 1][1])
            if level < log.FATAL:
                method(*(args + ["baz %s %d"]), "blah", -3)
            else:
                with self.assertRaises(SystemExit):
                    method(*(args + ["baz %s %d"]), "blah", -3)
            self.assertEqual(level, rl.sink.logs[len(rl.sink.logs) - 1][0])
            self.assertEqual("baz blah -3", rl.sink.logs[len(rl.sink.logs) - 1][1])
            if level < log.FATAL:
                method(*(args + ["invalid %d"]))
            else:
                with self.assertRaises(SystemExit):
                    method(*(args + ["invalid %d"]))
            self.assertEqual(max(level, log.ERR), rl.sink.logs[len(rl.sink.logs) - 1][0])
            self.assertIn("log format error", rl.sink.logs[len(rl.sink.logs) - 1][1])
            self.assertIn("invalid %d", rl.sink.logs[len(rl.sink.logs) - 1][1])
            self.assertIn("", rl.sink.logs[len(rl.sink.logs) - 1][1])
            self.assertRegex(rl.sink.logs[len(rl.sink.logs) - 1][1], "log/test\.py.*line \d+")


class test_default(unittest.TestCase):
    def setUp(self):
        self.backup_default = log.default_logger

    def tearDown(self):
        log.default_logger = self.backup_default

    def test_default_logging(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            log.v(1, "verbose")
            log.dbg("dbg")
            log.info("info %d", 2)
            log.tell("tell %s", "blah")
            log.warn("warn %(foo)s", foo="foo")
            log.err("err")
            with self.assertRaises(SystemExit):
                log.fatal("fatal")
        s = buf.getvalue()
        self.assertNotIn("verbose", s)
        self.assertNotIn("dbg", s)
        self.assertNotIn("info 2", s)
        self.assertIn("tell blah", s)
        self.assertIn("warn foo", s)
        self.assertIn("err", s)
        self.assertIn("fatal", s)

    def test_setup_default(self):
        with tempfile.NamedTemporaryFile() as f:
            log.setup_default(file_path=f.name)
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                log.info("info")
                log.tell("tell")
                log.warn("warn")
            stderr = buf.getvalue()
            with open(f.name) as f2:
                file_content = f2.read()
        self.assertIn("info", file_content)
        self.assertNotIn("info", stderr)
        self.assertIn("tell", file_content)
        self.assertIn("tell", stderr)
        self.assertIn("warn", file_content)
        self.assertIn("warn", stderr)


class test_stack_trace(unittest.TestCase):
    def test_stack_trace(self):
        try:
            raise BaseException("blah")
        except BaseException as e:
            s = log.stack_trace(e)
            self.assertRegex(s, "log/test.py.*line \d.*test_stack_trace")
            self.assertIn('raise BaseException("blah")', s)
            self.assertFalse(s.startswith("\n"))
            self.assertFalse(s.endswith("\n"))

    def test_stack_trace_logger(self):
        sink = log.ram_sink()
        try:
            with log.stack_trace_logger(log_func=log.logger(sink).err):
                raise Exception("blah")
        except Exception:
            pass
        self.assertEqual(log.ERR, sink.logs[0][0])
        self.assertRegex(sink.logs[0][1], "log/test.py.*line \d.*test_stack_trace_logger")
        self.assertIn('raise Exception("blah")', sink.logs[0][1])
