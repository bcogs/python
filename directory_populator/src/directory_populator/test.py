import datetime
import os
import shutil
import tempfile
import time
import unittest
import zoneinfo

try:
    import directory_populator
except ModuleNotFoundError:
    import __init__ as directory_populator


class ExpectedException(Exception):
    pass


class DirectoryPopulatorTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.archive, self.target = [os.path.realpath(os.path.join(self.test_dir, d)) for d in ("archive", "target")]
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create(self, path: str, content: str):
        with open(path, "w") as f:
            f.write(content)

    def assertContent(self, path: str, content: str):
        with open(path, "r") as f:
            self.assertEqual(content, f.read())

    def test_nominal(self):
        parent = os.path.join(self.test_dir, "parent")
        os.mkdir(parent)
        with open(os.path.join(parent, "bar"), "w") as f:
            f.write("test_nominal")
        for i in range(2):
            with directory_populator.DirectoryPopulator(self.target, tmp_parent=parent, tmp_suffix="zzzzz") as p:
                self.assertRegex(os.getcwd(), ".*zzzzz$")
                self.assertContent(os.path.join("..", "bar"), "test_nominal")
                self.create("foo", str(i))
                self.assertContent(os.path.join(p.tmp_dir, "foo"), str(i))
                self.assertFalse(hasattr(p, "committed"))
            self.assertTrue(p.committed)
            self.assertContent(os.path.join(self.target, "foo"), str(i))

    def test_commit_in_context(self):
        for i in range(2):
            raised = False
            try:
                with directory_populator.DirectoryPopulator(self.target, self.archive, tmp_parent=self.test_dir) as p:
                    self.create("foo", str(i))
                    self.assertFalse(hasattr(p, "committed"))
                    p.commit()
                    self.assertContent(os.path.join(self.target, "foo"), str(i))
                    if i == 1:
                        raise (ExpectedException("expected exception"))
                    self.assertTrue(hasattr(p, "committed"))
            except ExpectedException:
                raised = True
            self.assertEqual(i == 1, raised)
            with self.assertRaises(Exception):
                p.commit()

    def test_exception_prevents_implicit_commit(self):
        raised = False
        try:
            with directory_populator.DirectoryPopulator(self.target, self.archive, tmp_parent=self.test_dir) as p:
                self.create("foo", "foo")
                self.assertFalse(hasattr(p, "committed"))
                raise (ExpectedException("expected exception"))
        except ExpectedException:
            raised = True
        self.assertTrue(raised)
        foo = os.path.join(self.target, "foo")
        self.assertFalse(os.path.exists(foo))
        p.commit()
        self.assertTrue(p.committed)
        self.assertContent(foo, "foo")
        with self.assertRaises(Exception):
            p.commit()

    def test_nochdir(self):
        with open("bar", "w") as f:
            f.write("test_nochdir")
        for i in range(2):
            with directory_populator.DirectoryPopulator(
                self.target,
                tmp_prefix="prefix",
                tmp_parent=self.test_dir,
                chdir=bool(i),
            ):
                self.assertContent(os.path.join(".." if i else ".", "bar"), "test_nochdir")
                self.create("foo" + str(i), str(i))
            self.assertContent("bar", "test_nochdir")
        self.assertContent("foo0", "0")
        self.assertContent(os.path.join("target", "foo1"), "1")

    def test_relative_target(self):
        for i in range(2):
            with directory_populator.DirectoryPopulator("target", tmp_parent=self.test_dir, chdir=bool(i)):
                self.create("foo" + str(i), str(i))
        self.assertContent("foo0", "0")
        self.assertContent(os.path.join("target", "foo1"), "1")

    def test_relative_archive(self):
        for i in range(3):
            with directory_populator.DirectoryPopulator(
                self.target, archive="archive", tmp_parent=self.test_dir, chdir=bool(i)
            ):
                self.create("foo" + str(i), str(i))
            self.assertFalse(os.path.exists("archive"))
        self.assertContent("foo0", "0")
        self.assertFalse(os.path.exists(os.path.join("target", "foo1")))
        self.assertContent(os.path.join("target", "foo2"), "2")

    def test_strftime(self):
        for tz in ("UTC", "local", "America/Los_Angeles", "Europe/Paris"):
            if tz != "local":
                now = datetime.datetime.now(zoneinfo.ZoneInfo(tz)).timetuple()
            else:
                now = time.localtime()
            with directory_populator.DirectoryPopulator(
                "target", tmp_tz=tz, tmp_prefix="%H%M-", tmp_suffix="-%H%M"
            ) as p:
                self.create("foo", tz)
                self.assertTrue(os.getcwd().endswith(p.tmp_dir))
            self.assertContent(os.path.join("target", "foo"), tz)
            t0 = now.tm_hour * 60 + now.tm_min
            for s in (p.tmp_dir[:4], p.tmp_dir[-4:]):
                t1 = int(s[:2]) * 60 + int(s[2:])
                self.assertLess(t1, 24 * 60)
                self.assertLess((t1 - t0 + 60) % 60, 2)

    def test_nostrftime(self):
        with directory_populator.DirectoryPopulator("target", tmp_tz="", tmp_prefix="%H%M-", tmp_suffix="-%H%M") as p:
            self.create("foo", "nostrftime")
            self.assertTrue(os.getcwd().endswith(p.tmp_dir))
        self.assertRegex(p.tmp_dir, "^%H%M-.*-%H%M$")
        self.assertContent(os.path.join("target", "foo"), "nostrftime")

    def _test_forced_tmp_dir(self, create: bool):
        if create:
            os.mkdir("forced-tmp-dir")
        with directory_populator.DirectoryPopulator("target", tmp_dir="forced-tmp-dir"):
            self.create("foo", "forced-tmp-dir")
            self.assertTrue(os.getcwd().endswith("forced-tmp-dir"))
        self.assertContent(os.path.join("target", "foo"), "forced-tmp-dir")
        if create:
            os.mkdir("forced-tmp-dir")
        with directory_populator.DirectoryPopulator("target", chdir=False, tmp_dir="forced-tmp-dir"):
            self.create(os.path.join("forced-tmp-dir", "bar"), "forced-tmp-dir")
        self.assertContent(os.path.join("target", "bar"), "forced-tmp-dir")

    def test_forced_tmp_dir_create(self):
        self._test_forced_tmp_dir(True)

    def test_forced_tmp_dir_nocreate(self):
        self._test_forced_tmp_dir(False)
