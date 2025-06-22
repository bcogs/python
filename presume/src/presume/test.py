# to run the tests:   pushd .. && python3 -m unittest presume.test; popd
import functools
import os
import pickle
import shutil
import signal
import socket
import sys
import tempfile
import unittest

try:
    import presume
except ModuleNotFoundError:
    import __init__ as presume


class failure(Exception):
    pass


class state(object):
    def __init__(self):
        self.step = 0
        self.started_steps = []
        self.completed_steps = []

    def main(self, failing_step, fail_func, sequence=None, set_pos=None):
        iterator = (
            range(self.step, 10)
            if sequence is None
            else self.__dict__.setdefault("iterator", presume.iterator(sequence))
        )
        if set_pos is not None:
            iterator.set_position(set_pos)
        for self.step in iterator:
            self.started_steps.append(self.step)
            if self.step == failing_step:
                if fail_func:
                    fail_func()
                raise failure(self.step)
            self.completed_steps.append(self.step)


def sigpipe():
    # SIGPIPE raises an exception only during blocking I/Os
    parent, child = socket.socketpair()
    child.close()
    try:
        parent.send(b"trigger SIGPIPE")
    finally:
        parent.close()


class TestContext(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_no_failure(self):
        with presume.context(state()) as p:
            p.state.main(-1, None)
            self.assertEqual(9, p.state.step)
            self.assertEqual(list(range(10)), p.state.started_steps)
            self.assertEqual(list(range(10)), p.state.completed_steps)

    def test_fail_and_resume(self):
        state_file = os.path.join(self.test_dir, "state")
        exit = functools.partial(sys.exit, 1)
        sigint = functools.partial(os.kill, os.getpid(), signal.SIGINT)
        for fail_func in (None, exit, sigint, sigpipe):
            for failing_step in range(0, 3):
                s = state()
                try:
                    with presume.context(s, state_filename=state_file) as p:
                        p.state.main(failing_step, fail_func)
                    self.assertFalse("this code shouldn't be reached")
                except failure as f:
                    self.assertIsNone(fail_func)
                    self.assertEqual(str(failing_step), str(f))
                except BrokenPipeError:
                    self.assertEqual(sigpipe, fail_func)
                except KeyboardInterrupt:
                    self.assertEqual(sigint, fail_func)
                except SystemExit:
                    self.assertEqual(exit, fail_func)
                self.assertEqual(list(range(failing_step + 1)), s.started_steps)
                self.assertEqual(list(range(failing_step)), s.completed_steps)
                with presume.context(state(), state_filename=state_file) as p:
                    p.state.main(-1, None)
                    # check this second call to main() resumes where the previous call left
                    self.assertEqual(
                        list(range(failing_step + 1)) + list(range(failing_step, 10)), p.state.started_steps
                    )
                    self.assertEqual(list(range(10)), p.state.completed_steps)
                with open(state_file, "rb") as f:
                    self.assertEqual(9, pickle.load(f).step)
                os.remove(state_file)


class TestIterator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_with_no_failure(self):
        with presume.context(state()) as p:
            p.state.main(-1, None, [0, 1, 2])
            self.assertEqual([0, 1, 2], p.state.completed_steps)
        with presume.context(state()) as p:
            p.state.main(-1, None, (0, 1, 2))
            self.assertEqual([0, 1, 2], p.state.completed_steps)
        with presume.context(state()) as p:
            p.state.main(-1, None, [])
            self.assertEqual([], p.state.completed_steps)

    def test_with_failure(self):
        state_file = os.path.join(self.test_dir, "state")
        with presume.context(state(), state_filename=state_file) as p:
            try:
                p.state.main(1, None, [0, 1, 2])
            except failure:
                self.assertEqual([0, 1], p.state.started_steps)
            p.state.main(-1, None, [0, 1, 2])
            self.assertEqual([0, 1, 1, 2], p.state.started_steps)
            self.assertEqual([0, 1, 2], p.state.completed_steps)

    def test_set_position(self):
        for pos in [0, 1, 2]:
            with presume.context(state(), state_filename=os.path.join(self.test_dir, "nofailure-%d" % pos)) as p:
                p.state.main(-1, None, [0, 1, 2], pos)
                self.assertEqual(list(range(pos, 3)), p.state.completed_steps)
                with self.assertRaises(Exception) as cm:
                    p.state.main(-1, None, [0, 1, 2], 1)
                self.assertIn("supported", str(cm.exception))
            for fail_at in range(pos, 3):
                with presume.context(
                    state(), state_filename=os.path.join(self.test_dir, "failat-%d-%d" % (pos, fail_at))
                ) as p:
                    try:
                        p.state.main(fail_at, None, [0, 1, 2], pos)
                    except failure:
                        self.assertEqual(list(range(pos, fail_at + 1)), p.state.started_steps)
                    p.state.main(-1, None, [0, 1, 2], pos)
                    self.assertEqual(list(range(pos, fail_at + 1)) + list(range(pos, 3)), p.state.started_steps)
                    self.assertEqual(list(range(pos, fail_at)) + list(range(pos, 3)), p.state.completed_steps)
                    with self.assertRaises(Exception) as cm:
                        p.state.main(-1, None, [0, 1, 2], pos)
                    self.assertIn("supported", str(cm.exception))
