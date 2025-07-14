# to run the tests:   pushd .. && python3 -m unittest presume.test; popd
import functools
import os
import pickle
import shutil
import signal
import socket
import sys
import tempfile
import time
import unittest

try:
    import presume
except ModuleNotFoundError:
    import __init__ as presume


class Failure(Exception):
    pass


class State(object):
    def __init__(self):
        self.step = 0
        self.started_steps = []
        self.completed_steps = []

    def main(
        self,
        failing_step: int,
        fail_func,
        sequence=None,
        set_pos=None,
        set_pos_while_iterating={},
        test_case: unittest.TestCase = None,
    ):
        if sequence is None:
            expected_final_pos = 10
            iterator = range(self.step, expected_final_pos)
        else:
            expected_final_pos = len(sequence)
            iterator = self.__dict__.setdefault("iterator", presume.Iterator(sequence))
        if set_pos is not None:
            iterator.set_position(set_pos)
        for self.step in iterator:
            sp = set_pos_while_iterating.get(self.step, None)
            if sp is not None:
                iterator.set_position(sp)
                del set_pos_while_iterating[self.step]
                self.step = sp
            if test_case:
                test_case.assertEqual(self.step, iterator.get_position())
            self.started_steps.append(self.step)
            if self.step == failing_step:
                if fail_func:
                    fail_func()
                raise Failure(self.step)
            self.completed_steps.append(self.step)
        if test_case:
            test_case.assertEqual(expected_final_pos, iterator.get_position())


def sigpipe():
    # SIGPIPE raises an exception only during blocking I/Os
    parent, child = socket.socketpair()
    child.close()
    try:
        parent.send(b"trigger SIGPIPE")
    finally:
        parent.close()


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_no_failure(self):
        with presume.Context(State()) as p:
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
                s = State()
                try:
                    with presume.Context(s, state_filename=state_file) as p:
                        p.state.main(failing_step, fail_func)
                    self.assertFalse("this code shouldn't be reached")
                except Failure as f:
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
                with presume.Context(State(), state_filename=state_file) as p:
                    p.state.main(-1, None)
                    # check this second call to main() resumes where the previous call left
                    self.assertEqual(
                        list(range(failing_step + 1)) + list(range(failing_step, 10)), p.state.started_steps
                    )
                    self.assertEqual(list(range(10)), p.state.completed_steps)
                with open(state_file, "rb") as f:
                    self.assertEqual(9, pickle.load(f).step)
                os.remove(state_file)


class IteratorCombinedWithContextTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_with_no_failure(self):
        with presume.Context(State()) as p:
            p.state.main(-1, None, [0, 1, 2])
            self.assertEqual([0, 1, 2], p.state.completed_steps)
        with presume.Context(State()) as p:
            p.state.main(-1, None, (0, 1, 2))
            self.assertEqual([0, 1, 2], p.state.completed_steps)
        with presume.Context(State()) as p:
            p.state.main(-1, None, [])
            self.assertEqual([], p.state.completed_steps)

    def test_with_failure(self):
        state_file = os.path.join(self.test_dir, "state")
        with presume.Context(State(), state_filename=state_file) as p:
            try:
                p.state.main(1, None, [0, 1, 2])
            except Failure:
                self.assertEqual([0, 1], p.state.started_steps)
            p.state.main(-1, None, [0, 1, 2])
            self.assertEqual([0, 1, 1, 2], p.state.started_steps)
            self.assertEqual([0, 1, 2], p.state.completed_steps)

    def test_set_position_before_iterating(self):
        for pos in [0, 1, 2]:
            with presume.Context(State(), state_filename=os.path.join(self.test_dir, "nofailure-%d" % pos)) as p:
                p.state.main(-1, None, [0, 1, 2], pos)
                self.assertEqual(list(range(pos, 3)), p.state.completed_steps)
                with self.assertRaises(Exception) as cm:
                    p.state.main(-1, None, [0, 1, 2], 1)
                self.assertIn("supported", str(cm.exception))
            for fail_at in range(pos, 3):
                with presume.Context(
                    State(), state_filename=os.path.join(self.test_dir, "failat-%d-%d" % (pos, fail_at))
                ) as p:
                    try:
                        p.state.main(fail_at, None, sequence=[0, 1, 2], set_pos=pos)
                    except Failure:
                        self.assertEqual(list(range(pos, fail_at + 1)), p.state.started_steps)
                    p.state.main(-1, None, [0, 1, 2], pos)
                    self.assertEqual(list(range(pos, fail_at + 1)) + list(range(pos, 3)), p.state.started_steps)
                    self.assertEqual(list(range(pos, fail_at)) + list(range(pos, 3)), p.state.completed_steps)
                    with self.assertRaises(Exception) as cm:
                        p.state.main(-1, None, sequence=[0, 1, 2], set_pos=pos)
                    self.assertIn("supported", str(cm.exception))

    def test_set_position_while_iterating_without_failure(self):
        with presume.Context(
            State(), state_filename=os.path.join(self.test_dir, "set-pos-while-iterating-without-failure")
        ) as p:
            p.state.main(-1, None, sequence=list(range(5)), set_pos_while_iterating={3: 1}, test_case=self)
            self.assertEqual([0, 1, 2, 1, 2, 3, 4], p.state.started_steps)
            self.assertEqual([0, 1, 2, 1, 2, 3, 4], p.state.completed_steps)

    def test_set_position_while_iterating_with_failure(self):
        for failing_step in (3, 4):
            with presume.Context(
                State(),
                state_filename=os.path.join(self.test_dir, "set-pos-while-iterating-with-failure%d" % failing_step),
            ) as p:
                with self.assertRaises(Failure):
                    p.state.main(
                        failing_step, None, sequence=list(range(5)), set_pos_while_iterating={1: 3}, test_case=self
                    )
                p.state.main(-1, None, sequence=list(range(5)), set_pos_while_iterating={1: 3}, test_case=self)
                self.assertEqual(
                    [0] + list(range(3, failing_step + 1)) + list(range(failing_step, 5)), p.state.started_steps
                )
                self.assertEqual([0, 3, 4], p.state.completed_steps)


class SignalsMaskerTest(unittest.TestCase):
    def setUp(self):
        self.mask = signal.pthread_sigmask(signal.SIG_BLOCK, {})

    def tearDown(self):
        signal.pthread_sigmask(signal.SIG_SETMASK, self.mask)

    def test_deferal(self):
        received = {}

        def handler(signum, frame):
            received[signum] = 1

        for signum in presume.SignalsMasker.MASKABLE_TERMINATING_SIGNALS:
            received[signum] = 0
            signal.signal(signum, handler)
        pid = os.getpid()
        with presume.SignalsMasker():
            for signum in received:
                os.kill(pid, signum)
            time.sleep(1)
            self.assertEqual(0, sum(received.values()))
        slept = 0
        while sum(received.values()) < len(received):
            time.sleep(0.05)
            slept += 0.05
            self.assertLess(slept, 3)
