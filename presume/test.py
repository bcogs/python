# to run the tests:   pushd .. && python3 -m unittest presume.test; popd
import os
import pickle
import shutil
import tempfile
import unittest

import presume


class failure(Exception):
    pass


class state(object):
    def __init__(self):
        self.step = 0

    def main(self, failing_step):
        self.initial_step = self.step
        for self.step in range(self.step, 10):
            if self.step == failing_step:
                raise failure(self.step)


class TestPresume(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_no_failure(self):
        state_file = os.path.join(self.test_dir, "state")
        with presume.context(state(), state_filename=state_file) as p:
            p.state.main(-1)
        with open(state_file, "rb") as f:
            self.assertEqual(9, pickle.load(f).step)

    def test_fail_and_resume(self):
        state_file = os.path.join(self.test_dir, "state")
        for failing_step in range(0, 3):
            try:
                with presume.context(state(), state_filename=state_file) as p:
                    p.state.main(failing_step)
                self.assertFalse("this code shouldn't be reached")
            except failure as f:
                self.assertEqual(str(failing_step), str(f))
            with presume.context(state(), state_filename=state_file) as p:
                p.state.main(-1)
                # check this second call to main() resumes where the previous call left
                self.assertEqual(failing_step, p.state.initial_step)
            with open(state_file, "rb") as f:
                self.assertEqual(9, pickle.load(f).step)
            os.remove(state_file)
