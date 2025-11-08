# to run the tests:   pushd .. && python3 -m unittest psched.test; popd
import concurrent.futures
import dataclasses
import glob
import os
import random
import re
import shelve
import signal
import tempfile
import threading
import time
import traceback
import unittest

try:
    import psched
except ModuleNotFoundError:
    import __init__ as psched


class Failure(Exception):
    pass


class UnrecoverableFailure(psched.UnrecoverableException):
    pass


class SignalsMaskerTest(unittest.TestCase):
    def setUp(self):
        self.mask = signal.pthread_sigmask(signal.SIG_BLOCK, {})

    def tearDown(self):
        signal.pthread_sigmask(signal.SIG_SETMASK, self.mask)

    def test_deferal(self):
        received = {}

        def handler(signum, frame):
            received[signum] = 1

        for signum in psched.SignalsMasker.MASKABLE_TERMINATING_SIGNALS:
            received[signum] = 0
            signal.signal(signum, handler)
        pid = os.getpid()
        with psched.SignalsMasker():
            for signum in received:
                os.kill(pid, signum)
            time.sleep(1)
            self.assertEqual(0, sum(received.values()))
        slept = 0
        while sum(received.values()) < len(received):
            time.sleep(0.05)
            slept += 0.05
            self.assertLess(slept, 3)


def noop(arg, live: dict):
    live.setdefault("args", []).append(arg)


def fail_first_pass(arg, live: dict):
    noop(arg, live)
    if live["pass"] < 2:
        raise Failure("simulated failure in fail_first_pass (%d)" % live["pass"])


class CallRecorder(object):
    call_recorders = {}  # id -> CallRecorder

    @staticmethod
    def get_by_id(call_recorder_id):
        return CallRecorder.call_recorders[call_recorder_id]

    def __init__(self, scheduler_test):
        self.calls, self._calls_lock = [], threading.Lock()
        self._scheduler_test = scheduler_test
        self.call_recorders[id(self)] = self

    def __reduce__(self):
        "override the pickling system to keep the same CallRecorder upon pickling/unpickling"
        return (CallRecorder.get_by_id, (id(self),))

    def do(
        self,
        *args,
        recursive_creations=None,
        multi_recursive_creations=None,
        scheduler=None,
        result=None,
        sleep_max=0.0,
        prevent_failures=False,
        **kwargs,
    ):
        if sleep_max > 0.0:
            time.sleep(random.uniform(0.0, sleep_max))
        self._record(args, kwargs)
        fail_until = kwargs.get("fail_until", None)
        if fail_until and not prevent_failures:
            if psched.Scheduler.get_time() < fail_until:
                wanted_in_stack_trace = Failure("simulated failure at %f" % psched.Scheduler.get_time())
                raise wanted_in_stack_trace
        fail_if = kwargs.get("fail_if", None)
        if args and fail_if and not prevent_failures:
            if re.search(fail_if, str(args[0])):
                wanted_in_stack_trace = Failure("simulated failure with %r" % args[0])
                raise wanted_in_stack_trace
        new_tasks = []
        if recursive_creations:
            try:
                args, push_kwargs = next(recursive_creations)
            except StopIteration:
                return None if result is None else psched.Out(result=result)
            kwargs.update({"scheduler": scheduler, "recursive_creations": recursive_creations})
            new_tasks.append(psched.Task(what=self.do, args=args, kwargs=kwargs))
        if multi_recursive_creations:
            for args, push_kwargs in multi_recursive_creations:
                push_kwargs.setdefault("kwargs", {}).update(
                    {"scheduler": scheduler, "recursive_creations": recursive_creations}
                )
                new_tasks.append(psched.Task(self.do, args=args, **push_kwargs))
        return psched.Out(result=result, tasks=new_tasks)

    def raise_exception(self, recoverable: bool, *args, **kwargs):
        self._record(args, kwargs)
        raise (Failure if recoverable else UnrecoverableFailure)("simulated failure (recoverable=%r)" % recoverable)

    def record(self, *args, **kwargs):
        self._record(args, kwargs)

    def _record(self, args, kwargs):
        with self._calls_lock:
            self.calls.append((self._scheduler_test.now_secs, args, kwargs))

    def task_a0(self, a: str, **kwargs):
        self.do(a, **kwargs)
        tasks = {}
        for x in "bcdefghij":
            kwargs = {"sleep_max": 0.3}
            if x in ("c", "h"):
                kwargs["result"] = "result-" + x
            tasks[x] = psched.Task(self.do, args=["arg-" + x], kwargs=kwargs)
            if x != "b" and x <= "g":
                tasks["b"].depend_on(tasks[x], "key_" + x)
        tasks["f"].depend_on(tasks["g"], "key_g")
        tasks["f"].depend_on(tasks["h"], "key_h")
        tasks["f"].depend_on(tasks["i"], "key_i")
        tasks["f"].depend_on(tasks["j"], "key_j")
        # test all cases: put the dependent before and after the dependee in the
        # list, and put the dependee in the list or not
        return psched.Out(tasks=[tasks["c"], tasks["b"], tasks["e"], tasks["i"]])

    _TASK_A1_DEPS = {"b": "cdefg", "f": "ghij"}

    def task_a1(self, a: str, **kwargs):
        self.do(a, **kwargs)
        tasks, kwargs["prevent_failures"] = {}, psched.Live("nofail")
        for x in "bcdefghij":
            if "result" in kwargs:
                kwargs = kwargs.copy()
                kwargs["result"] = "res-" + x
            tasks[x] = psched.Task(self.do, args=["arg-" + x], kwargs=kwargs)
        for k, deps in self._TASK_A1_DEPS.items():
            for dep in deps:
                tasks[k].depend_on(tasks[dep], "key_" + dep)
            if x != "b" and x <= "g":
                tasks["b"].depend_on(tasks[x], "key_" + x)
        # test all cases: put the dependent before and after the dependee in the
        # list, and put the dependee in the list or not
        return psched.Out(tasks=[tasks["c"], tasks["b"], tasks["e"], tasks["i"]])

    def generate_dependency_cycle(self, n: int, explicit: bool):
        tasks = [psched.Task(self.record, args=(i,)) for i in range(n)]
        tasks[-1].depend_on(tasks[0], "dep")
        tasks[0].depend_on(tasks[-1], "dep")
        return psched.Out(tasks=tasks[0 if explicit else 1 :])

    def generate_double_dependency(self, arg: str, **kwargs):
        if arg:
            self.do(arg, **kwargs)
        dep = psched.Task(self.do, ("dep",), kwargs=kwargs)
        return psched.Out(
            tasks=[
                psched.Task(self.do, "1", kwargs=kwargs, dependencies={"one": dep}),
                psched.Task(self.do, "2", kwargs=kwargs, delay_secs=1.0, dependencies={"two": dep}),
            ]
        )


def set_dict_entry(d, key, val):
    d[key] = val


def work_in_sync(locks, acquire_locks_i, release_locks_i, complete):
    time.sleep(0.05)  # to check that run() waits until the end
    acquire_locks, release_locks = locks[acquire_locks_i], locks[release_locks_i]
    for i in range(len(acquire_locks)):
        release_locks[i].release()
        with acquire_locks[i]:
            pass
    complete.add(acquire_locks_i)


class JournalStatsWrapper(psched.Journal):
    """decorator around a Journal that observes what happens"""

    def __init__(self, journal):
        self.journal, self.calls, self.garbage, self.max_garbage = journal, [], 0, 0

    def compact(self, blobs_iterator):
        self.calls.append("compact")
        self.journal.compact(blobs_iterator)
        self.garbage = 0

    def get_iterator(self):
        iterator = self.journal.get_iterator()
        self.calls.append("get_iterator")
        return iterator

    def record(self, blob):
        blobs = blob if isinstance(blob, list) else [blob]
        new_tasks, n = 0, 0
        for b in blobs:
            if isinstance(b, dict):
                new_tasks += 1
            else:
                n += 1
        self.garbage += n
        self.max_garbage = max(self.max_garbage, self.garbage)
        self.calls.append(("record", new_tasks, n))
        self.journal.record(blob)


class FileJournalTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def glob(self):
        return set(glob.glob(os.path.join(self.dir.name, "*"))).union(glob.glob(os.path.join(self.dir.name, ".*")))

    def test_compact_and_resume(self):
        with psched.FileJournal(os.path.join(self.dir.name, "foo")) as journal:
            for blob in range(1, 4):
                journal.record(blob)
            journal.compact(range(4, 7))
            for blob in range(7, 10):
                journal.record(blob)
        with psched.FileJournal(os.path.join(self.dir.name, "foo")) as journal:
            self.assertEqual(str(list(range(4, 10))), str(list(journal.get_iterator())))
            journal.compact(range(10, 15))
            self.assertEqual(str(list(range(10, 15))), str(list(journal.get_iterator())))
        self.assertEqual(set((os.path.join(self.dir.name, "foo"),)), self.glob())

    def test_record_and_resume(self):
        all_blobs = [1, "two", Exception("three")]
        with psched.FileJournal(os.path.join(self.dir.name, "foo")) as journal:
            for blob in all_blobs:
                journal.record(blob)
        with psched.FileJournal(os.path.join(self.dir.name, "foo")) as journal:
            self.assertEqual(str(all_blobs), str(list(journal.get_iterator())))
            new_blobs = [4, "five", Exception("six")]
            for blob in new_blobs:
                journal.record(blob)
            all_blobs += new_blobs
        with psched.FileJournal(os.path.join(self.dir.name, "foo")) as journal:
            self.assertEqual(str(all_blobs), str(list(journal.get_iterator())))
        self.assertEqual(set((os.path.join(self.dir.name, "foo"),)), self.glob())

    def test_garbage_deletion(self):
        garbage_file = os.path.join(self.dir.name, psched.FileJournal._PREFIX + "garbage_foo")
        non_garbage_files = set(
            (
                os.path.join(self.dir.name, "notgarbage_foo"),
                os.path.join(self.dir.name, psched.FileJournal._PREFIX + "notgarbage_bar"),
            )
        )
        for f in non_garbage_files.union(set((garbage_file,))):
            with open(f, "w+b"):
                pass
        non_garbage_files.add(os.path.join(self.dir.name, "foo"))
        with psched.FileJournal(os.path.join(self.dir.name, "foo")):
            self.assertEqual(non_garbage_files, self.glob())
        self.assertEqual(non_garbage_files, self.glob())


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        def sleep(secs):
            self.sleeps.append(secs)
            self.now_secs += secs

        self.journal_dir = tempfile.TemporaryDirectory()
        self._backup = psched.Scheduler.get_time, psched.Scheduler.sleep
        CallRecorder.call_recorders, self.sleeps, self.now_secs = {}, [], 1.0
        psched.Scheduler.get_time = staticmethod(lambda: self.now_secs)
        psched.Scheduler.sleep = staticmethod(sleep)

    def tearDown(self):
        psched.Scheduler.get_time, psched.Scheduler.sleep = self._backup
        self.journal_dir.cleanup()

    def test_priorities_are_honored(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        s.run(
            [
                psched.Task(cr.do),
                psched.Task(cr.do, args=[3, 3, 3], kwargs={"three": 3}, priority=3),
                psched.Task(cr.do, args=[-4], priority=-4),
                psched.Task(cr.do, kwargs={"one": 1}, priority=1),
                psched.Task(cr.do, args=[2], priority=2),
                psched.Task(cr.do, args=[-2], priority=-2),
                psched.Task(cr.do, args=[-6], priority=-6),
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual(
            [
                (1.0, (3, 3, 3), {"three": 3}),
                (1.0, (2,), {}),
                (1.0, (), {"one": 1}),
                (1.0, (), {}),
                (1.0, (-2,), {}),
                (1.0, (-4,), {}),
                (1.0, (-6,), {}),
            ],
            cr.calls,
        )

    def test_repeated_calls(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        s.run(
            [
                psched.Task(cr.do, args=["foo"], priority=2),
                psched.Task(cr.do, args=["foo"], priority=2),
                psched.Task(cr.do, args=["foo"], priority=2),
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual(
            [
                (1.0, ("foo",), {}),
                (1.0, ("foo",), {}),
                (1.0, ("foo",), {}),
            ],
            cr.calls,
        )

    def test_nested_simple_pushes(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        priorities = (5, -2, 4, 4, 7, -5, 8, 2)
        recursive_creations = [([p], {"priority": p}) for p in priorities]
        s.run(
            [psched.Task(cr.do, args=(0,), kwargs={"scheduler": s, "recursive_creations": iter(recursive_creations)})],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertEqual("0," + ",".join(str(p) for p in priorities), ",".join(str(c[1][0]) for c in cr.calls))

    def test_nested_multiple_pushes_one_level_of_depth(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        priorities = [5, -2, 4, 4, 7, -5, 8, 2]
        recursive_creations = [([p], {"priority": p}) for p in priorities]
        s.run(
            [
                psched.Task(
                    cr.do, args=(0,), kwargs={"scheduler": s, "multi_recursive_creations": iter(recursive_creations)}
                )
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        priorities.sort(reverse=True)
        self.assertEqual("0," + ",".join(str(p) for p in priorities), ",".join(str(c[1][0]) for c in cr.calls))

    def test_nested_multiple_pushes_with_multi_depth(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        recursive_creations2b = [
            ((6, "lvl 2", "e"), {"priority": 6}),
            ((6, "lvl 2", "f"), {"priority": 6}),
        ]
        recursive_creations2c = [
            ((6, "lvl 2", "g"), {"priority": 6}),
        ]
        recursive_creations2d = [
            ((6, "lvl 2", "h"), {"priority": 6}),
            ((6, "lvl 2", "i"), {"priority": 6}),
            ((6, "lvl 2", "j"), {"priority": 6}),
        ]
        recursive_creations1 = [
            ((3, "lvl 1", "b"), {"priority": 3, "kwargs": {"multi_recursive_creations": iter(recursive_creations2b)}}),
            ((3, "lvl 1", "c"), {"priority": 3, "kwargs": {"multi_recursive_creations": iter(recursive_creations2c)}}),
            ((3, "lvl 1", "d"), {"priority": 3, "kwargs": {"multi_recursive_creations": iter(recursive_creations2d)}}),
        ]
        s.run(
            [
                psched.Task(
                    cr.do,
                    args=(0, "lvl 0", "a"),
                    kwargs={"scheduler": s, "multi_recursive_creations": iter(recursive_creations1)},
                )
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        done, task_tree, l1task = [], {}, ""
        for c in cr.calls:
            self.assertEqual(1.0, c[0])
            self.assertEqual("lvl %d" % (c[1][0] // 3), c[1][1])
            self.assertEqual({"a": 0, "b": 3, "c": 3, "d": 3}.get(c[1][2], 6), c[1][0])
            done.append(c[1][2])
            if c[1][0] == 3:
                l1task = c[1][2]
            elif c[1][0] == 6:
                task_tree.setdefault(l1task, set()).add(c[1][2])
        done.sort()
        self.assertListEqual(list("abcdefghij"), done)
        self.assertEqual(0, cr.calls[0][1][0])
        expected_task_tree = {"b": set("ef"), "c": set("g"), "d": set("hij")}
        self.assertDictEqual(expected_task_tree, task_tree)

    def test_delays_with_priorities(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        dp = [(3, 9), (6, 5), (6, 4), (6, 7), (4, -1), (6, 8)]
        tasks = []
        for i in range(len(dp)):
            delay, priority = dp[i]
            if i % 2:
                delay_secs, when = 0.0, s.get_time() + delay
            else:
                delay_secs, when = delay, None
            tasks.append(
                psched.Task(what=cr.do, args=(delay, priority), priority=priority, delay_secs=delay_secs, when=when)
            )
        s.run(tasks, journal=os.path.join(self.journal_dir.name, "foo"))
        dp = [(delay, -priority) for delay, priority in dp]
        dp.sort()
        expected = [(1.0 + delay, (delay, -priority), {}) for delay, priority in dp]
        self.assertListEqual(expected, cr.calls)

    def test_live_dict(self):
        live_state = {"two": 2}
        s, cr = psched.Scheduler(live_state=live_state), CallRecorder(self)
        s.run(
            [
                psched.Task(cr.do, args=[psched.Live("two")], kwargs={"x": psched.Live("two")}),
                psched.Task(cr.do, args=[psched.Live()], kwargs={"x": psched.Live(None)}),
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual(
            [
                (1.0, (2,), {"x": 2}),
                (1.0, (live_state,), {"x": live_state}),
            ],
            cr.calls,
        )

    def test_live_object(self):
        @dataclasses.dataclass
        class LiveState(object):
            x: int

        live_state = LiveState(42)
        s, cr = psched.Scheduler(live_state=live_state), CallRecorder(self)
        s.run(
            [
                psched.Task(cr.do, args=[psched.Live("x")], kwargs={"x": psched.Live("x")}),
                psched.Task(cr.do, args=[psched.Live(None)], kwargs={"x": psched.Live()}),
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual(
            [
                (1.0, (42,), {"x": 42}),
                (1.0, (live_state,), {"x": live_state}),
            ],
            cr.calls,
        )

    def test_task_modifies_live_state(self):
        live_state = {"y": 175}
        s = psched.Scheduler(live_state=live_state)
        s.run(
            [psched.Task(set_dict_entry, args=[psched.Live(), "x", 42])],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertEqual({"x": 42, "y": 175}, live_state)

    def test_retries_that_eventually_work_with_on_fail(self):
        self._retries(True, True)

    def test_retries_that_eventually_work_without_on_fail(self):
        self._retries(False, True)

    def test_retries_that_never_work_with_on_fail(self):
        self._retries(True, False)

    def test_retries_that_never_work_without_on_fail(self):
        self._retries(False, False)

    def _retries(self, set_on_fail, eventually_work):
        s, cr = psched.Scheduler(), CallRecorder(self)
        fail_until, n = (10.0, 6) if eventually_work else (1000.0, 9)
        kwargs = {"kwargs": {"fail_until": fail_until}, "max_retry_delay_secs": 100.0}
        if set_on_fail:
            kwargs["on_fail"] = None
        try:
            s.run([psched.Task(cr.do, **kwargs)], journal=os.path.join(self.journal_dir.name, "foo"))
            self.assertTrue(set_on_fail or eventually_work)
        except Failure as e:
            self.assertFalse(set_on_fail)
            self.assertFalse(eventually_work)
            self.assertIn("simulated failure", str(e))
            self.assertIn("wanted_in_stack_trace", "\n".join(traceback.format_exception(type(e), e, e.__traceback__)))
        self.assertListEqual(
            [
                (1.0, (), {"fail_until": fail_until}),
                (1.5, (), {"fail_until": fail_until}),
                (2.5, (), {"fail_until": fail_until}),
                (4.5, (), {"fail_until": fail_until}),
                (8.5, (), {"fail_until": fail_until}),
                (16.5, (), {"fail_until": fail_until}),
                (32.5, (), {"fail_until": fail_until}),
                (64.5, (), {"fail_until": fail_until}),
                (128.5, (), {"fail_until": fail_until}),
            ][:n],
            cr.calls,
        )

    def test_prioritizes_first_created_tasks(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        recursive_creations = [([i], {}) for i in range(1, 10)]
        s.run(
            [
                psched.Task(
                    cr.do, args=(0,), kwargs={"scheduler": s, "multi_recursive_creations": iter(recursive_creations)}
                )
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertEqual(",".join(str(i) for i in range(10)), ",".join(str(c[1][0]) for c in cr.calls))

    def test_dependencies(self):
        def run_once(n: int):
            s, cr = psched.Scheduler(), CallRecorder(self)
            s.run(
                [psched.Task(cr.task_a0, args=("arg-a",))],
                journal=os.path.join(self.journal_dir.name, str(n)),
                parallelism=20,
            )
            return cr

        futures, N = [], 100
        with concurrent.futures.ThreadPoolExecutor(max_workers=N + 1) as executor:
            for n in range(N):
                futures.append(executor.submit(run_once, n))
        for future in concurrent.futures.as_completed(futures):
            cr = future.result()
            # for call in cr.calls: print(call)
            # a must be the first call
            self.assertEqual("arg-a", cr.calls[0][1][0])
            # b must be the last call
            self.assertEqual("arg-b", cr.calls[len(cr.calls) - 1][1][0])
            # check we have all the calls we expect
            expected = {
                "arg-a": "",
                "arg-c": "",
                "arg-d": "",
                "arg-e": "",
                "arg-g": "",
                "arg-h": "",
                "arg-i": "",
                "arg-j": "",
                "arg-f": "key_g=None, key_h=result-h, key_i=None, key_j=None",
                "arg-b": "key_c=result-c, key_d=None, key_e=None, key_f=None, key_g=None",
            }
            for t, args, kwargs in cr.calls:
                self.assertEqual(1.0, t)
                if args[0] == "arg-g":  # g must be called before f
                    self.assertTrue("arg-f" in expected)
                s = ", ".join(k + "=" + str(v) for k, v in sorted(kwargs.items()) if k != "sleep_max")
                self.assertEqual(expected[args[0]], s)
                del expected[args[0]]

    def test_on_fail_with_no_retry_and_string(self):
        self._on_fail(False, "it-failed")

    def test_on_fail_with_retries_and_string(self):
        self._on_fail(True, "it-failed")

    def test_on_fail_with_no_retry_and_none(self):
        self._on_fail(False, None)

    def test_on_fail_with_retries_and_none(self):
        self._on_fail(True, None)

    def _on_fail(self, with_retries, on_fail):
        s, cr = psched.Scheduler(), CallRecorder(self)
        kwargs = {"kwargs": {"fail_until": 100.0}, "on_fail": on_fail}
        expected = [
            (1.0, (), {"fail_until": 100.0}),
        ]
        if with_retries:
            kwargs["max_retry_delay_secs"] = 10.0
            kwargs["retry_delay_secs"] = 2.0
            expected += [
                (3.0, (), {"fail_until": 100.0}),
                (7.0, (), {"fail_until": 100.0}),
                (15.0, (), {"fail_until": 100.0}),
            ]
        expected.append((15.0 if with_retries else 1.0, (), {"failing_task_result": on_fail}))
        failing_task = psched.Task(cr.do, **kwargs)
        dependent_task = psched.Task(cr.record, dependencies={"failing_task_result": failing_task})
        s.run([failing_task, dependent_task], journal=os.path.join(self.journal_dir.name, "foo"), parallelism=4)
        # for call in expected: print(call)
        self.assertListEqual(expected, cr.calls)

    def test_tasks_actually_run_in_parallel(self):
        TASKS, STEPS = 3, 4
        locks = []
        for task in range(TASKS):
            task_locks = []
            for i in range(STEPS):
                lock = threading.Lock()
                lock.acquire()
                task_locks.append(lock)
            locks.append(task_locks)
        t, s = [], psched.Scheduler(live_state={"locks": locks, "complete": set()})
        for i in range(TASKS):
            t.append(
                psched.Task(work_in_sync, args=(psched.Live("locks"), i, (i + 1) % TASKS, psched.Live("complete")))
            )

        def run_tasks():
            s.run(t, journal=os.path.join(self.journal_dir.name, "foo"), parallelism=TASKS)
            self.assertEqual(set([0, 1, 2]), s.live_state["complete"])
            for task_locks in locks:
                for lock in task_locks:
                    lock.acquire()

        thread = threading.Thread(target=run_tasks)
        thread.start()
        thread.join(3)
        self.assertFalse(thread.is_alive())  # thread still running

    def test_tasks_with_both_dependencies_and_time(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        task0 = psched.Task(cr.record)
        s.run(
            [
                task0,
                psched.Task(cr.record, delay_secs=2.0, dependencies={"a": task0}),
                psched.Task(cr.record, args=(4,), when=4.0),
                psched.Task(cr.record, when=5.0, dependencies={"b": task0}),
            ],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual(
            [
                (1.0, (), {}),
                (3.0, (), {"a": None}),
                (4.0, (4,), {}),
                (5.0, (), {"b": None}),
            ],
            cr.calls,
        )

    def test_dependency_failure_with_on_fail_str(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        task0 = psched.Task(cr.do, args=(0,), kwargs={"fail_until": 100.0}, on_fail="foo")
        s.run(
            [task0, psched.Task(cr.record, args=(1,), dependencies={"bar": task0})],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual([(1.0, (0,), {"fail_until": 100.0}), (1.0, (1,), {"bar": "foo"})], cr.calls)

    def test_dependency_failure_with_on_fail_none(self):
        s, cr = psched.Scheduler(), CallRecorder(self)
        task0 = psched.Task(cr.do, args=(0,), kwargs={"fail_until": 100.0}, on_fail=None)
        s.run(
            [task0, psched.Task(cr.record, args=(1,), dependencies={"bar": task0})],
            journal=os.path.join(self.journal_dir.name, "foo"),
        )
        self.assertListEqual([(1.0, (0,), {"fail_until": 100.0}), (1.0, (1,), {"bar": None})], cr.calls)

    def test_trivial_journal_recovery_with_no_compaction(self):
        journal, live_state = os.path.join(self.journal_dir.name, "journal"), {"pass": 1}
        tasks = [psched.Task(fail_first_pass, args=("foo", psched.Live()))]
        with self.assertRaises(Failure):
            psched.Scheduler(live_state=live_state).run(tasks, journal)
        self.assertEqual(["foo"], live_state["args"])
        live_state2 = live_state.copy()
        live_state2["pass"] = 2
        psched.Scheduler(live_state=live_state2).run([], journal)
        self.assertEqual(["foo", "foo"], live_state2["args"])
        psched.Scheduler(live_state=live_state2).run([], journal)
        self.assertEqual(["foo", "foo"], live_state2["args"])

    def test_trivial_journal_recovery_after_compaction(self):
        journal, live_state = os.path.join(self.journal_dir.name, "journal"), {"pass": 1}
        tasks = [psched.Task(noop, args=("%d" % i, psched.Live())) for i in range(4)]
        tasks.append(psched.Task(fail_first_pass, args=("foo", psched.Live())))
        with self.assertRaises(Failure):
            psched.Scheduler(live_state=live_state).run(tasks, journal, compaction=2)
        self.assertEqual(["0", "1", "2", "3", "foo"], live_state["args"])
        live_state2 = live_state.copy()
        live_state2["pass"] = 2
        psched.Scheduler(live_state=live_state2).run([], journal)
        self.assertEqual(["0", "1", "2", "3", "foo", "foo"], live_state2["args"])
        psched.Scheduler(live_state=live_state2).run([], journal)
        self.assertEqual(["0", "1", "2", "3", "foo", "foo"], live_state2["args"])

    def test_basic_journal_recovery(self):
        for fail_at in range(3):
            journal, live_state = os.path.join(self.journal_dir.name, str(fail_at)), {"nofail": False}
            s, cr, tasks = psched.Scheduler(live_state=live_state), CallRecorder(self), []
            for i in range(3):
                kwargs = {}
                if i == fail_at:
                    kwargs["fail_until"] = float("+Inf")
                    kwargs["prevent_failures"] = psched.Live("nofail")
                tasks.append(psched.Task(cr.do, args=(i,), kwargs=kwargs))
            with self.assertRaises(Failure):
                s.run(tasks, journal)
            live_state["nofail"] = True
            self.assertListEqual([(1.0, (i,), {}) for i in range(fail_at)], cr.calls[:fail_at])
            self.assertEqual((fail_at,), cr.calls[-1][1])
            s, cr.calls = psched.Scheduler(live_state=live_state), []
            s.run(tasks, journal)
            self.assertEqual([i for i in range(fail_at, 3)], [args[0] for t, args, kwargs in cr.calls])

    def test_journal_recovery_with_when(self):
        for fail_at in range(3):
            journal, live_state, self.now_secs = (
                os.path.join(self.journal_dir.name, str(fail_at)),
                {"nofail": False},
                1.0,
            )
            s, cr, tasks = psched.Scheduler(live_state=live_state), CallRecorder(self), []
            for i in range(3):
                kwargs = {}
                if i == fail_at:
                    kwargs["fail_until"] = float("+Inf")
                    kwargs["prevent_failures"] = psched.Live("nofail")
                tasks.append(psched.Task(cr.do, args=(i,), kwargs=kwargs, when=i + 1.0))
            with self.assertRaises(Failure):
                s.run(tasks, journal)
            self.assertListEqual([(i + 1.0, (i,), {}) for i in range(fail_at)], cr.calls[:fail_at])
            self.assertEqual((fail_at,), cr.calls[-1][1])
            live_state["nofail"] = True
            s, cr.calls = psched.Scheduler(live_state=live_state), []
            s.run(tasks, journal)
            self.assertEqual([i for i in range(fail_at, 3)], [args[0] for t, args, kwargs in cr.calls])

    def _journal_recovery_with_dependencies(self, with_results):
        for fail in "abcdefghij":
            journal, live_state = os.path.join(self.journal_dir.name, fail), {"nofail": False}
            s, cr = psched.Scheduler(live_state=live_state), CallRecorder(self)
            kwargs = {"prevent_failures": psched.Live("nofail"), "fail_if": "arg-" + fail}
            if with_results:
                kwargs["result"] = "res-a"
            task_a = psched.Task(
                cr.task_a1,
                args=("arg-a",),
                kwargs=kwargs,
            )
            with self.assertRaises(Failure):
                s.run([task_a], journal)
            live_state["nofail"] = True
            s.run([task_a], journal)
            calls = {}
            for call in cr.calls:
                arg0 = call[1][0]
                calls[arg0] = calls.get(arg0, 0) + 1
            expected, real = "", ""
            for c in "abcdefghij":
                expected += "arg-" + c + ":" + ("1" if c != fail else "2") + " "
                real += "arg-" + c + ":" + str(calls.get("arg-" + c, 0)) + " "
            self.assertEqual(expected, real)
            for call in cr.calls:
                t, kwargs = call[1][0][4], call[2]
                deps = cr._TASK_A1_DEPS.get(t, "")
                for dep in deps:
                    self.assertEqual("res-" + dep if with_results else None, kwargs["key_" + dep])

    def test_journal_recovery_with_dependencies_without_results(self):
        self._journal_recovery_with_dependencies(False)

    def test_journal_recovery_with_dependencies_with_results(self):
        self._journal_recovery_with_dependencies(True)

    def test_double_dependency_on_the_same_task(self):
        expected_args = [("dep",), ("1",), ("2",)]
        expected_kwargs = [{}, {"one": None, "two": None}, {"one": None, "two": None}]
        s, cr = psched.Scheduler(), CallRecorder(self)
        s.run(
            cr.generate_double_dependency("").tasks, journal=os.path.join(self.journal_dir.name, "foo"), parallelism=3
        )
        self.assertListEqual(expected_args, [call[1] for call in cr.calls])
        self.assertListEqual(expected_kwargs, [call[2] for call in cr.calls])
        cr.calls = []
        expected_args = [("0",)] + expected_args
        expected_kwargs = [{}] + expected_kwargs
        s.run(
            [psched.Task(cr.generate_double_dependency, args=("0",))],
            journal=os.path.join(self.journal_dir.name, "bar"),
            parallelism=2,
        )
        self.assertListEqual(expected_args, [call[1] for call in cr.calls])
        self.assertListEqual(expected_kwargs, [call[2] for call in cr.calls])

    def test_dependency_cycle(self):
        for explicit in (True, False):
            for n in range(2, 5):
                s, cr = psched.Scheduler(), CallRecorder(self)
                with self.assertRaises(psched.UnrecoverableException) as cm:
                    s.run(
                        cr.generate_dependency_cycle(n, explicit).tasks,
                        journal=os.path.join(self.journal_dir.name, "%s-%d" % (explicit, n)),
                    )
                self.assertIn("dependency loop", str(cm.exception))
                with self.assertRaises(psched.UnrecoverableException) as cm:
                    s.run(
                        [psched.Task(cr.generate_dependency_cycle, args=(n, explicit))],
                        journal=os.path.join(self.journal_dir.name, "%s-%d-2" % (explicit, n)),
                    )
                self.assertIn("dependency loop", str(cm.exception))

    def test_unrecoverable_exceptions_are_not_recovered(self):
        # what we're really testing is the behavior when recoverable is False,
        # but setting it to True checks the test does what it thinks it does
        for recoverable in (True, False):
            for max_retry_delay_secs in (0.0, 4.0):
                for on_fail in (psched.Fatal(), "foo"):
                    s, cr, exn = psched.Scheduler(), CallRecorder(self), None
                    try:
                        s.run(
                            [
                                psched.Task(
                                    cr.raise_exception,
                                    args=(recoverable,),
                                    max_retry_delay_secs=max_retry_delay_secs,
                                    on_fail=on_fail,
                                )
                            ],
                            journal=os.path.join(
                                self.journal_dir.name,
                                "%s-%f-%s" % (recoverable, max_retry_delay_secs, isinstance(on_fail, str)),
                            ),
                        )
                    except Exception as e:
                        exn = e
                    if not recoverable:
                        self.assertTrue(isinstance(exn, UnrecoverableFailure))
                        self.assertEqual(len(cr.calls), 1)
                        continue
                    if isinstance(on_fail, psched.Fatal):
                        self.assertTrue(isinstance(exn, Failure))
                    else:
                        self.assertIsNone(exn)
                    self.assertEqual(5 if max_retry_delay_secs else 1, len(cr.calls))

    def test_scheduler_does_compact_when_needed(self):
        with psched.FileJournal(os.path.join(self.journal_dir.name, "foo")) as fjournal:
            s, cr, journal = psched.Scheduler(), CallRecorder(self), JournalStatsWrapper(fjournal)
            s.run([psched.Task(cr.record, args=(i,)) for i in range(20)], compaction=3, parallelism=5, journal=journal)
            four = [("record", 0, 1), ("record", 0, 1), ("record", 0, 1), "compact"]
            twenty = four + four + four + four + four
            self.assertListEqual(["get_iterator", ("record", 20, 0)] + twenty, journal.calls)
            self.assertEqual(list(range(20)), sorted([call[1][0] for call in cr.calls]))

    def test_recovery_after_compaction_without_children_tasks(self):
        with psched.FileJournal(os.path.join(self.journal_dir.name, "foo")) as fjournal:
            s, cr, journal = psched.Scheduler(), CallRecorder(self), JournalStatsWrapper(fjournal)
            tasks = [psched.Task(cr.record, args=(i,)) for i in range(5)]
            t5 = psched.Task(cr.do, args=(5,), kwargs={"fail_until": 2.0})
            for i in range(len(tasks)):
                t5.depend_on(tasks[i], "k" + str(i))
            tasks.append(t5)
            tasks += [psched.Task(cr.record, args=(i,), when=2.0) for i in range(6, 10)]
            with self.assertRaises(Failure):
                s.run(tasks, compaction=2, parallelism=5, journal=journal)
            self.assertListEqual(
                [
                    "get_iterator",
                    ("record", 9, 0),
                    ("record", 0, 1),
                    ("record", 0, 1),
                    "compact",
                    ("record", 0, 1),
                    ("record", 0, 1),
                ],
                journal.calls,
            )
            self.now_secs, journal.calls, s = 2.0, [], psched.Scheduler()
            s.run(tasks, compaction=2, parallelism=5, journal=journal)
            self.assertListEqual(
                [
                    "get_iterator",
                    "compact",
                    ("record", 0, 1),
                    ("record", 0, 1),
                    "compact",
                    ("record", 0, 1),
                ],
                journal.calls,
            )
            calls = [call[1][0] for call in cr.calls]
            self.assertEqual(list(range(6)), sorted(calls[:6]))
            self.assertEqual(list(range(5, 10)), sorted(calls[6:]))

    def test_recovery_after_compaction_with_children_tasks(self):
        with psched.FileJournal(os.path.join(self.journal_dir.name, "foo")) as fjournal:
            s, cr, journal = psched.Scheduler(), CallRecorder(self), JournalStatsWrapper(fjournal)
            tasks = [
                psched.Task(
                    cr.do,
                    args=(i,),
                    kwargs={"fail_until": 11.1 if i == 10 else -1.0, "recursive_creations": iter([((-i,), {})])},
                    when=i + 1,
                )
                for i in range(20)
            ]
            with self.assertRaises(Failure):
                s.run(tasks, compaction=2, journal=journal)
            self.assertLess(1, journal.max_garbage)
            self.assertLess(journal.max_garbage, 3)
            calls = {}
            for call in cr.calls:
                arg = call[1][0]
                calls.setdefault(arg, 0)
                calls[arg] += 1
            expected = dict((i, 1) for i in range(-9, 11))
            expected[0] = 2
            self.assertEqual(expected, calls)
            self.now_secs, cr.calls, journal.calls, s = 11.5, [], [], psched.Scheduler()
            s.run(tasks, compaction=2, journal=journal)
            calls = {}
            for call in cr.calls:
                arg = call[1][0]
                calls.setdefault(arg, 0)
                calls[arg] += 1
            expected = dict((i, 1) for i in range(10, 20))
            expected.update(dict((-i, 1) for i in range(10, 20)))
            self.assertEqual(expected, calls)
            self.assertLess(1, journal.max_garbage)
            self.assertLess(journal.max_garbage, 3)


class ClaimLedgerWithShelveTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.reopen_shelf()

    def tearDown(self):
        self.dir.cleanup()
        if self.shelf:
            self.shelf = self.shelf.close()

    def reopen_shelf(self):
        if self.__dict__.get("shelf"):
            self.shelf.close()
        self.shelf = shelve.open(os.path.join(self.dir.name, "shelf"))

    def _test_nominal(self, cache: bool):
        N = 2
        for i in range(N):
            ledger = psched.ClaimLedger(self.shelf, cache, sigmask=frozenset())
            for j in range(3):
                self.assertTrue(ledger.claim("what", "by who"))
                self.assertFalse(ledger.claim("what", "by someone else"))
                self.assertTrue(ledger.claim("what2", "foo"))
                self.assertFalse(ledger.claim("what2", "bar"))
            if i + 1 < N:
                self.reopen_shelf()
                ledger = psched.ClaimLedger(self.shelf, cache, sigmask=frozenset())

    def test_nominal_with_cache(self):
        self._test_nominal(True)

    def test_nominal_without_cache(self):
        self._test_nominal(False)

    def test_caching(self):
        ledger = psched.ClaimLedger(self.shelf, True, sigmask=frozenset())
        ledger.claim("what", "by who")
        self.shelf.close()
        self.shelf = None
        self.assertTrue(ledger.claim("what", "by who"))
        self.assertFalse(ledger.claim("what", "by someone else"))


class FakeDBAndLock(object):
    def __init__(self):
        self.data, self.calls, self.locked = {}, [], False

    def __setitem__(self, key, value):
        self.calls.append(("setitem", key, value))
        self.data[key] = value

    def get(self, key, default=None):
        self.calls.append(("get", key, default))
        return self.data.get(key, default)

    def __enter__(self):
        self.calls.append(("lock",))
        assert not self.locked
        self.locked = True
        return self

    def __exit__(self, *args):
        self.calls.append(("unlock",))
        self.locked = False


class ClaimLedgerWithFakeDBTest(unittest.TestCase):
    def setUp(self):
        self.db = FakeDBAndLock()

    def _test_locking_and_caching(self, cache: bool):
        ledger = psched.ClaimLedger(self.db, cache, lock=self.db)
        self.assertTrue(ledger.claim("what", "by who"))
        self.assertFalse(ledger.claim("what", "by someone else"))
        self.assertTrue(ledger.claim("what2", "foo"))
        self.assertFalse(ledger.claim("what2", "bar"))
        expected_calls = [("lock",), ("get", "what", None), ("setitem", "what", "by who"), ("unlock",), ("lock",)]
        if not cache:
            expected_calls.append(("get", "what", None))
        expected_calls += [
            ("unlock",),
            ("lock",),
            ("get", "what2", None),
            ("setitem", "what2", "foo"),
            ("unlock",),
            ("lock",),
        ]
        if not cache:
            expected_calls.append(("get", "what2", None))
        expected_calls.append(("unlock",))
        self.assertListEqual(expected_calls, self.db.calls)

    def test_locking_without_cache(self):
        self._test_locking_and_caching(False)

    def test_locking_with_cache(self):
        self._test_locking_and_caching(True)
