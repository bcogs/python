"""The psched module is meant to create and execute tasks, and to recover execution from where it left in case of task failure.  The intent is to allow resuming after fixing the code of a task that raised an exception.

To use it, create a Scheduler and call its run() method with a list of initial Task instances.  A Task encapsulates a callable and its arguments, and it can return new tasks to execute.  Tasks can also depend on other tasks and receive their result in argument.  Task instances must be serializable with pickle, as the scheduler state is recorded to a journal for later recovery.

The module also has a number of whistles and bells such as parallel execution of tasks, priorities, task retries, deferred tasks, and more.

Example use:
    scheduler = psched.Scheduler()
    initial_tasks = [
        psched.Task(do_sth, args=("arg1", "arg2"), kwargs={"kwarg1": 1}),
        psched.Task(do_sth_else, max_retry_delay_secs=4.5, on_fail="oh, noes!"),
    ]
    scheduler.run(initial_tasks, "/path/to/the/journal", parallelism=3, logger=log.default_logger)

Objects that shouldn't be persisted but are still needed by the tasks (eg. an http client) can be passed as kwargs to the tasks using a mechanism called live state.  Example:
    def fetch(client: SomeHTTPClient=None):
        client.fetch(some_url)
    scheduler = psched.Scheduler(live_state={"shc": SomeHTTPClient()})
    initial_tasks = [psched.Task(fetch, kwargs={"client": psched.Live("shc")})]
    scheduler.run(initial_tasks, "/path/to/journal")
"""

import abc
import concurrent.futures
import contextlib
import dataclasses
import glob
import heapq
import itertools
import numbers
import os
import pickle
import signal
import sys
import tempfile
import threading
import time


class _NullLogger(object):
    @staticmethod
    def replace_none(logger):
        return logger if logger else _NullLogger()

    def __getattr__(self, name):
        return self._ignore

    def _ignore(self, *args, **kwargs):
        pass


class SignalsMasker(object):
    """Context manager that masks signals."""

    MASKABLE_TERMINATING_SIGNALS = frozenset(
        {signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT, signal.SIGALRM, signal.SIGUSR1, signal.SIGUSR2}
    )

    def __init__(self, signals_to_mask=MASKABLE_TERMINATING_SIGNALS):
        "signals_to_mask is a set {signal.SIGTERM, ...} that will be masked in addition to whatever was already masked before entering the context"
        self.signals_to_mask = signals_to_mask

    def __enter__(self):
        self._before = signal.pthread_sigmask(signal.SIG_BLOCK, self.signals_to_mask)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        signal.pthread_sigmask(signal.SIG_SETMASK, self._before)


class ClaimLedger(object):
    """Class that simplifies the avoidance of duplicate actions.

    When there's a possibility that a given action might be created more than
    once, the creators of the task can use the ClaimLedger to make sure only one
    of them get ownership of the action.  For this, the task must be identified
    by a unique string, and all potential creators should identify themselves
    with another unique string and try to claim the task. Only one of them will
    claim it successfully.  Repeated claims will succeed if they're made by the
    same claimant.

    This class supports concurrent access and works accross process restarts as
    long as the claims use stable ids and nothing meddles with the shelf.

    Example use:
        with shelve.open("shelf") as shelf:
            ledger = psched.ClaimLedger(shelf, True)

        # assuming several workers call the following, only one of them will
        # actually call do_task()
        def work(worker_id, ledger: ClaimLedger):
            if ledger.claim("some task", "worker %r" % worker_id): do_task()
    """

    def __init__(self, shelf: "shelve.Shelf", cache: bool = False, sigmask=SignalsMasker.MASKABLE_TERMINATING_SIGNALS):  # noqa: F821
        """Args:

        shelf: an open shelve.Shelf where the claims will be persisted
        cache: whether or not accesses to the shelf should be cached (this isn't
               based on the shelve.Shelf builtin cache, because writes are
               always synced immediately)
        sigmask: set of signals to mask when writing to the shelf"""
        self._cache, self._lock, self.shelf, self.sigmask = ({} if cache else None), threading.Lock(), shelf, sigmask

    def claim(self, what_is_claimed: str, who_is_claiming_it: str) -> bool:
        with self._lock:
            owner = None if self._cache is None else self._cache.get(what_is_claimed, None)
            if owner is None:
                owner = self.shelf.get(what_is_claimed, None)
                if owner is None:
                    with SignalsMasker(signals_to_mask=self.sigmask):
                        self.shelf[what_is_claimed] = who_is_claiming_it
                        self.shelf.sync()
                    self._observe(what_is_claimed, who_is_claiming_it)
                    return True
                self._observe(what_is_claimed, owner)
        return who_is_claiming_it == owner

    def _observe(self, what_is_claimed: str, who_is_claiming_it: str):
        if self._cache is not None:
            self._cache[what_is_claimed] = who_is_claiming_it


class Fatal(object):
    pass


@dataclasses.dataclass
class Task(object):
    """Wrapper around a callable that can be executed by a Scheduler.

    It's an ephemeral object that mustn't be accessed after returning it.
    All arguments passed to __init__ must be serializable with pickle, and they
    mustn't reference anything that will be modified after returning the Task.

    To execute the Task, the Scheduler calls what(*args, **kwargs).

    The call to what(...) must return None or an Out instance that contains an
    optional result and new Task instances to execute.

    Tasks with the highest priority are executed first by the Scheduler.
    Negative priorities are supported, and the default priority is 0.

    A Task can have dependencies that will be executed before the Task.
    Each dependency is a Task itself, and its optional result is passed to the
    Task as a kwarg.

    A Task can be delayed until a certain time.  This is done by setting the
    when parameter to a number of seconds relative to the epoch, or delay_secs
    to a number of seconds relative to the current time.  It won't be executed
    until both conditions are met:
      - the specified time is reached
      - and all its dependencies were executed.

    If a Task fails (by raising an Exception), its result can be set to a
    default value defined in the on_fail parameter.  If on_fail is a Fatal()
    instance (the default), the Scheduler execution stops and the Exception
    propagates.

    A Task can be retried later if it fails, if max_retry_delay_secs is strictly
    positive.  The retry delay starts at retry_delay_secs and doubles at each
    attempt, until it exceeds max_retry_delay_secs, at which point the Task
    fails as described above, causing the Exception to propagate or the Task to
    output its on_fail value.
    """

    what: callable
    args: tuple = ()
    kwargs: dict = dataclasses.field(default_factory=dict)
    priority: int = 0
    when: float = None
    delay_secs: float = 0.0
    retry_delay_secs: float = 0.5
    max_retry_delay_secs: float = 0.0
    on_fail: any = Fatal()
    dependencies: dict = dataclasses.field(default_factory=dict)  # str(kwarg) -> Task

    def depend_on(self, task: "Task", kwarg: str):
        """add task as a dependency, whose result will be passed to what() in the given kwarg"""
        assert kwarg not in self.dependencies
        self.dependencies[kwarg] = task

    def get_start_time(self, get_time):  # get_time is a function returning the current time as a float
        if self.when is not None:
            return self.when
        if self.delay_secs > 0.0:
            return get_time() + self.delay_secs
        return None

    def __repr__(self):
        return "Task(what=%r, args=%r, kwargs=%r)" % (self.what, self.args, self.kwargs)


class Out(object):
    """the return type of the callable wrapped by a Task"""

    def __init__(self, result=None, tasks=()):
        """Args:

        result: Optional result, that will be passed as a kwarg to dependent tasks.  Must be picklable.
        tasks:
            An iterable of new Task instances to schedule.
            If they have dependencies, those will be recursively scheduled too."""
        self.result, self.tasks = result, tasks


class _PersistableTask(object):
    @classmethod
    def init_from_pickled_dict(cls, d: dict):
        pt = cls.__new__(cls)
        pt.__dict__ = {k: v for k, v in d.items() if k not in {"args", "kwargs", "live"}}
        n, kwargs, live = -1, d.get("kwargs", {}), d.get("live", {})
        for k, v in live.items():
            if isinstance(k, str):
                kwargs[k] = Live(v)
                continue
            n = max(n, k)
        new_args, i, args = [], 0, d.get("args", ())
        while i < len(args) or len(new_args) < n:
            if len(new_args) in live:
                new_args.append(Live(live[len(new_args)]))
            else:
                new_args.append(args[i])
                i += 1
        pt.args, pt.kwargs = new_args, kwargs
        return pt

    def __init__(self, task: Task, sequence_number: int, when: float = None):
        # DON'T SET MEMBERS THAT HAVE THEIR DEFAULT VALUE
        # the class should be kept small, because it's written to disk
        self.sequence_number = sequence_number
        for k, v in task.__dict__.items():
            if k in frozenset({"args", "kwargs", "priority", "what"}):
                self.__dict__[k] = v
        if task.max_retry_delay_secs > 0.0:
            self.retry_delay_secs, self.max_retry_delay_secs = task.retry_delay_secs, task.max_retry_delay_secs
        deps = getattr(task, "dependencies", None)
        if deps:
            self.nblockers = len(deps)
        if not isinstance(task.on_fail, Fatal):
            self.on_fail = task.on_fail
        if when is not None:
            self.when = when

    def __lt__(self, other):
        """comparison operator for the _TasksQueue

        For tasks that have a .when member, pick lowest .when first.

        Then, pick highest priority first, then first created task first."""
        i = other.__dict__.get("when", -sys.float_info.max) - self.__dict__.get("when", -sys.float_info.max)
        if i:
            return i > 0
        i = self.priority - other.priority
        return (i > 0) if i else (other.sequence_number - self.sequence_number > 0)

    def __repr__(self):
        s = "_PersistableTask(%r" % self.what
        for x in ("args", "kwargs", "priority", "sequence_number", "when", "nblockers"):
            y = self.__dict__.get(x, None)
            if y is not None:
                s += ", %s=%r" % (x, y)
        if self.__dict__.get("max_retry_delay_secs", 0.0) > 0.0:
            s += ", max_retry_delay=%fs" % self.max_retry_delay_secs
        if hasattr(self, "on_fail") and not isinstance(self.on_fail, Fatal):
            s += ", on_fail=%r" % self.on_fail
        return s + ")"

    def explore(self, ptasks_by_seq_num):
        ptasks_by_seq_num[self.sequence_number] = self
        triggers = self.__dict__.get("triggers", None)
        max_seq_num = max(t.explore(ptasks_by_seq_num) for _, t in triggers) if triggers else -1
        return max(max_seq_num, self.sequence_number)

    def get_when(self):
        return self.__dict__.get("when", float("-Inf"))

    def handle_triggers(self, task_result):
        ready, triggers = [], self.__dict__.get("triggers", ())
        for kwarg, dependent_ptask in triggers:
            dependent_ptask.kwargs[kwarg] = task_result
            dependent_ptask.nblockers -= 1
            if dependent_ptask.nblockers <= 0:
                ready.append(dependent_ptask)
        return ready

    def prepare_pickling(self) -> dict:
        d, live = self.__dict__.copy(), {}
        args = d.get("args", None)
        if args:
            d["args"] = tuple(arg for i, arg in enumerate(args) if self._prepare_pickling(i, arg, live))
        kwargs = d.get("kwargs", None)
        if kwargs:
            d["kwargs"] = {k: v for k, v in kwargs.items() if self._prepare_pickling(k, v, live)}
        if live:
            d["live"] = live
        return d

    def process_failure(self, e: Exception):
        if "on_fail" in self.__dict__:
            return Out(result=self.on_fail)
        raise e

    def set_trigger(self, kwarg, dependent_ptask):
        self.__dict__.setdefault("triggers", []).append((kwarg, dependent_ptask))

    def _prepare_pickling(self, key, val, live):
        if isinstance(val, Live):
            live[key] = val.key
            return False
        return True


@dataclasses.dataclass
class Live(object):
    """Non-persisted info retrieved from the Scheduler live state and passed in argument to tasks.

    When creating a Task, if some of the args or kwargs are instances of Live,
    they'll be replaced when executing the Task by a value retrieved from
    the Scheduler's live state.

    If the Live instance key is None, the Live argument will be replaced by the
    entire live state.  Otherwise, it'll be replaced by one of:
        - live_state[key] if the Scheduler live state is a dict
        - or getattr(live_state, key).
    """

    key: str = None


class CompactionStrategy(object):
    """controls when to compact a Journal"""

    def __init__(self, max_garbage: int):
        self.garbage, self.max_garbage = 0, max_garbage

    def record_compaction(self):
        self.garbage = 0

    def record_garbage(self, n: int, **kwargs):
        self.garbage += n
        return self.garbage > self.max_garbage


class _TasksQueue(object):
    @classmethod
    def init_from_journal(cls, journal_iterator, cs: CompactionStrategy):
        """return a _TasksQueue or None if the journal is empty"""
        queued, ptasks, max_seq_num = {}, {}, -1
        for record in journal_iterator:
            if not isinstance(record, list):
                record = [record]
            completed = record[-1]
            if isinstance(completed, dict):
                completed = None
            else:
                record = record[:-1]
                if isinstance(completed, int):  # else it's a pair (seq num, result)
                    completed = (completed, None)
                del queued[completed[0]]
            for ptask_dict in record:
                ptask = _PersistableTask.init_from_pickled_dict(ptask_dict)
                max_seq_num = max(max_seq_num, ptask.explore(ptasks))
                queued[ptask.sequence_number] = ptask
            if completed is not None:
                seq_num, completed_result = completed
                for ptask in ptasks[seq_num].handle_triggers(completed_result):
                    queued[ptask.sequence_number] = ptask
                del ptasks[seq_num]
                cs.record_garbage(1)
        if max_seq_num < 0:  # no task was recorded in this journal
            return None
        ready, deferred = [], []
        for ptask in queued.values():
            (ready if ptask.__dict__.get("when", None) is None else deferred).append(ptask)
        heapq.heapify(ready)
        heapq.heapify(deferred)
        return cls.__new__(cls)._init(deferred, ready, deferred[0].when if deferred else float("+Inf"), max_seq_num + 1)

    def __init__(self):
        self._init([], [], float("+Inf"), -1)

    def get_compaction_iterator(self):
        return itertools.chain(self._deferred, self._ready)

    def new_task_sequence_number(self):
        self._seq_num += 1
        return self._seq_num

    def next(self, scheduler: "Scheduler"):  # -> _PersistableTask|float|None
        if self._deferred:
            t = scheduler.get_time()
            while self._deferred and t >= self._next_deferred_time:
                task = heapq.heappop(self._deferred)
                assert task.when <= t
                del task.when
                heapq.heappush(self._ready, task)
                self._next_deferred_time = self._deferred[0].when if self._deferred else float("+Inf")
        if self._ready:
            return heapq.heappop(self._ready)
        return self._next_deferred_time if self._deferred else None

    def push(self, task: _PersistableTask):
        when = task.get_when()
        if when <= 0.0:
            heapq.heappush(self._ready, task)
        else:
            self._next_deferred_time = min(self._next_deferred_time, when)
            heapq.heappush(self._deferred, task)

    def _init(self, deferred: list, ready: list, next_deferred_time: float, seq_num: int):
        self._deferred = deferred  # heap of _PersistableTask with a .when member
        self._ready = ready  # heap of _PersistableTask with no .when member
        self._next_deferred_time = next_deferred_time
        self._seq_num = seq_num
        return self


class Journal(abc.ABC):
    """Interface that must be implemented by journals used to track and resume work"""

    def compact(self, blobs_iterator):
        pass

    @abc.abstractmethod
    def get_iterator(self):
        """return an iterator that provides the journal entries in the exact same order they were record()ed"""
        pass

    @abc.abstractmethod
    def record(self, blob):
        pass

    # TODO: should the Journal be deleted when the scheduler ends?
    # If it isn't, recovery will kick in next time the program is run and
    # no task will be executed.


class FileJournal(Journal):
    """Journal that writes to a file, to be used as a context manager (or call close())"""

    _PREFIX = ".sched-journal_"

    def __init__(self, file_path):
        for f in glob.iglob(
            os.path.join(os.path.dirname(file_path), self._PREFIX + "*_" + os.path.basename(file_path))
        ):
            try:
                os.unlink(f)
            except Exception:
                pass
        self._file = open(file_path, "a+b")
        self.file_path, self._pickler = file_path, pickle.Pickler(self._file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def __next__(self):
        try:
            return self._unpickler.load()
        except EOFError:
            raise StopIteration
        except pickle.UnpicklingError:
            if self._file.read():  # truly corrupt
                raise
            # FIXME it sucks: we ignore the last entry if it was only partially
            # added, but if the journal is corrupt and we keep appending, it
            # won't be readable
            raise StopIteration

    def __iter__(self):
        self._file.seek(0)
        self._unpickler = pickle.Unpickler(self._file)
        return self

    def close(self):
        self._file.close()

    def compact(self, blobs_iterator):
        fd, path = tempfile.mkstemp(
            dir=os.path.dirname(self.file_path),
            prefix=self._PREFIX,
            suffix="_" + os.path.basename(self.file_path),
            text=False,
        )
        file = os.fdopen(fd, "w+b")
        file_to_close, pickler = None, pickle.Pickler(file)
        try:
            for blob in blobs_iterator:
                pickler.dump(blob)
            file.flush()
            os.fsync(file.fileno())
            os.rename(path, self.file_path)
            file_to_close = self._file
            self._file, self._pickler = file, pickler
        finally:
            if file_to_close:
                file_to_close.close()
            else:
                file.close()
                os.unlink(path)

    def get_iterator(self):
        return self

    def record(self, blob):
        self._pickler.dump(blob)
        self._file.flush()
        os.fsync(self._file.fileno())


class UnrecoverableException(Exception):
    """Exception that passes through the retry and default mechanisms of the Scheduler"""

    pass


class Scheduler(object):
    """executes user supplied tasks, and persists state to resume from where it left in case of failure"""

    get_time = staticmethod(time.time)
    sleep = staticmethod(time.sleep)

    def __init__(self, live_state=None, logger=None):
        """
        live_state can have any type, it's used to pass non-persisted arguments
        to the tasks (see the docstring of the Live class).

        logger is an optional https://github.com/bcogs/python log.Logger.
        """
        self.live_state, self.logger = live_state, _NullLogger.replace_none(logger)

    def run(self, initial_tasks, journal, parallelism: int = 1, compaction: int = 100):
        """Execute a number of tasks and any other tasks they create.

        initial_tasks is an iterable of Task.

        journal is either a Journal instance or the path of a file that will be
        used to initialize a FileJournal

        parallelism is the max number of tasks that can be run in parallel.

        compaction controls how often the journal gets compacted: when it
        contains more completed tasks than this number, it triggers a compaction
        """
        if isinstance(journal, str):
            journal = FileJournal(journal)
            journal_context = contextlib.closing(journal)
        else:
            journal_context = contextlib.nullcontext()
        with journal_context:
            future2pt, cs = {}, CompactionStrategy(compaction)
            queue = self._init_queue(initial_tasks, journal, cs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
                while self._run_tasks(executor, parallelism, queue, future2pt, journal, cs):
                    pass

    def _create_persistable_tasks(self, persisted_state, tasks, tasks_by_id, dependents):
        """tasks is an iterable of Task, tasks_by_id is a dict to initialize"""
        for task in tasks:
            if id(task) in dependents:
                raise UnrecoverableException("%r has a dependency loop" % task)
            if id(task) in tasks_by_id:
                continue
            tasks_by_id[id(task)] = (
                task,
                _PersistableTask(
                    task, persisted_state.new_task_sequence_number(), when=task.get_start_time(self.get_time)
                ),
            )
            dependents.add(id(task))
            self._create_persistable_tasks(persisted_state, task.dependencies.values(), tasks_by_id, dependents)
            dependents.remove(id(task))

    def _get_task_output(self, future, queue: _TasksQueue, ptask: _PersistableTask):
        """returns an Out, or, if it will be retried, None"""
        e = future.exception()
        if not e or isinstance(e, UnrecoverableException):
            result = future.result()
            return Out(result=None) if result is None else result
        max_delay = ptask.__dict__.get("max_retry_delay_secs", -1.0)
        if max_delay <= 0.0:
            return ptask.process_failure(e)
        delay = ptask.retry_delay_secs
        if delay <= max_delay:
            self.logger.info("%s failed (%s), retrying in %f seconds", ptask, e, delay)
            ptask.retry_delay_secs *= 2.0
            ptask.when = self.get_time() + delay
            queue.push(ptask)
            return None
        self.logger.error("%s failed and won't be retried - %s", ptask, e)
        return ptask.process_failure(e)

    def _init_queue(self, default_initial_tasks, journal: Journal, cs: CompactionStrategy):
        queue = _TasksQueue.init_from_journal(journal.get_iterator(), cs)
        if not queue:
            queue = _TasksQueue()
            self._push(queue, default_initial_tasks, journal, None, None, None)
        return queue

    def _push(
        self,
        queue: _TasksQueue,
        tasks,  # iterable of Task
        journal: Journal,
        completed_seq_num: int,
        completed_out: Out,
        cs: CompactionStrategy,
    ):
        tasks_by_id, record = {}, []
        self._create_persistable_tasks(queue, tasks, tasks_by_id, set())
        for task_id, task_and_ptask in tasks_by_id.items():
            task, ptask = task_and_ptask
            for kwarg, dependency in task.dependencies.items():
                tasks_by_id[id(dependency)][1].set_trigger(kwarg, ptask)
            if not task.dependencies:
                queue.push(ptask)
                record.append(ptask.__dict__)
        if completed_seq_num is not None:
            if cs and cs.record_garbage(1):
                return True
            record.append(
                completed_seq_num if completed_out.result is None else (completed_seq_num, completed_out.result)
            )
        if record:
            journal.record(record if len(record) > 1 else record[0])

    def _resolve_live(self, live_param: Live):
        key, ls = live_param.key, self.live_state
        if key is None:
            return ls
        return ls[key] if isinstance(ls, dict) else getattr(ls, key)

    def _run_tasks(
        self,
        executor: concurrent.futures.Executor,
        parallelism: int,
        queue: _TasksQueue,
        future2pt,
        journal: Journal,
        cs: CompactionStrategy,
    ) -> bool:  # returns False if there's nothing left to do
        future = self._start_futures_until_one_completes(future2pt, queue, executor, parallelism)
        if not future:  # all tasks were executed
            return False
        ptask = future2pt[future]
        del future2pt[future]
        out = self._get_task_output(future, queue, ptask)
        if out is None:  # was rescheduled to retry
            return True
        for ready_st in ptask.handle_triggers(out.result):
            queue.push(ready_st)
        if self._push(queue, out.tasks, journal, ptask.sequence_number, out, cs):
            journal.compact(
                ptask.prepare_pickling()
                for ptask in itertools.chain(queue.get_compaction_iterator(), future2pt.values())
            )
            cs.record_compaction()
        return True

    def _start_futures_until_one_completes(self, future2pt, queue, executor, parallelism):
        while True:
            if len(future2pt) >= parallelism:
                return self._wait(future2pt, None)
            st_or_time = queue.next(self)
            if isinstance(st_or_time, _PersistableTask):
                future2pt[self._submit(executor, st_or_time)] = st_or_time
            elif isinstance(st_or_time, numbers.Number):  # st_or_time is a time
                future = self._wait(future2pt, st_or_time)
                if future is not None:
                    return future
            else:  # none of the queue tasks is ready to start
                assert st_or_time is None
                return self._wait(future2pt, None) if future2pt else None

    def _submit(self, executor: concurrent.futures.Executor, task: Task) -> concurrent.futures.Future:
        args = (self._resolve_live(arg) if isinstance(arg, Live) else arg for arg in task.args)
        kwargs = dict((k, self._resolve_live(v)) if isinstance(v, Live) else (k, v) for k, v in task.kwargs.items())
        return executor.submit(task.what, *args, **kwargs)

    def _wait(self, futures, deadline_or_none):  # returns a future or None
        timeout = None if deadline_or_none is None else max(0.0, (deadline_or_none - self.get_time()))
        try:
            return next(concurrent.futures.as_completed(futures, timeout=timeout))
        except StopIteration:
            if timeout > 0.0:
                self.logger.info("sleeping %f seconds", timeout)
                self.sleep(timeout)
        except TimeoutError:
            return None
