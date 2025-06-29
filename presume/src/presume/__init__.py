import os
import pickle
import signal
import tempfile


class signals_masker(object):
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


class context(object):
    """Context that facilitates resuming a program from where it left after it exits.

    Works if it exits due to uncaught exceptions, SIGINT, SIGPIPE, or sys.exit.
    Can't do anything about most lethal signals (e.g. SIGTERM) or os._exit, unfortunately.

    Example use:
      class my_cool_class(object):

        def main(self, ...):
          if not hasattr(self, 'result0'): self.result0 = self.step0(...)
          if not hasattr(self, 'result1'): self.result1 = self.step1(...)

        def step0(...): ...

        def step1(...): ...

      with presume.context(my_cool_class(), state_filename='state_file') as p:
        # p.state is either:
        #   - the unpickled content of state_file if it exists
        #   - my_cool_class() otherwise
        p.state.main()
        # the above will run step0() then step1() unless those steps already succeeded and were persisted, which happens when the context exits
    """

    def __init__(self, state, state_filename=None, mask_signals=signals_masker.MASKABLE_TERMINATING_SIGNALS):
        """Ctor.

        Parameters:
            state: Initial state to use when the state file is absent.
            state_filename: Initial state to use, if present.  It should contain a pickled object of the same type as state.
            mask_signals: None or a set {signal.SIGTERM, ...} that will be masked while persisting state."
        """
        self.state_filename, self.mask_signals = state_filename, mask_signals
        if state_filename:
            try:
                with open(state_filename, "rb") as f:
                    self.state = pickle.load(f)
                    return
            except FileNotFoundError:
                pass
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Pickle self.state and save it to the state file."""
        if not self.state_filename:
            return
        dir_name = os.path.dirname(self.state_filename)
        mask = self.mask_signals if self.mask_signals else {}
        with signals_masker(signals_to_mask=mask):
            with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tmp:
                pickle.dump(self.state, tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, self.state_filename)


class iterator(object):
    """Iterator that can be pickled and thus persisted.

    Example use:
      with presume.context(...) as p:
        for n in p.state.__dict__.setdefault('it', presume.iterator([1, 2, 3])):
          do_something (n)
    """

    def __init__(self, sequence):
        self._index, self._sequence = 0, sequence

    def __iter__(self):
        if hasattr(self, "_index"):
            self._index -= 1
        return self

    def __next__(self):
        if not hasattr(self, "_index"):
            raise StopIteration
        self._index += 1
        if self._index >= len(self._sequence):
            del self._index
            del self._sequence
            raise StopIteration
        return self._sequence[self._index]

    def set_position(self, position=0):
        """Set the iterator to a given position.

        Musnt't be called after the iteration ends successfully.
        """
        if not hasattr(self, "_sequence"):
            # it isn't supported because we del self._sequence at the end of the
            # iteration, so we can't iterate again
            raise Exception("calling set_position after the iteration ended isn't supported")
        self._index = position
