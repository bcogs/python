import os
import pickle
import tempfile


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

    def __init__(self, state, state_filename=None):
        """Ctor.

        Parameters:
            state: Initial state to use when the state file is absent.
            state_filename: Initial state to use, if present.  It should contain a pickled object of the same type as state.
        """
        self.state_filename = state_filename
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
        with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tmp:
            pickle.dump(self.state, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, self.state_filename)
