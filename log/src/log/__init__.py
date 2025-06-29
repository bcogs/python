import os
import sys
import tempfile
import traceback
import time

from typing import Any

DBG = 0  # or negative for verbose debug
INFO = 1  # lowest level written to file by default, if a log file is setup
TELL = 2  # lowest level written to stderr by default
WARN = 3
ERR = 4
FATAL = 5  # by default, exits the program after logging
LEVELS = ("debug", "info", "tell", "warning", "error", "fatal")


def setup_default(*args, **kwargs):
    """Initialize the default logger, used for the module top level logging functions."""
    global default_logger
    default_logger = logger(file_sink(*args, **kwargs))
    return default_logger


def setup_default_as_ram_sink():
    """Initialize the default logger with a ram sink and return the sink.

    Meant as a convenience function for unit tests that want to check the content of the logs.
    Can be called multiple times, resetting the ram sink each time."""
    global default_logger
    default_logger = logger(ram_sink())
    return default_logger


def log(*args, **kwargs):
    """Log at an arbitrary level, args are the same as those of make()."""
    default_logger.log(*args, **kwargs)


def v(*args, **kwargs):
    """Log at a verbose debug level, the higher the more verbose."""
    default_logger.v(*args, **kwargs)


def dbg(*args, **kwargs):
    """Log at level debug."""
    default_logger.dbg(*args, **kwargs)


def info(*args, **kwargs):
    """Log at level info."""
    default_logger.info(*args, **kwargs)


def tell(*args, **kwargs):
    """Log at level tell."""
    default_logger.tell(*args, **kwargs)


def warn(*args, **kwargs):
    """Log at level warning."""
    default_logger.warn(*args, **kwargs)


def err(*args, **kwargs):
    """Log at level error."""
    default_logger.err(*args, **kwargs)


def fatal(*args, **kwargs):
    """Log at level fatal, which exits by default."""
    default_logger.fatal(*args, **kwargs)


class logger(object):
    """Main logging class."""

    def __init__(self, sink):
        """Ctor.

        Parameters:
          sink: any object with a log(self, level, fmt, *args, **kwargs) method,
                it will be used to actually write the messages; the log() arguments
                are the same as those of make()
        """
        self.sink = sink

    def die(self):
        """Called after logging at level fatal(), exits the program."""
        sys.exit(1)

    def log(self, level, fmt, *args, **kwargs):
        """Log at an arbitrary level, args are the same as those of make()."""
        self.sink.log(level, fmt, *args, **kwargs)
        if level >= FATAL:
            self.die()

    def v(self, level: int, fmt: str, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at a verbose debug level, the higher the more verbose."""
        self.log(-level, fmt, *args, **kwargs)

    def dbg(self, fmt, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at level debug."""
        self.log(DBG, fmt, *args, **kwargs)

    def info(self, fmt, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at level info."""
        self.log(INFO, fmt, *args, **kwargs)

    def tell(self, fmt, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at level tell."""
        self.log(TELL, fmt, *args, **kwargs)

    def warn(self, fmt, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at level warning."""
        self.log(WARN, fmt, *args, **kwargs)

    def err(self, fmt, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at level error."""
        self.log(ERR, fmt, *args, **kwargs)

    def fatal(self, fmt, *args: list[Any], **kwargs: dict[str, Any]):
        """Log at level fatal and call die()."""
        self.log(FATAL, fmt, *args, **kwargs)


def stack_trace(e: BaseException) -> str:
    """Stringify an exception as a python stack trace."""
    return "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()


class stack_trace_logger(object):
    """Context manager that writes the stack trace to logs if there's an exception.

    Example use:
      with log.stack_trace_logger(log_func=log.err): do_something()
    """

    def __init__(self, log_func=err):
        self.log_func = log_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, ext_tb):
        if exc_type is not None:
            self.log_func("%s", stack_trace(exc_value))


def make(level: int, fmt: str, *args, **kwargs) -> (int, str):
    """Turn a format and args into a single string, handling bad arguments well.

    Parameters:
      level: one of DBG, INFO etc or a negative int representing a verbose debug level (the larger the absolute value, the more verbose)
      fmt: format string
      args, kwargs: at most one should be non-0, the string will be fmt % that_one

    Returns:
      a pair (level, message) where the level could be different from the level supplied in argument in case of formatting error
    """
    try:
        msg = fmt % (args if args else kwargs)
        if level <= FATAL:
            return level, msg
        level, msg = FATAL, ("(invalid log level %d) " % level) + msg
    except Exception:
        level = max(min(level, FATAL), ERR)
        msg = f"log format error - level={level!r}, fmt={fmt!r}, args={args!r}, kwargs={kwargs!r}"
    caller_stack = "".join(traceback.format_stack(limit=10)[:-1])
    return level, msg + "\n at " + caller_stack


class file_sink(object):
    """Default sink that logs to files and stderr.

    It's also a context manager, it flushes and closes any open file when exiting.
    """

    def __init__(self, file_path=None, stderr_level=TELL, file_level=DBG):
        self.stderr_level, self.file_level = stderr_level, file_level
        if file_path is None:
            self.file = tempfile.NamedTemporaryFile(
                dir=tempfile.gettempdir(),
                delete=False,
                prefix=os.path.basename(sys.argv[0]) + "." + str(os.getpid()) + "." + time.strftime("%y%m%d-%H%M%S"),
            )
            return
        self.file = open(file_path, "ab") if file_path else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._close()

    def __del__(self):
        self._close()

    def _close(self):
        if not self.__dict__.get("file", None):
            return
        try:
            self.file.flush()
        except BaseException:
            pass
        try:
            self.file.close()
        except BaseException:
            pass
        self.file = None

    def log(self, level, fmt, *args, **kwargs):
        """Log a message, args are the same as those of make()."""
        if level < self.stderr_level and level < self.file_level:
            return
        level, msg = make(level, fmt, *args, **kwargs)
        msg = LEVELS[max(level, 0)][0].upper() + time.strftime("%m%d %H:%M:%S ") + msg + "\n"
        if level >= self.stderr_level:
            sys.stderr.write(msg)
        if level >= self.file_level and self.file:
            self.file.write(msg.encode("utf-8", errors="surrogateescape"))
            self.file.flush()


class ram_sink(object):
    """In memory sink, useful for unit tests that want to check log content."""

    def __init__(self):
        self.logs = []  # (level, message) pairs

    def log(self, level, fmt, *args, **kwargs):
        self.logs.append(make(level, fmt, *args, **kwargs))


default_logger = logger(file_sink())
