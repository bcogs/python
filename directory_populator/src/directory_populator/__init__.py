import dataclasses
import datetime
import os
import shutil
import tempfile
import time
import zoneinfo


# shorthand strftime constants
DATE = "%Y-%m-%d"
TIME = "$H:%M:%S"
DATETIME = DATE + "-" + TIME


@dataclasses.dataclass
class DirectoryPopulator(object):
    """Context manager that creates a temporary directory on enter, and renames it to a target path on exit if there was no exception.

    The renaming of the directories isn't atomic.  Providing equivalent functionality atomically would require something less convenient to use and/or less portable, with a file or symlink pointing to the proper directory.
    """

    target: str  # final name of the directory
    archive: str = ""  # if the target already exists, rename it to this path before renaming the temp dir to the target
    tmp_prefix: str = ""  # prefix of the temp dir name, same as tempfile.mkstemp
    tmp_suffix: str = ""  # end of the temp dir name, same as tempfile.mkstemp
    tmp_parent: str = ""  # directory containing the temp dir, same as tempfile.mkstemp's dir
    chdir: bool = True  # if True, chdir to the temp dir when entering the context, and chdir back to the original cwd when exiting it; if False, the path of the temporary directory is in the tmp_dir member
    tmp_tz: str = "UTC"  # if non-empty, tmp_prefix and tmp_suffix will be interpreted with strftime in the given timezone (use "local" for the default tz)

    def __enter__(self):
        if self.tmp_tz:
            if self.tmp_tz != "local":
                now = datetime.datetime.now(zoneinfo.ZoneInfo(self.tmp_tz)).timetuple()
            else:
                now = time.localtime()
            prefix, suffix = time.strftime(self.tmp_prefix, now), time.strftime(self.tmp_suffix, now)
        else:
            prefix, suffix = self.tmp_prefix, self.tmp_suffix
        self.tmp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=self.tmp_parent)
        if self.chdir:
            self._original_cwd = os.getcwd()
            os.chdir(self.tmp_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        original_cwd = self.__dict__.get("_original_cwd", None)
        if original_cwd:
            os.chdir(original_cwd)
        if exc_type is None and not self.__dict__.get("committed", False):
            self.commit()

    def commit(self):
        if self.__dict__.get("committed", False):
            raise Exception("repeated attempt to commit DirectoryPopulator(target=%r)" % self.target)
        to_del = None
        try:
            if self.archive:
                # if the archive exists, likely a previous run was interrupted
                # while moving the directories; better investigate than break
                # stuff
                assert not os.path.exists(self.archive)
                os.rename(self.target, self.archive)
                to_del = self.archive
            else:
                shutil.rmtree(self.target)
        except FileNotFoundError:
            pass
        os.rename(self.tmp_dir, self.target)
        if to_del:
            shutil.rmtree(to_del)
        self.committed = True
