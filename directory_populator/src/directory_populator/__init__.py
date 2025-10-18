import dataclasses
import os
import shutil
import tempfile


@dataclasses.dataclass
class DirectoryPopulator(object):
    """Context manager that creates a temporary directory on enter, and renames it to a target path on exit if there was no exception.

    The renaming of the directories isn't atomic.  Providing equivalent functionality atomically would require something less convenient to use and/or less portable, with a file or symlink pointing to the proper directory.
    """

    target: str  # final name of the directory
    archive: str = ""  # if the target already exists, rename it to this path before renaming the temp dir to the target
    tmp_prefix: str = None  # prefix of the temp dir name, same as tempfile.mkstemp
    tmp_suffix: str = None  # end of the temp dir name, same as tempfile.mkstemp
    tmp_parent: str = None  # directory containing the temp dir, same as tempfile.mkstemp's dir
    chdir: bool = True  # if True, chdir to the temp dir when entering the context, and chdir back to the original cwd when exiting it; if False, the path of the temporary directory is in the tmp_dir member

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix=self.tmp_prefix, suffix=self.tmp_suffix, dir=self.tmp_parent)
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
