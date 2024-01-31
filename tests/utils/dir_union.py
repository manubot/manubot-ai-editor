import os
from pathlib import Path
from unittest import mock

from contextlib import contextmanager

@contextmanager
def set_directory(new):
    """
    Given a path, sets it as the current working directory,
    then sets it back once the context has been exited.

    Note that if we upgrade to Python 3.11, this method can be replaced
    with https://docs.python.org/3/library/contextlib.html#contextlib.chdir
    """
    
    # store the current path so we can return to it
    original = Path().absolute()

    try:
        os.chdir(new)
        yield
    finally:
        os.chdir(original)

def mock_unify_open(original, patched):
    """
    Given paths to an 'original' and 'patched' folder,
    patches open() to first check the patched folder for the
    target file, then checks the original folder if it's not found
    in the patched folder.
    """
    builtin_open = open

    def unify_open(*args, **kwargs):
        try:
            # first, try to open the file from within patched

            # resolve all paths: the original, patched, and requested file
            target_full_path = Path(args[0]).absolute()
            rewritten_path = (
                str(target_full_path)
                    .replace(
                        str(original.absolute()),
                        str(patched.absolute())
                    )
            )

            return builtin_open(rewritten_path, *(args[1:]), **kwargs)
        except FileNotFoundError:
            # resort to opening it normally
            return builtin_open(*args, **kwargs)

    return unify_open
