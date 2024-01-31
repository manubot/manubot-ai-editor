from pathlib import Path
from unittest import mock

from .dir_union import mock_unify_open, set_directory

# tests for mock_unify_open

UNIFY_TEST_DIR = Path(__file__).parent / "dir_union_fixtures"
UNIFY_ORIG_DIR = UNIFY_TEST_DIR / "original"
UNIFY_PATCHED_DIR = UNIFY_TEST_DIR / "patched"

@mock.patch("builtins.open", mock_unify_open(UNIFY_ORIG_DIR, UNIFY_PATCHED_DIR))
def test_unify_folder_mock():
    # test that we can still open files in the original folder
    with open(UNIFY_ORIG_DIR / "test.txt") as fp:
        assert fp.read().strip() == "hello, world!"
    # test that the patched folder takes precedence
    with open(UNIFY_ORIG_DIR / "another.txt") as fp:
        assert fp.read().strip() == "patched in via unify mock"

@mock.patch("builtins.open", mock_unify_open(UNIFY_ORIG_DIR, UNIFY_PATCHED_DIR))
def test_unify_folder_mock_relative_paths():
    with set_directory(UNIFY_ORIG_DIR):
        # test that we can still open files in the original folder
        with open("./test.txt") as fp:
            assert fp.read().strip() == "hello, world!"
        # test that the patched folder takes precedence
        with open("./another.txt") as fp:
            assert fp.read().strip() == "patched in via unify mock"
        # test that subfolders in the patched folder can be used
        with open("./sub/third.txt") as fp:
            assert fp.read().strip() == "a third file"
