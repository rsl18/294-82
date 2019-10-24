"""Add h4d proj root to sys.path so we can import stuff that's in ./h4d/h4dlib. what if
we make this line a little longer.

Example: suppose you want to import something from ./h4d/h4dlib/submodule/, from a
script that's called from command line.

    import _import_helper  # pylint: disable=unused-import # noqa: F401
    import h4dlib.submodule

    or:

    import _import_helper  # pylint: disable=unused-import # noqa: F401
    from h4dlib.submodule import foo


Note: The '#pylint:...' parts are to suppress linter warnings. Don't remove.

Note 2: _import_helper starts w/ underscore so that if imports are auto-sorted
_import_helper gets imported before any h4dlib imports. This ensures the below sys.path
hack is in place by the time we try to import anything h4dlib
"""
import subprocess
import sys
from pathlib import Path


def add_path(path: Path):
    path = str(path)
    if path not in sys.path:
        sys.path.append(path)


root = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    .strip()
    .decode("utf-8")
)
add_path(root)
add_path(root / "centernet-master/src/tools/voc_eval_lib/utils")
add_path(root / "centernet-master/src/tools/voc_eval_lib/datasets")
add_path(root / "centernet-master/src/lib")
# print("PATH::::::", sys.path)
