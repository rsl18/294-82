
"""
When you import this, you will use the h4dconfig object.

Example usage:
    from h4d.config import h4dconfig
    print("Data dir: ", h4dconfig.DATA_DIR)

Enumerate the h4dconfig values availabe to you:
    print(h4dconfig)
"""
from pathlib import Path
from easydict import EasyDict

__C = EasyDict()
h4dconfig: EasyDict = __C

# Root directory of project
__C.ROOT_DIR = (Path(__file__).parent / "..").absolute().resolve()

# Data directory
__C.DATA_DIR = __C.ROOT_DIR / 'datasets'

# Experiments directory
__C.EXPERIMENTS_DIR = __C.ROOT_DIR / 'experiments'

__C.XVIEW_COCO_PREFIX = "xview_coco_v2"
