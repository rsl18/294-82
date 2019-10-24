"""
create_remap.py is a script that remaps categories for a COCO formatted dataset.
Remappings are provided by a .csv file.
"""
from pathlib import Path

from pycocotools.coco import COCO

import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.config import h4dconfig

folder = Path('../datasets/Xview/coco_vanilla')
old_ds_annotations = folder / 'xview_coco_v0.json'
csv_file = folder / 'xview_{}.csv'.format(h4dconfig.XVIEW_COCO_PREFIX[-2:])
JSON_path = folder / '{}.json'.format(h4dconfig.XVIEW_COCO_PREFIX)

coco = COCO(old_ds_annotations, False, csv_file, JSON_path)
