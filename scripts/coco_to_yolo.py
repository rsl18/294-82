"""
coco_to_yolo.py is a script to convert an MS COCO formatted dataset to VOC format, so it
can be used in yolov3.
"""
# Standard Library imports:
from pathlib import Path, PosixPath
from typing import List

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: E402, E501, F401
from h4dlib.config import h4dconfig
from h4dlib.data.cocohelpers import CocoToDarknet


def main():
    """Main function"""
    xview_path: Path = h4dconfig.DATA_DIR / "Xview/coco_chipped/"
    xview_datasets: List[PosixPath] = list(xview_path.glob("xview_coco*.json"))
    print(xview_datasets)
    for xview_dataset in xview_datasets:
        ds: PosixPath = xview_dataset
        ds_name, _, data_split = ds.name.replace("_chipped.json", "").rpartition("_")
        print("")
        print("================================================")
        print(f"Converting to yolo: dataset:{ds_name}, split: {data_split}")
        CocoToDarknet.convert(
            xview_dataset, h4dconfig.ROOT_DIR / "yolov3/data/", ds_name, data_split
        )


if __name__ == "__main__":
    main()
