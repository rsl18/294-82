"""
create_tiny_dataset.py is a script that helps create toy-sized versions of a dataset.
"""
# Standard Library imports:
import argparse
from pathlib import Path
from typing import Dict

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.config import h4dconfig
from h4dlib.data.cocohelpers import COCOShrinker


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ucb_coco_v1",
        help="Dataset to create stiny version of",
    )
    opt = parser.parse_args()

    # TODO: #1 Might want to move these dicts into a standalone file / class in h4dlib,
    # and if we add more info to this dictionary it could be an easy way to select
    # between different datasets from any of our training frameworks (e.g., rcnn, yolo,
    # ssd). Maybe a .cfg file.
    # TODO: #2: maybe make the values in the dict something more structured, like
    # a class or a named tuple, and that way could make it a single dict, rather than
    # two. The values could differentiate between train/val properties.
    ucb_coco_dir = h4dconfig.DATA_DIR / "enclave/coco"
    ds_paths_train: Dict[str, Path] = {
        "ucb_coco_v1": ucb_coco_dir / "ucb_coco_v1_train.json"
    }
    ds_paths_val: Dict[str, Path] = {
        "ucb_coco_v1": ucb_coco_dir / "ucb_coco_v1_val.json"
    }

    def get_tiny_filename(big_name: str, big_filename: str):
        return str(big_filename).replace(big_name, f"{big_name}_tiny")

    # Shrink:
    COCOShrinker(ds_paths_train[opt.dataset_name]).shrink(
        get_tiny_filename(opt.dataset_name, ds_paths_train[opt.dataset_name]), size=128
    )
    COCOShrinker(ds_paths_val[opt.dataset_name]).shrink(
        get_tiny_filename(opt.dataset_name, ds_paths_val[opt.dataset_name]), size=64
    )


if __name__ == "__main__":
    main()
