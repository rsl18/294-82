"""
coco_zero_index.py is a script that changes a coco formatted json to have 0 as background category
"""
# Standard Library imports:
import argparse
import copy
import json
from pathlib import Path
from typing import Dict

# 3rd Party imports:
from pycocotools.coco import COCO

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.config import h4dconfig
from h4dlib.data.cocohelpers import COCOShrinker


def main(opt):
    coco = COCO(opt.input_file)
    cats = coco.dataset["categories"]
    print("Cats: ", cats)
    print(coco.dataset.keys())
    print(coco.dataset["annotations"][0])
    is_zero_background_catid = coco_has_zero_as_background_id(coco)
    print("is_zero_background_catid: ", is_zero_background_catid)
    if not is_zero_background_catid:
        new_cats, new_anns = adjust_cat_ids(coco)
        print("")
        print(coco.dataset["annotations"][0])
        print(new_anns[0])
        root_json = {}
        root_json["categories"] = new_cats
        root_json["info"] = coco.dataset["info"]
        root_json["licenses"] = coco.dataset["licenses"]
        root_json["images"] = coco.dataset["images"]
        root_json["annotations"] = new_anns
        save(opt.input_file, root_json)


def save(input_file: Path, root_json: Dict) -> None:
    """Saves the json to the dest_path/dest_name location."""
    file_path = str(input_file).replace(".json", ".reindexed.json")
    print(f"Writing output to: '{file_path}'")
    with open(file_path, "w") as coco_file:
        coco_file.write(json.dumps(root_json))


def adjust_cat_ids(coco):
    cats = coco.dataset["categories"]
    new_cats = []
    for cat in cats:
        new_cat = {
            "supercategory": cat["supercategory"],
            "id": int(cat["id"]) + 1,
            "name": cat["name"],
        }
        new_cats.append(new_cat)
    print("new cats: ", new_cats)
    # Adjust annotations:
    anns = coco.dataset["annotations"]
    new_anns = copy.deepcopy(anns)
    for ann in new_anns:
        ann["category_id"] = ann["category_id"] + 1
    return new_cats, new_anns


def coco_has_zero_as_background_id(coco):
    """Return true if category_id=0 is either unused, or used for background class. Else return false."""
    cats = coco.dataset["categories"]
    cat_id_zero_nonbackground_exists = False
    for cat in cats:
        if cat["id"] == 0:
            if cat["name"] not in ["background", "__background__"]:
                cat_id_zero_nonbackground_exists = True
                break
    # id:0 isn't used for any categories, so by default can assume it can be used for background class:
    # if not cat_id_zero_nonbackground_exists:
    #     return True
    return not cat_id_zero_nonbackground_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        default="xview_coco_v2_tiny_train_chipped.json",
        help="Dataset to create stiny version of",
    )
    opt = parser.parse_args()
    opt.input_file = h4dconfig.DATA_DIR / "Xview/coco_chipped" / opt.input_file

    main(opt)
