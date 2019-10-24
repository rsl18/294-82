import json
from pathlib import Path

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: E402, E501, F401
from h4dlib.config import h4dconfig

def main(voc_path: Path):
    ANNOT_PATH = voc_path / "annotations"
    OUT_PATH = ANNOT_PATH
    INPUT_FILES = [
        "pascal_train2012.json",
        "pascal_val2012.json",
        "pascal_train2007.json",
        "pascal_val2007.json",
    ]
    OUTPUT_FILE = "pascal_trainval0712.json"
    KEYS = ["images", "type", "annotations", "categories"]
    MERGE_KEYS = ["images", "annotations"]

    out = {}
    tot_anns = 0
    for i, file_name in enumerate(INPUT_FILES):
        data = json.load(open(ANNOT_PATH / file_name, "r"))
        print("keys", data.keys())
        if i == 0:
            for key in KEYS:
                out[key] = data[key]
                print(file_name, key, len(data[key]))
        else:
            out["images"] += data["images"]
            for j in range(len(data["annotations"])):
                data["annotations"][j]["id"] += tot_anns
            out["annotations"] += data["annotations"]
            print(file_name, "images", len(data["images"]))
            print(file_name, "annotations", len(data["annotations"]))
        tot_anns = len(out["annotations"])
    print("tot", len(out["annotations"]))
    json.dump(out, open(OUT_PATH / OUTPUT_FILE, "w"))


if __name__ == "__main__":
    print(h4dconfig.DATA_DIR)
    DATADIR: Path = h4dconfig.DATA_DIR
    VOC_DIR: Path = DATADIR / "voc/voc_combined"
    print(VOC_DIR)
    main(VOC_DIR)