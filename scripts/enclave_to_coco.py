"""
coco_to_yolo.py is a script to convert an MS COCO formatted dataset to VOC format, so it
can be used in yolov3.
"""
# Standard Library imports:
from pathlib import Path, PosixPath
import os
import csv
import json

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: E402, E501, F401
from h4dlib.config import h4dconfig
import remapenclave

def convert(input_path, csv_file, output_path):
    """Main function"""

    cocoDS: dict = dict()
    cocoDS["images"] = []
    cocoDS["annotations"] = []
    cocoDS["categories"] = []
    map_rows = []
    created_cats: dict = dict()
    dataset = json.load(open(input_path, 'r'))
    enclaveDS = dataset 
    read_CSV = csv_file
    JSON_path = output_path / 'ucb_coco_raw_v1.json'
    with open(read_CSV, newline='') as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            map_rows.append(row)
    map_rows = map_rows[1:]
    remapenclave.enclave_to_coco(enclaveDS, read_CSV, JSON_path)


if __name__ == "__main__":
    csv_file = h4dconfig.ROOT_DIR / 'ucbv1.csv'
    output_path: Path = Path("/h4d_root/ucb/ENCLAVE_DTV/coco")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path: Path = Path("/h4d_root/ucb/ENCLAVE_DTV/all_annotations.json")

    convert(input_path, csv_file, output_path)

