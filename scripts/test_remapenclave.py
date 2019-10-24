from remapenclave import *
from pathlib import Path

import os
import json
import csv

import _import_helper
from h4dlib.config import h4dconfig

ucb_path: Path = h4dconfig.DATA_DIR / "enclave/vanilla/all_annotations.json"
csv_file = h4dconfig.ROOT_DIR / 'ucbv1.csv'
output_path: Path = h4dconfig.DATA_DIR / "enclave/coco"
if not os.path.exists(output_path):
    os.makedirs(output_path)

cocoDS: dict = dict()
cocoDS["images"] = []
cocoDS["annotations"] = []
cocoDS["categories"] = []
map_rows = []
created_cats: dict = dict()
dataset = json.load(open(ucb_path, 'r'))
enclaveDS = dataset 
read_CSV = csv_file
JSON_path = output_path / 'ucb_coco_v1.json'
with open(read_CSV, newline='') as csvfile:
    spamreader = csv.reader(csvfile, quotechar='|')
    for row in spamreader:

        map_rows.append(row)
map_rows = map_rows[1:]

def test_enclave_to_coco():
    enclave_to_coco(enclaveDS, read_CSV, JSON_path)
    currpos = seqgen()
    outDS = json.load(open(JSON_path))
    dates_same = []
    filenames_same = []
    heights_same = []
    widths_same = []
    prev_img_id = -1
    img_ids_unique_seq = []
    bbox_dims_correct = []
    ann_id_matches = []
    ann_ind = 0
    cats_well_mapped = []
    coco_ann_id_img_id_match = []
    cat_ids_unique_seq = []
    total_num_imgs_same = []
    total_enclave_anns = 0
    for enclave_img in enclaveDS:
        coco_img = outDS["images"][next(currpos)]
        next_img_id = prev_img_id + 1
        dates_same.append(enclave_img["CreationDate"] == coco_img["date_created"])
        filenames_same.append(
            enclave_img["frames"]["FileName"] == coco_img["file_name"]
        )
        if "MP4" in enclave_img.keys():
            heights_same.append(int(enclave_img["MP4"]["@height"]) == coco_img["height"])
            widths_same.append(int(enclave_img["MP4"]["@width"]) == coco_img["width"])
        elif "src_metadata" in enclave_img["sequence_record"].keys():
            heights_same.append(int(enclave_img["sequence_record"]["src_metadata"]["height"]) == coco_img["height"])
            widths_same.append(int(enclave_img["sequence_record"]["src_metadata"]["width"]) == coco_img["width"])
        elif "metadata" in enclave_img["sequence_record"].keys():
            heights_same.append(int(enclave_img["sequence_record"]["metadata"]["height"]) == coco_img["height"])
            widths_same.append(int(enclave_img["sequence_record"]["metadata"]["width"]) == coco_img["width"])
        elif "feed" in enclave_img["Telemetry"].keys():
            if "height_pix" in enclave_img["Telemetry"]["feed"].keys():
                heights_same.append(int(enclave_img["Telemetry"]["feed"]["height_pix"]) == coco_img["height"])
                widths_same.append(int(enclave_img["Telemetry"]["feed"]["width_pix"]) == coco_img["width"])
        img_ids_unique_seq.append(coco_img["id"] == next_img_id)
        prev_img_id += 1
        for enclave_ann in enclave_img["frames"]["FrameLabel"]["annotations"]:
            total_enclave_anns += 1
            check_bbox = [
                enclave_ann["x"],
                enclave_ann["y"],
                enclave_ann["width"],
                enclave_ann["height"],
            ]
            coco_ann = outDS["annotations"][ann_ind]
            ann_id_matches.append(enclave_ann["id"] == coco_ann["id"])
            bbox_dims_correct.append(check_bbox == coco_ann["bbox"])
            ann_ind += 1
            for row in map_rows:
                if row[0] == enclave_ann["f8_category"]:
                    NL, NP = row[2], row[3]
                    for cat in outDS["categories"]:
                        if cat["id"] == coco_ann["category_id"]:
                            cats_well_mapped.append(
                                cat["super"] == NP and cat["name"] == NL
                            )
            coco_ann_id_img_id_match.append(coco_ann["image_id"] == coco_img["id"])
    prev_cat_id = -1
    unique_NL_NP_pairs = []
    for cat in outDS["categories"]:
        cat_ids_unique_seq.append(cat["id"] == prev_cat_id + 1)
        prev_cat_id += 1
        unique_NL_NP_pairs.append((cat["name"], cat["super"]))
    assert False not in dates_same
    assert False not in filenames_same
    assert False not in heights_same
    assert False not in widths_same
    assert False not in img_ids_unique_seq
    assert False not in ann_id_matches
    assert False not in bbox_dims_correct
    assert False not in cats_well_mapped
    assert False not in coco_ann_id_img_id_match
    assert False not in cat_ids_unique_seq
    assert sorted(unique_NL_NP_pairs) == sorted(set(unique_NL_NP_pairs))
    assert len(enclaveDS) == len(outDS["images"])
    assert total_enclave_anns == len(outDS["annotations"])
    assert len(outDS["categories"]) <= len(map_rows)
