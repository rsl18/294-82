import csv
import time
import json


def seqgen():
    """ 
    Returns sequential integer generator object.
    """
    n = 0
    while True:
        yield n
        n += 1


# global sequential integer generators
imggen, anngen, catgen = seqgen(), seqgen(), seqgen()

EXCLUDED_F8 = ["aircraft", "missile_platform", "other_weapon", "small_arm"]


def enclave_to_coco(enclaveDS: list, read_CSV: str, JSON_path: str) -> None:

    """
    Convert enclave image dataset JSON into coco dataset json format.
    :param enclaveDS: the list of images (dictionaries) from the enclave dataset.
    :param read_CSV: path to the CSV file which details the scheme for remapping of f8_category to new label, new parent.
    :param JSON_path: desired output path for created coco dataset JSON.
    """
    # set up
    print("updating mapping...")
    tic = time.time()
    cocoDS: dict = dict()
    cocoDS["images"] = []
    cocoDS["annotations"] = []
    cocoDS["categories"] = []
    map_rows = []
    created_cats: dict = dict()
    # read the CSV
    with open(read_CSV, newline="") as csvfile:
        spamreader = csv.reader(csvfile, quotechar="|")
        for row in spamreader:
            map_rows.append(row)
    # skpping column headers row in CSV
    map_rows = map_rows[1:]
    total = 0
    hasnodim = 0
    is1280720 = 0
    # iterate through all images in enclave DS and extract info for coco dataset formatting
    for img in enclaveDS:
        img_id = next(imggen)
        cocoDS["images"].append(dict())
        currimg = cocoDS["images"][img_id]
        currimg["id"] = img_id
        currimg["date_created"] = img["CreationDate"]
        currimg["file_name"] = img["frames"]["FileName"]
        if "MP4" in img.keys():
            currimg["height"] = img["MP4"]["@height"]
            currimg["width"] = img["MP4"]["@width"]
        elif "src_metadata" in img["sequence_record"].keys():
            currimg["height"] = img["sequence_record"]["src_metadata"]["height"]
            currimg["width"] = img["sequence_record"]["src_metadata"]["width"]
        elif "metadata" in img["sequence_record"].keys():
            currimg["height"] = img["sequence_record"]["metadata"]["height"]
            currimg["width"] = img["sequence_record"]["metadata"]["width"]
        elif "feed" in img["Telemetry"].keys():
            if "height_pix" in img["Telemetry"]["feed"].keys():
                currimg["height"] = img["Telemetry"]["feed"]["height_pix"]
                currimg["width"] = img["Telemetry"]["feed"]["width_pix"]

        if "height" in currimg.keys():
            currimg["height"] = int(currimg["height"])
            currimg["width"] = int(currimg["width"])
        else:
            hasnodim += 1
        if currimg["height"] == 720 and currimg["width"] == 1280:
            is1280720 += 1
        total += 1
        skip = False
        for ann in img["frames"]["FrameLabel"]["annotations"]:
            target_f8 = ann["f8_category"]
            new_label, new_id = get_cat_id(target_f8, map_rows, created_cats, cocoDS)
            if new_label in EXCLUDED_F8:
                continue
            ann_ind = next(anngen)
            cocoDS["annotations"].append(dict())
            currann = cocoDS["annotations"][ann_ind]
            currann["id"] = ann_ind
            currann["bbox"] = [ann["x"], ann["y"], ann["width"], ann["height"]]
            currann["image_id"] = img_id
            currann["segmentation"] = [currann["bbox"]]
            currann["area"] = int(ann["width"]) * int(ann["height"])
            currann["iscrowd"] = 0
            currann["category_id"] = new_id
    # dump final coco JSON here
    print(
        "TOTAL: {}; HAS NO DIMENSION: {}; IS 1280x720: {}".format(
            total, hasnodim, is1280720
        )
    )
    print("writing to JSON...")
    with open(JSON_path, "w") as json_file:
        json.dump(cocoDS, json_file)
    # display final clock time
    stamp = time.time() - tic
    minutes = stamp // 60
    seconds = stamp % 60
    print("Done (t={} min {:0.2f} sec)".format(minutes, seconds))


def get_cat_id(target_f8: str, csv_rows: list, created_cats: dict, DS: dict) -> int:
    """
    Return category_id corresponding to target_f8 category if already created. If not created, create it and return its category_id.
    :param target_f8: f8_category from this individual annotation, corresponding to first column in csv.
    :param csv_rows: the already read csv file.
    :param created_cats: dictionary with f8_category-category_id key-value pairs to indicate whether category has already been created
    :param DS: current dataset being created for output JSON.
    """
    for row in csv_rows:
        curr_f8 = row[0]
        if curr_f8 == target_f8:
            NL, NP = row[2], row[3]
            if NL in EXCLUDED_F8:
                return NL, -1
            if created_cats.get((NL, NP)) is None:
                new_cat_id = make_cat_id(NL, NP, DS, created_cats, curr_f8)
            else:
                new_cat_id = get_existing_cat_id(NL, NP, DS)
            return NL, new_cat_id
    raise KeyError("no such f8_category")


def make_cat_id(NL: str, NP: str, DS: dict, created_cats: dict, f8_cat: str) -> int:
    """
    Create a new category, assign it a unique category id, add this category to created_cats dictionary and return this category id.
    :param NL: New Label value from csv lookup.
    :param NP: New Parent value from csv lookup.
    :param DS: current dataset being created for output JSON.
    :param f8_cat: f8_category value from enclave dataset 
    """
    new_cat = dict()
    new_cat["supercategory"] = NP
    new_cat["name"] = NL
    new_cat["id"] = next(catgen)
    DS["categories"].append(new_cat)
    created_cats[(NL, NP)] = new_cat["id"]
    return new_cat["id"]


def get_existing_cat_id(NL: str, NP: str, DS: dict) -> int:
    for cat in DS["categories"]:
        if cat["name"] == NL and cat["supercategory"] == NP:
            return cat["id"]
    raise KeyError("category not found")
