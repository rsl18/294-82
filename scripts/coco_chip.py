# Standard Library imports:
import io as _io
import json
import os
from pathlib import Path

# 3rd Party imports:
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.config import h4dconfig
from h4dlib.data.cocohelpers import *
import h4dlib.data.wv_util as wv


def generate_images_and_annotations(
    class_ids,
    input_json,
    source_imgs_dir,
    dest_imgs_dir,
    img_tag,
    width=300,
    height=300,
):
    # coco_images = {
    #   "license": 5,
    #   "file_name": "COCO_train2014_000000057870.jpg",
    #   "coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg",
    #   "height": 480,
    #   "width": 640,
    #   "date_captured": "2013-11-14 16:28:13",
    #   "flickr_url": "http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg",
    #   "id": 57870
    # }

    # coco_annotations = {
    #   "segmentation": [[312.29,562.89,402.25,511.49,400.96,425.38,398.39,372.69,388.11,332.85,318.71,325.14,295.58,305.86,269.88,314.86,258.31,337.99,217.19,321.29,182.49,343.13,141.37,348.27,132.37,358.55,159.36,377.83,116.95,421.53,167.07,499.92,232.61,560.32,300.72,571.89]],
    #   "area": 54652.9556,
    #   "iscrowd": 0,
    #   "image_id": 480023,
    #   "bbox": [116.95,305.86,285.3,266.03],
    #   "category_id": 58,
    #   "id": 86
    #   }
    # all of these arrays are the same length and correspond to all the entries in the geojson
    # so each image file appears multiple times in all_imgs.

    coco = COCO(input_json)

    ann_len = len(coco.anns)

    all_coords = np.zeros((ann_len, 4))
    all_chips = np.zeros(ann_len, dtype="object")
    all_classes = np.zeros(ann_len)

    i = 0
    for ann_id in coco.anns.keys():
        ann = coco.anns[ann_id]
        img_id = ann["image_id"]

        all_chips[i] = img_id

        bbox = ann["bbox"]
        coord = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        all_coords[i] = np.array([int(num) for num in coord])

        all_classes[i] = ann["category_id"]
        i += 1

    images = []
    file_index = 0
    annotation_index = 0
    annotations = []

    print("Chipping images")
    for img_id in tqdm(coco.imgs.keys()):
        img = coco.loadImgs(img_id)[0]
        I = io.imread("%s/%s" % (source_imgs_dir, img["file_name"]))

        chip_name = img["id"]
        coords = all_coords[all_chips == chip_name]
        classes = all_classes[all_chips == chip_name].astype(np.int64)
        chips, chip_boxes, chip_classes = wv.chip_image(
            I, coords, classes, shape=(width, height)
        )

        for i in range(len(chips)):
            image_dict = {}
            image_dict["license"] = 1
            image_dict["file_name"] = (
                "COCO_" + img_tag + "_" + str(file_index).zfill(12) + ".jpg"
            )
            image_dict["coco_url"] = ""
            image_dict["width"] = width
            image_dict["height"] = height
            image_dict["date_captured"] = "2018-02-22 00:00:00"
            image_dict["flickr_url"] = ""
            image_dict["id"] = file_index
            images.append(image_dict)
            new_image = convertToJpeg(chips[i])
            with open(
                os.path.join(
                    dest_imgs_dir,
                    "COCO_" + img_tag + "_" + str(file_index).zfill(12) + ".jpg",
                ),
                "wb",
            ) as image_file:
                image_file.write(new_image)

            for j in range(len(chip_boxes[i])):
                class_id = int(chip_classes[i][j])

                if not class_id in class_ids:
                    continue

                box = chip_boxes[i][j]
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                if x_min == y_min == x_max == y_max == 0:
                    continue

                x, y = int(x_min), int(y_min)
                w, h = int(x_max - x_min), int(y_max - y_min)

                annotation_dict = {}
                annotation_dict["bbox"] = [x, y, w, h]
                annotation_dict["segmentation"] = [
                    [x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]
                ]
                annotation_dict["area"] = w * h
                annotation_dict["iscrowd"] = 0
                annotation_dict["image_id"] = file_index
                annotation_dict["category_id"] = class_id
                annotation_dict["id"] = str(annotation_index)

                annotations.append(annotation_dict)
                annotation_index += 1
            file_index += 1

    return images, annotations


def generate_info():
    info_json = {
        "description": "XView Dataset",
        "url": "http://xviewdataset.org/",
        "version": "1.0",
        "year": 2018,
        "contributor": "Defense Innovation Unit Experimental (DIUx)",
        "date_created": "2018/02/22",
    }
    return info_json


def generate_licenses():
    licenses = []
    license = {
        "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
    }
    licenses.append(license)
    return licenses


def generate_categories(input_json):

    class_ids = []
    categories = []
    print(input_json)
    with open(input_json, "r") as coco_file:
        anns_file = json.load(coco_file)
        categories = anns_file["categories"]
        for entry in categories:
            class_ids.append(entry["id"])

    return categories, class_ids


def convertToJpeg(im):
    """
    (copied from tfr_util.py, so we don't have to import tensorflow)
    Converts an image array into an encoded JPEG string.

    Args:
        im: an image array

    Output:
        an encoded byte string containing the converted JPEG image.
    """
    with _io.BytesIO() as f:
        im = Image.fromarray(im)
        im.save(f, format="JPEG")
        return f.getvalue()


def main(datatype):

    print(h4dconfig.DATA_DIR)
    DATADIR: Path = h4dconfig.DATA_DIR

    INPUT_JSON = DATADIR / "Xview/coco_vanilla/{}_{}.json".format(
        h4dconfig.XVIEW_COCO_PREFIX, datatype
    )
    DEST_JSON = DATADIR / "Xview/coco_chipped/{}_{}_chipped.json".format(
        h4dconfig.XVIEW_COCO_PREFIX, datatype
    )
    SOURCE_IMGS_DIR = DATADIR / "Xview/coco_vanilla"
    DEST_IMGS_DIR = DATADIR / "Xview/coco_chipped/{}".format(datatype)
    IMG_TAG = datatype

    if not os.path.exists(DEST_IMGS_DIR):
        os.makedirs(DEST_IMGS_DIR)

    print(f"Processing {datatype}")
    with open(DEST_JSON, "w") as coco_file:
        root_json = {}
        categories_json, class_ids = generate_categories(INPUT_JSON)
        root_json["categories"] = categories_json
        root_json["info"] = generate_info()
        root_json["licenses"] = generate_licenses()
        images, annotations = generate_images_and_annotations(
            class_ids,
            INPUT_JSON,
            SOURCE_IMGS_DIR,
            DEST_IMGS_DIR,
            IMG_TAG,
            width=512,
            height=512,
        )
        root_json["images"] = images
        root_json["annotations"] = annotations
        coco_file.write(json.dumps(root_json))


if __name__ == "__main__":
    main("val")
    main("train")
