import glob
import shutil
import json
import os
import random

ROOTDIR = "/data/ucb/ENCLAVE_DTV/"
IMAGE_DIRS = glob.glob(ROOTDIR + "unzip/images*")
IMAGE_DIR = ROOTDIR + "enclave/images"


def source_annotation_mapping(image_dir):
    sources = sorted(glob.iglob(image_dir + "/v1/*/production/*/*"))
    annotation = image_dir + "/mapp_meta_data.json"
    source_annotation_map = {}
    for source in sources:
        source_annotation_map[source] = annotation
    return source_annotation_map


source_annotation_map = {}
for directory in IMAGE_DIRS:
    source_annotation_map.update(source_annotation_mapping(directory))

if not os.path.isdir(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

orig_count = 0
annotations = []

sources = sorted(source_annotation_map.keys())
num_dir = len(sources)


def copy_images(source, target_dir, source_annotation_map):
    global orig_count

    image_list = []
    annotations = []
    images = sorted(glob.iglob(source + "seq_jpeg/*"))
    for image in images:
        shutil.copy(image, target_dir)
        image_list.append(os.path.basename(image))

    with open(source_annotation_map[source]) as orig_anns:
        anns = json.load(orig_anns)
        orig_count += len(anns)
        for entry in anns:
            if entry["frames"]["FileName"] in image_list:
                annotations.append(entry)

    return annotations


counter = 1
for source in sources:
    print("Processing directory {} of {}".format(counter, num_dir))
    annotations += copy_images(source, IMAGE_DIR, source_annotation_map)
    counter += 1


print("Original annotation count: {}".format(orig_count))
print("Final annotation count: {}".format(len(annotations)))

with open(ROOTDIR + "all_annotations.json", "w") as final:
    json.dump(annotations, final)
