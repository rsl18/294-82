"""
trainvalsplit.py is a script that splits an MS COCO formatted dataset into train and val
partitions. For sample usage, run from command line:

Example:
    python trainvalsplit.py --help
"""
# Standard Library imports:
from pathlib import Path

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.config import h4dconfig
from h4dlib.data.cocohelpers import CocoClassDistHelper, CocoJsonBuilder, split2

# Used to check the results of the split--all classes in both splits
# should have at least this many annotations:
_CLASS_COUNT_THRESHOLD = 20

# Seed value 341589 was chosen via the train-val-split-xviewcoco notebook:
_RANDOM_SEED = 201859

# Size of val split. The train split size will be 1 - _TEST_SIZE.
_TEST_SIZE = 0.3


def create_split(input_json, output_path, output_json_name):
    """
    Creates train/val split for the coco-formatted dataset defined by
    input_json. params: input_json: full path or Path object to coco-formatted
    input json file. output_path: full path or Path object to directory where
    outputted json will be saved. output_json_name:
    """
    coco = CocoClassDistHelper(input_json)

    # Map from image filenames to id's
    img_ids = coco.img_ids
    name_id_map = {}
    for id in img_ids:
        name = coco.loadImgs(id)[0]["file_name"]
        name_id_map[name] = id
    img_names = name_id_map.keys()

    # Map mp4 stream names (filename prefixes) to lists to list of filenames
    source_map = {}
    for name in img_names:
        prefix = name[::-1].split("-", 1)[1][::-1]
        if not prefix in source_map.keys():
            source_map[prefix] = [name]
        else:
            source_map[prefix] += [name]

    # Sort sources and images so random seeds in split2 are determinstic
    sources = sorted(source_map.keys())
    for source in sources:
        images = sorted(source_map[source])
        source_map[source] = images

    train_names, val_names = split2(
        source_map, sources, test_size=_TEST_SIZE, random_state=_RANDOM_SEED
    )
    train_img_ids = [name_id_map[name] for name in train_names]
    val_img_ids = [name_id_map[name] for name in val_names]

    train_counts, train_percents = coco.get_class_dist(train_img_ids)
    val_counts, val_percents = coco.get_class_dist(val_img_ids)

    # Generate coco-formatted json's for train and val:
    def generate_coco_json(coco, split_type, img_ids):
        coco_builder = CocoJsonBuilder(
            coco.cats, output_path, output_json_name.format(split_type)
        )
        for idx, img_id in enumerate(img_ids):
            img_entry = coco.imgs[img_id]
            img_entry['id'] = idx
            anns = []
            for ann in coco.imgToAnns[img_id]:
                new_ann = ann
                new_ann['image_id'] = idx
                anns.append(new_ann)
            coco_builder.add_image(img_entry, anns)
        coco_builder.save()

    generate_coco_json(coco, "train", train_img_ids)
    generate_coco_json(coco, "val", val_img_ids)
    return coco


def verify_output(original_coco, output_path, output_json_name):
    """
    Verify that the outputted json's for the train/val split can be loaded, and
    have correct number of annotations, and minimum count for each class meets
    our threshold.
    """

    def verify_split_part(output_json_name, split_part):
        json_path = output_path / output_json_name.format(split_part)
        print(f"Checking if we can load json via coco api:{json_path}...")
        coco = CocoClassDistHelper(json_path)
        counts, _ = coco.get_class_dist()
        assert min(counts.values()) >= _CLASS_COUNT_THRESHOLD, (
            f"min class count ({min(counts.values())}) is "
            + f"lower than threshold of {_CLASS_COUNT_THRESHOLD}"
        )
        print(f"{split_part} class counts: ", counts)
        return coco

    train_coco = verify_split_part(output_json_name, "train")
    val_coco = verify_split_part(output_json_name, "val")
    # assert len(original_coco.imgs) == len(train_coco.imgs) + len(
    #     val_coco.imgs
    # ), "Num Images in original data should equal sum of imgs in splits."
    # assert len(original_coco.anns) == len(train_coco.anns) + len(
    #     val_coco.anns
    # ), "Num annotations in original data should equal sum of those in splits."


def _main(opt):
    """
    Creates train/val split and verifies output.
    params:
        opt: command line options (there are none right now)
        output_json_name: format-string of output file names, with a '{}'
            style placeholder where split type will be inserted.
    """
    print(h4dconfig.DATA_DIR)
    datadir: Path = h4dconfig.DATA_DIR
    output_json_name = "ucb_coco_v1_{}.json"
    input_json = datadir / "enclave/coco/{}.json".format("ucb_coco_raw_v1")
    output_path = datadir / "enclave/coco"
    original_coco = create_split(input_json, output_path, output_json_name)
    verify_output(original_coco, output_path, output_json_name)


if __name__ == "__main__":
    opt = None
    # parser = argparse.ArgumentParser()
    # opt = parser.parse_args()
    _main(opt)
