"""
cocohelpers is a module with helper classes and functions related to the MS
COCO API. Includes helpers for building COCO formatted json, inspecting class
distribution, and generating a train/val split.
"""
# Standard Library imports:
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 3rd Party imports:
import numpy as np
from pycocotools.coco import COCO

__all__ = ["CocoJsonBuilder", "COCOShrinker", "CocoClassDistHelper", "split"]


class CocoJsonBuilder(object):
    """
    A class used to help build coco-formatted json from scratch.
    """

    def __init__(self, categories: List[Dict[str, object]], dest_path="", dest_name=""):
        """
        Args:
            categories: this can be the COCO.dataset['categories'] property if you
                are building a COCO json derived from an existing COCO json and don't
                want to modify the classes. It's a list of dictionary objects. Each dict has
                three keys: "id":int = category id, "supercatetory": str = name of parent
                category, and a "name": str = name of category.
            dest_path: str or pathlib.Path instance, holding the path to directory where
                the new COCO formatted annotations
            file (dest_name) will be saved.
            dest_name: str of the filename where the generated json will be saved to.
        """
        self.categories = categories
        self.dest_path = Path(dest_path)
        self.dest_name = dest_name
        self.images = []
        self.annotations: List[Dict[str, object]] = []
        assert self.dest_path.exists(), f"dest_path: '{self.dest_path}' does not exist"
        assert (
            self.dest_path.is_dir()
        ), f"dest_path: '{self.dest_path}' is not a directory"

    def generate_info(self) -> Dict[str, str]:
        """returns: dictionary of descriptive info about the dataset."""
        info_json = {
            "description": "XView Dataset",
            "url": "http://xviewdataset.org/",
            "version": "1.0",
            "year": 2018,
            "contributor": "Defense Innovation Unit Experimental (DIUx)",
            "date_created": "2018/02/22",
        }
        return info_json

    def generate_licenses(self) -> Dict[str, str]:
        """Returns the json hash for the licensing info."""
        return [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ]

    def add_image(self, img: Dict[str, Any], annotations: List[Dict]) -> None:
        """
        Add an image and it's annotations to the coco json.

        Args:
            img: A dictionary of image attributes. This gets added verbatim to the
                json, so in typical use cases when you are building a coco json from an
                existing coco json, you would just pull the entire coco.imgs[img_id]
                object and pass it as the value for this parameter.
            annotations: annotations of the image to add. list of dictionaries.
                Each dict is one annotation, it contains all the properties of the
                annotation that should appear in the coco json. For example, when using
                this json builder to build JSON's for a train/val split, the
                annotations can be copied straight from the coco object for the full
                dataset, and passed into this parameter.

        Returns: None
        """
        self.images.append(img)
        for ann in annotations:
            self.annotations.append(ann)

    def get_json(self) -> Dict[str, object]:
        """Returns the full json for this instance of coco json builder."""
        root_json = {}
        root_json["categories"] = self.categories
        root_json["info"] = self.generate_info()
        root_json["licenses"] = self.generate_licenses()
        root_json["images"] = self.images
        root_json["annotations"] = self.annotations
        return root_json

    def save(self) -> None:
        """Saves the json to the dest_path/dest_name location."""
        file_path = self.dest_path / self.dest_name
        print(f"Writing output to: '{file_path}'")
        root_json = self.get_json()
        with open(file_path, "w") as coco_file:
            coco_file.write(json.dumps(root_json))


class COCOShrinker:
    """Shrinker takes an MS COCO formatted dataset and creates a tiny version of it.
    """

    def __init__(self, dataset_path: Path) -> None:
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path

    def shrink(self, target_filename: str, size: int = 512) -> None:
        """
        Create a toy sized version of dataset so we can use it just for testing if code
        runs, not for real training.

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Nothing, but the output dataset is saved to disk in the same directory
            where the input .json lives, with the same filename but with "_tiny" added
            to the filename.
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        dest_path: Path = self.base_path / target_filename
        print(
            f"Creating subset of {self.dataset_path}, of size: {size}, at: {dest_path}"
        )
        coco = COCO(self.dataset_path)
        builder = CocoJsonBuilder(
            coco.dataset["categories"], dest_path.parent, dest_path.name
        )
        subset_img_ids = coco.getImgIds()[:size]
        for img_id in subset_img_ids:
            builder.add_image(coco.imgs[img_id], coco.imgToAnns[img_id])
        builder.save()
        return dest_path


class CocoClassDistHelper(COCO):
    """
    A subclass of pycococtools.coco that adds a method(s) to calculate class
    distribution.
    """

    def __init__(
        self,
        annotation_file: str = None,
        create_mapping: bool = False,
        mapping_csv: str = None,
        write_to_JSON: bool = None,
    ):
        super().__init__(annotation_file, create_mapping, mapping_csv, write_to_JSON)
        # list of dictionaries. 3 keys each: (supercategory, id, name):
        self.cats = self.loadCats(self.getCatIds())
        list.sort(self.cats, key=lambda c: c["id"])
        # Dictionaries to lookup category and supercategory names from category
        # id:
        self.cat_name_lookup = {c["id"]: c["name"] for c in self.cats}
        self.supercat_name_lookup = {c["id"]: c["supercategory"] for c in self.cats}
        # List of integers, image id's:
        self.img_ids = self.getImgIds()
        # List of strings, each is an annotation id:
        self.ann_ids = self.getAnnIds(imgIds=self.img_ids)
        self.anns_list = self.loadAnns(self.ann_ids)
        print(f"num images: {len(self.img_ids)}")
        # print(F"num annotation id's: {len(self.ann_ids)}")
        print(f"num annotations: {len(self.anns)}")
        #         print(F"First annotation: {self.anns[0]}")
        #         Create self.img_ann_counts, a dictionary keyed off of img_id. For
        #         each img_id it stores a collections.Counter object, that has a count
        #         of how many annotations for each category/class there are for that
        #         img_id
        self.img_ann_counts = {}
        for img_id in self.imgToAnns.keys():
            imgAnnCounter = Counter({cat["name"]: 0 for cat in self.cats})
            anns = self.imgToAnns[img_id]
            for ann in anns:
                imgAnnCounter[self.cat_name_lookup[ann["category_id"]]] += 1
            self.img_ann_counts[img_id] = imgAnnCounter
        self.num_cats = len(self.cats)

    def get_class_dist(self, img_ids: List[int] = None):
        """
        Args:
            img_ids: List of image id's. If None, distribution is calculated for
                all image id's in the dataset.

        Returns: A dictionary representing the class distribution. Keys are category
            names Values are counts (e.g., how many annotations are there with that
            category/class label) np.array of class percentages. Entries are sorted by
            category_id (same as self.cats)
        """
        cat_counter = Counter({cat["name"]: 0 for cat in self.cats})
        if img_ids is None:
            img_ids = self.imgToAnns.keys()

        for img_id in img_ids:
            if img_id not in self.imgToAnns:
                continue
            cat_counter += self.img_ann_counts[img_id]

        # Convert to np array where entries correspond to cat_id's sorted asc.:
        total = float(sum(cat_counter.values()))
        cat_names = [c["name"] for c in self.cats]
        cat_percents = np.zeros((self.num_cats))
        for idx, cat_name in enumerate(sorted(cat_names)):
            cat_percents[idx] = cat_counter[cat_name] / total

        return cat_counter, cat_percents


def split(
    data: List, test_size: float = 0.2, random_state=None
) -> Tuple[List[Any], List[Any]]:
    """
    Similar to scikit learn, creates train/test splits of the passed in data.

    Args:
        data: A list or iterable type, of data to split.
        test_size: value in [0, 1.0] indicating the size of the test split.
        random_state: an int or RandomState object to seed the numpy randomness.

    Returns: 2-tuple of lists; (train, test), where each item in data has been placed
        into either the train or test split.
    """
    n = len(data)
    num_test = int(np.ceil(test_size * n))
    #     print(F"n:{n}, num_test:{num_test}, num_train:{num_train}")
    np.random.seed(random_state)
    test_idx = set(np.random.choice(range(n), num_test))
    data_test, data_train = list(), list()
    for idx, datum in enumerate(data):
        if idx in test_idx:
            data_test.append(data[idx])
        else:
            data_train.append(data[idx])
    return data_train, data_test


def split2(
    source_map: Dict,
    sources: List,
    test_size: float = 0.2,
    random_state=None,
    sample_rate: float = 0.05,
) -> Tuple[List[Any], List[Any]]:
    """
    Similar to scikit learn, creates train/test splits of the passed in data.
    Assumes that splits need to be senstive to input source (name prefix). Checks by first 
    mapping and splitting data sources with seed. Then samples randomly within each
    source with the seed.

    Args:
        source_map: A dictionary of source prefixes mapped to a list of sorted (for deterministic splits) image file names.
        source: A sorted list of source prefixes (for deterministic splits) 
        test_size: value in [0, 1.0] indicating the size of the test split.
        random_state: an int or RandomState object to seed the numpy randomness.
        sample_rate: float in [0,1.0] dictating 

    Returns: 2-tuple of lists; (train, test), where each item in data has been placed
        into either the train or test split.
    """

    num_sources = len(sources)
    num_test = int(np.ceil(test_size * num_sources))

    np.random.seed(random_state)
    test_source_idxs = set(np.random.choice(range(num_sources), num_test))

    def sample_from_source(images):
        num_images = len(images)
        num_sample = int(np.ceil(sample_rate * num_images))
        np.random.seed(random_state)
        sample_image_idx = set(np.random.choice(range(num_images), num_sample))
        data_test = list()
        for idx, image in enumerate(images):
            if idx in sample_image_idx:
                data_test.append(images[idx])
        return data_test

    data_test, data_train = list(), list()
    for idx, datum in enumerate(sources):
        if idx in test_source_idxs:
            data_test += sample_from_source(source_map[sources[idx]])
        else:
            data_train += sample_from_source(source_map[sources[idx]])

    return data_train, data_test


@dataclass
class bbox:
    """
    Data class to store a bounding box annotation instance
    """

    img_id: int
    cat_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class Img:
    """A helper class to store image info and annotations."""

    anns: List[bbox]

    def __init__(self, id: int, filename: str, width: float, height: float) -> None:
        self.id: int = id
        self.filename: str = filename
        self.width: float = width
        self.height: float = height
        self.anns = []

    def add_ann(self, ann: bbox) -> None:
        """Add an annotation to the image"""
        self.anns.append(ann)

    def get_anns(self) -> List[bbox]:
        """
        Gets annotations, possibly filters them in prep for converting to yolo/Darknet
        format.
        """
        return self.anns

    def to_darknet(self, box: bbox) -> bbox:
        """Convert a BBox from coco to Darknet format"""
        # COCO bboxes define the topleft corner of the box, but yolo expects the x/y
        # coords to reference the center of the box. yolo also requires the coordinates
        # and widths to be scaled by image dims, down to the range [0.0, 1.0]
        return bbox(
            self.id,
            box.cat_id,
            (box.x_center + (box.width / 2.0)) / self.width,
            (box.y_center + (box.height / 2.0)) / self.height,
            box.width / self.width,
            box.height / self.height,
        )

    def write_darknet_anns(self, label_file) -> None:
        """Writes bounding boxes to specified file in yolo/Darknet format"""
        # It's a bit leaky abstraction to have Img handle writing to file but it's
        # convenient b/c we have access to img height and width here to scale the bbox
        # dims. Same goes for .to_darknet()
        anns = self.get_anns()
        for box in anns:
            box = self.to_darknet(box)
            label_file.write(
                f"{box.cat_id} {box.x_center} {box.y_center} {box.width} {box.height}\n"
            )

    def has_anns(self) -> bool:
        """
        Returns true if this image instance has at least one bounding box (after any
        filters are applied)
        """
        # TODO: Can add filter to only return true if annotations have non-zero area: I
        # saw around ~5 or 6 annotations in the v2_train_chipped.json that had zero
        # area, not sure if those might cause problems for yolo
        return self.anns

    def get_label_path(self, base_path: Path) -> str:
        return base_path / self.filename.replace("jpeg", "txt").replace("jpg", "txt")

    def get_img_path(self, base_path: Path, dataset_name: str, data_split: str) -> str:
        return (
            base_path
            / dataset_name.replace("_tiny", "")
            / "images"
            / data_split
            / self.filename
        )


class CocoToDarknet:
    """Class that helps convert an MS COCO formatted dataset to yolo/Darknet format"""

    @staticmethod
    def convert(
        ann_path: Path, base_path: Path, dataset_name: str, data_split: str
    ) -> None:
        """Convert specified dataset to Darknet format.

        Details:
            - Labels are written to base_path/<dataset_name>/labels/<data_split>/*.txt
            - A file containing list of category names, is written to
                <base_path>/<dataset_name>.names
        """
        coco = COCO(ann_path)
        images = CocoToDarknet.build_db(coco)
        # Make paths:
        labels_path = base_path / dataset_name / "labels" / data_split
        labels_path.mkdir(parents=True, exist_ok=True)
        names_path = base_path / f"{dataset_name}.names"
        image_paths = CocoToDarknet.generate_label_files(
            images, labels_path, base_path, dataset_name, data_split
        )
        CocoToDarknet.generate_image_list(
            base_path, dataset_name, image_paths, data_split
        )
        CocoToDarknet.generate_names(names_path, coco)

    @staticmethod
    def generate_names(names_path: Path, coco: COCO) -> None:
        categories = [c["name"] + "\n" for c in coco.dataset["categories"]]
        with open(names_path, "w") as names_file:
            names_file.writelines(categories)

    @staticmethod
    def generate_label_files(
        images: Dict[int, Img],
        labels_path: Path,
        base_path: Path,
        dataset_name: str,
        data_split: str,
    ) -> List[str]:
        """
        Generates one .txt file for each image in the coco-formatted dataset. The .txt
        files contain the annotations in yolo/Darknet format.
        """
        # Convert:
        img_paths = set()
        for img_id, img in images.items():
            if img.has_anns():
                label_path = labels_path / img.get_label_path(labels_path)
                with open(label_path, "w") as label_file:
                    img.write_darknet_anns(label_file)
                img_path = img.get_img_path(base_path, dataset_name, data_split)
                assert img_path.exists(), f"Image doesn't exist {img_path}"
                img_paths.add(str(img_path) + "\n")
        return list(img_paths)

    @staticmethod
    def generate_image_list(
        base_path: Path, dataset_name: str, image_paths: List[str], data_split: str
    ) -> None:
        """Generates train.txt, val.txt, etc, txt file with list of image paths."""
        listing_path = base_path / dataset_name / f"{data_split}.txt"
        print("Listing path: ", listing_path)
        with open(listing_path, "w") as listing_file:
            listing_file.writelines(image_paths)

    @staticmethod
    def build_db(coco: COCO) -> Dict[int, Img]:
        """
        Builds a datastructure of images. All annotations are grouped into their
        corresponding images to facilitate generating the Darknet formatted metadata.

        Args:
            coco: a pycocotools.coco COCO instance

        Returns: Dictionary whose keys are image id's, and values are Img instances that
            are loaded with all the image info and annotations from the coco-formatted
            json
        """
        anns = coco.dataset["annotations"]
        images: Dict[int, Img] = {}
        # Build images data structure:
        for i, ann in enumerate(anns):
            ann = CocoToDarknet.get_ann(ann)
            if ann.img_id not in images:
                coco_img = coco.dataset["images"][ann.img_id]
                images[ann.img_id] = Img(
                    ann.img_id,
                    coco_img["file_name"],
                    float(coco_img["width"]),
                    float(coco_img["height"]),
                )
            img = images[ann.img_id]
            img.add_ann(ann)
        return images

    @staticmethod
    def get_ann(ann):
        """
        Gets a bbox instance from an annotation element pulled from the coco-formatted
        json
        """
        box = ann["bbox"]
        return bbox(ann["image_id"], ann["category_id"], box[0], box[1], box[2], box[3])
