import os
from pathlib import Path

from h4dlib.config import h4dconfig


class DatasetCatalog:
    DATA_DIR = h4dconfig.DATA_DIR
    DATASETS = {
        "voc_2007_train": {"data_dir": "VOC2007", "split": "train"},
        "voc_2007_val": {"data_dir": "VOC2007", "split": "val"},
        "voc_2007_trainval": {"data_dir": "VOC2007", "split": "trainval"},
        "voc_2007_test": {"data_dir": "VOC2007", "split": "test"},
        "voc_2012_train": {"data_dir": "VOC2012", "split": "train"},
        "voc_2012_val": {"data_dir": "VOC2012", "split": "val"},
        "voc_2012_trainval": {"data_dir": "VOC2012", "split": "trainval"},
        "voc_2012_test": {"data_dir": "VOC2012", "split": "test"},
        "coco_2014_valminusminival": {
            "data_dir": "images/val2014",
            "ann_file": "annotations/instances_valminusminival2014.json",
        },
        "coco_2014_minival": {
            "data_dir": "images/val2014",
            "ann_file": "annotations/instances_minival2014.json",
        },
        "coco_2014_train": {
            "data_dir": "images/train2014",
            "ann_file": "annotations/instances_train2014.json",
        },
        "coco_2014_val": {
            "data_dir": "images/val2014",
            "ann_file": "annotations/instances_val2014.json",
        },
        "xview_coco_v2_train": {
            "data_dir": "train",
            "ann_file": "xview_coco_v2_train_chipped.json",
        },
        "xview_coco_v2_val": {
            "data_dir": "val",
            "ann_file": "xview_coco_v2_val_chipped.json",
        },
        "xview_coco_v2_tiny_train": {
            "data_dir": "train",
            "ann_file": "xview_coco_v2_tiny_train_chipped.json",
        },
        "xview_coco_v2_tiny_val": {
            "data_dir": "val",
            "ann_file": "xview_coco_v2_tiny_val_chipped.json",
        },
        "ucb_coco_v1_train": {
            "data_dir": "images",
            "ann_file": "ucb_coco_v1_train.json",
        },
        "ucb_coco_v1_val": {"data_dir": "images", "ann_file": "ucb_coco_v1_val.json"},
        "coco_tiny_test": {
            "data_dir": "images/train2014",
            "ann_file": "annotations/coco_tiny_test_train.json"
        } 
    }

    @staticmethod
    def get(name):
        print(f"Getting dataset for dsname: {name}")
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR / "voc"
            if "VOC_ROOT" in os.environ:
                voc_root = Path(os.environ["VOC_ROOT"])

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(data_dir=voc_root / attrs["data_dir"], split=attrs["split"])
            return dict(factory="VOCDataset", args=args)
        elif "xview_coco" in name:
            xview_coco_root = DatasetCatalog.DATA_DIR / "Xview" / "coco_chipped"
            if "XVIEW_COCO_ROOT" in os.environ:
                xview_coco_root = Path(os.environ["XVIEW_COCO_ROOT"])

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=xview_coco_root / attrs["data_dir"],
                ann_file=xview_coco_root / attrs["ann_file"],
            )
            return dict(factory="XVIEWCOCODataset", args=args)
        elif "ucb_coco" in name:
            ucb_coco_root = DatasetCatalog.DATA_DIR / "enclave" / "coco"
            if "UCB_COCO_ROOT" in os.environ:
                ucb_coco_root = Path(os.environ["UCB_COCO_ROOT"])

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=ucb_coco_root / attrs["data_dir"],
                ann_file=ucb_coco_root / attrs["ann_file"],
            )
            return dict(factory="UCBCOCODataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR / "coco"
            if "COCO_ROOT" in os.environ:
                coco_root = Path(os.environ["COCO_ROOT"])

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=coco_root / attrs["data_dir"],
                ann_file=coco_root / attrs["ann_file"],
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
