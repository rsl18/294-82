# Future imports:
from __future__ import absolute_import, division, print_function

# Standard Library imports:
import json
import os
from pathlib import Path
import time

# 3rd Party imports:
import cv2
import numpy as np
from progress.bar import Bar
import torch

# 1st Party imports:
import _init_paths
from detectors.detector_factory import detector_factory
from external.nms import soft_nms
from lib.datasets.dataset_factory import dataset_factory
from logger import Logger
from models.model import create_model, load_model
from opts import opts
from utils.utils import AverageMeter

# h4dlib imports:
import _import_helper
from h4dlib.config import h4dconfig

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            print("TEST SCALE: ", scale)
            if opt.task == "ddd":
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info["calib"]
                )
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {"images": images, "image": image, "meta": meta}

    def __len__(self):
        return len(self.images)


def get_model(opt, model_path):
    if opt.gpus[0] >= 0:
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    print("Creating model...")
    # model_path = opt.load_model
    # model = create_model(opt.arch, opt.heads, opt.head_conv)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('Loaded {}, #epochs: {}'.format(model_path, checkpoint['epoch']))
    # model = load_model(model, opt.load_model)


def get_model_paths():
    exp_path :Path = h4dconfig.ROOT_DIR / "centernet-master/exp"
    print("exp_path: ", exp_path)
    models = []
    # Scan for best and/or most recent model:
    models.extend(exp_path.glob("**/*_last.pth"))
    # models.extend(exp_path.glob("**/*_best.pth"))
    print("Models found: ")
    for m in models:
        print(m)
    return models

def main(opt):
    """
    Inspect models, e.g., loads all saved models from exp folder and prints out details
    like how many epochs they were trained for.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    # print(opt)
    logger = Logger(opt)
    Detector = detector_factory[opt.task]

    split = "val" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    # detector = Detector(opt)
    model_paths = get_model_paths()
    for model_path in model_paths:
        get_model(opt, model_path)


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
