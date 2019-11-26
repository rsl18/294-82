# Future imports:
from __future__ import absolute_import, division, print_function

# Standard Library imports:
import json
from pathlib import Path
from typing import Dict

# 3rd Party imports:
import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data
from pycocotools.cocoeval import COCOeval


class Xview(data.Dataset):
    num_classes = 35
    default_resolution = [512, 512]
    # mean = np.array([0.40622682, 0.44545975, 0.46866511], dtype=np.float32).reshape(
    #     1, 1, 3
    # )
    # std = np.array([0.28533737, 0.2703044, 0.27486498], dtype=np.float32).reshape(
    #     1, 1, 3
    # )
    mean = np.array([0.15197755, 0.18627881, 0.2248358], dtype=np.float32).reshape(
        1, 1, 3
    )
    std = np.array([0.12559805, 0.13540234, 0.16740804], dtype=np.float32).reshape(
        1, 1, 3
    )

    def __init__(self, opt, split):
        super().__init__()
        self.data_dir = Path(opt.data_dir) / "Xview" / "coco_chipped"
        self.img_dir = self.data_dir / f"{split}"
        if split == "test":
            self.annot_path = (
                self.data_dir / "xview_coco_v2_val_chipped.json"
            )
        else:
            if opt.task == "exdet":
                raise NotImplementedError(
                    "exdet annotations are not (yet) implemented for xview"
                )
                self.annot_path = (
                    self.data_dir / f"instances_extreme_{split}2017.json"
                )
            else:
                self.annot_path = (
                    self.data_dir / f"xview_coco_v2_{split}_chipped.json"
                )
        self.max_objs = 1000
        self.class_name = [
            "__background__",
            "Other Aircraft",
            "Small Plane",
            "Passenger/Cargo Plane",
            "Helicopter",
            "Other Passenger Vehicle",
            "Car",
            "Bus",
            "Other Truck",
            "Truck w/Trailer Bed",
            "Truck w/Liquid Tank",
            "Crane Truck",
            "Passenger Car",
            "Other Railway Vehicle",
            "Locomotive",
            "Motor/Sail/Small Boat",
            "Barge",
            "Other Maritime Vessel",
            "Container Ship",
            "Tower Crane",
            "Container Crane",
            "Straddle Carrier",
            "Dump/Haul Truck",
            "Loader/Dozer/Tractor/Scraper",
            "Excavator",
            "Other Engineering Vehicle",
            "Hut/Tent",
            "Shed",
            "Other Building",
            "Aircraft Hangar",
            "Damaged Building",
            "Facility",
            "Helipad",
            "Storage Tank",
            "Pylon",
            "Tower Structure",
        ]
        self._valid_ids = list(range(1, self.num_classes + 1))
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        # Can't find any part of the code that uses voc_color for anything:
        self.voc_color = [
            (v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
            for v in range(1, self.num_classes + 1)
        ]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print("==> initializing xview {} data.".format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print("Loaded {} {} samples".format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(
            self.convert_eval_format(results), open(f"{save_dir}/results.json", "w")
        )

    def run_eval(self, results, save_dir, logger):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes("{}/results.json".format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        if logger is not None:
            write_coco_results(coco_eval, logger)
        return coco_eval.stats

def write_coco_results(coco_eval, logger):
    eval_stats = coco_eval.stats
    per_class_maps: Dict[str, float] = coco_eval.per_class_maps
    metrics = [
          ("Average Precision  (AP)",   "0.50:0.95",     "   all",   "100")
        , ("Average Precision  (AP)",   "0.50     ",     "   all",   "100")
        , ("Average Precision  (AP)",   "0.75     ",     "   all",   "100")
        , ("Average Precision  (AP)",   "0.50:0.95",     " small",   "100")
        , ("Average Precision  (AP)",   "0.50:0.95",     "medium",   "100")
        , ("Average Precision  (AP)",   "0.50:0.95",     " large",   "100")
        , ("Average Recall     (AR)",   "0.50:0.95",     "   all",   "  1")
        , ("Average Recall     (AR)",   "0.50:0.95",     "   all",   " 10")
        , ("Average Recall     (AR)",   "0.50:0.95",     "   all",   "100")
        , ("Average Recall     (AR)",   "0.50:0.95",     " small",   "100")
        , ("Average Recall     (AR)",   "0.50:0.95",     "medium",   "100")
        , ("Average Recall     (AR)",   "0.50:0.95",     " large",   "100")
        , ("Average Precision  (AP)",   "0.25     ",     "   all",   "100")
    ]
    logger.write("\n")
    for i, metric in enumerate(metrics):
        logger.write(
            f"{metric[0]} @[ IoU={metric[1]} | area={metric[2]} | maxDets={metric[3]} ] = {eval_stats[i]}\n"
        )
    logger.write("\nPer-class mAP's:\n")
    for k in per_class_maps.keys():
        logger.write(f"{k}-> {per_class_maps[k]}\n")
    logger.write("\n")
