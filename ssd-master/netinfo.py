# Standard Library imports:
from argparse import REMAINDER, ArgumentParser, Namespace
import logging
import os
from pathlib import Path
from typing import Any, Dict

from yacs.config import CfgNode
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# from pytorch_modelsummary import ModelSummary

# 1st Party imports:
from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train, do_best_model_checkpointing
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.utils.modelsummary import summary


def get_model_info(args, cfg):
    print(cfg)
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    # Magic number. Just plugged in a value to satisfy the error message:
    if cfg.INPUT.IMAGE_SIZE == 512:
        NUM_PRIORS = 24564
    elif cfg.INPUT.IMAGE_SIZE == 300:
        NUM_PRIORS = 10830
    else:
        raise "No implemented"
    for show_input in [False, True]:
        summary(
            model,
            torch.zeros((1, 3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)).to(device),
            {
                "labels": torch.ones((1, NUM_PRIORS)).type(torch.LongTensor).to(device),
                "boxes": torch.ones((1, NUM_PRIORS, 4)).type(torch.FloatTensor).to(device),
            },
            show_input=show_input
        )
    


def main():
    parser = ArgumentParser(
        description="Single Shot MultiBox Detector Training With PyTorch"
    )
    parser.add_argument(
        "--config_file",
        default="vgg_ssd300_coco_simple_sigma_in_backbone.yaml",
        metavar="FILE",
        help="config file name or path (relative to the configs/ folder) ",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--log_step", default=50, type=int, help="Print logs every log_step"
    )
    parser.add_argument(
        "--save_step", default=5000, type=int, help="Save checkpoint every save_step"
    )
    parser.add_argument(
        "--eval_step",
        default=5000,
        type=int,
        help="Evaluate dataset every eval_step, disabled when eval_step < 0",
    )
    parser.add_argument("--use_tensorboard", default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=REMAINDER,
    )
    parser.add_argument(
        "--resume_experiment",
        default="None",
        dest="resume",
        type=str,
        help="Checkpoint state_dict file to resume training from",
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
        print("Using cuda")
    else:
        cfg.MODEL.DEVICE = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.config_file = str(Path(__file__).parent / "configs" / args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    get_model_info(args, cfg)


if __name__ == "__main__":
    main()
