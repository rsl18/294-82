import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import build_detection_model
from ssd.utils import dist_util
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger


def evaluation(cfg, ckpt, distributed):
    logger: logging.RootLogger = logging.getLogger("SSD.inference")

    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)

    for scale in np.linspace(0.5, 1.0, 5):
        logger.info(f"Running eval with rescale factor: {scale}")
        eval_result = do_evaluation(cfg, model, distributed, rescale=scale)


def main():
    parser = argparse.ArgumentParser(
        description="SSD Evaluation on VOC and COCO dataset."
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default="eval_results",
        type=str,
        help="The directory to store evaluation results.",
    )
    parser.add_argument(
        "--rescale",
        help="""Enable rescaling of inputs across a range of scale factors [0.1, 1.0].
                One eval is performed for each scale factor.""",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.config_file = str(Path(__file__).parent / "configs" / args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    make_dirs(Path(cfg.OUTPUT_DIR))
    print("OUTPUT_DIR: ", cfg.OUTPUT_DIR)
    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt, distributed=distributed)


def make_dirs(path: Path):
    "Create dir (including missing parents) if it doesn't exist."
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
