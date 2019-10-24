
# Standard Library imports:
from argparse import REMAINDER, ArgumentParser, Namespace
import logging
import os
from pathlib import Path
from typing import Any, Dict
import sys

# 3rd Party imports:
# import torch.distributed as dist
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from yacs.config import CfgNode

# 1st Party imports:
from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train, do_best_model_checkpointing
from ssd.modeling.detector import build_detection_model
from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
from h4dlib.experiments import ExperimentManager


def train(cfg: CfgNode, args: Namespace, output_dir: Path, model_manager: Dict[str, Any], freeze_non_sigma: bool = False):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    resume_from = checkpointer.get_best_from_experiment_dir(cfg)
    extra_checkpoint_data = checkpointer.load(f=resume_from)
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])

    # Weight freezing test:
    # print_model(model)
    # freeze_weights(model)
    print_model(model)

    model = do_train(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments, args, output_dir, model_manager)
    return model


def print_model(model) -> None:
    for child_counter, child in enumerate(model.children()):
        print(" child", type(child), child_counter, "is -")
        print(child)


def freeze_weights(model) -> None:
    freeze: bool = True
    for child_counter, child in enumerate(model.children()):
        print(f"Isinstance: {isinstance(child, SSDBoxHead)}, counter: {child_counter}")
        if isinstance(child, SSDBoxHead):
            print("No longer freezing layers")
            freeze = False
        if freeze:
            print("child ", child_counter, " was frozen, counter: {child_counter}")
            for param in child.parameters():
                param.requires_grad = False
        else:
            print("child ", child_counter, " was not frozen, counter: {child_counter}")
    print("Finished freeze_weights()")

def main():
    parser = ArgumentParser(
        description="Single Shot MultiBox Detector Training With PyTorch"
    )
    parser.add_argument(
        "--config-file",
        default="",
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
    else:
        cfg.MODEL.DEVICE = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eman = ExperimentManager("ssd")
    output_dir = eman.get_output_dir()

    args.config_file = str(Path(__file__).parent / "configs" / args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()

    eman.start({"cfg": cfg, "args": vars(args)})
    # We use our own output dir, set by ExperimentManager:
    # if cfg.OUTPUT_DIR:
    #     mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), output_dir / "logs")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info(f"Output dir: {output_dir}")

    model_manager = {"best": None, "new": None}
    model = train(cfg, args, output_dir, model_manager)

    if not args.skip_test:
        logger.info("Start evaluating...")
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        eval_results = do_evaluation(
            cfg,
            model,
            distributed=args.distributed,
        )
        do_best_model_checkpointing(
            cfg,
            output_dir / "model_final.pth",
            eval_results,
            model_manager,
            logger,
            is_final=True,
        )

    eman.mark_dir_if_complete()

if __name__ == "__main__":
    main()
