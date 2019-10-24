"""
ssd.trainer module, contains training code for SSD.
"""
import collections
import datetime
import logging
import time
from argparse import Namespace
from pathlib import Path, PosixPath
from shutil import copy, copytree, rmtree
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch import device
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.optim import SGD
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import SSDDetector
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = "{}/{}".format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg: CfgNode,
    model: SSDDetector,
    data_loader: DataLoader,
    optimizer: SGD,
    scheduler: MultiStepLR,
    checkpointer,
    device: device,
    arguments,
    args: Namespace,
    output_dir: Path,
    model_manager: Dict[str, Any],
) -> SSDDetector:
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(logdir=output_dir / "logs")
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    logger.info("MAX_ITER: {}".format(max_iter))

    # GB: 2019-09-08:
    # For rescaling tests, do an eval before fine-tuning-training, so we know what
    # the eval results are before any weights are updated:
    # do_evaluation(
    #     cfg,
    #     model,
    #     distributed=args.distributed,
    #     iteration=0,
    # )
    # model.train()  # *IMPORTANT*: change to train mode after eval.

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        # TODO: Print learning rate:
        iteration = iteration + 1
        arguments["iteration"] = iteration
        scheduler.step()

        images = images.to(device)
        targets = targets.to(device)
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss = sum(loss for loss in loss_dict.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join(
                    [
                        "iter: {iter:06d}",
                        "lr: {lr:.5f}",
                        "{meters}",
                        "eta: {eta}",
                        "mem: {mem}M",
                    ]
                ).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar(
                    "losses/total_loss", losses_reduced, global_step=global_step
                )
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar(
                        "losses/{}".format(loss_name),
                        loss_item,
                        global_step=global_step,
                    )
                summary_writer.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], global_step=global_step
                )

        # This project doesn't use epochs, it does something with batch samplers
        # instead, so there is only a concept of "iteration". For now hardcode epoch as
        # zero to put into file name:
        epoch = 0
        save_name = f"ssd{cfg.INPUT.IMAGE_SIZE}-vgg_{cfg.DATASETS.TRAIN[0]}_0_{epoch}_{iteration:06d}"
        model_path = Path(output_dir) / f"{save_name}.pth"

        # Above if block would be replaced by this:
        if iteration % args.save_step == 0:
            checkpointer.save(save_name, **arguments)

        # Do eval when training, to trace the mAP changes and see performance improved
        # whether or nor
        if (
            args.eval_step > 0
            and iteration % args.eval_step == 0
            and not iteration == max_iter
        ):
            eval_results = do_evaluation(
                cfg, model,
                distributed=args.distributed,
                iteration=iteration,
            )
            do_best_model_checkpointing(
                cfg, model_path, eval_results, model_manager, logger
            )
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(
                        eval_result["metrics"],
                        "metrics/" + dataset,
                        summary_writer,
                        iteration,
                    )
            model.train()  # *IMPORTANT*: change to train mode after eval.

        if iteration % args.save_step == 0:
            remove_extra_checkpoints(output_dir, [model_path], logger)

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )
    return model


def do_best_model_checkpointing(
    cfg,
    model_path,
    dataset_metrics,
    model_manager: Dict,
    logger: Path,
    is_final: bool = False,
) -> None:
    """
    h4d custom checkpointing/eval code. Keep one copy of best model and one copy of most
    recent model  on disk. Possible issue with using up too much system RAM.
    """
    # keys of the metrics dict: ["AP", "AP50", "AP75", "APs", "APm", "APl"] These
    # values are for maxDets=100. See summarize() in cocoeval.py for source, and
    # also: h4d/ssd-master/ssd/data/datasets/evaluation/coco/__init__.py
    new_coco_eval = dataset_metrics[0]["metrics"]

    model_manager["new"] = {"checkpoint_file": model_path, "coco_eval": new_coco_eval}
    if "voc" in cfg.DATASETS.TEST[0]:
        metric = "mAP"
    else:
        metric = "AP50"
    logger.info(f"new: {model_manager['new']}")
    logger.info(f"best: {model_manager['best']}")
    inference_dir: Path = (Path(cfg.OUTPUT_DIR) / "inference" / cfg.DATASETS.TEST[0])
    best_inference_dir: Path = (
        Path(cfg.OUTPUT_DIR) / "inference" / f"best_{cfg.DATASETS.TEST[0]}"
    )

    if not model_manager["best"] is None:
        old_coco_eval = model_manager["best"]["coco_eval"]

        if new_coco_eval[metric] > old_coco_eval[metric]:
            logger.info(
                (
                    f"new is better than old \n new stats: {new_coco_eval[metric]}, "
                    f"old stats: {old_coco_eval[metric]}"
                )
            )
            os.remove(model_manager["best"]["checkpoint_file"])
            if os.path.exists(best_inference_dir):
                rmtree(best_inference_dir)
            copytree(inference_dir, best_inference_dir)
            model_manager["best"] = model_manager["new"]
    else:
        model_manager["best"] = model_manager["new"]
        copytree(inference_dir, best_inference_dir)
    # Rename checkpoint for best model (if is_final, copy instead of rename):
    best_checkpoint: Path = model_manager["best"]["checkpoint_file"]
    logger.info(f"best checkpoint: {best_checkpoint.name}")
    if not best_checkpoint.name.startswith("best_"):
        logger.info("Renaming best checkpoint")
        new_best_checkpoint_path = (
            best_checkpoint.parent / f"best_{best_checkpoint.name}"
        )
        if is_final:
            copy(best_checkpoint, new_best_checkpoint_path)
        else:
            os.rename(best_checkpoint, new_best_checkpoint_path)
            # best_checkpoint.rename(new_best_checkpoint_path)
        model_manager["best"]["checkpoint_file"] = new_best_checkpoint_path


def remove_extra_checkpoints(output_dir: Path, keep: List[Path], logger) -> None:
    """
    Removes checkpoints that aren't in the keep list.
    """
    keep = [path for path in keep if path is not None]
    logger.info(f"Keep: {keep}")
    for checkpoint in output_dir.glob("*.pth"):
        checkpoint: PosixPath = checkpoint
        if checkpoint not in keep and not checkpoint.name.startswith("best_"):
            logger.info(f"Deleting old checkpoint: {checkpoint}")
            if checkpoint.exists():
                try:
                    checkpoint.unlink()
                except:
                    pass
