import argparse

import torch

from segmentron.data import DATASETS, PREPROCS
from segmentron.model import ARCHS, BACKBONES, HEADS


def prepare_parser(train=True):
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_data_args(parser, train)
    if train:
        add_optimizer_args(parser)
    add_run_args(parser, train)
    return parser


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument("--arch", choices=ARCHS, required=True,
            help="Model architecture family")
    group.add_argument("--backbone", choices=BACKBONES, required=True,
            help="Model backbone for input -> feature")
    group.add_argument("--head", choices=HEADS, required=True,
            help="Model head for feature -> task output")
    group.add_argument("--weights", type=str,
            help="Path to checkpoint for loading weights")


def add_data_args(parser, train):
    group = parser.add_argument_group('Data')
    if train:
        group.add_argument("--dataset", choices=DATASETS, required=True,
                help="Dataset to train on")
        group.add_argument("--split", type=str,
                help="Dataset split to train on")
    group.add_argument("--eval-dataset", choices=DATASETS, required=not train,
            help="Dataset to evaluate on")
    group.add_argument("--eval-split", type=str,
            help="Dataset split to evaluate on")
    group.add_argument("--preproc", choices=PREPROCS, required=True,
            help="Data preprocessing")


def add_optimizer_args(parser):
    group = parser.add_argument_group('Optimizer')
    group.add_argument("--lr", default=1e-10, type=float,
            help="Learning rate")
    group.add_argument("--max-epoch", default=20, type=int,
            help="Number of epochs")
    group.add_argument("--max-iter", default=int(1e5), type=int,
            help="Number of updates for the optimization")
    group.add_argument("--iter-size", default=1, type=int,
            help="Number of forward-backward passes for one update")


def add_run_args(parser, train):
    group = parser.add_argument_group('Run')
    group.add_argument("--output", type=str,
            help="Path to dir for output")
    group.add_argument("--user", type=str,
            help="Username for folder creation")
    group.add_argument("--timestamp", type=str,
            help="Timestamp for folder creation")
    group.add_argument("--seed", default=1337, type=int,
            help="Seed for random number generation")
    group.add_argument("--gpu", type=int, required=True,
            help="GPU to run on")
    if train:
        group.add_argument("--checkpoint-interval", default=4000, type=int,
                help="Number of updates between checkpoints")
    else:
        group.add_argument('--eval-output', default='eval', type=str,
                help="Identifier for this evaluation run")
