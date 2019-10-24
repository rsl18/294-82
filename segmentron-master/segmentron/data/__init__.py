import torch

from . import transforms
from .collate import ImTargetAuxCollate
from .datasets import DATASETS
from .sampler import ReplacementSampler


PREPROCS = ('torch', 'caffe')


def prepare_data(dataset_name, split, preproc):
    """
    Configure transforms and make dataset.
    """
    dataset = DATASETS[dataset_name]
    if preproc == 'torch':
        image_transform = transforms.Compose([
            transforms.ImToNp(),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset.mean, std=dataset.std),
        ])
    else:
        image_transform = transforms.Compose([
            transforms.ImToNp(),
            transforms.ImToCaffe(mean=dataset.mean),  # coupled to Caffe VGG-16
            transforms.NpToTensor(),
        ])
    target_transform = transforms.Compose([
        transforms.ImToNp(is_target=True),
        transforms.SegToTensor(),
    ])
    return dataset(split=split, image_transform=image_transform,
                   target_transform=target_transform)


def prepare_loader(dataset, evaluation=False):
    """
    Make image-by-image loader for random sampling with replacemnet during
    training (evaluation=False) and deterministic, complete loading during
    evaluation (evaluation=True).
    """
    args = {'batch_size': 1, 'num_workers': 1, 'pin_memory': True,
            'collate_fn': ImTargetAuxCollate()}
    if evaluation:
        args['shuffle'] = False
    else:
        args['sampler'] = ReplacementSampler(dataset)
    return torch.utils.data.DataLoader(dataset, **args)
