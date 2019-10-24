import torch

from .archs import ARCHS
from .backbones import BACKBONES
from .heads import HEADS


def prepare_model(arch, backbone, head, out_dim, weights=None):
    model = ARCHS[arch](BACKBONES[backbone], HEADS[head], out_dim)
    if weights:
        model.load_state_dict(torch.load(weights))
    return model
