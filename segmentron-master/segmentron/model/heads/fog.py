import torch
import torch.nn as nn

from sigma.fog import FoG2d, FoGSync2d

from . import register_head


@register_head('fog3')
def HeadFoG3x3(in_dim, out_dim):
    return FoG2d(
        in_dim, out_dim,
        kernel_size=3, padding=1,
        sigma=1.)


@register_head('fog3sync1')
def HeadFoGSynck3x3(in_dim, out_dim):
    return FoGSync2d(
        in_dim, out_dim,
        kernel_size=3, padding=1,
        sigma=1.)


@register_head('fog3sync2')
def HeadFoGSync2k3x3(in_dim, out_dim):
    return FoGSync2d(
        in_dim, out_dim,
        kernel_size=3, padding=1,
        sigma=1., rate=2.)


@register_head('fog3sync1.5')
def HeadFoGSync2k3x3(in_dim, out_dim):
    return FoGSync2d(
        in_dim, out_dim,
        kernel_size=3, padding=1,
        sigma=1., rate=1.5)
