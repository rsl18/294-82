import torch
import torch.nn as nn

from def_conv import DeformConvPack
from sigma.deform import GaussSphereDeformConvPack

from . import register_head


@register_head('deform-free-3')
def HeadDeformFree(in_dim, out_dim):
    return DeformConvPack(
        in_dim, out_dim,
        kernel_size=3, padding=1)


@register_head('deform-sphere-3')
def HeadDeformSphere(in_dim, out_dim):
    return GaussSphereDeformConvPack(
        in_dim, out_dim,
        kernel_size=3, padding=1)
