import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .blur import blur2d_sphere, blur2d_diag, blur2d_full
from .deform import deform_conv, UNIT_CIRCLE


class FoG2d(nn.Module):
    """Composition of free-form and gaussian filtering by convolution.
    Blurs the effective filter according to the covariance."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 sigma=1.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.weight = nn.Parameter(torch.Tensor(out_channels,
                                                in_channels // self.groups,
                                                *self.kernel_size))
        nn.init.xavier_uniform_(self.weight)  # default for nn.Conv2d

    def forward(self, x):
        x = blur2d_sphere(x, self.sigma)
        x = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
        return x


class FoGSync2d(FoG2d):
    """Composition of free-form and gaussian filtering by convolution and
    dilation. Transforms the effective filter without blurring by synchronizing
    the smoothing and dilation rates."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 sigma=1.,
                 rate=1.):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, sigma)
        self.rate = torch.tensor(rate)
        # standard Gaussian coordinates to warp by covariance
        self.standard_offset = UNIT_CIRCLE.clone().float()
        # shape for scale broadcasting and deform conv argument
        self.standard_offset = self.standard_offset.view(1, -1, 1, 1)

    def forward(self, x):
        x = blur2d_sphere(x, self.sigma)
        b, _, h, w = x.size()
        # dilate by scaling coordinates then tiling across batch + spatial dims
        dilation = self.sigma * self.rate
        offset = self.standard_offset.to(x.device) * dilation
        offset = offset.repeat(b, 1, h, w)
        return deform_conv(x, offset, self.weight, self.stride,
                           self.padding, self.dilation, self.groups, 1)
