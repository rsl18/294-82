import torch
import torch.nn as nn

from . import register_head


class ConvHead(nn.Module):

    def __init__(self, feature_dim, num_outputs, kernel_size=1, dilation=1):
        super().__init__()
        # determine effective kernel size and pad by half for alignment
        eff_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = eff_kernel_size // 2
        self.net = nn.Conv2d(feature_dim, num_outputs, kernel_size,
                             padding=padding, dilation=dilation)
        # zero init
        for p in self.net.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, x):
        return self.net(x)


@register_head('k1')
class Head1x1(ConvHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, 1)


@register_head('k3')
class Head3x3(ConvHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, 3)


@register_head('k5')
class Head5x5(ConvHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, 5)


@register_head('k3d2')
class Head3x3d2(ConvHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, 3, 2)


@register_head('k5d3')
class Head5x5d3(ConvHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, 5, 3)


@register_head('k5d12')
class Head5x5d12(ConvHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, 5, 12)
