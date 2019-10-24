import torch
import torch.nn as nn

from . import register_arch
from ..core import Upsampler


@register_arch('simple')
class SimpleArch(nn.Module):
    """
    Simple fully convolutional network with a backbone for feature encoding, a
    head for task inference, and a decoder for interpolation.
    """

    def __init__(self, backbone, head, out_dim):
        super().__init__()
        # feature extraction
        self.encoder = backbone()
        # task inference
        self.head = head(self.encoder.dim, out_dim)
        # decode by interpolation to output stride (== downsampling rate)
        # and crop into the projective field center
        self.decoder = Upsampler(self.encoder.stride, odd=self.encoder.aligned)
        if self.encoder.aligned:
            self.offset = self.encoder.stride - 1
        else:
            # hack: workaround unaligned backbones by manually setting offset
            self.offset = self.encoder.offset

    def forward(self, x):
        h, w = x.size()[-2:]
        x = self.encoder(x)
        x = self.head(x)
        x = self.decoder(x)
        x = self.crop(x, h, w)
        return x

    def crop(self, x, h, w):
        return x[..., self.offset:self.offset + h, self.offset:self.offset + w]
