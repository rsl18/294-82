import torch
import torch.nn as nn

from . import register_arch
from ..core import Upsampler


@register_arch('skip')
class SkipArch(nn.Module):
    """
    Skip architecture with a backbone for feature encoding, skips across the
    feature hierarchy, and a head + decoder for each skip. The model output is
    merged across skips by summing and interpolating the skip outs in
    deep-to-shallow order.
    """

    def __init__(self, backbone, head, out_dim):
        super().__init__()
        # feature extraction
        self.encoder = backbone()
        if not self.encoder.aligned:
            raise ValueError("Skip architecture is incompatible "
                            f"with unaligned backbone ({backbone}).")
        if isinstance(self.encoder.dim, int):
            raise ValueError("Skip architecture requires backbone with "
                            f"multiple outputs but the backbone only has one.")
        # task inference
        self.heads = []
        for o, out in enumerate(self.encoder.outputs):
            skip_head = head(self.encoder.dim[o], out_dim)
            self.add_module(f'head{out}', skip_head)
            self.heads.append(skip_head)
        # resolution/stride ratios between skips
        strides = self.encoder.stride
        ratios = [deep // shallow
                  for deep, shallow in zip(strides[1:], strides)]
        # skip decoders interpolate in a cascade
        self.decoders = []
        self.offsets = []
        for out, ratio in zip(self.encoder.outputs, ratios):
            dec = Upsampler(ratio, odd=self.encoder.aligned)
            self.add_module(f'decoder{out}', dec)
            self.decoders.append(dec)
            self.offsets.append(ratio - 1)
        # interpolate from merged skips to full output
        output_stride = strides[0]
        self.full_decoder = Upsampler(output_stride, odd=self.encoder.aligned)
        self.full_offset = output_stride - 1

    def forward(self, x):
        feats = self.encoder(x)
        skips = [head(feat) for head, feat in zip(self.heads, feats)]
        # skip decoding: merge by interp., crop, and sum from deep to shallow
        skips, last = skips[:-1], skips[-1]
        for skip, dec, off in zip(
            skips[::-1], self.decoders[::-1], self.offsets[::-1]
        ):
            h, w = skip.size()[2:]
            upcropped = self.crop(dec(last), h, w, offset=off)
            merged = skip + upcropped
            last = merged
        # interpolate + crop full output from merged skips
        h, w = x.size()[2:]
        out = self.crop(self.full_decoder(merged),
                        h, w, offset=self.full_offset)
        return out

    def crop(self, x, h, w, offset=0):
        return x[..., offset:offset + h, offset:offset + w]
