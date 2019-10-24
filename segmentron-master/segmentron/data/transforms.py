import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class NpToTensor(object):
    """
    Convert `np.array` to `torch.Tensor`, but don't rescale the values
    unlike `ToTensor()` from `torchvision`.
    """

    def __call__(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NpToIm(object):
    """
    Convert `np.array` to PIL `Image`, using mode appropriate for the
    number of channels.
    """

    def __call__(self, arr):
        if arr.shape[-1] == 1:
            return Image.fromarray(arr, mode='P')
        else:
            return Image.fromarray(arr, mode='RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImToNp(object):
    """
    Convert PIL `Image` to `np.array`.
    """

    def __init__(self, is_target=False):
        self.is_target = is_target

    def __call__(self, im):
        arr = np.array(im, dtype=np.uint8)
        if self.is_target:
            return arr
        if arr.ndim == 2:  # indexed images are missing the channel dimension
            arr = arr[..., None]
        elif arr.shape[-1] == 4:  # alpha images have one channel too many
            arr = arr[..., :-1]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImToCaffe(object):
    """
    Prepare image for input to Caffe-style network:
     - permute RGB channels to BGR
     - subtract mean
     - swap axes to C x H x W order
    """

    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.mean = self.mean[::-1]  # BGR
        self.mean *= 255.

    def __call__(self, im):
        im = im.astype(np.float32)[..., ::-1].transpose((2, 0, 1))
        im -= self.mean
        return im

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SegToTensor(object):
    """
    Convert `np.array` of integer segmentation targets to `torch.Tensor`
    without channel dimension to satisfy loss.
    """

    def __call__(self, seg):
        seg = torch.from_numpy(seg).long()
        seg = seg.squeeze(-1)  # H x W
        return seg

    def __repr__(self):
        return self.__class__.__name__ + '()'
