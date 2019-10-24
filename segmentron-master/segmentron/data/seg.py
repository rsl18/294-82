from abc import abstractmethod
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class SegData(Dataset):
    """
    Skeleton for loading datasets with input and target image pairs.

    Take
        root_dir: path to dataset root dir
        split: specialized to the dataset, but usually train/val/test
        joint_transform: list of `Transform` to apply identically and jointly
                         to the image and target, such as horizontal flipping
        image_transform: list of `Transform`s for the input image
        target_transform: list of `Transform`s for the target image

    Note that joint transforms are done first so that tensor conversion can
    follow transformations done more simply on images/arrays, such as resizing.
    """

    # list of class names ordered by index
    classes = []

    # pixel statistics
    mean = []
    std = []

    # reserved target value to exclude from loss, evaluation, ...
    ignore_index = None

    def __init__(self, root_dir=None, split=None,
                 joint_transform=None, image_transform=None,
                 target_transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.slugs = self.load_slugs()

    @abstractmethod
    def load_slugs(self):
        pass

    @abstractmethod
    def slug_to_image_path(self, slug):
        pass

    def load_image(self, path):
        return Image.open(path)

    @abstractmethod
    def slug_to_annotation_path(self, slug):
        pass

    def load_annotation(self, path):
        return Image.open(path)

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        slug = self.slugs[idx]
        im = self.load_image(self.slug_to_image_path(slug))
        target = self.load_annotation(self.slug_to_annotation_path(slug))
        aux = {'slug': slug}
        if self.joint_transform is not None:
            im, target = self.joint_transform((im, target))
        if self.image_transform is not None:
            im = self.image_transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return im, target, aux

    def __len__(self):
        return len(self.slugs)
