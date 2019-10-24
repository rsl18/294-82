from ..seg import SegData
from . import register_dataset


@register_dataset('voc')
class VOC(SegData):
    """
    Load semantic segmentation data in the style of PASCAL VOC.

    Take
        root_dir: path to PASCAL VOC year dir
        split: {train,val,test}
    """

    classes = [
        '__background__',
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # pixel statistics (RGB)
    mean = (0.48109378, 0.45752457, 0.40787054)
    std = (0.27363777, 0.26949592, 0.28480016)

    # reserved target value to exclude from loss, evaluation, ...
    ignore_index = 255

    def __init__(self, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', None) or 'data/voc2012'
        super().__init__(**kwargs)

    def load_slugs(self):
        with open(self.listing_path(), 'r') as f:
            slugs = f.read().splitlines()
        return slugs

    def listing_path(self):
        return (self.root_dir / 'ImageSets' / 'Segmentation'
                / '{}.txt'.format(self.split))

    def slug_to_image_path(self, slug):
        return self.root_dir / 'JPEGImages' / '{}.jpg'.format(slug)

    def slug_to_annotation_path(self, slug):
        return self.root_dir / 'SegmentationClass' / '{}.png'.format(slug)
