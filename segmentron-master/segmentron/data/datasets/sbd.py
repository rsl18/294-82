import numpy as np
import scipy.io
from PIL import Image

from . import register_dataset
from .voc import VOC


@register_dataset('sbd')
class SBD(VOC):
    """
    Load semantic segmentation data in the style of PASCAL VOC extended by the
    Semantic Boundaries Dataset. It's essentially the same, but more, and in
    .mat format.

    Take
        root_dir: path to SBD dir (with img and cls dirs and split text files)
        split: {train,val} n.b. these are *not* the same as VOC train/val
    """
    def __init__(self, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', None) or 'data/sbd'
        super().__init__(**kwargs)

    def listing_path(self):
        return (self.root_dir / '{}.txt'.format(self.split))

    def slug_to_image_path(self, slug):
        return str(self.root_dir / 'img' / '{}.jpg'.format(slug))

    def slug_to_annotation_path(self, slug):
        anno_format = 'mat' if self.split != 'trainaug' else 'png'
        return str(self.root_dir / 'cls' / '{}.{}'.format(slug, anno_format))

    def load_annotation(self, path):
        mat = scipy.io.loadmat(path)['GTcls'][0]['Segmentation'][0]
        return Image.fromarray(mat.astype(np.uint8))
