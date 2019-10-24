from . import register_dataset
from .voc import VOC
from .sbd import SBD


@register_dataset('vocvalid')
class VOCValid(VOC):
    """
    Load the "segvalid" split of the 904 non-intersecting inputs from PASCAL VOC
    val and SBD train. The intersection is taken out of VOC val to keep the
    standard 8,498 inputs in the SBD train set.

    Take
        voc_dir: path to PASCAL VOC year dir
        sbd_dir: path to SBD dir
    """
    def __init__(self, voc_dir=None, sbd_dir=None, **kwargs):
        self.voc = VOC(root_dir=voc_dir, split='val')
        self.sbd = SBD(root_dir=sbd_dir, split='train')
        super().__init__(**kwargs)

    def load_slugs(self):
        # take sbd train out of voc val by set difference
        # and keep list for indexing
        voc_val, sbd_train = set(self.voc.slugs), set(self.sbd.slugs)
        return list(voc_val - sbd_train)
