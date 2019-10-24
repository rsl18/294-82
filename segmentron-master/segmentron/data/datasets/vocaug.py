from . import register_dataset
from .voc import VOC
from .sbd import SBD
from .vocvalid import VOCValid


@register_dataset('vocaug')
class VOCAug(VOC):
    """
    Load the "trainaug" split of union(segtrain, sbdtrain, sbdval) - segvalid
    11,127 inputs to check the improvement from more training data. Where VOC
    and SBD intersect, take the VOC truths since the annotations are more
    reliable. For segvalid, the canonical validation split for VOC/SBD eval,
    refer to the VOCValid Dataset.

    Note (dataset, ID) slugs for delegation of the image/annotation loading.

    Take
        voc_dir: path to PASCAL VOC year dir
        sbd_dir: path to SBD dir
    """
    def __init__(self, voc_dir=None, sbd_dir=None, **kwargs):
        # load segtrain from VOC, train + val from SBD, and segvalid
        self.segtrain = VOC(root_dir=voc_dir, split='train')
        self.sbdtrain = SBD(root_dir=sbd_dir, split='train')
        self.sbdval = SBD(root_dir=sbd_dir, split='val')
        self.vocvalid = VOCValid(voc_dir=voc_dir, sbd_dir=sbd_dir)
        # init w/o root and split since this is a composite dataset
        super().__init__(**kwargs)

    def load_slugs(self):
        # give VOC priority over SBD by taking out duplicates in SBD
        for sbd in (self.sbdtrain, self.sbdval):
            sbd.slugs = list(set(sbd.slugs) - set(self.segtrain.slugs))
        # collect the training sets, take out vocvalid from each, and then
        # unpack into a flat list of (dataset, idx) slugs
        slugs = []
        for dataset in (self.segtrain, self.sbdtrain, self.sbdval):
            dataset.slugs = list(set(dataset.slugs) - set(self.vocvalid.slugs))
            dataset.slugs.sort()  # for determinism (sets don't keep order)
            slugs.extend([(dataset, s) for s in dataset.slugs])
        return slugs

    def slug_to_image_path(self, dataset_slug):
        return dataset_slug

    def load_image(self, dataset_slug):
        dataset, slug = dataset_slug
        return dataset.load_image(dataset.slug_to_image_path(slug))

    def slug_to_annotation_path(self, dataset_slug):
        return dataset_slug

    def load_annotation(self, dataset_slug):
        dataset, slug = dataset_slug
        return dataset.load_annotation(dataset.slug_to_annotation_path(slug))
