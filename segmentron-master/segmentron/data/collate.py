from torch.utils.data.dataloader import default_collate


class ImTargetAuxCollate(object):
    """
    Custom collate to check the no. of elements, insist that the last element
    is a dict (for auxiliary information), and pass the auxiliary unchanged.
    """

    def __call__(self, batch):
        batch = batch[0]
        if len(batch) == 3 and isinstance(batch[-1], dict):
            im, target, aux = batch
            im, target = default_collate([im]), default_collate([target])
            return im, target, aux
        raise TypeError("Data should be an (im, target, aux) tuple; "
                        "found: {}".format(type(batch)))
