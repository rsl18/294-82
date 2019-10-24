import numpy as np
from torch.utils.data.sampler import Sampler


class ReplacementSampler(Sampler):
    """
    Sample elements randomly with replacement.
    `WeightedRandomSampler` can do this, but that usage is less clear.

    Take
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Sample a dataset of equal size from all indices with replacement.
        indices = np.random.choice(len(self.data_source),
                                   size=len(self.data_source),
                                   replace=True)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
