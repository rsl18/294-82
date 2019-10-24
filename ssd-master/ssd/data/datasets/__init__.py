from typing import Dict, List

from torch.utils.data import ConcatDataset, Dataset

from ssd.config.path_catlog import DatasetCatalog

from .voc import VOCDataset
from .coco import COCODataset
from .xview_coco import XVIEWCOCODataset
from .ucb_coco import UCBCOCODataset

_DATASETS: Dict[str, Dataset] = {
    "VOCDataset": VOCDataset,
    "COCODataset": COCODataset,
    "XVIEWCOCODataset": XVIEWCOCODataset,
    "UCBCOCODataset": UCBCOCODataset,
}


def build_dataset(
    dataset_list, transform=None, target_transform=None, is_train=True
) -> Dataset:
    """
    returns: a torch.data.dataset.Dataset instance
    """
    assert dataset_list, "dataset_list should not be empty"
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data["args"]
        factory = _DATASETS[data["factory"]]
        args["transform"] = transform
        args["target_transform"] = target_transform
        if factory == VOCDataset:
            args["keep_difficult"] = not is_train
        elif factory == XVIEWCOCODataset:
            args["remove_empty"] = is_train
        elif factory == UCBCOCODataset:
            args["remove_empty"] = is_train
        elif factory == COCODataset:
            args["remove_empty"] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
