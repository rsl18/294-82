import os
import importlib


DATASETS = {}


def register_dataset(name):
    """Registers a dataset by name for instantiation in segmentron."""

    def register_dataset_cls(cls):
        if name in DATASETS:
            raise ValueError(f"Cannot register duplicate dataset {name}")
        DATASETS[name] = cls
        return cls

    return register_dataset_cls


# automatically import any datasets in the dataset/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('segmentron.data.datasets.' + module)
