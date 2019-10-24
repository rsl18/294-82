import os
import importlib


HEADS = {}


def register_head(name):
    """Registers a head by name for instantiation in segmentron."""

    def register_head_cls(cls):
        if name in HEADS:
            raise ValueError(f"Cannot register duplicate head {name}")
        HEADS[name] = cls
        return cls

    return register_head_cls


# automatically import any heads in the heads/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('segmentron.model.heads.' + module)
