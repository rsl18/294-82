import os
import importlib


BACKBONES = {}


def register_backbone(name):
    """Registers a backbone by name for instantiation in segmentron."""

    def register_backbone_fn(fn):
        if name in BACKBONES:
            raise ValueError(f"Cannot register duplicate backbone {name}")
        if not callable(fn):
            raise TypeError(f"Backbone {name} must be callable")
        BACKBONES[name] = fn
        return fn

    return register_backbone_fn


# automatically import any backbones in the backbones/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('segmentron.model.backbones.' + module)
