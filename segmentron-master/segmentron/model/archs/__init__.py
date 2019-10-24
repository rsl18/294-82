import os
import importlib


ARCHS = {}


def register_arch(name):
    """Registers a arch by name for instantiation in segmentron."""

    def register_arch_cls(cls):
        if name in ARCHS:
            raise ValueError(f"Cannot register duplicate arch {name}")
        ARCHS[name] = cls
        return cls

    return register_arch_cls


# automatically import any archs in the archs/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('segmentron.model.archs.' + module)
