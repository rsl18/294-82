import torch.nn as nn


def constant_init(module, v, b=0.):
    nn.init.constant_(module.weight, v)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, b)


def normal_init(module, mean=0., std=1., b=0.):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, b)


def msra_init(module, b=0.):
    """
    The equal variance init in the normal/fan out/ReLU edition from

        Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification. He. et al, ICCV 2015.
    """
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, b)
