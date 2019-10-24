import torch.nn as nn

from ..params import msra_init, normal_init
from . import register_backbone


def conv3x3(in_dim, out_dim, dilation=1):
    # 3x3 convolution with padding 1 for centering
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1,
                     dilation=dilation)


def make_stage(in_dim, dim, num_layers, stride=2, dilation=1):
    # a VGG stage of convs + max pooling
    layers = []
    for _ in range(num_layers):
        layers.append(conv3x3(in_dim, dim, dilation=dilation))
        layers.append(nn.ReLU(inplace=True))
        in_dim = dim
    layers.append(nn.MaxPool2d(kernel_size=2, stride=stride, ceil_mode=True))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """
    Define VGG-16 as a fully convolutional network, following its original
    definition in Caffe *except* for weight initialization, for which we import
    the MSRA initialization from what was then the future.
    - cast into fully convolutional form by substituting conv for fc
    - omit last classifier layer
    - load pre-training from Caffe or torchvision, or initialize from scratch
    - include incomplete pooling regions, for fuller resolution, as in Caffe
    """
    aligned = False  # not aligned b.c. of even pooling and missing padding
    offset = 0       # padded s.t. there's no excess padding and offset is zero

    def __init__(self,
                 outputs=(5,),
                 strides=(2, 2, 2, 2, 2),  # note: no stride for last stage
                 dilations=(1, 1, 1, 1, 1, 1),
                 frozen_to=-1,
                 pretrained=None):
        super().__init__()
        self.outputs = outputs
        self.dims = []  # stage output channels
        self.strides = []  # stage output stride w.r.t. input
        # convolutional stages
        self.stages = []
        in_dim = 3
        stride = 1
        stage_layers = (2, 2, 3, 3, 3)
        for i, num_layers in enumerate(stage_layers):
            dim = 64 * 2**i if i < 4 else 512
            stage = make_stage(in_dim, dim, num_layers, stride=strides[i],
                               dilation=dilations[i])
            stage_name = f'layer{i+1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage)
            in_dim = dim
            self.dims.append(dim)
            stride *= strides[i]
            self.strides.append(stride)
        # the layers formally known as fully-connected, but now convolutional
        fconv = nn.Sequential(
            nn.Conv2d(512, 4096, 7, dilation=dilations[-1]),  # was fc6
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),  # was fc7
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.add_module('layer6', fconv)
        self.stages.append(fconv)
        self.dims.append(4096)
        self.strides.append(self.strides[-1])
        self.layer1[0].padding = (81, 81)  # pad conv1 to align output to input
        self.param_init(pretrained)
        self.frozen_to = frozen_to

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.outputs:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode):
        super().train(mode)
        # freeze stages during training
        if mode and self.frozen_to >= 0:
            for m in self.stages[:self.frozen_to + 1]:
                for p in m.parameters():
                    p.requires_grad = False

    def param_init(self, pretrained=None):
        if isinstance(pretrained, str):
            if pretrained == 'caffe':
                # original Caffe parameters for improved accuracy
                # see https://github.com/jcjohnson/pytorch-vgg
                from torch.utils import model_zoo
                params = model_zoo.load_url(
                    'https://s3-us-west-2.amazonaws.com/'
                    'jcjohns-models/vgg16-00b39a1b.pth').values()
            elif pretrained == 'torch':
                from torchvision import models
                params = [p.data for p in
                          models.vgg16(pretrained=True).parameters()]
            # recklessly take parameters in order
            for p, pretrained_p in zip(self.parameters(), params):
                    p.data.copy_(pretrained_p.view_as(p))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.out_channels < 4096:  # conv from stages
                        msra_init(m)
                    else:  # conv from fully-connected
                        normal_init(m, std=0.01)

    @property
    def dim(self):
        dims = [self.dims[o] for o in self.outputs]
        if len(dims) == 1:
            dims = dims[0]
        return dims

    @property
    def stride(self):
        strides = [self.strides[o] for o in self.outputs]
        if len(strides) == 1:
            strides = strides[0]
        return strides


@register_backbone('vgg16')
def VGG16Caffe():
    """
    VGG-16 backbone with ILSVRC pre-training from Caffe.

    note: inputs should be in [0, 255] range in BGR channel order.
    """
    return VGG16(pretrained='caffe')


@register_backbone('vgg16-dil8')
def VGG16CaffeDil8():
    """
    VGG-16 backbone with ILSVRC pre-training from Caffe, dilated to stride 8.
    """
    vgg = VGG16(pretrained='caffe',
                strides=(2, 2, 2, 1, 1),
                dilations=(1, 1, 1, 1, 2, 4),)
    # pad backbone s.t. output, once decoded, is aligned with input
    vgg.layer1[0].padding = (125, 125)
    return vgg
