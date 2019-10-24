import torch.nn as nn
from torchvision import models

from ..params import msra_init, constant_init
from . import register_backbone


def conv(in_dim, out_dim, kernel_size=3, stride=1, dilation=1):
    # convolution with half padding for centering and no bias b.c. batch norm
    eff_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = eff_kernel_size // 2
    return nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=False)


def make_stage(block, in_dim, dim, num_blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or in_dim != dim * block.expansion:
        downsample = nn.Sequential(
            conv(in_dim, dim * block.expansion, 1, stride=stride),
            nn.BatchNorm2d(dim * block.expansion),
        )

    layers = []
    # TODO(shelhamer) fix this gross hack!
    layers.append(block(in_dim, dim, stride, dilation // 2 or 1, downsample))
    in_dim = dim * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(in_dim, dim, 1, dilation))
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_dim, dim, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(in_dim, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        # pytorch-style: stride is on the 3x3, not the 1x1
        self.conv2 = conv(dim, dim, 3, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = conv(dim, dim * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(dim * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Define ResNet for image-to-image learning, following torchvision edition,
    with image-to-image learning surgery:
    - omit average pooling and last classifier layer
    - freeze batch norm statistics and scale/shift parameters, but keep the
      batch norm ops, to preserve the conditioning for optimization.
    - include incomplete pooling regions, for fuller resolution, as in Caffe
    - load pre-training from torchvision or FAIR/MSRA
    """
    aligned = True  # aligned by odd kernels and half padding throughout

    archs = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth,
                 outputs=(4,),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 frozen_to=-1,
                 pretrained='torch'):
        super().__init__()
        self.depth = depth
        self.outputs = outputs
        block_type, blocks_per_stage = self.archs[depth]
        self.dims = []  # stage output channels
        self.strides = []  # stage output stride w.r.t. input
        # stem
        stem_dim = 64
        stem_stride = 4
        self.conv1 = conv(3, stem_dim, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(stem_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                    ceil_mode=True)
        stem = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        # stages
        self.stages = []
        self.stages.append(stem)
        self.dims.append(stem_dim)
        self.strides.append(stem_stride)
        in_dim = stem_dim
        stride = stem_stride
        for i, num_blocks in enumerate(blocks_per_stage):
            dim = stem_dim * 2**i
            stage = make_stage(block_type, in_dim, dim, num_blocks,
                               stride=strides[i], dilation=dilations[i])
            stage_name = f'layer{i+1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage)
            in_dim = dim * block_type.expansion
            self.dims.append(in_dim)
            stride *= strides[i]
            self.strides.append(stride)
        self.param_init(pretrained)
        if pretrained:
            # freeze scale/shift batch norm params
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False
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
        # freeze batch norm stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        # freeze stages during training
        if mode and self.frozen_to >= 0:
            for m in self.stages[:self.frozen_to + 1]:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def param_init(self, pretrained=None):
        if pretrained == 'torch':
            from torchvision import models
            pretrained_model = getattr(models, f'resnet{self.depth}')
            pretrained_state = pretrained_model(pretrained=True).state_dict()
            # drop fc
            fc = [k for k in pretrained_state if 'fc' in k]
            for k in fc:
                del pretrained_state[k]
            self.load_state_dict(pretrained_state)
        elif pretrained == 'fair':
            from segmentron.model.serialization import load_fair_params
            pretrained_state = load_fair_params(self.depth)
            # recklessly load w/o strict b.c. unused biases and batch norm stats
            # are harmlessly missing
            self.load_state_dict(pretrained_state, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    msra_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1.)

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


@register_backbone('res50')
def Res50():
    """
    ResNet-50 backbone with ILSVRC pre-training from torchvision.
    """
    return ResNet(50, pretrained='torch')


@register_backbone('res50-fair')
def Res50FAIR():
    """
    ResNet-50 backbone with ILSVRC pre-training from FAIR/MSRA in Caffe2/Caffe.
    """
    return ResNet(50, pretrained='fair')


@register_backbone('res50-frozen')
def Res50Frozen():
    """
    ResNet-50 backbone with ILSVRC pre-training from torchvision.
    """
    return ResNet(50, pretrained='torch', frozen_to=4)


@register_backbone('res50-stage234')
def Res50Stage234():
    """
    ResNet-50 backbone with ILSVRC pre-training from torchvision, and
    outputs from stages 2, 3, and 4 (last stage).
    """
    return ResNet(50, pretrained='torch', outputs=(2, 3, 4))


@register_backbone('res50-dil8')
def Res50Dil8():
    """
    ResNet-50 backbone with ILSVRC pre-training from torchvision,
    dilated to have stride 8 at the output.
    """
    return ResNet(50,
                  pretrained='torch',
                  strides=(1, 2, 1, 1),
                  dilations=(1, 1, 2, 4),)


@register_backbone('res101')
def Res101():
    """
    ResNet-101 backbone with ILSVRC pre-training from torchvision.
    """
    return ResNet(101, pretrained='torch')


@register_backbone('res101-fair')
def Res101FAIR():
    """
    ResNet-101 backbone with ILSVRC pre-training from FAIR/MSRA in Caffe2/Caffe.
    """
    return ResNet(101, pretrained='fair')


@register_backbone('res101-dil8')
def Res101Dil8(BaseFCN):
    """
    ResNet-101 backbone with ILSVRC pre-training from torchvision,
    dilated to have stride 8 at the output.
    """
    return ResNet(101,
                  pretrained='torch',
                  strides=(1, 2, 1, 1),
                  dilations=(1, 1, 2, 4),)


@register_backbone('res152')
def Res152():
    """
    ResNet-152 backbone with ILSVRC pre-training from torchvision.
    """
    return ResNet(152, pretrained='torch')
