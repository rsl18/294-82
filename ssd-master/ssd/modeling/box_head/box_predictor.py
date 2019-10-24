import torch
from torch import nn

from ssd.layers import SeparableConv2d
from ssd.modeling import registry

from ssd.sigma.fog import FoG2d, FoGSync2d
from ssd.sigma.deform import GaussSphereDeformConvPack, GaussFullDeformConvPack



class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register('SSDLiteBoxPredictor')
class SSDLiteBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register("SSDSigmaBoxPredictor")
class SSDSigmaBoxPredictor(BoxPredictor):
    MIN_FEATURE_MAP_SIGMA = 3
    def cls_block(self, level, out_channels, boxes_per_location):
        # Tried this one but it's not working, looks like the code for it is missing
        # a property. Complained about not finding an "anchors_offset" property but I
        # looked in segmentron and that property is referenced but never set
        #
        # Update: After looking at GaussFullDeformConvPack() a little more I think it
        # might be fixable. It might just be a simple typo -- self.anchor_offset maybe
        # should be changed to self.standard_offset
        #
        # return GaussFullDeformConvPack(
        #   out_channels
        #   , boxes_per_location * self.cfg.MODEL.NUM_CLASSES
        #   , kernel_size=3
        #   , stride=1
        #   , padding=1
        # )
        if self.cfg.MODEL.PRIORS.FEATURE_MAPS[level] >= SSDSigmaBoxPredictor.MIN_FEATURE_MAP_SIGMA:
            print(f"level: {level}, out_channels: {out_channels}, feature_maps: {self.cfg.MODEL.PRIORS.FEATURE_MAPS[level]}")
            return GaussSphereDeformConvPack(
                out_channels,
                boxes_per_location * self.cfg.MODEL.NUM_CLASSES,
                kernel_size=3,
                stride=1,
                padding=1,
                # level=level,
            )
        else:
            return nn.Conv2d(
                out_channels,
                boxes_per_location * self.cfg.MODEL.NUM_CLASSES,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def reg_block(self, level, out_channels, boxes_per_location):
        if self.cfg.MODEL.PRIORS.FEATURE_MAPS[level] >= SSDSigmaBoxPredictor.MIN_FEATURE_MAP_SIGMA:
            print(f"level: {level}, out_channels: {out_channels}")
            return GaussSphereDeformConvPack(
                out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1
            )
        else:
            return nn.Conv2d(
                out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1
            )


class GaussWrapper(nn.Module):
    """
    Dummy wrapper that doesn't do anything, but useful for debugging to intercept the
    tensors being passed through SSDSigmaBoxPredictor during forward pass
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.level = kwargs["level"]
        kwargs.pop("level")
        pass_thru_kwargs = {
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "deformable_groups",
            "bias",
        }
        self.inner = GaussSphereDeformConvPack(
            *args, **{k: v for (k, v) in kwargs.items() if k in pass_thru_kwargs}
        )

    def forward(self, input):
        # print(f"Level: {self.level}, size: {x.size()}")
        return self.inner(input)


def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
