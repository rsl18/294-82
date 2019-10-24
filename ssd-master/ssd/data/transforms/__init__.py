from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.TRAIN.RESCALE_FACTOR > 0:
            print(f"USING TRAIN.RESCALE_FACTOR: {cfg.TRAIN.RESCALE_FACTOR}")
            transform = [
                ConvertFromInts(),
                # PhotometricDistort(),
                # Expand(cfg.INPUT.PIXEL_MEAN),
                # RandomSampleCrop(),
                # RandomMirror(),
                ToPercentCoords(),
                ShrinkAndPad(cfg.INPUT.IMAGE_SIZE, is_train=True, rescale_factor=cfg.TRAIN.RESCALE_FACTOR, mean=cfg.INPUT.PIXEL_MEAN),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]
        else:
            print("NOT using any rescaling for training data.")
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]
    else:
        # For rescaling test:
        if cfg.TEST.RESCALE_FACTOR > 0:
            print(f"USING RESCALE_TEST FACTOR: {cfg.TEST.RESCALE_FACTOR}")
            transform = [
                ShrinkAndPad(cfg.INPUT.IMAGE_SIZE, is_train=False, rescale_factor=cfg.TEST.RESCALE_FACTOR, mean=cfg.INPUT.PIXEL_MEAN),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor()
            ]
        else:
            print("NOT using any rescale factor.")
            # Original (no rescaling):
            transform = [
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor()
            ]

    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
