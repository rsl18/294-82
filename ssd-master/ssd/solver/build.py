import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, lr=None) -> torch.optim.SGD:
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    # return torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=cfg.SOLVER.MOMENTUM,
    #     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    # )
    # Make optimizer ignore weights that are frozen:
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )


def make_lr_scheduler(cfg, optimizer, milestones=None):
    return WarmupMultiStepLR(
        optimizer=optimizer,
        milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
    )
