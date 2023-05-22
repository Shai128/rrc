import torch
from bisect import bisect_right


def make_lr_scheduler_from_cfg(cfg, optimizer):
    return make_lr_scheduler(optimizer,
                             cfg.TRAIN.LR_SCHEDULER_MULTISTEPS,
                             cfg.TRAIN.LR_SCHEDULER_GAMMA,
                             warmup_factor=cfg.TRAIN.WARMUP_FACTOR,
                             warmup_iters=cfg.TRAIN.WARMUP_ITERS,
                             warmup_method=cfg.TRAIN.WARMUP_METHOD, )


def make_lr_scheduler(optimizer,
                      milestones=None,
                      gamma=0.1,
                      warmup_factor=1.0 / 3,
                      warmup_iters=500,
                      warmup_method="linear",
                      last_epoch=-1):
    if milestones is None:
        milestones = [30000, 120000, 200000]
    return WarmupMultiStepLR(
        optimizer,
        milestones,
        gamma,
        warmup_factor,
        warmup_iters,
        warmup_method,
        last_epoch,
    )


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
