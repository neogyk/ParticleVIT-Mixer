from __future__ import annotations

import math
import os
import torch
import random
import warnings
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torch_geometric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    # From Baler
    """PyTorch implementation to fix the seeds
    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, torch_geometric.nn.CGConv):
        torch.nn.init.xavier_normal_(m.lin.weight)
        if m.lin.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, torch.nn.RMSNorm):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0.1)


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        warmup_steps=350,
        decay=1,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch, verbose)

        # Decay attributes
        self.decay = decay
        self.initial_lrs = self.base_lrs

        # Warmup attributes
        self.warmup_steps = warmup_steps
        self.current_steps = 0

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [
            (self.current_steps / self.warmup_steps)
            * (
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                / 2
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.T_cur + 1 == self.T_i:
            if self.verbose:
                print(f"multiplying base_lrs by {self.decay:.4f}")
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.current_steps < self.warmup_steps:
                self.current_steps += 1

            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
