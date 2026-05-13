from __future__ import annotations

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
