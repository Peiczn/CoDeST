from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = F.huber_loss(pred, target, delta=delta, reduction="none").mean(dim=1)
    return (loss * confidence).mean()
