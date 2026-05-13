from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    return torch.sum(target * (torch.log(target) - torch.log(pred)), dim=1).mean()


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def weighted_mae_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
    diff = torch.abs(pred - target)
    if class_weight is not None:
        diff = diff * class_weight.unsqueeze(0)
    diff = diff.mean(dim=1) * sample_weight
    return diff.mean()


def prior_penalty(pred: torch.Tensor, class_prior: torch.Tensor) -> torch.Tensor:
    mean_pred = pred.mean(dim=0)
    return F.mse_loss(mean_pred, class_prior)
