from __future__ import annotations

import torch


def edge_preserving_spatial_loss(
    pred: torch.Tensor,
    x_repr: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    temperature: float = 5.0,
) -> torch.Tensor:
    src, dst = edge_index
    pred_diff = torch.mean((pred[src] - pred[dst]) ** 2, dim=1)
    repr_diff = torch.mean((x_repr[src] - x_repr[dst]) ** 2, dim=1)
    adaptive = torch.exp(-temperature * repr_diff).detach()
    return torch.mean(pred_diff * adaptive * edge_weight)
