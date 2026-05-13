from __future__ import annotations

import torch
import torch.nn.functional as F


def marker_consistency_loss(pred: torch.Tensor, marker_activity: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return pred.new_tensor(0.0)
    pred_sel = pred[:, valid_mask]
    marker_sel = marker_activity[:, valid_mask]
    pred_sel = pred_sel - pred_sel.mean(dim=0, keepdim=True)
    marker_sel = marker_sel - marker_sel.mean(dim=0, keepdim=True)
    pred_sel = F.normalize(pred_sel, dim=0)
    marker_sel = F.normalize(marker_sel, dim=0)
    return torch.mean((pred_sel - marker_sel) ** 2)
