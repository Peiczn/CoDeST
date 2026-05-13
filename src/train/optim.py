from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
