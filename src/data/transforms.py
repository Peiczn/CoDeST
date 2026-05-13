from __future__ import annotations

import numpy as np
import torch


def make_masked_input(x: torch.Tensor, mask_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(x) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def zscore_numpy(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)
