from __future__ import annotations

import numpy as np
import torch


def numpy_edge_to_torch(edge_index: np.ndarray, edge_weight: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(edge_index, dtype=torch.long, device=device),
        torch.as_tensor(edge_weight, dtype=torch.float32, device=device),
    )
