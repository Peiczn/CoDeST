from __future__ import annotations

import numpy as np


def morans_i(values: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray) -> float:
    n = values.shape[0]
    x = values.astype(np.float64)
    x = x - x.mean()
    src, dst = edge_index
    w = edge_weight.astype(np.float64)
    num = np.sum(w * x[src] * x[dst])
    den = np.sum(x ** 2)
    if den <= 0 or w.sum() <= 0:
        return 0.0
    return float((n / w.sum()) * (num / den))
