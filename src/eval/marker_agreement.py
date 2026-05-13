from __future__ import annotations

import numpy as np


def per_celltype_marker_agreement(pred: np.ndarray, marker_activity: np.ndarray, cell_type_order: list[str]) -> list[dict]:
    rows: list[dict] = []
    for idx, cell_type in enumerate(cell_type_order):
        x = pred[:, idx]
        y = marker_activity[:, idx]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        rows.append({"metric": "marker_agreement", "cell_type": cell_type, "value": corr})
    return rows
