from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(fn(x, y)[0])


def compute_deconv_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    cell_type_order: list[str],
    rare_mask: np.ndarray,
) -> list[dict]:
    eps = 1e-8
    rows: list[dict] = []
    rows.append({"metric": "mae_overall", "cell_type": "ALL", "value": float(np.mean(np.abs(pred - target)))})
    rows.append(
        {
            "metric": "kl_overall",
            "cell_type": "ALL",
            "value": float(np.mean(np.sum(target * (np.log(target + eps) - np.log(pred + eps)), axis=1))),
        }
    )
    dom_acc = float(np.mean(np.argmax(pred, axis=1) == np.argmax(target, axis=1)))
    rows.append({"metric": "dominant_cell_type_accuracy", "cell_type": "ALL", "value": dom_acc})
    rows.append(
        {"metric": "rare_cell_mae", "cell_type": "ALL", "value": float(np.mean(np.abs(pred[:, rare_mask] - target[:, rare_mask])))}
    )
    for idx, cell_type in enumerate(cell_type_order):
        rows.extend(
            [
                {"metric": "mae_per_cell_type", "cell_type": cell_type, "value": float(np.mean(np.abs(pred[:, idx] - target[:, idx])))},
                {"metric": "pearson_per_cell_type", "cell_type": cell_type, "value": safe_corr(pearsonr, pred[:, idx], target[:, idx])},
                {"metric": "spearman_per_cell_type", "cell_type": cell_type, "value": safe_corr(spearmanr, pred[:, idx], target[:, idx])},
            ]
        )
    return rows
