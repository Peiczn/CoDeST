from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np


def load_marker_sets(path: str, genes_ref: list[str], cell_type_order: list[str]) -> OrderedDict[str, list[int]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes_ref)}
    out: OrderedDict[str, list[int]] = OrderedDict()
    for cell_type in cell_type_order:
        genes = [g for g in raw.get(cell_type, []) if g in gene_to_idx]
        out[cell_type] = [gene_to_idx[g] for g in genes]
    return out


def compute_marker_activity_from_dense(
    x: np.ndarray,
    marker_sets: OrderedDict[str, list[int]],
) -> np.ndarray:
    activity = np.zeros((x.shape[0], len(marker_sets)), dtype=np.float32)
    for col_idx, indices in enumerate(marker_sets.values()):
        if indices:
            activity[:, col_idx] = x[:, indices].mean(axis=1)
    return activity
