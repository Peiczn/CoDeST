from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PCD_OBS_COLUMNS = [
    "HALLMARK_APOPTOSIS_aucell",
    "HALLMARK_APOPTOSIS_ssgsea",
    "REACTOME_PYROPTOSIS_aucell",
    "REACTOME_PYROPTOSIS_ssgsea",
    "GOBP_NECROPTOTIC_SIGNALING_PATHWAY_aucell",
    "GOBP_NECROPTOTIC_SIGNALING_PATHWAY_ssgsea",
    "FERROPTOSIS_CORE_aucell",
    "FERROPTOSIS_CORE_ssgsea",
]

DEATH_PROGRAM_ORDER = ["Apoptosis", "Pyroptosis", "Necroptosis", "Ferroptosis"]


@dataclass
class STGraphSample:
    slice_id: str
    x: np.ndarray
    counts: sp.csr_matrix
    spatial: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    pcd_weak: np.ndarray
    marker_activity: np.ndarray | None = None


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def assert_no_duplicates(values: list[str], label: str) -> None:
    idx = pd.Index(values)
    if idx.has_duplicates:
        raise ValueError(f"{label} contains duplicates: {idx[idx.duplicated()].unique().tolist()[:10]}")


def assert_gene_alignment(var_names: list[str], genes_ref: list[str]) -> None:
    if len(var_names) != len(genes_ref):
        raise ValueError(f"Gene length mismatch: {len(var_names)} vs {len(genes_ref)}")
    for idx, (lhs, rhs) in enumerate(zip(var_names, genes_ref)):
        if lhs != rhs:
            raise ValueError(
                f"Gene alignment mismatch at position {idx}: var_names[{idx}]={lhs}, genes_ref[{idx}]={rhs}"
            )


def ensure_finite_non_negative(X: Any, label: str) -> None:
    values = X.data if sp.issparse(X) else np.asarray(X)
    if np.isnan(values).any():
        raise ValueError(f"{label} contains NaN")
    if np.isinf(values).any():
        raise ValueError(f"{label} contains inf")
    if (values < 0).any():
        raise ValueError(f"{label} contains negative values")


def to_csr(X: Any) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr()
    if hasattr(X, "to_memory"):
        mem = X.to_memory()
        return mem.tocsr() if sp.issparse(mem) else sp.csr_matrix(np.asarray(mem))
    return sp.csr_matrix(np.asarray(X))


def sparse_to_float32_dense(X: Any) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def load_pseudo_backed(path: str, genes_ref: list[str], cell_type_order: list[str]) -> ad.AnnData:
    adata = ad.read_h5ad(path, backed="r")
    assert_gene_alignment(adata.var_names.astype(str).tolist(), genes_ref)
    if "y_cellprop" not in adata.obsm:
        raise ValueError(f"{path} missing obsm['y_cellprop']")
    y = np.asarray(adata.obsm["y_cellprop"], dtype=np.float32)
    if y.shape[1] != len(cell_type_order):
        raise ValueError(f"{path} y_cellprop width mismatch: {y.shape[1]} vs {len(cell_type_order)}")
    if np.isnan(y).any() or np.isinf(y).any() or (y < 0).any():
        raise ValueError(f"{path} y_cellprop contains invalid values")
    if not np.allclose(y.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{path} y_cellprop rows do not sum to 1")
    if list(adata.uns["cell_type_order"]) != list(cell_type_order):
        raise ValueError(f"{path} cell_type_order mismatch")
    return adata


def load_celltype_prior(path: str, cell_type_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df["cell_type"].tolist() != list(cell_type_order):
        raise ValueError("celltype_prior.csv cell_type order mismatch")
    return df


def load_st_graph_sample(path: str, genes_ref: list[str]) -> STGraphSample:
    adata = ad.read_h5ad(path)
    assert_gene_alignment(adata.var_names.astype(str).tolist(), genes_ref)
    if "lognorm" not in adata.layers or "counts" not in adata.layers:
        raise ValueError(f"{path} missing required layers")
    if "spatial_connectivities" not in adata.obsp:
        raise ValueError(f"{path} missing obsp['spatial_connectivities']")
    if "spatial" not in adata.obsm:
        raise ValueError(f"{path} missing obsm['spatial']")
    for col in PCD_OBS_COLUMNS:
        if col not in adata.obs:
            raise ValueError(f"{path} missing obs['{col}']")

    lognorm = to_csr(adata.layers["lognorm"])
    counts = to_csr(adata.layers["counts"])
    ensure_finite_non_negative(lognorm, f"{path} lognorm")
    ensure_finite_non_negative(counts, f"{path} counts")
    conn = to_csr(adata.obsp["spatial_connectivities"])
    conn = conn.tocoo()
    edge_index = np.vstack([conn.row, conn.col]).astype(np.int64)
    edge_weight = conn.data.astype(np.float32, copy=False)
    pcd = adata.obs[PCD_OBS_COLUMNS].to_numpy(dtype=np.float32)
    slice_id = Path(path).name.replace("_ready.h5ad", "")
    return STGraphSample(
        slice_id=slice_id,
        x=sparse_to_float32_dense(lognorm),
        counts=counts,
        spatial=np.asarray(adata.obsm["spatial"], dtype=np.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
        pcd_weak=pcd,
    )


def save_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
