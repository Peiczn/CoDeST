from __future__ import annotations

from src.data.io_h5ad import read_lines


def load_st_split_paths(st_root: str, slice_ids: list[str]) -> list[str]:
    return [f"{st_root}/{slice_id}/{slice_id}_ready.h5ad" for slice_id in slice_ids]


def read_st_splits(train_file: str, val_file: str | None = None, test_file: str | None = None) -> dict[str, list[str]]:
    splits = {"train": read_lines(train_file)}
    if val_file:
        splits["val"] = read_lines(val_file)
    if test_file:
        splits["test"] = read_lines(test_file)
    return splits
