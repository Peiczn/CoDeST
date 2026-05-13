from __future__ import annotations

from pathlib import Path

import pandas as pd


class CSVLogger:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(row)

    def to_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)
