import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from ..config import Config
from .dataclasses import NMRSample
from .preprocessing import prepare_peaks


class NMRDataset(Dataset):
    # Peaks outside these ranges are sentinel/error values and should be dropped
    _DC_MAX = 300.0   # δC: typical HSQC range 0–220 ppm
    _DH_MAX = 20.0    # δH: typical HSQC range 0–12 ppm

    def __init__(self, cfg: Config, split: str, metadata: dict) -> None:
        super().__init__()

        # load the parquet file for this split into memory
        path = Path(cfg.app.data_path) / "arrow" / split / "HSQC_NMR.parquet"
        table = pq.read_table(path)
        rows = table.to_pydict()

        # store as parallel lists — random access by row index in __getitem__
        self._indices: list[int] = rows["idx"]
        self._data: list[list[float]] = rows["data"]
        self._shapes: list[list[int]] = rows["shape"]

        # metadata dict keyed by str(idx), pre-loaded by DataModule to avoid
        # reading the JSON file once per split
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, row_idx: int) -> NMRSample:
        peaks = self._load_peaks(row_idx)
        token_ids = self._load_selfies(row_idx)
        return NMRSample(peaks=peaks, token_ids=token_ids)

    def _load_peaks(self, row_idx: int) -> torch.Tensor:
        return prepare_peaks(
            self._data[row_idx],
            self._shapes[row_idx]
        )

    def _load_selfies(self):
        pass