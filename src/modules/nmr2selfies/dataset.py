import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from ..config import Config
from .dataclass import NMRSample

log = logging.getLogger(__name__)

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
        # reconstruct (N, 3) array from flat data list and shape
        arr = np.array(self._data[row_idx], dtype=np.float32)
        arr = arr.reshape(self._shapes[row_idx])  # (N, 3)

        # drop peaks with sentinel values outside the valid HSQC window
        # inspect_data.py showed min δC/δH of -100,000 — clearly not real peaks
        valid = (np.abs(arr[:, 0]) <= self._DC_MAX) & (np.abs(arr[:, 1]) <= self._DH_MAX)
        dropped = int((~valid).sum())
        if dropped:
            log.debug(f"idx={self._indices[row_idx]}: dropped {dropped} sentinel peaks")
        arr = arr[valid]

        arr[:, 2] = self._normalize_intensity(arr[:, 2])
        return torch.from_numpy(arr)

    def _normalize_intensity(self, I: np.ndarray) -> np.ndarray:
        # local scaling: p95 of positives -> +1, p05 of negatives -> -1
        # preserves sign information while bringing intensities into [-1, 1]
        I_scaled = np.zeros_like(I)
        eps = 1e-12

        pos = I > 0
        neg = I < 0

        # scale positives so p95 -> +1
        if pos.any():
            p_hi = np.percentile(I[pos], 95)
            if p_hi > eps:
                I_scaled[pos] = I[pos] / p_hi

        # scale negatives so p05 (most negative) -> -1
        if neg.any():
            n_lo = np.percentile(I[neg], 5)
            denom = abs(n_lo)
            if denom > eps:
                I_scaled[neg] = I[neg] / denom

        return np.clip(I_scaled, -1.0, 1.0)

    def _load_selfies(self, row_idx: int) -> torch.Tensor:
        # TODO: implement once tokenizer is built
        # look up SMILES from metadata, convert to SELFIES, tokenize
        raise NotImplementedError
