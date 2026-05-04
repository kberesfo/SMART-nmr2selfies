import numpy as np
import torch

# global caps
DC_MAX = 300.0
DH_MAX = 20.0

def prepare_peaks(raw_data: list[float], shape: list[int]) -> torch.Tensor:
    """
    Full peak preparation pipeline — safe to call from both training
    and production inference.
    
    Returns (N, 3) float tensor of valid, normalized peaks.
    """
    arr = np.array(raw_data, dtype=np.float32).reshape(shape)
    arr, n_dropped = _drop_sentinel_peaks(arr)
    
    if n_dropped:
        log.debug("dropped %d sentinel peaks", n_dropped)
    if len(arr) == 0:
        log.warning("all peaks dropped — input may be malformed")

    arr[:, 2] = _normalize_intensity(arr[:, 2])
    return torch.from_numpy(arr)

def _drop_sentinel_peaks(arr: np.ndarray) -> tuple[np.ndarray, int]:
    valid = (np.abs(arr[:, 0]) <= DC_MAX) & (np.abs(arr[:, 1]) <= DH_MAX)
    return arr[valid], int((~valid).sum())

def _normalize_intensity(I: np.ndarray) -> np.ndarray:
    I_scaled = np.zeros_like(I)
    eps = 1e-12
    pos = I > 0
    neg = I < 0

    if pos.any():
        p_hi = np.percentile(I[pos], 95)
        if p_hi > eps:
            I_scaled[pos] = I[pos] / p_hi

    if neg.any():
        n_lo = np.percentile(I[neg], 5)
        if abs(n_lo) > eps:
            I_scaled[neg] = I[neg] / abs(n_lo)

    return np.clip(I_scaled, -1.0, 1.0)