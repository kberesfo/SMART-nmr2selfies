import torch
import logging
import numpy as np

log = logging.getLogger(__name__)

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

    return torch.from_numpy(arr)


def _drop_sentinel_peaks(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    This drops the peaks that are out of expected bounds and masks the values

    Args:
        arr (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, int]: _description_
    """
    valid = (np.abs(arr[:, 0]) <= DC_MAX) & (np.abs(arr[:, 1]) <= DH_MAX)
    return arr[valid], int((~valid).sum())
