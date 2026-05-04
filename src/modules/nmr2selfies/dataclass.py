from dataclasses import dataclass
import torch

@dataclass
class NMRSample:
    peaks: torch.Tensor      # (N, 3) — δC, δH, intensity
    token_ids: torch.Tensor  # (seq_len,)
