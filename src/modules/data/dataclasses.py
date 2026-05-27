import torch

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NMRSample:
    peaks: torch.Tensor      # (N, 3) — δC, δH, intensity
    token_ids: torch.Tensor  # (seq_len,)


@dataclass
class NMRBatch:
    peaks: torch.Tensor              # (B, N, 3)
    token_ids: torch.Tensor          # (B, seq_len)
    peak_padding_mask: torch.Tensor  # (B, N) bool — True = padding, ignore
    token_padding_mask: torch.Tensor # (B, seq_len) bool — True = padding, ignore
    labels: Optional[torch.Tensor] = field(default=None)  # (B, seq_len) — only used for training loss
