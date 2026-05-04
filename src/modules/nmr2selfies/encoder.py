
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional
from __future__ import annotations

from ..config import EncoderSettings


@dataclass
class ModelConfig:
    peak_feature_dim: int = 3
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    vocab_size: int = 200
    max_token_len: int = 128
    peak_mean: tuple = (5.0, 100.0, 0.5)
    peak_std: tuple = (2.0, 50.0, 0.3)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderSettings):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model)
        )

        # i am already normalizing the input peaks
        # self.norm1 = nn.LayerNorm(cfg.d_model)
        # self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, peak_mask=None) -> torch.Tensor:
        # (B, N, D)
        B, N, _ = x.shape
        m = peak_mask.unsqueeze(1).expand(B, N, N)
        return x
