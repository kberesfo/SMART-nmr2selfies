import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from .artifacts import TokenizerConfig
from ..config import DecoderSettings


class TokenEmbedding(nn.Module):
    def __init__(self, model_cfg: DecoderSettings, tokenizer_cfg: TokenizerConfig):
        super().__init__()
        self.tok = nn.Embedding(
            tokenizer_cfg.vocab_size,
            model_cfg.d_model,
            padding_idx=tokenizer_cfg.pad_token_id
        )
        self.d_model = model_cfg.d_model
        pe = self._sinusoidal(
            tokenizer_cfg.max_token_len,
            model_cfg.d_model
        )
        self.register_buffer("pe", pe)

    @staticmethod
    def _sinusoidal(T: int, D: int) -> torch.Tensor:
        pos = torch.arange(T).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
        pe = torch.zeros(T, D)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.size(1)
        return self.tok(tokens) * math.sqrt(self.d_model) + self.pe[:, :T]


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderSettings):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.num_heads,
            dropout=cfg.dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model)
        )

    def forward(self, x, mask=None):
        raise NotImplementedError(
            "DecoderLayer forward pass not implemented yet"
        )


class Decoder():
    def __init__(self, cfg: DecoderSettings):
        super().__init__()
        pass

    def decode(self, token_ids):
        raise NotImplementedError(
            "Decoder decode method not implemented yet"
        )
