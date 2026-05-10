import math
import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
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
        pos: torch.Tensor = torch.arange(T).unsqueeze(1)
        div: torch.Tensor = torch.exp(
            torch.arange(0, D, 2) * (-math.log(10000.0) / D))
        pe: torch.Tensor = torch.zeros(T, D)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        return pe.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.size(1)
        return self.tok(tokens) * math.sqrt(self.d_model) + self.pe[:, :T]


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderSettings):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            cfg.d_model, cfg.num_heads, cfg.dropout)
        self.cross_attn = MultiHeadAttention(
            cfg.d_model, cfg.num_heads, cfg.dropout)

        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.last_cross_attn: Optional[torch.Tensor] = None

    def forward(self, tgt, memory, tgt_mask, memory_mask, capture_attn=False):
        a, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.drop(a))
        a, w = self.cross_attn(tgt, memory, memory,
                               mask=memory_mask, need_weights=capture_attn)
        if capture_attn:
            self.last_cross_attn = w.detach()
            
        tgt = self.norm2(tgt + self.drop(a))
        tgt = self.norm3(tgt + self.drop(self.ff(tgt)))

        return tgt


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderSettings, tokenizer_cfg: TokenizerConfig):
        super().__init__()
        self.embed = TokenEmbedding(cfg, tokenizer_cfg)
        self.layers = nn.ModuleList([DecoderLayer(cfg)
                                    for _ in range(cfg.n_decoder_layers)])
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, tokens, memory, tgt_mask, memory_mask, capture_attn=False):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask,
                      capture_attn=capture_attn)
        return self.out_proj(x)

    def cross_attn_stack(self) -> torch.Tensor:
        w = [layer.last_cross_attn for layer in self.layers]
        assert all(
            x is not None for x in w), "run forward(..., capture_attn=True) first"
        return torch.stack(w, dim=0)
