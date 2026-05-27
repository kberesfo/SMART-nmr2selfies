import math
import torch
import torch.nn as nn

from .artifacts import TokenizerConfig

from ..config import DecoderSettings


class TokenEmbedding(nn.Module):
    def __init__(self, model_cfg: DecoderSettings, tokenizer_cfg: TokenizerConfig):
        super().__init__()
        # look up table for token embeddings; padding_idx ensures the pad token gets zeroed out
        self.tok = nn.Embedding(
            tokenizer_cfg.vocab_size,
            model_cfg.d_model,
            padding_idx=tokenizer_cfg.pad_token_id
        )

        self.d_model = model_cfg.d_model
        # positional encoding buffer
        pe = self._sinusoidal(
            tokenizer_cfg.max_seq_length,
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
        # (1, T, D) — batch dim added for broadcasting
        return pe.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.size(1)
        return self.tok(tokens) * math.sqrt(self.d_model) + self.pe[:, :T]
