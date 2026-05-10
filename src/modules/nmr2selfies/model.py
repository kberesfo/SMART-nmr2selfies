import torch
import torch.nn as nn

from .artifacts import PeakNormConfig, TokenizerConfig

from ..config import ModelSettings
from .encoder import Encoder
from .decoder import Decoder

# NOTE: STILL NEEDS REVIEW

class NMR2SELFIES(nn.Module):
    def __init__(self, cfg: ModelSettings, peak_cfg: PeakNormConfig, tokenizer_cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.encoder, peak_cfg)
        self.decoder = Decoder(cfg.decoder, tokenizer_cfg)

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.int32)).unsqueeze(0)

    def forward(self, peaks, peak_mask, tokens, token_mask, capture_attn=False):
        B, N = peak_mask.shape
        T = tokens.size(1)

        device = peaks.device

        memory = self.encoder(peaks, peak_mask)

        causal = self._causal_mask(T, device)
        pad_q = token_mask.unsqueeze(2).expand(B, T, T)
        pad_k = token_mask.unsqueeze(1).expand(B, T, T)
        tgt_mask = causal * pad_q * pad_k

        memory_mask = peak_mask.unsqueeze(1).expand(B, T, N)

        return self.decoder(tokens, memory, tgt_mask, memory_mask, capture_attn=capture_attn)

    @torch.no_grad()
    def generate(self, peaks, peak_mask, max_len=None, capture_attn=False):
        self.eval()
        cfg = self.cfg
        B = peaks.size(0)
        device = peaks.device
        max_len = max_len or cfg.max_token_len

        tokens = torch.full((B, 1), cfg.bos_token_id,
                            dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tmask = torch.ones_like(tokens, dtype=torch.int32)
            logits = self.forward(peaks, peak_mask, tokens,
                                  tmask, capture_attn=capture_attn)
            next_tok = logits[:, -1].argmax(dim=-1)
            next_tok = torch.where(finished, torch.full_like(
                next_tok, cfg.pad_token_id), next_tok)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == cfg.eos_token_id)
            if finished.all():
                break

        attn = self.decoder.cross_attn_stack() if capture_attn else None
        return tokens, attn
