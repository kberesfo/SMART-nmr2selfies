import torch
import torch.nn as nn

from .artifacts import PeakNormConfig, TokenizerConfig

from .encoder import Encoder
from .decoder import Decoder

from ..config import ModelSettings
from ..data import NMRBatch

# NOTE: STILL NEEDS REVIEW

class NMR2SELFIES(nn.Module):
    def __init__(self, cfg: ModelSettings, peak_cfg: PeakNormConfig, tokenizer_cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer_cfg = tokenizer_cfg
        self.encoder = Encoder(cfg.encoder, peak_cfg)
        self.decoder = Decoder(cfg.decoder, tokenizer_cfg)

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.int32)).unsqueeze(0)

    def _encode(self, peaks: torch.Tensor, peak_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of NMR peaks into memory for the decoder to attend to.

        Centralized here so that both forward (training) and generate (inference)
        go through the same path — any change to the encoding step only needs to
        happen in one place.

        Args:
            peaks (torch.Tensor): Peak features, shape (B, S, 3) — (δH, δC, intensity).
            peak_mask (torch.Tensor): Boolean padding mask, shape (B, S). True at
                padded positions; prevents cross-attention from attending to them.

        Returns:
            torch.Tensor: Encoder output (memory), shape (B, S, d_model).
        """
        return self.encoder(peaks, peak_mask)

    def forward(self, batch: NMRBatch, capture_attn=False) -> torch.Tensor:
        peaks = batch.peaks
        peak_mask = batch.peak_padding_mask
        tokens = batch.token_ids
        token_mask = batch.token_padding_mask

        T = tokens.size(1)
        device = peaks.device

        memory = self._encode(peaks, peak_mask)

        # causal mask only — padding is handled separately by tgt_padding_mask / memory_padding_mask
        tgt_mask = self._causal_mask(T, device)

        return self.decoder(tokens, memory, tgt_mask, token_mask, peak_mask, capture_attn=capture_attn)

    @torch.no_grad()
    def generate(
        self,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        max_len: int | None = None,
        capture_attn: bool = False,
    ):
        self.eval()
        tok = self.tokenizer_cfg
        B = peaks.size(0)
        device = peaks.device
        max_len = max_len or self.tokenizer_cfg.max_seq_length

        # seed each sequence in the batch with BOS
        tokens = torch.full((B, 1), tok.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # encode peaks once — reused by every decoder step
        memory = self._encode(peaks, peak_mask)

        for _ in range(max_len - 1):
            T = tokens.size(1)
            tgt_mask = self._causal_mask(T, device)
            # no padding in the growing token sequence, so tgt_padding_mask is all-false
            logits = self.decoder(
                tokens,
                memory,
                tgt_mask,
                tgt_padding_mask=torch.zeros_like(tokens, dtype=torch.bool),
                memory_padding_mask=peak_mask,
                capture_attn=capture_attn,
            )
            next_tok = logits[:, -1].argmax(dim=-1)
            # keep already-finished rows emitting PAD so output lengths stay consistent
            next_tok = torch.where(finished, torch.full_like(next_tok, tok.pad_token_id), next_tok)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == tok.eos_token_id)
            if finished.all():
                break

        attn = self.decoder.cross_attn_stack() if capture_attn else None
        return tokens, attn
