import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .artifacts import TokenizerConfig
from .tokenizer import TokenEmbedding

from ..config import DecoderSettings


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderSettings):
        super().__init__()
        # self-attention layer for the decoder
        self.self_attn = MultiHeadAttention(cfg)
        # cross-attention layer for attending to encoder outputs
        self.cross_attn = MultiHeadAttention(cfg)

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

    def forward(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_padding_mask, capture_attn=False):
        """Single decoder layer: causal self-attention → cross-attention → FFN (Pre-LN).

        Args:
            tgt (torch.Tensor): Target sequence embeddings, shape (B, T, d_model).
            memory (torch.Tensor): Encoder output (NMR peak encodings), shape (B, S, d_model).
            tgt_mask (torch.Tensor): Causal mask, shape (1, T, T) or (B, T, T). Entry 0 at
                position (i, j) where j > i fills the score with -inf, preventing the model
                from attending to future tokens during teacher-forced training.
            tgt_padding_mask (torch.Tensor): Boolean padding mask for target tokens,
                shape (B, T). True at positions that are padding; those positions receive
                -inf attention scores in the self-attention sublayer.
            memory_padding_mask (torch.Tensor): Boolean padding mask for encoder input,
                shape (B, S). True at padded peak positions; prevents cross-attention
                from attending to peaks that were added only to fill the batch.
            capture_attn (bool): If True, stores the cross-attention weight tensor
                (B, n_heads, T, S) in self.last_cross_attn for later inspection.
                Defaults to False.

        Returns:
            torch.Tensor: Updated target representations, shape (B, T, d_model).
        """
        # Pre-LN: normalize input before each sublayer, add back to original tgt

        # self-attention sublayer
        normed = self.norm1(tgt)
        a, _ = self.self_attn(
            normed, normed, normed,
            key_padding_mask=tgt_padding_mask, attn_mask=tgt_mask)
        tgt = tgt + self.drop(a)

        # cross-attention sublayer
        normed = self.norm2(tgt)
        a, w = self.cross_attn(
            normed, memory, memory,
            key_padding_mask=memory_padding_mask,
            attn_mask=None,
            need_weights=capture_attn,
        )

        if capture_attn:
            self.last_cross_attn = w.detach()

        # FFN sublayer
        tgt = tgt + self.drop(a)
        tgt = tgt + self.drop(self.ff(self.norm3(tgt)))

        return tgt


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderSettings, tokenizer_cfg: TokenizerConfig):
        super().__init__()
        self.embed = TokenEmbedding(cfg, tokenizer_cfg)

        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )

        # optional final norm (common in Pre-LN models)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, tokenizer_cfg.vocab_size)

    def forward(self, tokens, memory, tgt_mask, tgt_padding_mask, memory_padding_mask, capture_attn=False):
        """Embed tokens and run them through all decoder layers.

        Args:
            tokens (torch.Tensor): Integer token IDs, shape (B, T). Sequence starts with
                BOS; during teacher forcing T is the full target length (up to max_seq_length=160).
            memory (torch.Tensor): Encoder output (NMR peak encodings), shape (B, S, d_model).
            tgt_mask (torch.Tensor): Causal mask for decoder self-attention, shape (1, T, T)
                or (B, T, T). See DecoderLayer.forward for semantics.
            tgt_padding_mask (torch.Tensor): Boolean padding mask for target tokens, shape (B, T).
                True at padding positions. Passed to the self-attention sublayer of each layer.
            memory_padding_mask (torch.Tensor): Boolean padding mask for encoder input, shape (B, S).
                True at padded peak positions. Passed to the cross-attention sublayer of each layer.
            capture_attn (bool): If True, each layer stores its cross-attention weights for
                later retrieval via cross_attn_stack(). Defaults to False.

        Returns:
            torch.Tensor: Unnormalized logits over the SELFIES vocabulary,
                shape (B, T, vocab_size).
        """
        x = self.embed(tokens)

        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask,
                tgt_padding_mask,
                memory_padding_mask,
                capture_attn=capture_attn
            )

        return self.out_proj(self.final_norm(x))

    def cross_attn_stack(self) -> torch.Tensor:
        w = [layer.last_cross_attn for layer in self.layers]

        assert all(
            x is not None for x in w), "run forward(..., capture_attn=True) first"

        return torch.stack(w, dim=0)
