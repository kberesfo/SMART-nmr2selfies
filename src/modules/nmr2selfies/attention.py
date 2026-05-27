import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerBlockSettings


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerBlockSettings):
        super().__init__()
        # Throw assertion error if d_model is not divisible by num_heads
        assert cfg.d_model % cfg.num_heads == 0
        self.d_k = cfg.d_model // cfg.num_heads
        self.n_heads = cfg.num_heads

        # Create linear layers for query, key, value projections and output projection
        self.qkv_proj = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(3)]
        )

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False
    ):
        """
        Multi-head scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (B, T_q, D).
            k (torch.Tensor): Key tensor of shape (B, T_k, D).
            v (torch.Tensor): Value tensor of shape (B, T_k, D).
            key_padding_mask (torch.Tensor, optional): Boolean mask of shape (B, T_k).
                True = padding position, will be filled with -inf before softmax.
                Used to prevent attention to padding tokens or peaks.
            attn_mask (torch.Tensor, optional): Integer mask of shape (1, T_q, T_k)
                or (B, T_q, T_k). 0 = ignore, 1 = attend. Used for causal masking
                in decoder self-attention. Applied before key_padding_mask.
            need_weights (bool): If True, returns pre-dropout attention weights for
                analysis. Post-dropout weights are not returned as they don't sum to
                1 and are misleading in heatmaps. Defaults to False.

        Returns:
            out (torch.Tensor): Attended output of shape (B, T_q, D).
            attn (torch.Tensor | None): Attention weights of shape
                (B, n_heads, T_q, T_k) if need_weights=True, else None.
        """
        ## RESHAPE ##
        # unpack batch size (B), sequence length (Tq), and model dimension (D)
        B, Tq, D = q.shape

        # splits last dim into heads → (B, N, n_heads, d_k). .transpose(1,2) → (B, n_heads, N, d_k)
        def split(x, proj):
            return proj(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        qh = split(q, self.qkv_proj[0])
        kh = split(k, self.qkv_proj[1])
        vh = split(v, self.qkv_proj[2])
        ## SCORE ##
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_k)
        if key_padding_mask is not None:
            # (B, N) → (B, 1, 1, N), True = padding = -inf
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        if attn_mask is not None:
            # (1, T, T) or (B, T, T), 0 = ignore = -inf
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_dropped = self.dropout(attn)

        out = torch.matmul(attn_dropped, vh)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        # calling forward on out_proj applies the linear transformation
        out = self.out_proj(out)

        # return pre-dropout weights for analysis -- post-dropout values
        # don't sum to 1 and are misleading in heatmaps
        return out, (attn if need_weights else None)
