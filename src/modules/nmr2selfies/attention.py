import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerBlockSettings


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerBlockSettings):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.d_k = cfg.d_model // cfg.num_heads
        self.n_heads = cfg.num_heads
        ## TODO: ASK ABOUT THIS ##
        self.qkv_proj = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(3)]
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, q, k, v, mask=None, need_weights=False):
        B, Tq, D = q.shape

        def split(x, proj):
            return proj(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        qh = split(q, self.qkv_proj[0])
        kh = split(k, self.qkv_proj[1])
        vh = split(v, self.qkv_proj[2])

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn)

        out = torch.matmul(attn_dropped, vh)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.out_proj(out)

        # return pre-dropout weights for analysis -- post-dropout values
        # don't sum to 1 and are misleading in heatmaps
        return out, (attn if need_weights else None)
