import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .artifacts import PeakNormConfig

from ..config import EncoderSettings


class PeakEmbedding(nn.Module):
    def __init__(self, cfg: EncoderSettings, stats: PeakNormConfig):
        super().__init__()
        # buffer stores mean for each of the 3 peak features; it is a not learnable parameters
        # reshapes the stats from shape (3,) to (1, 1, 3) so they broadcast
        self.register_buffer(
            "mean",
            torch.tensor(stats.peak_mean).view(1, 1, -1)
        )
        # buffer stores std for each of the 3 peak features; it is a not learnable parameters
        # reshapes the stats from shape (3,) to (1, 1, 3) so they broadcast
        self.register_buffer(
            "std",
            torch.tensor(stats.peak_std).view(1, 1, -1)
        )

        # std multilayer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(cfg.peak_feature_dim, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model),
        )

    def _normalize_intensity(self, peaks: torch.Tensor) -> torch.Tensor:
        """
        Per-molecule intensity normalization.
        Positive intensities scaled by that molecule's own p95.
        Negative intensities scaled by that molecule's own p05.
        Pure tensor ops — GPU safe.

        Args:
            peaks (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        intensity = peaks[..., 2].clone()  # (B, N)
        eps = 1e-12

        pos_mask = intensity > 0
        neg_mask = intensity < 0

        # compute per-molecule percentiles — shape (B,)
        # we need to handle molecules that have no positive or negative peaks
        scaled = torch.zeros_like(intensity)

        for b in range(intensity.shape[0]):
            pos_vals = intensity[b][pos_mask[b]]
            neg_vals = intensity[b][neg_mask[b]]

            if pos_vals.numel() > 0:
                p_hi = torch.quantile(pos_vals, 0.95)
                if p_hi > eps:
                    scaled[b][pos_mask[b]] = pos_vals / p_hi

            if neg_vals.numel() > 0:
                p_lo = torch.quantile(neg_vals, 0.05)
                if p_lo.abs() > eps:
                    scaled[b][neg_mask[b]] = neg_vals / p_lo.abs()

        peaks = peaks.clone()
        peaks[..., 2] = torch.clamp(scaled, -1.0, 1.0)
        return peaks

    def _normalize_shifts(self, peaks: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalization for δC (col 0) and δH (col 1)
        using training set mean and std stored as buffers.

        Args:
            peaks (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        peaks = peaks.clone()

        peaks[..., :2] = (
            peaks[..., :2] - self.mean[..., :2]
        ) / (self.std[..., :2] + 1e-12)

        return peaks

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        # normalize shifts and intensity separately, then concatenate and project
        x: torch.Tensor = self._normalize_shifts(peaks)
        x: torch.Tensor = self._normalize_intensity(peaks)

        return self.mlp(x)


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
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, peak_mask=None) -> torch.Tensor:
        # (B, N, D)
        B, N, _ = x.shape
        m = peak_mask.unsqueeze(1).expand(B, N, N)

        a, _ = self.self_attn(x, x, x, mask=m)

        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        
        return x


class Encoder(nn.Module):

    def __init__(self, cfg: EncoderSettings):
        super().__init__()
        pass

    def encode(self, peaks, peak_mask):
        raise NotImplementedError(
            "Encoder encode method not implemented yet"
        )
