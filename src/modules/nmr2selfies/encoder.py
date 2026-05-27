import torch
import torch.nn as nn

from .artifacts import PeakNormConfig
from .attention import MultiHeadAttention

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
        Per-molecule intensity normalization. Scales positive intensities by the
        molecule's own p95 and negative intensities by its p05 magnitude, then clamps
        to [-1, 1]. Operates per-molecule so batch composition does not affect values.

        Args:
            peaks (torch.Tensor): Raw peak features, shape (B, S, 3) where col 2 is intensity.

        Returns:
            torch.Tensor: Shape (B, S, 3) with col 2 replaced by normalized intensity.
                Cols 0-1 (δC, δH) are unchanged.
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
        Z-score normalization for δC (col 0) and δH (col 1) using training-set
        mean and std stored as non-learnable buffers. Intensity (col 2) is unchanged.

        Args:
            peaks (torch.Tensor): Raw peak features, shape (B, S, 3).

        Returns:
            torch.Tensor: Shape (B, S, 3) with cols 0-1 z-scored. Col 2 is unchanged.
        """
        peaks = peaks.clone()

        peaks[..., :2] = (
            peaks[..., :2] - self.mean[..., :2]
        ) / (self.std[..., :2] + 1e-12)

        return peaks

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        # normalize shifts and intensity separately, then concatenate and project
        # x: torch.Tensor = self._normalize_shifts(peaks)
        x: torch.Tensor = self._normalize_intensity(peaks)

        return self.mlp(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderSettings):
        super().__init__()

        self.self_attn = MultiHeadAttention(cfg)

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
        """
        Single encoder layer: self-attention → FFN (Pre-LN). No positional encoding;
        NMR peaks are treated as an unordered set.

        Args:
            x (torch.Tensor): Peak embeddings, shape (B, S, d_model).
            peak_mask (torch.Tensor, optional): Boolean padding mask, shape (B, S).
                True at positions that are padding peaks added to fill the batch.
                Defaults to None (no masking).

        Returns:
            torch.Tensor: Updated peak representations, shape (B, S, d_model).
        """
        # Pre norm architecture: https://arxiv.org/pdf/2002.04745.pdf
        normed = self.norm1(x)
        a, _ = self.self_attn(
            normed, normed, normed,
            key_padding_mask=peak_mask
        )

        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))

        return x


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderSettings, stats: PeakNormConfig):
        super().__init__()
        self.embed = PeakEmbedding(cfg, stats)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.num_layers)]
        )

    def forward(self, peaks, peak_mask):
        """
        Embed raw NMR peaks and encode them through all encoder layers.

        Args:
            peaks (torch.Tensor): Raw peak features, shape (B, S, 3) where the 3 features
                are (δC, δH, intensity). S is the padded number of peaks per molecule.
            peak_mask (torch.Tensor): Boolean padding mask, shape (B, S). True at positions
                that are padding peaks. Propagated to every layer to prevent attention to them.

        Returns:
            torch.Tensor: Contextualised peak encodings, shape (B, S, d_model).
                Passed directly to the decoder as memory.
        """
        x = self.embed(peaks)
        for layer in self.layers:
            x = layer(x, peak_mask)
        return x
