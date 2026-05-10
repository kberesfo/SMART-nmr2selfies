
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

from .dataclasses import (
    FieldStats, HSQCStats, IntensityStats,
    PercentileStats, SentinelStats
)

def print_hsqc_stats(stats: HSQCStats) -> None:
    print(f"\n  Peak counts (raw):      min={stats.peak_counts_raw.min:.0f}  max={stats.peak_counts_raw.max:.0f}  mean={stats.peak_counts_raw.mean:.1f}  p95={stats.peak_counts_raw.p95:.0f}  p99={stats.peak_counts_raw.p99:.0f}")
    print(f"  Peak counts (filtered): min={stats.peak_counts_filtered.min:.0f}  max={stats.peak_counts_filtered.max:.0f}  mean={stats.peak_counts_filtered.mean:.1f}  p95={stats.peak_counts_filtered.p95:.0f}  p99={stats.peak_counts_filtered.p99:.0f}")
    print(
        f"  Sentinel peaks: {stats.sentinel_peaks.total:,} total, affecting {stats.sentinel_peaks.fraction_affected:.2%} of molecules")
    print(
        f"\n  δC:        mean={stats.dC.mean:.2f}  std={stats.dC.std:.2f}  range=[{stats.dC.min:.1f}, {stats.dC.max:.1f}]")
    print(
        f"  δH:        mean={stats.dH.mean:.2f}  std={stats.dH.std:.2f}  range=[{stats.dH.min:.1f}, {stats.dH.max:.1f}]")
    print(
        f"  Intensity: mean={stats.intensity.mean:.1f}  std={stats.intensity.std:.1f}  negative_fraction={stats.intensity.negative_fraction:.3f}")


def hsqc_stats(data_path: Path, split: str, dc_max: float, dh_max: float) -> HSQCStats:
    """Compute HSQC peak statistics for a split, filtering sentinel values."""
    path = data_path / "arrow" / split / "HSQC_NMR.parquet"
    table = pq.read_table(path)
    rows = table.to_pydict()

    peak_counts_raw = []
    peak_counts_filtered = []
    dC_vals, dH_vals, intensity_vals = [], [], []
    sentinel_counts = []

    for data, shape in zip(rows["data"], rows["shape"]):
        arr = np.array(data).reshape(shape)  # (N, 3)
        peak_counts_raw.append(shape[0])

        # apply same sentinel filtering as dataset.py
        valid = (np.abs(arr[:, 0]) <= dc_max) & (np.abs(arr[:, 1]) <= dh_max)
        sentinel_counts.append(int((~valid).sum()))
        arr = arr[valid]

        peak_counts_filtered.append(len(arr))
        dC_vals.extend(arr[:, 0].tolist())
        dH_vals.extend(arr[:, 1].tolist())
        intensity_vals.extend(arr[:, 2].tolist())

    peak_counts_raw = np.array(peak_counts_raw)
    peak_counts_filtered = np.array(peak_counts_filtered)
    dC_vals = np.array(dC_vals)
    dH_vals = np.array(dH_vals)
    intensity_vals = np.array(intensity_vals)
    sentinel_counts = np.array(sentinel_counts)

    stats = HSQCStats(
        split=split,
        n_molecules=len(peak_counts_raw),
        peak_counts_raw=PercentileStats(
            min=float(peak_counts_raw.min()),
            max=float(peak_counts_raw.max()),
            mean=float(peak_counts_raw.mean()),
            std=float(peak_counts_raw.std()),
            p05=float(np.percentile(peak_counts_raw, 5)),
            p50=float(np.percentile(peak_counts_raw, 50)),
            p95=float(np.percentile(peak_counts_raw, 95)),
            p99=float(np.percentile(peak_counts_raw, 99)),
        ),
        peak_counts_filtered=PercentileStats(
            min=float(peak_counts_filtered.min()),
            max=float(peak_counts_filtered.max()),
            mean=float(peak_counts_filtered.mean()),
            std=float(peak_counts_filtered.std()),
            p05=float(np.percentile(peak_counts_filtered, 5)),
            p50=float(np.percentile(peak_counts_filtered, 50)),
            p95=float(np.percentile(peak_counts_filtered, 95)),
            p99=float(np.percentile(peak_counts_filtered, 99)),
        ),
        sentinel_peaks=SentinelStats(
            total=int(sentinel_counts.sum()),
            molecules_affected=int((sentinel_counts > 0).sum()),
            fraction_affected=float((sentinel_counts > 0).mean()),
        ),
        dC=FieldStats(
            min=float(dC_vals.min()),
            max=float(dC_vals.max()),
            mean=float(dC_vals.mean()),
            std=float(dC_vals.std()),
        ),
        dH=FieldStats(
            min=float(dH_vals.min()),
            max=float(dH_vals.max()),
            mean=float(dH_vals.mean()),
            std=float(dH_vals.std()),
        ),
        intensity=IntensityStats(
            min=float(intensity_vals.min()),
            max=float(intensity_vals.max()),
            mean=float(intensity_vals.mean()),
            std=float(intensity_vals.std()),

            negative_fraction=float((intensity_vals < 0).mean()),
        ),
    )

    print_hsqc_stats(stats)

    return stats
