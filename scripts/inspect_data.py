"""
Inspect the data directory and print schema, statistics, and sample records.
Writes dataset_stats.json and peak_norm_stats.json to the artifacts/ directory.
Run with: pixi run -e dev python scripts/inspect_data.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import selfies
from pydantic import BaseModel

DATA_ROOT: Path = Path(os.environ.get("APP_DATA_PATH", "/data"))
STATS_DIR: Path = Path("artifacts/stats/")
SPLITS = ["train", "val", "test"]
PARQUET_FILES = ["HSQC_NMR", "C_NMR", "H_NMR", "FragIdx"]

# must match dataset.py sentinel thresholds
_DC_MAX = 300.0
_DH_MAX = 20.0


# --- stats models ---

class FieldStats(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class IntensityStats(FieldStats):
    negative_fraction: float


class PercentileStats(FieldStats):
    p05: float
    p50: float
    p95: float
    p99: float


class SentinelStats(BaseModel):
    total: int
    molecules_affected: int
    fraction_affected: float


class HSQCStats(BaseModel):
    split: str
    n_molecules: int
    peak_counts_raw: PercentileStats
    peak_counts_filtered: PercentileStats
    sentinel_peaks: SentinelStats
    dC: FieldStats
    dH: FieldStats
    intensity: IntensityStats


class TokenLengthStats(BaseModel):
    min: int
    max: int
    mean: float
    p95: float
    p99: float


class SELFIESConversionStats(BaseModel):
    n_sample: int
    failures: int
    failure_rate: float
    token_length: TokenLengthStats


class CrossReferenceStats(BaseModel):
    split: str
    hsqc_rows: int
    metadata_entries: int
    matched: int
    hsqc_missing_metadata: int
    metadata_without_hsqc: int


class PeakNormStats(BaseModel):
    dC: FieldStats
    dH: FieldStats
    intensity: IntensityStats
    note: str


class DatasetStats(BaseModel):
    row_counts: dict[str, dict[str, int]]
    hsqc_train: HSQCStats
    selfies_conversion: SELFIESConversionStats
    cross_reference: CrossReferenceStats


# --- inspection functions ---

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def inspect_parquet(split: str, name: str) -> None:
    path = DATA_ROOT / "arrow" / split / f"{name}.parquet"
    if not path.exists():
        print(f"  MISSING: {path}")
        return

    table = pq.read_table(path)
    print(f"\n  {name}.parquet  ({len(table):,} rows)")
    print(f"  Schema: {table.schema.to_string(show_field_metadata=False)}")

    sample = table.slice(0, 1).to_pydict()
    print(f"  Sample idx={sample['idx'][0]}:")
    for col, val in sample.items():
        v = val[0]
        if isinstance(v, list) and len(v) > 6:
            print(f"    {col}: [{', '.join(f'{x:.3f}' if isinstance(
                x, float) else str(x) for x in v[:6])}, ...] (len={len(v)})")
        else:
            print(f"    {col}: {v}")


def hsqc_stats(split: str) -> HSQCStats:
    """Compute HSQC peak statistics for a split, filtering sentinel values."""
    path = DATA_ROOT / "arrow" / split / "HSQC_NMR.parquet"
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
        valid = (np.abs(arr[:, 0]) <= _DC_MAX) & (np.abs(arr[:, 1]) <= _DH_MAX)
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

    return stats


def inspect_metadata() -> dict:
    path = DATA_ROOT / "metadata.json"
    if not path.exists():
        print("  MISSING: metadata.json")
        return {}

    with open(path) as f:
        meta = json.load(f)

    print(f"\n  metadata.json: {len(meta):,} entries")
    print(
        f"  Keys are: {type(list(meta.keys())[0]).__name__} (sample: {list(meta.keys())[:3]})")

    sample = meta[list(meta.keys())[0]]
    print(f"  Fields per entry: {list(sample.keys())}")
    print(f"  Sample smiles: {sample.get('smiles', 'MISSING')[:80]}")

    return meta


def selfies_conversion_check(meta: dict, n_sample: int = 1000) -> SELFIESConversionStats:
    keys = list(meta.keys())[:n_sample]
    failures = 0
    lengths = []

    for k in keys:
        smiles = meta[k].get(
            "canonical_2d_smiles") or meta[k].get("smiles", "")
        try:
            sf = selfies.encoder(smiles)
            tokens = list(selfies.split_selfies(sf))
            lengths.append(len(tokens))
        except Exception:
            failures += 1

    rate = failures / len(keys)
    lengths = np.array(lengths) if lengths else np.array([0])

    stats = SELFIESConversionStats(
        n_sample=len(keys),
        failures=failures,
        failure_rate=float(rate),
        token_length=TokenLengthStats(
            min=int(lengths.min()),
            max=int(lengths.max()),
            mean=float(lengths.mean()),
            p95=float(np.percentile(lengths, 95)),
            p99=float(np.percentile(lengths, 99)),
        ),
    )

    print(f"\n  SELFIES conversion (sample n={stats.n_sample}):")
    print(f"    failures: {stats.failures} ({stats.failure_rate:.2%})")
    if stats.failure_rate > 0.01:
        print("    WARNING: failure rate exceeds 1% threshold")
    print(f"    token length: min={stats.token_length.min}  max={stats.token_length.max}  mean={stats.token_length.mean:.1f}  p95={stats.token_length.p95:.0f}  p99={stats.token_length.p99:.0f}")

    return stats


def cross_reference_check(meta: dict, split: str) -> CrossReferenceStats:
    hsqc = pq.read_table(DATA_ROOT / "arrow" / split / "HSQC_NMR.parquet")
    hsqc_ids = set(hsqc.column("idx").to_pylist())
    meta_ids = set(int(k) for k in meta.keys())

    in_both = hsqc_ids & meta_ids
    hsqc_only = hsqc_ids - meta_ids
    meta_only = meta_ids - hsqc_ids

    stats = CrossReferenceStats(
        split=split,
        hsqc_rows=len(hsqc_ids),
        metadata_entries=len(meta_ids),
        matched=len(in_both),
        hsqc_missing_metadata=len(hsqc_only),
        metadata_without_hsqc=len(meta_only),
    )

    print(
        f"\n  HSQC rows: {stats.hsqc_rows:,}  |  metadata entries: {stats.metadata_entries:,}  |  matched: {stats.matched:,}")
    if hsqc_only:
        print(
            f"  WARNING: {stats.hsqc_missing_metadata:,} HSQC rows have no metadata entry")
    if meta_only:
        print(
            f"  {stats.metadata_without_hsqc:,} metadata entries have no HSQC row (other modalities only)")

    return stats


def row_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name in PARQUET_FILES:
        counts[name] = {}
        for split in SPLITS:
            path = DATA_ROOT / "arrow" / split / f"{name}.parquet"
            if path.exists():
                t = pq.read_table(path)
                counts[name][split] = len(t)
                print(f"  {name} / {split}: {len(t):,}")
    return counts


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    section("Directory layout")
    for split in SPLITS:
        split_dir = DATA_ROOT / "arrow" / split
        if split_dir.exists():
            files = [f.name for f in sorted(split_dir.iterdir())]
            print(f"  {split}/: {files}")
        else:
            print(f"  {split}/: MISSING")

    section("Parquet schemas and samples")
    for split in SPLITS[:1]:  # schema is identical across splits, show train only
        print(f"\n[{split}]")
        for name in PARQUET_FILES:
            inspect_parquet(split, name)

    section("Row counts across splits")
    counts = row_counts()

    section("HSQC peak statistics (train split)")
    train_hsqc_stats = hsqc_stats("train")

    section("metadata.json")
    meta = inspect_metadata()

    section("SELFIES conversion check")
    selfies_stats = selfies_conversion_check(meta, n_sample=1000)

    section("Cross-reference check")
    xref_stats = cross_reference_check(meta, "train")

    # --- write artifacts ---
    section("Writing artifacts")

    dataset_stats = DatasetStats(
        row_counts=counts,
        hsqc_train=train_hsqc_stats,
        selfies_conversion=selfies_stats,
        cross_reference=xref_stats,
    )
    stats_path = STATS_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dataset_stats.model_dump(), f, indent=2)
    print(f"  Wrote {stats_path}")

    # peak_norm_stats: train-split mean/std after sentinel filtering
    # used by PeakNormSettings once global normalization is implemented
    peak_norm = PeakNormStats(
        dC=train_hsqc_stats.dC,
        dH=train_hsqc_stats.dH,
        intensity=train_hsqc_stats.intensity,
        note="computed on train split after filtering sentinel peaks (|dC| > 300 or |dH| > 20)",
    )
    norm_path = STATS_DIR / "peak_norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(peak_norm.model_dump(), f, indent=2)
    print(f"  Wrote {norm_path}")


if __name__ == "__main__":
    main()
