"""
Inspect the data directory and print schema, statistics, and sample records.
Run with: pixi run -e dev python scripts/inspect_data.py
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import selfies

DATA_ROOT = Path("/data")
SPLITS = ["train", "val", "test"]
PARQUET_FILES = ["HSQC_NMR", "C_NMR", "H_NMR", "FragIdx"]


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
            print(f"    {col}: [{', '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in v[:6])}, ...] (len={len(v)})")
        else:
            print(f"    {col}: {v}")


def hsqc_stats(split: str) -> None:
    path = DATA_ROOT / "arrow" / split / "HSQC_NMR.parquet"
    table = pq.read_table(path)
    rows = table.to_pydict()

    peak_counts = []
    dC_vals, dH_vals, intensity_vals = [], [], []

    for data, shape in zip(rows["data"], rows["shape"]):
        n_peaks = shape[0]
        peak_counts.append(n_peaks)
        arr = np.array(data).reshape(shape)
        dC_vals.extend(arr[:, 0].tolist())
        dH_vals.extend(arr[:, 1].tolist())
        intensity_vals.extend(arr[:, 2].tolist())

    peak_counts = np.array(peak_counts)
    dC_vals = np.array(dC_vals)
    dH_vals = np.array(dH_vals)
    intensity_vals = np.array(intensity_vals)

    print(f"\n  Peak counts per molecule:")
    print(f"    min={peak_counts.min()}  max={peak_counts.max()}  "
          f"mean={peak_counts.mean():.1f}  p50={np.percentile(peak_counts, 50):.0f}  "
          f"p95={np.percentile(peak_counts, 95):.0f}  p99={np.percentile(peak_counts, 99):.0f}")

    print(f"\n  δC (carbon shift):")
    print(f"    min={dC_vals.min():.2f}  max={dC_vals.max():.2f}  "
          f"mean={dC_vals.mean():.2f}  std={dC_vals.std():.2f}")

    print(f"\n  δH (proton shift):")
    print(f"    min={dH_vals.min():.2f}  max={dH_vals.max():.2f}  "
          f"mean={dH_vals.mean():.2f}  std={dH_vals.std():.2f}")

    print(f"\n  Intensity:")
    print(f"    min={intensity_vals.min():.1f}  max={intensity_vals.max():.1f}  "
          f"mean={intensity_vals.mean():.1f}  std={intensity_vals.std():.1f}")
    print(f"    negative fraction: {(intensity_vals < 0).mean():.3f}")


def inspect_metadata() -> dict:
    path = DATA_ROOT / "metadata.json"
    if not path.exists():
        print("  MISSING: metadata.json")
        return {}

    with open(path) as f:
        meta = json.load(f)

    print(f"\n  metadata.json: {len(meta):,} entries")
    print(f"  Keys are: {type(list(meta.keys())[0]).__name__} (sample: {list(meta.keys())[:3]})")

    sample = meta[list(meta.keys())[0]]
    print(f"  Fields per entry: {list(sample.keys())}")
    print(f"  Sample smiles: {sample.get('smiles', 'MISSING')[:80]}")

    return meta


def selfies_conversion_check(meta: dict, n_sample: int = 1000) -> None:
    keys = list(meta.keys())[:n_sample]
    failures = 0
    lengths = []

    for k in keys:
        smiles = meta[k].get("canonical_2d_smiles") or meta[k].get("smiles", "")
        try:
            sf = selfies.encoder(smiles)
            tokens = list(selfies.split_selfies(sf))
            lengths.append(len(tokens))
        except Exception:
            failures += 1

    rate = failures / len(keys)
    print(f"\n  SELFIES conversion (sample n={len(keys)}):")
    print(f"    failures: {failures} ({rate:.2%})")
    if rate > 0.01:
        print(f"    WARNING: failure rate exceeds 1% threshold")
    if lengths:
        lengths = np.array(lengths)
        print(f"    token length: min={lengths.min()}  max={lengths.max()}  "
              f"mean={lengths.mean():.1f}  p95={np.percentile(lengths, 95):.0f}  "
              f"p99={np.percentile(lengths, 99):.0f}")


def cross_reference_check(meta: dict, split: str) -> None:
    hsqc = pq.read_table(DATA_ROOT / "arrow" / split / "HSQC_NMR.parquet")
    hsqc_ids = set(hsqc.column("idx").to_pylist())
    meta_ids = set(int(k) for k in meta.keys())

    in_both = hsqc_ids & meta_ids
    hsqc_only = hsqc_ids - meta_ids
    meta_only = meta_ids - hsqc_ids

    print(f"\n  Cross-reference HSQC ↔ metadata ({split}):")
    print(f"    HSQC rows: {len(hsqc_ids):,}")
    print(f"    metadata entries: {len(meta_ids):,}")
    print(f"    matched: {len(in_both):,}")
    if hsqc_only:
        print(f"    WARNING: {len(hsqc_only):,} HSQC rows have no metadata entry")
    if meta_only:
        print(f"    {len(meta_only):,} metadata entries have no HSQC row (other modalities only)")


def main() -> None:
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
    for name in PARQUET_FILES:
        counts = []
        for split in SPLITS:
            path = DATA_ROOT / "arrow" / split / f"{name}.parquet"
            if path.exists():
                t = pq.read_table(path)
                counts.append(f"{split}={len(t):,}")
        print(f"  {name}: {' | '.join(counts)}")

    section("HSQC peak statistics (train split)")
    hsqc_stats("train")

    section("metadata.json")
    meta = inspect_metadata()

    section("SELFIES conversion check")
    if meta:
        selfies_conversion_check(meta, n_sample=1000)

    section("Cross-reference check")
    if meta:
        cross_reference_check(meta, "train")

    section("Summary for dataset implementation")
    print("""
  Data format:  Apache Arrow / Parquet  (NOT LMDB)
  HSQC peaks:  /data/arrow/{split}/HSQC_NMR.parquet
                columns: idx (int64), data (flat float64 list), shape ([N, 3])
                layout:  data[i*3+0] = δC,  data[i*3+1] = δH,  data[i*3+2] = intensity
  SMILES:       /data/metadata.json  keyed by str(idx)
                use 'canonical_2d_smiles' for SELFIES conversion
  Splits:       /data/arrow/train|val|test
  """)


if __name__ == "__main__":
    main()
