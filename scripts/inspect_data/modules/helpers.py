import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import selfies

from .dataclasses import CrossReferenceStats, SELFIESConversionStats, TokenLengthStats

# --- inspection functions ---


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def inspect_parquet(data_path: Path,  split: str, name: str) -> None:
    path = data_path / "arrow" / split / f"{name}.parquet"

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


def inspect_metadata(data_path: Path) -> dict:
    path = data_path / "metadata.json"

    if not path.exists():
        print("  MISSING: metadata.json")
        return {}

    with open(path) as f:
        meta = json.load(f)

    print(f"\n  metadata.json: {len(meta):,} entries")
    print(
        f"  Keys are: {type(list(meta.keys())[0]).__name__} (sample: {list(meta.keys())[:3]})"
    )

    sample = meta[list(meta.keys())[0]]
    print(f"  Fields per entry: {list(sample.keys())}")
    print(f"  Sample smiles: {sample.get('smiles', 'MISSING')[:80]}")

    return meta


def row_counts(data_path: Path, parquet_files: list[str], splits: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name in parquet_files:

        counts[name] = {}

        for split in splits:
            path = data_path / "arrow" / split / f"{name}.parquet"

            if path.exists():
                t = pq.read_table(path)
                counts[name][split] = len(t)
                print(f"  {name} / {split}: {len(t):,}")

    return counts


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


def cross_reference_check(data_path: Path, meta: dict, split: str) -> CrossReferenceStats:
    hsqc = pq.read_table(data_path / "arrow" / split / "HSQC_NMR.parquet")
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
