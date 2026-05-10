import os
import json
from pathlib import Path

from .modules import (
    section,
    inspect_parquet,
    inspect_metadata,
    row_counts,
    hsqc_stats,
    selfies_conversion_check,
    cross_reference_check,
    DatasetStats,
    PeakNormStats
)

DATA_PATH: Path = Path(os.environ.get("APP_DATA_PATH", "/data"))
STATS_PATH: Path = Path("artifacts/stats/")
SPLITS = ["train", "val", "test"]
PARQUET_FILES = ["HSQC_NMR", "C_NMR", "H_NMR", "FragIdx"]

# must match dataset.py sentinel thresholds
_DC_MAX = 300.0
_DH_MAX = 20.0


def main() -> None:
    """
    Inspect the data directory and print schema, statistics, and sample records.
    Writes dataset_stats.json and peak_norm_stats.json to the artifacts/ directory.
    Run with: pixi run -e dev python scripts/inspect_data.py
    """

    STATS_PATH.mkdir(parents=True, exist_ok=True)

    section("Directory layout")
    for split in SPLITS:
        split_dir = DATA_PATH / "arrow" / split
        if split_dir.exists():
            files = [f.name for f in sorted(split_dir.iterdir())]
            print(f"  {split}/: {files}")
        else:
            print(f"  {split}/: MISSING")

    section("Parquet schemas and samples")
    for split in SPLITS[:1]:  # schema is identical across splits, show train only
        print(f"\n[{split}]")
        for name in PARQUET_FILES:
            inspect_parquet(DATA_PATH, split, name)

    section("Row counts across splits")
    counts = row_counts(DATA_PATH, PARQUET_FILES, SPLITS)

    section("HSQC peak statistics (train split)")
    train_hsqc_stats = hsqc_stats(DATA_PATH, "train", _DC_MAX, _DH_MAX)

    section("metadata.json")
    meta = inspect_metadata(DATA_PATH)

    section("SELFIES conversion check")
    selfies_stats = selfies_conversion_check(meta, n_sample=1000)

    section("Cross-reference check")
    xref_stats = cross_reference_check(DATA_PATH, meta, "train")

    # --- write artifacts ---
    section("Writing artifacts")

    dataset_stats = DatasetStats(
        row_counts=counts,
        hsqc_train=train_hsqc_stats,
        selfies_conversion=selfies_stats,
        cross_reference=xref_stats,
    )
    stats_path = STATS_PATH / "dataset_stats.json"
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
    norm_path = STATS_PATH / "peak_norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(peak_norm.model_dump(), f, indent=2)
    print(f"  Wrote {norm_path}")


if __name__ == "__main__":
    main()
