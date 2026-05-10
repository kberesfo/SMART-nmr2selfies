import os
import json
import logging
from pathlib import Path

from .modules import load_train_ids, load_metadata, build_vocab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("APP_DATA_PATH", "/data"))
STATS_DIR = Path("artifacts/stats")
TOKENIZER_DIR = Path("artifacts/tokenizer")


def main() -> None:
    """
    Build the SELFIES tokenizer vocabulary from the training split.
    Only runs over the train split to avoid data leakage into val/test.

    Outputs:
    artifacts/tokenizer/tokenizer.json  — vocab mapping + special token IDs
    artifacts/stats/vocab_stats.json    — conversion stats and token frequencies

    Run with: pixi run python scripts/build_vocab.py
    """
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading training split molecule IDs...")
    train_ids = load_train_ids(DATA_ROOT)
    log.info(f"Found {len(train_ids):,} molecules in train split")

    log.info("Loading metadata...")
    metadata = load_metadata(DATA_ROOT)
    log.info(f"Loaded {len(metadata):,} metadata entries")

    tokenizer, stats = build_vocab(train_ids, metadata)

    # write tokenizer
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer.model_dump(), f, indent=2)

    log.info(f"Wrote {tokenizer_path}  (vocab_size={tokenizer.vocab_size})")

    # write vocab stats
    stats_path = STATS_DIR / "vocab_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.model_dump(), f, indent=2)
        
    log.info(f"Wrote {stats_path}")

    # print summary
    print(f"\n  vocab_size:    {stats.vocab_size}")
    print(f"  converted:     {stats.n_converted:,} / {stats.n_molecules:,}")
    print(f"  failure_rate:  {stats.failure_rate:.4%}")
    print(f"\n  Top 10 tokens:")
    for tf in stats.token_frequencies[:10]:
        print(f"    id={tf.id:>4}  count={tf.count:>8,}  {tf.token}")


if __name__ == "__main__":
    main()
