"""
Build the SELFIES tokenizer vocabulary from the training split.
Only runs over the train split to avoid data leakage into val/test.

Outputs:
  artifacts/tokenizer/tokenizer.json  — vocab mapping + special token IDs
  artifacts/stats/vocab_stats.json    — conversion stats and token frequencies

Run with: pixi run python scripts/build_vocab.py
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import selfies as sf
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("APP_DATA_PATH", "/data"))
STATS_DIR = Path("artifacts/stats")
TOKENIZER_DIR = Path("artifacts/tokenizer")

# special tokens — order defines their IDs and must never change after training
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


# --- output models ---

class TokenFrequency(BaseModel):
    token: str
    count: int
    id: int


class VocabStats(BaseModel):
    n_molecules: int
    n_converted: int
    n_failed: int
    failure_rate: float
    vocab_size: int
    special_tokens: list[str]
    token_frequencies: list[TokenFrequency]


class Tokenizer(BaseModel):
    vocab: dict[str, int]        # token string -> token ID
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int
    special_tokens: list[str]


# --- build functions ---

def load_train_ids(data_root: Path) -> list[int]:
    """Return molecule IDs present in the training split HSQC parquet."""
    path = data_root / "arrow" / "train" / "HSQC_NMR.parquet"
    table = pq.read_table(path, columns=["idx"])
    return table.column("idx").to_pylist()


def load_metadata(data_root: Path) -> dict:
    path = data_root / "metadata.json"
    with open(path) as f:
        return json.load(f)


def build_vocab(
    train_ids: list[int],
    metadata: dict,
) -> tuple[Tokenizer, VocabStats]:
    n_molecules = len(train_ids)
    n_failed = 0
    token_counts: Counter = Counter()
    lengths = []

    log.info(f"Converting {n_molecules:,} training molecules to SELFIES...")

    for idx in train_ids:
        entry = metadata.get(str(idx))
        if entry is None:
            n_failed += 1
            log.debug(f"idx={idx}: missing from metadata")
            continue

        smiles = entry.get("canonical_2d_smiles") or entry.get("smiles", "")
        if not smiles:
            n_failed += 1
            log.debug(f"idx={idx}: no SMILES found")
            continue

        try:
            selfies_str = sf.encoder(smiles)
            tokens = list(sf.split_selfies(selfies_str))
            token_counts.update(tokens)
            lengths.append(len(tokens))
        except Exception as e:
            n_failed += 1
            log.debug(f"idx={idx}: SELFIES conversion failed — {e}")

    n_converted = n_molecules - n_failed
    failure_rate = n_failed / n_molecules if n_molecules > 0 else 0.0

    if failure_rate > 0.01:
        log.warning(f"SELFIES failure rate {failure_rate:.2%} exceeds 1% threshold")

    log.info(f"Converted {n_converted:,} / {n_molecules:,} molecules ({n_failed:,} failed)")
    log.info(f"Unique SELFIES tokens: {len(token_counts):,}")

    # build vocab: special tokens first (IDs 0-3), then SELFIES tokens by frequency
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for token, _ in token_counts.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)

    tokenizer = Tokenizer(
        vocab=vocab,
        vocab_size=len(vocab),
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        unk_token_id=UNK_TOKEN_ID,
        special_tokens=SPECIAL_TOKENS,
    )

    # build stats — include all tokens with their assigned IDs
    token_frequencies = [
        TokenFrequency(token=tok, count=count, id=vocab[tok])
        for tok, count in token_counts.most_common()
    ]

    stats = VocabStats(
        n_molecules=n_molecules,
        n_converted=n_converted,
        n_failed=n_failed,
        failure_rate=failure_rate,
        vocab_size=len(vocab),
        special_tokens=SPECIAL_TOKENS,
        token_frequencies=token_frequencies,
    )

    return tokenizer, stats


def main() -> None:
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
