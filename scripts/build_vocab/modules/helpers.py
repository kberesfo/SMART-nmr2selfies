import json
import logging
import pyarrow.parquet as pq
import selfies as sf

from pathlib import Path
from collections import Counter

from .dataclasses import Tokenizer, VocabStats, TokenFrequency

log = logging.getLogger(__name__)


# special tokens — order defines their IDs and must never change after training
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


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
        log.warning(
            f"SELFIES failure rate {failure_rate:.2%} exceeds 1% threshold")

    log.info(
        f"Converted {n_converted:,} / {n_molecules:,} molecules ({n_failed:,} failed)")
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
