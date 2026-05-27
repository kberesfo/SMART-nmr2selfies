import json

from pathlib import Path
from pydantic import BaseModel, model_validator


class FieldStats(BaseModel):
    """Per-field descriptive statistics for shift fields (dC, dH)."""
    min: float
    max: float
    mean: float
    std: float


class PeakNormConfig(BaseModel):
    """Loaded from artifacts/stats/peak_norm_stats.json."""
    dC: FieldStats
    dH: FieldStats
    intensity: FieldStats
    note: str

    @property
    def peak_mean(self) -> tuple[float, float, float]:
        return (self.dC.mean, self.dH.mean, self.intensity.mean)

    @property
    def peak_std(self) -> tuple[float, float, float]:
        return (self.dC.std,  self.dH.std,  self.intensity.std)

    @classmethod
    def from_file(cls, path: Path) -> "PeakNormConfig":
        with open(path) as f:
            return cls.model_validate(json.load(f))


class TokenizerConfig(BaseModel):
    """Loaded from artifacts/tokenizer/tokenizer.json."""
    vocab: dict[str, int]
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int
    special_tokens: list[str]
    max_seq_length: int  # p99 token length from dataset stats

    @model_validator(mode="after")
    def check_vocab_size(self) -> "TokenizerConfig":
        if len(self.vocab) != self.vocab_size:
            raise ValueError(
                f"vocab length {len(self.vocab)} does not match vocab_size {self.vocab_size}"
            )
        return self

    @model_validator(mode="after")
    def check_special_token_ids(self) -> "TokenizerConfig":
        expected = {
            "[PAD]": self.pad_token_id,
            "[BOS]": self.bos_token_id,
            "[EOS]": self.eos_token_id,
            "[UNK]": self.unk_token_id,
        }
        for token, expected_id in expected.items():
            actual_id = self.vocab.get(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"{token} has id {actual_id} in vocab but expected {expected_id}"
                )
        return self

    @model_validator(mode="after")
    def check_max_seq_length(self) -> "TokenizerConfig":
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )
        return self

    @classmethod
    def from_file(cls, path: Path) -> "TokenizerConfig":
        with open(path) as f:
            return cls.model_validate(json.load(f))
