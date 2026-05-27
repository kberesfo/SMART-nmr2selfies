from pydantic import BaseModel


class TokenFrequency(BaseModel):
    token: str
    count: int
    id: int


class SeqLenStats(BaseModel):
    min: int
    max: int
    mean: float
    p50: int
    p99: int


class VocabStats(BaseModel):
    n_molecules: int
    n_converted: int
    n_failed: int
    failure_rate: float
    vocab_size: int
    special_tokens: list[str]
    seq_len: SeqLenStats
    token_frequencies: list[TokenFrequency]


class Tokenizer(BaseModel):
    vocab: dict[str, int]        # token string -> token ID
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int
    special_tokens: list[str]
    max_seq_length: int          # p99 token length from training split