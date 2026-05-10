from pydantic import BaseModel


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
