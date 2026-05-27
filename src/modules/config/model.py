from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class TransformerBlockSettings(BaseModel):
    """Shared hyperparameters for any transformer encoder or decoder block."""
    num_layers: int = 4
    d_model: int = 256
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


class EncoderSettings(TransformerBlockSettings):
    # number of input features per peak (δC, δH, intensity)
    peak_feature_dim: int = 3


class DecoderSettings(TransformerBlockSettings):
    d_ff: int = 1024
    

class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    encoder: EncoderSettings = EncoderSettings()
    decoder: DecoderSettings = DecoderSettings()
