from pydantic import BaseModel

from .app import AppSettings
from .model import DecoderSettings, EncoderSettings, ModelSettings, TransformerBlockSettings
from .train import TrainSettings


class Config(BaseModel):
    app: AppSettings = AppSettings()
    model: ModelSettings = ModelSettings()
    train: TrainSettings = TrainSettings()


__all__ = [
    "AppSettings",
    "ModelSettings",
    "TransformerBlockSettings",
    "EncoderSettings",
    "DecoderSettings",
    "TrainSettings",
    "Config",
]
