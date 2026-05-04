from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_", extra="ignore")

    lr: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
