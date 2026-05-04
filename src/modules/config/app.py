from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_path: Path = Path("/data")
    checkpoint_path: Path = Path("/checkpoints")
    wandb_project: str = "nmr2selfies"
    wandb_entity: str | None = None
    device: str = "cuda"
