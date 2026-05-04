# Configuration

Configuration is split into three concerns so env-level secrets stay out of git while experiment settings are versioned.

## Classes

### `AppSettings` (`config/app.py`)

Environment-level settings. Load from `.env` or shell env vars. Never commit these values.

| Field | Env var | Default | Description |
|---|---|---|---|
| `data_path` | `APP_DATA_PATH` | `/data` | Path to LMDB dataset |
| `checkpoint_path` | `APP_CHECKPOINT_PATH` | `/checkpoints` | Checkpoint output directory |
| `wandb_project` | `APP_WANDB_PROJECT` | `nmr2selfies` | W&B project name |
| `wandb_entity` | `APP_WANDB_ENTITY` | `None` | W&B team/entity |
| `device` | `APP_DEVICE` | `cuda` | Training device |

### `ModelSettings` (`config/model.py`)

Architecture hyperparameters. Version these in `configs/base.yaml`.

| Field | Env var | Default | Description |
|---|---|---|---|
| `encoder.num_layers` | — | `4` | Encoder transformer layers |
| `encoder.d_model` | — | `256` | Model dimension |
| `encoder.num_heads` | — | `8` | Attention heads |
| `encoder.ffn_dim` | — | `1024` | FFN inner dimension |
| `encoder.dropout` | — | `0.1` | Dropout rate |
| `decoder.*` | — | (same as encoder) | Decoder mirrors encoder defaults |
| `vocab_size` | `MODEL_VOCAB_SIZE` | `0` | Set after tokenizer is built |

Nested fields (`encoder`, `decoder`) can be overridden via a JSON env var:
```bash
MODEL_ENCODER='{"num_layers": 6, "d_model": 512}'
```
In practice, override these in `configs/base.yaml` instead.

### `TrainSettings` (`config/train.py`)

Training hyperparameters. Version these in `configs/base.yaml`.

| Field | Env var | Default | Description |
|---|---|---|---|
| `lr` | `TRAIN_LR` | `1e-4` | Peak learning rate |
| `batch_size` | `TRAIN_BATCH_SIZE` | `64` | Batch size |
| `max_epochs` | `TRAIN_MAX_EPOCHS` | `50` | Training epochs |
| `warmup_steps` | `TRAIN_WARMUP_STEPS` | `1000` | LR warmup steps |
| `weight_decay` | `TRAIN_WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `grad_clip` | `TRAIN_GRAD_CLIP` | `1.0` | Gradient clip norm |
| `num_workers` | `TRAIN_NUM_WORKERS` | `4` | DataLoader workers |

## Composed root

`Config` composes all three and is the single import for most code:

```python
from modules.config import Config

cfg = Config()
print(cfg.app.data_path)
print(cfg.model.encoder.num_layers)
print(cfg.train.lr)
```

## Overriding values

**Via env var** (useful in CI, Docker, or one-off runs):
```bash
TRAIN_LR=3e-4 TRAIN_BATCH_SIZE=128 pixi run train
```

**Via `.env` file** (local dev secrets, never committed):
```bash
APP_WANDB_PROJECT=my-project
APP_DATA_PATH=/mnt/data
```

**Via LightningCLI** (hyperparameter sweeps):
```bash
pixi run train --trainer.max_epochs 30 --model.lr 1e-4
```
