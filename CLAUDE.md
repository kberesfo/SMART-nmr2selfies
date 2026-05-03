# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nmr2selfies** trains a transformer encoder-decoder to predict SELFIES molecular strings from HSQC NMR spectra. Each input is a variable-length set of `(δH, δC, intensity)` peaks; each output is a SELFIES string. The project is currently in Phase 1 (go/no-go validation). See `plan.md` for the full phased roadmap, architecture rationale, and success criteria.

## Development Environment

The project runs in a VS Code Dev Container (Python 3.13, NVIDIA CUDA). Data is mounted from the host at `/data`; checkpoints at `/checkpoints`. Container env vars (`DATA_PATH`, `CHECKPOINT_PATH`) are set in `.devcontainer/devcontainer.json`.

Secrets (W&B API key) go in `.devcontainer/.env` — this file must not be committed.

## Commands

The project will use `pixi` for environment management (`pixi.toml` not yet written). Once set up, the primary entry point is LightningCLI:

```bash
# Train with base config
pixi run train

# Override config values on the CLI
pixi run train --trainer.max_epochs 30 --model.lr 1e-4

# Single-GPU overfit smoke test (verify loss reaches near-zero on 4 batches)
pixi run train --trainer.overfit_batches 4 --trainer.max_epochs 50
```

Configuration is in `.config/base.yaml`. Individual fields can be overridden on the CLI.

## Architecture

Standard encoder-decoder transformer (~12M params):

- **Encoder**: 4 layers, 8 heads, `d_model=256`, FFN dim 1024. Input peaks are projected through a small MLP. **No positional encoding** — NMR peaks are an unordered set; positional encodings would let the model exploit arbitrary ordering.
- **Decoder**: 4 layers, 8 heads, `d_model=256`. Standard causal self-attention + cross-attention to encoder output. Sinusoidal positional encoding on output tokens.
- **Output**: linear projection to SELFIES vocab size.

Source lives in `src/`. The planned package layout (from `plan.md`) is `src/nmr2selfies/` with modules: `settings.py`, `model.py`, `tokenizer.py`, `datamodule.py`, `lightning_module.py`, `analysis.py`. Currently `src/modules/` contains stubs (`encoder.py`, `decoder.py`, `datamodule.py`).

## Configuration

Three Pydantic Settings classes (once implemented):

- `AppSettings` — env-level: paths, W&B project, device. Loaded from env vars / `.env`.
- `ModelSettings` — architecture hyperparameters. Versioned in `.config/base.yaml`.
- `TrainSettings` — training hyperparameters. Versioned in `.config/base.yaml`.

The split allows env secrets to stay out of git while experiment settings are tracked.

## Data

- **Source**: LMDB at `/data` containing 1.7M molecules. Keys are molecule IDs; values contain peak arrays and SMILES strings. **Schema is not yet confirmed** — run `scripts/inspect_lmdb.py` (to be written) before writing any dataset code.
- **SELFIES conversion**: done in `__getitem__` for Phase 1 (simpler). Switch to a precomputed side-car LMDB if GPU utilization shows the dataloader is the bottleneck.
- **Splits**: 90/5/5 train/val/test by molecule ID hash, persisted as JSON.
- **Peak normalization**: `(mean, std)` computed once over the training split, hard-coded into config.
- **Tokenizer**: built once from the training split by streaming SMILES → SELFIES conversions. Persisted as `tokenizer.json`. Never rebuild without also invalidating any cached SELFIES data.

## Critical Implementation Notes

- **LMDB + multiprocessing**: do not open the LMDB in the parent process. Open it lazily inside the first `__getitem__` call to avoid corrupted file handles across DataLoader worker forks.
- **Padding mask on cross-attention**: padded peak positions must receive exactly zero attention weight. This is the most likely source of silent bugs — add explicit unit tests for it.
- **SMILES → SELFIES failures**: catch `selfies.EncoderError`, skip the record, and log the failure count. Warn if the rate exceeds 1% of records.
- **`selfies` version must be pinned** in `pixi.toml`. Different versions can produce different canonical SELFIES orderings, which would corrupt the tokenizer.
- **LMDB on cloud**: copy to local SSD at job startup; never stream over a network filesystem.
