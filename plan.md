# SMART: NMR-to-SELFIES — project plan

## Goal

Train a sequence model that, given an HSQC NMR spectrum (a variable-length set of `(δH, δC, intensity)` peaks), produces the SELFIES string of the molecule that generated it. The end goal is a deployable model that takes raw spectra and outputs valid molecular structures.

The hypothesis driving the architecture: individual peaks (or small groups of peaks) act as many-to-many indicators for the presence of specific substructures in the molecule. A peak at `δC ≈ 125, δH ≈ 7.2` indicates aromatic CH; a methyl group typically produces peaks near `δC ≈ 15–25, δH ≈ 0.9`. If this hypothesis holds, a transformer with cross-attention from output tokens to input peaks should learn to attend to chemically-relevant peaks when emitting tokens for the corresponding substructures.

## Phase 1 — go/no-go validation (current focus)

The first goal is **not** a production model — it is to confirm the peaks-to-substructures hypothesis is strong enough to be worth pursuing. The cheapest test is to train a small transformer end-to-end and inspect cross-attention weights:

1. Train a small encoder-decoder transformer to convergence (loss plateau, ≥90% SELFIES validity on held-out data).
2. On the validation set, capture decoder cross-attention weights for every emitted token.
3. For each token type, aggregate the attention-weighted distribution of attended peaks across all occurrences.
4. Check three conditions:
   - **Sharpness**: per-token attention entropy is meaningfully below uniform (model is selecting peaks, not averaging them).
   - **Consistency**: for a given token type, attended-peak `(δH, δC)` coordinates have low spread across molecules (model attends to similar peaks for similar substructures).
   - **Chemical plausibility**: the mean attended `(δH, δC)` for known substructure tokens (aromatic CH, methyl, carbonyl-adjacent carbons) lands in the chemical-shift regions chemists would expect.

If all three hold, proceed to Phase 2. If they don't, document the negative result and stop — don't sink more time into the approach.

> **Note on attention as evidence**: Attention weights are correlational, not causal. They are sufficient for go/no-go because if the model is *not* using peaks the way we hypothesize, attention will be diffuse or chemically nonsensical, which is enough to kill the project. If attention *does* look right, that's promising but not proof — Phase 2 includes proper causal validation.

## Phase 2 — chemistry-informed clustering and validation

If Phase 1 succeeds:

1. Research theoretical `(δH, δC)` regions for common substructures from NMR literature.
2. Build a constrained clustering of peaks based on those regions, and use cluster assignments as auxiliary multi-hot features to the encoder.
3. Run causal attribution (integrated gradients or peak-ablation studies) to confirm specific peaks drive specific tokens, not just correlate.
4. Compare model performance with and without the auxiliary cluster features.

## Phase 3 — production model

If the approach is validated:

1. Scale up model size and training duration.
2. Add SMILES output as an alternative head (some downstream tools prefer SMILES).
3. Beam search and constrained decoding for higher SELFIES validity.
4. Deployment as a containerized inference service.

## Architecture

A standard transformer encoder-decoder, with one important deviation from the textbook setup: **no positional encoding on the encoder**. NMR peaks are an unordered set, and any ordering in the data is arbitrary. Adding positional encodings would let the model latch onto spurious order-based shortcuts.

- **Encoder**: 4 layers, 8 heads, `d_model=256`, FFN inner dim 1024. Input is per-peak `(δH, δC, intensity)` projected through a small MLP. Self-attention is permutation-equivariant (no positional encoding).
- **Decoder**: 4 layers, 8 heads, `d_model=256`. Standard causal self-attention plus cross-attention to encoder output. Token embeddings include sinusoidal positional encoding.
- **Output head**: linear projection to SELFIES vocab size.
- **Total parameters**: ~12M, trainable on a single mid-range GPU in a few hours given 1.7M training molecules.

## Data

- **Source**: 1.7M molecules with paired HSQC spectra and SMILES strings, stored in an existing LMDB.
- **Storage layout (existing)**: keys are molecule IDs, values are serialized records containing the peak array and the SMILES string. The store does **not** contain SELFIES.
- **SMILES → SELFIES conversion**: handled by the `selfies` package. There are two reasonable places to do the conversion:
  1. **At training time, inside the dataset's `__getitem__`**. Simple to implement, no preprocessing step. Cost: ~34M conversions over a 20-epoch training run, plus the same again for any analysis run. With `num_workers ≥ 4` this is unlikely to bottleneck the GPU, but it does waste cycles across multiple runs.
  2. **Once, into a side-car LMDB or a new column**. A small `prepare_selfies.py` script reads each record, converts SMILES → SELFIES, writes the SELFIES into a parallel LMDB keyed by the same molecule ID. The training dataset reads from both stores. More upfront work, but training and analysis become CPU-cheap and reproducible.
  
  Recommendation: start with option 1 for Phase 1 (lower friction, the cost is real but manageable). Switch to option 2 if profiling shows the dataloader is GPU-starved, or before any large hyperparameter sweep where the conversion cost compounds.
- **Conversion failures**: not every SMILES converts cleanly to SELFIES — some valences or stereochemistry edge cases fail. The dataset must catch `selfies.EncoderError` (or whatever the package raises) and either skip the record or substitute a placeholder. Skipping is safer; log the count of skipped records on first epoch and confirm it's a small fraction of the dataset before trusting the run.
- **Peak normalization statistics**: compute `mean` and `std` for `(δH, δC, intensity)` once over the training split, hard-code into config. This is a one-time scan over the LMDB.
- **Splits**: 90% train, 5% val, 5% test by molecule ID. Stratify by molecular weight or scaffold if the dataset has class imbalance. If the existing LMDB doesn't already have splits, build a deterministic split list (hash on molecule ID, persist to JSON) and load it in the DataModule.
- **Tokenizer**: SELFIES atom-level tokenization. Build the vocab by streaming through the training-split SMILES, converting each to SELFIES, and accumulating the unique tokens. Persist as JSON. The vocab build is the **only** place a full pass over the dataset is required; cache the result.

## Tech stack

- **Compute**: cloud GPUs. Architecture targets a single GPU per run for Phase 1, with multi-GPU support deferred until needed.
- **PyTorch + Lightning**: pure-PyTorch model code, Lightning for training loop, callbacks, and distributed training. Keep the model framework-free so it can be used outside Lightning at inference time.
- **LightningDataModule**: wraps the LMDB dataset, tokenizer, and DataLoaders.
- **Pydantic Settings**: configuration management. All config (model hyperparameters, paths, training settings, W&B project names) flows through pydantic-settings classes, with environment-variable and `.env` overrides for cloud deployment.
- **Weights & Biases**: experiment tracking. Logs training/validation losses, learning rate, SELFIES validity, and the attention-analysis tables and scatter plots in Phase 1.
- **Pixi**: environment management. All dependencies (Python version, PyTorch+CUDA, RDKit, etc.) pinned in `pixi.toml`, reproducible across local dev and cloud workers.
- **RDKit**: SELFIES validity checking and substructure matching for the attention analysis (mapping tokens back to chemical groups).

## Configuration approach

Pydantic Settings is the source of truth for all configuration. Three concerns are kept separate:

1. **`AppSettings`** — environment-level config: paths, cloud credentials, W&B project, device. Loaded from environment variables and `.env` files. This is what changes between local and cloud runs.
2. **`ModelSettings`** — model architecture: dimensions, layer counts, dropout. Stable across runs of the same experiment.
3. **`TrainSettings`** — training hyperparameters: learning rate, batch size, schedule, epochs.

A top-level `Config` composes the three. Lightning's CLI hooks read from this; any field can be overridden on the command line for sweeps.

The reason for splitting: env-level settings are secrets-adjacent and should not be checked into git, but model and training settings are experiment artifacts that should be versioned. Splitting them lets the env layer load from `.env` while the experiment layer loads from a YAML file in the repo.

## Project structure

```
nmr2selfies/
├── pixi.toml                    # env + task definitions
├── .env.example                 # template for env-level secrets/paths
├── plan.md                      # this file
├── configs/
│   └── base.yaml                # model + training settings
├── scripts/
│   ├── inspect_lmdb.py          # one-time schema discovery
│   ├── make_splits.py           # train/val/test split generation
│   ├── compute_peak_stats.py    # peak normalization stats
│   ├── build_tokenizer.py       # one-time vocab construction
│   ├── prepare_selfies.py       # optional: precompute SELFIES into side-car LMDB
│   ├── train.py                 # Lightning entry point
│   └── analyze_attention.py     # post-training analysis
└── src/nmr2selfies/
    ├── __init__.py
    ├── settings.py              # pydantic-settings config classes
    ├── model.py                 # pure pytorch transformer
    ├── tokenizer.py             # SELFIES tokenizer
    ├── datamodule.py            # LightningDataModule + LMDB dataset
    ├── lightning_module.py      # training/eval loop
    └── analysis.py              # cross-attention extraction
```

## Implementation steps

The order is chosen so that each step produces something testable before moving on. Don't proceed to step N+1 until step N is verified.

1. **Environment** — write `pixi.toml` with all dependencies pinned (including `lmdb` and `selfies`). Create `.env.example`. Confirm the project installs cleanly on a fresh machine.
2. **Settings** — write the pydantic-settings classes. Verify env-var and YAML loading both work, including overrides.
3. **LMDB inspection** — write a small script that opens the existing LMDB read-only and prints schema, key format, value serialization, and record counts. Don't write any model code until this is understood — guessing at the LMDB schema is the most likely source of silent dataloader bugs.
4. **Splits + normalization** — generate the train/val/test split list (deterministic, hashed on molecule ID), persist as JSON. Compute peak `(mean, std)` over the training split, write into the config.
5. **Tokenizer** — stream through the training split, convert SMILES → SELFIES on the fly, build the vocab. Inspect for surprises (very rare tokens, unexpected `[UNK]` candidates, conversion failure rate).
6. **Pure model** — implement `model.py`. Run the smoke test from the previous skeleton: synthetic batch, verify shapes, verify masking behavior on padded positions.
7. **Dataset + DataModule** — implement the LMDB-backed dataset that reads `(peaks, smiles)` and converts to SELFIES in `__getitem__`. Verify a single batch loads correctly: shapes match, padding masks are correct, conversion failures are handled gracefully.
8. **LightningModule** — implement `lightning_module.py`. Run the **overfit** task on 4 batches; if loss does not go to near-zero, there's a bug.
9. **Full training run** — kick off training with W&B logging. Watch the first epoch for loss going down and validity rate going up. Don't leave it unattended on the first run; a misconfigured scheduler or data bug is far cheaper to catch in the first 30 minutes than after 12 hours.
10. **Profile the dataloader** — once training is stable, check GPU utilization. If GPU is idle waiting for batches, the SMILES→SELFIES conversion is the most likely culprit; precompute SELFIES into a side-car LMDB (option 2 in the data section) before any larger sweep.
11. **Attention analysis** — once a checkpoint converges, run `analyze_attention.py`. Inspect the W&B table and scatter plot. Make the go/no-go call.

## Risks and mitigations

- **Bug in attention masking** silently invalidates the entire analysis. Mitigation: explicit unit tests on the masking — for a batch with padded peaks, verify cross-attention weights to padded positions are exactly zero.
- **Padding-mask bug** masquerades as model learning. Mitigation: the smoke test in step 6 explicitly checks that padded peaks receive zero attention.
- **LMDB schema assumptions are wrong**. Mitigation: step 3 forces explicit schema discovery before any model code is written. If the LMDB stores peaks in a format the dataset code doesn't expect, every batch is silently corrupted.
- **SMILES → SELFIES conversion failures** — some records won't convert. If they're skipped silently and silently dominate one split, training metrics become misleading. Mitigation: log conversion failure counts on first epoch and emit a warning if the rate exceeds 1% of records.
- **Conversion non-determinism** — the `selfies` library can return different canonical orderings for equivalent SMILES across versions. Mitigation: pin the `selfies` version in `pixi.toml`, and if the side-car LMDB approach is taken, never rebuild it without invalidating the tokenizer too.
- **Tokenizer drift** — building the tokenizer at training time and rebuilding at analysis time produces different vocab IDs. Mitigation: tokenizer is built once, persisted, and loaded read-only thereafter. Vocab is part of every checkpoint's hyperparameters.
- **Confusing correlation with causation** in attention analysis. Mitigation: Phase 1 explicitly treats attention as preliminary evidence; Phase 2 includes ablation studies for causal confirmation.
- **LMDB on cloud storage** can be very slow if accessed over a network filesystem. Mitigation: copy the LMDB file to local SSD on each cloud worker at job startup; do not stream over network FS.
- **LMDB readers and multiprocessing** — LMDB environments cannot be opened in the parent process and inherited across `DataLoader` workers; the file handles get corrupted. Mitigation: open the LMDB lazily inside the worker (in `Dataset.__init__` won't work if the dataset is constructed before workers fork; do it in the first `__getitem__` call or use a worker init function).
- **Cost overruns on cloud GPUs** from leaving training jobs running unattended. Mitigation: W&B alerts on stalled runs, and a hard `max_epochs` cap with early stopping enabled by default.

## Success criteria

Phase 1 is successful if:

- The model achieves ≥90% SELFIES validity on held-out data.
- For at least three substructure classes (suggested: aromatic CH, aliphatic methyl, carbonyl carbons), the cross-attention analysis shows mean attended `(δH, δC)` within the chemically expected region.
- Per-token attention entropy is below 70% of the entropy of a uniform distribution over peaks (rough sharpness threshold; tune based on actual peak counts).
- Different substructure classes attend to clearly distinguishable peak regions (visual separation in the `(δH, δC)` scatter plot).

Phase 1 is a clean negative if:

- The model fails to reach 80% validity even with a larger architecture (suggesting peaks alone don't carry enough information).
- Attention is diffuse across all tokens (suggesting the model isn't using peak structure).
- Attention is sharp but chemically nonsensical (suggesting the model is exploiting dataset artifacts rather than the hypothesized peak-substructure relationship).

In the negative case, document the negative result, archive the run, and reconsider the approach before sinking more time.

## Open questions

- What's the schema of the existing LMDB? Specifically: key format (string IDs? integer IDs?), value serialization (pickle? msgpack? raw numpy?), peak array shape and dtype. Step 3 of the implementation steps answers this.
- What's the SMILES → SELFIES conversion failure rate on this dataset? If it's >1%, investigate before training; could indicate stereochemistry edge cases or invalid SMILES that need cleaning.
- Is intensity calibrated across the dataset, or does it need per-spectrum normalization? If intensities aren't comparable across molecules, the model is effectively learning from `(δH, δC)` alone and intensity is noise.
- Are there molecules with peaks at chemical shifts outside the typical HSQC window? They may need filtering or wider normalization bounds.
- What's the distribution of peak counts per molecule? This drives the padding overhead and informs whether to bucket batches by sequence length.
