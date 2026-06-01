# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Neural network guitar amplifier simulation. A causal TCN (Temporal Convolutional Network) learns to map a dry DI guitar signal (`data/input.wav`) to an amp-recorded signal (`data/output.wav`). Training data is stored in git LFS.

## Commands

```bash
# Setup
make install          # uv sync + install pre-commit hooks

# Train locally
uv run python src/train.py

# Train on HPC (SLURM)
make train_hpc        # sbatch job.sh

# Lint / format
uv run ruff check --fix src/
uv run ruff format src/
```

Pre-commit runs `ruff check --fix` and `ruff format` automatically on commit.

There are no tests.

## Architecture

All source lives in `src/`. Scripts import each other directly (no package install); run everything with `uv run python src/train.py` from the project root.

**Data flow:**
`data/input.wav` + `data/output.wav` → `LongAudioDataset` (randomly sampled chunks) → `AmpTCN` → `CombinedLoss` → saved to `runs/<timestamp>_<EXP_NAME>/`

**Model (`src/model.py`):**  
`AmpTCN` — input projection (1→channels), N stacks × dilation series of `ResidualBlock`s (causal dilated conv + PReLU + 1×1 mix + residual skip), output projection (channels→1). Controlled by `CHANNELS`, `DILATIONS`, `STACKS` in config.

**Loss (`src/loss.py`):**  
`CombinedLoss` = `0.8 × ESR + 0.2 × MR-STFT`. ESR (Error-to-Signal Ratio) is the primary time-domain metric; MR-STFT via `auraloss` adds multi-resolution spectral supervision.

**Config (`src/config.py`):**  
Single file for all hyperparameters and paths. To run a new experiment, edit `EXP_NAME` and any hyperparameters here — a new timestamped run directory is created automatically on import.

**Run outputs (`runs/<name>/`):**  
`best_model.pt`, `last_model.pt`, `metrics.csv`, `training_curve.png`

## HPC notes

`job.sh` targets an A100-40G GPU on a SLURM cluster (`ealloc_ati-1-neur` account). It expects the repo cloned to `~/amp_sim/` on the remote. Logs go to `logs/slurm-<jobid>.out`.



claude --resume ea75a121-d12b-4f22-90a3-51db763ee47e
