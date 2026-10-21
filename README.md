# Aerohash

Weather time-series hashing and retrieval: encode weather windows into binary codes and retrieve similar windows by Hamming distance.

The model uses time-domain/frequency-domain encoders, gated fusion, a Transformer encoder, and **GAP + MLP**. Training uses ADSH, with optional center, TFC, ordinal, and bit-regularization losses.

## Installation

Requires Python 3.10+ and a PyTorch build compatible with your hardware.

```bash
pip install -e .
```

## Training and Evaluation

The default data directory is `dataset/weather/`, containing window data `X_*.npy`, labels `Y_*.npy`, similarity matrices, and metadata such as `config.json`. Training uses `train_core`, and model selection uses an independent validation set.

```bash
# Run when a validation split is needed
aerohash build-val --stations weather

# Uses GPU by default; writes outputs to checkpoints/<run>/
aerohash train --config configs/default.yaml

# Standalone evaluation; writes results to a separate directory
aerohash evaluate --ckpt checkpoints/<run>/model.t --outdir results/eval
```

The main outputs are the model `model.t`, configuration `resolved_config.yaml`, evaluation metrics, and query/database hash codes. Evaluation reports include weather, clear, and overall mAP, along with code-space statistics.

## Configuration

See [configs/default.yaml](configs/default.yaml) for all parameters. Configuration precedence: command line > environment variables > specified YAML > defaults.

```bash
# Multi-station training
aerohash train --stations weather weather_ZBAD

# Set code length and random seed
aerohash train --code-length 32 --seed 42

# Disable the TFC auxiliary loss
aerohash train --loss.use-tfc false
```

Training and inference use the same code length. The frequency-domain encoder and TFC loss can be controlled independently.

## Experiments and Development

```bash
# Single-seed / multi-seed experiments
SEEDS="42" bash scripts/run_multiseed.sh
bash scripts/run_multiseed.sh

# Install test dependencies and run tests
pip install -e '.[dev]'
pytest -q
```

Source code is in `src/aerohash/`, tests in `tests/`, experiment scripts in `scripts/`, and baselines in `baseline/`.

See `aerohash --help` and `aerohash <command> --help` for data cleaning, preprocessing, and other commands.
