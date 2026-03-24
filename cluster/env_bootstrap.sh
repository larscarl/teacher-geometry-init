#!/usr/bin/env bash
set -euo pipefail

# Pick a venv name; allow override from environment
VENV="${VENV:-llm_compression_3_12_4}"
export VENV

# trying to debug fine-tuning on sanjose...
export HF_HUB_DISABLE_XET=1

module purge || true
module load python/3.12.4

# Create/activate uv venv (same name you used before)
VENV="$VENV" module load uv/0.6.12

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not on PATH after 'module load uv/0.6.12'." >&2
  exit 1
fi

# Create venv if missing (safe to run repeatedly)
uv venv
source "/scratch/venvs/${VENV}/bin/activate"

python --version
which uv || true

# Packages (same as your .submit)
uv pip install "torch==2.8.0" "torchvision==0.23.0"
uv pip install "transformers>=4.50.0" "accelerate>=0.34" sentencepiece wandb

uv pip install hydra-submitit-launcher # deprecated

# Project paths & caches
export HF_HOME="/scratch/hf_home/"
export WANDB_DIR="/scratch/wandb/"
# CODE_DIRECTORY is expected from .env or caller; fallback to current dir.
export CODE_DIRECTORY="${CODE_DIRECTORY:-$(pwd)}"
export PYTHONPATH="${CODE_DIRECTORY}/src:${CODE_DIRECTORY}:${PYTHONPATH:-}"

# Freeze for debugging (Python code will also write a freeze into the Hydra run dir)
uv pip freeze > /tmp/pip_freeze_env_bootstrap.txt || true
