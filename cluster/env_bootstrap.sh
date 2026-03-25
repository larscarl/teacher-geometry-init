#!/usr/bin/env bash
set -euo pipefail

# Pick a venv name and path; allow overrides from environment.
# On cluster we default to scratch-backed envs to avoid home quota issues.
VENV="${VENV:-teacher_geometry_init_3_12_4}"
export VENV

# Work around occasional XET-related HF download issues on cluster.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

module purge || true
module load python/3.12.4

# Load uv from module environment.
VENV="$VENV" module load uv/0.6.12

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not on PATH after 'module load uv/0.6.12'." >&2
  exit 1
fi

# Resolve project root; caller can pass CODE_DIRECTORY via .env
CODE_DIRECTORY="${CODE_DIRECTORY:-$(pwd)}"
cd "${CODE_DIRECTORY}"

# Use uv-managed project environment path.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/venvs/${VENV}}"
if [[ "${UV_PROJECT_ENVIRONMENT}" = /* ]]; then
  VENV_PATH="${UV_PROJECT_ENVIRONMENT}"
else
  VENV_PATH="${CODE_DIRECTORY}/${UV_PROJECT_ENVIRONMENT}"
fi

# Install exactly what the project specifies.
# Prefer lockfile-reproducible installs when uv.lock is present.
if [[ -f "${CODE_DIRECTORY}/uv.lock" ]]; then
  uv sync --frozen
else
  uv sync
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ERROR: Expected venv activation script not found at ${VENV_PATH}/bin/activate" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

python --version
which uv || true

# Project paths & caches
export HF_HOME="${HF_HOME:-/scratch/hf_home}"
export WANDB_DIR="${WANDB_DIR:-/scratch/wandb}"
mkdir -p "${HF_HOME}" "${WANDB_DIR}"
export PYTHONPATH="${CODE_DIRECTORY}/src:${CODE_DIRECTORY}:${PYTHONPATH:-}"

# Freeze for debugging (Python code will also write a freeze into the Hydra run dir)
uv pip freeze > /tmp/pip_freeze_env_bootstrap.txt || true
