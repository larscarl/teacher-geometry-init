# teacher-geometry-init

Distillation framework for teacher-guided student model initialization and training.

## Setup

1. Sync dependencies:

```bash
uv sync
```

2. Create `.env` from the template:

```bash
cp .env.example .env
```

Then update values as needed:

```bash
HF_TOKEN=hf_xxx
CODE_DIRECTORY=/cluster/path/to/teacher-geometry-init
```

Notes:
- `HF_TOKEN` is read from environment by `run_distill.py` and is also loaded from `.env`.
- `CODE_DIRECTORY` is used by `sbatchs/lmic_distill_hydra.submit` (for cluster runs).

## Registry Files

- Active registry (current runs): `experiments/registry_distill.csv`
- Deprecated historical registry (information only): `experiments/registry_distill_deprecated.csv`

## Smoke Test

This repository includes three smoke rows in `experiments/registry_distill.csv`:
- `entry_id=smoke_local_tiny_llama_random`
- `entry_id=smoke_local_tiny_llama_copy_subset`
- `entry_id=smoke_local_tiny_llama_pca_layerwise`
- `submit_via=python` (runs locally, not via `sbatch`)
- tiny Llama model + tiny dataset (`experiments/smoke_data/train.jsonl`)

Run it:

```bash
uv run python -m src.experiments.distill_registry validate --registry experiments/registry_distill.csv
uv run python -m src.experiments.distill_registry run --registry experiments/registry_distill.csv --enabled-only
```

Run only a specific smoke entry:

```bash
uv run python -m src.experiments.distill_registry run \
  --registry experiments/registry_distill.csv \
  --ids smoke_local_tiny_llama_pca_layerwise
```

Duplicate safeguard (default):
- `run` now fails if `run_artifact_id` was already used (detected via `experiments/submissions_distill.jsonl`, `experiments/results_distill.jsonl`, or an existing artifact directory).

Intentional rerun of the same config with separate folder:

```bash
uv run python -m src.experiments.distill_registry run \
  --registry experiments/registry_distill.csv \
  --enabled-only \
  --duplicate-policy=timestamp
```

This keeps the registry row stable but submits with a timestamp-suffixed `run_artifact_id` (for example `smoke-local-tiny-llama-random-20260324_164742`).

Artifacts are written to:
- `experiments/runs_distill/<run_artifact_id>`
- `experiments/results_distill.jsonl`

To increase verbosity for debugging, set `--log-level=DEBUG` in `extra_overrides` for the smoke row.
