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
WANDB_API_KEY=wandb_xxx
WANDB_PROJECT=teacher-geometry-init
WANDB_ENTITY=
WANDB_ENABLED=false
CODE_DIRECTORY=/cluster/path/to/teacher-geometry-init
```

Notes:
- `HF_TOKEN` is read from environment by `run_distill.py` and is also loaded from `.env`.
- `CODE_DIRECTORY` is used by `sbatchs/lmic_distill_hydra.submit` (for cluster runs).
- W&B logging is opt-in. Enable it with `WANDB_ENABLED=true` or `wandb.enabled=true` in overrides.
- Cluster bootstrap (`cluster/env_bootstrap.sh`) installs from `pyproject.toml`/`uv.lock` via `uv sync` (no manual `uv pip install` needed).

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
- `src.distillation.run_distill` now also fails fast when `run.mode=new` and the target `run_dir` is non-empty, unless `run.force_clear=true` is explicitly set.

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

## Logging

Shell output:
- `INFO`/`DEBUG` logs (from `run_distill.py`) go to stdout.
- `WARNING`+ logs go to stderr.
- Some modules still use `print()` (student init + KLD trainer), so those appear on stdout.

Files:
- `experiments/runs_distill/<run_artifact_id>/run_distill.log` (full DEBUG log).
- `experiments/runs_distill/<run_artifact_id>/RUN_META.json` (resolved overrides).
- `experiments/runs_distill/<run_artifact_id>/DISTILL_METRICS.json` (train/eval metrics).
- `experiments/results_distill.jsonl` (one-line summary per run).
- `experiments/submissions_distill.jsonl` (registry submissions).

## Weights & Biases

1. Create a project at https://wandb.ai/ and generate an API key.
2. Set `WANDB_API_KEY` in `.env` and optionally `WANDB_PROJECT` / `WANDB_ENTITY`.
3. Enable logging by setting `WANDB_ENABLED=true` in `.env` or by adding `wandb.enabled=true` in your registry row overrides.

When enabled, `run_distill.py` logs training metrics and lm-eval metrics to W&B.

## lm-eval Benchmarks

`run_distill.py` can run lm-eval benchmarks during training and after training:

Example overrides (add to `extra_overrides`):
- `lm_eval.enabled=true`
- `lm_eval.tasks=hellaswag,arc_easy`
- `lm_eval.device=cuda:0`
- `lm_eval.batch_size=auto`
- `lm_eval.limit=` (empty means full)

During training (evaluated on each checkpoint save):
- `lm_eval.train.enabled=true`
- `lm_eval.train.limit=64` (default if unset)

Results are appended to:
- `experiments/runs_distill/<run_artifact_id>/LM_EVAL_RESULTS.jsonl`

Security note:
- `LM_EVAL_RESULTS.jsonl` is redacted before write. Sensitive keys such as `token`, `hf_token`, `api_key`, `access_token`, and `authorization` are stored as `***REDACTED***`.

W&B note:
- lm-eval metrics are logged under `lm_eval/<phase>/<task>/<metric>`.
- `phase` is either `train` (checkpoint-time eval) or `post_train` (after final save).
- `lm_eval/<phase>/global_step` is also logged for easier charting.
- For apples-to-apples trend comparisons across checkpoints, keep `lm_eval.tasks`, `lm_eval.limit`, and `lm_eval.num_fewshot` fixed.

Post-hoc evaluation (base/teacher/student models):

```bash
uv run python -m src.experiments.lm_eval_runner \
  --models base=gpt2,teacher=hf-internal-testing/tiny-random-LlamaForCausalLM,student=/path/to/run_dir \
  --tasks hellaswag,arc_easy \
  --device cuda:0 \
  --batch-size auto \
  --limit 64
```

Model/task timing matrix (smoke + full suite):

```bash
uv run python -m src.experiments.lm_eval_suite_runner \
  --models gemma3_270m=google/gemma-3-270m,gemma3_1b_pt=google/gemma-3-1b-pt,apertus_8b=swiss-ai/Apertus-8B-2509,llama32_1b=meta-llama/Llama-3.2-1B \
  --tasks mmlu_stem,hellaswag,arc_challenge,piqa,winogrande,boolq,gsm8k_cot_llama \
  --device cpu \
  --batch-size auto \
  --smoke \
  --smoke-limit 2 \
  --suite-id local_smoke_model_matrix
```

For full-dataset runtime measurement (for cluster planning):

```bash
uv run python -m src.experiments.lm_eval_suite_runner \
  --models gemma3_270m=google/gemma-3-270m,gemma3_1b_pt=google/gemma-3-1b-pt,apertus_8b=swiss-ai/Apertus-8B-2509,llama32_1b=meta-llama/Llama-3.2-1B \
  --tasks mmlu_stem,hellaswag,arc_challenge,piqa,winogrande,boolq,gsm8k_cot_llama \
  --device cuda:0 \
  --batch-size auto \
  --suite-id cluster_full_model_matrix
```

Submit the same full suite via Slurm:

```bash
sbatch --time=24:00:00 --nodes=1 --ntasks=1 --mem=80G --partition=gpu_top --gres=gpu:1 --cpus-per-task=8 \
  --job-name=lm_eval_full_matrix \
  sbatchs/lmic_lm_eval_suite.submit \
  --models gemma3_270m=google/gemma-3-270m,gemma3_1b_pt=google/gemma-3-1b-pt,apertus_8b=swiss-ai/Apertus-8B-2509,llama32_1b=meta-llama/Llama-3.2-1B \
  --tasks mmlu_stem,hellaswag,arc_challenge,piqa,winogrande,boolq,gsm8k_cot_llama \
  --device cuda:0 \
  --batch-size auto \
  --suite-id cluster_full_model_matrix
```

Outputs:
- `experiments/lm_eval_suite_results.jsonl` (detailed per model/task results, redacted).
- `experiments/lm_eval_suite_timing.csv` (per-task runtime table with `duration_s`, sample counts, and primary metric).
- stdout JSON lines include per-task rows plus `summary_type=model_total` and `summary_type=suite_total`.

## Meaningful Local Benchmark Smoke

The tiny 4-sample smoke set is useful for plumbing checks, but benchmark scores often do not move in a meaningful way.  
For a stronger local signal, use a larger repeated local dataset and `wikitext` benchmark eval:

```bash
mkdir -p experiments/smoke_data_medium
: > experiments/smoke_data_medium/train.jsonl
: > experiments/smoke_data_medium/eval.jsonl
for i in {1..128}; do cat experiments/smoke_data/train.jsonl >> experiments/smoke_data_medium/train.jsonl; done
for i in {1..32}; do cat experiments/smoke_data/train.jsonl >> experiments/smoke_data_medium/eval.jsonl; done
```

```bash
uv run python -m src.distillation.run_distill \
  run_artifact_id=smoke-lmeval-local_v5 \
  distillation.teacher_model=sshleifer/tiny-gpt2 \
  distillation.student.init_strategy=copy_subset \
  distillation.student.hidden_size=2 \
  distillation.loss=cross_entropy \
  data.input_path=smoke_data_medium \
  data.prepared_dir=experiments/smoke_data_medium \
  data.prepared_split=train \
  data.eval_prepared_dir=experiments/smoke_data_medium \
  data.eval_prepared_split=eval \
  tokenizer.chunk_length=128 \
  model.train_args.num_train_epochs=1 \
  model.train_args.per_device_train_batch_size=8 \
  model.train_args.per_device_eval_batch_size=8 \
  model.train_args.gradient_accumulation_steps=1 \
  model.train_args.learning_rate=5e-5 \
  model.train_args.logging_steps=2 \
  model.train_args.save_strategy=steps \
  model.train_args.save_steps=8 \
  model.train_args.save_total_limit=4 \
  model.train_args.seed=42 \
  distillation.max_train_samples=256 \
  distillation.max_eval_samples=64 \
  lm_eval.enabled=true \
  lm_eval.tasks=wikitext \
  lm_eval.limit=32 \
  lm_eval.device=cpu \
  lm_eval.batch_size=auto \
  lm_eval.train.enabled=true \
  lm_eval.train.limit=16 \
  --log-level=INFO
```
