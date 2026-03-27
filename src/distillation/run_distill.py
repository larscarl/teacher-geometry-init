import argparse
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from src.distillation.kld_loss_trainer import KLDistillationTrainer
from src.distillation.student_factory import create_student_model
from src.utils.helper import is_rank0, setup_logging
from src.utils.lm_eval_utils import (
    append_jsonl as append_jsonl_safe,
    flatten_lm_eval_scalars,
    make_json_safe,
    parse_batch_size,
    parse_limit,
    parse_task_list,
    redact_lm_eval_result,
    run_lm_eval,
)


DEFAULT_RESULTS_PATH = "experiments/results_distill.jsonl"
DEFAULT_RUN_DIR_BASE = "experiments/runs_distill"
LM_EVAL_RESULTS_FILENAME = "LM_EVAL_RESULTS.jsonl"


def _is_iterable_dataset(dataset: object) -> bool:
    return isinstance(dataset, IterableDataset)


def _dataset_len(dataset: object) -> Optional[int]:
    if dataset is None or _is_iterable_dataset(dataset):
        return None
    if hasattr(dataset, "__len__"):
        try:
            return int(len(dataset))
        except Exception:
            return None
    return None


def _dataset_size_str(dataset: object) -> str:
    n = _dataset_len(dataset)
    if n is None:
        return "streaming/unknown"
    return str(n)


def _load_env_file(path: Path, *, override: bool = False) -> None:
    """
    Lightweight .env loader (no external dependency).
    Supports lines like:
      KEY=value
      export KEY=value
      KEY="value"
    Existing environment variables are preserved unless override=True.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue

        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()

        if override or key not in os.environ:
            os.environ[key] = value


def _strip(value: Optional[object]) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_slug(value: str, fallback: str = "distill-run") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", _strip(value))
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = fallback
    return text


def _parse_int(value: Optional[str], default: int) -> int:
    text = _strip(value)
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _parse_float(value: Optional[str], default: float) -> float:
    text = _strip(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_bool(value: Optional[str], default: bool) -> bool:
    text = _strip(value).lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    text = _strip(value)
    if not text:
        return None
    return _parse_bool(text, default=False)


def _parse_overrides(tokens: List[str]) -> Dict[str, str]:
    """
    Parse Hydra-like CLI overrides passed as plain tokens, e.g.:
      run.exp_name=my-run distillation.loss=kld +foo=bar
    """
    out: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.lstrip("+").strip()
        if not key:
            continue
        out[key] = value.strip()
    return out


def _get_first(overrides: Dict[str, str], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = _strip(overrides.get(key))
        if value:
            return value
    return default


def _parse_list(value: Optional[str]) -> List[str]:
    return parse_task_list(value or "")


def _resolve_wandb_settings(
    overrides: Dict[str, str],
    *,
    run_name: str,
    run_dir: Path,
) -> Dict[str, object]:
    enabled_raw = _get_first(
        overrides,
        ["wandb.enabled", "logging.wandb", "wandb", "log.wandb"],
        default="",
    )
    enabled_env = os.environ.get("WANDB_ENABLED", "")
    enabled = _parse_bool(enabled_raw, _parse_bool(enabled_env, False))

    project = _get_first(
        overrides,
        ["wandb.project", "wandb_project", "logging.wandb_project"],
        default=os.environ.get("WANDB_PROJECT", "teacher-geometry-init"),
    )
    entity = _get_first(
        overrides,
        ["wandb.entity", "wandb_entity", "logging.wandb_entity"],
        default=os.environ.get("WANDB_ENTITY", ""),
    )
    group = _get_first(
        overrides,
        ["wandb.group", "wandb_group"],
        default=os.environ.get("WANDB_GROUP", ""),
    )
    tags = _parse_list(
        _get_first(
            overrides,
            ["wandb.tags", "wandb_tags"],
            default=os.environ.get("WANDB_TAGS", ""),
        )
    )
    mode = _get_first(
        overrides,
        ["wandb.mode", "wandb_mode"],
        default=os.environ.get("WANDB_MODE", ""),
    )
    wandb_dir = _get_first(
        overrides,
        ["wandb.dir", "wandb_dir"],
        default=os.environ.get("WANDB_DIR", ""),
    )

    name = _get_first(
        overrides,
        ["wandb.name", "wandb_name"],
        default=run_name,
    )

    return {
        "enabled": enabled,
        "project": project,
        "entity": entity,
        "group": group,
        "tags": tags,
        "mode": mode,
        "dir": wandb_dir or str(run_dir),
        "name": name,
    }


def _maybe_init_wandb(
    settings: Dict[str, object],
    *,
    overrides: Dict[str, str],
    log: logging.Logger,
) -> Optional[object]:
    if not settings.get("enabled"):
        return None
    if not is_rank0():
        return None
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is enabled but not installed. Add it to dependencies.")

    import wandb

    mode = str(settings.get("mode") or "").strip()
    if mode:
        os.environ["WANDB_MODE"] = mode

    wandb_dir = str(settings.get("dir") or "").strip()
    if wandb_dir:
        os.environ["WANDB_DIR"] = wandb_dir

    run = wandb.init(
        project=str(settings.get("project") or "teacher-geometry-init"),
        entity=str(settings.get("entity") or "") or None,
        group=str(settings.get("group") or "") or None,
        name=str(settings.get("name") or "") or None,
        tags=list(settings.get("tags") or []),
        config={"overrides": overrides},
    )
    log.info("Weights & Biases run initialized: %s", getattr(run, "name", "unknown"))
    return run


def _resolve_lm_eval_settings(
    overrides: Dict[str, str],
    *,
    run_dir: Path,
) -> Dict[str, object]:
    enabled = _parse_bool(
        _get_first(overrides, ["lm_eval.enabled", "lm_eval.enable"], default=""),
        False,
    )
    tasks = _parse_list(_get_first(overrides, ["lm_eval.tasks"], default=""))

    train_enabled = _parse_bool(
        _get_first(
            overrides,
            ["lm_eval.train.enabled", "lm_eval.train_on_save", "lm_eval.train.on_save"],
            default="",
        ),
        False,
    )
    train_tasks = _parse_list(
        _get_first(overrides, ["lm_eval.train.tasks"], default="")
    )

    batch_size = parse_batch_size(_get_first(overrides, ["lm_eval.batch_size"], ""), default="auto")
    num_fewshot = _parse_int(_get_first(overrides, ["lm_eval.num_fewshot"], ""), 0)
    limit_raw = _get_first(overrides, ["lm_eval.limit"], "")
    limit = parse_limit(limit_raw, default=None)
    device = _get_first(overrides, ["lm_eval.device"], default="")
    log_samples = _parse_bool(_get_first(overrides, ["lm_eval.log_samples"], ""), False)
    apply_chat_template = _parse_optional_bool(
        _get_first(overrides, ["lm_eval.apply_chat_template"], "")
    )
    trust_remote_code = _parse_optional_bool(
        _get_first(overrides, ["lm_eval.trust_remote_code"], "")
    )

    train_batch_size = parse_batch_size(
        _get_first(overrides, ["lm_eval.train.batch_size"], ""),
        default=batch_size,
    )
    train_num_fewshot = _parse_int(
        _get_first(overrides, ["lm_eval.train.num_fewshot"], ""), num_fewshot
    )
    train_limit_raw = _get_first(overrides, ["lm_eval.train.limit"], "")
    train_limit = parse_limit(train_limit_raw, default=None)
    if train_enabled and not train_limit_raw:
        train_limit = 64
    train_device = _get_first(overrides, ["lm_eval.train.device"], default=device)

    results_path = _get_first(
        overrides, ["lm_eval.results_path"], default=str(run_dir / LM_EVAL_RESULTS_FILENAME)
    )

    return {
        "enabled": enabled,
        "tasks": tasks,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "device": device,
        "log_samples": log_samples,
        "apply_chat_template": apply_chat_template,
        "trust_remote_code": trust_remote_code,
        "train_enabled": train_enabled,
        "train_tasks": train_tasks,
        "train_batch_size": train_batch_size,
        "train_num_fewshot": train_num_fewshot,
        "train_limit": train_limit,
        "train_device": train_device,
        "results_path": results_path,
    }


def _run_lm_eval_and_log(
    *,
    log: logging.Logger,
    hf_token: str,
    model_ref: str,
    tasks: List[str],
    batch_size: Union[int, str],
    device: str,
    num_fewshot: int,
    limit: Optional[Union[int, float]],
    log_samples: bool,
    apply_chat_template: Optional[bool],
    trust_remote_code: Optional[bool],
    phase: str,
    results_path: Path,
    run_meta: Dict[str, object],
    wandb_run: Optional[object] = None,
    global_step: Optional[int] = None,
) -> Dict[str, Any]:
    if not tasks:
        raise ValueError("lm_eval tasks list is empty.")

    log.info(
        "lm_eval (%s): model=%s tasks=%s limit=%s batch_size=%s device=%s",
        phase,
        model_ref,
        ",".join(tasks),
        limit,
        batch_size,
        device or "auto",
    )
    results = run_lm_eval(
        model_ref=model_ref,
        tasks=tasks,
        batch_size=batch_size,
        device=device or "cpu",
        num_fewshot=num_fewshot,
        limit=limit,
        hf_token=hf_token,
        log_samples=log_samples,
        apply_chat_template=apply_chat_template,
        trust_remote_code=trust_remote_code,
    )

    redacted_results = redact_lm_eval_result(results)
    payload = {
        "created_at": _now_iso(),
        "phase": phase,
        "model_ref": model_ref,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "limit": limit,
        "log_samples": log_samples,
        "apply_chat_template": apply_chat_template,
        "trust_remote_code": trust_remote_code,
        "meta": run_meta,
        "results": redacted_results,
    }
    append_jsonl_safe(results_path, payload)

    if wandb_run is not None:
        metrics = flatten_lm_eval_scalars(results, prefix=f"lm_eval/{phase}")
        if metrics:
            import wandb

            if global_step is not None:
                metrics[f"lm_eval/{phase}/global_step"] = float(global_step)
            wandb.log(metrics)

    return results


def _append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(row, ensure_ascii=True) + "\n")


def _init_file_logger(log: logging.Logger, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(handler)


def _split_candidates(split: str) -> List[str]:
    split = _strip(split) or "train"
    out = [f"{split}.jsonl", f"{split}set.jsonl"]
    if split == "train":
        out.append("trainset.jsonl")
    return out


def _resolve_dir_payload_path(data_dir: Path, split: str) -> Optional[Path]:
    for name in _split_candidates(split):
        candidate = data_dir / name
        if candidate.exists():
            return candidate

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if len(jsonl_files) == 1:
        return jsonl_files[0]

    return None


def _prepare_hf_dataset_kwargs(
    *,
    hf_path: str,
    hf_config: str,
    streaming: bool,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    path = _strip(hf_path)
    config = _strip(hf_config)
    if path:
        kwargs["path"] = path
    if config:
        kwargs["name"] = config
    kwargs["streaming"] = bool(streaming)
    return kwargs


def _load_dataset_from_path(path: Path, split: str) -> Dataset:
    split = _strip(split) or "train"

    if path.is_file():
        return load_dataset("json", data_files={split: str(path)}, split=split)

    if path.is_dir() and (path / "dataset_info.json").exists():
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            first_split = next(iter(ds.keys()))
            return ds[first_split]
        return ds

    if path.is_dir():
        payload = _resolve_dir_payload_path(path, split=split)
        if payload is not None:
            return load_dataset("json", data_files={split: str(payload)}, split=split)

    raise FileNotFoundError(f"Could not load dataset from path={path} split={split}")


def _load_raw_dataset(
    *,
    dataset_name: str,
    hf_path: str,
    hf_config: str,
    split: str,
    prepared_path: str,
    prepared_dir: str,
    streaming: bool,
) -> Union[Dataset, IterableDataset]:
    explicit_path = _strip(prepared_path)
    explicit_dir = _strip(prepared_dir)

    if explicit_path:
        return _load_dataset_from_path(Path(explicit_path), split=split)

    if explicit_dir:
        return _load_dataset_from_path(Path(explicit_dir), split=split)

    if streaming and (_strip(explicit_path) or _strip(explicit_dir)):
        raise ValueError(
            "Streaming mode does not support local prepared_path/prepared_dir inputs. "
            "Use Hugging Face dataset path/config instead."
        )

    hf_path = _strip(hf_path)
    hf_config = _strip(hf_config)
    if hf_path:
        kwargs = _prepare_hf_dataset_kwargs(
            hf_path=hf_path,
            hf_config=hf_config,
            streaming=streaming,
        )
        kwargs["split"] = _strip(split) or "train"
        return load_dataset(**kwargs)

    dataset_name = _strip(dataset_name)
    if dataset_name and Path(dataset_name).exists():
        return _load_dataset_from_path(Path(dataset_name), split=split)

    if dataset_name:
        kwargs = _prepare_hf_dataset_kwargs(
            hf_path=dataset_name,
            hf_config=hf_config,
            streaming=streaming,
        )
        kwargs["split"] = _strip(split) or "train"
        return load_dataset(**kwargs)

    raise ValueError(
        "No data source resolved. Set data.prepared_path or data.prepared_dir "
        "(recommended for VT2 protocol runs)."
    )


def _limit_raw_dataset(
    dataset: Union[Dataset, IterableDataset], *, max_samples: int
) -> Union[Dataset, IterableDataset]:
    if max_samples <= 0:
        return dataset
    if _is_iterable_dataset(dataset):
        return dataset.take(max_samples)
    return dataset.select(range(min(max_samples, len(dataset))))


def _tokenize_or_normalize_dataset(
    *,
    dataset: Union[Dataset, IterableDataset],
    tokenizer,
    max_seq_length: int,
) -> Union[Dataset, IterableDataset]:
    has_input_ids = "input_ids" in dataset.column_names

    if has_input_ids:
        remove_cols = [c for c in dataset.column_names if c != "input_ids"]

        def _normalize(example):
            ids = example.get("input_ids")
            if ids is None:
                ids = []
            if (
                tokenizer.bos_token_id is not None
                and len(ids) > 0
                and ids[0] == tokenizer.bos_token_id
            ):
                ids = ids[1:]
            if max_seq_length > 0:
                ids = ids[:max_seq_length]
            return {"input_ids": ids}

        tokenized = dataset.map(_normalize, remove_columns=remove_cols)
    elif "text" in dataset.column_names:
        remove_cols = list(dataset.column_names)

        def _tokenize(batch):
            kwargs = {
                "add_special_tokens": True,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }
            if max_seq_length > 0:
                kwargs.update({"truncation": True, "max_length": max_seq_length})
            enc = tokenizer(batch["text"], **kwargs)
            return {"input_ids": enc["input_ids"]}

        tokenized = dataset.map(_tokenize, batched=True, remove_columns=remove_cols)
    else:
        raise ValueError(
            "Dataset must expose either `input_ids` or `text` column. "
            f"Columns: {dataset.column_names}"
        )

    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 1)
    return tokenized


def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"

    return "cuda"


def _count_model_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _total_safetensors_mb(run_dir: Path) -> float:
    safetensors_files = sorted(run_dir.glob("*.safetensors"))
    if not safetensors_files:
        return -1.0
    total_bytes = sum(p.stat().st_size for p in safetensors_files)
    return total_bytes / (1024.0 * 1024.0)


def _resolve_attn_impl(model_name: str) -> str:
    return "eager" if "gemma-3" in model_name.lower() else "sdpa"


def _clear_dir_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass


class LmEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        log: logging.Logger,
        hf_token: str,
        tasks: List[str],
        batch_size: Union[int, str],
        device: str,
        num_fewshot: int,
        limit: Optional[Union[int, float]],
        log_samples: bool,
        apply_chat_template: Optional[bool],
        trust_remote_code: Optional[bool],
        results_path: Path,
        run_meta: Dict[str, object],
        wandb_run: Optional[object] = None,
    ) -> None:
        super().__init__()
        self._log = log
        self._hf_token = hf_token
        self._tasks = tasks
        self._batch_size = batch_size
        self._device = device
        self._num_fewshot = num_fewshot
        self._limit = limit
        self._log_samples = log_samples
        self._apply_chat_template = apply_chat_template
        self._trust_remote_code = trust_remote_code
        self._results_path = results_path
        self._run_meta = run_meta
        self._wandb_run = wandb_run
        self._last_step: Optional[int] = None

    def on_save(self, args, state, control, **kwargs):
        if not is_rank0():
            return control

        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return control
        if self._last_step == step:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.exists():
            self._log.warning(
                "lm_eval skipped: checkpoint directory missing at %s", checkpoint_dir
            )
            return control

        self._last_step = step
        try:
            _run_lm_eval_and_log(
                log=self._log,
                hf_token=self._hf_token,
                model_ref=str(checkpoint_dir),
                tasks=self._tasks,
                batch_size=self._batch_size,
                device=self._device,
                num_fewshot=self._num_fewshot,
                limit=self._limit,
                log_samples=self._log_samples,
                apply_chat_template=self._apply_chat_template,
                trust_remote_code=self._trust_remote_code,
                phase="train",
                results_path=self._results_path,
                run_meta=self._run_meta,
                wandb_run=self._wandb_run,
                global_step=step,
            )
        except Exception as exc:
            self._log.exception("lm_eval during training failed: %s", exc)

        return control


class _SamplerTrainerMixin:
    def __init__(self, *args, sampler_mode: str = "random", **kwargs):
        self.sampler_mode = sampler_mode
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not hasattr(train_dataset, "__len__"):
            return None

        mode = str(self.sampler_mode or "random").lower()
        world_size = getattr(self.args, "world_size", 1)
        rank = getattr(self.args, "process_index", 0)

        if mode == "sequential":
            if world_size > 1:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
            return SequentialSampler(train_dataset)

        if mode == "distributed":
            return DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )

        return RandomSampler(train_dataset)


class DistillCETrainer(_SamplerTrainerMixin, Trainer):
    pass


class DistillKLDTrainer(_SamplerTrainerMixin, KLDistillationTrainer):
    pass


def _build_result_row(
    *,
    overrides: Dict[str, str],
    status: str,
    submitted_at: str,
    finished_at: str,
    run_dir: Path,
    log_path: Path,
    error_note: str = "",
) -> Dict[str, object]:
    entry_id = _get_first(overrides, ["entry_id", "run.entry_id"], default=run_dir.name)
    run_artifact_id = _get_first(
        overrides,
        ["run_artifact_id", "run.artifact_id", "run.exp_name", "run.run_id"],
        default=run_dir.name,
    )
    dataset = _get_first(
        overrides,
        ["dataset", "data.input_path", "data.hf_path"],
        default="unknown",
    )
    split = _get_first(
        overrides,
        ["split", "data.prepared_split", "data.split"],
        default="unspecified",
    )
    seed = _parse_int(
        _get_first(
            overrides,
            ["seed", "run.seed", "model.train_args.seed", "train_seed"],
            default="",
        ),
        default=-1,
    )
    teacher_model = _get_first(
        overrides,
        ["teacher_model", "distillation.teacher_model"],
        default="unknown",
    )
    teacher_variant = _get_first(
        overrides,
        ["teacher_variant", "distillation.teacher_variant"],
        default="unspecified",
    )
    student_family = _get_first(
        overrides,
        ["student_family", "distillation.student_family"],
        default="unknown",
    )
    student_arch = _get_first(
        overrides,
        ["student_arch", "distillation.student_arch"],
        default="unknown",
    )
    student_target_params = _parse_int(
        _get_first(
            overrides,
            ["student_target_params", "distillation.student_target_params"],
            default="",
        ),
        default=-1,
    )
    student_actual_params = _parse_int(
        _get_first(overrides, ["student_actual_params"], default=""),
        default=-1,
    )
    student_safetensors_mb = _parse_float(
        _get_first(overrides, ["student_safetensors_mb"], default=""),
        default=-1.0,
    )
    distill_loss = _get_first(
        overrides, ["distill_loss", "distillation.loss"], default="unknown"
    )
    distill_method = _get_first(
        overrides, ["distill_method", "distillation.method"], default="unknown"
    )

    total_compression_rate = _parse_float(
        _get_first(overrides, ["total_compression_rate"], default=""),
        default=-1.0,
    )
    compression_speed_mb_s = _parse_float(
        _get_first(overrides, ["compression_speed_mb_s"], default=""),
        default=-1.0,
    )

    train_loss = _parse_float(
        _get_first(overrides, ["train_loss"], default=""), default=-1.0
    )
    eval_loss = _parse_float(
        _get_first(overrides, ["eval_loss"], default=""), default=-1.0
    )
    train_runtime_s = _parse_float(
        _get_first(overrides, ["train_runtime_s"], default=""),
        default=-1.0,
    )
    train_samples_per_second = _parse_float(
        _get_first(overrides, ["train_samples_per_second"], default=""),
        default=-1.0,
    )
    train_steps_per_second = _parse_float(
        _get_first(overrides, ["train_steps_per_second"], default=""),
        default=-1.0,
    )
    train_global_step = _parse_int(
        _get_first(overrides, ["train_global_step"], default=""),
        default=-1,
    )
    distill_temperature = _parse_float(
        _get_first(overrides, ["distillation.temperature"], default=""),
        default=-1.0,
    )
    distill_alpha = _parse_float(
        _get_first(overrides, ["distillation.alpha"], default=""),
        default=-1.0,
    )
    train_num_samples = _parse_int(
        _get_first(overrides, ["train_num_samples"], default=""),
        default=-1,
    )
    eval_num_samples = _parse_int(
        _get_first(overrides, ["eval_num_samples"], default=""),
        default=-1,
    )

    notes = _get_first(overrides, ["notes"], default="")
    if error_note:
        notes = f"{notes} | error={error_note}".strip(" |")

    row: Dict[str, object] = {
        "entry_id": entry_id,
        "run_artifact_id": run_artifact_id,
        "status": status,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "teacher_model": teacher_model,
        "teacher_variant": teacher_variant,
        "student_family": student_family,
        "student_arch": student_arch,
        "student_target_params": student_target_params,
        "student_actual_params": student_actual_params,
        "student_safetensors_mb": student_safetensors_mb,
        "distill_loss": distill_loss,
        "distill_method": distill_method,
        "total_compression_rate": total_compression_rate,
        "compression_speed_mb_s": compression_speed_mb_s,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_runtime_s": train_runtime_s,
        "train_samples_per_second": train_samples_per_second,
        "train_steps_per_second": train_steps_per_second,
        "train_global_step": train_global_step,
        "distill_temperature": distill_temperature,
        "distill_alpha": distill_alpha,
        "train_num_samples": train_num_samples,
        "eval_num_samples": eval_num_samples,
        "submitted_at": submitted_at,
        "finished_at": finished_at,
        "artifact_path": str(run_dir.resolve()),
        "log_path": str(log_path.resolve()),
    }

    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if node:
        row["node"] = node
    if gpu:
        row["gpu"] = gpu
    if notes:
        row["notes"] = notes

    return row


def _write_run_meta(
    run_dir: Path,
    overrides: Dict[str, str],
    payload: Dict[str, object],
    filename: str,
) -> None:
    (run_dir / filename).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _build_student_init_cfg(overrides: Dict[str, str]) -> Optional[dict]:
    cache_dir = _get_first(
        overrides,
        [
            "distillation.student.init_cache_dir",
            "distillation.student.init.cache_dir",
            "distillation.student.umap_cache_dir",
            "distillation.student.cache_dir",
            "distillation.umap.cache_dir",
        ],
        default="",
    )
    if not cache_dir:
        return None

    dataset_name = _get_first(
        overrides,
        ["data.input_path", "dataset", "data.hf_path"],
        default="wikitext",
    )
    dataset_path = _get_first(
        overrides,
        [
            "distillation.student.dataset_path",
            "data.prepared_path",
            "data.prepared_dir",
        ],
        default=dataset_name,
    )
    hf_path = _get_first(overrides, ["data.hf_path"], default=dataset_name)
    hf_config = _get_first(overrides, ["data.hf_config"], default="")
    hf_split = _get_first(
        overrides,
        ["data.hf_train_split", "data.prepared_split", "data.split", "split"],
        default="train",
    )
    streaming = _parse_bool(
        _get_first(overrides, ["data.streaming"], default=""),
        default=False,
    )

    cfg = {
        "cache_dir": cache_dir,
        "num_samples": _parse_int(
            _get_first(
                overrides,
                [
                    "distillation.student.init_num_samples",
                    "distillation.student.init.num_samples",
                    "distillation.student.umap_num_samples",
                ],
                default="",
            ),
            5000,
        ),
        "max_length": _parse_int(
            _get_first(
                overrides,
                [
                    "distillation.student.init_max_length",
                    "distillation.student.init.max_length",
                    "distillation.student.umap_max_length",
                ],
                default="",
            ),
            128,
        ),
        "umap_n_neighbors": _parse_int(
            _get_first(
                overrides, ["distillation.student.umap_n_neighbors"], default=""
            ),
            15,
        ),
        "umap_min_dist": _parse_float(
            _get_first(overrides, ["distillation.student.umap_min_dist"], default=""),
            0.1,
        ),
        "umap_metric": _get_first(
            overrides, ["distillation.student.umap_metric"], default="cosine"
        ),
        "init_epochs": _parse_int(
            _get_first(overrides, ["distillation.student.init_epochs"], default=""), 10
        ),
        "init_batch_size": _parse_int(
            _get_first(overrides, ["distillation.student.init_batch_size"], default=""),
            16,
        ),
        "init_lr": _parse_float(
            _get_first(overrides, ["distillation.student.init_lr"], default=""), 1e-4
        ),
        "init_weight_decay": _parse_float(
            _get_first(
                overrides, ["distillation.student.init_weight_decay"], default=""
            ),
            0.01,
        ),
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "dataset_hf_path": hf_path,
        "dataset_hf_config": hf_config,
        "dataset_hf_split": hf_split,
        "dataset_streaming": streaming,
    }
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VT2 distillation runner that accepts Hydra-style key=value overrides, "
            "runs CE or KLD distillation training, and emits standardized JSONL result records."
        )
    )
    parser.add_argument(
        "--results-path",
        default=DEFAULT_RESULTS_PATH,
        help="Append-only JSONL path for distillation run records.",
    )
    parser.add_argument(
        "--run-dir-base",
        default=DEFAULT_RUN_DIR_BASE,
        help="Base directory for per-run artifacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Console log level.",
    )
    return parser


def _run_training(
    *,
    run_dir: Path,
    overrides: Dict[str, str],
    log: logging.Logger,
    wandb_enabled: bool = False,
    wandb_run: Optional[object] = None,
    lm_eval_settings: Optional[Dict[str, object]] = None,
) -> None:
    hf_token = os.environ.get("HF_TOKEN")
    device = _resolve_device()

    lm_eval_settings = lm_eval_settings or {}
    lm_eval_enabled = bool(lm_eval_settings.get("enabled"))
    lm_eval_train_enabled = bool(lm_eval_settings.get("train_enabled"))
    if (lm_eval_enabled or lm_eval_train_enabled) and importlib.util.find_spec(
        "lm_eval"
    ) is None:
        raise RuntimeError(
            "lm-eval is enabled but not installed. Add lm-eval[hf] to dependencies."
        )

    lm_eval_tasks = list(lm_eval_settings.get("tasks") or [])
    lm_eval_train_tasks = list(lm_eval_settings.get("train_tasks") or []) or lm_eval_tasks
    if lm_eval_enabled and not lm_eval_tasks:
        raise ValueError("lm_eval.enabled=true but no lm_eval.tasks provided.")
    if lm_eval_train_enabled and not lm_eval_train_tasks:
        raise ValueError(
            "lm_eval.train.enabled=true but no lm_eval.train.tasks provided."
        )

    lm_eval_results_path = Path(
        lm_eval_settings.get("results_path") or run_dir / LM_EVAL_RESULTS_FILENAME
    )

    teacher_model_name = _get_first(
        overrides,
        ["distillation.teacher_model", "teacher_model", "pretrained.model_name"],
        default="",
    )
    if not teacher_model_name:
        raise ValueError(
            "Missing teacher model. Set distillation.teacher_model=<hf-or-path>."
        )

    tokenizer_model = _get_first(
        overrides,
        [
            "distillation.tokenizer_model",
            "tokenizer.model_name",
            "pretrained.model_name",
        ],
        default=teacher_model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = _get_first(
        overrides, ["data.input_path", "dataset", "data.hf_path"], default=""
    )
    dataset_hf_path = _get_first(overrides, ["data.hf_path"], default="")
    dataset_hf_config = _get_first(overrides, ["data.hf_config"], default="")
    data_streaming = _parse_bool(
        _get_first(overrides, ["data.streaming"], default=""),
        default=False,
    )
    train_split = _get_first(
        overrides,
        ["data.hf_train_split", "data.prepared_split", "data.split", "split"],
        default="train",
    )
    train_prepared_path = _get_first(overrides, ["data.prepared_path"], default="")
    train_prepared_dir = _get_first(overrides, ["data.prepared_dir"], default="")

    max_train_samples = _parse_int(
        _get_first(
            overrides,
            ["distillation.max_train_samples", "data.max_train_samples"],
            default="",
        ),
        default=-1,
    )

    train_raw = _load_raw_dataset(
        dataset_name=dataset_name,
        hf_path=dataset_hf_path,
        hf_config=dataset_hf_config,
        split=train_split,
        prepared_path=train_prepared_path,
        prepared_dir=train_prepared_dir,
        streaming=data_streaming,
    )
    train_raw = _limit_raw_dataset(train_raw, max_samples=max_train_samples)

    max_seq_length = _parse_int(
        _get_first(
            overrides,
            ["distillation.max_seq_length", "tokenizer.chunk_length"],
            default="",
        ),
        default=0,
    )
    train_dataset = _tokenize_or_normalize_dataset(
        dataset=train_raw,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    eval_dataset = None
    eval_prepared_path = _get_first(overrides, ["data.eval_prepared_path"], default="")
    eval_prepared_dir = _get_first(overrides, ["data.eval_prepared_dir"], default="")
    eval_split = _get_first(
        overrides,
        ["data.hf_eval_split", "data.eval_prepared_split", "data.eval_split"],
        default="",
    )
    if eval_prepared_path or eval_prepared_dir or eval_split:
        max_eval_samples = _parse_int(
            _get_first(
                overrides,
                ["distillation.max_eval_samples", "data.max_eval_samples"],
                default="",
            ),
            default=-1,
        )
        eval_raw = _load_raw_dataset(
            dataset_name=dataset_name,
            hf_path=dataset_hf_path,
            hf_config=dataset_hf_config,
            split=(eval_split or "eval"),
            prepared_path=eval_prepared_path,
            prepared_dir=eval_prepared_dir,
            streaming=data_streaming,
        )
        eval_raw = _limit_raw_dataset(eval_raw, max_samples=max_eval_samples)
        eval_dataset = _tokenize_or_normalize_dataset(
            dataset=eval_raw,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

    log.info("Device: %s", device)
    log.info("Teacher model: %s", teacher_model_name)
    log.info("Tokenizer model: %s", tokenizer_model)
    log.info(
        "Data source: hf_path=%s hf_config=%s streaming=%s input_path=%s",
        dataset_hf_path or "<auto>",
        dataset_hf_config or "<none>",
        data_streaming,
        dataset_name or "<none>",
    )
    log.info(
        "Train split: %s | #samples=%s", train_split, _dataset_size_str(train_dataset)
    )
    log.info("Eval enabled: %s", eval_dataset is not None)
    if eval_dataset is not None:
        log.info("Eval split: %s | #samples=%s", eval_split or "eval", _dataset_size_str(eval_dataset))

    student_scale_factor = _parse_float(
        _get_first(overrides, ["distillation.student.scale_factor"], default=""),
        default=0.25,
    )
    student_hidden_size = _parse_int(
        _get_first(overrides, ["distillation.student.hidden_size"], default=""),
        default=-1,
    )
    student_hidden_size_opt: Optional[int] = (
        student_hidden_size if student_hidden_size > 0 else None
    )

    student_init_strategy = _get_first(
        overrides,
        ["distillation.student.init_strategy"],
        default="random",
    )

    student_init_cfg = _build_student_init_cfg(overrides)
    if (
        student_init_strategy in {"umap_layerwise", "pca_layerwise"}
        and student_init_cfg is None
    ):
        raise ValueError(
            f"distillation.student.init_strategy={student_init_strategy} requires "
            "distillation.student.init_cache_dir."
        )

    student_model, _student_config = create_student_model(
        teacher_model_name=teacher_model_name,
        scale_factor=student_scale_factor,
        init_strategy=student_init_strategy,
        hidden_size=student_hidden_size_opt,
        device=device,
        student_init_cfg=student_init_cfg,
        attn_implementation=_resolve_attn_impl(teacher_model_name),
    )

    # Some tiny checkpoints ship invalid generation config values (e.g. pad_token_id=-1),
    # which breaks checkpoint/model saving. Normalize from tokenizer when available.
    generation_config = getattr(student_model, "generation_config", None)
    if generation_config is not None:
        tok_pad = getattr(tokenizer, "pad_token_id", None)
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        tok_bos = getattr(tokenizer, "bos_token_id", None)

        if tok_pad is not None and (
            getattr(generation_config, "pad_token_id", None) is None
            or int(getattr(generation_config, "pad_token_id")) < 0
        ):
            generation_config.pad_token_id = int(tok_pad)
            log.info(
                "Normalized student generation_config.pad_token_id=%s",
                generation_config.pad_token_id,
            )
        if tok_eos is not None and getattr(generation_config, "eos_token_id", None) is None:
            generation_config.eos_token_id = int(tok_eos)
        if tok_bos is not None and getattr(generation_config, "bos_token_id", None) is None:
            generation_config.bos_token_id = int(tok_bos)

    model_config = getattr(student_model, "config", None)
    if model_config is not None:
        tok_pad = getattr(tokenizer, "pad_token_id", None)
        if tok_pad is not None and (
            getattr(model_config, "pad_token_id", None) is None
            or int(getattr(model_config, "pad_token_id")) < 0
        ):
            model_config.pad_token_id = int(tok_pad)

    distill_loss = _get_first(
        overrides,
        ["distillation.loss", "distill_loss"],
        default="cross_entropy",
    ).lower()
    if distill_loss not in {"cross_entropy", "kld"}:
        raise ValueError("distillation.loss must be one of: cross_entropy, kld")

    teacher_fp16 = _parse_bool(
        _get_first(overrides, ["distillation.teacher_fp16"], default=""),
        default=torch.cuda.is_available(),
    )
    allow_gemma_teacher_fp16 = _parse_bool(
        _get_first(
            overrides,
            ["distillation.allow_gemma_teacher_fp16", "allow_gemma_teacher_fp16"],
            default="",
        ),
        default=False,
    )
    if not torch.cuda.is_available():
        teacher_fp16 = False
    if (
        "gemma-3" in teacher_model_name.lower()
        and teacher_fp16
        and not allow_gemma_teacher_fp16
    ):
        log.warning("Forcing distillation.teacher_fp16=false for Gemma3 KLD stability.")
        teacher_fp16 = False
    elif "gemma-3" in teacher_model_name.lower() and teacher_fp16:
        log.warning(
            "Allowing Gemma3 teacher_fp16 via distillation.allow_gemma_teacher_fp16=true; "
            "use only for dedicated smoke tests."
        )

    temperature = _parse_float(
        _get_first(overrides, ["distillation.temperature"], default=""),
        default=2.0,
    )
    alpha = _parse_float(
        _get_first(overrides, ["distillation.alpha"], default=""),
        default=0.9,
    )
    teacher_batch_size = _parse_int(
        _get_first(overrides, ["distillation.teacher_batch_size"], default=""),
        default=-1,
    )
    teacher_batch_size_opt: Optional[int] = (
        teacher_batch_size if teacher_batch_size > 0 else None
    )

    if distill_loss != "kld":
        temperature = -1.0
        alpha = -1.0

    epochs = _parse_float(
        _get_first(
            overrides,
            [
                "model.train_args.num_train_epochs",
                "distillation.train.num_train_epochs",
            ],
            default="",
        ),
        default=1.0,
    )
    per_device_train_batch_size = _parse_int(
        _get_first(
            overrides,
            [
                "model.train_args.per_device_train_batch_size",
                "distillation.train.per_device_train_batch_size",
            ],
            default="",
        ),
        default=4,
    )
    per_device_eval_batch_size = _parse_int(
        _get_first(
            overrides,
            [
                "model.train_args.per_device_eval_batch_size",
                "distillation.train.per_device_eval_batch_size",
            ],
            default="",
        ),
        default=per_device_train_batch_size,
    )
    grad_accum = _parse_int(
        _get_first(
            overrides,
            [
                "model.train_args.gradient_accumulation_steps",
                "distillation.train.gradient_accumulation_steps",
            ],
            default="",
        ),
        default=1,
    )
    learning_rate = _parse_float(
        _get_first(
            overrides,
            ["model.train_args.learning_rate", "distillation.train.learning_rate"],
            default="",
        ),
        default=5e-5,
    )
    weight_decay = _parse_float(
        _get_first(
            overrides,
            ["model.train_args.weight_decay", "distillation.train.weight_decay"],
            default="",
        ),
        default=0.0,
    )
    logging_steps = _parse_int(
        _get_first(
            overrides,
            ["model.train_args.logging_steps", "distillation.train.logging_steps"],
            default="",
        ),
        default=10,
    )
    save_strategy = _get_first(
        overrides,
        ["model.train_args.save_strategy", "distillation.train.save_strategy"],
        default="steps",
    ).lower()
    if save_strategy not in {"steps", "epoch"}:
        raise ValueError("model.train_args.save_strategy must be one of: steps, epoch")
    save_steps = _parse_int(
        _get_first(
            overrides,
            ["model.train_args.save_steps", "distillation.train.save_steps"],
            default="",
        ),
        default=500,
    )
    save_total_limit_raw = _get_first(
        overrides,
        ["model.train_args.save_total_limit", "distillation.train.save_total_limit"],
        default="",
    )
    if save_strategy == "epoch":
        epoch_ckpt_default = max(1, int(math.ceil(epochs)))
        save_total_limit = _parse_int(
            save_total_limit_raw,
            default=epoch_ckpt_default,
        )
    else:
        save_total_limit = _parse_int(save_total_limit_raw, default=2)
    max_grad_norm = _parse_float(
        _get_first(
            overrides,
            ["model.train_args.max_grad_norm", "distillation.train.max_grad_norm"],
            default="",
        ),
        default=1.0,
    )
    train_seed = _parse_int(
        _get_first(
            overrides, ["model.train_args.seed", "run.seed", "seed"], default=""
        ),
        default=42,
    )

    bf16 = _parse_bool(
        _get_first(
            overrides, ["model.train_args.bf16", "distillation.train.bf16"], default=""
        ),
        default=torch.cuda.is_available(),
    )
    fp16 = _parse_bool(
        _get_first(
            overrides, ["model.train_args.fp16", "distillation.train.fp16"], default=""
        ),
        default=False,
    )
    if bf16 and fp16:
        fp16 = False

    lr_scheduler_type = _get_first(
        overrides,
        ["model.train_args.lr_scheduler_type", "distillation.train.lr_scheduler_type"],
        default="linear",
    )
    warmup_ratio = _parse_float(
        _get_first(
            overrides,
            ["model.train_args.warmup_ratio", "distillation.train.warmup_ratio"],
            default="",
        ),
        default=0.0,
    )
    warmup_steps = _parse_int(
        _get_first(
            overrides,
            ["model.train_args.warmup_steps", "distillation.train.warmup_steps"],
            default="",
        ),
        default=0,
    )
    max_steps = _parse_int(
        _get_first(
            overrides,
            ["model.train_args.max_steps", "distillation.train.max_steps"],
            default="",
        ),
        default=-1,
    )
    if _is_iterable_dataset(train_dataset) and max_steps <= 0:
        raise ValueError(
            "Training dataset is streaming/iterable, so max_steps must be set "
            "(model.train_args.max_steps or distillation.train.max_steps > 0)."
        )
    if _is_iterable_dataset(eval_dataset):
        raise ValueError(
            "Eval dataset is streaming/iterable. Please provide a finite eval split "
            "(non-streaming prepared eval files/dataset) for evaluation."
        )

    dataloader_num_workers = _parse_int(
        _get_first(
            overrides, ["distillation.train.dataloader_num_workers"], default=""
        ),
        default=0,
    )

    sampler_mode = _get_first(
        overrides,
        ["model.train_args.sampler", "sampler", "distillation.train.sampler"],
        default="random",
    )

    evaluation_strategy = "no"
    if eval_dataset is not None:
        evaluation_strategy = _get_first(
            overrides,
            [
                "distillation.train.evaluation_strategy",
                "model.train_args.evaluation_strategy",
            ],
            default="steps",
        )

    run_name = _get_first(
        overrides, ["run.exp_name", "run_artifact_id", "entry_id"], default=run_dir.name
    )

    if wandb_run is not None:
        try:
            wandb_run.config.update(
                {
                    "teacher_model": teacher_model_name,
                    "tokenizer_model": tokenizer_model,
                    "distill_loss": distill_loss,
                    "distill_method": _get_first(
                        overrides, ["distill_method", "distillation.method"], default=""
                    ),
                    "dataset": dataset_name,
                    "train_split": train_split,
                    "train_samples": _dataset_len(train_dataset) or -1,
                    "eval_samples": _dataset_len(eval_dataset) if eval_dataset is not None else 0,
                    "data_streaming": data_streaming,
                    "dataset_hf_path": dataset_hf_path,
                    "dataset_hf_config": dataset_hf_config,
                    "student_init_strategy": student_init_strategy,
                    "student_hidden_size": student_hidden_size_opt or student_hidden_size,
                },
                allow_val_change=True,
            )
        except Exception as exc:
            log.warning("wandb config update failed: %s", exc)

    training_args_kwargs = {
        "output_dir": str(run_dir),
        "overwrite_output_dir": True,
        "run_name": run_name,
        "report_to": ["wandb"] if wandb_enabled else "none",
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "fp16": fp16,
        "bf16": bf16,
        "bf16_full_eval": bf16,
        "fp16_full_eval": fp16,
        "seed": train_seed,
        "max_grad_norm": max_grad_norm,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "ddp_find_unused_parameters": False,
        "dataloader_drop_last": False,
        "dataloader_num_workers": max(dataloader_num_workers, 0),
    }
    if save_strategy == "steps":
        training_args_kwargs["save_steps"] = save_steps

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_args_kwargs["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in ta_params:
        training_args_kwargs["eval_strategy"] = evaluation_strategy
    else:
        log.warning(
            "TrainingArguments has neither `evaluation_strategy` nor `eval_strategy`; "
            "evaluation scheduling will use library defaults."
        )

    unsupported_ta_keys = [
        k for k in list(training_args_kwargs.keys()) if k not in ta_params
    ]
    if unsupported_ta_keys:
        for key in sorted(unsupported_ta_keys):
            log.warning("TrainingArguments does not support `%s`; skipping.", key)
            training_args_kwargs.pop(key, None)

    training_args = TrainingArguments(**training_args_kwargs)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        "sampler_mode": sampler_mode,
        "model": student_model,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    run_meta = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "artifact_id": run_dir.name,
        "teacher_model": teacher_model_name,
        "student_init_strategy": student_init_strategy,
        "distill_loss": distill_loss,
    }
    callbacks: List[TrainerCallback] = []
    if lm_eval_train_enabled:
        callbacks.append(
            LmEvalCallback(
                log=log,
                hf_token=hf_token or "",
                tasks=lm_eval_train_tasks,
                batch_size=lm_eval_settings.get("train_batch_size") or "auto",
                device=str(lm_eval_settings.get("train_device") or device),
                num_fewshot=int(lm_eval_settings.get("train_num_fewshot") or 0),
                limit=lm_eval_settings.get("train_limit"),
                log_samples=bool(lm_eval_settings.get("log_samples") or False),
                apply_chat_template=lm_eval_settings.get("apply_chat_template"),
                trust_remote_code=lm_eval_settings.get("trust_remote_code"),
                results_path=lm_eval_results_path,
                run_meta=run_meta,
                wandb_run=wandb_run,
            )
        )
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    teacher_model = None
    if distill_loss == "kld":
        teacher_dtype = torch.float16 if teacher_fp16 else torch.float32
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            token=hf_token,
            torch_dtype=teacher_dtype,
            attn_implementation=_resolve_attn_impl(teacher_model_name),
        )

        trainer = DistillKLDTrainer(
            teacher_model=teacher_model,
            temperature=temperature,
            alpha=alpha,
            teacher_fp16=teacher_fp16,
            teacher_batch_size=teacher_batch_size_opt,
            **trainer_kwargs,
        )
    else:
        trainer = DistillCETrainer(**trainer_kwargs)

    mode = _get_first(overrides, ["run.mode"], default="new").lower()
    if mode not in {"auto", "new", "resume"}:
        raise ValueError(f"Unsupported run.mode={mode!r}. Use auto|new|resume.")

    if mode == "new":
        resume_ckpt = None
    else:
        resume_ckpt = get_last_checkpoint(str(run_dir))

    if mode == "resume" and not resume_ckpt:
        raise RuntimeError(f"run.mode=resume but no checkpoint found under {run_dir}")

    if resume_ckpt:
        log.info("Resuming training from checkpoint: %s", resume_ckpt)
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        train_result = trainer.train()

    eval_metrics = {}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()

    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    student_actual_params = _count_model_params(student_model)
    student_safetensors_mb = _total_safetensors_mb(run_dir)

    overrides["student_actual_params"] = str(student_actual_params)
    overrides["student_safetensors_mb"] = f"{student_safetensors_mb:.6f}"
    overrides["distillation.temperature"] = f"{temperature}"
    overrides["distillation.alpha"] = f"{alpha}"
    train_n = _dataset_len(train_dataset)
    eval_n = _dataset_len(eval_dataset)
    overrides["train_num_samples"] = str(train_n if train_n is not None else -1)
    overrides["eval_num_samples"] = str(eval_n if eval_n is not None else 0)

    metrics = dict(train_result.metrics or {})
    if "train_loss" in metrics:
        overrides["train_loss"] = f"{metrics['train_loss']}"
    if "train_runtime" in metrics:
        overrides["train_runtime_s"] = f"{metrics['train_runtime']}"
    if "train_samples_per_second" in metrics:
        overrides["train_samples_per_second"] = f"{metrics['train_samples_per_second']}"
    if "train_steps_per_second" in metrics:
        overrides["train_steps_per_second"] = f"{metrics['train_steps_per_second']}"
    if "global_step" in metrics:
        overrides["train_global_step"] = str(int(metrics["global_step"]))

    state_global_step = int(getattr(trainer.state, "global_step", -1))
    if state_global_step >= 0:
        overrides["train_global_step"] = str(state_global_step)

    if "eval_loss" in eval_metrics:
        overrides["eval_loss"] = f"{eval_metrics['eval_loss']}"

    log.info("Training complete. student_actual_params=%s", student_actual_params)
    log.info("Saved safetensors size (MB): %.4f", student_safetensors_mb)

    if lm_eval_enabled and is_rank0():
        _run_lm_eval_and_log(
            log=log,
            hf_token=hf_token or "",
            model_ref=str(run_dir),
            tasks=lm_eval_tasks,
            batch_size=lm_eval_settings.get("batch_size") or "auto",
            device=str(lm_eval_settings.get("device") or device),
            num_fewshot=int(lm_eval_settings.get("num_fewshot") or 0),
            limit=lm_eval_settings.get("limit"),
            log_samples=bool(lm_eval_settings.get("log_samples") or False),
            apply_chat_template=lm_eval_settings.get("apply_chat_template"),
            trust_remote_code=lm_eval_settings.get("trust_remote_code"),
            phase="post_train",
            results_path=lm_eval_results_path,
            run_meta=run_meta,
            wandb_run=wandb_run,
            global_step=state_global_step if state_global_step >= 0 else None,
        )

    run_metrics_payload = {
        "created_at": _now_iso(),
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "student_actual_params": student_actual_params,
        "student_safetensors_mb": student_safetensors_mb,
        "distill_loss": distill_loss,
        "distill_temperature": temperature,
        "distill_alpha": alpha,
        "lm_eval_results_path": str(lm_eval_results_path)
        if (lm_eval_enabled or lm_eval_train_enabled)
        else "",
    }
    _write_run_meta(
        run_dir,
        overrides,
        payload=run_metrics_payload,
        filename="DISTILL_METRICS.json",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = _parse_overrides(unknown)
    _load_env_file(Path(".env"), override=False)

    artifact_id = _safe_slug(
        _get_first(
            overrides,
            [
                "run_artifact_id",
                "run.artifact_id",
                "run.exp_name",
                "run.run_id",
                "entry_id",
            ],
            default="",
        ),
        fallback=f"distill_{datetime.now():%Y%m%d_%H%M%S}",
    )

    run_dir = Path(args.run_dir_base) / artifact_id
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_mode = _get_first(overrides, ["run.mode"], default="new").lower()
    force_clear = _parse_bool(
        _get_first(overrides, ["run.force_clear"], default=""), default=False
    )
    if requested_mode == "new":
        run_dir_has_contents = any(run_dir.iterdir())
        if run_dir_has_contents:
            if force_clear:
                _clear_dir_contents(run_dir)
            else:
                raise RuntimeError(
                    "run.mode=new but target run_dir is not empty: "
                    f"{run_dir}. Use a unique run_artifact_id, "
                    "set run.mode=resume/auto, or set run.force_clear=true."
                )

    log = setup_logging("run_distill", level=args.log_level)
    log_path = run_dir / "run_distill.log"
    _init_file_logger(log, log_path)

    run_name = _get_first(
        overrides, ["run.exp_name", "run_artifact_id", "entry_id"], default=artifact_id
    )
    wandb_settings = _resolve_wandb_settings(
        overrides, run_name=run_name, run_dir=run_dir
    )
    wandb_run = _maybe_init_wandb(wandb_settings, overrides=overrides, log=log)
    lm_eval_settings = _resolve_lm_eval_settings(overrides, run_dir=run_dir)

    submitted_at = _now_iso()
    status = "success"
    error_note = ""

    try:
        log.info("Starting VT2 distillation run")
        log.info("Artifact ID: %s", artifact_id)
        log.info("Resolved overrides: %s", json.dumps(overrides, ensure_ascii=True))

        _write_run_meta(
            run_dir,
            overrides,
            payload={
                "created_at": _now_iso(),
                "overrides": overrides,
                "wandb": {
                    "enabled": bool(wandb_settings.get("enabled")),
                    "project": wandb_settings.get("project"),
                    "entity": wandb_settings.get("entity"),
                    "group": wandb_settings.get("group"),
                    "name": wandb_settings.get("name"),
                },
                "lm_eval": make_json_safe(lm_eval_settings),
            },
            filename="RUN_META.json",
        )

        _run_training(
            run_dir=run_dir,
            overrides=overrides,
            log=log,
            wandb_enabled=bool(wandb_settings.get("enabled")) and is_rank0(),
            wandb_run=wandb_run,
            lm_eval_settings=lm_eval_settings,
        )

    except Exception as exc:
        status = "failed"
        error_note = str(exc)
        log.exception("Distillation run failed: %s", exc)
    finally:
        finished_at = _now_iso()
        result_row = _build_result_row(
            overrides=overrides,
            status=status,
            submitted_at=submitted_at,
            finished_at=finished_at,
            run_dir=run_dir,
            log_path=log_path,
            error_note=error_note,
        )

        _append_jsonl(Path(args.results_path), result_row)
        print(json.dumps(result_row, ensure_ascii=True))
        if wandb_run is not None and is_rank0():
            try:
                import wandb

                summary_payload = {
                    "status": status,
                    "train_loss": result_row.get("train_loss"),
                    "eval_loss": result_row.get("eval_loss"),
                    "train_runtime_s": result_row.get("train_runtime_s"),
                    "train_samples_per_second": result_row.get("train_samples_per_second"),
                    "train_steps_per_second": result_row.get("train_steps_per_second"),
                    "student_actual_params": result_row.get("student_actual_params"),
                    "student_safetensors_mb": result_row.get("student_safetensors_mb"),
                }
                wandb.run.summary.update(summary_payload)
                wandb.finish()
            except Exception as exc:
                log.warning("wandb finalize failed: %s", exc)

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
