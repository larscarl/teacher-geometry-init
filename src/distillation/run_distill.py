import argparse
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
from typing import Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from src.distillation.kld_loss_trainer import KLDistillationTrainer
from src.distillation.student_factory import create_student_model
from src.utils.helper import setup_logging


DEFAULT_RESULTS_PATH = "experiments/results_distill.jsonl"
DEFAULT_RUN_DIR_BASE = "experiments/runs_distill"


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
    split: str,
    prepared_path: str,
    prepared_dir: str,
) -> Dataset:
    explicit_path = _strip(prepared_path)
    explicit_dir = _strip(prepared_dir)

    if explicit_path:
        return _load_dataset_from_path(Path(explicit_path), split=split)

    if explicit_dir:
        return _load_dataset_from_path(Path(explicit_dir), split=split)

    dataset_name = _strip(dataset_name)
    if dataset_name and Path(dataset_name).exists():
        return _load_dataset_from_path(Path(dataset_name), split=split)

    if dataset_name:
        return load_dataset(dataset_name, split=split)

    raise ValueError(
        "No data source resolved. Set data.prepared_path or data.prepared_dir "
        "(recommended for VT2 protocol runs)."
    )


def _tokenize_or_normalize_dataset(
    *,
    dataset: Dataset,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
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
    dataset = _get_first(overrides, ["dataset", "data.input_path"], default="unknown")
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

    dataset_name = _get_first(overrides, ["data.input_path"], default="wikitext")
    dataset_path = _get_first(
        overrides,
        [
            "distillation.student.dataset_path",
            "data.prepared_path",
            "data.prepared_dir",
        ],
        default=dataset_name,
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
) -> None:
    hf_token = os.environ.get("HF_TOKEN")
    device = _resolve_device()

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

    dataset_name = _get_first(overrides, ["data.input_path", "dataset"], default="")
    train_split = _get_first(
        overrides,
        ["data.prepared_split", "data.split", "split"],
        default="train",
    )
    train_prepared_path = _get_first(overrides, ["data.prepared_path"], default="")
    train_prepared_dir = _get_first(overrides, ["data.prepared_dir"], default="")

    train_raw = _load_raw_dataset(
        dataset_name=dataset_name,
        split=train_split,
        prepared_path=train_prepared_path,
        prepared_dir=train_prepared_dir,
    )

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

    max_train_samples = _parse_int(
        _get_first(
            overrides,
            ["distillation.max_train_samples", "data.max_train_samples"],
            default="",
        ),
        default=-1,
    )
    if max_train_samples > 0:
        train_dataset = train_dataset.select(
            range(min(max_train_samples, len(train_dataset)))
        )

    eval_dataset = None
    eval_prepared_path = _get_first(overrides, ["data.eval_prepared_path"], default="")
    eval_prepared_dir = _get_first(overrides, ["data.eval_prepared_dir"], default="")
    eval_split = _get_first(
        overrides, ["data.eval_prepared_split", "data.eval_split"], default=""
    )
    if eval_prepared_path or eval_prepared_dir or eval_split:
        eval_raw = _load_raw_dataset(
            dataset_name=dataset_name,
            split=(eval_split or "eval"),
            prepared_path=eval_prepared_path,
            prepared_dir=eval_prepared_dir,
        )
        eval_dataset = _tokenize_or_normalize_dataset(
            dataset=eval_raw,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        max_eval_samples = _parse_int(
            _get_first(
                overrides,
                ["distillation.max_eval_samples", "data.max_eval_samples"],
                default="",
            ),
            default=-1,
        )
        if max_eval_samples > 0:
            eval_dataset = eval_dataset.select(
                range(min(max_eval_samples, len(eval_dataset)))
            )

    log.info("Device: %s", device)
    log.info("Teacher model: %s", teacher_model_name)
    log.info("Tokenizer model: %s", tokenizer_model)
    log.info("Train split: %s | #samples=%s", train_split, len(train_dataset))
    log.info("Eval enabled: %s", eval_dataset is not None)

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

    training_args_kwargs = {
        "output_dir": str(run_dir),
        "overwrite_output_dir": True,
        "run_name": run_name,
        "report_to": "none",
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

    if mode == "new" and _parse_bool(
        _get_first(overrides, ["run.force_clear"], default=""), default=False
    ):
        _clear_dir_contents(run_dir)

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
    overrides["train_num_samples"] = str(len(train_dataset))
    overrides["eval_num_samples"] = str(
        len(eval_dataset) if eval_dataset is not None else 0
    )

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

    run_metrics_payload = {
        "created_at": _now_iso(),
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "student_actual_params": student_actual_params,
        "student_safetensors_mb": student_safetensors_mb,
        "distill_loss": distill_loss,
        "distill_temperature": temperature,
        "distill_alpha": alpha,
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

    log = setup_logging("run_distill", level=args.log_level)
    log_path = run_dir / "run_distill.log"
    _init_file_logger(log, log_path)

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
            },
            filename="RUN_META.json",
        )

        _run_training(run_dir=run_dir, overrides=overrides, log=log)

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

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
