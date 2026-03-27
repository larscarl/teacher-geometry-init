import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


DEFAULT_REGISTRY = "experiments/registry_distill.csv"
DEFAULT_SUBMISSIONS_LOG = "experiments/submissions_distill.jsonl"
DEFAULT_RESULTS_LOG = "experiments/results_distill.jsonl"
DEFAULT_RUN_DIR_BASE = "experiments/runs_distill"

REQUIRED_COLUMNS = [
    "entry_id",
    "enabled",
    "submit_via",
    "dataset",
    "split",
    "seed",
    "sampler",
    "teacher_model",
    "student_family",
    "student_arch",
    "student_target_params",
    "distill_loss",
    "distill_method",
    "run_artifact_id",
    "run_mode",
    "run_exp_name",
    "student_init_strategy",
    "student_hidden_size",
    "tokenizer_chunk_length",
    "data_input_path",
    "data_prepared_dir",
    "data_prepared_split",
    "data_eval_prepared_dir",
    "data_eval_prepared_split",
    "train_num_epochs",
    "train_batch_size",
    "train_grad_accum",
    "train_learning_rate",
    "train_logging_steps",
    "train_save_strategy",
    "train_save_steps",
    "train_save_total_limit",
    "train_bf16",
    "train_fp16",
    "distillation_max_train_samples",
    "distillation_max_eval_samples",
]

ALLOWED_SUBMIT_VIA = {"sbatch", "python"}
ALLOWED_DISTILL_LOSS = {"cross_entropy", "kld"}
ALLOWED_TRAIN_SAVE_STRATEGY = {"steps", "epoch"}

OPTIONAL_EMPTY_COLUMNS = {
    "train_save_steps",
    "train_save_total_limit",
}

BOOL_COLUMNS = {
    "distillation_teacher_fp16",
    "train_bf16",
    "train_fp16",
    "data_streaming",
}

INT_COLUMNS = {
    "seed",
    "student_target_params",
    "student_hidden_size",
    "tokenizer_chunk_length",
    "distillation_teacher_batch_size",
    "train_num_epochs",
    "train_batch_size",
    "train_grad_accum",
    "train_logging_steps",
    "train_save_steps",
    "train_save_total_limit",
    "distillation_max_train_samples",
    "distillation_max_eval_samples",
}

FLOAT_COLUMNS = {
    "distillation_temperature",
    "distillation_alpha",
    "train_learning_rate",
}

DISTILL_RUNTIME_COLUMN_MAP = [
    ("run_mode", "run.mode"),
    ("run_exp_name", "run.exp_name"),
    ("distillation_temperature", "distillation.temperature"),
    ("distillation_alpha", "distillation.alpha"),
    ("distillation_teacher_fp16", "distillation.teacher_fp16"),
    ("distillation_teacher_batch_size", "distillation.teacher_batch_size"),
    ("student_init_strategy", "distillation.student.init_strategy"),
    ("student_hidden_size", "distillation.student.hidden_size"),
    ("tokenizer_chunk_length", "tokenizer.chunk_length"),
    ("data_input_path", "data.input_path"),
    ("data_hf_path", "data.hf_path"),
    ("data_hf_config", "data.hf_config"),
    ("data_hf_train_split", "data.hf_train_split"),
    ("data_hf_eval_split", "data.hf_eval_split"),
    ("data_streaming", "data.streaming"),
    ("data_prepared_path", "data.prepared_path"),
    ("data_prepared_dir", "data.prepared_dir"),
    ("data_prepared_split", "data.prepared_split"),
    ("data_eval_prepared_path", "data.eval_prepared_path"),
    ("data_eval_prepared_dir", "data.eval_prepared_dir"),
    ("data_eval_prepared_split", "data.eval_prepared_split"),
    ("train_num_epochs", "model.train_args.num_train_epochs"),
    ("train_batch_size", "model.train_args.per_device_train_batch_size"),
    ("train_grad_accum", "model.train_args.gradient_accumulation_steps"),
    ("train_learning_rate", "model.train_args.learning_rate"),
    ("train_logging_steps", "model.train_args.logging_steps"),
    ("train_save_strategy", "model.train_args.save_strategy"),
    ("train_save_steps", "model.train_args.save_steps"),
    ("train_save_total_limit", "model.train_args.save_total_limit"),
    ("train_bf16", "model.train_args.bf16"),
    ("train_fp16", "model.train_args.fp16"),
    ("distillation_max_train_samples", "distillation.max_train_samples"),
    ("distillation_max_eval_samples", "distillation.max_eval_samples"),
]

DISTILL_RUNTIME_DERIVED_MAP = [
    ("distillation.loss", "distill_loss"),
    ("distillation.method", "distill_method"),
    ("distillation.teacher_model", "teacher_model"),
    ("model.train_args.seed", "seed"),
    ("model.train_args.sampler", "sampler"),
]


def _strip(value: Optional[str]) -> str:
    return (value or "").strip()


def _nonnull(value: Optional[str]) -> Optional[str]:
    text = _strip(value)
    return text if text else None


def _as_bool(value: Optional[str], default: Optional[bool] = None) -> Optional[bool]:
    text = _strip(value).lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value!r}")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_registry(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {', '.join(missing_cols)}"
            )

        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(reader, start=2):
            rows.append(
                {
                    "_line": str(idx),
                    **{k: (v or "") for k, v in row.items()},
                }
            )
    return rows


def _split_pipe_list(value: Optional[str]) -> List[str]:
    raw = _strip(value)
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split("|") if chunk.strip()]


def _append_override(overrides: List[str], key: str, value: Optional[str]) -> None:
    if value is None:
        return
    overrides.append(f"{key}={value}")


def _extract_artifact_id_override(extra_overrides: List[str]) -> Optional[str]:
    run_artifact_id: Optional[str] = None
    for token in extra_overrides:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.strip() != "run_artifact_id":
            continue
        text = value.strip()
        if text:
            run_artifact_id = text
    return run_artifact_id


def _resolved_row_artifact_id(row: Dict[str, str]) -> str:
    fallback = _strip(row.get("entry_id")) or "distill_run"
    base = _strip(row.get("run_artifact_id")) or fallback
    override = _extract_artifact_id_override(_split_pipe_list(row.get("extra_overrides")))
    return override or base


def _read_jsonl_artifact_ids(path: Path) -> Set[str]:
    artifact_ids: Set[str] = set()
    if not path.exists():
        return artifact_ids

    with path.open("r", encoding="utf-8") as rf:
        for line in rf:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            run_artifact_id = _strip(str(payload.get("run_artifact_id", "")))
            if run_artifact_id:
                artifact_ids.add(run_artifact_id)

    return artifact_ids


def _find_duplicate_reasons(
    *,
    artifact_id: str,
    run_dir_base: Path,
    seen_submission_ids: Set[str],
    seen_result_ids: Set[str],
    selected_seen_ids: Set[str],
    submissions_log: Path,
    results_log: Path,
) -> List[str]:
    reasons: List[str] = []

    if artifact_id in selected_seen_ids:
        reasons.append("artifact id already selected in this invocation")
    if artifact_id in seen_submission_ids:
        reasons.append(f"artifact id already exists in submissions log ({submissions_log})")
    if artifact_id in seen_result_ids:
        reasons.append(f"artifact id already exists in results log ({results_log})")

    run_dir = run_dir_base / artifact_id
    if run_dir.exists():
        reasons.append(f"artifact directory already exists ({run_dir})")

    return reasons


def _build_timestamped_artifact_id(
    *,
    base_artifact_id: str,
    run_dir_base: Path,
    seen_submission_ids: Set[str],
    seen_result_ids: Set[str],
    selected_seen_ids: Set[str],
    submissions_log: Path,
    results_log: Path,
    timestamp_format: str,
) -> str:
    stamp = datetime.now().strftime(timestamp_format)
    stem = f"{base_artifact_id}-{stamp}"

    for idx in range(0, 1000):
        candidate = stem if idx == 0 else f"{stem}-{idx:02d}"
        reasons = _find_duplicate_reasons(
            artifact_id=candidate,
            run_dir_base=run_dir_base,
            seen_submission_ids=seen_submission_ids,
            seen_result_ids=seen_result_ids,
            selected_seen_ids=selected_seen_ids,
            submissions_log=submissions_log,
            results_log=results_log,
        )
        if not reasons:
            return candidate

    raise RuntimeError(
        f"Unable to create unique timestamped artifact id for base `{base_artifact_id}`."
    )


def _validate_rows(rows: Iterable[Dict[str, str]]) -> List[str]:
    errors: List[str] = []
    seen_entry_ids: Dict[str, str] = {}
    seen_artifact_ids: Dict[str, str] = {}

    for row in rows:
        line = row["_line"]
        entry_id = _strip(row.get("entry_id"))
        if not entry_id:
            errors.append(f"line {line}: missing entry_id")
            continue

        for col in REQUIRED_COLUMNS:
            if col in OPTIONAL_EMPTY_COLUMNS:
                continue
            if _strip(row.get(col)) == "":
                errors.append(f"line {line} [{entry_id}]: missing required field `{col}`")

        prev_line = seen_entry_ids.get(entry_id)
        if prev_line is not None:
            errors.append(
                f"line {line} [{entry_id}]: duplicate entry_id (already at line {prev_line})"
            )
        else:
            seen_entry_ids[entry_id] = line

        artifact_id = _strip(row.get("run_artifact_id"))
        if artifact_id:
            prev_artifact_line = seen_artifact_ids.get(artifact_id)
            if prev_artifact_line is not None:
                errors.append(
                    f"line {line} [{entry_id}]: duplicate run_artifact_id `{artifact_id}` "
                    f"(already at line {prev_artifact_line})"
                )
            else:
                seen_artifact_ids[artifact_id] = line

        submit_via = _strip(row.get("submit_via")).lower()
        if submit_via and submit_via not in ALLOWED_SUBMIT_VIA:
            errors.append(
                f"line {line} [{entry_id}]: invalid submit_via `{submit_via}` "
                f"(allowed: {', '.join(sorted(ALLOWED_SUBMIT_VIA))})"
            )

        distill_loss = _strip(row.get("distill_loss")).lower()
        if distill_loss and distill_loss not in ALLOWED_DISTILL_LOSS:
            errors.append(
                f"line {line} [{entry_id}]: invalid distill_loss `{distill_loss}` "
                f"(allowed: {', '.join(sorted(ALLOWED_DISTILL_LOSS))})"
            )

        save_strategy = _strip(row.get("train_save_strategy")).lower()
        if save_strategy and save_strategy not in ALLOWED_TRAIN_SAVE_STRATEGY:
            errors.append(
                f"line {line} [{entry_id}]: invalid train_save_strategy `{save_strategy}` "
                f"(allowed: {', '.join(sorted(ALLOWED_TRAIN_SAVE_STRATEGY))})"
            )
        if save_strategy == "steps" and _strip(row.get("train_save_steps")) == "":
            errors.append(
                f"line {line} [{entry_id}]: train_save_strategy=steps requires `train_save_steps`"
            )

        seed_text = _strip(row.get("seed"))
        if seed_text:
            try:
                int(seed_text)
            except ValueError:
                errors.append(
                    f"line {line} [{entry_id}]: seed must be an integer, got `{seed_text}`"
                )

        if submit_via == "sbatch" and not _nonnull(row.get("submit_script")):
            errors.append(
                f"line {line} [{entry_id}]: submit_via=sbatch requires `submit_script`"
            )
        if submit_via == "python" and not _nonnull(row.get("python_module")):
            errors.append(
                f"line {line} [{entry_id}]: submit_via=python requires `python_module`"
            )

        for col in INT_COLUMNS:
            value = _strip(row.get(col))
            if not value:
                continue
            try:
                int(value)
            except ValueError:
                errors.append(
                    f"line {line} [{entry_id}]: `{col}` must be an integer, got `{value}`"
                )

        for col in FLOAT_COLUMNS:
            value = _strip(row.get(col))
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                errors.append(
                    f"line {line} [{entry_id}]: `{col}` must be numeric, got `{value}`"
                )

        for col in BOOL_COLUMNS:
            value = _strip(row.get(col))
            if not value:
                continue
            try:
                _as_bool(value, default=None)
            except ValueError:
                errors.append(
                    f"line {line} [{entry_id}]: `{col}` must be boolean, got `{value}`"
                )

        if distill_loss == "kld":
            for col in ("distillation_temperature", "distillation_alpha"):
                if _strip(row.get(col)) == "":
                    errors.append(
                        f"line {line} [{entry_id}]: distill_loss=kld requires `{col}`"
                    )

        for token in _split_pipe_list(row.get("extra_overrides")):
            if "=" not in token:
                errors.append(
                    f"line {line} [{entry_id}]: invalid extra override `{token}` (missing '=')"
                )

    return errors


def _selected_rows(
    rows: List[Dict[str, str]],
    ids: List[str],
    enabled_only: bool,
) -> List[Dict[str, str]]:
    by_id = {_strip(r.get("entry_id")): r for r in rows}

    if ids:
        selected: List[Dict[str, str]] = []
        for entry_id in ids:
            row = by_id.get(entry_id)
            if row is None:
                raise ValueError(f"Unknown entry_id: {entry_id}")
            selected.append(row)
    else:
        selected = list(rows)

    if enabled_only:
        selected = [r for r in selected if _as_bool(r.get("enabled"), default=False)]

    return selected


def _build_command(
    row: Dict[str, str],
    artifact_id_override: Optional[str] = None,
) -> List[str]:
    submit_via = _strip(row.get("submit_via")).lower()
    entry_id = _strip(row.get("entry_id"))

    # Forward key registry metadata as CLI overrides so run_distill can emit
    # result rows with stable IDs and planning dimensions.
    metadata_overrides: List[str] = []
    metadata_pairs = [
        ("entry_id", _strip(row.get("entry_id"))),
        ("dataset", _strip(row.get("dataset"))),
        ("split", _strip(row.get("split"))),
        ("seed", _strip(row.get("seed"))),
        ("sampler", _strip(row.get("sampler"))),
        ("teacher_model", _strip(row.get("teacher_model"))),
        ("teacher_variant", _strip(row.get("teacher_variant"))),
        ("student_family", _strip(row.get("student_family"))),
        ("student_arch", _strip(row.get("student_arch"))),
        ("student_target_params", _strip(row.get("student_target_params"))),
        ("distill_loss", _strip(row.get("distill_loss"))),
        ("distill_method", _strip(row.get("distill_method"))),
        (
            "run_artifact_id",
            artifact_id_override
            if artifact_id_override is not None
            else _strip(row.get("run_artifact_id")),
        ),
    ]
    for out_key, value in metadata_pairs:
        if value:
            metadata_overrides.append(f"{out_key}={value}")

    runtime_overrides: List[str] = []
    for col, key in DISTILL_RUNTIME_COLUMN_MAP:
        value = _nonnull(row.get(col))
        if value is None:
            continue
        if col in BOOL_COLUMNS:
            parsed = _as_bool(value, default=None)
            if parsed is None:
                continue
            value = "true" if parsed else "false"
        _append_override(runtime_overrides, key, value)

    for key, row_key in DISTILL_RUNTIME_DERIVED_MAP:
        _append_override(runtime_overrides, key, _nonnull(row.get(row_key)))

    runtime_overrides.extend(_split_pipe_list(row.get("extra_overrides")))

    full_overrides = [*metadata_overrides, *runtime_overrides]

    if submit_via == "python":
        python_module = _nonnull(row.get("python_module"))
        if not python_module:
            raise ValueError(f"[{entry_id}] submit_via=python requires python_module")
        return ["python", "-m", python_module, *full_overrides]

    if submit_via == "sbatch":
        submit_script = _nonnull(row.get("submit_script"))
        if not submit_script:
            raise ValueError(f"[{entry_id}] submit_via=sbatch requires submit_script")
        sbatch_args = shlex.split(_strip(row.get("sbatch_args")))
        has_job_name = any(
            token == "--job-name"
            or token.startswith("--job-name=")
            or token.startswith("-J")
            for token in sbatch_args
        )
        if not has_job_name:
            sbatch_args = [*sbatch_args, f"--job-name={entry_id}"]
        return ["sbatch", *sbatch_args, submit_script, *full_overrides]

    raise ValueError(f"[{entry_id}] unsupported submit_via={submit_via!r}")


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _parse_sbatch_job_id(text: str) -> Optional[str]:
    for line in text.splitlines():
        marker = "Submitted batch job "
        if marker in line:
            return line.split(marker, 1)[1].strip() or None
    return None


def cmd_validate(args: argparse.Namespace) -> int:
    rows = _read_registry(Path(args.registry))
    errors = _validate_rows(rows)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Validation OK: {len(rows)} row(s)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = _read_registry(Path(args.registry))
    errors = _validate_rows(rows)
    if errors and not args.allow_invalid:
        print("Validation failed. Run `validate` for details.", file=sys.stderr)
        return 1

    selected = _selected_rows(
        rows=rows,
        ids=[x.strip() for x in _strip(args.ids).split(",") if x.strip()],
        enabled_only=args.enabled_only,
    )

    for row in selected:
        print(
            ",".join(
                [
                    _strip(row.get("entry_id")),
                    _strip(row.get("enabled")),
                    _strip(row.get("submit_via")),
                    _strip(row.get("dataset")),
                    _strip(row.get("distill_loss")),
                    _strip(row.get("run_artifact_id")),
                ]
            )
        )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    rows = _read_registry(Path(args.registry))
    errors = _validate_rows(rows)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    selected = _selected_rows(
        rows=rows,
        ids=[x.strip() for x in _strip(args.ids).split(",") if x.strip()],
        enabled_only=args.enabled_only,
    )
    if not selected:
        print("No rows selected.", file=sys.stderr)
        return 1

    for row in selected:
        cmd = _build_command(row)
        text = shlex.join(cmd)
        if args.dry_run:
            print(text)
        else:
            # v0 scope is intentionally build-only.
            print(text)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rows = _read_registry(Path(args.registry))
    errors = _validate_rows(rows)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    selected = _selected_rows(
        rows=rows,
        ids=[x.strip() for x in _strip(args.ids).split(",") if x.strip()],
        enabled_only=args.enabled_only,
    )
    if not selected:
        print("No rows selected.", file=sys.stderr)
        return 1

    submissions_log = Path(args.submissions_log)
    results_log = Path(args.results_log)
    run_dir_base = Path(args.run_dir_base)
    seen_submission_ids = _read_jsonl_artifact_ids(submissions_log)
    seen_result_ids = _read_jsonl_artifact_ids(results_log)
    selected_seen_ids: Set[str] = set()
    had_failure = False

    for row in selected:
        entry_id = _strip(row.get("entry_id"))
        submit_via = _strip(row.get("submit_via"))
        base_artifact_id = _resolved_row_artifact_id(row)
        artifact_id = base_artifact_id

        duplicate_reasons = _find_duplicate_reasons(
            artifact_id=artifact_id,
            run_dir_base=run_dir_base,
            seen_submission_ids=seen_submission_ids,
            seen_result_ids=seen_result_ids,
            selected_seen_ids=selected_seen_ids,
            submissions_log=submissions_log,
            results_log=results_log,
        )
        if duplicate_reasons:
            if args.duplicate_policy == "timestamp":
                artifact_id = _build_timestamped_artifact_id(
                    base_artifact_id=base_artifact_id,
                    run_dir_base=run_dir_base,
                    seen_submission_ids=seen_submission_ids,
                    seen_result_ids=seen_result_ids,
                    selected_seen_ids=selected_seen_ids,
                    submissions_log=submissions_log,
                    results_log=results_log,
                    timestamp_format=args.timestamp_format,
                )
                print(
                    (
                        f"[{entry_id}] duplicate artifact id `{base_artifact_id}` "
                        f"detected; using `{artifact_id}`"
                    ),
                    file=sys.stderr,
                )
            elif args.duplicate_policy == "skip":
                print(
                    (
                        f"[{entry_id}] skipping duplicate artifact id `{artifact_id}`: "
                        + "; ".join(duplicate_reasons)
                    ),
                    file=sys.stderr,
                )
                continue
            else:
                print(
                    (
                        f"[{entry_id}] duplicate artifact id `{artifact_id}` blocked: "
                        + "; ".join(duplicate_reasons)
                    ),
                    file=sys.stderr,
                )
                had_failure = True
                if not args.continue_on_error:
                    break
                continue

        selected_seen_ids.add(artifact_id)
        cmd = _build_command(row, artifact_id_override=artifact_id)
        cmd_text = shlex.join(cmd)
        print(cmd_text)

        if args.dry_run:
            continue

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)

        merged_output = "\n".join([_strip(proc.stdout), _strip(proc.stderr)]).strip()
        job_id = _parse_sbatch_job_id(merged_output) if submit_via == "sbatch" else None
        status = "submitted" if proc.returncode == 0 else "failed"

        event: Dict[str, object] = {
            "event_version": 1,
            "submitted_at": _now_iso(),
            "entry_id": entry_id,
            "run_artifact_id": artifact_id,
            "submit_via": submit_via,
            "status": status,
            "return_code": proc.returncode,
            "command": cmd,
            "command_text": cmd_text,
            "job_id": job_id,
        }
        _append_jsonl(submissions_log, event)
        seen_submission_ids.add(artifact_id)

        if proc.returncode != 0:
            had_failure = True
            if not args.continue_on_error:
                break

    return 1 if had_failure else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help="Path to VT2 distillation registry CSV",
    )
    common.add_argument(
        "--ids",
        default="",
        help="Comma-separated entry_id list. Empty means all selected rows.",
    )
    common.add_argument(
        "--enabled-only",
        action="store_true",
        help="Select only rows where enabled=true.",
    )

    parser = argparse.ArgumentParser(
        description="VT2 distillation registry validator, command builder, and runner."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", parents=[common], help="Validate registry CSV")
    p_validate.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list", parents=[common], help="List selected registry entries")
    p_list.add_argument(
        "--allow-invalid",
        action="store_true",
        help="List even if validation fails.",
    )
    p_list.set_defaults(func=cmd_list)

    p_build = sub.add_parser(
        "build",
        parents=[common],
        help="Print resolved commands for selected rows.",
    )
    p_build.add_argument(
        "--dry-run",
        action="store_true",
        help="Compatibility flag for v0 (build already prints only).",
    )
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser(
        "run",
        parents=[common],
        help="Execute selected rows and append submission events.",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    p_run.add_argument(
        "--submissions-log",
        default=DEFAULT_SUBMISSIONS_LOG,
        help="JSONL file to append submission events.",
    )
    p_run.add_argument(
        "--results-log",
        default=DEFAULT_RESULTS_LOG,
        help="JSONL file used to detect already-used run_artifact_id values.",
    )
    p_run.add_argument(
        "--run-dir-base",
        default=DEFAULT_RUN_DIR_BASE,
        help="Base artifact directory used to detect already-existing run directories.",
    )
    p_run.add_argument(
        "--duplicate-policy",
        default="error",
        choices=["error", "skip", "timestamp"],
        help=(
            "How to handle duplicate run_artifact_id values across prior runs: "
            "error (default), skip, or create a timestamp-suffixed artifact id."
        ),
    )
    p_run.add_argument(
        "--timestamp-format",
        default="%Y%m%d_%H%M%S",
        help=(
            "strftime format used when --duplicate-policy=timestamp. "
            "Default: %%Y%%m%%d_%%H%%M%%S"
        ),
    )
    p_run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining rows after a failed command.",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
