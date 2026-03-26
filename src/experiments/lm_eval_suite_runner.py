import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.lm_eval_utils import (
    append_jsonl,
    make_json_safe,
    parse_batch_size,
    parse_limit,
    parse_task_list,
    redact_lm_eval_result,
    run_lm_eval,
)


DEFAULT_RESULTS_PATH = "experiments/lm_eval_suite_results.jsonl"
DEFAULT_TIMING_CSV_PATH = "experiments/lm_eval_suite_timing.csv"

TIMING_FIELDNAMES = [
    "suite_id",
    "mode",
    "model_label",
    "model_ref",
    "task",
    "status",
    "started_at",
    "finished_at",
    "duration_s",
    "device",
    "batch_size",
    "num_fewshot",
    "limit",
    "n_samples_original",
    "n_samples_effective",
    "primary_metric_name",
    "primary_metric_value",
    "error",
]


def _load_env_file(path: Path, *, override: bool = False) -> None:
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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_models(text: str) -> List[Tuple[str, str]]:
    raw = str(text or "")
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("No models provided. Use --models label=model_ref[,label2=ref2].")

    out: List[Tuple[str, str]] = []
    for idx, part in enumerate(parts, start=1):
        if "=" in part:
            label, ref = part.split("=", 1)
            label = label.strip()
            ref = ref.strip()
        else:
            label = f"model_{idx}"
            ref = part.strip()
        if not ref:
            raise ValueError(f"Invalid model entry: {part!r}")
        out.append((label, ref))
    return out


def _first_numeric_metric(metrics: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    preferred = [
        "acc_norm,none",
        "acc,none",
        "exact_match,strict-match",
        "exact_match,none",
        "f1,none",
        "word_perplexity,none",
        "perplexity,none",
        "bits_per_byte,none",
        "byte_perplexity,none",
    ]
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and ",stderr" not in key:
            return key, float(value)

    for key in sorted(metrics.keys()):
        if ",stderr" in key:
            continue
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    return "", None


def _append_timing_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=TIMING_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TIMING_FIELDNAMES})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an lm-eval model/task matrix with per-task timing output "
            "(JSONL + CSV)."
        )
    )
    parser.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated list: label=model_ref "
            "(e.g. gemma_270m=google/gemma-3-270m,llama=meta-llama/Llama-3.2-1B)."
        ),
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help=(
            "Comma-separated task list "
            "(e.g. mmlu_stem,hellaswag,arc_challenge,piqa,winogrande,boolq,gsm8k_cot_llama)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size for lm-eval (int or 'auto').",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu, cuda, cuda:0, mps).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Few-shot examples.",
    )
    parser.add_argument(
        "--limit",
        default="",
        help="Limit examples per task for non-smoke mode (empty means full dataset).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode with a small per-task sample limit.",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=2,
        help="Per-task limit used when --smoke is enabled (default: 2).",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Use model chat template if available.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote code for HF models.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Store model inputs/outputs in lm-eval results payload.",
    )
    parser.add_argument(
        "--suite-id",
        default="",
        help="Optional suite identifier (default: timestamped id).",
    )
    parser.add_argument(
        "--results-path",
        default=DEFAULT_RESULTS_PATH,
        help="Append-only JSONL for detailed per-task results.",
    )
    parser.add_argument(
        "--timing-csv-path",
        default=DEFAULT_TIMING_CSV_PATH,
        help="CSV file with per-task timing summary.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining model/task entries when one fails.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _load_env_file(Path(".env"), override=False)
    hf_token = os.environ.get("HF_TOKEN", "")

    models = _parse_models(args.models)
    tasks = parse_task_list(args.tasks)
    if not tasks:
        raise ValueError("Tasks list is empty. Provide --tasks.")

    batch_size = parse_batch_size(args.batch_size, default="auto")
    if args.smoke:
        if args.smoke_limit <= 0:
            raise ValueError("--smoke-limit must be > 0")
        limit = args.smoke_limit
        mode = "smoke"
    else:
        limit = parse_limit(args.limit, default=None)
        mode = "full"

    suite_id = args.suite_id.strip() or f"lm_eval_suite_{datetime.now():%Y%m%d_%H%M%S}"
    results_path = Path(args.results_path)
    timing_csv_path = Path(args.timing_csv_path)

    had_failure = False
    model_total_duration: Dict[str, float] = {}
    model_success_count: Dict[str, int] = {}
    model_failure_count: Dict[str, int] = {}

    for model_label, model_ref in models:
        for task in tasks:
            started_at = _now_iso()
            t0 = time.perf_counter()
            status = "success"
            error_message = ""

            redacted_results: Dict[str, Any] = {}
            task_metrics: Dict[str, float] = {}
            n_samples_original: Optional[int] = None
            n_samples_effective: Optional[int] = None
            primary_metric_name = ""
            primary_metric_value: Optional[float] = None

            try:
                result = run_lm_eval(
                    model_ref=model_ref,
                    tasks=[task],
                    batch_size=batch_size,
                    device=args.device,
                    num_fewshot=args.num_fewshot,
                    limit=limit,
                    hf_token=hf_token,
                    log_samples=bool(args.log_samples),
                    apply_chat_template=True if args.apply_chat_template else None,
                    trust_remote_code=True if args.trust_remote_code else None,
                )
                redacted_results = redact_lm_eval_result(result)

                task_result = result.get("results", {}).get(task, {})
                if isinstance(task_result, dict):
                    for metric_name, metric_value in task_result.items():
                        if ",stderr" in str(metric_name):
                            continue
                        if isinstance(metric_value, bool):
                            continue
                        if isinstance(metric_value, (int, float)):
                            task_metrics[str(metric_name)] = float(metric_value)
                    primary_metric_name, primary_metric_value = _first_numeric_metric(task_result)

                n_samples = result.get("n-samples", {}).get(task, {})
                if isinstance(n_samples, dict):
                    original = n_samples.get("original")
                    effective = n_samples.get("effective")
                    if isinstance(original, int):
                        n_samples_original = original
                    if isinstance(effective, int):
                        n_samples_effective = effective

            except Exception as exc:
                status = "failed"
                had_failure = True
                error_message = f"{type(exc).__name__}: {exc}"

            finished_at = _now_iso()
            duration_s = time.perf_counter() - t0

            json_row: Dict[str, Any] = {
                "suite_id": suite_id,
                "mode": mode,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_s": duration_s,
                "model_label": model_label,
                "model_ref": model_ref,
                "task": task,
                "device": args.device,
                "batch_size": batch_size,
                "num_fewshot": args.num_fewshot,
                "limit": limit,
                "n_samples_original": n_samples_original,
                "n_samples_effective": n_samples_effective,
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "task_metrics": task_metrics,
                "error": error_message,
                "results": redacted_results,
            }
            append_jsonl(results_path, make_json_safe(json_row))

            csv_row: Dict[str, Any] = {
                "suite_id": suite_id,
                "mode": mode,
                "model_label": model_label,
                "model_ref": model_ref,
                "task": task,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_s": f"{duration_s:.6f}",
                "device": args.device,
                "batch_size": batch_size,
                "num_fewshot": args.num_fewshot,
                "limit": limit if limit is not None else "",
                "n_samples_original": n_samples_original if n_samples_original is not None else "",
                "n_samples_effective": n_samples_effective if n_samples_effective is not None else "",
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": (
                    f"{primary_metric_value:.12g}" if primary_metric_value is not None else ""
                ),
                "error": error_message,
            }
            _append_timing_csv(timing_csv_path, csv_row)

            model_total_duration[model_label] = (
                model_total_duration.get(model_label, 0.0) + duration_s
            )
            if status == "success":
                model_success_count[model_label] = model_success_count.get(model_label, 0) + 1
            else:
                model_failure_count[model_label] = model_failure_count.get(model_label, 0) + 1

            summary = {
                "suite_id": suite_id,
                "status": status,
                "model_label": model_label,
                "task": task,
                "duration_s": round(duration_s, 3),
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
            }
            if error_message:
                summary["error"] = error_message
            print(json.dumps(make_json_safe(summary), ensure_ascii=True))

            if status == "failed" and not args.continue_on_error:
                return 1

    for model_label, total_duration in sorted(model_total_duration.items()):
        model_summary = {
            "suite_id": suite_id,
            "summary_type": "model_total",
            "model_label": model_label,
            "duration_s": round(total_duration, 3),
            "task_successes": model_success_count.get(model_label, 0),
            "task_failures": model_failure_count.get(model_label, 0),
        }
        print(json.dumps(make_json_safe(model_summary), ensure_ascii=True))

    overall_summary = {
        "suite_id": suite_id,
        "summary_type": "suite_total",
        "duration_s": round(sum(model_total_duration.values()), 3),
        "models": len(models),
        "tasks_per_model": len(tasks),
        "had_failure": had_failure,
    }
    print(json.dumps(make_json_safe(overall_summary), ensure_ascii=True))

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
