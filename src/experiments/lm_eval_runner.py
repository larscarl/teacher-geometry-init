import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.lm_eval_utils import (
    append_jsonl,
    flatten_lm_eval_scalars,
    make_json_safe,
    parse_batch_size,
    parse_limit,
    parse_task_list,
    redact_lm_eval_result,
    run_lm_eval,
)


DEFAULT_RESULTS_PATH = "experiments/lm_eval_results.jsonl"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lm-eval on one or more models and append JSONL results."
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list: label=model_ref (e.g. base=gpt2,student=/path).",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated task list (e.g. hellaswag,arc_easy).",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size for lm-eval (int or 'auto').",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run evaluation on (cpu, cuda, mps, cuda:0).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples.",
    )
    parser.add_argument(
        "--limit",
        default="",
        help="Limit examples per task (int or float). Empty means full.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Store model inputs/outputs in the lm-eval results.",
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
        "--results-path",
        default=DEFAULT_RESULTS_PATH,
        help="Append-only JSONL file for lm-eval results.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _load_env_file(Path(".env"), override=False)
    hf_token = os.environ.get("HF_TOKEN", "")

    tasks = parse_task_list(args.tasks)
    if not tasks:
        raise ValueError("Tasks list is empty. Provide --tasks.")

    batch_size = parse_batch_size(args.batch_size, default="auto")
    limit = parse_limit(args.limit, default=None)
    model_entries = _parse_models(args.models)
    results_path = Path(args.results_path)

    for label, model_ref in model_entries:
        results = run_lm_eval(
            model_ref=model_ref,
            tasks=tasks,
            batch_size=batch_size,
            device=args.device,
            num_fewshot=args.num_fewshot,
            limit=limit,
            hf_token=hf_token,
            log_samples=bool(args.log_samples),
            apply_chat_template=True if args.apply_chat_template else None,
            trust_remote_code=True if args.trust_remote_code else None,
        )

        payload = {
            "created_at": _now_iso(),
            "label": label,
            "model_ref": model_ref,
            "tasks": tasks,
            "num_fewshot": args.num_fewshot,
            "batch_size": batch_size,
            "device": args.device,
            "limit": limit,
            "log_samples": bool(args.log_samples),
            "apply_chat_template": bool(args.apply_chat_template),
            "trust_remote_code": bool(args.trust_remote_code),
            "results": redact_lm_eval_result(results),
        }
        append_jsonl(results_path, payload)

        summary = flatten_lm_eval_scalars(results, prefix=f"lm_eval/{label}")
        print(json.dumps(make_json_safe(summary), ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
