import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


LimitValue = Optional[Union[int, float]]
BatchSizeValue = Union[int, str]
REDACTED_VALUE = "***REDACTED***"
SENSITIVE_KEYS = {
    "token",
    "hf_token",
    "api_key",
    "access_token",
    "auth_token",
    "authorization",
}


def parse_task_list(text: str) -> List[str]:
    raw = str(text or "")
    parts = raw.replace("|", ",").split(",")
    return [chunk.strip() for chunk in parts if chunk.strip()]


def parse_batch_size(text: str, *, default: BatchSizeValue = "auto") -> BatchSizeValue:
    raw = str(text or "").strip()
    if not raw:
        return default

    lowered = raw.lower()
    if lowered == "auto" or lowered.startswith("auto:"):
        return raw

    try:
        value = int(raw)
    except ValueError:
        return default

    if value <= 0:
        return default
    return value


def parse_limit(text: str, *, default: LimitValue = None) -> LimitValue:
    raw = str(text or "").strip()
    if not raw:
        return default

    lowered = raw.lower()
    if lowered in {"none", "full", "all", "-1"}:
        return None

    try:
        int_value = int(raw)
        if int_value <= 0:
            return None
        return int_value
    except ValueError:
        pass

    try:
        float_value = float(raw)
    except ValueError:
        return default

    if float_value <= 0:
        return None
    return float_value


def run_lm_eval(
    *,
    model_ref: str,
    tasks: Sequence[str],
    batch_size: BatchSizeValue = "auto",
    device: str = "cpu",
    num_fewshot: int = 0,
    limit: LimitValue = None,
    hf_token: str = "",
    log_samples: bool = False,
    apply_chat_template: Optional[bool] = None,
    trust_remote_code: Optional[bool] = None,
) -> Dict[str, Any]:
    try:
        import lm_eval
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "lm-eval is required but not installed. Install with `uv sync` "
            "(dependency: lm-eval[hf])."
        ) from exc

    model_args: Dict[str, Any] = {
        "pretrained": str(model_ref),
    }
    if hf_token:
        model_args["token"] = hf_token
    if trust_remote_code is not None:
        model_args["trust_remote_code"] = bool(trust_remote_code)

    kwargs: Dict[str, Any] = {
        "model": "hf",
        "model_args": model_args,
        "tasks": list(tasks),
        "num_fewshot": int(num_fewshot),
        "batch_size": batch_size,
        "device": str(device),
        "limit": limit,
        "log_samples": bool(log_samples),
    }
    if apply_chat_template is not None:
        kwargs["apply_chat_template"] = bool(apply_chat_template)

    return lm_eval.simple_evaluate(**kwargs)


def flatten_lm_eval_scalars(result: Dict[str, Any], *, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    task_results = result.get("results", {})
    if not isinstance(task_results, dict):
        return out

    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            if ",stderr" in str(metric_name):
                continue
            if isinstance(metric_value, bool):
                continue
            if not isinstance(metric_value, (int, float)):
                continue
            out[f"{prefix}/{task_name}/{metric_name}"] = float(metric_value)

    return out


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in SENSITIVE_KEYS:
                out[str(key)] = REDACTED_VALUE
            else:
                out[str(key)] = _redact_sensitive(item)
        return out
    if isinstance(value, list):
        return [_redact_sensitive(v) for v in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(v) for v in value]
    return value


def redact_lm_eval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return _redact_sensitive(result)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) or value is None or isinstance(value, str):
        return value

    # numpy scalar support without hard dependency
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass

    return str(value)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(make_json_safe(payload), ensure_ascii=True) + "\n")
