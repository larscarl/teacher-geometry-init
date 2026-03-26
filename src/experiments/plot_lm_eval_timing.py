import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TIMING_CSV_PATH = "experiments/lm_eval_suite_timing.csv"
DEFAULT_OUTPUT_PATH = "experiments/lm_eval_suite_timing_by_model_task.png"


def _load_rows(path: Path, suite_id: str = "") -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Timing CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        rows = [row for row in reader]

    if suite_id:
        rows = [row for row in rows if row.get("suite_id", "") == suite_id]

    rows = [row for row in rows if row.get("status", "") == "success"]
    if not rows:
        scope = f" for suite_id={suite_id}" if suite_id else ""
        raise ValueError(f"No successful timing rows found{scope}.")
    return rows


def _to_matrix(rows: List[Dict[str, str]]) -> tuple[List[str], List[str], np.ndarray]:
    models = sorted({row["model_label"] for row in rows})
    tasks = sorted({row["task"] for row in rows})

    matrix = np.zeros((len(models), len(tasks)), dtype=float)
    model_index = {name: idx for idx, name in enumerate(models)}
    task_index = {name: idx for idx, name in enumerate(tasks)}

    for row in rows:
        m = model_index[row["model_label"]]
        t = task_index[row["task"]]
        matrix[m, t] = float(row["duration_s"]) / 60.0

    return models, tasks, matrix


def _plot(models: List[str], tasks: List[str], minutes: np.ndarray, output_path: Path) -> None:
    totals = minutes.sum(axis=1)
    model_order = list(np.argsort(totals)[::-1])
    task_order = list(np.argsort(minutes.sum(axis=0))[::-1])

    models_sorted = [models[idx] for idx in model_order]
    tasks_sorted = [tasks[idx] for idx in task_order]
    values = minutes[np.ix_(model_order, task_order)]

    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models_sorted))
    bottoms = np.zeros(len(models_sorted), dtype=float)
    cmap = plt.get_cmap("tab20")

    for idx, task in enumerate(tasks_sorted):
        vals = values[:, idx]
        ax1.bar(
            x,
            vals,
            bottom=bottoms,
            label=task,
            color=cmap(idx % 20),
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals

    ax1.set_title("lm-eval Runtime by Model (stacked by task)")
    ax1.set_ylabel("Duration (minutes)")
    ax1.set_xlabel("Model")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_sorted, rotation=18, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax1.set_ylim(0, max(values.sum(axis=1)) * 1.12)

    for idx, total_min in enumerate(values.sum(axis=1)):
        ax1.text(idx, total_min + 1.2, f"{total_min:.1f}m", ha="center", va="bottom", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    heat = ax2.imshow(values.T, aspect="auto", cmap="YlOrRd")
    ax2.set_title("Task Runtime Heatmap")
    ax2.set_xticks(np.arange(len(models_sorted)))
    ax2.set_xticklabels(models_sorted, rotation=18, ha="right")
    ax2.set_yticks(np.arange(len(tasks_sorted)))
    ax2.set_yticklabels(tasks_sorted)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Task")

    for yi in range(values.shape[1]):
        for xi in range(values.shape[0]):
            ax2.text(xi, yi, f"{values[xi, yi]:.1f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(heat, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Duration (minutes)")

    fig.suptitle("Benchmark Runtime Matrix", fontsize=14, y=0.97)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.24, wspace=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot lm-eval suite runtime by model and task.")
    parser.add_argument(
        "--timing-csv-path",
        default=DEFAULT_TIMING_CSV_PATH,
        help="Path to lm_eval_suite_timing.csv",
    )
    parser.add_argument(
        "--suite-id",
        default="",
        help="Optional suite id filter (e.g. cluster_full_model_matrix).",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output image (PNG).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(Path(args.timing_csv_path), suite_id=args.suite_id.strip())
    models, tasks, matrix = _to_matrix(rows)
    _plot(models, tasks, matrix, Path(args.output_path))
    print(f"Wrote plot: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
