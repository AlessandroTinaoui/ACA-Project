#!/usr/bin/env python3
"""Plot gem5/workload metrics for KAN model variants and the MLP baseline."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAN_MODELS = ["1x1", "1x2x1", "1x4x1", "1x8x1"]


def root_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparison metrics from gem5 KAN/MLP result directories."
    )
    parser.add_argument(
        "--kan-models",
        nargs="+",
        default=DEFAULT_KAN_MODELS,
        help="KAN model result names under results/cache.",
    )
    parser.add_argument(
        "--mlp-dir",
        type=Path,
        default=root_path("results", "mlp_l1_l2"),
        help="MLP L1+L2 result directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root_path("plots", "model_metrics"),
        help="Directory where CSV and PNG plots are written.",
    )
    return parser.parse_args()


def parse_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.partition("#")[0].split()
            if len(parts) >= 2:
                stats[parts[0]] = parts[1]
    return stats


def parse_simout(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    rx = re.compile(r"^(MSE|MAE|MAX_ABS_ERROR|CHECKSUM)\s*=\s*(\S+)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            match = rx.match(stripped)
            if match:
                metrics[match.group(1)] = match.group(2)
            elif stripped.startswith("N = "):
                metrics["N"] = stripped.split("=", 1)[1].strip()
    return metrics


def first(stats: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        if key in stats:
            return stats[key]
    return "nan"


def sum_matching(stats: dict[str, str], pattern: str) -> str:
    rx = re.compile(pattern)
    total = 0.0
    found = False
    for key, value in stats.items():
        if not rx.search(key):
            continue
        try:
            total += float(value)
            found = True
        except ValueError:
            pass
    return str(total) if found else "nan"


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def as_intish(value: str) -> float:
    return as_float(value)


def model_row(label: str, kind: str, directory: Path) -> dict[str, object]:
    stats = parse_stats(directory / "stats.txt")
    simout = parse_simout(directory / "simout")
    return {
        "model": label,
        "kind": kind,
        "result_dir": str(directory),
        "N": as_intish(simout.get("N", "nan")),
        "simInsts": as_intish(
            first(
                stats,
                [
                    "simInsts",
                    "system.cpu.committedInsts",
                    "board.processor.cores.core.commitStats0.numInsts",
                    "board.processor.cores.core.thread_0.numInsts",
                ],
            )
        ),
        "simTicks": as_intish(first(stats, ["simTicks"])),
        "cycles": as_intish(
            first(stats, ["system.cpu.numCycles", "board.processor.cores.core.numCycles"])
        ),
        "IPC": as_float(
            first(
                stats,
                [
                    "system.cpu.ipc",
                    "board.processor.cores.core.ipc",
                    "board.processor.cores.core.commitStats0.ipc",
                ],
            )
        ),
        "L1D_misses": as_intish(
            sum_matching(
                stats,
                r"l1d-cache.*\.overallMisses::total$|dcache.*\.overallMisses::total$",
            )
        ),
        "L2_misses": as_intish(
            sum_matching(stats, r"l2-cache.*\.overallMisses::total$|l2.*\.overallMisses::total$")
        ),
        "MSE": as_float(simout.get("MSE", "nan")),
        "MAE": as_float(simout.get("MAE", "nan")),
        "MAX_ABS_ERROR": as_float(simout.get("MAX_ABS_ERROR", "nan")),
        "CHECKSUM": as_float(simout.get("CHECKSUM", "nan")),
    }


def valid_rows(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if isinstance(row.get(metric), float) and not math.isnan(row[metric])  # type: ignore[index]
    ]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "model",
        "kind",
        "result_dir",
        "N",
        "simInsts",
        "simTicks",
        "cycles",
        "IPC",
        "L1D_misses",
        "L2_misses",
        "MSE",
        "MAE",
        "MAX_ABS_ERROR",
        "CHECKSUM",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def bar_plot(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    path: Path,
    *,
    log_y: bool = False,
) -> None:
    filtered = valid_rows(rows, metric)
    labels = [str(row["model"]) for row in filtered]
    values = [float(row[metric]) for row in filtered]
    colors = ["#2f6fbb" if row["kind"] == "KAN" else "#c75b39" for row in filtered]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} comparison")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    if log_y:
        ax.set_yscale("log")
    ax.bar_label(bars, labels=[format_value(value) for value in values], padding=3, fontsize=8)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tradeoff_plot(
    rows: list[dict[str, object]],
    x_metric: str,
    y_metric: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    log_y: bool = True,
) -> None:
    filtered = [
        row
        for row in rows
        if not math.isnan(float(row[x_metric])) and not math.isnan(float(row[y_metric]))
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for row in filtered:
        color = "#2f6fbb" if row["kind"] == "KAN" else "#c75b39"
        marker = "o" if row["kind"] == "KAN" else "s"
        ax.scatter(float(row[x_metric]), float(row[y_metric]), color=color, marker=marker, s=75)
        ax.annotate(str(row["model"]), (float(row[x_metric]), float(row[y_metric])), xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def normalized_bar_plot(rows: list[dict[str, object]], baseline: str, path: Path) -> None:
    metrics = ["simInsts", "cycles", "simTicks", "MSE"]
    baseline_row = next((row for row in rows if row["model"] == baseline), None)
    if baseline_row is None:
        return

    labels = [str(row["model"]) for row in rows]
    x_positions = range(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for metric_index, metric in enumerate(metrics):
        base = float(baseline_row[metric])
        if math.isnan(base) or base == 0.0:
            continue
        values = [float(row[metric]) / base for row in rows]
        offsets = [pos + (metric_index - 1.5) * width for pos in x_positions]
        ax.bar(offsets, values, width=width, label=f"{metric} / {baseline}")

    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_title(f"Metrics normalized to {baseline}")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_value(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    if value == 0.0:
        return "0"
    if abs(value) >= 10000:
        return f"{value:.3g}"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4g}"


def main() -> None:
    args = parse_args()
    mlp_dir = resolve_path(args.mlp_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        model_row(f"KAN {model}", "KAN", root_path("results", "cache", model))
        for model in args.kan_models
    ]
    rows.append(model_row("MLP 24x24", "MLP", mlp_dir))

    write_csv(rows, out_dir / "metrics.csv")

    bar_plot(rows, "MSE", "MSE (log scale)", out_dir / "mse.png", log_y=True)
    bar_plot(rows, "cycles", "Clock cycles", out_dir / "cycles.png")
    bar_plot(rows, "simTicks", "simTicks", out_dir / "sim_ticks.png")
    bar_plot(rows, "simInsts", "Simulated instructions", out_dir / "instructions.png")
    bar_plot(rows, "IPC", "IPC", out_dir / "ipc.png")
    bar_plot(rows, "L1D_misses", "L1D misses", out_dir / "l1d_misses.png")
    bar_plot(rows, "L2_misses", "L2 misses", out_dir / "l2_misses.png")
    tradeoff_plot(
        rows,
        "cycles",
        "MSE",
        "Clock cycles",
        "MSE",
        out_dir / "mse_vs_cycles.png",
    )
    tradeoff_plot(
        rows,
        "simInsts",
        "MSE",
        "Simulated instructions",
        "MSE",
        out_dir / "mse_vs_instructions.png",
    )
    normalized_bar_plot(rows, "MLP 24x24", out_dir / "normalized_to_mlp.png")

    print(f"Wrote {out_dir / 'metrics.csv'}")
    for path in sorted(out_dir.glob("*.png")):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
