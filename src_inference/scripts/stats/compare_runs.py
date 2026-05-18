#!/usr/bin/env python3
"""Build a compact markdown comparison for fp32/PTQ/true-int runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_METRICS = ("mse", "rmse", "mae", "r2", "accuracy_within_tolerance")
SIMOUT_MODEL_METRICS = ("mse", "rmse", "mae")
SYSTEM_METRICS = (
    "simulated instructions",
    "cpu cycles",
    "IPC",
    "CPI",
    "host runtime",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fp32, PTQ, and true-int runs.")
    parser.add_argument("--fp32-metrics", type=Path, required=True)
    parser.add_argument("--fp32-summary", type=Path, required=True)
    parser.add_argument("--q16-metrics", type=Path, required=True)
    parser.add_argument("--q16-summary", type=Path, required=True)
    parser.add_argument("--q8-metrics", type=Path, required=True)
    parser.add_argument("--q8-summary", type=Path, required=True)
    parser.add_argument("--fp32-simout", type=Path)
    parser.add_argument("--q16-simout", type=Path)
    parser.add_argument("--q8-simout", type=Path)
    parser.add_argument("--ti16-summary", type=Path)
    parser.add_argument("--ti8-summary", type=Path)
    parser.add_argument("--ti16-simout", type=Path)
    parser.add_argument("--ti8-simout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_model_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing test metrics in {path}")
    return {key: float(test[key]) for key in MODEL_METRICS}


def load_summary_metrics(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip().split("|")[1:-1]]
        if len(parts) < 2:
            continue
        key, value = parts[0], parts[1]
        if key in SYSTEM_METRICS:
            values[key] = value
    return values


def load_simout_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    key_map = {
        "MSE": "mse",
        "RMSE": "rmse",
        "MAE": "mae",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        raw_key, raw_value = line.split(" = ", 1)
        key = raw_key.strip()
        if key in key_map:
            values[key_map[key]] = float(raw_value.strip())
    missing = [key for key in SIMOUT_MODEL_METRICS if key not in values]
    if missing:
        raise ValueError(f"Missing simout metrics {missing} in {path}")
    return values


def model_table(
    fp32: dict[str, float],
    q16: dict[str, float],
    q8: dict[str, float],
    ti16: dict[str, float] | None = None,
    ti8: dict[str, float] | None = None,
) -> str:
    columns = ["fp32", "q16", "q8"]
    if ti16 is not None:
        columns.append("ti16")
    if ti8 is not None:
        columns.append("ti8")
    rows = [
        "| metrica | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for key in MODEL_METRICS:
        values = [f"{fp32[key]:.9g}", f"{q16[key]:.9g}", f"{q8[key]:.9g}"]
        if ti16 is not None:
            values.append(f"{ti16[key]:.9g}" if key in ti16 else "n/d")
        if ti8 is not None:
            values.append(f"{ti8[key]:.9g}" if key in ti8 else "n/d")
        rows.append(f"| {key} | " + " | ".join(values) + " |")
    return "\n".join(rows)


def system_table(columns: list[tuple[str, dict[str, str]]]) -> str:
    header = "| metrica | " + " | ".join(label for label, _ in columns) + " |"
    separator = "| --- | " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for key in SYSTEM_METRICS:
        values = " | ".join(metrics.get(key, "n/d") for _, metrics in columns)
        rows.append(f"| {key} | {values} |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    fp32_model = load_model_metrics(args.fp32_metrics)
    q16_model = load_model_metrics(args.q16_metrics)
    q8_model = load_model_metrics(args.q8_metrics)
    if args.fp32_simout:
        fp32_model.update(load_simout_metrics(args.fp32_simout))
    if args.q16_simout:
        q16_model.update(load_simout_metrics(args.q16_simout))
    if args.q8_simout:
        q8_model.update(load_simout_metrics(args.q8_simout))

    fp32_system = load_summary_metrics(args.fp32_summary)
    q16_system = load_summary_metrics(args.q16_summary)
    q8_system = load_summary_metrics(args.q8_summary)
    ti16_system = load_summary_metrics(args.ti16_summary) if args.ti16_summary else {}
    ti8_system = load_summary_metrics(args.ti8_summary) if args.ti8_summary else {}
    ti16_model = load_simout_metrics(args.ti16_simout) if args.ti16_simout else None
    ti8_model = load_simout_metrics(args.ti8_simout) if args.ti8_simout else None

    system_columns: list[tuple[str, dict[str, str]]] = [
        ("fp32", fp32_system),
        ("q16", q16_system),
        ("q8", q8_system),
    ]
    if args.ti16_summary:
        system_columns.append(("ti16", ti16_system))
    if args.ti8_summary:
        system_columns.append(("ti8", ti8_system))

    source_lines = [
        "## Sources",
        f"- fp32 metrics: `{args.fp32_metrics}`",
        f"- fp32 summary: `{args.fp32_summary}`",
        f"- q16 metrics: `{args.q16_metrics}`",
        f"- q16 summary: `{args.q16_summary}`",
        f"- q8 metrics: `{args.q8_metrics}`",
        f"- q8 summary: `{args.q8_summary}`",
    ]
    if args.fp32_simout:
        source_lines.append(f"- fp32 simout: `{args.fp32_simout}`")
    if args.q16_simout:
        source_lines.append(f"- q16 simout: `{args.q16_simout}`")
    if args.q8_simout:
        source_lines.append(f"- q8 simout: `{args.q8_simout}`")
    if args.ti16_summary:
        source_lines.append(f"- ti16 summary: `{args.ti16_summary}`")
    if args.ti8_summary:
        source_lines.append(f"- ti8 summary: `{args.ti8_summary}`")
    if args.ti16_simout:
        source_lines.append(f"- ti16 simout: `{args.ti16_simout}`")
    if args.ti8_simout:
        source_lines.append(f"- ti8 simout: `{args.ti8_simout}`")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(
            [
                "# fp32 vs PTQ vs true-int",
                "",
                "## Model metrics",
                model_table(fp32_model, q16_model, q8_model, ti16_model, ti8_model),
                "",
                "## gem5 summary",
                system_table(system_columns),
                "",
                *source_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Comparison written to: {args.out}")


if __name__ == "__main__":
    main()
