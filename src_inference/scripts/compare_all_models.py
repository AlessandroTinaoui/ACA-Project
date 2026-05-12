#!/usr/bin/env python3
"""Print a compact comparison table for multiple KAN runs plus one MLP run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Compare all KAN runs and the MLP run.")
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
    return "N/A"


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
    if not found:
        return "N/A"
    return str(int(total)) if total.is_integer() else f"{total:.9g}"


def row(model: str, config: str, directory: Path) -> dict[str, str]:
    stats = parse_stats(directory / "stats.txt")
    simout = parse_simout(directory / "simout")
    return {
        "Model": model,
        "Config": config,
        "N": simout.get("N", "N/A"),
        "simInsts": first(
            stats,
            [
                "simInsts",
                "system.cpu.committedInsts",
                "board.processor.cores.core.commitStats0.numInsts",
                "board.processor.cores.core.thread_0.numInsts",
            ],
        ),
        "simTicks": first(stats, ["simTicks"]),
        "cycles": first(
            stats,
            ["system.cpu.numCycles", "board.processor.cores.core.numCycles"],
        ),
        "IPC": first(
            stats,
            [
                "system.cpu.ipc",
                "board.processor.cores.core.ipc",
                "board.processor.cores.core.commitStats0.ipc",
            ],
        ),
        "L1D_misses": sum_matching(
            stats, r"l1d-cache.*\.overallMisses::total$|dcache.*\.overallMisses::total$"
        ),
        "L2_misses": sum_matching(
            stats, r"l2-cache.*\.overallMisses::total$|l2.*\.overallMisses::total$"
        ),
        "MSE": simout.get("MSE", "N/A"),
        "CHECKSUM": simout.get("CHECKSUM", "N/A"),
    }


def print_table(rows: list[dict[str, str]]) -> None:
    headers = [
        "Model",
        "Config",
        "N",
        "simInsts",
        "simTicks",
        "cycles",
        "IPC",
        "L1D_misses",
        "L2_misses",
        "MSE",
        "CHECKSUM",
    ]
    widths = {
        header: max(len(header), *(len(row.get(header, "N/A")) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for item in rows:
        print("  ".join(item.get(header, "N/A").ljust(widths[header]) for header in headers))


def main() -> None:
    args = parse_args()
    mlp_dir = resolve_path(args.mlp_dir)
    rows = [
        row(f"KAN {model}", "L1+L2", root_path("results", "cache", model))
        for model in args.kan_models
    ]
    rows.append(row("MLP 24x24", "L1+L2", mlp_dir))
    print_table(rows)


if __name__ == "__main__":
    main()
