#!/usr/bin/env python3
"""Print selected gem5 and workload metrics for the MLP L1+L2 run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS = ROOT / "results" / "mlp_l1_l2" / "stats.txt"
DEFAULT_SIMOUT = ROOT / "results" / "mlp_l1_l2" / "simout"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect selected metrics from MLP gem5 stats.txt."
    )
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--simout", type=Path, default=DEFAULT_SIMOUT)
    return parser.parse_args()


def parse_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            before_comment = line.partition("#")[0]
            parts = before_comment.split()
            if len(parts) >= 2:
                stats[parts[0]] = parts[1]
    return stats


def get_first(stats: dict[str, str], keys: list[str]) -> str:
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


def parse_workload_metrics(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics

    rx = re.compile(r"^(MSE|MAE|MAX_ABS_ERROR|CHECKSUM)\s*=\s*(\S+)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = rx.match(line.strip())
            if match:
                metrics[match.group(1)] = match.group(2)
            elif line.startswith("N = "):
                metrics["N"] = line.split("=", 1)[1].strip()
            elif line.strip() == "DONE":
                metrics["DONE"] = "yes"
    return metrics


def print_row(name: str, value: str) -> None:
    print(f"{name:32s} {value}")


def main() -> int:
    args = parse_args()
    stats_path = resolve_path(args.stats)
    simout_path = resolve_path(args.simout)
    stats = parse_stats(stats_path)
    workload = parse_workload_metrics(simout_path)

    if not stats:
        print(f"Warning: stats file not found or empty: {stats_path}")
    if not workload:
        print(f"Warning: workload output not found or empty: {simout_path}")

    print("MLP L1+L2 metrics")
    print_row("stats", str(stats_path))
    print_row("simout", str(simout_path))
    print_row("N", workload.get("N", "N/A"))
    print_row("simTicks", stats.get("simTicks", "N/A"))
    print_row("simInsts", stats.get("simInsts", "N/A"))
    print_row("simOps", stats.get("simOps", "N/A"))
    print_row("hostSeconds", stats.get("hostSeconds", "N/A"))
    print_row("hostTickRate", stats.get("hostTickRate", "N/A"))
    print_row(
        "system.cpu.numCycles",
        get_first(
            stats,
            ["system.cpu.numCycles", "board.processor.cores.core.numCycles"],
        ),
    )
    print_row(
        "system.cpu.committedInsts",
        get_first(
            stats,
            [
                "system.cpu.committedInsts",
                "board.processor.cores.core.commitStats0.numInsts",
                "board.processor.cores.core.thread_0.numInsts",
            ],
        ),
    )
    print_row(
        "system.cpu.ipc",
        get_first(
            stats,
            [
                "system.cpu.ipc",
                "board.processor.cores.core.ipc",
                "board.processor.cores.core.commitStats0.ipc",
            ],
        ),
    )
    print_row("cache hits", sum_matching(stats, r"\.overallHits::total$"))
    print_row("cache misses", sum_matching(stats, r"\.overallMisses::total$"))
    print_row(
        "L1I misses",
        sum_matching(stats, r"l1i-cache.*\.overallMisses::total$|icache.*\.overallMisses::total$"),
    )
    print_row(
        "L1D misses",
        sum_matching(stats, r"l1d-cache.*\.overallMisses::total$|dcache.*\.overallMisses::total$"),
    )
    print_row(
        "L2 misses",
        sum_matching(stats, r"l2-cache.*\.overallMisses::total$|l2.*\.overallMisses::total$"),
    )
    print_row("MSE", workload.get("MSE", "N/A"))
    print_row("MAE", workload.get("MAE", "N/A"))
    print_row("MAX_ABS_ERROR", workload.get("MAX_ABS_ERROR", "N/A"))
    print_row("CHECKSUM", workload.get("CHECKSUM", "N/A"))
    print_row("DONE", workload.get("DONE", "N/A"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
