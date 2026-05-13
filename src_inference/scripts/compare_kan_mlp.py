#!/usr/bin/env python3
"""Print a compact KAN vs MLP comparison table from gem5 result directories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from stats_sections import parse_stats

ROOT = Path(__file__).resolve().parents[1]


def root_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KAN and MLP gem5 metrics.")
    parser.add_argument(
        "--kan-dir",
        type=Path,
        default=root_path("results", "cache", "1x4x1"),
        help="KAN L1+L2 result directory.",
    )
    parser.add_argument(
        "--mlp-dir",
        type=Path,
        default=root_path("results", "mlp_l1_l2"),
        help="MLP L1+L2 result directory.",
    )
    return parser.parse_args()


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
        "simInsts": stats.get("simInsts", "N/A"),
        "simTicks": stats.get("simTicks", "N/A"),
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
            stats,
            r"l1d-cache.*\.overallMisses::total$|dcache.*\.overallMisses::total$",
        ),
        "L2_misses": sum_matching(
            stats,
            r"l2-cache.*\.overallMisses::total$|l2.*\.overallMisses::total$",
        ),
        "MSE": simout.get("MSE", "N/A"),
        "CHECKSUM": simout.get("CHECKSUM", "N/A"),
    }


def print_table(rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for item in rows:
        print("  ".join(item[header].ljust(widths[header]) for header in headers))


def main() -> int:
    args = parse_args()
    kan_dir = resolve_path(args.kan_dir)
    mlp_dir = resolve_path(args.mlp_dir)
    print_table(
        [
            row("KAN", "L1+L2", kan_dir),
            row("MLP", "L1+L2", mlp_dir),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
