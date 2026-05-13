#!/usr/bin/env python3
"""Helpers for reading one gem5 stats dump section from stats.txt."""

from __future__ import annotations

from pathlib import Path

BEGIN_STATS = "---------- Begin Simulation Statistics"
END_STATS = "---------- End Simulation Statistics"


def _parse_stat_line(line: str) -> tuple[str, str] | None:
    parts = line.partition("#")[0].split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _select_section(sections: list[dict[str, str]], section: str) -> dict[str, str]:
    if not sections:
        return {}
    if section == "first":
        return sections[0]
    if section == "last":
        return sections[-1]
    raise ValueError(f"Unsupported stats section: {section}")


def parse_stats(path: Path, section: str = "first") -> dict[str, str]:
    """Parse stats.txt and return the requested dump section.

    gem5 writes one section for each explicit m5_dump_stats call and another
    one at final simulation exit. The inference benchmarks use the first
    section for the clean, reset-to-dump measurement window.
    """
    if not path.exists():
        return {}

    sections: list[dict[str, str]] = []
    unmarked_stats: dict[str, str] = {}
    current: dict[str, str] = {}
    saw_section_marker = False
    in_section = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(BEGIN_STATS):
                saw_section_marker = True
                current = {}
                in_section = True
                continue

            if stripped.startswith(END_STATS):
                if in_section:
                    sections.append(current)
                current = {}
                in_section = False
                continue

            parsed = _parse_stat_line(line)
            if parsed is None:
                continue

            name, value = parsed
            if saw_section_marker:
                if in_section:
                    current[name] = value
            else:
                unmarked_stats[name] = value

    if in_section:
        sections.append(current)
    if not saw_section_marker:
        sections.append(unmarked_stats)

    return _select_section(sections, section)
