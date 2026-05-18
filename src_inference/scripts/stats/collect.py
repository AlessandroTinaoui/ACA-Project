#!/usr/bin/env python3
"""
gem5_report_data_only.py

Genera un report Markdown/HTML/JSON a partire da un file stats.txt di gem5.
Il report contiene solo contatori presenti nel file e metriche derivate con formula esplicita.
Supporta opzionalmente un config.json prodotto da gem5.

Esempi:
    python gem5_report_data_only.py stats.txt --out report.md
    python gem5_report_data_only.py stats.txt --format html --out report.html
    python gem5_report_data_only.py stats.txt --format json --out report.json
    python gem5_report_data_only.py stats.txt --config config.json --out report.md
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
ROOT = Path(__file__).resolve().parents[2]
BEGIN_STATS = "---------- Begin Simulation Statistics"
END_STATS = "---------- End Simulation Statistics"


@dataclass
class StatEntry:
    name: str
    value: Any
    raw_value: str
    description: str = ""
    unit: str = ""


class Gem5Stats:
    def __init__(self, entries: Dict[str, StatEntry]) -> None:
        self.entries = entries

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        entry = self.entries.get(key)
        if entry is None:
            return default
        if isinstance(entry.value, (int, float)):
            return float(entry.value)
        return default

    def get_raw(self, key: str, default: str = "n/d") -> str:
        entry = self.entries.get(key)
        return entry.raw_value if entry else default

    def first_existing(self, keys: Sequence[str]) -> Optional[Tuple[str, float]]:
        for key in keys:
            value = self.get(key)
            if value is not None:
                return key, value
        return None

    def find(self, pattern: str) -> Dict[str, float]:
        rx = re.compile(pattern)
        out: Dict[str, float] = {}
        for name, entry in self.entries.items():
            if rx.search(name) and isinstance(entry.value, (int, float)):
                out[name] = float(entry.value)
        return out


def parse_number(token: str) -> Any:
    token = token.strip()
    if not NUMBER_RE.match(token):
        return token
    try:
        value = float(token)
    except ValueError:
        return token
    if math.isfinite(value) and value.is_integer() and not any(c in token for c in ".eE"):
        try:
            return int(token)
        except ValueError:
            return value
    return value


def split_description(comment: str) -> Tuple[str, str]:
    comment = comment.strip()
    if comment.endswith(")") and "(" in comment:
        idx = comment.rfind("(")
        unit = comment[idx + 1 : -1].strip()
        desc = comment[:idx].strip()
        return desc, unit
    return comment, ""


def select_stats_section(sections: List[Dict[str, StatEntry]], section: str) -> Dict[str, StatEntry]:
    if not sections:
        return {}
    if section == "first":
        return sections[0]
    if section == "last":
        return sections[-1]
    raise ValueError(f"Sezione stats non supportata: {section}")


def parse_stats(path: Path, section: str = "first") -> Gem5Stats:
    sections: List[Dict[str, StatEntry]] = []
    unmarked_entries: Dict[str, StatEntry] = {}
    current_entries: Dict[str, StatEntry] = {}
    saw_section_marker = False
    in_section = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith(BEGIN_STATS):
                saw_section_marker = True
                current_entries = {}
                in_section = True
                continue
            if stripped.startswith(END_STATS):
                if in_section:
                    sections.append(current_entries)
                current_entries = {}
                in_section = False
                continue
            if not stripped or stripped.startswith("----------"):
                continue
            before_comment, sep, comment = line.partition("#")
            parts = before_comment.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            raw_value = parts[1]
            desc, unit = split_description(comment) if sep else ("", "")
            target_entries = current_entries if saw_section_marker else unmarked_entries
            if saw_section_marker and not in_section:
                continue
            target_entries[name] = StatEntry(
                name=name,
                value=parse_number(raw_value),
                raw_value=raw_value,
                description=desc,
                unit=unit,
            )

    if in_section:
        sections.append(current_entries)
    if not saw_section_marker:
        sections.append(unmarked_entries)

    return Gem5Stats(select_stats_section(sections, section))


def load_config(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Errore lettura config.json: {exc}", file=sys.stderr)
        return None


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def pct(num: Optional[float], den: Optional[float]) -> Optional[float]:
    value = safe_div(num, den)
    return None if value is None else value * 100.0


def ticks_to_ns(ticks: Optional[float], sim_freq: Optional[float]) -> Optional[float]:
    if ticks is None or sim_freq is None or sim_freq == 0:
        return None
    return ticks / sim_freq * 1e9


def ghz_from_clock(sim_freq: Optional[float], clock_period_ticks: Optional[float]) -> Optional[float]:
    if sim_freq is None or clock_period_ticks is None or clock_period_ticks == 0:
        return None
    return sim_freq / clock_period_ticks / 1e9


def fmt_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/d"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/d"
    abs_v = abs(float(value))
    if abs_v >= 1_000_000_000:
        return f"{value/1_000_000_000:.{digits}f}B"
    if abs_v >= 1_000_000:
        return f"{value/1_000_000:.{digits}f}M"
    if abs_v >= 1_000:
        return f"{value/1_000:.{digits}f}K"
    if abs_v != 0 and abs_v < 0.001:
        return f"{value:.{digits}e}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def fmt_percent(value: Optional[float]) -> str:
    return "n/d" if value is None else f"{value:.3f}%"


def fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/d"
    if value >= 1:
        return f"{value:.6f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.6f} ms"
    if value >= 1e-6:
        return f"{value * 1e6:.6f} us"
    return f"{value * 1e9:.6f} ns"


def bytes_to_human(value: Optional[float]) -> str:
    if value is None:
        return "n/d"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(value)
    idx = 0
    while abs(v) >= 1024 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    return f"{v:.3f} {units[idx]}"


def short_name(name: str, max_len: int = 72) -> str:
    if len(name) <= max_len:
        return name
    parts = name.split(".")
    if len(parts) > 3:
        candidate = "..." + ".".join(parts[-3:])
        if len(candidate) <= max_len:
            return candidate
    return "..." + name[-max_len + 3 :]


def value_or_nd(item: Optional[Tuple[str, float]]) -> Optional[float]:
    return item[1] if item else None


def source_or_nd(item: Optional[Tuple[str, float]]) -> str:
    return item[0] if item else "n/d"


def metric_row(metric: str, value: Optional[float], formatted: str, source: str, unit: str = "") -> Dict[str, str]:
    return {
        "metrica": metric,
        "valore": formatted,
        "valore_raw": "n/d" if value is None else str(value),
        "unità": unit,
        "origine": source,
    }


def raw_row(stats: Gem5Stats, metric: str, key: str, formatter=None, unit: str = "") -> Dict[str, str]:
    value = stats.get(key)
    if formatter is None:
        formatted = fmt_num(value)
    else:
        formatted = formatter(value)
    return metric_row(metric, value, formatted, key, unit)


def derived_row(metric: str, value: Optional[float], formatted: str, formula: str, unit: str = "") -> Dict[str, str]:
    return metric_row(metric, value, formatted, formula, unit)


def extract_instruction_mix(stats: Gem5Stats, limit: int) -> List[Dict[str, str]]:
    candidates = stats.find(r"\.commitStats0\.committedInstType::[^:]+$")
    if not candidates:
        candidates = stats.find(r"\.issuedInstType_0::[^:]+$")
    total = None
    total_key = "n/d"
    for key, value in candidates.items():
        if key.endswith("::total"):
            total = value
            total_key = key
            break
    items = []
    for key, value in candidates.items():
        cls = key.split("::")[-1]
        if cls == "total" or value <= 0:
            continue
        items.append((cls, key, value, pct(value, total)))
    items.sort(key=lambda x: x[2], reverse=True)
    rows = []
    for cls, key, count, share in items[:limit]:
        rows.append({
            "classe": cls,
            "conteggio": fmt_num(count),
            "conteggio_raw": str(count),
            "% totale": fmt_percent(share),
            "origine": f"{key}; totale={total_key}",
        })
    return rows


def extract_cache_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    cache_names = set()
    suffixes = [
        "overallHits::total",
        "overallMisses::total",
        "overallHitRate::total",
        "demandHits::total",
        "demandMisses::total",
        "demandHitRate::total",
        "ReadReq.hits::total",
        "ReadReq.misses::total",
        "WriteReq.hits::total",
        "WriteReq.misses::total",
    ]
    for name in stats.entries:
        for suffix in suffixes:
            tail = "." + suffix
            if name.endswith(tail):
                cache_names.add(name[: -len(tail)])

    rows: List[Dict[str, str]] = []
    for cache in sorted(cache_names):
        hits = stats.get(f"{cache}.overallHits::total")
        misses = stats.get(f"{cache}.overallMisses::total")
        hit_rate = stats.get(f"{cache}.overallHitRate::total")
        if hit_rate is not None and hit_rate <= 1.0:
            hit_rate *= 100.0
        if hit_rate is None:
            hit_rate = pct(hits, (hits or 0) + (misses or 0)) if hits is not None or misses is not None else None

        demand_hits = stats.get(f"{cache}.demandHits::total")
        demand_misses = stats.get(f"{cache}.demandMisses::total")
        demand_hit_rate = stats.get(f"{cache}.demandHitRate::total")
        if demand_hit_rate is not None and demand_hit_rate <= 1.0:
            demand_hit_rate *= 100.0
        if demand_hit_rate is None:
            demand_hit_rate = pct(demand_hits, (demand_hits or 0) + (demand_misses or 0)) if demand_hits is not None or demand_misses is not None else None

        rows.append({
            "cache": short_name(cache),
            "hits": fmt_num(hits),
            "misses": fmt_num(misses),
            "hit_rate": fmt_percent(hit_rate),
            "demand_hits": fmt_num(demand_hits),
            "demand_misses": fmt_num(demand_misses),
            "demand_hit_rate": fmt_percent(demand_hit_rate),
        })
    return rows


def build_general_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    sim_freq = stats.get("simFreq")
    clock = stats.get("board.clk_domain.clock")
    freq_ghz = ghz_from_clock(sim_freq, clock)
    return [
        raw_row(stats, "simSeconds", "simSeconds", fmt_seconds, "s"),
        raw_row(stats, "simTicks", "simTicks", fmt_num, "ticks"),
        raw_row(stats, "finalTick", "finalTick", fmt_num, "ticks"),
        raw_row(stats, "simFreq", "simFreq", fmt_num, "ticks/s"),
        raw_row(stats, "hostSeconds", "hostSeconds", fmt_seconds, "s"),
        raw_row(stats, "hostTickRate", "hostTickRate", lambda v: f"{fmt_num(v)} ticks/s", "ticks/s"),
        raw_row(stats, "hostMemory", "hostMemory", bytes_to_human, "B"),
        raw_row(stats, "simInsts", "simInsts", fmt_num, "count"),
        raw_row(stats, "simOps", "simOps", fmt_num, "count"),
        raw_row(stats, "hostInstRate", "hostInstRate", lambda v: f"{fmt_num(v)} inst/s", "inst/s"),
        raw_row(stats, "hostOpRate", "hostOpRate", lambda v: f"{fmt_num(v)} op/s", "op/s"),
        derived_row("clock stimato", freq_ghz, "n/d" if freq_ghz is None else f"{freq_ghz:.6f} GHz", "simFreq / board.clk_domain.clock", "GHz"),
    ]


def build_cpu_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    sim_insts = stats.get("simInsts")
    num_cycles = stats.first_existing([
        "board.processor.cores.core.numCycles",
        "system.cpu.numCycles",
    ])
    ipc = stats.first_existing([
        "board.processor.cores.core.ipc",
        "board.processor.cores.core.commitStats0.ipc",
        "system.cpu.ipc",
    ])
    cpi = stats.first_existing([
        "board.processor.cores.core.cpi",
        "board.processor.cores.core.commitStats0.cpi",
        "system.cpu.cpi",
    ])
    committed = stats.first_existing([
        "board.processor.cores.core.commitStats0.numInsts",
        "board.processor.cores.core.thread_0.numInsts",
    ])
    committed_ops = stats.first_existing([
        "board.processor.cores.core.commitStats0.numOps",
        "board.processor.cores.core.thread_0.numOps",
    ])
    fetched = stats.first_existing([
        "board.processor.cores.core.fetchStats0.numInsts",
    ])
    executed = stats.first_existing([
        "board.processor.cores.core.executeStats0.numInsts",
    ])
    busy = stats.first_existing([
        "board.processor.cores.core.exec_context.thread_0.numBusyCycles",
    ])
    idle = stats.first_existing([
        "board.processor.cores.core.exec_context.thread_0.numIdleCycles",
    ])
    not_idle = stats.first_existing([
        "board.processor.cores.core.exec_context.thread_0.notIdleFraction",
    ])
    rows = [
        metric_row("numCycles", value_or_nd(num_cycles), fmt_num(value_or_nd(num_cycles)), source_or_nd(num_cycles), "cycles"),
        metric_row("IPC", value_or_nd(ipc), "n/d" if ipc is None else f"{ipc[1]:.9f}", source_or_nd(ipc), "inst/cycle"),
        metric_row("CPI", value_or_nd(cpi), "n/d" if cpi is None else f"{cpi[1]:.9f}", source_or_nd(cpi), "cycle/inst"),
        metric_row("committed instructions", value_or_nd(committed), fmt_num(value_or_nd(committed)), source_or_nd(committed), "count"),
        metric_row("committed ops", value_or_nd(committed_ops), fmt_num(value_or_nd(committed_ops)), source_or_nd(committed_ops), "count"),
        metric_row("fetched instructions", value_or_nd(fetched), fmt_num(value_or_nd(fetched)), source_or_nd(fetched), "count"),
        metric_row("executed instructions", value_or_nd(executed), fmt_num(value_or_nd(executed)), source_or_nd(executed), "count"),
        metric_row("busy cycles", value_or_nd(busy), fmt_num(value_or_nd(busy)), source_or_nd(busy), "cycles"),
        metric_row("idle cycles", value_or_nd(idle), fmt_num(value_or_nd(idle)), source_or_nd(idle), "cycles"),
        metric_row("notIdleFraction", value_or_nd(not_idle), "n/d" if not_idle is None else f"{not_idle[1]:.9f}", source_or_nd(not_idle), "ratio"),
    ]
    if num_cycles and sim_insts:
        derived_ipc = safe_div(sim_insts, num_cycles[1])
        derived_cpi = safe_div(num_cycles[1], sim_insts)
        rows.extend([
            derived_row("IPC derivato", derived_ipc, "n/d" if derived_ipc is None else f"{derived_ipc:.9f}", "simInsts / numCycles", "inst/cycle"),
            derived_row("CPI derivato", derived_cpi, "n/d" if derived_cpi is None else f"{derived_cpi:.9f}", "numCycles / simInsts", "cycle/inst"),
        ])
    return rows


def build_instruction_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    sim_insts = stats.get("simInsts")
    items = [
        ("integer instructions", stats.first_existing(["board.processor.cores.core.commitStats0.numIntInsts", "board.processor.cores.core.executeStats0.numIntAluAccesses"])),
        ("floating-point instructions/accesses", stats.first_existing(["board.processor.cores.core.commitStats0.numFpInsts", "board.processor.cores.core.executeStats0.numFpAluAccesses"])),
        ("vector instructions/accesses", stats.first_existing(["board.processor.cores.core.commitStats0.numVecInsts", "board.processor.cores.core.executeStats0.numVecAluAccesses"])),
        ("memory references", stats.first_existing(["board.processor.cores.core.commitStats0.numMemRefs", "board.processor.cores.core.thread_0.numMemRefs", "board.processor.cores.core.executeStats0.numMemRefs"])),
        ("load instructions", stats.first_existing(["board.processor.cores.core.commitStats0.numLoadInsts", "board.processor.cores.core.executeStats0.numLoadInsts"])),
        ("store instructions", stats.first_existing(["board.processor.cores.core.commitStats0.numStoreInsts", "board.processor.cores.core.executeStats0.numStoreInsts"])),
        ("branches/control", stats.first_existing(["board.processor.cores.core.executeStats0.numBranches", "board.processor.cores.core.commitStats0.committedControl::IsControl"])),
        ("function calls", stats.first_existing(["board.processor.cores.core.commitStats0.functionCalls"])),
        ("syscalls", stats.first_existing(["board.processor.cores.core.workload.numSyscalls"])),
    ]
    rows = []
    for label, item in items:
        value = value_or_nd(item)
        rows.append({
            "metrica": label,
            "conteggio": fmt_num(value),
            "conteggio_raw": "n/d" if value is None else str(value),
            "% su simInsts": fmt_percent(pct(value, sim_insts)),
            "origine": source_or_nd(item),
        })
    return rows


def build_memory_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    sim_freq = stats.get("simFreq")
    key_defs = [
        ("readReqs", "board.memory.mem_ctrl.readReqs", fmt_num, "count"),
        ("writeReqs", "board.memory.mem_ctrl.writeReqs", fmt_num, "count"),
        ("readBursts", "board.memory.mem_ctrl.readBursts", fmt_num, "count"),
        ("writeBursts", "board.memory.mem_ctrl.writeBursts", fmt_num, "count"),
        ("dram.readBursts", "board.memory.mem_ctrl.dram.readBursts", fmt_num, "count"),
        ("dram.writeBursts", "board.memory.mem_ctrl.dram.writeBursts", fmt_num, "count"),
        ("bytesReadSys", "board.memory.mem_ctrl.bytesReadSys", bytes_to_human, "B"),
        ("bytesWrittenSys", "board.memory.mem_ctrl.bytesWrittenSys", bytes_to_human, "B"),
        ("dram.bytesRead.total", "board.memory.mem_ctrl.dram.bytesRead::total", bytes_to_human, "B"),
        ("dram.bytesWritten.total", "board.memory.mem_ctrl.dram.bytesWritten::total", bytes_to_human, "B"),
        ("dram.dramBytesRead", "board.memory.mem_ctrl.dram.dramBytesRead", bytes_to_human, "B"),
        ("dram.dramBytesWritten", "board.memory.mem_ctrl.dram.dramBytesWritten", bytes_to_human, "B"),
        ("avgRdBWSys", "board.memory.mem_ctrl.avgRdBWSys", lambda v: "n/d" if v is None else f"{v:.6f} B/s", "B/s"),
        ("avgWrBWSys", "board.memory.mem_ctrl.avgWrBWSys", lambda v: "n/d" if v is None else f"{v:.6f} B/s", "B/s"),
        ("dram.avgRdBW", "board.memory.mem_ctrl.dram.avgRdBW", lambda v: "n/d" if v is None else f"{v:.6f} MiB/s", "MiB/s"),
        ("dram.avgWrBW", "board.memory.mem_ctrl.dram.avgWrBW", lambda v: "n/d" if v is None else f"{v:.6f} MiB/s", "MiB/s"),
        ("dram.peakBW", "board.memory.mem_ctrl.dram.peakBW", lambda v: "n/d" if v is None else f"{v:.6f} MiB/s", "MiB/s"),
        ("dram.busUtil", "board.memory.mem_ctrl.dram.busUtil", fmt_percent, "%"),
        ("dram.busUtilRead", "board.memory.mem_ctrl.dram.busUtilRead", fmt_percent, "%"),
        ("dram.busUtilWrite", "board.memory.mem_ctrl.dram.busUtilWrite", fmt_percent, "%"),
        ("dram.pageHitRate", "board.memory.mem_ctrl.dram.pageHitRate", fmt_percent, "%"),
        ("dram.readRowHitRate", "board.memory.mem_ctrl.dram.readRowHitRate", fmt_percent, "%"),
        ("dram.writeRowHitRate", "board.memory.mem_ctrl.dram.writeRowHitRate", fmt_percent, "%"),
        ("avgRdQLen", "board.memory.mem_ctrl.avgRdQLen", fmt_num, "count/tick"),
        ("avgWrQLen", "board.memory.mem_ctrl.avgWrQLen", fmt_num, "count/tick"),
        ("numRdRetry", "board.memory.mem_ctrl.numRdRetry", fmt_num, "count"),
        ("numWrRetry", "board.memory.mem_ctrl.numWrRetry", fmt_num, "count"),
        ("mergedWrBursts", "board.memory.mem_ctrl.mergedWrBursts", fmt_num, "count"),
        ("servicedByWrQ", "board.memory.mem_ctrl.servicedByWrQ", fmt_num, "count"),
    ]
    rows = []
    for label, key, formatter, unit in key_defs:
        if key in stats.entries:
            rows.append(raw_row(stats, label, key, formatter, unit))

    latency_defs = [
        ("dram.avgQLat", "board.memory.mem_ctrl.dram.avgQLat"),
        ("dram.avgBusLat", "board.memory.mem_ctrl.dram.avgBusLat"),
        ("dram.avgMemAccLat", "board.memory.mem_ctrl.dram.avgMemAccLat"),
        ("priorityMinLatency", "board.memory.mem_ctrl.priorityMinLatency"),
        ("priorityMaxLatency", "board.memory.mem_ctrl.priorityMaxLatency"),
    ]
    for label, key in latency_defs:
        value = stats.get(key)
        if value is None:
            continue
        if key.endswith("Latency"):
            rows.append(metric_row(label, value, fmt_seconds(value), key, "s"))
        else:
            ns = ticks_to_ns(value, sim_freq)
            rows.append(derived_row(f"{label} ns", ns, "n/d" if ns is None else f"{ns:.6f} ns", f"{key} / simFreq * 1e9", "ns"))
            rows.append(raw_row(stats, f"{label} ticks", key, fmt_num, "ticks"))
    return rows


def get_suffix_value(mapping: Dict[str, float], suffix: str) -> Optional[float]:
    for key, value in mapping.items():
        if key.endswith("::" + suffix):
            return value
    return None


def build_requestor_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    read_bytes = stats.find(r"\.requestorReadBytes::")
    write_bytes = stats.find(r"\.requestorWriteBytes::")
    read_accesses = stats.find(r"\.requestorReadAccesses::")
    write_accesses = stats.find(r"\.requestorWriteAccesses::")
    read_avg_lat = stats.find(r"\.requestorReadAvgLat::")
    write_avg_lat = stats.find(r"\.requestorWriteAvgLat::")
    sim_freq = stats.get("simFreq")
    requestors = set()
    for mapping in [read_bytes, write_bytes, read_accesses, write_accesses, read_avg_lat, write_avg_lat]:
        for key in mapping:
            requestors.add(key.split("::", 1)[1])
    rows = []
    for req in sorted(requestors):
        rb = get_suffix_value(read_bytes, req)
        wb = get_suffix_value(write_bytes, req)
        ra = get_suffix_value(read_accesses, req)
        wa = get_suffix_value(write_accesses, req)
        rlat = ticks_to_ns(get_suffix_value(read_avg_lat, req), sim_freq)
        wlat = ticks_to_ns(get_suffix_value(write_avg_lat, req), sim_freq)
        rows.append({
            "requestor": short_name(req, 48),
            "read_bytes": bytes_to_human(rb),
            "read_accesses": fmt_num(ra),
            "avg_read_lat_ns": "n/d" if rlat is None else f"{rlat:.6f}",
            "write_bytes": bytes_to_human(wb),
            "write_accesses": fmt_num(wa),
            "avg_write_lat_ns": "n/d" if wlat is None else f"{wlat:.6f}",
        })
    return rows


def build_membus_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    keys = [
        ("membus pktCount total", "board.cache_hierarchy.membus.pktCount::total", fmt_num, "count"),
        ("membus pktSize total", "board.cache_hierarchy.membus.pktSize::total", bytes_to_human, "B"),
        ("membus snoops", "board.cache_hierarchy.membus.snoops", fmt_num, "count"),
        ("membus snoopTraffic", "board.cache_hierarchy.membus.snoopTraffic", bytes_to_human, "B"),
        ("ReadReq", "board.cache_hierarchy.membus.transDist::ReadReq", fmt_num, "count"),
        ("ReadResp", "board.cache_hierarchy.membus.transDist::ReadResp", fmt_num, "count"),
        ("WriteReq", "board.cache_hierarchy.membus.transDist::WriteReq", fmt_num, "count"),
        ("WriteResp", "board.cache_hierarchy.membus.transDist::WriteResp", fmt_num, "count"),
    ]
    rows = []
    for label, key, formatter, unit in keys:
        if key in stats.entries:
            rows.append(raw_row(stats, label, key, formatter, unit))
    return rows


def build_bank_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    reads = stats.find(r"\.dram\.perBankRdBursts::\d+$")
    writes = stats.find(r"\.dram\.perBankWrBursts::\d+$")
    banks = sorted({int(k.split("::")[-1]) for k in reads} | {int(k.split("::")[-1]) for k in writes})
    total_reads = sum(reads.values()) if reads else None
    total_writes = sum(writes.values()) if writes else None
    rows = []
    for bank in banks:
        r = get_suffix_value(reads, str(bank))
        w = get_suffix_value(writes, str(bank))
        rows.append({
            "bank": str(bank),
            "read_bursts": fmt_num(r),
            "read_bursts_raw": "n/d" if r is None else str(r),
            "% read": fmt_percent(pct(r, total_reads)),
            "write_bursts": fmt_num(w),
            "write_bursts_raw": "n/d" if w is None else str(w),
            "% write": fmt_percent(pct(w, total_writes)),
        })
    return rows


def build_energy_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    patterns = [
        r"\.dram\.rank\d+\.actEnergy$",
        r"\.dram\.rank\d+\.preEnergy$",
        r"\.dram\.rank\d+\.readEnergy$",
        r"\.dram\.rank\d+\.writeEnergy$",
        r"\.dram\.rank\d+\.refreshEnergy$",
        r"\.dram\.rank\d+\.actBackEnergy$",
        r"\.dram\.rank\d+\.preBackEnergy$",
        r"\.dram\.rank\d+\.totalEnergy$",
        r"\.dram\.rank\d+\.averagePower$",
    ]
    rx = re.compile("|".join(f"(?:{p})" for p in patterns))
    rows = []
    for key in sorted(stats.entries):
        if not rx.search(key):
            continue
        value = stats.get(key)
        metric = key.split(".dram.", 1)[-1]
        unit = stats.entries[key].unit or ""
        formatted = f"{value:.6f}" if isinstance(value, float) else fmt_num(value)
        rows.append(metric_row(metric, value, formatted, key, unit))
    return rows


def build_tlb_rows(stats: Gem5Stats) -> List[Dict[str, str]]:
    prefixes = sorted(set(k.rsplit(".", 1)[0] for k in stats.entries if re.search(r"\.mmu\.[id]tb\.accesses$", k)))
    rows = []
    for prefix in prefixes:
        accesses = stats.get(f"{prefix}.accesses")
        hits = stats.get(f"{prefix}.hits")
        misses = stats.get(f"{prefix}.misses")
        rows.append({
            "tlb": short_name(prefix),
            "accesses": fmt_num(accesses),
            "hits": fmt_num(hits),
            "misses": fmt_num(misses),
            "hit_rate": fmt_percent(pct(hits, accesses)),
        })
    return rows


def build_summary_rows(stats: Gem5Stats, config: Optional[Dict[str, Any]], top: int) -> Dict[str, List[Dict[str, str]]]:
    sim_freq = stats.get("simFreq")
    sim_seconds = stats.get("simSeconds")
    sim_insts = stats.get("simInsts")
    num_cycles = value_or_nd(stats.first_existing([
        "board.processor.cores.core.numCycles",
        "system.cpu.numCycles",
    ]))
    ipc = value_or_nd(stats.first_existing([
        "board.processor.cores.core.ipc",
        "board.processor.cores.core.commitStats0.ipc",
        "system.cpu.ipc",
    ]))
    cpi = value_or_nd(stats.first_existing([
        "board.processor.cores.core.cpi",
        "board.processor.cores.core.commitStats0.cpi",
        "system.cpu.cpi",
    ]))
    host_seconds = stats.get("hostSeconds")
    host_inst_rate = stats.get("hostInstRate")
    freq_ghz = ghz_from_clock(sim_freq, stats.get("board.clk_domain.clock"))
    mem_refs = value_or_nd(stats.first_existing([
        "board.processor.cores.core.commitStats0.numMemRefs",
        "board.processor.cores.core.thread_0.numMemRefs",
        "board.processor.cores.core.executeStats0.numMemRefs",
    ]))
    load_insts = value_or_nd(stats.first_existing([
        "board.processor.cores.core.commitStats0.numLoadInsts",
        "board.processor.cores.core.executeStats0.numLoadInsts",
    ]))
    store_insts = value_or_nd(stats.first_existing([
        "board.processor.cores.core.commitStats0.numStoreInsts",
        "board.processor.cores.core.executeStats0.numStoreInsts",
    ]))
    branch_insts = value_or_nd(stats.first_existing([
        "board.processor.cores.core.executeStats0.numBranches",
        "board.processor.cores.core.commitStats0.committedControl::IsControl",
    ]))
    fp_insts = value_or_nd(stats.first_existing([
        "board.processor.cores.core.commitStats0.numFpInsts",
        "board.processor.cores.core.executeStats0.numFpAluAccesses",
    ]))
    int_insts = value_or_nd(stats.first_existing([
        "board.processor.cores.core.commitStats0.numIntInsts",
        "board.processor.cores.core.executeStats0.numIntAluAccesses",
    ]))
    dram_read_bytes = stats.get("board.memory.mem_ctrl.bytesReadSys")
    dram_write_bytes = stats.get("board.memory.mem_ctrl.bytesWrittenSys")
    dram_read_bw = stats.get("board.memory.mem_ctrl.dram.avgRdBW")
    dram_write_bw = stats.get("board.memory.mem_ctrl.dram.avgWrBW")
    dram_peak_bw = stats.get("board.memory.mem_ctrl.dram.peakBW")
    dram_bus_util = stats.get("board.memory.mem_ctrl.dram.busUtil")
    dram_page_hit = stats.get("board.memory.mem_ctrl.dram.pageHitRate")
    dram_avg_mem_lat_ns = ticks_to_ns(stats.get("board.memory.mem_ctrl.dram.avgMemAccLat"), sim_freq)
    dram_avg_q_lat_ns = ticks_to_ns(stats.get("board.memory.mem_ctrl.dram.avgQLat"), sim_freq)
    dram_avg_bus_lat_ns = ticks_to_ns(stats.get("board.memory.mem_ctrl.dram.avgBusLat"), sim_freq)

    summary_overview = [
        {"metrica": "simulated time", "valore": fmt_seconds(sim_seconds)},
        {"metrica": "simulated instructions", "valore": fmt_num(sim_insts)},
        {"metrica": "cpu cycles", "valore": fmt_num(num_cycles)},
        {"metrica": "estimated clock", "valore": "n/d" if freq_ghz is None else f"{freq_ghz:.6f} GHz"},
        {"metrica": "host runtime", "valore": fmt_seconds(host_seconds)},
        {"metrica": "host instruction rate", "valore": "n/d" if host_inst_rate is None else f"{fmt_num(host_inst_rate)} inst/s"},
    ]

    summary_cpu = [
        {"metrica": "IPC", "valore": "n/d" if ipc is None else f"{ipc:.9f}"},
        {"metrica": "CPI", "valore": "n/d" if cpi is None else f"{cpi:.9f}"},
        {"metrica": "memory refs share", "valore": fmt_percent(pct(mem_refs, sim_insts))},
        {"metrica": "load share", "valore": fmt_percent(pct(load_insts, sim_insts))},
        {"metrica": "store share", "valore": fmt_percent(pct(store_insts, sim_insts))},
        {"metrica": "branch share", "valore": fmt_percent(pct(branch_insts, sim_insts))},
        {"metrica": "integer share", "valore": fmt_percent(pct(int_insts, sim_insts))},
        {"metrica": "floating-point share", "valore": fmt_percent(pct(fp_insts, sim_insts))},
    ]

    summary_memory = [
        {"metrica": "bytes read by system", "valore": bytes_to_human(dram_read_bytes)},
        {"metrica": "bytes written by system", "valore": bytes_to_human(dram_write_bytes)},
        {"metrica": "avg DRAM read bandwidth", "valore": "n/d" if dram_read_bw is None else f"{dram_read_bw:.6f} MiB/s"},
        {"metrica": "avg DRAM write bandwidth", "valore": "n/d" if dram_write_bw is None else f"{dram_write_bw:.6f} MiB/s"},
        {"metrica": "peak DRAM bandwidth", "valore": "n/d" if dram_peak_bw is None else f"{dram_peak_bw:.6f} MiB/s"},
        {"metrica": "DRAM bus utilization", "valore": fmt_percent(dram_bus_util)},
        {"metrica": "DRAM page hit rate", "valore": fmt_percent(dram_page_hit)},
        {"metrica": "avg DRAM access latency", "valore": "n/d" if dram_avg_mem_lat_ns is None else f"{dram_avg_mem_lat_ns:.6f} ns"},
        {"metrica": "avg DRAM queue latency", "valore": "n/d" if dram_avg_q_lat_ns is None else f"{dram_avg_q_lat_ns:.6f} ns"},
        {"metrica": "avg DRAM bus latency", "valore": "n/d" if dram_avg_bus_lat_ns is None else f"{dram_avg_bus_lat_ns:.6f} ns"},
    ]

    cache_rows = extract_cache_rows(stats)
    top_caches = cache_rows[: min(len(cache_rows), 6)]
    mix_rows = extract_instruction_mix(stats, max(min(top, 8), 0))
    requestor_rows = build_requestor_rows(stats)[:4]
    config_rows = build_config_rows(config, max_rows=12)

    data = {
        "summary_overview": summary_overview,
        "summary_cpu": summary_cpu,
        "summary_memory": summary_memory,
        "top_instruction_mix": mix_rows,
        "top_requestors": requestor_rows,
        "cache_summary": top_caches,
        "config_summary": config_rows,
    }
    return {k: v for k, v in data.items() if v}


def walk_json(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else str(k)
            yield child, v
            yield from walk_json(v, child)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child = f"{path}[{i}]"
            yield child, v
            yield from walk_json(v, child)


def build_config_rows(config: Optional[Dict[str, Any]], max_rows: int = 80) -> List[Dict[str, str]]:
    if not config:
        return []
    key_rx = re.compile(
        r"(^|\.)(type|cpu_type|isa|num_cpus|clk_domain|clock|voltage|cache_line_size|mem_ranges|range|size|dram|mem_ctrl|workload|executable|cmd|system)$",
        re.IGNORECASE,
    )
    rows = []
    for path, value in walk_json(config):
        if len(rows) >= max_rows:
            break
        if not key_rx.search(path):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = str(value)
        elif isinstance(value, list) and len(value) <= 8 and all(not isinstance(x, (dict, list)) for x in value):
            text = ", ".join(map(str, value))
        else:
            continue
        if len(text) > 160:
            text = text[:157] + "..."
        rows.append({"campo": short_name(path, 90), "valore": text})
    return rows


def report_data(stats: Gem5Stats, config: Optional[Dict[str, Any]], top: int) -> Dict[str, List[Dict[str, str]]]:
    data = {
        "simulazione": build_general_rows(stats),
        "cpu": build_cpu_rows(stats),
        "istruzioni": build_instruction_rows(stats),
        "mix_istruzioni": extract_instruction_mix(stats, top),
        "membus": build_membus_rows(stats),
        "memoria_dram": build_memory_rows(stats),
        "requestor": build_requestor_rows(stats),
        "dram_bank": build_bank_rows(stats),
        "dram_energy_power": build_energy_rows(stats),
        "cache": extract_cache_rows(stats),
        "tlb_mmu": build_tlb_rows(stats),
        "config": build_config_rows(config),
    }
    return {k: v for k, v in data.items() if v}


def build_report(data: Dict[str, List[Dict[str, str]]], fmt: str, title: str) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)
    md = to_markdown(data, title)
    return to_html(md) if fmt == "html" else md


def default_summary_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_summary{out_path.suffix}")


def markdown_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return ""
    headers: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(escape_md(str(row.get(h, ""))) for h in headers) + " |")
    return "\n".join(out)


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(data: Dict[str, List[Dict[str, str]]], title: str) -> str:
    lines = [f"# {title}", ""]
    for section, rows in data.items():
        lines.append(f"## {section.replace('_', ' ')}")
        lines.append(markdown_table(rows))
        lines.append("")
    return "\n".join(lines)


def to_html(md: str) -> str:
    out: List[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            table_lines = []
            while i < len(lines) and lines[i].startswith("| "):
                table_lines.append(lines[i])
                i += 1
            i -= 1
            out.append(table_to_html(table_lines))
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
        i += 1
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; line-height: 1.4; color: #111827; }
    h1 { margin-bottom: 0.6rem; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.25rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
    th, td { border: 1px solid #e5e7eb; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }
    th { background: #f9fafb; }
    tr:nth-child(even) td { background: #fcfcfd; }
    """
    return "<!doctype html>\n<html><head><meta charset='utf-8'><title>Report gem5 data-only</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>\n"


def table_to_html(table_lines: List[str]) -> str:
    rows = []
    for idx, line in enumerate(table_lines):
        cells = [c.strip().replace("\\|", "|") for c in line.strip().strip("|").split("|")]
        if idx == 1 and all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        tag = "th" if idx == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def root_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return path
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Genera un report data-only da stats.txt di gem5.")
    parser.add_argument("--stats", type=Path, default=root_path("results", "cache", "stats.txt"), help="Percorso al file stats.txt")
    parser.add_argument("--config", type=Path, default=root_path("results", "cache", "config.json"), help="Percorso opzionale a config.json")
    parser.add_argument("--out", type=Path, default=root_path("simulation_metrics", "cache_metrics", "simulation_metrics.md"), help="File di output. Default: stdout")
    parser.add_argument("--summary-out", type=Path, default=None, help="File di output per il report riassuntivo. Default: <out>_summary")
    parser.add_argument("--format", choices=["md", "html", "json"], default="md", help="Formato del report")
    parser.add_argument("--title", default="Report gem5 data-only", help="Titolo del report")
    parser.add_argument("--top", type=int, default=20, help="Numero massimo di classi di istruzioni nel mix")
    parser.add_argument(
        "--stats-section",
        choices=["first", "last"],
        default="first",
        help="Sezione di stats.txt da usare. Con m5_dump_stats, 'first' isola la finestra reset-to-dump.",
    )
    args = parser.parse_args(argv)

    args.stats = resolve_path(args.stats)
    args.config = resolve_path(args.config)
    args.out = resolve_path(args.out)
    args.summary_out = resolve_path(args.summary_out)

    if not args.stats.exists():
        print(f"File stats non trovato: {args.stats}", file=sys.stderr)
        return 2

    stats = parse_stats(args.stats, section=args.stats_section)
    config = load_config(args.config)
    data = report_data(stats, config, top=max(args.top, 0))
    summary_data = build_summary_rows(stats, config, top=max(args.top, 0))
    report = build_report(data, args.format, args.title)
    summary_title = f"{args.title} - Summary"
    summary_report = build_report(summary_data, args.format, summary_title)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Report scritto in: {args.out}")
        summary_out = args.summary_out or default_summary_path(args.out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(summary_report, encoding="utf-8")
        print(f"Summary scritto in: {summary_out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
