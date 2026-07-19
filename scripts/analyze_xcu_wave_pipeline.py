#!/usr/bin/env python3
"""Compare instruction-phase timing across XCU pipeline event CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    timestamp: int
    category: str
    opcode: str
    peers: str


def category(row: dict[str, str]) -> str:
    opcode = row["Opcode"]
    group = row["Group"]
    if group in {"MMOP", "MMAC"} or opcode.startswith(("v_mmac", "mmop_")):
        return "MMAC"
    if opcode.startswith("s_waitcnt"):
        return "WAIT"
    if "abarrier" in opcode or opcode == "s_barrier":
        return "BARRIER"
    if group.startswith(("Vector Memory", "Global")):
        return "VMEM"
    if group == "VALU" or group.startswith("Vector"):
        return "VALU"
    if group.startswith("LDS"):
        return "LDS"
    return "OTHER"


def read_events(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Kind"] != "inst":
                continue
            events.append(Event(
                int(row["Issue TS"]), category(row), row["Opcode"],
                row["CoIssue Peers"],
            ))
    return sorted(events, key=lambda event: event.timestamp)


def parse_wave(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("wave must be LABEL=PIPELINE_EVENTS.csv")
    label, path = value.split("=", 1)
    return label, Path(path)


def mmac_runs(events: list[Event]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = -1
    end = -1
    count = 0
    for event in events:
        if event.category == "MMAC":
            if count == 0:
                start = event.timestamp
            end = event.timestamp
            count += 1
        elif count:
            runs.append((start, end, count))
            start = -1
            end = -1
            count = 0
    if count:
        runs.append((start, end, count))
    return runs


def peer_vector_opcode(peers: str) -> list[str]:
    if not peers or peers == "N/A":
        return []
    found: list[str] = []
    for peer in peers.split(";"):
        fields = peer.rsplit(":", 2)
        if len(fields) != 3:
            continue
        group = fields[1].strip()
        if group != "VALU" and not (
                group.startswith("Vector") and not group.startswith("Vector Memory")):
            continue
        found.append(fields[2].strip())
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", action="append", required=True, type=parse_wave)
    parser.add_argument("--bin-cycles", type=int, default=256)
    parser.add_argument("--top-runs", type=int, default=12)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if len(args.wave) != 2:
        parser.error("exactly two --wave inputs are required")

    waves = {label: read_events(path) for label, path in args.wave}
    labels = list(waves)
    if any(not events for events in waves.values()):
        raise ValueError("a pipeline CSV contains no instruction events")

    window_start = min(events[0].timestamp for events in waves.values())
    window_end = max(events[-1].timestamp for events in waves.values()) + 1
    print(f"window={window_start}:{window_end} bin_cycles={args.bin_cycles}")

    for label, events in waves.items():
        counts = Counter(event.category for event in events)
        mmac_events = [event for event in events if event.category == "MMAC"]
        peer_valu = Counter(
            opcode for event in mmac_events
            for opcode in peer_vector_opcode(event.peers))
        actual_coissue = sum(peer_valu.values())
        print(
            f"{label}: instructions={len(events)} categories={dict(counts)} "
            f"mmac_with_vector_peer={actual_coissue}/{len(mmac_events)}")
        if peer_valu:
            print(f"  top MMAC coissue peers: {peer_valu.most_common(8)}")
        runs = sorted(mmac_runs(events), key=lambda run: run[2], reverse=True)
        print(f"  MMAC runs (start,end,count), top {args.top_runs}:")
        for run in runs[:args.top_runs]:
            print(f"    {run[0]},{run[1]},{run[2]}")

    bins: dict[str, dict[int, Counter[str]]] = {
        label: defaultdict(Counter) for label in labels
    }
    for label, events in waves.items():
        for event in events:
            index = (event.timestamp - window_start) // args.bin_cycles
            bins[label][index][event.category] += 1

    phase_counts = Counter()
    rows: list[str] = []
    bin_count = (window_end - window_start + args.bin_cycles - 1) // args.bin_cycles
    for index in range(bin_count):
        left = bins[labels[0]][index]
        right = bins[labels[1]][index]
        left_mmac = left["MMAC"] > 0
        right_mmac = right["MMAC"] > 0
        left_valu = left["VALU"] > 0
        right_valu = right["VALU"] > 0
        if left_mmac and right_mmac:
            phase = "MMAC_vs_MMAC"
        elif (left_mmac and right_valu) or (right_mmac and left_valu):
            phase = "MMAC_vs_VALU"
        elif left_mmac or right_mmac:
            phase = "MMAC_without_peer_VALU"
        else:
            phase = "no_MMAC"
        phase_counts[phase] += 1
        start = window_start + index * args.bin_cycles
        end = min(start + args.bin_cycles, window_end)
        rows.append(
            f"{start}:{end},{phase},"
            f"{labels[0]}[M={left['MMAC']} V={left['VALU']} L={left['LDS']} "
            f"W={left['WAIT']} B={left['BARRIER']}],"
            f"{labels[1]}[M={right['MMAC']} V={right['VALU']} L={right['LDS']} "
            f"W={right['WAIT']} B={right['BARRIER']}]"
        )

    print(f"\nBinned phase classification: {dict(phase_counts)}")
    if not args.summary_only:
        print("time,phase,wave0,wave1")
        for row in rows:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
