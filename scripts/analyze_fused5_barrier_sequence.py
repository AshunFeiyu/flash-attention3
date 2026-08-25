#!/usr/bin/env python3
"""Recover fused5 ABarrier ownership costs from an XCU pipeline CSV.

The canonical 16-wave role mapping gives each SIMD one producer, consumer0,
consumer1, and dQ-writer wave.  For causal H1 runs, each role's ordered wait
sequence uniquely determines the barrier token and q-tile count, so no marker
instruction or diagnostic kernel build is required.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaitGap:
    wf_id: str
    wave_slot: int
    start: int
    duration: int
    se: int
    cu: int
    simd: int


ROLE_BY_SLOT = {
    0: "producer",
    1: "consumer0",
    2: "consumer1",
    3: "dq_writer",
}


def classify_producer(gaps: list[WaitGap]) -> tuple[int, list[str]]:
    q_tiles = len(gaps) - 1
    if q_tiles < 2:
        raise ValueError(f"producer wait count {len(gaps)} cannot encode q>=2")
    tokens = ["VSidecarReady"]
    for qi in range(2, q_tiles):
        tokens.append("RawUsed0.reuse" if qi % 2 == 0 else "RawUsed1.reuse")
    tokens.extend(("RawUsed0.final", "RawUsed1.final"))
    return q_tiles, tokens


def classify_consumer0(gaps: list[WaitGap]) -> tuple[int, list[str]]:
    if len(gaps) < 3 or len(gaps) % 2 == 0:
        raise ValueError(f"consumer0 wait count {len(gaps)} must be odd and >=3")
    q_tiles = (len(gaps) + 1) // 2
    tokens = ["ResidentFilled"]
    for qi in range(q_tiles):
        page = qi & 1
        tokens.append(f"RawFilled{page}")
        if qi >= 2:
            tokens.append("DqDone0" if page == 0 else "DqDone0Alt")
    return q_tiles, tokens


def classify_consumer1(gaps: list[WaitGap]) -> tuple[int, list[str]]:
    if len(gaps) < 4 or len(gaps) % 2 != 0:
        raise ValueError(f"consumer1 wait count {len(gaps)} must be even and >=4")
    q_tiles = len(gaps) // 2
    tokens = ["ResidentFilled", "RawFilled0"]
    for qi in range(1, q_tiles):
        tokens.extend((f"RawFilled{qi & 1}", "DqDone1"))
    return q_tiles, tokens


def classify_dq_writer(gaps: list[WaitGap]) -> tuple[int, list[str]]:
    if len(gaps) < 5 or len(gaps) % 2 == 0:
        raise ValueError(f"dQ-writer wait count {len(gaps)} must be odd and >=5")
    q_tiles = (len(gaps) - 1) // 2
    tokens = ["ResidentFilled"]
    for qi in range(q_tiles):
        tokens.extend(("BatchDsFilled1",
                       "BatchDsFilled0" if qi % 2 == 0
                       else "BatchDsFilled0Alt"))
    return q_tiles, tokens


CLASSIFIER_BY_ROLE = {
    "producer": classify_producer,
    "consumer0": classify_consumer0,
    "consumer1": classify_consumer1,
    "dq_writer": classify_dq_writer,
}


def read_wait_gaps(path: Path) -> dict[str, list[WaitGap]]:
    by_wave: dict[str, list[WaitGap]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("Kind") != "bubble" or
                    row.get("Before") != "s_abarrier_try_wait" or
                    row.get("After") != "s_xor_b32"):
                continue
            gap = WaitGap(
                wf_id=row["WF ID"],
                wave_slot=int(row["WaveSlot"]),
                start=int(row["Start"]),
                duration=int(row["Duration"]),
                se=int(row["SE"]),
                cu=int(row["CU"]),
                simd=int(row["SIMD"]),
            )
            by_wave[gap.wf_id].append(gap)
    for gaps in by_wave.values():
        gaps.sort(key=lambda gap: gap.start)
    return by_wave


def percentage(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    by_wave = read_wait_gaps(args.csv)
    if not by_wave:
        raise SystemExit("no s_abarrier_try_wait -> s_xor_b32 gaps found")

    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    q_tile_histogram: dict[tuple[str, int], int] = defaultdict(int)
    failures: list[str] = []
    for wf_id, gaps in by_wave.items():
        slot = gaps[0].wave_slot
        role = ROLE_BY_SLOT.get(slot)
        if role is None:
            failures.append(f"wf={wf_id} unsupported WaveSlot={slot}")
            continue
        try:
            q_tiles, tokens = CLASSIFIER_BY_ROLE[role](gaps)
        except ValueError as error:
            failures.append(f"wf={wf_id} role={role}: {error}")
            continue
        if len(tokens) != len(gaps):
            failures.append(
                f"wf={wf_id} role={role} tokens={len(tokens)} gaps={len(gaps)}")
            continue
        q_tile_histogram[(role, q_tiles)] += 1
        for token, gap in zip(tokens, gaps):
            values = totals[(role, token)]
            values[0] += 1
            values[1] += gap.duration
            values[2] = max(values[2], gap.duration)

    if failures:
        print("fused5 barrier sequence analysis: FAIL")
        for failure in failures[:40]:
            print(f"  - {failure}")
        return 1

    grand_duration = sum(values[1] for values in totals.values())
    critical_duration = sum(
        values[1] for (role, _), values in totals.items()
        if role != "producer")
    print(f"waves={len(by_wave)} wait_duration={grand_duration} "
          f"consumer_critical_duration={critical_duration}")
    print("role,token,count,duration,total_share,role_share,max_duration")
    role_duration = defaultdict(int)
    for (role, _), values in totals.items():
        role_duration[role] += values[1]
    ordered = sorted(totals.items(), key=lambda item: item[1][1], reverse=True)
    for (role, token), (count, duration, max_duration) in ordered:
        print(f"{role},{token},{count},{duration},"
              f"{percentage(duration, grand_duration):.4f},"
              f"{percentage(duration, role_duration[role]):.4f},"
              f"{max_duration}")

    print("\nq_tile_histogram role,q_tiles,waves")
    for (role, q_tiles), count in sorted(q_tile_histogram.items()):
        print(f"{role},{q_tiles},{count}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "role", "token", "count", "duration", "total_share_percent",
                "role_share_percent", "max_duration",
            ])
            for (role, token), (count, duration, max_duration) in ordered:
                writer.writerow([
                    role, token, count, duration,
                    f"{percentage(duration, grand_duration):.6f}",
                    f"{percentage(duration, role_duration[role]):.6f}",
                    max_duration,
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
