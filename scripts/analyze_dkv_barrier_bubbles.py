#!/usr/bin/env python3
"""Aggregate XCU SQTT issue gaps by dKV ABarrier ownership site."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaitSite:
    function: str
    token: str
    barrier_id: int


def parse_contract(path: Path) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    pattern = re.compile(r"static constexpr int k(\w+)\s*=\s*(\d+)\s*;")
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            token_ids[match.group(1)] = int(match.group(2))
    return token_ids


def parse_wait_sites(header: Path, token_ids: dict[str, int]) -> dict[int, WaitSite]:
    sites: dict[int, WaitSite] = {}
    function = ""
    function_pattern = re.compile(r"\bvoid\s+(wait_\w+)\s*\(")
    token_pattern = re.compile(r"Barrier::k(\w+)")
    for line_number, line in enumerate(header.read_text().splitlines(), start=1):
        function_match = function_pattern.search(line)
        if function_match:
            function = function_match.group(1)
        token_match = token_pattern.search(line)
        if not function or not token_match:
            continue
        token = token_match.group(1)
        if token not in token_ids:
            raise ValueError(f"token k{token} has no numeric ID in {header}")
        sites[line_number] = WaitSite(function, token, token_ids[token])
    return sites


def source_line(source: str, filename: str) -> int | None:
    match = re.search(rf"(?:^|/){re.escape(filename)}:(\d+)(?::\d+)?", source)
    return int(match.group(1)) if match else None


def asm_barrier_id(asm: str) -> int | None:
    if "s_abarrier_try_wait" not in asm:
        return None
    values = re.findall(r"(?<![A-Za-z_])\d+", asm)
    return int(values[-1]) if values else None


def percentage(value: int, total: int) -> str:
    return f"{100.0 * value / total:.2f}%" if total else "0.00%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument(
        "--header", type=Path, default=Path("include/dkv_barrier_tomography.h"))
    parser.add_argument(
        "--contract", type=Path, default=Path("include/dkv_contract.h"))
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    token_ids = parse_contract(args.contract)
    sites_by_line = parse_wait_sites(args.header, token_ids)
    sites_by_id = {
        site.barrier_id: site for site in sites_by_line.values()
        if site.token == "AllDone"
    }

    all_count = 0
    all_duration = 0
    wait_count = 0
    wait_duration = 0
    by_site: dict[tuple[WaitSite, str], list[int]] = defaultdict(lambda: [0, 0])
    by_after: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_location: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    top_event: dict[tuple[WaitSite, str], dict[str, str]] = {}
    unknown: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    with args.csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            duration = int(row["Duration"])
            all_count += 1
            all_duration += duration
            if row["Before"] != "s_abarrier_try_wait":
                continue

            wait_count += 1
            wait_duration += duration
            after = row["After"]
            by_after[after][0] += 1
            by_after[after][1] += duration

            line = source_line(row["Before Source"], args.header.name)
            site = sites_by_line.get(line) if line is not None else None
            barrier_id = asm_barrier_id(row["Before ASM"])
            if site is None and barrier_id is not None:
                site = sites_by_id.get(barrier_id)

            if site is None:
                key = (row["Before Source"], row["Before ASM"])
                unknown[key][0] += 1
                unknown[key][1] += duration
                continue

            by_site[(site, after)][0] += 1
            by_site[(site, after)][1] += duration
            location = f"se{row['SE']}/cu{row['CU']}/simd{row['SIMD']}"
            by_location[(site.function, location)][0] += 1
            by_location[(site.function, location)][1] += duration
            key = (site, after)
            if key not in top_event or duration > int(top_event[key]["Duration"]):
                top_event[key] = row

    print(f"input rows: count={all_count} duration={all_duration}")
    print(
        "ABarrier wait gaps: "
        f"count={wait_count} duration={wait_duration} "
        f"share_of_exported_duration={percentage(wait_duration, all_duration)}")
    print("\nBy wait site and successor:")
    print("site,token,barrier_id,after,count,duration,wait_share")
    ordered = sorted(by_site.items(), key=lambda item: item[1][1], reverse=True)
    for (site, after), (count, duration) in ordered:
        print(
            f"{site.function},{site.token},{site.barrier_id},{after},"
            f"{count},{duration},{percentage(duration, wait_duration)}")

    print("\nLongest event per wait site:")
    print("site,duration,se,cu,simd,wave_slot,start,end,wf_id")
    for (site, after), _ in ordered:
        row = top_event[(site, after)]
        print(
            f"{site.function},{row['Duration']},{row['SE']},{row['CU']},"
            f"{row['SIMD']},{row['WaveSlot']},{row['Start']},{row['End']},"
            f"{row['WF ID']}")

    print("\nBy successor:")
    for after, (count, duration) in sorted(
            by_after.items(), key=lambda item: item[1][1], reverse=True):
        print(f"{after}: count={count} duration={duration} "
              f"share={percentage(duration, wait_duration)}")

    if unknown:
        print("\nUnmapped wait sources:")
        for (source, asm), (count, duration) in sorted(
                unknown.items(), key=lambda item: item[1][1], reverse=True):
            print(f"count={count} duration={duration} source={source} asm={asm}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "site", "token", "barrier_id", "after", "count", "duration",
                "wait_share_percent",
            ])
            for (site, after), (count, duration) in ordered:
                writer.writerow([
                    site.function, site.token, site.barrier_id, after, count,
                    duration, f"{100.0 * duration / wait_duration:.6f}"
                    if wait_duration else "0.000000",
                ])

    # Location data is retained for programmatic extensions; longest events
    # above provide deterministic xcu time-window inputs without noisy tables.
    _ = by_location
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
