#!/usr/bin/env python3
"""Parse the fixed dot_do_o -> dKV -> dQ PMD dispatch sequence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


LABELS = ("dot_do_o", "dKV", "dQ")
INSTRUCTION_TYPES = ("VALU", "SCA", "VMEM", "LDS", "MMOP", "FLAT")


def parse_counter(text: str, name: str) -> int:
    match = re.search(rf"^.*\b{name}\s+(\d+)\s+", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"missing counter {name}")
    return int(match.group(1))


def stats_sort_key(path: Path) -> tuple[int, ...]:
    numeric = [int(part) for part in path.parts if part.isdigit()]
    return tuple(numeric)


def sum_cu_counter(text: str, counter: str) -> int:
    pattern = re.compile(
        rf"^system\.shaders\.CUs\d+\.{re.escape(counter)}\s+([0-9]+)",
        flags=re.MULTILINE,
    )
    return sum(int(match.group(1)) for match in pattern.finditer(text))


def parse_shader_float(text: str, counter: str) -> float:
    match = re.search(
        rf"^system\.shaders\.{re.escape(counter)}\s+([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else 0.0


def parse_simd_balance(text: str) -> dict[str, float | int]:
    matches = re.finditer(
        r"^system\.shaders\.CUs(\d+)\.SIMD(\d+)\.activeTimeCounter\s+([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )
    active_by_cu: dict[int, float] = {}
    active_simds: list[float] = []
    for match in matches:
        cu = int(match.group(1))
        value = float(match.group(3))
        if value <= 0:
            continue
        active_by_cu[cu] = active_by_cu.get(cu, 0.0) + value
        active_simds.append(value)

    if not active_simds:
        return {
            "active_cus": 0,
            "active_simds": 0,
            "simd_active_min": 0.0,
            "simd_active_max": 0.0,
            "simd_active_mean": 0.0,
            "simd_active_cv": 0.0,
        }

    mean = sum(active_simds) / len(active_simds)
    variance = sum((value - mean) ** 2 for value in active_simds) / len(active_simds)
    return {
        "active_cus": len(active_by_cu),
        "active_simds": len(active_simds),
        "simd_active_min": min(active_simds),
        "simd_active_max": max(active_simds),
        "simd_active_mean": mean,
        "simd_active_cv": math.sqrt(variance) / mean,
    }


def parse_dispatch_evidence(text: str) -> dict[str, object]:
    executed = {
        kind: sum_cu_counter(text, f"numInstrTypeExecuted::{kind}")
        for kind in INSTRUCTION_TYPES
    }
    issued_cycles = {
        kind: sum_cu_counter(text, f"numCyclesWithInstrTypeIssued::{kind}")
        for kind in ("VALU", "SCA", "VMEM", "LDS")
    }
    return {
        "executed": executed,
        "issued_cycles": issued_cycles,
        "coissue_success": sum_cu_counter(
            text, "SpPipeline.numSuccessCoIssue"
        ),
        "coissue_failed": sum_cu_counter(text, "SpPipeline.numFailedCoIssue"),
        "no_valu_ready_cycles": sum_cu_counter(text, "numCyclesWithNoVALUReady"),
        "lds_bank_conflict": sum_cu_counter(text, "ldsBankConflict"),
        "global_data_cycles": sum_cu_counter(
            text, "GlobalMemPipeline.numCyclesSpTaData"
        ),
        "global_addr_cycles": sum_cu_counter(
            text, "GlobalMemPipeline.numCyclesSpTaAddr"
        ),
        "shader_busy": {
            "valu": parse_shader_float(text, "valuBusy"),
            "valu_fp": parse_shader_float(text, "valuBusyFp"),
            "global_data": parse_shader_float(text, "spTaDataBusy"),
            "global_addr": parse_shader_float(text, "spTaAddrBusy"),
            "lds": parse_shader_float(text, "spLdsBusy"),
        },
        "simd_balance": parse_simd_balance(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5out", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-dot-share", type=float)
    args = parser.parse_args()

    stats_paths = []
    for path in args.m5out.rglob("stats.txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "firstWaveStartTick" in text and "lastWaveEndTick" in text:
            stats_paths.append(path)
    stats_paths.sort(key=stats_sort_key)
    if len(stats_paths) != len(LABELS):
        raise SystemExit(
            f"expected {len(LABELS)} dispatch stats, found {len(stats_paths)}"
        )

    dispatches = []
    for label, path in zip(LABELS, stats_paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        first = parse_counter(text, "firstWaveStartTick")
        last = parse_counter(text, "lastWaveEndTick")
        dispatches.append(
            {
                "label": label,
                "stats": str(path),
                "first_wave_start_tick": first,
                "last_wave_end_tick": last,
                "kernel_ticks": last - first,
                "evidence": parse_dispatch_evidence(text),
            }
        )

    total = sum(row["kernel_ticks"] for row in dispatches)
    dot_ticks = dispatches[0]["kernel_ticks"]
    peer_ticks = total - dot_ticks
    dot_share = dot_ticks / total
    target_ticks = (0.05 / 0.95) * peer_ticks
    payload = {
        "m5out": str(args.m5out),
        "dispatches": dispatches,
        "kernel_ticks_sum": total,
        "dot_share": dot_share,
        "dot_target_share": 0.05,
        "dot_max_ticks_for_current_peers": target_ticks,
        "dot_target_pass": dot_share <= 0.05,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for row in dispatches:
        print(f"{row['label']}_kernel_ticks={row['kernel_ticks']}")
    dot_evidence = dispatches[0]["evidence"]
    dot_balance = dot_evidence["simd_balance"]
    print(
        "dot_instruction_counts="
        + ",".join(
            f"{kind}:{dot_evidence['executed'][kind]}"
            for kind in INSTRUCTION_TYPES
        )
    )
    print(
        "dot_cu_simd_balance="
        f"active_cus:{dot_balance['active_cus']},"
        f"active_simds:{dot_balance['active_simds']},"
        f"min:{dot_balance['simd_active_min']:.3f},"
        f"max:{dot_balance['simd_active_max']:.3f},"
        f"cv:{dot_balance['simd_active_cv']:.6f}"
    )
    print(
        "dot_shader_busy="
        + ",".join(
            f"{kind}:{value:.6f}"
            for kind, value in dot_evidence["shader_busy"].items()
        )
    )
    print(f"full_bwd_kernel_ticks_sum={total}")
    print(f"dot_share={dot_share:.8%}")
    print(f"dot_max_ticks_for_current_peers={target_ticks:.0f}")
    print(f"dot_target_pass={int(dot_share <= 0.05)}")

    if args.max_dot_share is not None and dot_share > args.max_dot_share:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
