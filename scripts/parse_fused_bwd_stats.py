#!/usr/bin/env python3
"""Parse one fused backward PMD stats.txt file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


INSTRUCTION_TYPES = ("VALU", "SCA", "VMEM", "LDS", "MMOP", "FLAT")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_value(text: str, key: str) -> float:
    match = re.search(rf"^\s*{re.escape(key)}\s+({NUMBER})\b", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing counter {key}")
    return float(match.group(1))


def sum_cu_counter(text: str, counter: str) -> float:
    pattern = re.compile(
        rf"^\s*system\.shaders\.CUs\d+\.{re.escape(counter)}\s+({NUMBER})\b",
        re.MULTILINE,
    )
    return sum(float(match.group(1)) for match in pattern.finditer(text))


def parse_simd_rows(text: str) -> dict[int, list[dict[str, float]]]:
    pattern = re.compile(
        rf"^\s*system\.shaders\.CUs(\d+)\.SIMD(\d+)\."
        rf"(activeTimeCounter|mmopRunTimeCounter|runTimeCounter|waitVmCounter|"
        rf"waitLgkmCounter|barrierCounter)\s+({NUMBER})\b",
        re.MULTILINE,
    )
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for match in pattern.finditer(text):
        cu, simd, field, value = int(match.group(1)), int(match.group(2)), match.group(3), float(match.group(4))
        rows.setdefault((cu, simd), {})[field] = value

    by_cu: dict[int, list[dict[str, float]]] = {}
    for (cu, _simd), row in rows.items():
        by_cu.setdefault(cu, []).append(row)
    return by_cu


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_stats(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sim_ticks = parse_value(text, "system.simTicks")
    first_wave = parse_value(text, "system.shaders.firstWaveStartTick")
    last_wave = parse_value(text, "system.shaders.lastWaveEndTick")
    kernel_ticks = last_wave - first_wave

    instruction_counts = {
        kind: int(sum_cu_counter(text, f"numInstrTypeExecuted::{kind}"))
        for kind in INSTRUCTION_TYPES
    }
    coissue = {
        "success": int(sum_cu_counter(text, "SpPipeline.numSuccessCoIssue")),
        "fail": int(sum_cu_counter(text, "SpPipeline.numFailedCoIssue")),
    }
    by_cu = parse_simd_rows(text)
    active_cus: list[dict[str, object]] = []
    active_simds = 0
    total_active_time = 0.0
    total_mmop_runtime = 0.0
    total_wait_vm = 0.0
    total_wait_lgkm = 0.0
    total_barrier = 0.0
    for cu in sorted(by_cu):
        rows = [row for row in by_cu[cu] if row.get("activeTimeCounter", 0.0) > 0.0]
        if not rows:
            continue
        active_simds += len(rows)
        active_time = sum(row.get("activeTimeCounter", 0.0) for row in rows)
        mmop_runtime = sum(row.get("mmopRunTimeCounter", 0.0) for row in rows)
        total_active_time += active_time
        total_mmop_runtime += mmop_runtime
        total_wait_vm += sum(row.get("waitVmCounter", 0.0) for row in rows)
        total_wait_lgkm += sum(row.get("waitLgkmCounter", 0.0) for row in rows)
        total_barrier += sum(row.get("barrierCounter", 0.0) for row in rows)
        active_cus.append(
            {
                "cu": cu,
                "simds": len(rows),
                "mmac_active": ratio(mmop_runtime, active_time),
            }
        )

    per_cu_mmac = [float(row["mmac_active"]) for row in active_cus]
    mmac_stats = {
        "per_active_cu": active_cus,
        "avg": mean(per_cu_mmac) if per_cu_mmac else 0.0,
        "min": min(per_cu_mmac) if per_cu_mmac else 0.0,
        "max": max(per_cu_mmac) if per_cu_mmac else 0.0,
    }
    return {
        "stats": str(path),
        "simTicks": sim_ticks,
        "firstWave": first_wave,
        "lastWave": last_wave,
        "kernel_ticks": kernel_ticks,
        "instruction_counts": instruction_counts,
        "ldsBankConflict": int(sum_cu_counter(text, "ldsBankConflict")),
        "coissue": coissue,
        "active": {"cus": len(active_cus), "simds": active_simds},
        "mmac_active": {
            "value": ratio(total_mmop_runtime, total_active_time),
            **mmac_stats,
        },
        "wait_shares": {
            "waitVm": ratio(total_wait_vm, total_active_time),
            "waitLgkm": ratio(total_wait_lgkm, total_active_time),
            "barrier": ratio(total_barrier, total_active_time),
        },
    }


def print_text(payload: dict[str, object]) -> None:
    counts = payload["instruction_counts"]
    coissue = payload["coissue"]
    active = payload["active"]
    mmac = payload["mmac_active"]
    waits = payload["wait_shares"]
    print(
        f"simTicks={payload['simTicks']:.0f} firstWave={payload['firstWave']:.0f} "
        f"lastWave={payload['lastWave']:.0f} kernel_ticks={payload['kernel_ticks']:.0f}"
    )
    print("instr=" + ",".join(f"{kind}:{counts[kind]}" for kind in INSTRUCTION_TYPES))
    print(
        f"ldsBankConflict={payload['ldsBankConflict']} "
        f"coissue={coissue['success']}/{coissue['fail']} "
        f"active_cus={active['cus']} active_simds={active['simds']}"
    )
    print(
        f"mmac_active={mmac['value']:.6%} "
        f"per_active_cu_avg={mmac['avg']:.6%} "
        f"min={mmac['min']:.6%} max={mmac['max']:.6%}"
    )
    print(
        f"shares=waitVm:{waits['waitVm']:.6%},"
        f"waitLgkm:{waits['waitLgkm']:.6%},barrier:{waits['barrier']:.6%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path, help="one dispatch stats.txt")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()
    payload = parse_stats(args.stats)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
