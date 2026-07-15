#!/usr/bin/env python3
"""Measure MMAC and matrix-read fragmentation in generated assembly."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


def instruction_class(opcode: str, operands: str) -> str:
    if opcode.startswith("v_mmac"):
        return "MMAC"
    if opcode.startswith("ds_read_matrix"):
        return "MATRIX_READ"
    if opcode.startswith("ds_read"):
        return "DS_READ"
    if opcode.startswith("ds_write"):
        return "DS_WRITE"
    if opcode.startswith("buffer_load") and "lds" in operands:
        return "BPS_LOAD"
    if opcode.startswith(("buffer_load", "global_load", "flat_load")):
        return "VMEM_LOAD"
    if opcode.startswith(("buffer_store", "global_store", "flat_store")):
        return "VMEM_STORE"
    if opcode == "s_waitcnt":
        return "WAIT"
    if "barrier" in opcode:
        return "BARRIER"
    if opcode == "s_setprio":
        return "SETPRIO"
    if opcode.startswith("s_"):
        return "SALU"
    if opcode.startswith("v_"):
        return "VALU"
    return "OTHER"


def parse_function(path: Path, symbol: str) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if symbol in line
            and re.match(r"^\S*" + re.escape(symbol) + r"\S*:", line) is not None
        ),
        None,
    )
    if start is None:
        raise ValueError(f"symbol not found: {symbol}")
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if line.strip().startswith(".Lfunc_end")
        ),
        len(lines),
    )

    instructions: list[tuple[int, str, str]] = []
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*(.*)")
    for line_number, line in enumerate(lines[start:end], start + 1):
        text = line.strip()
        if not text or text.startswith((".", ";", "#")) or text.endswith(":"):
            continue
        match = pattern.match(text)
        if match is None:
            continue
        opcode, operands = match.groups()
        if opcode.startswith(("s_", "v_", "ds_", "buffer_", "flat_", "global_")):
            instructions.append((line_number, opcode, operands))
    return instructions


def consecutive_runs(
    instructions: list[tuple[int, str, str]],
) -> list[tuple[str, list[tuple[int, str, str]]]]:
    runs: list[tuple[str, list[tuple[int, str, str]]]] = []
    for instruction in instructions:
        category = instruction_class(instruction[1], instruction[2])
        if runs and runs[-1][0] == category:
            runs[-1][1].append(instruction)
        else:
            runs.append((category, [instruction]))
    return runs


def run_metrics(lengths: list[int]) -> dict[str, float | int | list[int]]:
    if not lengths:
        return {
            "instructions": 0,
            "runs": 0,
            "mean": 0.0,
            "median": 0.0,
            "maximum": 0,
            "singletons": 0,
            "singleton_pct": 0.0,
            "histogram": [],
        }
    singleton_count = sum(length == 1 for length in lengths)
    return {
        "instructions": sum(lengths),
        "runs": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "maximum": max(lengths),
        "singletons": singleton_count,
        "singleton_pct": 100.0 * singleton_count / len(lengths),
        "histogram": sorted(Counter(lengths).items()),
    }


def semantic_mmac_lengths(
    instructions: list[tuple[int, str, str]],
) -> list[int]:
    lengths: list[int] = []
    current = 0
    for _, opcode, operands in instructions:
        category = instruction_class(opcode, operands)
        if category == "MMAC":
            current += 1
        elif category == "SETPRIO" and current:
            continue
        else:
            if current:
                lengths.append(current)
                current = 0
    if current:
        lengths.append(current)
    return lengths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asm", type=Path, required=True)
    parser.add_argument("--symbol", default="fa3_bwd_dkv_kernel")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-mean-mmac", type=float)
    parser.add_argument("--max-singleton-mmac-pct", type=float)
    parser.add_argument("--max-singleton-read-pct", type=float)
    args = parser.parse_args()

    try:
        instructions = parse_function(args.asm, args.symbol)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    runs = consecutive_runs(instructions)
    strict_mmac = [len(run) for category, run in runs if category == "MMAC"]
    matrix_reads = [
        len(run) for category, run in runs if category == "MATRIX_READ"
    ]
    result = {
        "asm": str(args.asm),
        "symbol": args.symbol,
        "mmac_strict": run_metrics(strict_mmac),
        "mmac_semantic": run_metrics(semantic_mmac_lengths(instructions)),
        "matrix_read": run_metrics(matrix_reads),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for name in ("mmac_strict", "mmac_semantic", "matrix_read"):
            metrics = result[name]
            print(
                f"{name}: instructions={metrics['instructions']} "
                f"runs={metrics['runs']} mean={metrics['mean']:.2f} "
                f"median={metrics['median']:.2f} max={metrics['maximum']} "
                f"singletons={metrics['singletons']} "
                f"singleton_pct={metrics['singleton_pct']:.2f}"
            )
            print(f"  histogram={metrics['histogram']}")

    failures: list[str] = []
    semantic = result["mmac_semantic"]
    reads = result["matrix_read"]
    if args.min_mean_mmac is not None and semantic["mean"] < args.min_mean_mmac:
        failures.append("mean_mmac_below_gate")
    if (
        args.max_singleton_mmac_pct is not None
        and semantic["singleton_pct"] > args.max_singleton_mmac_pct
    ):
        failures.append("singleton_mmac_pct_above_gate")
    if (
        args.max_singleton_read_pct is not None
        and reads["singleton_pct"] > args.max_singleton_read_pct
    ):
        failures.append("singleton_read_pct_above_gate")
    if failures:
        print("FAIL: " + ",".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
