#!/usr/bin/env python3
"""Verify that diagnostic ABarrier source mapping preserves kernel codegen."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def kernel_body(text: str, symbol_fragment: str) -> str:
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if ".type" in line and symbol_fragment in line and "@function" in line),
        None,
    )
    if start is None:
        raise ValueError(f"kernel symbol containing {symbol_fragment!r} not found")
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i].startswith(".Lfunc_end")),
        None,
    )
    if end is None:
        raise ValueError("kernel end label not found")
    return "\n".join(lines[start:end])


def instructions(body: str) -> list[str]:
    result: list[str] = []
    for raw in body.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith((".", "#", "//")) or line.endswith(":"):
            continue
        if not re.match(r"[a-z][a-z0-9_.]*\s", line):
            continue
        line = re.sub(r"\.LBB\d+_\d+", ".LBB", line)
        result.append(re.sub(r"\s+", " ", line))
    return result


def opcode_counts(insts: list[str]) -> Counter[str]:
    return Counter(line.split(" ", 1)[0] for line in insts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--tomography", required=True)
    parser.add_argument("--symbol", default="fa3_bwd_dkv_kernel")
    args = parser.parse_args()

    control = instructions(kernel_body(
        Path(args.control).read_text(errors="ignore"), args.symbol))
    tomography = instructions(kernel_body(
        Path(args.tomography).read_text(errors="ignore"), args.symbol))

    failures: list[str] = []
    if len(control) != len(tomography):
        failures.append(
            f"instruction_count control={len(control)} tomography={len(tomography)}")
    if opcode_counts(control) != opcode_counts(tomography):
        failures.append("opcode_histogram_changed")

    control_waits = [x for x in control if x.startswith("s_abarrier_try_wait ")]
    tomography_waits = [x for x in tomography
                        if x.startswith("s_abarrier_try_wait ")]
    if Counter(control_waits) != Counter(tomography_waits):
        failures.append("abarrier_try_wait_operands_changed")

    if control != tomography:
        first = next(
            (i for i, pair in enumerate(zip(control, tomography))
             if pair[0] != pair[1]),
            min(len(control), len(tomography)),
        )
        failures.append(f"instruction_stream_changed_at={first}")
        lo = max(0, first - 2)
        hi = first + 3
        print("control context:")
        for line in control[lo:hi]:
            print(f"  {line}")
        print("tomography context:")
        for line in tomography[lo:hi]:
            print(f"  {line}")

    print(f"control instructions={len(control)} waits={len(control_waits)}")
    print(f"tomography instructions={len(tomography)} waits={len(tomography_waits)}")
    if failures:
        print("dKV barrier tomography gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("dKV barrier tomography gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
