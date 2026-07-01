#!/usr/bin/env python3
"""Symbol-scoped AMDGPU metadata gate for the clean Shaobo FA3 BWD repo."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class KernelMetadata:
    name: str
    private_segment_fixed_size: int = 0
    sgpr_spill_count: int = 0
    vgpr_spill_count: int = 0
    vgpr_count: int = 0
    sgpr_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asm", required=True, type=Path)
    parser.add_argument("--symbol-regex", required=True, action="append")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--max-private-segment", type=int, default=0)
    parser.add_argument("--max-sgpr-spill", type=int, default=0)
    parser.add_argument("--max-vgpr-spill", type=int, default=0)
    parser.add_argument("--max-vgpr-count", type=int)
    return parser.parse_args()


def parse_int_after_colon(line: str) -> int:
    return int(line.split(":", 1)[1].strip())


def parse_metadata(text: str) -> list[KernelMetadata]:
    kernels: list[KernelMetadata] = []
    current: KernelMetadata | None = None
    for line in text.splitlines():
        if line.startswith("    .name:"):
            if current is not None:
                kernels.append(current)
            current = KernelMetadata(name=line.split(":", 1)[1].strip())
            continue
        if current is None:
            continue
        if ".private_segment_fixed_size:" in line:
            current.private_segment_fixed_size = parse_int_after_colon(line)
        elif ".sgpr_spill_count:" in line:
            current.sgpr_spill_count = parse_int_after_colon(line)
        elif ".vgpr_spill_count:" in line:
            current.vgpr_spill_count = parse_int_after_colon(line)
        elif ".vgpr_count:" in line:
            current.vgpr_count = parse_int_after_colon(line)
        elif ".sgpr_count:" in line:
            current.sgpr_count = parse_int_after_colon(line)
    if current is not None:
        kernels.append(current)
    return kernels


def row_failures(row: KernelMetadata, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if row.private_segment_fixed_size > args.max_private_segment:
        failures.append(
            "private_segment_fixed_size="
            f"{row.private_segment_fixed_size}>max{args.max_private_segment}"
        )
    if row.sgpr_spill_count > args.max_sgpr_spill:
        failures.append(
            f"sgpr_spill_count={row.sgpr_spill_count}>max{args.max_sgpr_spill}"
        )
    if row.vgpr_spill_count > args.max_vgpr_spill:
        failures.append(
            f"vgpr_spill_count={row.vgpr_spill_count}>max{args.max_vgpr_spill}"
        )
    if args.max_vgpr_count is not None and row.vgpr_count > args.max_vgpr_count:
        failures.append(f"vgpr_count={row.vgpr_count}>max{args.max_vgpr_count}")
    return failures


def print_csv(rows: list[KernelMetadata],
              failures: dict[str, list[str]]) -> None:
    print(
        "symbol,private_segment_fixed_size,sgpr_spill_count,vgpr_spill_count,"
        "sgpr_count,vgpr_count,failures"
    )
    for row in rows:
        print(
            ",".join(
                [
                    row.name,
                    str(row.private_segment_fixed_size),
                    str(row.sgpr_spill_count),
                    str(row.vgpr_spill_count),
                    str(row.sgpr_count),
                    str(row.vgpr_count),
                    "|".join(failures.get(row.name, [])),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    text = args.asm.read_text(encoding="utf-8", errors="replace")
    patterns = [re.compile(pattern) for pattern in args.symbol_regex]
    matched = [
        row
        for row in parse_metadata(text)
        if any(pattern.search(row.name) for pattern in patterns)
    ]
    failures = {
        row.name: row_failures(row, args)
        for row in matched
        if row_failures(row, args)
    }
    payload = {
        "asm": str(args.asm),
        "symbol_regex": [pattern.pattern for pattern in patterns],
        "symbols": [asdict(row) for row in matched],
        "failures": failures,
        "status": "PASS" if matched and not failures else "FAIL",
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.csv:
        print_csv(matched, failures)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    if not matched:
        print("symbol_metadata_gate_status=FAIL no_matching_symbols")
        return 2
    if failures:
        print("symbol_metadata_gate_status=FAIL")
        return 2
    print("symbol_metadata_gate_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
