#!/usr/bin/env python3
"""Summarize v_mov instructions and source locations in generated asm."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asm", default="build/fa3_bwd_wasp_clean.asm")
    parser.add_argument(
        "--symbol-regex",
        help="Optional function label regex; counts only that function body.",
    )
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def symbol_body(lines: list[str], pattern: str | None) -> list[str]:
    if not pattern:
        return lines
    regex = re.compile(pattern)
    hint = None
    for idx, line in enumerate(lines):
        if regex.search(line):
            hint = idx
            break
    if hint is None:
        raise SystemExit(f"no symbol label matched: {pattern}")

    start = None
    label_re = re.compile(r"^\S.*:\s*(;.*)?$")
    for idx in range(hint, -1, -1):
        stripped = lines[idx].strip()
        if label_re.match(stripped) and not stripped.startswith("."):
            start = idx
            break
    if start is None:
        for idx in range(hint, len(lines)):
            stripped = lines[idx].strip()
            if label_re.match(stripped) and not stripped.startswith("."):
                start = idx
                break
    if start is None:
        raise SystemExit(f"matched pattern but could not find symbol label: {pattern}")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith(".Lfunc_end"):
            end = idx
            break
    return lines[start:end]


def main() -> int:
    args = parse_args()
    lines = Path(args.asm).read_text(errors="ignore").splitlines()
    lines = symbol_body(lines, args.symbol_regex)

    op_counts: Counter[str] = Counter()
    zero_counts: Counter[str] = Counter()
    loc_counts: Counter[tuple[str, str]] = Counter()
    last_loc = "NO_LOC"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(".loc"):
            last_loc = stripped
        match = re.search(r"\b(v_mov_[a-z0-9_]+|v_mov_b64)\b", stripped)
        if not match:
            continue
        op = match.group(1)
        op_counts[op] += 1
        if re.search(r",\s*(0|0x0)\b", stripped):
            zero_counts[op] += 1
        loc_counts[(last_loc, op)] += 1

    print("op_counts")
    for op, count in op_counts.most_common():
        print(f"{op},{count}")
    print("zero_counts")
    for op, count in zero_counts.most_common():
        print(f"{op},{count}")
    print("top_locations")
    for (loc, op), count in loc_counts.most_common(args.top):
        print(f"{count},{op},{loc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
