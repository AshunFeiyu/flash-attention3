#!/usr/bin/env python3
"""Static source/ASM gate for the canonical dot_do_o wave reduction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/dot_do_o_kernel.cpp")
    parser.add_argument("--asm", default="build/dot/dot_do_o_kernel.asm")
    args = parser.parse_args()

    source = Path(args.source).read_text(encoding="utf-8", errors="replace")
    asm_path = Path(args.asm)
    asm = asm_path.read_text(encoding="utf-8", errors="replace") \
        if asm_path.exists() else ""
    failures: list[str] = []

    source_required = {
        "one_wave_per_row": r"blockIdx\.x \* kWavesPerBlock \+ wave",
        "four_waves_per_block": r"kWavesPerBlock\s*=\s*4",
        "wave64_reduce": r"__shfl_down\(value, offset, kWaveSize\)",
        "d128_only": r"kHeadDim\s*=\s*128",
    }
    source_forbidden = {
        "cta_barrier": r"__syncthreads|s_barrier|s_abarrier",
        "lds_reduction": r"__shared__",
        "atomic": r"atomicAdd|atomic_add|global_atomic",
    }
    for name, pattern in source_required.items():
        if not re.search(pattern, source, flags=re.MULTILINE):
            failures.append(f"source_missing_{name}")
    for name, pattern in source_forbidden.items():
        if re.search(pattern, source, flags=re.MULTILINE):
            failures.append(f"source_forbidden_{name}")

    if not asm_path.exists():
        failures.append("asm_file_missing")
    else:
        if "dot_do_o_kernel" not in asm:
            failures.append("asm_missing_dot_kernel")
        if len(re.findall(r"\bds_bpermute_b32\b", asm)) < 6:
            failures.append("asm_missing_six_wave_permute_steps")
        if re.search(r"^\s*s_(?:a|e)?barrier\b", asm, flags=re.MULTILINE):
            failures.append("asm_contains_barrier")
        if re.search(r"^\s*s_set_vgpr_size\b", asm, flags=re.MULTILINE):
            failures.append("asm_contains_wdra_resize")
        if re.search(r"^\s*s_trap\b", asm, flags=re.MULTILINE):
            failures.append("asm_contains_trap")

    if failures:
        print("dot_do_o kernel gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("dot_do_o kernel gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
