#!/usr/bin/env python3
"""Static gate for the clean FA3 BWD dQ bringup implementation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def require(text: str, pattern: str, failures: list[str], name: str) -> None:
    if not re.search(pattern, text, flags=re.MULTILINE | re.DOTALL):
        failures.append(name)


def forbid(text: str, pattern: str, failures: list[str], name: str) -> None:
    if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL):
        failures.append(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/dq_kernel.cpp")
    parser.add_argument("--contract", default="include/dq_contract.h")
    parser.add_argument("--asm", default="build/fa3_bwd_dq_clean.asm")
    args = parser.parse_args()

    source = Path(args.source).read_text(errors="ignore")
    contract = Path(args.contract).read_text(errors="ignore")
    asm_path = Path(args.asm)
    asm = asm_path.read_text(errors="ignore") if asm_path.exists() else ""
    failures: list[str] = []

    require(source, r"fa3_bwd_dq_ref_softmax_kernel",
            failures, "missing_ref_softmax_kernel")
    require(source, r"fa3_bwd_dq_ref_delta_kernel",
            failures, "missing_ref_delta_kernel")
    require(source, r"fa3_bwd_dq_ref_dp_kernel",
            failures, "missing_ref_dp_kernel")
    require(source, r"fa3_bwd_dq_ref_output_kernel",
            failures, "missing_ref_output_kernel")
    require(source, r"cpu_reference_dq", failures, "missing_cpu_reference_dq")
    require(source, r"fa3_bwd_dq_correctness", failures,
            "missing_correctness_status_line")
    require(source, r"kDqPathReferenceCorrectness", failures,
            "missing_reference_path")
    require(source, r"kDqPathCanonicalDq", failures,
            "missing_canonical_path_contract_use")
    require(contract, r"using\s+ActiveDqTile\s*=\s*DqTileD128MqNk<64,\s*128>",
            failures, "missing_active_mq64_nk128_tile")
    require(contract, r"kForbidDuplicateScoreDpInsideDq\s*=\s*true",
            failures, "missing_no_duplicate_score_dp_contract")
    require(contract, r"kForbidDqAtomicAdd\s*=\s*true",
            failures, "missing_no_atomic_contract")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*60",
            failures, "missing_mmac_active_target")
    require(contract, r"kPlannedLdsBytes\s*=", failures,
            "missing_lds_budget_contract")
    require(source, r"dq_out", failures, "missing_dq_output_name")
    forbid(source, r"atomicAdd|atomic_add|global_atomic", failures,
           "dq_bringup_must_not_use_atomic")
    forbid(source, r"61C\d+|C1\d{2}", failures,
           "clean_source_must_not_define_cxx_phase_stack")
    forbid(source + contract, r"kDkvPath|DkvTile|fa3_bwd_dkv",
           failures, "dq_source_must_not_depend_on_dkv_path")

    if asm:
        require(asm, r"fa3_bwd_dq_ref_output_kernel", failures,
                "asm_missing_dq_ref_output_kernel")

    if failures:
        print("dQ kernel gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("dQ kernel gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
