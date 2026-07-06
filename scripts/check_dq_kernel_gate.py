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
    require(source, r"fa3_bwd_dq_kernel", failures,
            "missing_canonical_dq_kernel")
    require(source, r"hcu_wdra_waves_per_tg\(12\)", failures,
            "missing_12wave_wdra_attribute")
    require(source, r"dq_publish_ds_chunk", failures,
            "missing_split_ds_publisher")
    require(source, r"dq_load_sidecar_tile", failures,
            "missing_producer_sidecar_lds_staging")
    require(source, r"dq_consume_ds_kt_full_dtile", failures,
            "missing_dq_mmac_consumer")
    require(source, r"materialize_k_t_source", failures,
            "missing_kt_source_layout_helper")
    require(source, r"CANONICAL_DQ", failures,
            "missing_canonical_standalone_switch")
    require(contract, r"using\s+ActiveDqTile\s*=\s*DqTileD128MqNk<32,\s*64>",
            failures, "missing_active_mq32_nk64_tile")
    require(contract, r"kForbidDuplicateScoreDpInsideDq\s*=\s*true",
            failures, "missing_no_duplicate_score_dp_contract")
    require(contract, r"kForbidDqAtomicAdd\s*=\s*true",
            failures, "missing_no_atomic_contract")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*40",
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
        require(asm, r"fa3_bwd_dq_kernel", failures,
                "asm_missing_canonical_dq_kernel")
        require(asm, r"matrix_load_32x32_b16.*bps lds", failures,
                "asm_missing_matrix_load_bps_lds")
        require(asm, r"ds_read_matrix_trans_format", failures,
                "asm_missing_ds_read_matrix_trans")
        require(asm, r"v_mmac", failures, "asm_missing_mmac")
        require(asm, r"s_set_vgpr_size", failures,
                "asm_missing_wdra_vgpr_resize")

    if failures:
        print("dQ kernel gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("dQ kernel gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
