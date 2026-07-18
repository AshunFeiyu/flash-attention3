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
    if not asm_path.exists():
        failures.append("asm_file_missing")

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
    require(source, r"hcu_wdra_waves_per_tg\(16\)", failures,
            "missing_16wave_wdra_attribute")
    require(source, r"dq_consumer_full3gemm_role", failures,
            "missing_full3gemm_consumer")
    require(source, r"dq_load_sidecar_group", failures,
            "missing_producer_sidecar_lds_staging")
    require(source, r"dq_wait_qdo_filled", failures,
            "missing_consumer_qdo_filled_wait")
    require(source, r"dq_arrive_qdo_filled", failures,
            "missing_producer_qdo_filled_arrive")
    require(source, r"kSidecarBase\s*=\s*kQDoBytes", failures,
            "missing_sidecar_after_qdo_startup_layout")
    require(source, r"kPage0Base\s*=\s*kQBase", failures,
            "missing_kv_page0_startup_overlay")
    require(source, r"dq_update_from_ds_(?:vec|pair)", failures,
            "missing_vgpr_ds_to_dq_mmac")
    require(source, r"CANONICAL_DQ", failures,
            "missing_canonical_standalone_switch")
    require(contract, r"using\s+ActiveDqTile\s*=\s*DqTileD128MqNk<192,\s*128>",
            failures, "missing_active_mq192_nk128_tile")
    require(contract, r"kStartupLdsBytes", failures,
            "missing_latched_sidecar_startup_lds_budget")
    require(contract, r"kSteadyLdsBytes", failures,
            "missing_kv_doublepage_steady_lds_budget")
    require(contract, r"kQDoFilled", failures,
            "missing_qdo_filled_startup_barrier")
    require(contract, r"kForbidDuplicateScoreDpInsideDq\s*=\s*true",
            failures, "missing_no_duplicate_score_dp_contract")
    require(contract, r"kForbidDqAtomicAdd\s*=\s*true",
            failures, "missing_no_atomic_contract")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*50",
            failures, "missing_mmac_active_target")
    require(contract, r"kConsumerGroups\s*=\s*3", failures,
            "missing_three_consumer_contract")
    require(contract, r"kConsumerTargetVgprs\s*=\s*160", failures,
            "missing_three_consumer_vgpr_window")
    require(contract, r"kPlannedLdsBytes\s*=", failures,
            "missing_lds_budget_contract")
    require(source, r"dq_out", failures, "missing_dq_output_name")
    forbid(source, r"atomicAdd|atomic_add|global_atomic", failures,
           "dq_bringup_must_not_use_atomic")
    forbid(source, r"dq_load_kt_tile|dq_store_kt_tile_scalar|kKtBase|"
                   r"materialize_k_t_source|k_t_source",
           failures, "canonical_dq_must_not_restore_kt_lds_page")
    forbid(source + contract, r"dq_publish_ds_chunk|dq_consume_ds|DsFilled|"
                              r"kDsBase|kDsPageBytes|KToDs",
           failures, "canonical_dq_must_not_stage_ds_in_lds")
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
        require(asm, r"ds_read_matrix_format", failures,
                "asm_missing_ds_read_matrix_normal")
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
