#!/usr/bin/env python3
"""Static gate for the clean FA3 BWD dKV implementation."""

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
    parser.add_argument("--source", default="src/dkv_kernel.cpp")
    parser.add_argument("--contract", default="include/dkv_contract.h")
    parser.add_argument("--asm", default="build/fa3_bwd_wasp_clean.asm")
    args = parser.parse_args()

    source = Path(args.source).read_text(errors="ignore")
    contract = Path(args.contract).read_text(errors="ignore")
    asm_path = Path(args.asm)
    asm = asm_path.read_text(errors="ignore") if asm_path.exists() else ""
    failures: list[str] = []

    require(source, r"fa3_bwd_dkv_probe_kernel",
            failures, "missing_probe_kernel")
    require(source, r"hcu_wdra_waves_per_tg\(16\)", failures,
            "missing_wdra_attribute")
    require(source, r"producer_qk_loop", failures, "missing_producer_qk_loop")
    require(source, r"producer_dout_v_loop", failures,
            "missing_producer_dout_v_loop")
    require(source, r"consumer_score_dp_loop", failures,
            "missing_consumer_score_dp_loop")
    require(source, r"score_dp_mmac_probe", failures,
            "missing_score_dp_mmac_probe")
    require(source, r"wave_id\s*<\s*4", failures, "missing_producer_a_branch")
    require(source, r"wave_id\s*<\s*8", failures, "missing_consumer0_branch")
    require(source, r"wave_id\s*<\s*12", failures, "missing_consumer1_branch")
    require(source, r"s_set_vgpr_size\(Vgpr::kProducerVgprs\)", failures,
            "missing_producer_vgpr_window")
    require(source, r"s_set_vgpr_size\(Vgpr::kConsumerVgprs\)", failures,
            "missing_consumer_vgpr_window")
    require(source, r"s_abarrier_init\(Bar::kRawFilled,\s*4\)", failures,
            "missing_q_filled_count")
    require(source, r"s_abarrier_init\(Bar::kTransFilled,\s*4\)", failures,
            "missing_dout_filled_count")
    require(source, r"s_abarrier_init\(Bar::kKv0Filled,\s*4\)", failures,
            "missing_k_filled_count")
    require(source, r"s_abarrier_init\(Bar::kKv1Filled,\s*4\)", failures,
            "missing_v_filled_count")
    require(source, r"abarrier_try_wait<false>\(Bar::kAllDone", failures,
            "missing_all_done_wait")
    require(source, r"matrix_load_32x32_b16_bps_lds", failures,
            "missing_mls_bps_helper")
    require(source, r"ds_read_matrix_trans_pair", failures,
            "missing_ds_read_matrix_helper")
    require(source, r"mmac_f16_lit", failures, "missing_mmac_lit_helper")
    require(source, r"raise_priority_2", failures, "missing_s_setprio_helper")
    require(contract, r"struct\s+DkvBarrierLedger", failures,
            "missing_barrier_ledger")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*60", failures,
            "missing_active_share_target")
    require(contract, r"kForbidDuplicateScoreDp\s*=\s*true", failures,
            "missing_no_duplicate_contract")
    forbid(source, r"61C\d+|C1\d{2}", failures,
           "clean_source_must_not_define_cxx_phase_stack")
    forbid(source,
           r"matrix_load_32x32_b16_bps_lds\([\s\S]{0,360}?"
           r"wait_lgkm\(0\)[\s\S]{0,160}?abarrier_arrive_cnt",
           failures, "post_mls_wait_before_publication")

    if asm:
        require(asm, r"s_set_vgpr_size", failures, "asm_missing_s_set_vgpr_size")
        require(asm, r"s_abarrier", failures, "asm_missing_s_abarrier")
        require(asm, r"matrix_load_32x32_b16.*bps.*lds", failures,
                "asm_missing_matrix_load_bps_lds")
        require(asm, r"ds_read_matrix_.*format", failures,
                "asm_missing_ds_read_matrix")
        require(asm, r"v_mmac_.*lit", failures, "asm_missing_v_mmac_lit")
        require(asm, r"s_setprio", failures, "asm_missing_s_setprio")
        require(asm, r"fa3_bwd_dkv_probe_kernel", failures,
                "asm_missing_kernel_symbol")

    if failures:
        print("dKV kernel gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("dKV kernel gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
