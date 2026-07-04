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
    perf_source = source
    contract = Path(args.contract).read_text(errors="ignore")
    asm_path = Path(args.asm)
    asm = asm_path.read_text(errors="ignore") if asm_path.exists() else ""
    failures: list[str] = []

    require(source, r"fa3_bwd_dkv_ref_softmax_kernel",
            failures, "missing_ref_softmax_kernel")
    require(source, r"fa3_bwd_dkv_ref_output_kernel",
            failures, "missing_ref_output_kernel")
    require(source, r"cpu_reference_dkv", failures,
            "missing_cpu_reference_dkv")
    require(source, r"fa3_bwd_dkv_correctness", failures,
            "missing_correctness_status_line")
    require(perf_source, r"fa3_bwd_dkv_kernel", failures,
            "missing_canonical_dkv_kernel")
    require(perf_source, r"consumer_dkv_mmac_loop", failures,
            "missing_dkv_mmac_consumer_loop")
    require(perf_source, r"score_dp_mmac_owner16", failures,
            "missing_owner16_score_dp_mmac")
    require(perf_source, r"dv_dk_mmac_owner16", failures,
            "missing_owner16_dv_dk_mmac")
    require(source, r"kDkvPathCanonicalDkv", failures,
            "missing_canonical_path")
    require(perf_source, r"hcu_wdra_waves_per_tg\(16\)", failures,
            "missing_wdra16_attribute")
    require(perf_source, r"producer_kq_loop", failures,
            "missing_16wave_kq_producer")
    require(perf_source, r"producer_vdout_loop", failures,
            "missing_16wave_vdout_producer")
    require(perf_source, r"wave_id\s*<\s*4", failures,
            "missing_producer_a_branch")
    require(perf_source, r"wave_id\s*<\s*8", failures,
            "missing_consumer0_branch")
    require(perf_source, r"wave_id\s*<\s*12", failures,
            "missing_consumer1_branch")
    require(perf_source, r"consumer_dkv_mmac_loop<Tile,\s*Bar,\s*1,\s*true>",
            failures, "missing_consumer1_branch")
    require(perf_source, r"s_set_vgpr_size\(Vgpr::kProducerVgprs\)",
            failures,
            "missing_producer_vgpr_window")
    require(perf_source, r"s_set_vgpr_size\(Vgpr::kConsumerVgprs\)",
            failures,
            "missing_consumer_vgpr_window")
    require(perf_source, r"s_abarrier_init\(Bar::kResidentFilled,\s*8\)",
            failures, "missing_resident_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kResidentUsed,\s*8\)",
            failures, "missing_resident_used_count")
    require(perf_source, r"s_abarrier_init\(Bar::kRaw0Filled,\s*8\)",
            failures,
            "missing_raw0_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kRaw0Used,\s*8\)", failures,
            "missing_raw0_used_count")
    require(perf_source, r"q_tile\s*<\s*q_tiles", failures,
            "missing_q_tile_stream_loop")
    require(perf_source, r"abarrier_try_wait<false>\(Bar::kAllDone", failures,
            "missing_all_done_wait")
    require(source, r"matrix_load_32x32_b16_bps_lds", failures,
            "missing_mls_bps_helper")
    require(source, r"ds_read_matrix_trans_pair", failures,
            "missing_ds_read_matrix_helper")
    require(source, r"mmac_f16_lit", failures, "missing_mmac_lit_helper")
    require(source, r"raise_priority_2", failures, "missing_s_setprio_helper")
    require(contract, r"struct\s+DkvBarrierLedger", failures,
            "missing_barrier_ledger")
    require(contract, r"kDkvPathReferenceCorrectness\s*=\s*1", failures,
            "missing_reference_path_contract")
    require(contract, r"kDkvPathCanonicalDkv", failures,
            "missing_canonical_path_contract")
    require(contract, r"template\s*<\s*int\s+BlockMq\s*>", failures,
            "missing_blockmq_template_contract")
    require(contract, r"using\s+ActiveDkvTile\s*=\s*DkvTileD128MqNk128<64>",
            failures, "missing_active_mq64_tile_contract")
    require(contract, r"kRawBuffers\s*=\s*1", failures,
            "canonical_route_must_use_single_raw_buffer")
    require(contract, r"kOverlayRawOnResidentKv", failures,
            "missing_kv_raw_overlay_contract")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*60", failures,
            "missing_active_share_target")
    require(contract, r"kForbidDuplicateScoreDp\s*=\s*true", failures,
            "missing_no_duplicate_contract")
    forbid(source, r"61C\d+|C1\d{2}", failures,
           "clean_source_must_not_define_cxx_phase_stack")
    forbid(perf_source,
           r"matrix_load_32x32_b16_bps_lds\([\s\S]{0,360}?"
           r"wait_lgkm\(0\)[\s\S]{0,160}?abarrier_arrive_cnt",
           failures, "post_mls_wait_before_publication")
    forbid(source, r"kRawFilled|kRawUsed|kTransFilled|kTransUsed|"
           r"kKv0Filled|kKv0Used|kKv1Filled|kKv1Used",
           failures, "legacy_fragment_barrier_tokens")
    forbid(source, r"kPacketAFilled|kPacketAUsed|kPacketBFilled|kPacketBUsed",
           failures, "single_packet_probe_barrier_tokens")
    forbid(source + contract,
           r"kSourceFilled|kSourceUsed|wait_source_|arrive_source_|"
           r"kSource0Filled|kSource0Used|kSource1Filled|kSource1Used",
           failures, "source_layout_must_share_page_packet_token")
    forbid(source + contract, r"kRaw1Filled|kRaw1Used", failures,
           "raw1_tokens_must_not_exist_in_single_buffer_route")
    forbid(source + contract,
           r"kDkvPathWasp|DkvSemanticMq64|"
           r"fa3_bwd_dkv_probe_kernel|fa3_bwd_dkv_mmac_kernel|"
           r"fa3_bwd_dkv_mmac12_|consumer_score_dp_loop|"
           r"consumer_dkv_mmac_loop_(causal_skip|score_dp_brick|"
           r"sidecar_overlay|mq64)|producer_(qk|dout_v)_loop|"
           r"producer_all_loop_(causal_skip|sidecar_overlay|mq64)|"
           r"score_dp_mmac_fragments|softmax_ds_fragment_from_sidecar|"
           r"probe_check|fragment_check|fa3_bwd_dkv_probe",
           failures, "stale_experiment_route_or_probe_code")

    if asm:
        require(asm, r"s_set_vgpr_size", failures, "asm_missing_s_set_vgpr_size")
        require(asm, r"s_abarrier", failures, "asm_missing_s_abarrier")
        require(asm, r"matrix_load_32x32_b16.*bps.*lds", failures,
                "asm_missing_matrix_load_bps_lds")
        require(asm, r"ds_read_matrix_.*format", failures,
                "asm_missing_ds_read_matrix")
        require(asm, r"v_mmac_.*lit", failures, "asm_missing_v_mmac_lit")
        require(asm, r"s_setprio", failures, "asm_missing_s_setprio")
        require(asm, r"fa3_bwd_dkv_kernel", failures,
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
