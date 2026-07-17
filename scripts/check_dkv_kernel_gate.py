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
    require(perf_source, r"score_dp_mmac_owner32", failures,
            "missing_owner32_score_dp_mmac")
    require(perf_source, r"owner32_dv_dk_read_mmac", failures,
            "missing_owner32_dv_dk_mmac")
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
    require(perf_source, r"consumer_dkv_mmac_loop<Tile,\s*Bar,\s*1>",
            failures, "missing_consumer1_branch")
    require(perf_source, r"s_set_vgpr_size\(Vgpr::kProducerKqVgprs\)",
            failures, "missing_kq_producer_vgpr_window")
    require(perf_source, r"s_set_vgpr_size\(Vgpr::kConsumer0Vgprs\)",
            failures, "missing_c0_read8_vgpr_window")
    require(perf_source, r"s_set_vgpr_size\(Vgpr::kConsumer1Vgprs\)",
            failures, "missing_c1_canonical_vgpr_window")
    require(perf_source, r"s_set_vgpr_size\(Vgpr::kProducerVdoutVgprs\)",
            failures, "missing_vdout_producer_vgpr_window")
    require(perf_source, r"constexpr bool kBatchDvDkSources\s*=\s*true",
            failures, "missing_symmetric_dvdk_read8_schedule")
    require(perf_source,
            r"if constexpr \(BatchDvDkSources\)[\s\S]{0,500}"
            r"read_owner32_dv_dk_sources<Tile, 0, MBlockBase>"
            r"[\s\S]{0,240}read_owner32_dv_dk_sources<Tile, 2, MBlockBase>"
            r"[\s\S]{0,120}wait_lgkm\(0\)", failures,
            "missing_read8_before_single_wait")
    require(perf_source, r"s_abarrier_init\(Bar::kResidentFilled,\s*8\)",
            failures, "missing_resident_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kResidentUsed,\s*8\)",
            failures, "missing_resident_used_count")
    require(perf_source, r"s_abarrier_init\(Bar::kQ0Filled,\s*8\)",
            failures, "missing_q0_combined_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kQ0Used,\s*8\)",
            failures, "missing_q0_used_count")
    forbid(perf_source, r"s_abarrier_init\(Bar::kDout0Filled,",
           failures, "dout0_filled_should_share_q0_filled")
    require(perf_source, r"s_abarrier_init\(Bar::kDout0Used,\s*8\)",
            failures, "missing_dout0_used_count")
    require(perf_source, r"s_abarrier_init\(Bar::kQ1Filled,\s*8\)",
            failures, "missing_q1_combined_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kQ1Used,\s*8\)",
            failures, "missing_q1_used_count")
    forbid(perf_source, r"s_abarrier_init\(Bar::kDout1Filled,",
           failures, "dout1_filled_should_share_q1_filled")
    require(perf_source, r"s_abarrier_init\(Bar::kDout1Used,\s*8\)",
            failures, "missing_dout1_used_count")
    require(perf_source, r"publish_mq_half_tile<Tile,\s*0>", failures,
            "missing_qdo_half0_publisher")
    require(perf_source, r"publish_mq_half_tile<Tile,\s*1>", failures,
            "missing_qdo_half1_publisher")
    require(perf_source, r"publish_sidecar_half_tile_to_lds<Tile,\s*0>",
            failures, "missing_sidecar_half0_publisher")
    require(perf_source, r"publish_sidecar_half_tile_to_lds<Tile,\s*1>",
            failures, "missing_sidecar_half1_publisher")
    require(perf_source, r"arrive_dout_half_used<Wdra,\s*Half>\(\)",
            failures, "missing_dout_half_lifetime_release")
    require(perf_source, r"arrive_q_half_used<Wdra,\s*Half>\(\)",
            failures, "missing_q_half_lifetime_release")
    require(perf_source, r"q_tile\s*<\s*q_tiles", failures,
            "missing_q_tile_stream_loop")
    require(perf_source,
            r"abarrier_try_wait<false>\(Bar::kAllDone",
            failures,
            "missing_all_done_wait")
    require(perf_source,
            r"__syncthreads\(\);\s*if\s*\(\s*wave_id\s*==\s*0\s*\)"
            r"\s*\{[\s\S]{0,900}s_abarrier_inv\(Bar::kResidentFilled\)",
            failures, "missing_wave0_terminal_invalidate")
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
    require(contract, r"struct\s+ActiveDkvTile", failures,
            "missing_active_tile_contract")
    require(contract, r"kBlockMq\s*=\s*128", failures,
            "missing_active_mq128_contract")
    require(contract, r"kNkPerConsumerWave\s*=\s*32", failures,
            "missing_owner32_contract")
    require(contract, r"kResidentNk\s*=", failures,
            "missing_resident_nk_contract")
    require(contract, r"kRawBuffers\s*=\s*1", failures,
            "missing_active_rawbuffer1_contract")
    require(contract, r"kProducerKqVgprs\s*=\s*16", failures,
            "missing_p0_16_vgpr_contract")
    require(contract, r"kConsumer0Vgprs\s*=\s*244", failures,
            "missing_c0_244_vgpr_contract")
    require(contract, r"kConsumer1Vgprs\s*=\s*244", failures,
            "missing_c1_244_vgpr_contract")
    require(contract, r"kProducerVdoutVgprs\s*=\s*8", failures,
            "missing_p1_8_vgpr_contract")
    require(source, r"softmax_ds_owner32_causal_exact_tile", failures,
            "missing_causal_exact_tile_helper")
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
    require(source, r"publish_resident_tile", failures,
            "missing_nk256_resident_publisher")
    require(source, r"latch_owner32_kv_regs", failures,
            "missing_owner32_kv_latch")
    require(source, r"store_dkv_owner32", failures,
            "missing_owner32_store")
    forbid(source, r"Owner16|owner16", failures,
           "owner16_route_must_not_remain")
    forbid(source, r"dkv_barrier_tomography", failures,
           "canonical_source_must_not_include_tomography")
    forbid(source, r"ds_read_b32|ds_bpermute|ds_permute", failures,
           "non_native_matrix_workaround_in_main_source")
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
