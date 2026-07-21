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
    failures: list[str] = []
    if not asm_path.is_file():
        failures.append("asm_file_missing")
    asm = asm_path.read_text(errors="ignore") if asm_path.exists() else ""

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
    require(perf_source, r"owner16_dv_dk_mmac_from_sources", failures,
            "missing_owner16_dv_dk_mmac")
    require(source, r"kDkvPathCanonicalDkv", failures,
            "missing_canonical_path")
    require(perf_source, r"hcu_wdra_waves_per_tg\(16\)", failures,
            "missing_wdra16_attribute")
    require(perf_source,
            r"__builtin_hcu_wdra_init\(\s*"
            r"dkv::WdraResourceWindows::kProducerVgprs,\s*"
            r"dkv::WdraResourceWindows::kConsumerVgprs,\s*"
            r"dkv::WdraResourceWindows::kConsumerVgprs,\s*"
            r"dkv::WdraResourceWindows::kProducerVgprs\s*\)",
            failures, "missing_physical_2p2c_wdra_init")
    require(perf_source, r"producer_resident_raw_loop", failures,
            "missing_resident_raw_producer")
    require(perf_source,
            r"producer_resident_raw_loop<Tile,\s*Bar,\s*false>",
            failures, "missing_k_q_producer")
    require(perf_source,
            r"producer_resident_raw_loop<Tile,\s*Bar,\s*true>",
            failures, "missing_v_dout_sidecar_producer")
    require(perf_source, r"wave_id\s*<\s*4", failures,
            "missing_producer_a_branch")
    require(perf_source, r"wave_id\s*<\s*8", failures,
            "missing_consumer0_branch")
    require(perf_source, r"wave_id\s*<\s*12", failures,
            "missing_consumer1_branch")
    require(perf_source, r"consumer_dkv_mmac_loop<Tile,\s*Bar,\s*1>",
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
    require(perf_source, r"s_abarrier_init\(Bar::kRawHeadFilled,\s*8\)",
            failures, "missing_raw_head_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kRawHeadUsed,\s*8\)",
            failures, "missing_raw_head_used_count")
    require(perf_source, r"s_abarrier_init\(Bar::kRawTailFilled,\s*8\)",
            failures, "missing_raw_tail_filled_count")
    require(perf_source, r"s_abarrier_init\(Bar::kRawTailUsed,\s*8\)",
            failures, "missing_raw_tail_used_count")
    require(perf_source, r"publish_mq_slice<Tile,\s*0,\s*kHeadMBlocks>",
            failures, "missing_qdo_head_publisher")
    require(perf_source,
            r"publish_mq_slice<Tile,\s*kHeadMBlocks,\s*kTotalMBlocks>",
            failures, "missing_qdo_tail_publisher")
    require(perf_source,
            r"publish_sidecar_slice_to_lds<Tile,\s*0,\s*Tile::kHeadReadyMq>",
            failures, "missing_sidecar_head_publisher")
    require(perf_source,
            r"Tile,\s*Tile::kHeadReadyMq,\s*Tile::kTailReadyMq>",
            failures, "missing_sidecar_tail_publisher")
    require(perf_source,
            r"wait_raw_ready<Wdra::kRawTailFilled>[\s\S]{0,500}"
            r"kHeadMBlocks,\s*kTotalMBlocks",
            failures, "missing_tail_readiness_before_tail_consume")
    require(perf_source, r"ds_read_matrix_32x16_trans_dual_base_imm2",
            failures, "missing_dual_base_trans_matrix_read")
    require(perf_source, r"ds_read_matrix_32x16_normal_dual_base_imm2",
            failures, "missing_dual_base_normal_matrix_read")
    require(perf_source, r"ds_read_b128_lds_imm3",
            failures, "missing_sidecar_read3_island")
    require(perf_source,
            r"read_sidecar_owner16<Tile,\s*MBlockBase>[\s\S]{0,500}"
            r"read_owner16_dv_dk_sources<Tile,\s*0,\s*MBlockBase>"
            r"[\s\S]{0,500}wait_lgkm\(4\)[\s\S]{0,500}"
            r"softmax_ds_owner16_tile[\s\S]{0,700}"
            r"wait_lgkm\(0\)[\s\S]{0,300}"
            r"read_owner16_dv_dk_sources<Tile,\s*2,\s*MBlockBase>"
            r"[\s\S]{0,400}owner16_dv_dk_mmac_from_sources",
            failures, "missing_sidecar_matrix_softmax_pipeline")
    require(perf_source,
            r"MBlockBase\s*\+\s*1\s*==\s*Tile::kBlockMq\s*/\s*16",
            failures, "missing_tail_release_contract")
    require(perf_source,
            r"MBlockBase\s*\+\s*1\s*==\s*Tile::kHeadReadyMq\s*/\s*16",
            failures, "missing_head_release_contract")
    require(perf_source,
            r"read_score_dp_owner16_dblock_pair<Tile,\s*NextMBlock,\s*0>"
            r"[\s\S]{0,500}wait_lgkm\(4\)",
            failures, "missing_next_m16_score_prefetch")
    require(perf_source,
            r"!PrefetchNext\s*\|\|\s*\(!ReleaseHead\s*&&\s*!ReleaseTail\)",
            failures, "missing_prefetch_ownership_boundary_gate")
    require(perf_source, r"arrive_raw_used<Wdra::kRawHeadUsed>\(\)",
            failures, "missing_head_lifetime_release")
    require(perf_source, r"arrive_raw_used<Wdra::kRawTailUsed>\(\)",
            failures, "missing_tail_lifetime_release")
    require(perf_source,
            r"wait_raw_used<Wdra::kRawHeadUsed>[\s\S]{0,2400}"
            r"arrive_raw_ready<Wdra::kRawHeadFilled>[\s\S]{0,500}"
            r"wait_raw_used<Wdra::kRawTailUsed>[\s\S]{0,2400}"
            r"arrive_raw_ready<Wdra::kRawTailFilled>",
            failures, "missing_split_used_cross_packet_order")
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
    require(contract, r"kBlockMq\s*%\s*kWaveSize\s*==\s*0", failures,
            "missing_full_wave_sidecar_coverage_contract")
    require(contract, r"kHeadReadyMq\s*=\s*64", failures,
            "missing_head64_readiness_contract")
    require(contract,
            r"kTailReadyMq\s*=\s*kBlockMq\s*-\s*kHeadReadyMq",
            failures, "missing_tail128_readiness_contract")
    require(contract, r"kNkPerConsumerWave\s*=\s*16", failures,
            "missing_owner16_contract")
    require(contract, r"kConsumerGroups\s*=\s*2", failures,
            "missing_two_physical_consumer_groups_contract")
    require(contract, r"kLogicalConsumer0Rows\s*=\s*64", failures,
            "missing_logical_consumer0_rows")
    require(contract, r"kLogicalConsumer1Rows\s*=\s*32", failures,
            "missing_logical_consumer1_rows")
    require(contract, r"kLogicalConsumer2Rows\s*=\s*32", failures,
            "missing_logical_consumer2_rows")
    require(contract, r"kResidentNk\s*=", failures,
            "missing_resident_nk_contract")
    require(contract, r"kResidentNk\s*==\s*128", failures,
            "missing_resident_nk128_assert")
    require(contract, r"kRawBuffers\s*=\s*1", failures,
            "missing_active_rawbuffer1_contract")
    require(source, r"softmax_ds_owner16<ApplyCausalMask>", failures,
            "missing_boundary_selective_softmax_helper")
    require(source,
            r"consume_m16_owner16_tile<[\s\S]{0,180}FirstQTile",
            failures, "missing_first_tile_causal_mask_contract")
    require(source,
            r"q_base\s*=\s*k_tile\s*\*\s*Tile::kBlockMq",
            failures, "missing_causal_q_start")
    require(source,
            r"q_tiles\s*=\s*seqlen\s*/\s*Tile::kBlockMq\s*-\s*k_tile",
            failures, "missing_causal_q_tile_count")
    require(contract, r"kOverlayRawOnResidentKv", failures,
            "missing_kv_raw_overlay_contract")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*60", failures,
            "missing_active_share_target")
    require(contract, r"kForbidDuplicateScoreDp\s*=\s*true", failures,
            "missing_no_duplicate_contract")
    require(contract,
            r"kTotalMmacPerConsumer\s*==\s*2\s*\*\s*kBlockMq",
            failures, "missing_exact_four_gemm_mmac_contract")
    forbid(source, r"61C\d+|C1\d{2}", failures,
           "clean_source_must_not_define_cxx_phase_stack")
    forbid(perf_source,
           r"matrix_load_32x32_b16_bps_lds\([\s\S]{0,360}?"
           r"wait_lgkm\(0\)[\s\S]{0,160}?abarrier_arrive_cnt",
           failures, "post_mls_wait_before_publication")
    forbid(source, r"kQ0Filled|kQ0Used|kQ1Filled|kQ1Used|"
           r"kDout0Filled|kDout0Used|kDout1Filled|kDout1Used|"
           r"kTransFilled|kTransUsed|kKv0Filled|kKv0Used|"
           r"kKv1Filled|kKv1Used",
           failures, "legacy_fragment_barrier_tokens")
    forbid(source, r"kPacketAFilled|kPacketAUsed|kPacketBFilled|kPacketBUsed",
           failures, "single_packet_probe_barrier_tokens")
    forbid(source + contract,
           r"kSourceFilled|kSourceUsed|wait_source_|arrive_source_|"
           r"kSource0Filled|kSource0Used|kSource1Filled|kSource1Used",
           failures, "source_layout_must_share_page_packet_token")
    forbid(source + contract, r"\bkRawFilled\b", failures,
           "unsplit_raw_filled_token_must_not_remain")
    forbid(source + contract, r"\bkRawUsed\b", failures,
           "combined_raw_used_token_must_not_remain")
    require(source, r"publish_resident_tile", failures,
            "missing_nk128_resident_publisher")
    require(source, r"latch_owner16_kv_regs", failures,
            "missing_owner16_kv_latch")
    require(source, r"store_dkv_owner16", failures,
            "missing_owner16_store")
    forbid(source, r"Owner32|owner32", failures,
           "owner32_route_must_not_remain")
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
        require(asm, r"ds_read_b128", failures,
                "asm_missing_sidecar_ds_read_b128")
        forbid(asm, r"s_waitcnt lgkmcnt\(8\)", failures,
               "asm_d1_wait8_must_not_remain")
        require(asm, r"v_mmac_.*lit", failures, "asm_missing_v_mmac_lit")
        require(asm, r"s_setprio", failures, "asm_missing_s_setprio")
        require(asm, r"fa3_bwd_dkv_kernel", failures,
                "asm_missing_kernel_symbol")
        forbid(asm, r"^\s*s_trap\b", failures,
               "asm_contains_real_wdra_trap")

    if failures:
        print("dKV kernel gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("dKV kernel gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
