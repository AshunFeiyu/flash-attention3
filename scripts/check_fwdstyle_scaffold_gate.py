#!/usr/bin/env python3
"""Static gate for the clean Stage61 dKV FWD-style scaffold."""

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
    parser.add_argument("--source", default="src/stage61_dkv_fwdstyle.cpp")
    parser.add_argument("--contract", default="include/stage61_dkv_contract.h")
    parser.add_argument("--asm", default="build/fa3_bwd_wasp_fwdstyle_clean.asm")
    args = parser.parse_args()

    source = Path(args.source).read_text(errors="ignore")
    contract = Path(args.contract).read_text(errors="ignore")
    asm_path = Path(args.asm)
    asm = asm_path.read_text(errors="ignore") if asm_path.exists() else ""
    failures: list[str] = []

    require(source, r"bwd_dkv_stage61_fwdstyle_scaffold_kernel",
            failures, "missing_kernel")
    require(source, r"hcu_wdra_waves_per_tg\(16\)", failures,
            "missing_wdra_attribute")
    require(source, r"wave_id\s*<\s*4", failures, "missing_producer_a_branch")
    require(source, r"wave_id\s*<\s*8", failures, "missing_consumer0_branch")
    require(source, r"wave_id\s*<\s*12", failures, "missing_consumer1_branch")
    require(source, r"s_set_vgpr_size\(Vgpr::kProducerVgprs\)", failures,
            "missing_producer_vgpr_window")
    require(source, r"s_set_vgpr_size\(Vgpr::kConsumerVgprs\)", failures,
            "missing_consumer_vgpr_window")
    require(source, r"s_abarrier_init\(Bar::kRawFilled,\s*8\)", failures,
            "missing_raw_filled_count")
    require(source, r"s_abarrier_try_wait\(Bar::kAllDone,\s*0\)", failures,
            "missing_all_done_wait")
    require(contract, r"struct\s+DkvBarrierLedger", failures,
            "missing_barrier_ledger")
    require(contract, r"kTargetMmacActiveSharePercent\s*=\s*60", failures,
            "missing_active_share_target")
    require(contract, r"kForbidDuplicateScoreDp\s*=\s*true", failures,
            "missing_no_duplicate_contract")
    forbid(source, r"61C\d+|C1\d{2}", failures,
           "clean_source_must_not_define_cxx_phase_stack")

    if asm:
        require(asm, r"s_set_vgpr_size", failures, "asm_missing_s_set_vgpr_size")
        require(asm, r"s_abarrier", failures, "asm_missing_s_abarrier")
        require(asm, r"bwd_dkv_stage61_fwdstyle_scaffold_kernel", failures,
                "asm_missing_kernel_symbol")

    if failures:
        print("FWD-style scaffold gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("FWD-style scaffold gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
