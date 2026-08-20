#!/usr/bin/env python3
"""Static source/ASM admission gate for one fused five-GEMM BWD kernel."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CANONICAL = "fa3_bwd_5gemm_kernel"
STAGES = ("score", "dp", "dv", "dk", "dq")


def require(text: str, pattern: str, failures: list[str], name: str) -> None:
    if not re.search(pattern, text, flags=re.MULTILINE | re.DOTALL):
        failures.append(name)


def forbid(text: str, pattern: str, failures: list[str], name: str) -> None:
    if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE):
        failures.append(name)


def function_names(text: str) -> list[str]:
    """Return names from simple C/C++ function definitions, not call sites."""
    pattern = re.compile(
        r"^\s*(?:template\s*<[^;{}]*>\s*)?"
        r"[^;{}\n]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
        r"(?:const\s*)?\{",
        flags=re.MULTILINE | re.DOTALL,
    )
    return [match.group(1) for match in pattern.finditer(text)]


def stage_definitions(source: str) -> dict[str, list[str]]:
    """Find named logical stage helpers while ignoring low-level MMAC helpers."""
    definitions = {stage: [] for stage in STAGES}
    for name in function_names(source):
        lowered = name.lower()
        if lowered.startswith(("mmac_", "read_", "load_", "store_")):
            continue
        for stage in STAGES:
            if re.search(rf"(?:^|_){stage}(?:_|$)", lowered):
                definitions[stage].append(name)
    return definitions


def metadata_block(asm: str, symbol: str) -> str:
    blocks = re.split(r"(?=^\s*\.name:\s*)", asm, flags=re.MULTILINE)
    for block in blocks:
        if re.search(rf"^\s*\.name:\s*{re.escape(symbol)}\s*$",
                     block, flags=re.MULTILINE):
            return block
    return ""


def check_asm(asm_path: Path, symbol: str, failures: list[str]) -> str:
    if not asm_path.is_file():
        failures.append("asm_file_missing")
        return ""

    asm = asm_path.read_text(encoding="utf-8", errors="replace")
    require(asm, rf"\b{re.escape(symbol)}\b", failures,
            "asm_missing_canonical_symbol")

    names = re.findall(r"^\s*\.name:\s*(\S+)\s*$", asm,
                       flags=re.MULTILINE)
    if names:
        if names.count(symbol) != 1:
            failures.append(f"asm_canonical_symbol_count={names.count(symbol)}")
        block = metadata_block(asm, symbol)
        if not block:
            failures.append("asm_missing_canonical_metadata")
            return asm
        for line in block.splitlines():
            if re.search(r"private|scratch|spill", line, re.IGNORECASE):
                values = [int(value) for value in re.findall(r"(?:[:=])\s*(\d+)", line)]
                if values and any(value != 0 for value in values):
                    failures.append("asm_nonzero_private_scratch_or_spill")
                    break
    return asm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("src/fused_bwd_kernel.cpp"))
    parser.add_argument("--contract", type=Path,
                        default=Path("include/fused_bwd_contract.h"))
    parser.add_argument("--asm", type=Path,
                        help="optional LLVM AMDGPU assembly/metadata text")
    parser.add_argument("--symbol", default=CANONICAL)
    args = parser.parse_args()
    failures: list[str] = []

    if not args.source.is_file():
        print(f"fused BWD kernel gate: FAIL\n  - source_missing:{args.source}")
        return 1
    if not args.contract.is_file():
        print(f"fused BWD kernel gate: FAIL\n  - contract_missing:{args.contract}")
        return 1

    source = args.source.read_text(encoding="utf-8", errors="replace")
    contract = args.contract.read_text(encoding="utf-8", errors="replace")
    asm = check_asm(args.asm, args.symbol, failures) if args.asm else ""
    code = source + "\n" + asm

    definitions = re.findall(
        rf"^\s*(?:extern\s+\"C\"\s+)?__global__[^;{{}}]*\b"
        rf"({re.escape(args.symbol)})\s*\([^;{{}}]*\)\s*\{{",
        source, flags=re.MULTILINE | re.DOTALL,
    )
    if len(definitions) != 1:
        failures.append(f"canonical_definition_count={len(definitions)}")

    production_symbols = re.findall(
        r"^\s*(?:extern\s+\"C\"\s+)?__global__[^;{}]*\b"
        r"([A-Za-z_]\w*(?:kernel|Kernel))\s*\([^;{}]*\)\s*\{",
        source, flags=re.MULTILINE | re.DOTALL,
    )
    extra_fused = [
        name for name in production_symbols
        if name != args.symbol and re.search(r"(?:fused|5gemm)", name,
                                             re.IGNORECASE)
    ]
    if extra_fused:
        failures.append("extra_production_fused_symbols=" + ",".join(extra_fused))
    require(source, rf"\b{re.escape(args.symbol)}\b", failures,
            "missing_canonical_symbol")

    stages = stage_definitions(source)
    for stage in STAGES:
        require(code, rf"(?i)\b{stage}\b", failures,
                f"missing_{stage}_stage_evidence")
        if not stages[stage]:
            failures.append(f"missing_{stage}_stage_definition")
    for stage in ("score", "dp"):
        if len(stages[stage]) > 1:
            failures.append(
                f"duplicate_{stage}_stage_definitions={len(stages[stage])}"
            )

    for name, pattern in (
        ("missing_mls_bps", r"matrix_load[^\n;{}]*bps[^\n;{}]*lds"),
        ("missing_ds_read_matrix", r"ds_read_matrix"),
        ("missing_mmac", r"(?:v_)?mmac|mmac_f16"),
        ("missing_abarrier_init", r"(?:s_)?abarrier_init"),
        ("missing_abarrier_use", r"(?:s_)?abarrier_(?:arrive|seq|wait|try_wait|inv)"),
    ):
        require(code, pattern, failures, name)

    for name, pattern in (
        ("forbidden_natural_wrong", r"\bnatural_wrong\b"),
        ("forbidden_bpermute", r"\bbpermute\b|__builtin_hcu_bpermute"),
        ("forbidden_explicit_ds_read_b32", r"\bds_read_b32\b"),
        ("forbidden_gather", r"\bgather\b"),
        ("forbidden_generic_atomic", r"\batomicAdd\b|\batomic_add\b"),
        ("forbidden_hcu_global_atomic", r"__builtin_hcu_global_atomic"),
        ("forbidden_fallback", r"\bfallback\b"),
        ("forbidden_phase_version_stack",
         r"production[_ -]?phase|\bphase[_ -]?\d+\b|\bversion[_ -]?\d+\b|"
         r"\b(?:fused|5gemm)_(?:v|ver|version|phase|fallback)\w*_kernel\b"),
    ):
        forbid(source, pattern, failures, name)

    for name, pattern in (
        ("contract_gemm_count", r"kGemmCount\s*=\s*5"),
        ("contract_score_formula", r"kScore\s*=\s*\"score=Q@K\^T\""),
        ("contract_dp_formula", r"kDp\s*=\s*\"dP=dO@V\^T\""),
        ("contract_dv_formula", r"kDv\s*=\s*\"dV=P\^T@dO\""),
        ("contract_dk_formula", r"kDk\s*=\s*\"dK=dS\^T@Q\""),
        ("contract_dq_formula", r"kDq\s*=\s*\"dQ=dS@K\""),
        ("resource_lds_budget", r"k(?:LdsLimitBytes|LdsBudgetBytes|PlannedLdsBytes|SteadyBytes|PhysicalLdsBytes)"),
        ("resource_vgpr_budget", r"k(?:Consumer|Producer).*Vgprs|kPerSimd.*Vgprs"),
        ("resource_no_private", r"kRequireNoPrivateSegment"),
        ("resource_no_scratch", r"kRequireNoScratch"),
        ("resource_no_spill", r"kRequireNoSpill"),
    ):
        require(contract, pattern, failures, name)

    if failures:
        print("fused BWD kernel gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("fused BWD kernel gate: PASS")
    print(f"  canonical: {args.symbol}, GEMMs=score/dp/dv/dk/dq")
    print("  MLS/BPS + ds_read_matrix + MMAC + ABarrier; no fallback/shortcut path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
