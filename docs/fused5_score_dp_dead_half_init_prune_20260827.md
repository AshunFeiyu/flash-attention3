# Fused5 score/dP dead-half initialization prune

Status: `REJECT_STATIC_NO_ISA_CHANGE_CANONICAL_RESTORED`.

## Hypothesis

Each score and dP product produces one `Vec4F16` in `Fragment::f16x4[0]`.
Only `score.scalar[0..3]` and `dp.scalar[0..3]` feed probability and dS.
`Fragment::f16x4[1]` is therefore dead for these two products, but the current
source zero-initializes the whole local fragment and writes the high half again
inside `f16_mmac_single`.

Remove only those dead-half initializations. Keep the shared role-local MMAC
zero seed for the first low-half accumulation.

## Invariants

- Exact five-GEMM DAG, M64/N128/D128 tile and 16-wave ownership are unchanged.
- MMAC, matrix-read, MLS/BPS, LDS and ABarrier counts are unchanged.
- Probability and dS still explicitly zero their required high halves before
  native writer/reader transport.
- Promotion requires full correctness, no spill/private/scratch, bank0 and
  lower repeated same-shape ticks.

## Static admission

The generated fused symbol must retain the exact MMAC and matrix-read counts
while reducing `v_mov_b64` or another mapped zero-initialization instruction.
If generated ISA is unchanged, reject before PMD.

## Result

Compiler `e0f10535` already removes the dead score/dP high-half writes. The
candidate and C111 control have identical causal-symbol instruction counts:

- MMAC: `1,344`
- matrix read: `768`
- `v_mov_b32_e32`: `157`
- `v_mov_b64_e32`: `42`
- zero `v_mov_b64`: `10`
- role VGPR use: `9/142/87/130`
- metadata: SGPR72, VGPR128, private/spill/scratch0

After removing the object filename line, `llvm-objdump -d` output has the same
SHA256 for control and candidate:
`0b2f1721b4c9595216e4a0f30601571743cb05712c9b7d2258e87a5d885d7be69`.
No PMD run is admitted because the generated machine instruction stream is
unchanged. Canonical source is restored.

## Evidence

- control: `/zys/fa3_bwd_main_dot2_a3_20260827`
- candidate: `/zys/fa3_bwd_score_dp_deadhalf_20260827`
- compiler: `e0f10535`
- command: `scripts/analyze_asm_vmov.py` plus normalized `llvm-objdump -d`

Do not repeat source-level score/dP dead-half initialization pruning under this
compiler. A future attempt requires a different generated-ISA mechanism, not
another C++ initialization spelling.
