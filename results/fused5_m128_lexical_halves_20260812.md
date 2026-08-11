# Fused5 M128 Lexical-Halves Result

Date: 2026-08-12

Classification: `REJECT_STATS_TICKS / CANONICAL_RESTORED`.

## What Was Tested

- M128/N128/D128, exact five GEMMs and MMOP92,160 on H1/S1024 causal.
- Two compile-time M64 ownership halves; no runtime panel dispatch.
- K/V LDS reuse for one half-sized P/dS batch and sidecar in proven writer
  padding; total LDS 131,072 bytes.
- Unchanged 16-wave P0/C0/C1/dQ ownership and seven-token ledger.

## Gates

- Static role use `8/157/158/84` inside `8/200/200/88`.
- Metadata SGPR64/VGPR124, private/spill/scratch0.
- H1/S128 causal/noncausal and H1/S1024 causal correctness PASS.
- Exact MMOP92,160 and `ldsBankConflict=0`.

## Performance

The accepted M64 stats runs are 72,208,045 and 71,888,180 fused ticks. The
M128 lexical runs are 72,743,125 and 72,836,855, a mean regression of 1.030%.
MMAC active falls from a 22.714850% mean to 22.173414%.

The candidate reduces SCA `37,808 -> 35,068` and LDS `64,096 -> 63,664`, but
VALU rises `120,800 -> 128,464`. Coissue counts rise without lowering ticks.
A bounded wait relocation runs at 72,748,585 and 73,061,625 ticks and is also
rejected.

## Root Cause

Same-compiler ASM shows the two-half expression loses packed softmax/dS
codegen. `v_pk_fma_f32` disappears and scalar FMA/multiply/subtract sequences
expand. The result is a compiler-expression loss, not duplicated GEMM work:
MMOP remains exact.

## Evidence

- Candidate stats:
  `/zys/shaobo_runs/fused5_m128_lexical_halves_20260812/s1024_stats`
- Candidate repeat:
  `/zys/shaobo_runs/fused5_m128_lexical_halves_20260812/repeat`
- Wait relocation:
  `/zys/shaobo_runs/fused5_m128_wait_relocate_20260812`
- Candidate ASM:
  `/zys/shaobo_runs/fused5_m128_lexical_halves_20260812/asm_audit/m128_wait_relocate.asm`
- Baseline ASM:
  `/zys/shaobo/fa3_bwd_5gemm_toolkit/build/fused5_b28_audit/fused_bwd_kernel.asm`
- Compiler e0f10535, PMD HEAD1694, `GPU_CHIP=sb`, SQ7.

## Decision

Restore `b28e73d`. Do not tune another wait in this expression. A future M128
retry must first recover packed probability/dS VALU in generated ASM; it then
repeats the same correctness/resource/ticks gates before any SQTT capture.
