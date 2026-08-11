# Fused5 Raw-Page Compile-Time Specialization Result

Date: 2026-08-12

Decision: `REJECT_TICKS_REGRESSION_CANONICAL_RESTORED`.

## Hypothesis

Traverse raw page0/page1 in compile-time-specialized pairs to remove the
runtime page-base, barrier-ID and sidecar-page branches introduced by the
accepted double-page ownership. Keep all formulas, tokens, pages, MMOP and
output ownership unchanged.

## Gates

- H1/S128 causal and noncausal dQ/dK/dV CPU golden PASS.
- Two H1/S1024 causal runs PASS.
- Actual role use `12/167/169/86`; SGPR70/VGPR128/LDS131,072B.
- private/spill/scratch0, bank0, compute MMOP92,160, reduction MMOP0.

## PMD Result

| Run | Compute | Reduction | Total | MMAC active |
|---|---:|---:|---:|---:|
| 1 | 50,379,875 | 2,957,500 | 53,337,375 | 31.673937% |
| 2 | 50,706,565 | 2,700,880 | 53,407,445 | 31.584976% |
| Mean | 50,543,220 | 2,829,190 | 53,372,410 | 31.629456% |

Versus accepted `d62c645`, compute regresses 0.503% and the complete lifecycle
regresses 0.310%. A 0.153 pp MMAC-active increase is not an optimization when
same-shape ticks rise.

## Attribution

The intended control reduction is real: dynamic SCA falls
58,336 -> 38,988. It is outweighed by generated-code and resource expansion:

- static kernel instruction lines 2,076 -> 4,255 (+104.96%);
- SGPR60 -> 70;
- producer/C0/C1 actual VGPR `9/161/163` -> `12/167/169`;
- dynamic VALU 127,352 -> 131,120;
- barrier share rises 15.940946% -> 16.298731%;
- lgkm wait share rises about 0.466 pp.

The source is restored to `d62c645`. Stop page-control micro-tuning; preserve
the accepted raw double-page ownership and next redesign useful C0/C1 work
staggering without duplicating the packet body.

## Evidence

- Runs:
  `/zys/shaobo_runs/fused5_raw_page_specialization_20260812`
- Build:
  `/zys/shaobo/fa3_bwd_5gemm_toolkit/build/fused5_raw_page_specialization`
- Compiler: LLVM `e0f10535`; PMD HEAD1694; `GPU_CHIP=sb`; SQC prefetch7.
