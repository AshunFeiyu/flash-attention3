# Fused5 dQ Workspace Reduction Result

Date: 2026-08-12

Decision: `ACCEPT_CANONICAL_STRUCTURAL_WIN`.

## Hypothesis

The persistent dQ writer serialized 9,216 FP32 global atomics and delayed the
CTA terminal barrier. Replace those atomics with uniquely owned FP32 partial
stores and charge a separate vector reduction dispatch in the complete
lifecycle.

The five GEMMs, M64/N128/D128 tile, 16-wave roles, ABarrier ledger, formulas,
and MMOP count are unchanged.

## Gates

- H1/S128 causal: dQ/dK/dV CPU golden PASS.
- H1/S128 noncausal: dQ/dK/dV CPU golden PASS.
- H1/S1024 causal: dQ/dK/dV CPU golden PASS.
- Compute: SGPR60/VGPR124/LDS115,456 B, role `8/163/166/86`.
- Reduction: SGPR26/VGPR36/LDS0.
- Both: private/spill/scratch0 and bank0.
- Compute MMOP92,160; reduction MMOP0; compute atomic opcode count0.

## PMD Result

Repeated stats complete ticks:

| Run | Compute | Reduction | Total |
|---|---:|---:|---:|
| 1 | 55,818,945 | 2,703,610 | 58,522,555 |
| 2 | 56,188,860 | 2,680,860 | 58,869,720 |
| Mean | 56,003,902.5 | 2,692,235.0 | 58,696,137.5 |

The accepted atomic baseline mean is 72,048,112.5 ticks. The complete mean
improves 18.532%. Fullperf is 56,162,925 + 2,785,965 = 58,948,890 ticks,
18.070% below its 71,950,060 baseline.

Compute MMAC active rises from 22.725077% to 28.851332% in fullperf. Dynamic
MMOP remains 92,160 and FLAT falls from 10,528 to 3,616.

## XCU Attribution

- Atomic issue gaps disappear: baseline 12.60% atomic-to-atomic plus 3.14%
  address-to-atomic.
- Terminal ebarrier issue gap falls 11.49% -> 6.12%.
- XCU duration falls 158,132 -> 123,436 cycles.
- Candidate compute still has a 40.72% ABarrier-following issue gap.
- Matrix read first-use gaps remain 7.39% trans plus 6.83% normal.
- C0/C1 bins remain substantially aligned: 162 MMAC-vs-MMAC versus 139
  MMAC-vs-VALU. The previous baseline was 158 versus 135.

Therefore the gain is an output-ownership win, not a consumer-pipeline
reordering win. The next hypothesis must shorten the dominant ABarrier
ownership interval without restoring atomic output or changing exact work.

## Evidence

- Stats root:
  `/zys/shaobo_runs/fused5_dq_workspace_reduction_20260812/stats`
- Fullperf:
  `/zys/shaobo_runs/fused5_dq_workspace_reduction_20260812/fullperf/5gemm_owner_s1024_c1_fullperf_20260812_034221`
- XCU:
  `/zys/shaobo_runs/fused5_dq_workspace_reduction_20260812/xcu`
- Compiler: LLVM `e0f10535`; PMD HEAD1694; `GPU_CHIP=sb`; SQC prefetch7.
