# Fused5 Group-Local dK Read-Ahead

Date: 2026-08-12  
Decision: `ACCEPT_TICKS_AND_ACTIVE_WIN`

## Hypothesis

The dK tail was four independent `Q/dS read -> wait_lgkm(0) -> 8 MMAC`
islands. Keep one current Q/dS fragment pair and issue the next pair while
the current dK MMAC island is active. The five-GEMM DAG, source layout,
output ownership, and ABarrier tokens remain unchanged.

## Gates

- H1/S128 causal full lifecycle: PASS.
- H1/S1024 causal full lifecycle: PASS.
- Exact MMOP `92,160`; `ldsBankConflict=0`.
- No private segment, scratch, SGPR spill, or VGPR spill.
- Main path remains MLS/BPS + `ds_read_matrix` + MMAC.

## H1/S1024 PMD stats

Compiler `e0f10535`, PMD `HEAD_1694`, `GPU_CHIP=sb`, `SQCIPfLines=7`.

| Run | fused ticks | full lifecycle ticks | MMAC active |
|---|---:|---:|---:|
| 1 | 47,682,635 | 52,819,130 | 33.355944% |
| 2 | 47,247,200 | 52,419,640 | 33.360532% |
| 3 | 47,749,065 | 53,086,215 | 33.415005% |

Mean fused ticks: `47,559,633`; mean full lifecycle ticks: `52,774,995`.
The three samples keep `VALU=127,352`, `SCA=58,720`, `LDS=63,872`,
`VMEM=1,408`, `FLAT=3,616`, and `MMOP=92,160` unchanged. Mean wait shares
are approximately `waitVm=2.20%`, `waitLgkm=9.90%`, and `barrier=15.11%`.

## Interpretation

This is a real local readiness improvement, not a reduction of useful work.
It raises MMAC active to about `33.38%` and improves the accepted raw-page
stats baseline. It does not reach the 50% goal; remaining debt is dQ-writer
and cross-island VALU/SCA scheduling. These runs produced stats only and no
`.perf`, so no SQTT claim is made.

The attempted two-panel score read batch was rejected: it raised consumer
VGPR usage and `waitLgkm`, with MMAC active falling to `32.53%`. The attempted
single-read dO reuse was rejected by H1/S128 correctness because dV needs a
normal dO fragment while dP needs the transposed source-layout fragment.

Next: evaluate dQ-writer read-ahead only after a WDRA ledger proves it fits
the 88-VGPR writer branch without spilling.
