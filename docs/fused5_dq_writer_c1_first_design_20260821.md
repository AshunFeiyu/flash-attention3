# Fused5 dQ Writer C1-First Ownership Order

## Trace Evidence

On the representative H1/S1024 SIMD, the dQ writer's steady ABarrier pattern
alternates an exposed `1.0--1.7K`-cycle wait with a ready `3--400`-cycle wait.
The exposed wait is group0; group1 normally becomes ready while the writer is
waiting for or consuming group0. The accepted group0 two-generation page
already allows C0 to advance without reusing one physical dS page every tile.

## Single Hypothesis

Consume the complete ready group1 dS batch before group0:

```text
canonical: wait G0 -> 32 dQ MMAC -> Done0 -> wait G1 -> 32 dQ MMAC -> Done1
candidate: wait G1 -> 32 dQ MMAC -> Done1 -> wait G0 -> 32 dQ MMAC -> Done0
```

Group1 MMAC is useful work that can cover C0 publication latency. This is not
the rejected panel-interleave or dual-group-read shape: each group remains one
contiguous four-panel ownership epoch and no wait is added or merged.

## Invariants And Budget

- Exact five-GEMM DAG, MMOP count, M64/N128/D128 tile and output ownership.
- Same dS pages, C0 generation0/generation1 alternation, token counts and LDS.
- Same writer accumulator and fragment live ranges; no WDRA rebudget expected.
- FP32 addition order changes only from `G0+G1` to `G1+G0`; full CPU-golden
  tolerances remain the correctness authority.

## Admission

1. Static metadata remains no-private/spill/scratch with exact MMOP92,160.
2. H1/S128 and H1/S1024 full correctness pass; bank conflict remains zero.
3. Repeated same-mode H1/S1024 fused ticks improve.
4. Winner-only SQTT must show the writer's first exposed ABarrier gap covered
   by group1 MMAC, without moving a larger wait to C1 reuse or terminal time.

## Result

Status: `REJECT_G0_COVERS_G1_READINESS_CANONICAL_RESTORED`.

Static resources are identical to canonical: branch use `9/187/87/182`,
SGPR82/VGPR128 and no private/spill/scratch. H1/S128 and both H1/S1024 runs
pass full correctness with bank conflict zero.

Two interleaved H1/S1024 pairs regress fused mean ticks
`45,172,173 -> 45,737,965` (`+1.25%`) and lifecycle mean ticks
`49,272,860 -> 49,942,848` (`+1.36%`). In the first pair, MMAC active falls
`33.73% -> 33.09%` and barrier share rises `13.73% -> 14.63%`.

The trace premise was incomplete: group1 being ready after the canonical G0
MMAC does not mean group1 is ready at writer entry. G0's 32 dQ MMACs already
cover group1 publication latency. Reversing the order exposes that latency and
removes the useful ownership stagger. Restore G0-first order.
