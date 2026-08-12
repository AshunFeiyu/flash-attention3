# Fused5 dKV MMAC Zero-Seed Experiment

Date: 2026-08-12

Status: `REJECT_TICKS_REGRESSION_CANONICAL_RESTORED`.

## Hypothesis

Use the existing branch-local `mmac_zero` as the accumulator input for the
first dV and dK MMAC of the first q tile, then keep the dV/dK accumulators live
for all later q tiles. This removes explicit dKV accumulator zero moves while
preserving the five-GEMM accumulation semantics.

## Gates

- Canonical source: `b626236`, two-slot dK read-ahead.
- Shape: `B=1,H=1,S=128/1024,D=128,causal=true`.
- Environment: `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`, LLVM
  `e0f10535`, PMD `HEAD1694`.
- Required: full lifecycle correctness, exact MMOP, no private/spill/scratch,
  `ldsBankConflict=0`.

## Evidence

Static resource and gate result for the candidate:

```text
producer=9/16, consumer0=165/204, consumer1=167/204, dq_writer=86/88
private=0, sgpr_spill=0, vgpr_spill=0, scratch=0, bank=0, MMOP=92160
```

Correctness passed for H1/S128 and H1/S1024, including dot, dV, dK and dQ.
H1/S1024 candidate runs were:

```text
fused ticks: 47841885, 48595365, 48191325
full ticks:  52989755, 53823315, 53292330
```

The candidate fused mean is about `48.21M` versus the canonical two-slot mean
`47.56M`; full lifecycle mean is about `53.37M` versus `52.77M`. PMD stats on
the first candidate run reported `MMAC active=33.033605%`,
`waitLgkm=9.178212%`, `barrier=15.697455%`, and `coissue=20406/24910`.

## Decision

`REJECT_TICKS_REGRESSION_CANONICAL_RESTORED`.

The dKV zero seed is semantically valid and resource-clean, but the compiler
does not turn this long-lived first-update template into a faster steady
pipeline. The reduction in explicit zero moves is not enough to offset the
new first-tile control/accumulator dependency. Keep zero-seed as a fixed
first-MMAC rule for score/dP islands, but do not use this dKV form on the
canonical path.

## Boundary

This result does not disprove zero-seed for a compile-time peeled, fixed
MMAC island. It rejects only the runtime `qi/m_block` dispatch around
long-lived dV/dK accumulators in this fused5 M64/N128 topology.
