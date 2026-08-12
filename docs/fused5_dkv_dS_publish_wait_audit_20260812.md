# Fused5 dS Publish Wait Audit

Date: 2026-08-12

## Hypothesis

The four `ds_write_matrix` operations that publish one dS panel generation
may be ordered for peer waves by `BatchDsFilled` ABarrier. Remove only the
producer-side `s_waitcnt lgkmcnt(0)` immediately before the local
`BatchDsFilled` arrival and test whether the ABarrier publish edge also covers
LDS write readiness.

No formula, tile, MMAC count, LDS address, output owner, or barrier token is
changed. The consumer and dQ writer waits remain unchanged.

## Required Gates

- H1/S128 full lifecycle correctness before any timing.
- H1/S1024 correctness and stats only if H1/S128 passes.
- Exact MMOP92,160, no spill/private/scratch, bank0, native matrix path.

## Expected Interpretation

- Correctness failure means ABarrier arrival does not replace producer-local
  LDS write readiness for this ds-write path. Restore the wait.
- Correctness pass but ticks regression means the wait is required for useful
  scheduling even if visibility is preserved.
- Correctness and ticks win would admit the change only after repeated S1024
  stats; no coissue claim is made from one run.
