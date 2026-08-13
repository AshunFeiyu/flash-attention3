# Fused5 dK Read-Island Batch2

Status: `HYPOTHESIS_OPEN`

## Scope

One canonical-source scheduling experiment on the existing five-GEMM kernel.
The tile, role ownership, LDS addresses, output ownership, and ABarrier ledger
remain unchanged.

## Evidence And Hypothesis

Canonical XCU ranks `ds_read_matrix_trans -> s_waitcnt` at about 11.04% and
`ds_read_matrix -> s_waitcnt` at about 5.53%. The current dK read-ahead keeps
two panel slots but drains panel 0 before issuing panel 1. This leaves the
first panel's read latency exposed before the first dK MMAC island.

Hypothesis: issue panel 0 and panel 1 reads back-to-back, drain them once,
then keep the existing two-slot lag-one conveyor. The first dK MMAC island
may then hide part of the initial LDS latency without increasing live storage
or synchronization count.

## Expected Schedule

```text
time0: read Q/dS panel0, read Q/dS panel1
time1: wait lgkm(0), dK MMAC(panel0), read panel2
time2: wait lgkm(0), dK MMAC(panel1), read panel3
time3: wait lgkm(0), dK MMAC(panel2)
time4: wait lgkm(0), dK MMAC(panel3)
```

The existing two fragment slots are reused only after their corresponding
MMAC update, so no fragment is consumed before its own read completes.

## Resource And Correctness Gates

- Exact five logical GEMMs and the canonical M64/N128/D128 contract.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- No new ABarrier, no ordinary matrix `ds_read_b*`, no gather/bpermute.
- No private segment, scratch, SGPR/VGPR spill, or LDS bank conflict.
- H1/S128 full lifecycle correctness must pass before S1024 stats.

## Promotion Rule

The candidate is useful only if repeated H1/S1024 fused ticks improve against
the same-build canonical control and the result is explainable by lower first
use/readiness exposure. Higher MMAC active alone is not sufficient.

## Rejection Boundary

Restore the canonical source if the read-ahead causes a PMD VGPR-init panic,
correctness error, resource failure, or same-shape tick regression. Record the
result as a rejected read-island change rather than stacking another read
placement variant.

## Result

The candidate passed static/resource gates, H1/S128 full lifecycle
correctness, and H1/S1024 full lifecycle correctness. It remained clean at
`private=0`, `scratch=0`, `sgpr_spill=0`, `vgpr_spill=0`, and
`ldsBankConflict=0`. H1/S1024 fused ticks were `47,177,585` versus canonical
`46,637,955` (`+1.16%`); MMAC active was about `32.2%`. The longer initial
readiness window outweighed the larger read island.

Classification: `REJECT_TICKS_CANONICAL_RESTORED`. The source was restored to
the canonical `2d1a522` equivalent. Do not stack another read-placement
variant without a new whole-kernel SQTT hypothesis.
