# Fused5 dO Prefetch Under dK

Status: `REJECT_BARRIER_COST_CANONICAL_RESTORED`

## Single Hypothesis

The canonical two-page raw packet keeps `Q+dO` under one `RawFilled/RawUsed`
ownership epoch. Both consumer groups finish their final dV read before they
start dK, so dO is dead while Q remains live. Add only `DoutUsed0/1`: the
producer may refill the dO half for tile `t+2` while dK still consumes Q from
tile `t`, then wait for `RawUsed` before overwriting Q and publishing the same
complete `RawFilled` packet.

## Expected Conveyor

```text
time0: producer publishes Q(t)+dO(t); consumers run score/dP/dS/dV
time1: both consumer groups arrive DoutUsed(t)
time2: producer loads dO(t+2) while consumers run dK(t)
time3: consumers arrive RawUsed(t); producer loads Q(t+2)+sidecar and
       publishes one complete RawFilled(t+2)
```

## Invariants

- Exactly five GEMMs; no duplicated score, dP, dV, dK, or dQ work.
- `M64/N128/D128`, 16 waves, and the four WDRA roles are unchanged.
- LDS remains 128KB; matrix layout and dynamic matrix traffic are unchanged.
- Main matrix transport remains MLS/BPS + `ds_read_matrix` + MMAC.
- `RawFilled/RawUsed` remain the complete-packet readiness and Q ownership
  tokens. No separate dO Filled token is added.
- dO release is after the final dV LDS read has completed in both consumer
  groups. Q is not overwritten until both groups finish dK and arrive
  `RawUsed`.

## Admission

Reject on any correctness error, PMD warning/panic, scratch/spill/private
segment, bank conflict, extra matrix transaction, generated-MMAC count drift,
or same-shape tick regression. If promoted, XCU must show useful producer BPS
inside the prior `RawUsed` gap and a shorter future raw-readiness wait.

The full derivation, resource ledger, and time0/time1/time2 schedule are in
`fa3_bwd_5gemm_clean_design_20260823_dout_prefetch_under_dk.xlsx`, section 51.

## Result

Generated ISA retains MMAC1472, `ds_read_matrix`840, waits340 and
`v_mov_b64`68, but adds two static BPS sites and two ABarrier token families.
Role use remains `9/171/87/164`; SGPR82/VGPR128 and
private/spill/scratch0 pass. Full S128 causal/noncausal and S1024 causal CPU
golden checks pass with warning0 and bank0.

Three interleaved S1024 pairs reject promotion:

```text
control:   44,261,035  44,093,595  44,797,935  mean 44,384,188
candidate: 44,341,115  45,060,470  44,463,965  mean 44,621,850
delta: +0.535%
```

The final paired stats sample raises MMAC active `34.693% -> 35.112%` and
coissue success `20,542 -> 20,788`, but SCA grows `46,776 -> 48,108`, barrier
share grows `13.748% -> 14.203%`, and wait-LGKM grows
`7.754% -> 7.932%`. The early BPS work is real, but the new ownership token
cost is larger than its critical-path benefit. No S2048 or candidate fullperf
is admitted. Canonical source is restored; do not retry a per-page dO Used
token without a way to reuse an existing synchronization event.

Evidence: `/zys/sb/runs/f5dop`, `/zys/sb/runs/f5dop_ab`, and workbook
section 51.
