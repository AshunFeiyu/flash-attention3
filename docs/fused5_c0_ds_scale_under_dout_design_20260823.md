# Fused5 C0 dS Scale Under dO Readiness

## Hypothesis

The accepted SQTT trace shows that the dQ writer repeatedly waits on C0's dS
publication. Two read-placement experiments failed because they migrated LDS
readiness into a later critical window. Keep every read and ownership point
unchanged, and instead move one existing arithmetic operation into C0's dO
readiness window.

The algebraic rewrite is:

```text
baseline:  dS = P * (dP - D) * softmax_scale
candidate: P_scaled = P * softmax_scale
           dS = P_scaled * (dP - D)
```

It preserves the operation count: two multiplies and one subtraction per useful
word. `P.f16`, captured before scaling, remains the unchanged dV operand.

## Exact Schedule

```text
score MMAC
probability/softmax VALU
save P.f16 for dV
issue the canonical four dO ds_read_matrix fragments
scale P.f32 for dS while dO is in flight
canonical first-use wait
dP MMAC
finish dS from P_scaled and row delta
publish the unchanged dS batch
```

## Invariants

- Exact five logical GEMMs and unchanged MMAC count.
- Unchanged `M64/N128/D128`, 16-wave roles, LDS layout, matrix-read count,
  ABarrier IDs, token generations, global traffic, and output ownership.
- C1, producers, and dQ writer remain byte-for-byte unchanged.
- No new fragment lifetime: `P.f32` is overwritten in place after `P.f16` is
  saved for dV.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- No private segment, spill, scratch, or LDS bank conflict.

## Admission

1. Generated ISA must retain the accepted MMAC/read/wait counts and form
   `dO read -> useful scale VALU -> first-use wait -> dP MMAC` in C0.
2. C0 must stay inside its WDRA role budget with no resource regression.
3. Full cached CPU-golden correctness must pass S128 causal/noncausal and
   S1024 causal with zero warnings and zero LDS bank conflicts.
4. Three interleaved S1024 A/B pairs decide ticks; S2048 confirms scaling.
5. Higher MMAC active without lower same-shape ticks is not promotion.

The corresponding workbook design is Section 53 of
`fa3_bwd_5gemm_clean_design_20260823_c0_ds_scale_under_dout.xlsx`.

## Result

Status: `ACCEPT_MICRO_TICKS_AND_READINESS_BARRIER_DEBT_OPEN`.

- The compiler initially sank `P *= scale` below dP. A zero-instruction VGPR
  scheduling anchor is required to produce the intended
  `four dO reads -> four useful v_mul -> wait -> dP MMAC` ISA.
- Static MMAC1472, symbol matrix-read840, wait340, ABarrier102, and
  `v_mov_b64`68 remain exact. Role usage is `9/176/87/164`, SGPR82/VGPR128,
  with private/spill/scratch0.
- Full cached CPU-golden correctness passes S128 causal/noncausal, S1024 and
  S2048 causal. PMD reports warning0 and `ldsBankConflict=0`.
- Three interleaved S1024 pairs improve fused mean
  `44,561,942 -> 44,060,532` (`-1.125%`). Two S2048 pairs improve
  `82,442,360 -> 82,095,878` (`-0.420%`).
- S1024 PMD means show MMAC active `34.799% -> 34.891%` and wait-LGKM
  `8.092% -> 7.617%`. Dynamic VALU falls `118,304 -> 117,156`; SCA falls
  `46,776 -> 46,656`. Barrier share rises `13.830% -> 14.007%`.
- XCU confirms that C0 transpose-read readiness plus the newly visible scale
  tail falls `10,787 -> 9,119` representative-wave cycles (`-15.5%`). Global
  transpose-read-to-wait falls `8.47% -> 6.73%`, with `1.34%` now attributed
  to useful scale-to-wait. Dispatch issues fall `370,864 -> 369,884`.
- The representative writer ABarrier total does not improve
  (`32,107 -> 32,643` cycles), so this is a real ticks/readiness micro-win,
  not a solution to ownership serialization. That barrier debt remains the
  next structural target.

Evidence:
`/zys/sb/runs/fused5_c0_ds_scale_under_dout_ab`,
`/zys/sb/runs/fused5_c0_ds_scale_under_dout_s2048_*`, and
`/zys/sb/runs/fused5_c0_ds_scale_fullperf_20260823`.
