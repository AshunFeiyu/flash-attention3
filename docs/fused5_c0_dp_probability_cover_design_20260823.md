# Fused5 C0 dP Reads Under Probability

## Hypothesis

Accepted-baseline SQTT shows that the dQ writer repeatedly waits on the C0
`dS` publication while C1 is normally ready a few cycles later.  On a
representative SIMD, C0 also exposes substantial transpose matrix-read
readiness time.  Move only C0's existing four dO matrix reads to immediately
after score and hide their readiness under the useful probability VALU block.

## Exact Schedule

```text
latch S_MAX/S_SUM
issue next-Q ping-pong packet
existing Q first-use wait also retires the older sidecar reads
score MMAC
issue four dO ds_read_matrix fragments
probability/softmax VALU
first-use wait
dP MMAC
dS VALU
publish the unchanged dS batch
```

The dO fragments live across probability only.  They do not cross dV, dS
publication, another panel, or an ownership epoch.

## Invariants

- Exact five logical GEMMs and unchanged MMAC count.
- `M64/N128/D128`, 16 waves, and all output ownership remain unchanged.
- No new ABarrier ID, event, LDS page, matrix transaction, or global access.
- C1, producer, and dQ-writer schedules remain unchanged.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- No spill, scratch, private segment, or LDS bank conflict.

## Historical Boundary

Earlier C0 dP prefetch experiments held fragments across probability plus dV,
or moved Q and dO reads together.  Those experiments increased resource or
ownership pressure and regressed ticks.  This candidate is admitted only
because fresh SQTT now proves C0 is the repeated writer pace setter and the new
lifetime ends at dP before dS/dV.

## Admission

1. Generated ISA must preserve MMAC/read counts and place dO reads before the
   probability VALU block. The existing Q first-use wait must retire sidecar
   reads, and static waits must not grow.
2. C0 must fit its 204-VGPR WDRA window with zero spill/private/scratch.
3. Full golden correctness must pass at S128 causal/noncausal and S1024 causal;
   LDS bank conflicts must remain zero.
4. Three interleaved S1024 A/B pairs decide ticks. S2048 confirms scaling.
5. A promoted candidate must reduce both C0 transpose-read wait and the dQ
   writer's G0 ABarrier wait without migrating the cost to C1 or control.

The corresponding workbook design is Section 52 of
`fa3_bwd_5gemm_clean_design_20260823_c0_dp_probability_cover.xlsx`.

## Result

Status: `REJECT_RUNTIME_WAIT_MIGRATION_CANONICAL_RESTORED`.

- Generated ISA forms the intended local order and preserves MMAC 1472,
  symbol-scoped matrix-read 840, ABarrier 102, and `v_mov_b64` 68. Static waits fall
  `340 -> 335`; C0 role use rises only `171 -> 173` within its 204-VGPR
  window. SGPR82/VGPR128 and private/spill/scratch0 pass.
- Full cached CPU-golden correctness passes S128 causal/noncausal and S1024
  causal. All delta/dK/dV/dQ checks pass with warning0 and bank0.
- Three interleaved S1024 pairs give control mean `44,203,705` and candidate
  mean `44,511,740`, a `0.697%` fused-ticks regression.
- MMAC active rises `34.786% -> 35.177%`, but runtime wait-LGKM rises
  `7.891% -> 8.408%` and barrier share rises `13.796% -> 13.931%`.
  Lower static wait count did not shorten the critical path; the compiler's
  `lgkmcnt(1/0)` sequence moved readiness pressure into the probability-to-dP
  window.
- No S2048 or candidate fullperf is admitted. The canonical source is
  restored to accepted `0085c6c` behavior.

Evidence: `/zys/sb/runs/fused5_c0_dp_probability_cover*` and workbook
section52.
