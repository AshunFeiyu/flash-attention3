# Fused5 dQ Panel-Pair Read Rejection

## Hypothesis

Reading both consumer dS groups for one dQ panel before the dQ MMAC island
could enlarge the matrix-read island and hide first-use LDS latency.

## Evidence

- The direct dual-group read variant failed the static resource gate with
  `private=36` and `vgpr_spill=8`.
- The bounded reduced-live variant passed static resources and full H1/S128,
  H1/S1024 correctness. It changed the dQ writer order to
  `panel0:g0 -> panel0:g1 -> panel1:g0 -> panel1:g1`.
- Two H1/S1024 stats-only repeats were `47,810,035` and `47,832,785` fused
  ticks, versus the canonical `47,624,850` reference. This is a
  `+0.39%` to `+0.44%` regression.

## Decision

`REJECT_TICKS_REGRESSION_CANONICAL_RESTORED`.

The larger dQ read island does not compensate for changing the existing
group-local dS stagger. Do not add another dQ read-order variant. The next
admitted experiment must change ownership topology, not matrix-read placement.

