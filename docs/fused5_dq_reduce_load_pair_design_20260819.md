# Fused5 dQ Reduction Paired Loads

Status: `REJECT_LIVE_RANGE_AND_WAIT_REGRESSION_CANONICAL_RESTORED`

## Evidence

After 2D ownership, H1/S1024 SQTT shows:

- `s_waitcnt`: 44.95% of Source latency;
- `global_load_dwordx4 -> s_waitcnt`: 29.91% of issue-bubble duration;
- reducer SGPR25/VGPR25, leaving ample register headroom.

The scalarized loop still consumes each `float4` immediately after issuing its
load, so PMD exposes first-use global latency on every workspace slice.

## Change

Issue two consecutive `float4` workspace loads, then accumulate both. Process
an odd final tile with the original single-load path. The 2D grid, causal tile
set, FP32 accumulation, packed FP16 output and total bytes remain unchanged.

## Gates

1. Generated ISA contains a repeated pair of `global_load_dwordx4` before the
   corresponding waits/adds.
2. No private segment, spill or scratch; packed FP16 output remains.
3. S128 causal/non-causal and S1024 causal correctness pass, bank0.
4. Three alternating S1024 A/B pairs compare reducer and full lifecycle ticks.
5. A fresh SQTT must show lower load-to-wait bubble duration; otherwise the
   source is restored to commit `1678545`.

## Result

- The compiler emitted paired `global_load_dwordx4` instructions, but reducer
  resources grew from SGPR25/VGPR25 to SGPR30/VGPR36.
- H1/S128 causal/non-causal and H1/S1024 causal correctness pass, bank0.
- Three alternating S1024 pairs regress reducer mean
  `2,133,495 -> 2,317,012` ticks (`+8.60%`) and full lifecycle mean
  `50,036,653 -> 50,122,042` (`+0.17%`).

The extra outstanding operand fragments lengthen live ranges without hiding
the PMD wait path. Source is restored to the single-load loop at `1678545`;
do not increase load-batch depth without a different asynchronous primitive or
ownership design.
