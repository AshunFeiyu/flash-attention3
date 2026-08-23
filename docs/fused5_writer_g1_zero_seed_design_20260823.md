# Fused5 dQ Writer G1-First Zero-Seed Design

Date: 2026-08-23

Status: `DESIGN_ADMITTED_ONE_BOUNDED_RETRY`.

## One Hypothesis

The accepted writer consumes group1 before group0 on every q tile. Its eight
FP32 dQ accumulators are still explicitly cleared before the first group1
MMAC. Seed each panel/half accumulator from that first useful group1 MMAC and
reuse one small `mmac_zero` fragment instead.

This is not the old C0-first experiment repeated unchanged. Since that test,
the canonical writer order became G1-first and the fused body became a
compile-time causal/noncausal specialization. This is the final bounded retry
of writer zero-seeding; a non-win closes the tier.

## Formula And Ownership

For every q tile and dQ output fragment:

```text
dQ = dS_group1 @ K_group1 + dS_group0 @ K_group0
```

The candidate changes only the accumulator's first update:

```text
control:   acc=0; acc+=G1_writer0; acc+=G1_writer1..3; acc+=G0_writer0..3
candidate: acc =G1_writer0; acc+=G1_writer1..3; acc+=G0_writer0..3
```

All eight source writers, both Filled/Done handoffs, all reads, all MMACs and
all stores remain exact. The causal q0 group1 contribution is still executed
in this experiment; zero-work pruning is deliberately separate.

## Resource And Pipeline Budget

| Item | Accepted C83 | Candidate gate |
| --- | ---: | ---: |
| Tile / roles | M64/N128/D128, 16 waves | exact |
| Formula | five GEMMs | exact |
| Dynamic MMOP/LDS/VMEM/FLAT | 88,064/61,056/1,408/3,616 | exact |
| ABarrier IDs / ownership | 12 / G1-first | exact |
| Writer clears | 8 accumulator clears/q tile | removed |
| Zero live state | none across first MMAC | one F16x8 zero fragment |
| WDRA pool | 16/204/204/88 | one 512-total rebudget to 16/204/196/96 allowed |
| Metadata | private/spill/scratch0 | exact |

Expected writer schedule:

```text
time0  wait G1 Filled; read G1 panel0
time1  first G1 writer MMAC seeds dQ; remaining G1 MMAC accumulates
time2  arrive G1 Done; wait/read/MMAC G0; store final dQ
```

The source must use one templated writer helper. No second writer body, runtime
first-update branch, extra MMAC, extra token, or alternate production path is
allowed.

## Admission Gates

1. Static ASM must keep causal/noncausal MMAC, matrix reads, waits, barriers
   and stores exact while reducing `v_mov_b64` sites.
2. Both symbols remain private/spill/scratch free. A single total-preserving
   WDRA rebudget is allowed because branch use 88 in an 88-register window
   spills under this compiler; no second rebudget is admitted.
3. Full CPU-golden S128 causal/noncausal and causal S1024 pass with warning0,
   nonfinite0 and bank0.
4. Three interleaved S1024 pairs decide promotion; any repeatable ticks win is
   admissible. MMAC active is explanatory, not a substitute for ticks.
5. A winning candidate receives fullperf/xcu. A loss restores C83 source and
   records the writer-zero-seed tier as closed.

## Result

Status: `REJECT_PAIRED_TICKS_WRITER_ZERO_SEED_TIER_CLOSED`.

- The intended ISA was generated: causal/noncausal `v_mov_b64` sites fall
  `10 -> 6`, while MMAC, matrix reads, waits, ABarrier and stores remain exact.
- The initial 88-in-88 writer window spilled. The single admitted total-pool
  rebudget `16/204/196/96` passed at roles `9/173/91/162`, SGPR70/VGPR128,
  private/spill/scratch0 for causal and noncausal symbols.
- Full lifecycle S128 causal/noncausal passes with warning0 and bank0.
- Three interleaved S1024 fused pairs are `-0.625%`, `+0.910%`, and
  `+1.767%`. Means regress `40,667,445 -> 40,944,540` (`+0.681%`).
- Lifecycle means regress `44,824,932 -> 45,119,165` (`+0.656%`). Two of
  three pairs regress, so no S2048/fullperf/xcu is admitted.

The cleaner writer ISA does not shorten the measured critical path. Carrying
the seed and changing the WDRA partition perturb scheduling enough to exceed
the removed clear cost. Production source and WDRA ledger are restored to C83.
Combined with the prior C0-first result, writer zero-seeding is closed unless
the hardware/compiler gains a zero-result MMAC form with no live seed state.
