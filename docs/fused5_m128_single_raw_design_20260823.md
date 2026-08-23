# Fused5 M128/N128 Single-Raw Ownership Epoch

Status: `REJECT_OWNERSHIP_SERIALIZATION_CANONICAL_RESTORED`.

## Why This Is Structural

Two writer-local experiments lowered local instruction/readiness debt but
moved more time into ABarrier. The next candidate changes the amount of useful
work protected by each ownership epoch rather than making one role arrive at
the same wait sooner.

The tile becomes `M128/N128/D128`. Each logical GEMM grows from 256 to 512
MMACs and each five-GEMM epoch from 1,280 to 2,560. For causal S1024, q epochs
fall from 72 to 36 while dynamic useful MMOP remains exactly 92,160. No score,
dP or output work is duplicated or pruned.

## LDS and Lifetime Proof

- Startup: one raw Q/dO M128 page is 64KiB; resident K/V is 64KiB. Total is
  exactly 128KiB.
- After K/V latch, each consumer group owns a private sidecar copy and 4KiB P
  conversion scratch inside its future 32KiB dS slice.
- C0 computes dV panel-locally, so probability fragments do not survive until
  the batch publish and scratch is dead before dS publication.
- Both groups' eight-panel dS batches occupy exactly the released 64KiB.
  Sidecar and scratch are dead at that point, so the alias is legal.
- Raw Q remains in the other 64KiB for dK; dQ reads native dS and resident K.

The candidate removes raw page1 and C0's alternate dS generation. A lifetime
stress pass further contracts the barrier ledger from 12 IDs to 7:
ResidentFilled, ResidentUsed, RawFilled, RawUsed, two BatchDsFilled and one
EpochDone. Producer preloads the next raw Q/dO after RawUsed, then waits
EpochDone before replacing current dS with the next sidecar copies.

## VGPR Proof

The admitted fixed split is `16/192/184/120`. Compiler branch accounting is
`11/180/180/116`, with SGPR76, VGPR128 and zero private/spill/scratch.

## Expected Pipeline

```text
time0  P: load resident K/V + raw M128; all heavy roles latch resident data
time1  C0/C1: eight score/dP/softmax/dS+dV panels; P waits RawUsed
time2  C0/C1: publish dS then dK; W: G1 dQ then G0 dQ
time3  P: prefetch next raw after RawUsed, wait EpochDone, publish sidecar;
       W stores current dQ partial
```

This sacrifices raw double-buffer overlap. Promotion therefore requires lower
same-shape ticks, not merely fewer barriers or higher MMAC active. Static,
full-golden, bank0 and no-debt gates precede three interleaved S1024 pairs.
S2048/fullperf/xcu are admitted only for a primary-gate winner.

## Result

S128 full CPU-golden correctness passes for delta/dK/dV/dQ with warning0 and
bank0. Three interleaved causal H1/S1024 pairs reject the candidate:

- fused ticks: `43,304,018 -> 54,579,070` (`+26.037%`), 0/3 wins;
- MMAC active: `35.265% -> 29.879%` (`-5.386 pp`);
- coissue success: `24,355 -> 17,019` (`-30.121%`);
- barrier share: `13.347% -> 24.127%` (`+10.780 pp`);
- wait-LGKM: `7.648% -> 6.226%` (`-1.422 pp`);
- dynamic work stays MMOP92,160 and VMEM1,408; SCA falls
  `46,656 -> 36,312`, while bank conflicts remain zero.

The local readiness/control reduction is real, but the single-page ownership
loop serializes producer publication behind the slowest dKV+dQ completion.
Larger MMAC islands cannot compensate for the lost peer overlap. No S2048 or
fullperf/xcu capture is admitted; source returns to commit `58e90fc`.

Workbook: section61 in
`fa3_bwd_5gemm_clean_design_20260823_m128_single_raw.xlsx`.
