# Fused5 M128 Q-Double and dO-Single Design

Status: `ACCEPT_PROBE_ADMIT_CANONICAL_STATIC_DRAFT`

## Hypothesis

Use one `M128/N128/D128` ownership epoch, but do not retain two complete
`Q+dO` raw pages. Keep two Q pages and one current dO page. Once dO, P scratch
and sidecar are dead, reuse their contiguous 64 KiB region as the native dS
writer page. This closes the storage hole that rejected the two-full-raw C91
design and lets Q for the next epoch age while the current dS is consumed.

This is not the rejected standalone Q-only double buffer: the larger M128
epoch halves the count8 dead-dO rendezvous cadence. It is also not the rejected
M128 single-raw design: only dO publication remains behind `EpochDone`; Q for
the next epoch is already resident.

## Formula and Causal Work

The exact five-GEMM DAG is unchanged. A full M128 tile has 512 MMAC per GEMM
and 2560 per ownership epoch. The causal implementation must remain two
compile-time M64 halves so the accepted C1 zero-front removes the fully invalid
lower-half work. Restoring masked-but-useless MMAC merely to raise active share
is forbidden.

## LDS Map

```text
startup:
  [  0,  32) KiB  Q0
  [ 32,  64) KiB  unused, later Q1
  [ 64, 128) KiB  resident K/V

steady compute:
  [  0,  32) KiB  Q0
  [ 32,  64) KiB  Q1
  [ 64,  96) KiB  current dO
  [ 96, 104) KiB  P ownership scratch
  [104,105.5) KiB current M128 sidecar

steady dS publication:
  [  0,  64) KiB  Q current + Q next
  [ 64, 128) KiB  physical M128 dS writer page
```

The dS overlay is legal only after all eight heavy consumer waves have
finished every dO read and sidecar use. The focused probe must prove this
count8 `DoutDead` edge over at least three generations.

## Barrier Ledger

```text
ResidentFilled      count4
ResidentUsed        count12
QFilled0/QUsed0     count4/count8
QFilled1/QUsed1     count4/count8
DoutFilled/DoutDead count4/count8
BatchDsFilled0      count4
BatchDsFilled1      count4
EpochDone           count12
```

There are 11 semantic IDs. `EpochDone` includes eight dK normal-read owners
and four dQ trans-read owners. It protects the next dO/sidecar publication;
Q page reuse is independent and may begin after the corresponding `QUsed`.

## Expected Pipeline

```text
time0  P: MLS K/V + Q0
       C0/C1/W: latch resident K/V
       P: MLS dO0 + Q1 + sidecar0

time1  C0/C1: two lexical M64 halves of score,dP,P,dV,dS
       Q1 is already resident; no duplicate score or dP

time2  C0/C1: arrive/wait DoutDead, publish one M128 dS page
       C0/C1: dK; W: G1-first then G0 dQ
       P: after QUsed0, MLS Q2 into Q0 while dS/dQ are active

time3  after EpochDone, P loads only dO1 + sidecar1
       C0/C1 begin the next M128 epoch with Q1 already ready
```

## Resource and Codegen Gates

- Startup/steady LDS must be at most 131,072 bytes with compile-time asserts.
- C0 must compute dV panel-locally so it holds eight dS panels without also
  retaining eight P panels. Both heavy roles must remain within 204 VGPR.
- The source must preserve C83 packed softmax/dS lowering. The prior M128
  lexical expression lost packed operations and is a direct static reject
  boundary.
- Main matrices remain MLS/BPS + `ds_read_matrix` + MMAC. No ordinary matrix
  DS read, gather, permute or wrong-layout path is allowed.
- Exact causal MMOP, full CPU golden, no private/spill/scratch and bank0 are
  hard gates.

## Admission Order

1. A0-A4 focused two-generation layout/lifecycle probe for Q ping-pong,
   dO/sidecar death, native dS overwrite, normal/trans consumers and count12
   release.
2. Static production draft only if the probe passes. Compare generated packed
   arithmetic against C83 before PMD.
3. H1/S128 causal/noncausal correctness, then interleaved H1/S1024 A/B.
4. Only a tick-competitive candidate receives S2048 and xcu capture.

The design is admitted because its complete resource graph closes and it
targets useful work per ownership generation. It is not promoted by design;
the count8 `DoutDead` rendezvous and serialized dO load remain explicit risks.

## Focused Probe Result

`probes/fused5_m128_qdouble_dout_single_probe.cpp` passes three generations
on compiler `e0f10535` and PMD `HEAD1694`:

- semantic mismatches: Q-before `0`, dO `0`, Q-after `0`, dS-normal `0`,
  dS-trans `0`;
- branch VGPR use: `1/134/132/27` within WDRA `16/204/204/88`;
- metadata: SGPR47/VGPR128, private/spill/scratch `0`;
- LDS128KiB, 11 ABarrier IDs, `ldsBankConflict=0`;
- 32 MLS, 48 native dS writes, normal/trans matrix reads and MMAC present;
- ordinary matrix DS read, permute and permlane counts are all `0`.

Producer ISA preserves the intended useful overlap order:
`wait QUsed0 -> MLS Q2 -> wait EpochDone -> MLS dO1`. This proves that Q-page
reuse is independent of the 12-reader dS release edge in generated code.

Evidence:
`/zys/sb/runs/fused5_c92_probe/layout_probes/fused5_m128_qdouble_dout_single_20260823_174228`.
