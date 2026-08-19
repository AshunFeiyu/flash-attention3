# Fused5 C0 Half-Batch dS Publication Design

## Evidence

The accepted two-generation C0 conveyor makes the dQ writer the final
representative SIMD role at timestamp 96,028, only 892 cycles after C1. Its
remaining 32 dS waits total 21,983 cycles and are dominated by the first C0
wait of each q tile; C1 normally becomes ready while the writer consumes C0.
C0 currently computes all four M16 dS panels before publishing one batch.

## Single Hypothesis

Split only C0's Filled event into two halves while retaining one physical page
and one Done ownership token per generation:

```text
time0  C0 compute dS panel0/1
time1  C0 publish half0 + FilledHalf0    W consume dQ panel0/1
time2  C0 compute dS panel2/3            W MMAC half0
time3  C0 publish half1 + FilledHalf1    W consume dQ panel2/3
time4  C0 dV/dK                          W continue with ready C1
```

This exposes useful C0 score/dP/softmax work under writer dQ MMAC. It does not
add an empty delay, duplicate a GEMM, change M64/N128/D128 ownership, or alter
the group1 schedule.

## Ownership Proof

- Generation0 and generation1 retain their accepted distinct 16 KiB pages.
- Existing `BatchDsFilled0` and `BatchDsFilled0Alt` become first-half Filled
  tokens; two new count-4 tokens announce panels2/3.
- Existing `DqDone0` and `DqDone0Alt` remain count-8 full-page reuse tokens.
  Four C0 waves arrive only after their dK reads; four writer waves arrive only
  after consuming both halves. Therefore neither half can be overwritten while
  either role still reads it.
- Group1's Filled/Done pair is unchanged.
- Total ABarrier count grows from 12 to 14, below the observed 16-ID surface.
  LDS allocation and dynamic matrix work remain unchanged.

## Implementation Shape

- Extract the canonical C0 panel body as a compile-time MBlock helper without
  changing its read/wait/MMAC order.
- Execute panels0/1, wait the generation's existing Done token when reusing,
  publish and signal half0; then execute/publish/signal panels2/3.
- Writer waits and consumes C0 half0, then C0 half1, then follows the canonical
  full-batch C1 path.

## Admission

1. Static gate: no private/spill/scratch; all roles fit existing WDRA pools;
   exact MMOP and native matrix path remain unchanged.
2. S128 causal/noncausal and S1024 causal correctness PASS; bank0.
3. Three paired S1024 runs improve fused ticks. SQTT must show the first C0
   writer wait shrinking and C0 panel2/3 VALU/MMAC overlapping writer dQ MMAC.
4. If barrier share or ticks regress, remove the split and keep
   `best/fused5-c0-ds-gen2-fixed-pair-20260819`.

## Result

Status: `REJECT_EARLY_WAKE_RESOURCE_CONTENTION`

The implementation passed static gates at C0 169/204 VGPR, kernel
SGPR76/VGPR128, and no private segment, spill, or scratch. S128 causal and
noncausal plus three S1024 pairs passed full correctness with exact MMOP and
bank0. The lower C0 VGPR use confirms that publishing after two panels shortens
the fragment lifetime as intended.

Performance rejects the schedule. Three alternating S1024 pairs regress fused
mean ticks `44,718,765 -> 45,857,327` (`+2.546%`) and lifecycle mean ticks
`48,875,038 -> 49,989,182` (`+2.279%`). Mean barrier share falls
`13.549% -> 12.454%`, but wait-LGKM rises `7.439% -> 8.351%`, wait-VM rises
`2.275% -> 2.804%`, and MMAC active falls `34.211% -> 33.949%`.

Early wake-up succeeds at removing ownership wait but makes writer dQ compete
with C0/C1 for LDS and MMAC issue slots. The two extra Filled tokens and half
MMAC islands cost more than the saved wait. Candidate code is removed. Do not
split dS below the accepted four-panel batch on the current 16-wave topology.
