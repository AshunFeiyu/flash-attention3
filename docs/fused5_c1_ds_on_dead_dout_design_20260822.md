# Fused5 C1 dS On Dead dO

Status: `FOCUSED_PROBE_PASS_CANONICAL_INTEGRATION_ADMITTED`.

## Evidence Boundary

Two same-layer scheduling experiments are closed: resident-read zero cover and
C1 dV-before-dS both passed correctness/resources but did not improve repeated
H1/S1024 ticks. The rejected M128 lexical-halves route already tested a coarse
raw ownership epoch and must not be repeated.

The current M64/N128/D128 LDS map leaves only 6.5KiB in released K/V storage,
so adding a conventional second C1 dS page would exceed the budget by 9.5KiB.
This candidate changes the lifetime map instead of adding storage.

## Single Structural Hypothesis

Each raw page is exactly:

```text
Q  = 16KiB
dO = 16KiB
```

After both consumer groups finish dV, dO is dead. One complete C1 dS page is
also exactly 16KiB. C1 can therefore overwrite the dead dO half of raw page0
or page1 with native `ds_write_matrix` output. The same page is then read by
the normal dK path and transposed dQ path.

No GEMM, tile, output owner, ordinary DS read, gather, permutation, or extra
LDS allocation is introduced.

## Ownership Ledger

```text
C0: publish its accepted dS generation early
writer: consume C0 dS and execute G0 dQ
C0: finish dV, then arrive DoutDead(page)
C1: wait DoutDead(page), overwrite dead dO with C1 dS, signal C1Filled(page)
C0/C1: execute dK, each contributes four RawUsed arrivals
writer: consume C1 dS, execute G1 dQ, contributes four RawUsed arrivals
producer: may refill the raw page only after RawUsed count 12
```

C1 completion is owned by the raw page, so its old `DqDone1` token is removed.
The candidate uses page-specific C1 Filled and DoutDead tokens. The complete
ledger is 14 IDs (`0..13`), below the observed 16-ID surface.

## Resource Proof

| Item | Current | Candidate |
|---|---:|---:|
| startup resident K/V + two raw pages | 128KiB | 128KiB |
| released-K/V steady use | 57.5KiB | 41.5KiB |
| released-K/V headroom | 6.5KiB | 22.5KiB |
| physical WDRA target | 512 VGPR | 512 VGPR target |
| ABarrier IDs | 12 | 14 |

The P scratch remains mandatory: the current q-left P fragment failed direct
P-to-dV correctness and its LDS roundtrip is a fragment ownership conversion.
This design does not claim otherwise.

## Expected Pipeline

```text
time0: producer publishes raw page p; C0/C1 consume score and dP inputs
time1: C0 publishes dS early; C1 completes dV; writer starts G0 dQ
time2: C0 dV overlaps writer G0; C1 waits only for useful DoutDead work
time3: C1 publishes dS into dead dO; C0/C1 dK; writer finishes G0
time4: writer runs G1; C0+C1+writer complete RawUsed count12
time5: producer reuses page p after the peer raw page supplied intervening work
```

The intended stagger is useful work, not a delay: C0 dV and writer G0 MMAC
cover the C1 publication edge.

## Admission Order

1. Focused two-page alias/layout oracle: dO read, DoutDead, dS overwrite,
   normal/trans dS readers, mismatch0 and bank0.
2. Four-role 14-token lifecycle probe over at least three generations:
   exact completion counts, no deadlock, stale read, PMD panic, or VGPR warning.
3. Only then integrate the canonical kernel and run H1/S128 causal/noncausal.
4. Run paired H1/S1024 stats. Ticks are primary; MMAC active explains.
5. Capture xcu only if same-build ticks are competitive.

Any failure is classified at the layer where it occurs. ABarrier phase/count
mistakes are design errors unless a minimal reproducer proves a PMD/compiler
defect.

## Focused Probe Result

`probes/fused5_c1_ds_on_dead_dout_probe.cpp` implements the exact two-page,
three-generation, four-role lifecycle. The generated ISA contains 12 MLS/BPS
loads, 24 native dS writers, 192 trans matrix reads, 24 normal matrix reads,
216 MMAC, four role-local VGPR resizes, and exactly 14 ABarrier init/invalidate
instructions. Ordinary DS reads and lane permutations are both zero.

PMD result:

```text
q_mismatches=0 dout_mismatches=0 normal_mismatches=0
trans_mismatches=0 pass=1
private=0 sgpr_spill=0 vgpr_spill=0 ldsBankConflict=0
```

Evidence directory:
`/zys/sb/fa3b/layout_probes/fused5_c1_ds_on_dead_dout_20260822_174738`.
Compiler commit: `e0f10535a0d681bcf3885ea2c398cc494bf6e332`.
Binary SHA256: `dd1664fcc76d9e421bedcf937077b4c82b9bf0afdffdaea4072f4b9f7eaa38ac`.
ASM SHA256: `20aee9b3a16dad1364f612b8fcfb62495bbafeb53b1014981699f02d2d95b74a`.

This admits the ownership map into the canonical kernel. It is not yet a
performance acceptance.

Workbook evidence:
`/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_5gemm_clean_design_20260822.xlsx`,
sheet `36 C1 dS on Dead dO`.
