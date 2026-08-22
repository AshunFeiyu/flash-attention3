# Fused5 C1 dS On Dead dO

Status: `REJECT_CTA_WIDE_DOUT_DEAD_GATE_CANONICAL_RESTORED`.

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
C1: finish dV, arrive+wait DoutDead(page), overwrite dead dO with C1 dS
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
three-generation, four-role lifecycle. All eight consumer waves arrive at
`DoutDead`; C1 then waits before any wave overwrites dO. The generated ISA
contains 12 MLS/BPS
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
`/zys/sb/fa3b/layout_probes/fused5_c1_ds_on_dead_dout_20260822_180152`.
Compiler commit: `e0f10535a0d681bcf3885ea2c398cc494bf6e332`.
Binary SHA256: `5a1fdd09b84a69b8f482f97038bbb297ac67c3924aa0a1d313bb291e7c26a39b`.
ASM SHA256: `b9ea007bb3f72fd145a29e31a4251a9fc7f9050b0ac0086158de0ccbb289d175`.

The first count-4 probe was insufficient: one C1 wave could overwrite dO
while another C1 wave still read the complete dO tile. Canonical S1024 exposed
this as nondeterministic dV error. Count 8 fixes the race and gives three
bit-stable S1024 correctness passes.

## Canonical Performance Result

The corrected ownership map passes S128 causal/noncausal, S256, S512, and
three consecutive S1024 full CPU-golden runs. Static resources remain
SGPR82/VGPR128, role use `9/187/87/182`, private/spill/scratch0, exact five
GEMMs, and bank0.

Two interleaved S1024 pairs reject the candidate:

| Pair | Control fused ticks | Candidate fused ticks | Delta |
|---|---:|---:|---:|
| 1 | 44,996,770 | 46,277,595 | +2.846% |
| 2 | 44,949,905 | 46,571,070 | +3.607% |
| mean | 44,973,338 | 46,424,333 | +3.226% |

Full-lifecycle means regress `49,124,075 -> 50,567,563` (`+2.938%`). The
saved 16KiB does not create an additional packet or MMAC island. Correctness
requires an eight-consumer `DoutDead` rendezvous on every q tile, replacing
group-local progress with a CTA-wide ownership gate. The candidate is removed
from canonical source; xcu capture is intentionally skipped because ticks are
not competitive.

Evidence:
`/zys/sb/fa3b/dead_dout_ab_20260822/paired`.

Workbook evidence:
`/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_5gemm_clean_design_20260822.xlsx`,
sheet `36 C1 dS on Dead dO`.
