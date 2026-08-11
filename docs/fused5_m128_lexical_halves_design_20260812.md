# Fused5 M128 Lexical-Halves Design

Status: `REJECT_STATS_TICKS / CANONICAL_RESTORED`.

## One Hypothesis

The accepted H1/S1024 SQTT attributes about 29.6% of issue-gap time to the
single-page `RawUsed -> producer BPS -> RawFilled` ownership chain. Keep the
same exact five GEMMs, N128/D128 ownership, 16-wave roles and native
P/dS/dQ handoff, but double M from 64 to 128 so one raw packet carries twice
the useful MMAC work.

This is not the rejected July M128 expression. That version avoided spills
with runtime panel dispatch and expanded VALU/SCA by about 70%/151%. The new
admission condition is two lexical, compile-time M64 halves. Each half reuses
the current four-panel source structure; no eight-panel dynamic array or
runtime helper selection is allowed.

## Formula And Exact Work

For one M128 x N128 tile:

```text
score = Q @ K^T        512 MMAC
dP    = dO @ V^T       512 MMAC
dV   += P^T @ dO       512 MMAC
dK   += dS^T @ Q       512 MMAC
dQ   += dS @ K         512 MMAC
total                  2,560 MMAC
```

Each heavy consumer wave owns N16 and performs 256 MMAC per tile. Each dQ
writer owns D32 and performs 128 MMAC. Across 8 heavy plus 4 writer waves:

```text
8 * 256 + 4 * 128 = 2,560
```

H1/S1024 causal remains exact MMOP92,160. The number of useful arithmetic
operations and output values is unchanged; only q-tile ownership epochs fall
from 72 to 36.

## LDS Budget And Lifetimes

Startup uses exactly 128KB:

| Region | Bytes | Startup | After resident latch |
|---|---:|---|---|
| Q M128 | 32,768 | raw Q | current Q |
| dO M128 | 32,768 | raw dO | current dO |
| K N128 | 32,768 | resident K | P scratch / dS batch |
| V N128 | 32,768 | resident V | P scratch / dS batch |
| Total | 131,072 | | |

After all 12 non-producer waves latch K/V into VGPR, the K/V 64KB region is
reused in two phases per q tile:

```text
compute half h: 16KB P writer pages + sidecar in writer padding
publish half h: four M16 dS panels occupy the full 64KB physical region
consume half h: heavy waves read normal dS for dK; writer reads trans dS for dQ
release half h: KvDsUsed generation completes, then the region is reused
```

The writer-padding probe proves one 768-byte M64 sidecar fits in one page's
upper 1KB. M128 sidecar uses two such regions: rows0-63 in writer page0 padding
and rows64-127 in writer page1 padding. No sidecar byte overlaps a native P or
dS writer footprint.

## WDRA Budget

The long-lived dK/dV accumulators, resident K/V fragments and dQ half
accumulators do not grow. Only source row offsets and one half index are new:

```text
window target        producer/C0/C1/dQ = 8/200/200/88
accepted M64 use                         8/163/166/84
M128 admission       every role within the same window
hard rejection       any private, scratch, SGPR spill or VGPR spill
```

The previous M128 role `15/185/184` is not a performance precedent; it only
proves the raw resource class can fit. Generated metadata and branch-role
evidence are authoritative.

## Ownership Ledger

Keep the canonical seven tokens:

```text
0 ResidentFilled0
1 ResidentFilled1
2 KvDsUsed
3 RawFilled
4 RawUsed
5 BatchDsFilled0
6 BatchDsFilled1
```

`KvDsUsed` generations are:

```text
generation0: all non-producer waves have latched resident K/V
generation1: half0 dK and dQ source reads are complete
generation2: half1 dK and dQ source reads are complete
... repeat two generations per q tile
```

The producer waits generation0 before placing the first sidecar into released
K/V padding. Later sidecar writes stay in writer padding and are protected by
the canonical `RawUsed` packet lifetime.

## Expected Pipeline

```text
time0:
  P0        load K/V and Q/dO; wait resident latch; publish sidecar+RawFilled
  C0/C1/WQ  latch resident K/V

time1, half0:
  C0/C1     score/dP -> softmax/P -> dV/dS -> publish dS -> dK
  WQ        wait dS half0 -> dQ half0 -> arrive KvDsUsed -> atomic half0

time2, half1:
  C0/C1     after KvDsUsed, reuse LDS and run the same fixed four-panel chain
  WQ        atomic half0 overlaps C MMAC/VALU; then dQ/atomic half1

time3:
  C0/C1     arrive RawUsed only after half1 dK
  P0        load next M128 Q/dO/sidecar once
  C0/C1     wait next RawFilled, then retire prior half1 KvDsUsed before
            writing the reused P/dS scratch
```

Expected gain: RawUsed/RawFilled/BPS ownership events per useful MMOP halve.
Expected cost: one extra intra-tile `KvDsUsed` rendezvous, equal to the
boundary that two independent M64 tiles already pay. Promotion requires the
generated source to avoid the old runtime-control expansion.

The prior-half release wait is deliberately carried across the q-tile
boundary. Producer Q/dO/sidecar publication does not overwrite the touched
P/dS source bytes, while the next consumer half cannot write P scratch until
that wait completes. This preserves the ownership ledger and lets the next raw
load age under the dQ half1 read instead of exposing an end-of-tile wait.

## Gates

1. Contract/source: exact five GEMMs, M128/N128/D128, two explicit lexical
   halves, no runtime panel dispatch and no alternate kernel path.
2. ASM/resources: MLS/BPS + ds_read_matrix + MMAC, role windows pass,
   LDS131072B, private/spill/scratch0.
3. Correctness: H1/S128 causal/noncausal, then full H1/S1024 causal; exact
   MMOP92,160 and bank0.
4. Performance: compare same-build M64 canonical and M128 lexical candidate.
   Ticks are primary; MMAC active, SCA/VALU, barrier and wait shares explain.
5. XCU only after ticks are competitive. Reject if VALU/SCA again expand
   materially, even when Raw barrier share falls.

## Result

The candidate passed static/resource gates, H1/S128 causal and noncausal, and
the complete H1/S1024 causal CPU golden. It retained exact MMOP92,160,
`ldsBankConflict=0`, role use `8/157/158/84`, metadata VGPR124 and no
private/spill/scratch. It did not pass the performance gate:

| Metric | M64 accepted mean | M128 lexical mean | Change |
|---|---:|---:|---:|
| fused ticks | 72,048,112.5 | 72,789,990 | +1.030% |
| MMAC active | 22.714850% | 22.173414% | -0.541 pp |
| VALU | 120,800 | 128,464 | +6.344% |
| SCA | 37,808 | 35,068 | -7.247% |
| LDS | 64,096 | 63,664 | -0.674% |

Moving the half1 `KvDsUsed` wait across the q-tile boundary also failed:
two runs averaged 72,905,105 ticks and 22.139918% MMAC active. It added 288
SCA instructions and did not lower barrier or LDS first-use exposure.

Same-compiler ASM explains the unexpected VALU growth. The M128 expression
loses packed softmax/dS codegen: static `v_pk_fma_f32` changes `8 -> 0` and
`v_pk_mul_f32` changes `40 -> 32`, while scalar `v_fma_mix_f32` changes
`16 -> 64`, `v_mul_f32` `50 -> 194`, and `v_sub_f32` `16 -> 80`. Dynamic
epochs halve, but the scalar expansion still leaves 6.3% more VALU. This
rejects the current source expression, not the M128 arithmetic tile itself.
