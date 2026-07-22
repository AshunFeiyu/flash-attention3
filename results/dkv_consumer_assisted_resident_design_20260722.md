# dKV Consumer-Assisted Resident Publication Design

Date: 2026-07-22

Status: `DESIGN_READY_FOR_STATIC_IMPLEMENTATION`

## Evidence That Selects This Boundary

The latest e0f10535 H1/S1024 aggregate dKV result is
`34,625,955 ticks / 38.538081% MMAC active`.  Per-CU attribution shows that
the longest causal CTA reaches about `45.36%` while the one-q-tile CTA reaches
only about `16.94%`.  The eight CTAs execute `8,7,...,1` q tiles and pay the
same resident/startup protocol on every CTA.

The current LDS overlay forces this startup order:

```text
P0/P1: load resident K/V to LDS
       -> ResidentFilled
C0/C1: read all owned K/V fragments to VGPR
       -> ResidentUsed
P0/P1: overwrite the same LDS 64KB with Q/dO + publish sidecar
       -> RawHeadFilled
C0/C1: first score/dP MMAC
```

XCU attributes the two largest dKV issue-gap families to ABarrier ownership:
`s_abarrier_try_wait -> s_xor_b32` is `30.35%`, and
`s_abarrier_try_wait -> s_waitcnt` is `14.89%`.  Store codegen and output
traffic have already been ruled out as the cause.

## Algorithm And Ownership Invariants

The four-GEMM DAG and output ownership do not change:

```text
score = K @ Q^T
dP    = V @ dO^T
P,dS  = softmax/elementwise(score,dP,max,invsum,delta)
dV   += P  @ dO
dK   += dS @ Q
```

- Tile remains `Mq=128,Nk=128,D=128`.
- Two physical four-wave consumers own disjoint `Nk=64` regions; each wave
  owns one `Nk=16` output stripe for full D128.
- No score/dP/dV/dK GEMM is duplicated.
- dK/dV remain FP32 and keep the direct `global_store_dwordx4` epilogue.
- Producer0 owns Q publication; producer1 owns dO plus sidecar publication.

## Rejected Layout Drafts

1. **Naive concurrent publication with the current overlay:** invalid.  K/V
   and Q/dO occupy the same 64KB, so concurrent MLS writes race.
2. **Resident K/V 64KB + full M128 Q/dO 64KB + sidecar 1.5KB:** invalid at
   `132,608B`, above the 128KB limit.
3. **One physical M64 raw page:** fits, but destroys the existing head/tail
   ping-pong and prevents next-page publication during current-page compute.
4. **Chosen layout:** resident K/V 64KB plus two physical M64 Q/dO pages
   totaling 64KB.  Sidecar pages reuse the K/V region only after all consumer
   waves have latched K/V and arrived at `ResidentUsed`.

## Chosen LDS Budget

```text
resident K/V startup region:
  2 * 128 * 128 * fp16                    = 65,536 B

two M64 Q/dO raw pages:
  2 pages * 2 tensors * 64 * 128 * fp16  = 65,536 B

static LDS allocation                     = 131,072 B
sidecar per M64 page:
  3 fields * 64 rows * fp32                 = 768 B
two sidecar pages                            = 1,536 B
```

Sidecar is a phase overlay inside the resident K/V region, so it adds no
static LDS bytes.  The overwrite proof is `8 consumer ResidentUsed arrivals`
after every owned K/V fragment has completed its `ds_read_matrix` latch.

## Wave Roles

```text
waves0-3   producer0: Q M64 page0/page1, then later Q packets
waves4-7   consumer0: publish K0/V0 resident Nk64, latch, then dKV compute
waves8-11  consumer1: publish K1/V1 resident Nk64, latch, then dKV compute
waves12-15 producer1: dO M64 page0/page1; after ResidentUsed publish sidecar
```

Consumer publication uses the same number of K/V MLS operations as the
canonical producers.  Work moves between roles; global bytes and matrix-load
count do not increase.

## ABarrier Ledger

The slot count stays at seven:

| Slot | Publisher | Waiter | Protected lifetime |
|---|---|---|---|
| ResidentFilled | 8 consumer waves | same 8 consumers | K/V BPS completion before K/V latch |
| ResidentUsed | 8 consumer waves | producer1 | all K/V reads complete before sidecar overlay |
| RawHeadFilled | 8 producer waves | 8 consumers | Q/dO page0 and sidecar0 ready |
| RawHeadUsed | 8 consumers | 8 producers | page0 safe to overwrite |
| RawTailFilled | 8 producer waves | 8 consumers | Q/dO page1 and sidecar1 ready |
| RawTailUsed | 8 consumers | 8 producers | page1 safe to overwrite |
| AllDone | 16 waves | 16 waves | safe ABarrier invalidation |

No new token is introduced.  Producer1 delays its first `RawHeadFilled`
arrival until `ResidentUsed` permits the sidecar overlay.  Producer0 can
publish both Q pages while K/V publication/latch is in progress.

## Expected Pipeline

```text
time0:
  P0  MLS Q page0
  C0  MLS K0/V0 resident
  C1  MLS K1/V1 resident
  P1  MLS dO page0

time1:
  P0  MLS Q page1
  C0/C1 wait ResidentFilled and latch K/V
  P1  waits ResidentUsed only for sidecar overwrite

time2:
  P0  page1 already available
  C0/C1 arrive ResidentUsed
  P1  publish sidecar0, arrive RawHeadFilled, then dO/sidecar page1

time3:
  C0 score/dP MMAC page0
  C1 score/dP MMAC page0 with the accepted useful stagger
  producers wait only on page reuse, not resident publication

steady:
  page0/page1 alternate exactly as the canonical head/tail protocol
```

The intended gain is removal of Q/dO MLS and its readiness wait from the
post-`ResidentUsed` startup chain.  It is not an attempt to remove required
first-use waits or to inflate MMAC count.

## Resource And Promotion Gates

- Consumer WDRA target starts at 160 VGPR.  If MLS address temporaries exceed
  it, at most 192 is available while keeping `32+192+192+32=448 < 512` VGPR
  per SIMD.  Any spill/private/scratch rejects the design.
- Static LDS must equal, not exceed, 131,072B.
- Main operands remain MLS/BPS + `ds_read_matrix` + MMAC; sidecar is the only
  ordinary LDS path.
- H1/S128 and H1/S1024 correctness must pass with exact dK/dV metrics.
- Dynamic MMOP, VMEM bytes, FLAT stores and output ownership must match
  canonical; `ldsBankConflict=0`.
- First stats gate: same-build H1/S1024 ticks must decrease.  Explain active
  per CU, especially q-tile counts 8 and 1, instead of relying only on the
  aggregate.
- Winner-only fullperf must show lower ResidentFilled/ResidentUsed/RawFilled
  critical gaps without replacing them with more wait/SCA or broken MMAC
  islands.

