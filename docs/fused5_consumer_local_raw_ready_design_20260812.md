# Fused5 Consumer-Local Raw Ready Design (Rejected)

Date: 2026-08-12

## Hypothesis

The canonical five-GEMM kernel closes every q tile through a cross-role loop:

```text
producer: Q/dO/sidecar load -> RawFilled
consumer0/1: score, softmax, dV, dP, dS, dK -> RawUsed
producer: wait RawUsed before reusing the page
```

This makes the hottest ownership edge recur once per q tile. The candidate
keeps K/V as a producer-only startup load, then lets each heavy consumer own
one raw Q/dO page and publish its own ready token. Q/dO are intentionally
loaded twice. The trade is admitted only if removing the reverse ownership
edge is worth the extra global traffic.

## Algorithm And Work

The formulas and exact five GEMMs are unchanged:

```text
S=Q@K^T, dP=dO@V^T, dV=P^T@dO, dK=dS^T@Q, dQ=dS@K
```

Each consumer still owns a distinct Nk=64 half and computes all four dKV
GEMM islands. The dQ writer still consumes both group-local dS pages. No
score/dP duplication is introduced.

## Resource Ledger

```text
resident K/V: 64 KiB, latched before raw staging
consumer0 raw Q+dO page: 32 KiB at LDS[0:32 KiB]
consumer1 raw Q+dO page: 32 KiB at LDS[96:128 KiB]
P/dS and sidecar reuse: 32 KiB K/V region after resident latch
```

The allocation remains exactly 128 KiB. The existing 204-VGPR heavy-role
windows and no-spill/no-scratch/bank0 gates are unchanged.

## Barrier Ledger

`ResidentFilled` remains the startup K/V token. `RawFilled0/1` become local
four-wave ready tokens, one per consumer group and q tile. `RawUsed0/1` and
`KvDsUsed` are removed because no producer reuses the raw pages or released
K/V region. The dS `BatchDsFilled` and `DqDone` tokens remain unchanged.

```text
consumer group: seq -> four waves matrix_load Q/dO + sidecar
                -> wait vmem/BPS -> arrive RawFilled
                -> wait RawFilled -> score/softmax/dV/dP/dS/dK
```

## Expected Pipeline

```text
time0: producer publishes K/V; C0/C1/dQ latch K/V
time1: C0 stages raw page0 while C1 stages raw page1
time2: C0/C1 enter independent score/dP MMAC islands
time3: C0/C1 softmax/dS and dV/dK islands overlap across SIMD
time4+: each group repeats local stage->ready->compute without peer reuse
```

## Result

The first source-only version compiled with role use `169/172/86`, SGPR73 and
no spill/scratch/private segment. H1/S128 passed, but H1/S1024 failed dK with
`rel_l2=0.136397`. It removed `KvDsUsed` while consumer-side staging reused
storage needed by the long-lived dK/P-dS path. The initial layout also lacked
a valid proof that the second raw page fit within the 128 KiB allocation.

The second version restored `KvDsUsed` and kept the nominal 128 KiB layout;
it still passed S128 but failed S1024 dK with `rel_l2=0.109135`, while dV/dQ
passed. Therefore the startup latch alone does not close the lifecycle: the
consumer-local raw ownership still has an unresolved S1024 dK source/layout
or generation hazard.

Decision: `REJECT_A5_LIFECYCLE_CANONICAL_RESTORED`.

## Admission

Run static gate, H1/S128 causal/noncausal, H1/S1024 causal, and resource
metadata before promotion. Require exact MMOP, correctness PASS, no
private/spill/scratch, and bank0. Compare same build/runtime against the
canonical wait-pruned baseline. A local-ready bubble or extra Q/dO traffic
that regresses ticks is a rejection even if MMAC active rises.
