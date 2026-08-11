# Fused5 Alternating Q / P-Scratch Design

Status: `HYPOTHESIS_READY / NOT_IMPLEMENTED`

## Evidence And Constraint

The locked complete-lifecycle H1/S1024 fullperf baseline is:

```text
fused kernel ticks       72,132,060
MMAC active              22.106908%
MMOP                     92,160
ABarrier issue gap       29.38%
dQ atomic issue gap      14.06%
terminal ebarrier gap    10.28%
matrix-read wait gap     10.15%
```

Barrier top5000 maps 1,777,244 cycles to producer `RawUsed` ID4 and
1,369,332 cycles to consumer `RawFilled` ID3. The current one-page protocol is
therefore:

```text
consumer finishes dK -> RawUsed -> producer loads next Q/dO -> RawFilled
```

Commits `4170797` and `9e19075` already reject separate Q/dO tokens and a
second BPS producer. Both move the wait into `s_waitcnt_vbcnt`; neither may be
repeated.

## Lifetime Proof

For one q tile:

```text
Q       last use: dK = dS^T @ Q
dO      last use: later of dP = dO @ V^T and dV = P^T @ dO
sidecar last use: softmax/dS in the four-panel loop
P page  last use: dV for the corresponding panel
```

After the four-panel score/P/dV/dP/dS loop, dO, sidecar and the 16KB P
conversion scratch are dead, but Q remains live through dK. This permits the
next Q to occupy the dead P-scratch page while the current Q remains in its
own page.

## LDS Plan

No LDS byte is added:

| Region | Bytes | Steady role at even t | Steady role at odd t |
|---|---:|---|---|
| page A | 16,384 | Q(t) | P scratch(t) |
| dO | 16,384 | dO(t), then dO(t+1) | dO(t), then dO(t+1) |
| resident K/V -> batch dS | 65,536 | unchanged | unchanged |
| page B | 16,384 | P scratch(t), then Q(t+1) | Q(t), then Q(t+1) |
| sidecar | 768 | sidecar(t), then sidecar(t+1) | same |
| total | 115,456 | within 128KB | within 128KB |

Startup remains phased: page B is the P scratch while K/V are loaded and
latched. The existing K/V region is reused for four final dS panels only after
all consumer and dQ-writer waves latch their resident fragments.

## Ownership

Replace combined `RawUsed` with two semantic release points:

```text
EarlyUsed: 8 dKV waves have finished dO, sidecar and current P-scratch use.
QUsed:     8 dKV waves have finished dK from the current Q page.
RawFilled: next Q/dO/sidecar packet is globally ready.
```

The producer may issue `Q(t+1)` into the opposite page and overwrite
dO/sidecar immediately after `EarlyUsed(t)`. Reuse of the same Q page occurs
only at `t+2`, so `QUsed(t)` should already be complete when checked.

Merge `ResidentFilled0/1` into one token because every non-producer role
currently waits both before any resident read. The barrier ledger remains at
seven IDs, below the eight-ID limit:

```text
ResidentFilled, KvDsUsed, RawFilled, EarlyUsed, QUsed,
BatchDsFilled0, BatchDsFilled1
```

## Expected Pipeline

```text
time0:
  P0: load/latch K/V, publish raw0(Q page A, dO, sidecar)
  C0/C1/WQ: latch resident K/V views

time1, tile t:
  C0/C1: score -> P -> dV -> dP -> dS on raw(t)
  WQ:     dQ/atomic from dS(t-1)
  P0:     waits EarlyUsed(t)

time2:
  P0:     issue Q(t+1) into opposite Q/P page plus dO(t+1)/sidecar(t+1),
          then one vbcnt drain and RawFilled(t+1)
  C0/C1: publish dS(t) -> dK(t) -> QUsed(t)
  WQ:     consume dS(t), with useful work overlapping next raw publication

time3:
  C0/C1: RawFilled(t+1) should be ready or substantially aged
  page roles swap; no tensor transpose, gather or duplicate GEMM is added
```

## Resource And Work Gates

- Exactly five logical GEMMs and dynamic H1/S1024 MMOP92,160.
- Role usage must fit `8/200/200/88`; no private/spill/scratch.
- LDS remains 115,456B and `ldsBankConflict=0`.
- Matrix paths remain MLS/BPS + `ds_read_matrix` + MMAC.
- No `ds_read_b32`, bpermute, gather, duplicate Q/dO loads, or empty delay.
- H1/S128 causal/noncausal and complete H1/S1024 causal correctness pass.

## Performance Gate

Compare against the same complete-lifecycle baseline. Promotion requires lower
fused and summed lifecycle ticks. XCU must show a lower consumer RawFilled ID3
duration without recreating the rejected `s_waitcnt_vbcnt` debt. MMAC active,
coissue and matrix-read waits are explanatory metrics; they cannot override a
tick regression.

Reject immediately if dynamic page selection creates spill/private memory,
if QUsed becomes a new exposed producer wait, or if total BPS vbcnt duration
offsets the RawFilled reduction.
