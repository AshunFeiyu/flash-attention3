# Fused5 Alternating Q / P-Scratch Design

Status: `REJECT_CORRECTNESS / CANONICAL_SOURCE_RESTORED`

## 2026-08-11 A5 Retry Gate

The exact reusable tuple now passes A0-A4 in
`dcu-kernel-toolkit` commit `0afa714`:

```text
matrix_load_32x32_b16 t bps lds
  -> LDS base 0 or 98304
  -> normal/trans ds_read_matrix
  -> dense score-like and dK-like FP32 MMAC
```

All low/high/generation-1 fragment permutations and CPU dense oracles pass;
LDS bank conflict, private, scratch and spill are zero. Therefore the two
earlier dK failures are not evidence against the MLS32 high-page tuple. They
leave one newly informed A5 retry: make the selected Q page explicit at every
score and dK call site, make the P scratch page its exact opposite, and prove
the cross-group release ledger below. No other algorithm, tile, instruction,
or scheduling change is admitted in this retry.

The retry is `ACCEPT` only if H1/S128 causal and noncausal plus H1/S1024 causal
pass the complete lifecycle, exact MMOP remains unchanged, and the same-build
baseline comparison lowers fused and summed ticks. A correctness failure
returns immediately to the canonical source and closes this architecture.

### Retry Result

Status: `REJECT_CORRECTNESS / ARCHITECTURE_CLOSED`.

- The exact selected Q page was threaded through every score and dK call site;
  P scratch always used the opposite page. Generated ISA selected bases `0`
  and `98304` as intended, with no spill, scratch or bank conflict.
- The explicit `QUsed` retry still passed delta/dV/dQ but failed dK. Replacing
  the new token with the already-proven `KvDsUsed` release also failed dK
  (`rel_l2=0.339` causal and `0.685` noncausal).
- Per-K16 diagnostics localized the corruption to q-tile0 contributions: the
  final causal K tile, whose q-tile0 contribution is empty, was nearly exact,
  while all tiles that consume q-tile0 were wrong.
- Since the isolated tuple passes A0-A4, the rejected result is an A5
  cross-iteration page-lifetime/integration failure, not a failure of
  MLS32 normal/trans dual-view itself. No performance result is admitted.

The canonical kernel, contract and correctness harness are restored exactly to
`c58272f`. Future work must retain the canonical one-page ownership protocol or
introduce a separately proven generation ledger; it must not revive this
alternating-Q implementation through another local wait adjustment.

## Experiment Outcome

Two H1/S128 causal implementations were rejected before performance capture:

1. The direct alternating-page implementation preserved static resources and
   passed delta/dV/dQ, but dK failed with `rel_l2=0.440201`.
2. A lifetime correction waited the previous `QUsed` before the first reuse
   of old Q as P scratch. It produced the same isolated dK failure
   (`rel_l2=0.440202`), so the initial race was real but not the only defect.

Both forms retained MMOP2,560, bank0, no spill/private/scratch and the exact
five-GEMM source contract. No performance claim is valid. The existing
high-page probe uses a different MLS shape from this production path, so it
does not prove the exact `matrix_load_32x32 -> high-page normal dK` chain.
Per the two-failure rule, restore `d44ff33` and require an isolated exact-shape
layout/lifetime probe before revisiting this architecture.

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

Stress correction: the two consumer groups are not lockstep. A faster group
can enter iteration `t+1` and reuse the old Q page as P scratch while a slower
group is still reading Q for `dK(t)`. Therefore every consumer waits
`QUsed(t)` immediately before its first P-scratch write in iteration `t+1`.
Score, dP, softmax and dS work that does not touch that scratch page remains
before the wait so the ownership latency can age behind useful work.

Stress revision before implementation: retain `ResidentFilled0/1`. Although
all non-producer roles eventually wait both, merging them would delay the
first consumer group and contaminate the raw-lifetime experiment with a
startup scheduling change. Use the eighth hardware ID for `EarlyUsed`:

```text
ResidentFilled0, ResidentFilled1, KvDsUsed, RawFilled, QUsed,
BatchDsFilled0, BatchDsFilled1, EarlyUsed
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
  C0/C1: RawFilled(t+1) should be ready or substantially aged; execute
         score/dP/softmax, then wait QUsed(t) only before first scratch reuse
  page roles swap; no tensor transpose, gather or duplicate GEMM is added
```

## Resource And Work Gates

- Exactly five logical GEMMs and dynamic H1/S1024 MMOP92,160.
- Role usage must fit `8/200/200/88`; no private/spill/scratch.
- LDS remains 115,456B and `ldsBankConflict=0`.
- Matrix paths remain MLS/BPS + `ds_read_matrix` + MMAC.
- No `ds_read_b32`, bpermute, gather, duplicate Q/dO loads, or empty delay.
- H1/S128 causal/noncausal and complete H1/S1024 causal correctness pass.

## Performance Gate (Not Reached)

Compare against the same complete-lifecycle baseline. Promotion requires lower
fused and summed lifecycle ticks. XCU must show a lower consumer RawFilled ID3
duration without recreating the rejected `s_waitcnt_vbcnt` debt. MMAC active,
coissue and matrix-read waits are explanatory metrics; they cannot override a
tick regression.

Reject immediately if dynamic page selection creates spill/private memory,
if QUsed becomes a new exposed producer wait, or if total BPS vbcnt duration
offsets the RawFilled reduction.
