# Fused5 Full Raw Q/dO Double-Page Design

Date: 2026-08-12

Status: `ACCEPT_CANONICAL_STRUCTURAL_WIN / MMAC50_OPEN`.

## One Hypothesis

The accepted H1/S1024 SQTT attributes the longest steady producer edge to:

```text
dK(t) -> RawUsed(t) -> producer Q/dO MLS(t+1) -> RawFilled(t+1)
```

Use the LDS released by the admitted 1KB P/dS writer stride to keep two
complete and physically disjoint raw packets. The producer may fill packet
`t+1` while consumers use packet `t`; page reuse waits only for `t-2`.

This round changes only raw ownership and LDS placement. The M64/N128/D128
tile, five logical GEMMs, output ownership, consumer arithmetic order, and dQ
workspace reduction remain fixed.

## Why This Is Not The Rejected Q-Only Experiment

The rejected qdouble path added only a second Q region while dO and sidecar
remained shared and generations used one ordered `RawFilled` plus
`QUsed/EarlyUsed`. It failed dK when next-generation BPS overlapped current dK.

This candidate must use:

- a full 32KB `Q+dO` page for each generation;
- page-local `RawFilled0/1` and `RawUsed0/1` barriers;
- no `EarlyUsed`, no single cross-page phase token, and no page alias;
- compile-time-stable native MLS/BPS and `ds_read_matrix` matrix paths.

If dK still fails, the evidence closes concurrent next-page BPS versus current
dK at operator A5; no wait-order micro-tuning is permitted.

## Exact Work

For every M64 x N128 tile:

```text
score = Q @ K^T
dP    = dO @ V^T
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

The operator remains exactly 1,280 MMAC per tile and H1/S1024 causal remains
MMOP92,160. No duplicate score/dP and no gather/permute path are admitted.

## LDS Ledger

Startup reaches, but does not exceed, 128KB:

| Address | Startup owner | Steady owner | Bytes |
|---:|---|---|---:|
| 0KB | raw Q0+dO0 | raw Q0+dO0 | 32KB |
| 32KB | resident K | four packed dS panels | 32KB |
| 64KB | resident V | P scratch + sidecar0/1 | 32KB |
| 96KB | raw page1 unused | raw Q1+dO1 | 32KB |

Detailed steady placement:

```text
batch dS       [32KB, 64KB)
P scratch      [64KB, 72KB)
sidecar0       [72KB, 72KB+768B)
sidecar1       [72KB+768B, 72KB+1536B)
raw page1      [96KB, 128KB)
```

The first raw packet is not published until all 12 non-producer waves latch
resident K/V and complete the initial `KvDsUsed` generation, making the
sidecar region legal to overwrite.

## Barrier Ledger

All eight hardware IDs have one semantic owner:

```text
0 ResidentFilled   count4: each producer wave arrives once after all K/V
1 KvDsUsed         count12: resident latch, then batch-dS reuse generations
2 RawFilled0       count4
3 RawUsed0         count8
4 RawFilled1       count4
5 RawUsed1         count8
6 BatchDsFilled0   count4
7 BatchDsFilled1   count4
```

Merging resident readiness is legal because every consumer and dQ writer
already waited for both old resident barriers before reading any K/V fragment.

## Expected Pipeline

```text
time0
  P0: MLS K/V -> ResidentFilled; issue raw0 Q/dO
  C0/C1/WQ: wait ResidentFilled, latch K/V, arrive KvDsUsed

time1
  P0: wait initial KvDsUsed; publish sidecar0 + RawFilled0
  C0/C1: wait page0, score/dP/P/dV/dS
  P0: concurrently MLS Q1/dO1 + sidecar1 -> RawFilled1

time2
  C0/C1: publish dS0, dK0, RawUsed0
  WQ: dQ0
  C0/C1: consume already-aged RawFilled1
  P0: wait RawUsed0 only before refilling page0 with packet2

time3+
  page0/page1 alternate; producer page reuse is two packets behind consumers
```

Expected SQTT change: producer RawUsed aggregate and consumer RawFilled gaps
fall without increasing MMOP, BPS traffic, or barrier events per packet.

## Admission Gates

1. Static: role resources within `16/204/204/88`, private/spill/scratch0,
   LDS exactly 131,072B, canonical source gate PASS.
2. Correctness: H1/S128 causal and noncausal, then H1/S1024 causal.
3. Dynamic: compute MMOP92,160, reduction MMOP0, `ldsBankConflict=0`.
4. Performance: repeated total ticks below 58,696,137.5; compact-stride-only
   59,148,407.5 is an implementation control, not the promotion baseline.
5. SQTT only for a performance candidate: lower page-local Filled/Used gaps,
   improved MMAC active, and no new vbcnt/LDS wait debt.

Immediate reject conditions are dK corruption, spill/private/scratch, bank
conflict, extra MMOP, or a same-shape total-tick regression.

## Measured Result

All admission gates pass on LLVM `e0f10535`, PMD HEAD1694, `GPU_CHIP=sb`,
and `GPU_ARGS=['--SQCIPfLines=7']`:

- H1/S128 causal and noncausal plus H1/S1024 causal dQ/dK/dV CPU golden PASS.
- Compile windows are `16/204/204/88`; actual role use is `9/161/163/86`.
- SGPR60/VGPR128/LDS131,072B, private/spill/scratch0, bank0.
- Compute MMOP92,160 and reduction MMOP0; the five-GEMM ledger is unchanged.

Repeated complete H1/S1024 lifecycle ticks are 53,093,495 and 53,320,995,
mean 53,207,245. This is 9.351% below the accepted 58,696,137.5 mean.
Compute ticks improve 10.202%; fullperf total improves 8.879%.

SQTT confirms the intended ownership change:

- fused compute duration falls 123,436 -> 111,792 cycles;
- MMAC active rises 28.897238% -> 31.476233% in repeated stats;
- barrier share falls 22.814064% -> 15.940946%;
- dominant ABarrier-following issue-gap falls 40.72% -> 29.23%;
- C0/C1 still execute 2,048 MMAC each, with 395/641 MMAC instructions
  paired with peer vector work. The remaining imbalance is real, not missing
  MMAC work.

The next hypothesis must preserve this two-page ownership while reducing the
new runtime page-selection/control cost. Dynamic SCA rises 38,192 -> 58,336;
therefore compile-time page specialization is the next bounded experiment.
