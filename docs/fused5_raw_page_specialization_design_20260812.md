# Fused5 Raw-Page Compile-Time Specialization Design

Date: 2026-08-12

Status: `REJECT_TICKS_REGRESSION / CANONICAL_RESTORED`.

## One Hypothesis

The accepted two-page ownership topology reduces complete H1/S1024 ticks by
9.351%, but runtime `page = qi & 1` selection expands scalar control work.
Specialize page0/page1 at compile time and traverse q tiles in fixed pairs.
This should retain the ownership win while removing repeated page-base,
barrier-ID, and sidecar-page branches.

This round changes only page-control expression. It must not change:

- the five-GEMM formula or MMOP92,160;
- M64/N128/D128, 16-wave roles, or output ownership;
- two 32KB raw pages or the eight-token ABarrier ledger;
- consumer arithmetic order, matrix-read/MMAC paths, or dQ reduction.

## SQTT Basis

Relative to the accepted single-page workspace-reduction baseline, the raw
double-page candidate changes representative same-SIMD dynamic scalar work:

| Wave | Role | Scalar instruction delta | Conditional branch delta | `s_xor` delta |
|---|---|---:|---:|---:|
| wave0 | producer | +459 | +85 | +1 |
| wave1 | consumer0 | +564 | +64 | -1 |
| wave2 | consumer1 | +595 | +64 | -1 |
| wave3 | dQ writer | -1 | 0 | -1 |

Therefore the added SCA is page CFG, not additional ABarrier polling. The
double-page candidate already lowers ABarrier-following issue-gap
40.72% -> 29.23%; changing the token topology again would target the wrong
cause.

## Pseudocode

Producer:

```text
publish resident K/V
fill page0(tile0), with the existing initial KvDsUsed handoff
if tile1 exists: fill page1(tile1)

for qi = 2; qi + 1 < q_tile_count; qi += 2:
    wait Used0; fill page0(qi)
    wait Used1; fill page1(qi + 1)

if odd tail exists:
    wait Used0; fill page0(tail)

wait final Used0 and final Used1 when present
```

Each `fill pageN` is a `template<int Page>` specialization with compile-time
raw base, sidecar base, Filled token and Used token.

Each heavy consumer uses the same paired traversal:

```text
for qi = 0; qi + 1 < q_tile_count; qi += 2:
    consume<Group, page0>(qi, phase0)
    consume<Group, page1>(qi + 1, phase1)

if odd tail exists:
    consume<Group, page0>(tail, phase0)
```

`consume<Group, Page>` contains one unchanged score/P/dV/dP/dS/dK packet and
selects its raw base, sidecar base, Filled token and Used token with
`if constexpr` or compile-time constants.

## Resource And Schedule Stress

- LDS remains exactly 131,072B; no new page, fragment, or accumulator exists.
- WDRA windows remain `16/204/204/88`; actual roles must stay within the 512
  physical VGPR pool and private/spill/scratch must remain zero.
- Runtime q-tile count can be odd. The tail consumes page0 exactly once; no
  page1 wait or release is issued for a missing packet.
- Code size may grow because two page-specialized consumer bodies can be
  emitted. This is the primary risk. Static ASM size and SQC/fetch evidence
  must be compared before promotion.
- Fixed pairing must not serialize page1 fill behind page0 consumption beyond
  the already accepted page-local Used tokens.

## Admission Gates

1. Static: canonical gate PASS, private/spill/scratch0, LDS131,072B, exact
   MMOP, no ordinary matrix-path gather.
2. Correctness: H1/S128 causal and noncausal, then H1/S1024 causal.
3. Dynamic: MMOP92,160, reduction MMOP0, bank0.
4. Performance: repeated complete ticks below 53,207,245 with no compute-tick
   regression hidden by reduction noise.
5. SQTT for a candidate: producer/C0/C1 conditional branches and SCA fall;
   ABarrier issue-gap does not regress; MMAC active and duration improve or
   remain consistent with lower complete ticks.

If code-size/fetch pressure erases the scalar saving, classify `REJECT`,
restore `d62c645`, and stop page-control micro-tuning. The next tier would be
consumer useful-stagger redesign on the accepted double-page ownership.

## Measured Result

The implementation passes H1/S128 causal/noncausal and repeated H1/S1024
causal correctness. It preserves MMOP92,160, bank0, LDS131,072B and
private/spill/scratch0. Actual role use is `12/167/169/86`, metadata is
SGPR70/VGPR128.

The predicted scalar reduction occurs: dynamic SCA falls 58,336 -> 38,988.
However the compiler emits two full page-specialized bodies:

- static kernel instruction lines: 2,076 -> 4,255 (+104.96%);
- dynamic VALU: 127,352 -> 131,120;
- repeated compute mean: 50,290,467.5 -> 50,543,220 (+0.503%);
- repeated complete mean: 53,207,245 -> 53,372,410 (+0.310%);
- MMAC active: 31.476233% -> 31.629456%, but ticks regress.

This is not promotable. Production source is restored exactly to `d62c645`.
Do not retry page-control specialization on compiler `e0f10535`; a lower SCA
count does not repay duplicated instruction footprint and resource pressure.
