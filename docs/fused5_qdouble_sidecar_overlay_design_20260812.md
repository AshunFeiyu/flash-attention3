# Fused5 Q-Double / Sidecar Overlay Design

Status: `REJECT_A5_DK_LIFETIME / CANONICAL_RESTORE_REQUIRED`.

## One Hypothesis

The accepted H1/S1024 SQTT maps the dominant ownership gap to the single raw
page:

```text
consumer dK finishes -> RawUsed(id4) -> producer Q/dO BPS -> RawFilled(id3)
```

Keep the exact five-GEMM math, M64/N128/D128 tile, 16-wave role split,
resident K/V fragments, batch-dS publication and dQ writer. Add one dedicated
16 KiB Q page so the producer can publish Q(t+1), dO(t+1) and sidecar(t+1)
while the consumers finish dS publication, dK and dQ for tile t.

This is an ownership/lifetime experiment. It must not change MMOP, output
ownership, matrix layout, or arithmetic order.

## Why The Historical Q Latch Is Not Reused

Commit `d7308d4` held all four Q panels in each old heavy consumer and improved
ticks by 1.03%. In the current 16-wave native lag-one topology the same
`QNormalBatch` adds 64 VGPR per heavy role. The attempted
`8/208/208/88` WDRA allocation spilled even though it consumed the full
physical 512-VGPR pool. The current panel-streamed Q path is therefore a hard
resource invariant.

The rejected alternating-Q experiment also cannot be copied: it alternated
one physical page between Q and the P writer scratch. Its A5 integration
failed dK correctness even after the exact high-page MLS/read tuple passed.
The new design gives Q1 a dedicated page and never aliases a Q generation with
P scratch.

## Formula And Exact Work

For each M64 x N128 tile:

```text
score = Q @ K^T
dP    = dO @ V^T
P,dS  = softmax-sidecar(score,dP)
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

There are exactly five logical GEMMs, 256 MMAC per logical GEMM and 1,280
MMAC per tile. H1/S1024 causal must remain MMOP=92,160.

## LDS Budget

The target physical layout is exactly 128 KiB:

| Region | Bytes | Startup | Steady state |
|---|---:|---|---|
| Q0 | 16,384 | Q(0) | Q(even) |
| dO | 16,384 | dO(0) | current dO |
| K/V | 65,536 | resident K/V | four-panel batch dS |
| Q1 | 16,384 | unused | Q(odd) |
| P scratch + sidecar overlay | 16,384 | sidecar(0), then P | sidecar(t), then P |
| Total | 131,072 | | |

The sidecar overlay is admitted only if a focused writer-footprint probe proves
that every sidecar byte is outside the addresses touched by
`ds_write_matrix_32x16_trans_f16`. The producer publishes one common 768-byte
sidecar copy in the untouched half of writer page 0. `RawFilled` already guards
that producer-to-consumer transfer, and every P writer touches only the lower
half of its own page, so replication and an extra latch barrier are unnecessary.

The locked e0f10535/HEAD1694 probe at
`/zys/shaobo_runs/fused5_padding_probe_20260812/layout_probes/`
proved an exact writer footprint of `[0,1024)` bytes and an unchanged aligned
`[1024,2048)` region. The probe passed with one native writer, no PMD panic,
SGPR13/VGPR17 and zero private/spill/scratch.

## WDRA Budget

```text
target windows       producer/C0/C1/dQ = 8/200/200/88 = 496
accepted role use                         8/163/166/84
new long-lived matrix fragments                       0
new state                            SGPR phases/bases only
physical guard                                      16 VGPR
```

The candidate is rejected before PMD if either heavy role exceeds 200 VGPR,
the four windows exceed 504, or any private/spill/scratch appears.

## Ownership Ledger

Use all eight ABarrier IDs. `QUsed` remains a single ordered token: after
publishing raw tile `t+1`, the producer immediately consumes `QUsed(t)` before
advancing. Thus at most one completed Q generation can be outstanding and the
phase bit cannot wrap, while the next raw packet has already been published.

```text
0 ResidentFilled0  producer group0 K/V -> all non-producer roles
1 ResidentFilled1  producer group1 K/V -> all non-producer roles
2 KvDsUsed         K/V latch release, then batch-dS generation release
3 RawFilled        Q(slot)+dO+sidecar packet ready
4 QUsed            current Q slot no longer needed by dK
5 BatchDsFilled0   consumer0 dS batch ready
6 BatchDsFilled1   consumer1 dS batch ready
7 EarlyUsed        dO, sidecar and P scratch no longer needed
```

Producer generation rules:

```text
tile0: load Q0+dO+sidecar, arrive RawFilled
tile1: wait EarlyUsed(0), load Q1+dO+sidecar, arrive RawFilled
tileN: wait EarlyUsed(N-1), load Q[N&1]+dO+sidecar, publish RawFilled(N),
       then wait QUsed(N-1) before advancing
```

Consumer generation rules:

```text
wait RawFilled(t)
score/dP/P/dV/dS for four M16 panels
arrive EarlyUsed(t)
publish batch dS, run dK from Q[t&1]
arrive QUsed(t)
arrive KvDsUsed(t)
```

The dQ writer remains unchanged and arrives `KvDsUsed` after consuming both
batch-dS groups.

## Expected Pipeline

```text
time0  P:  K/V startup + raw0
       C:  latch resident K/V, then consume raw0
       WQ: latch K normal

time1  P:  wait EarlyUsed0
       C:  score/dP -> softmax/P -> dV/dS (raw0)
       WQ: idle startup

time2  P:  load raw1 into Q1 + shared dO + sidecar overlay
       C:  batch dS0 -> dK0
       WQ: dQ0 -> atomic0

time3  P:  wait EarlyUsed1 and QUsed0 before raw2 reuses Q0
       C:  consume already-aged raw1
       WQ: finish dQ0 while C starts score/dP1
```

Success means consumer `RawFilled(id3)` becomes hidden or substantially
smaller. A producer `QUsed` wait may grow while peer consumers execute useful
work; that is not a regression unless same-shape ticks grow.

## Gates

1. A0/A1 writer-footprint probe: exact touched-byte map, no PMD panic.
2. A2 overlay proof: selected sidecar offsets remain unchanged after the
   production writer mode.
3. Static kernel: exact MMOP source contract, MLS/BPS + ds_read_matrix + MMAC,
   no scalar matrix workaround, no spill/private/scratch, LDS=131,072 B.
4. Correctness: H1/S128 causal and noncausal, then complete H1/S1024 causal.
5. Performance: repeated stats and one fullperf versus `b28e73d`.
6. XCU: compare RawFilled/RawUsed-QUsed/EarlyUsed durations, BPS vbcnt debt,
   MMAC active and same-SIMD MMAC+VALU overlap.

Promotion requires lower fused and lifecycle ticks. MMAC active or coissue
alone cannot promote the candidate.

## A5 Result

The reusable writer-padding probe passed, but the complete operator rejected
the overlapping packet schedule before performance:

| Case | dK rel-L2 | dV | dQ | MMOP | Bank conflict |
|---|---:|---|---|---:|---:|
| causal, raw(t+1) before QUsed(t) | 0.650218 | PASS | PASS | 2,560 | 0 |
| noncausal, raw(t+1) before QUsed(t) | 1.01811 | PASS | PASS | 2,560 | 0 |
| causal diagnostic, QUsed(t) before raw(t+1) | 0.00123603 | PASS | PASS | 2,560 | 0 |

All three builds retained role usage `8/163/166/84`, SGPR60/VGPR124,
private/spill/scratch0 and exact LDS=131,072B. Moving only the ordered QUsed
wait ahead of the next MLS/BPS packet restored dK, so the five-GEMM formula,
high-page standalone tuple, sidecar overlay and generated work are not the
failure. The unsupported point is the A5 overlap between a next-generation
raw MLS/BPS packet and current-generation dK LDS consumption. Whether that is
a PMD limitation or an undocumented production instruction-lifecycle rule is
not proven here.

This route is closed without a performance capture. The already-correct split
dO/sidecar lifetime experiment (`4170797`) reduced ABarrier time but regressed
ticks by 0.65% because the debt moved into `s_waitcnt_vbcnt`; do not repeat it.
Restore `b28e73d` and move the next hypothesis up to tile/ownership topology.
