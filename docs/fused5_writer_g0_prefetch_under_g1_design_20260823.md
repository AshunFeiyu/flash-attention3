# Fused5 Writer G0 Read Prefetch Under G1 MMAC

Status: `REJECT_SCALE_REGRESSION_CANONICAL_RESTORED`.

## Evidence And Single Hypothesis

C83's sampled dQ-writer wave attributes 13,056 bubble cycles across 80
`ds_read_matrix_trans_format -> s_waitcnt` gaps. The writer currently consumes
the complete single-buffered G1 dS batch, signals `DqDone1`, then waits for G0
and starts G0's first read/wait/MMAC panel from cold LDS readiness.

Move only G0 panel0's four matrix reads before G1's final eight-MMAC island:

```text
control:
  G1 panel3: read4 -> wait -> MMAC8 -> Done1
  wait G0 Filled -> G0 panel0: read4 -> wait -> MMAC8

candidate:
  G1 panel3: read4 -> wait
  wait G0 Filled -> issue G0 panel0 read4
  G1 panel3: MMAC8 -> Done1
  wait prefetched G0 panel0 -> MMAC8
```

The one admitted hypothesis is that eight useful G1 MMACs can age G0's first
four reads and remove one repeated writer first-use gap per q tile. G0 MMAC is
not moved earlier and no empty delay is inserted. `DqDone1` remains after the
final G1 MMAC semantically, but its wall-clock arrival can move later because
the G0 Filled wait and read issue now precede that MMAC; this cost must be
measured explicitly.

## Exact Work And Ownership

- Tile remains `M64/N128/D128`, 16 waves and 1,280 MMAC per useful CTA tile.
- Score, dP, dV, dK and dQ execute exactly once. Causal H1/S1024 dynamic MMOP
  remains `88,064`.
- Writer still owns unique D32 output slices. G1 is fully accumulated before
  `DqDone1`; G0 panel0 is only read early and is accumulated afterward. The
  candidate may delay `DqDone1` by the moved G0 readiness/issue interval even
  though it does not change the ownership condition.
- G0 `BatchDsFilled0/Alt` is waited before its early read, so no unready dS is
  accessed. `DqDone0/Alt` remains after all four G0 panels.
- MLS/BPS, P/dS native bridge, matrix reads, global stores, LDS addresses and
  all twelve ABarrier IDs remain unchanged.

## Resource Budget

The candidate holds four G0 dS fragments while G1 panel3 fragments and the
eight dQ accumulators remain live:

```text
G0 prefetched packet: 4 waves * Vec8F16 = 16 VGPR
C83 writer measured role                    87 VGPR
candidate target                            <=103 VGPR
candidate WDRA writer window                104 VGPR
```

Rebalance only the declared WDRA windows:

```text
control:   P16 + C0 204 + C1 204 + W 88  = 512
candidate: P16 + C0 204 + C1 188 + W 104 = 512
```

C83 C1 measures 162 VGPR, leaving 26 VGPR guard in its 188 window. LDS remains
exactly 128 KiB. Any role overflow, private segment, spill or scratch rejects
the candidate before PMD.

## Expected Pipeline

```text
time0  W: G1 panels0-2 read/wait/MMAC              C0/C1: useful dKV work
time1  W: G1 panel3 read/wait; wait G0 Filled; issue G0 panel0 read4
time2  W: G1 panel3 MMAC8                           G0 reads age in LDS pipe
time3  W: Done1 (possibly later); bounded wait G0 packet; G0 panel0 MMAC8
time4  W: canonical G0 panels1-3; Done0/Alt; store dQ
```

The expected gain is bounded: one of eight writer panel read/wait chains per
q tile. The candidate is useful only if moving the G0 Filled wait one MMAC
island earlier does not expose C0 publication latency and if the retained
packet does not disturb writer instruction packing.

## Admission And Stop Gates

1. Historical de-duplication must find no exact prior G0 packet-under-G1-MMAC
   experiment. A prior writer read8 batch is not equivalent because it raised
   outstanding LDS pressure without using peer MMAC to age the packet.
2. Static: exact causal MMAC/read/wait/ABarrier/BPS/store counts; roles fit
   `16/204/188/104`; no private/spill/scratch or duplicate body.
3. Correctness: H1/S128 causal and noncausal, then causal H1/S1024 full
   CPU-golden lifecycle; warning0 and `ldsBankConflict=0`.
4. Performance: three interleaved same-build H1/S1024 pairs against C83.
   Same-shape fused/lifecycle ticks decide; active/coissue alone cannot
   promote.
5. XCU only for a tick-competitive candidate. It must reduce writer
   trans-read first-use gaps by more than any increase in G0 Filled, C1 Done,
   producer RawUsed or terminal-store bubbles.

Reject immediately if G0 is not ready one G1 MMAC island earlier, if static
wait/read counts increase, or if the writer role exceeds 104 VGPR. Failed
source is removed; only design and result evidence remain.

## Actual Result

Static compilation preserved the intended order and exact work. The causal
symbol kept `MMAC/read/wait/ABarrier/BPS/store = 1344/768/272/102/20/56`;
roles were `P/C0/W/C1 = 9/173/101/162`, SGPR/VGPR were `70/128`, and private,
spill and scratch remained zero. Full CPU-golden correctness passed H1/S128
causal and noncausal plus H1/S1024 causal, with warning0 and bank0.

Three interleaved H1/S1024 pairs showed only a noise-scale fused improvement:

```text
C83 fused mean       41,074,063.3
candidate fused mean 40,980,333.3  (-0.228%)
C83 lifecycle mean   45,291,761.7
candidate lifecycle  45,152,380.0  (-0.308%)
```

Two interleaved H1/S2048 pairs reversed the result:

```text
C83 fused mean       76,498,695
candidate fused mean 77,069,265    (+0.746%)
C83 lifecycle mean   83,972,070
candidate lifecycle  84,424,795    (+0.539%)
```

The scaling result rejects the hypothesis before fullperf. Although G0's
first packet can age under eight G1 MMACs, the required G0 Filled wait and
read issue delay wall-clock `DqDone1`, retaining C1's page longer. The release
cost grows with the longer sequence and outweighs the covered first-use gap.
Canonical C83 source and WDRA `16/204/204/88` are restored; do not retry a
cross-group writer prefetch that places peer readiness work before `DqDone1`.

Evidence: `/zys/sb/runs/fused5_c96_correctness_20260823`,
`/zys/sb/runs/fused5_c96_ab_20260823`, and
`/zys/sb/runs/fused5_c96_s2048_ab_20260823`.
