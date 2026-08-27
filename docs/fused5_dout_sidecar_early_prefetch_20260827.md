# Fused5 dO/sidecar early prefetch

Status: `REJECT_BARRIER_MIGRATION_CANONICAL_RESTORED`.

## Hypothesis

C111 keeps each raw Q/dO page until both consumer groups finish dK, although
dO is dead after dV. Its SQTT attributes about 478K recurring producer bubble
cycles to page-local RawUsed waits. Split the existing lifetime without extra
LDS or extra matrix loads:

1. make both groups' dS-ready publication imply that dV is complete;
2. prefetch dO and sidecar for tile `t+2` into the dead half of page `t`;
3. retain Q until the unchanged RawUsed event after dK;
4. load only Q after RawUsed, then publish the complete raw packet.

The steady post-RawUsed burst falls from four BPS matrix loads per producer
wave to two. Total BPS, five-GEMM work, tile, output ownership and LDS bytes
remain exact.

## Barrier ledger

C0 already has page-specific dS-ready tokens. C1 gains one alternate
page-ready token so the thin producer cannot miss two phase transitions.
This uses 15 of 16 ABarrier IDs. No new consumer arrival is added: C1 chooses
one of two ready tokens instead of always using the same token.

The producer waits for page-local C0 and C1 dS-ready, publishes the next
sidecar, waits the existing RawUsed, then publishes Q/raw readiness. If these
two ready waits merely replace the RawUsed bubble, the candidate is rejected.

## Gates

- Exact MMAC, BPS, matrix-read and output-store counts.
- LDS 128 KiB; private/spill/scratch0; bank0.
- Full golden S128 causal/noncausal and S1024 causal.
- Three interleaved H1/S1024 same-build A/B pairs.
- Fullperf/xcu only if fused and lifecycle ticks improve.

Shared workbook sheet: `F5_DOut_EarlyPrefetch`.

## Result

Static gates pass with exact causal MMAC `1344`, matrix-read `768`,
LDS `128 KiB`, roles `9/142/87/130`, metadata VGPR `128`, and no
private/spill/scratch. ABarrier sites rise `124 -> 130`; waitcnt sites fall
`300 -> 293`. Complete golden correctness passes S128 causal/noncausal and
S1024 causal with warning0 and bank0.

Three interleaved S1024 pairs reject the schedule:

| Pair | Control fused | Candidate fused | Control lifecycle | Candidate lifecycle |
|---|---:|---:|---:|---:|
| 1 | 38,692,745 | 39,383,435 | 42,782,285 | 43,533,490 |
| 2 | 38,567,165 | 39,512,655 | 42,703,115 | 43,512,105 |
| 3 | 38,769,185 | 39,460,330 | 42,890,120 | 43,624,945 |
| Mean | 38,676,365 | 39,452,140 | 42,791,840 | 43,556,847 |

Fused ticks regress `2.006%`; lifecycle ticks regress `1.788%`, with all
three pairs losing. The extra page-local dS-ready waits and delayed C0 dS
publication replace rather than hide the former RawUsed bubble. Per the gate,
no fullperf/xcu run is admitted. Canonical source is restored to commit
`458a42a`; only this negative design evidence remains.

Evidence:
`/zys/sb/ab_dout_early_20260827`,
`/zys/sb/f5_dout_early_prefetch_20260827/correctness_s1024`, and remote
candidate archive `/zys/fa3_bwd_dout_early_prefetch_20260827`.
