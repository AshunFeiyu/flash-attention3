# Fused5 C0 dV-Tail dK0 Cover

Status: `REJECT_S2048_OUTSTANDING_READ_PRESSURE_CANONICAL_RESTORED`.

## Hypothesis

The canonical dK lag-one pipeline still exposes the first `Qx4+dSx1` packet.
Issue that packet after the final dV panel's P/dO operands, use
`lgkmcnt(5)` to make only dV ready, execute the useful eight-MMAC dV island,
then wait for and consume dK0.

```text
canonical: dV0 dV1 dV2 dV3 -> read dK0 -> wait0 -> dK0 -> lag-one dK1-3
candidate: dV0 dV1 dV2 -> read dV3 -> read dK0 -> wait5 -> dV3
           -> wait0 -> dK0 -> lag-one dK1-3
```

This differs from the rejected dK batch2 experiment, which issued two dK
packets before any useful work, and from the rejected dV read-ahead experiment,
which increased same-family dV outstanding work.

## Resource Gate

The overlap adds one five-fragment dK packet, approximately 20 VGPR. It is
placed after P panels0-2 are dead. The worst static estimate is near the C0
204-VGPR branch cap, so compile/metadata is the first admission gate. The
candidate may not raise the cap or introduce spill/private storage.

LDS remains 128 KiB, barrier IDs remain 12, MMOP remains 92,160, and all
matrix paths remain MLS/BPS + `ds_read_matrix` + MMAC.

## Promotion

Require exact ASM ordering, H1/S128 causal/noncausal correctness, two
interleaved H1/S1024 pairs, bank0 and lower mean ticks. Fullperf/xcu is admitted
only after the ticks gate and must show a shorter first dK first-use gap without
moving the bubble into no-VALU-ready or barrier time.

Workbook: sheet `42 C0 dV Tail dK0 Cover`.

## Result

All static gates pass unchanged: exact MMOP 92,160, LDS 128 KiB, role VGPR
`9/187/87/182`, no private/spill/scratch, and bank conflict zero. H1/S128
causal and noncausal correctness pass.

At H1/S1024 the intended local mechanism is real. Two interleaved pairs move
fused mean ticks from `45,001,093` to `44,792,475` (`-0.464%`). Same-build
fullperf moves `45,188,325 -> 44,941,260` (`-0.547%`), and xcu duration moves
`99,316 -> 98,772`. On the representative C0 wave, total bubble cycles fall
`89,312 -> 88,776`, the largest first-use matrix-read wait falls
`13,332 -> 11,059`, and MMAC+VALU coissue rises `18.00% -> 20.54%`.

The schedule fails the scaling gate. Two H1/S2048 pairs move fused mean ticks
from `83,211,765` to `87,305,628` (`+4.919%`). Exact MMOP and instruction
counts remain unchanged, but MMAC active falls `37.877% -> 37.536%`,
`waitLgkm` rises `7.937% -> 8.382%`, successful coissue falls
`93,697 -> 92,658`, and failed coissue rises `119,879 -> 124,322`.

The extra dK packet can hide one short-loop first-use gap, but it increases the
outstanding LDS-read pressure seen by the longer q-loop. Do not promote this
schedule or retry by moving only the partial wait. Future cross-GEMM overlap
must bound outstanding reads per SIMD across all resident consumer waves.
