# Fused5 C0 dV-Tail dK0 Cover

Status: `ADMIT_DESIGN_ONLY`.

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
