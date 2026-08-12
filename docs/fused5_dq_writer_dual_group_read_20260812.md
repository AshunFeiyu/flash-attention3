# Fused5 dQ Writer Dual-Group Read Island

Date: 2026-08-12

## Hypothesis

The dQ writer consumes two dS source groups into the same accumulator. The
canonical route performs one `4 ds_read_matrix_trans + wait + 8 MMAC` island
for group0 and repeats it for group1. Use the native Shaobo dual-base reader
to issue both groups together:

```text
8 ds_read_matrix_trans -> 1 wait -> 16 dQ MMAC
```

This keeps the exact five-GEMM DAG and output ownership, and changes no
consumer-side barrier token. It targets only dQ-writer read/MMAC fragmentation.

## Resource Ledger

The two source groups require eight `Vec8F16` fragments live through the first
MMAC island. The first `16/200/200/96` allocation spilled, so the bounded
resource retry uses `16/196/196/104`; the physical per-SIMD pool remains
exactly 512 VGPRs and the current compiled consumer usage is only 163/165.

## Expected Pipeline

```text
WQ: wait BatchDsFilled0 + BatchDsFilled1
WQ: dual-base read dS0/dS1 for M-panel
WQ: wait lgkmcnt(0)
WQ: 16 MMAC into the same dQ accumulators
WQ: DqDone0 + DqDone1
```

If the 96-VGPR writer budget spills or correctness fails at H1/S128, restore
the single-group reader. If it passes, compare H1/S1024 stats; no promotion
comes from coissue or MMAC-island size alone.
