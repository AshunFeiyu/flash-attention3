# Fused5 Probability Sidecar Latch

## Evidence

The current single-die H1/S1024 SQTT baseline is correct, spill-free and
bank-conflict-free, but the fused dispatch still executes 6,912 scalar
`ds_read_b32` instructions. Each heavy consumer reads three sidecar values for
each of four M16 panels. `S_MAX` and `S_SUM` are first used only after the
score/dP MMAC island, so reading them at that point fragments the hot path.

## Single Hypothesis

After `RawFilled`, latch only `S_MAX` and `S_SUM` for all four panels into
consumer VGPRs, then issue the canonical first matrix packet. Keep `delta`
panel-local because it is needed later by dS and carrying all 12 values would
extend live ranges unnecessarily.

```text
consumer tile start:
  wait RawFilled
  read 4 x S_MAX + 4 x S_SUM
  issue first Q or dO matrix packet
  score/dP MMAC hides sidecar readiness
  use latched S_MAX/S_SUM in softmax
  read panel-local delta immediately before dS
```

The five-GEMM DAG, M64/N128/D128 tile, MMOP count, raw/dS page ownership,
ABarrier ledger, LDS layout and output stores remain unchanged.

## Resource And Correctness Budget

- Eight additional FP32 values live for one q tile: estimated `+8 VGPR` per
  heavy consumer.
- Current measured branch use is C0/C1 `187/182` under 204-VGPR windows, so
  estimated use is `195/190`; generated metadata is authoritative.
- No ordinary DS instruction enters the matrix path. Sidecar scalar LDS reads
  remain permitted metadata traffic.
- Correctness requires the existing `RawFilled` edge before every latch.

## Admission

1. ASM shows an eight-read sidecar island before the first matrix packet, not
   a compiler-created immediate wait after every scalar read.
2. No private segment, spill or scratch; exact MMOP remains 92,160.
3. H1/S128 and H1/S1024 causal full correctness pass with bank conflict zero.
4. Repeated same-mode H1/S1024 fused ticks improve. A fullperf/xcu capture is
   admitted only after the stats gate; it must show lower sidecar/read wait or
   a cleaner MMAC/VALU cadence without increased ownership debt.

## Result

Status: `REJECT_LGKM_LIVE_RANGE_CANONICAL_RESTORED`.

The generated ISA contains the intended eight contiguous scalar sidecar
reads, followed by eight matrix reads, `lgkmcnt(4)` and the score MMAC island.
Resources remain clean at branch use `9/191/87/183`, SGPR82/VGPR128, with no
private segment, spill or scratch. H1/S128 and both H1/S1024 candidate runs
pass full correctness with bank conflict zero.

Two interleaved H1/S1024 pairs regress fused mean ticks
`44,944,900 -> 45,298,208` (`+0.79%`) and lifecycle mean ticks
`49,056,280 -> 49,480,795` (`+0.87%`). In the first pair, LDS instructions
fall `63,872 -> 62,080`, but wait-LGKM rises `7.29% -> 8.17%` and MMAC active
falls `33.92% -> 33.82%`. The longer sidecar live range perturbs the existing
matrix-packet readiness more than the regular read island helps. Remove the
candidate and do not pre-latch a full q tile of probability sidecar values.
