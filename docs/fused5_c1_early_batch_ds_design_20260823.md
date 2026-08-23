# Fused5 C1 Early Batch-dS Design

Status: `REJECT_MMAC_CONTENTION_WAIT_REGRESSION_CANONICAL_RESTORED`.

## Single Hypothesis

The C83 writer waits for C1's complete dS batch while C1 currently performs
one dV island inside every panel before publishing dS. Keep the exact C1
score/dP order, but retain four FP16 P fragments and move the unchanged four
dV islands after C1 publishes its four-panel dS batch:

```text
control:   (dP -> score -> P -> dS -> dV) x4 -> Filled1 -> dK
candidate: (dP -> score -> P -> dS) x4 -> Filled1 -> dV x4 -> dK
```

This advances useful writer dQ MMAC without adding a barrier, page, GEMM,
global transaction or empty delay. It deliberately does not split dS below a
four-panel batch; the C0 half-publish negative proves that finer publication
causes LDS/MMAC contention.

## Exact Work And Ownership

- Tile remains `M64/N128/D128`, 16 waves and 1,280 MMAC per useful tile.
- Score, dP, dV, dK and dQ each execute exactly once.
- C0/C1 retain unique N16 dK/dV ownership; writer waves retain unique D32 dQ
  ownership.
- Dynamic causal H1/S1024 MMOP must remain exactly `88,064` under the C83
  compile-time causal symbol.
- Main matrix traffic remains MLS/BPS + `ds_read_matrix` + MMAC. P still uses
  the proven native writer/read bridge; no ordinary matrix DS read is added.

## Resource And Lifetime Budget

LDS and all twelve C83 ABarrier IDs remain byte-for-byte unchanged at 128 KiB.
C1 adds only four retained `Fragment` values:

```text
4 P panels * Vec8F16 = 4 * 4 VGPR = 16 VGPR
C83 measured C1 role                      162 VGPR
estimated candidate                      <=178 VGPR
C1 WDRA window                            204 VGPR
```

Generated role evidence is authoritative. Any private segment, scratch,
SGPR/VGPR spill, role use above 204, or compiler body duplication rejects the
candidate before PMD.

P remains live only from its panel's probability stage to the post-publication
dV island. dO remains protected by the existing raw-page ownership until all
four dV islands finish. Q remains protected through dK. Therefore
`RawFilled/RawUsed` and page-reuse timing are unchanged.

## Expected Pipeline

```text
time0  C0: score/P/dP/dS panels       C1: dP/score/P/dS panels
       W : waits BatchDsFilled1       P : useful next-page work or RawUsed

time1  C1: publish dS batch + Filled1
       W : read G1 dS -> dQ MMAC      C1: dV MMAC batch

time2  C0: publish dS / dV / dK       C1: dK -> RawUsed/Done1
       W : finish G1 -> consume G0

time3  unchanged stores, raw reuse and next q-tile recurrence
```

The desired overlap is writer MMAC versus peer C0 VALU/read work and earlier
writer readiness. C1 dV and writer dQ are both MMAC and cannot coissue; if
their new contention exceeds the removed writer wait, same-shape ticks will
regress and the candidate is rejected.

## Admission And Stop Gates

1. Static: exact MMAC/read/BPS/store/barrier counts except scheduling-only
   movement; no private/spill/scratch; C1 within 204 VGPR.
2. Correctness: H1/S128 causal and noncausal, then causal H1/S1024 full
   lifecycle; warning0 and bank0.
3. Performance: at least three interleaved same-build H1/S1024 pairs against
   C83. Ticks are primary; active/coissue alone cannot promote.
4. XCU only for a tick-competitive candidate. It must show lower writer
   `BatchDsFilled1` exposure without equivalent growth in MMAC contention,
   RawUsed, wait-LGKM or wait-VM.

Historical de-duplication: `fused5_c1_dv_before_ds` is the opposite schedule;
`fused5_c0_ds_half_publish` changes publication granularity and is not this
candidate. No matching C1 full-batch early-publication result was found.

## Actual Result

- Static gates pass with exact causal MMAC/read/ABarrier counts
  `1344/768/102`, roles `9/173/87/165`, SGPR70/VGPR128 and no
  private/spill/scratch. Compared with C83, C1 grows by only three VGPR, but
  generated wait sites grow `268 -> 276`.
- Full CPU-golden correctness passes H1/S128 causal and noncausal and causal
  H1/S1024, with warning0 and `ldsBankConflict=0`.
- Three interleaved same-build H1/S1024 pairs give fused means
  `41,047,976.7 -> 41,856,056.7` ticks (`+1.969%`) and lifecycle means
  `45,179,680 -> 46,094,230` ticks (`+2.024%`).
- The candidate is therefore rejected before fullperf/XCU. Advancing
  `BatchDsFilled1` is insufficient because the newly awakened writer dQ MMAC
  competes with the deferred C1 dV MMAC; the eight extra static waits are a
  second readiness debt. C83's panel-local C1 dV remains canonical.

Evidence:

- Remote A/B: `/zys/sb/runs/fused5_c94_ab_20260823`.
- Remote correctness: `/zys/sb/runs/fused5_c94_correctness`.
- Local archive: `outputs/019ea61f-c117-76b2-abad-e776092d47a0/c94_rejected`.
