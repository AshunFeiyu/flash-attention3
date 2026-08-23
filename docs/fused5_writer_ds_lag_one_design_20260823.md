# Fused5 dQ Writer dS Panel Lag-One Read

Status: `REJECT_LOCAL_READINESS_WIN_BARRIER_MIGRATION_CANONICAL_RESTORED`.

## Measured Trigger

The accepted G1-first fullperf still spends `7.428%` in wait-LGKM. The dQ
writer consumes each published dS group as four serial panel islands:

```text
4 ds_read_matrix -> lgkmcnt(0) -> 8 MMAC
```

Every group is already fully published before the writer enters it, so panel
`t+1` has no ownership dependency on panel `t`'s MMAC result. Only its VGPR
destination must remain live.

## Exact Change

- Split dS panel reading from MMAC consumption.
- Read panel0 and wait as startup.
- For panels1-3, issue the next four matrix reads before consuming the current
  panel's eight MMACs; wait only at the next panel's first use.
- Preserve G1-first group order, all 12 ABarrier objects, LDS pages, formulas,
  MMAC count, matrix-read count, accumulator order and output stores.

Expected steady writer sequence:

```text
read panel(t+1) -> 8 MMAC panel(t) -> lgkmcnt(0) -> consume panel(t+1)
```

## Resource Proof

One additional four-fragment dS packet is live. Each fragment is eight FP16
values, or four VGPRs, so the expected writer increase is 16 VGPRs. The
accepted writer uses 87 of 88 while C1 uses 164 of 204. Reassign the fixed
512-register pool from `16/204/204/88` to `16/204/184/108`:

- producer: measured9, budget16
- C0: measured176, budget204
- C1: measured164, budget184
- writer: expected <=103, budget108

No physical VGPR capacity is added. LDS stays 131,072 bytes; SGPR target82,
MMAC1472, matrix-read840 and ABarrier102 must remain exact. Any private segment,
spill, scratch, bank conflict, extra read or hidden work rejects the candidate.

## Admission

Run static gates, S128 causal/noncausal and S1024 full CPU golden, then three
interleaved S1024 pairs. S2048/fullperf/xcu are admitted only if paired ticks
improve. XCU must prove that writer `ds_read_matrix -> lgkmcnt -> MMAC` gaps
fall without moving more time into ABarrier, wait-VM or terminal stores.

Workbook: section60 in
`fa3_bwd_5gemm_clean_design_20260823_writer_ds_lag_one_rejected.xlsx`.

## Result

The candidate is legal and the requested pipeline is present in generated
ISA. Writer use is `101/108`, C1 use is `164/184`, SGPR82/VGPR128 and
private/spill/scratch0 pass. Static MMAC1472, matrix-read840 and ABarrier102
are exact. The assembly contains the next four `ds_read_matrix` instructions
before the current eight-MMAC island. S128 causal/noncausal and S1024 full
CPU-golden correctness pass with bank0.

Three interleaved S1024 pairs nevertheless regress fused mean
`43,091,078 -> 44,139,853` (`+2.434%`) with zero candidate wins. The local
hypothesis succeeds because wait-LGKM falls `7.746% -> 7.226%` (`-0.520 pp`).
The closed loop does not improve: ABarrier rises `13.325% -> 14.385%`
(`+1.060 pp`), coissue success falls `3.791%`, and MMAC active falls
`35.229% -> 35.073%`.

The writer reaches the next ownership wait earlier, so dS-read readiness was
not the CTA critical edge. S2048/fullperf/xcu are not admitted. Canonical
source and WDRA are restored exactly to `58e90fc`. Do not retry writer-only
read-ahead without changing or eliminating the ownership wait it runs into.
