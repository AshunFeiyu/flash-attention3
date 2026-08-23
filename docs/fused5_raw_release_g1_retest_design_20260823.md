# Fused5 RawUsed Before Final dK on G1-First

Status: `REJECT_CLOSED_LOOP_BARRIER_MIGRATION_CANONICAL_RESTORED`.

## Changed Premise

The same release point was rejected on the old writer-G0-first topology at
`+0.686%` H1/S1024.  It is admissible for one bounded retest because accepted
commit `58e90fc` changed the measured critical edge:

- C1 `DqDone1` wait fell from `7,561` to `45` cycles.
- C1 reaches the next `RawFilled` wait earlier, exposing `7,901` cycles.
- producer ABarrier time remains the largest role-local debt at `49,679`
  cycles.

The retry is therefore not justified by the old local MMAC overlap claim.  It
tests whether the newly exposed producer-to-C1 raw-page edge can be shortened.

## Single Ownership Change

The canonical two-slot dK lag-one pipeline already completes the final
`Q3+dS3` matrix packet before the final eight-MMAC dK island:

```text
read Q2/dS2 -> dK1 -> wait
read Q3/dS3 -> dK2 -> wait -> dK3 -> RawUsed
```

Move the existing page-specific `RawUsed` arrival immediately after the final
wait and before `dK3`:

```text
read Q3/dS3 -> dK2 -> wait -> RawUsed -> dK3 from VGPR
```

After that wait, dK3 consumes only the already latched `q_buf[1]` and
`ds_buf[1]`.  No subsequent instruction may read the released raw page.

## Invariants

- Exactly five logical GEMMs; MMAC count and formulas are unchanged.
- M64/N128/D128, 16 waves and four role groups are unchanged.
- LDS remains 131,072 bytes; the two raw pages and dS pages are unchanged.
- ABarrier IDs, phase variables, participant counts and arrival count are
  unchanged.
- Matrix reads, BPS transactions, global stores and output ownership are
  unchanged.
- No extra outstanding LDS read is introduced.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- No private segment, spill, scratch or LDS bank conflict is allowed.

## Expected Pipeline

```text
time0  C0/C1 dK2 MMAC; Q3/dS3 packet ages; producer waits RawUsed
time1  Q3/dS3 ready; C0/C1 arrive RawUsed; producer may refill next page
time2  C0/C1 dK3 MMAC from VGPR; producer BPS overlaps useful MMAC
time3  next RawFilled becomes ready earlier; writer remains G1-first
```

## Admission

1. Generated ASM must keep MMAC1472, symbol matrix reads840 and ABarrier102;
   only the RawUsed arrival order may move.
2. Metadata must match SGPR82/VGPR128 with role use within the accepted WDRA
   windows and private/spill/scratch0.
3. Full CPU golden must pass H1/S128 causal and noncausal plus H1/S1024
   causal, with bank0.
4. Three interleaved H1/S1024 pairs decide the short-loop gate.
5. Two H1/S2048 pairs are mandatory.  Any repeatable scaling loss rejects the
   candidate even if S1024 wins.
6. Fullperf/xcu is winner-only and must show lower producer/RawFilled debt
   without moving the cost into wait-LGKM, barrier or MMAC contention.

Workbook: Section 58 in
`fa3_bwd_5gemm_clean_design_20260823_raw_release_g1_retest_rejected.xlsx`.

## Result

- Generated ISA remains exact at MMAC1472, symbol matrix-read840 and
  ABarrier102. Roles stay `9/176/87/164`, SGPR82/VGPR128, with zero
  private/spill/scratch.
- Full lifecycle golden passes S128 causal/noncausal and S1024 causal;
  `ldsBankConflict=0`.
- Three interleaved S1024 pairs improve mean fused ticks
  `43,289,610 -> 42,969,290` (`-0.740%`), but only two pairs win.
- Three interleaved S2048 pairs improve mean fused ticks
  `79,928,940 -> 79,628,943` (`-0.375%`), again with only two wins.
- Fullperf is nearly neutral: ticks `43,318,730 -> 43,197,245` (`-0.280%`)
  and MMAC active `35.0371% -> 35.0619%` (`+0.0248 pp`).
- The intended local edge improves: C1 RawFilled ABarrier cycles fall
  `7,901 -> 5,481`. The closed loop does not: producer ABarrier rises by
  `1,668` cycles, writer rises by `1,704`, wait-LGKM rises
  `7.4278% -> 7.7317%`, wait-VM rises `3.3248% -> 3.5661%`, and successful
  coissue falls `24,724 -> 24,423`.

The small tick signal is retained as an observation, but the hypothesis is
rejected because the cost moved from C1 to producer/writer instead of leaving
the ownership loop. Production source is restored to accepted commit
`58e90fc` behavior.
