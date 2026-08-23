# Fused5 Causal C0 Zero-Panel Prune

Date: 2026-08-23

Status: `REJECT_PMD_REGISTER_INIT / OBSERVE_PERF`

## Hypothesis

For a causal diagonal CTA with K rows `[k, k+127]`, the first Q tile owns
rows `[k, k+63]`. Consumer group C0 is split into four N16 owners:

| C0 owner | K rows | Fully invalid M16 panels |
|---|---|---|
| 0 | `k+0:k+15` | none |
| 1 | `k+16:k+31` | M0 |
| 2 | `k+32:k+47` | M0, M1 |
| 3 | `k+48:k+63` | M0, M1, M2 |

For each listed owner/panel rectangle, every K row is greater than every Q
row. Causal masking therefore makes P and dS exactly zero. Score, dP and the
softmax/dS elementwise path are dead work for six owner-panel pairs per
diagonal CTA.

This first candidate removes only that dead prefix. It deliberately keeps the
existing zero-contribution dV and dK MMACs so that C0's accepted first-MMAC
accumulator seed, dS publication layout, writer schedule and ABarrier protocol
remain unchanged.

## Work Ledger

One owner/panel score or dP product contains eight D128 MMAC instructions.
The candidate removes:

```text
6 owner-panels * 2 GEMMs * 8 MMAC = 96 MMAC / diagonal CTA
```

At H1/S1024 there are eight causal K tiles, so the expected dynamic reduction
is 768 MMOP instructions. dV and dK retain their zero-contribution MMACs in
this tier. A future full-panel prune is not admitted until this low-risk tier
shows that control cost is smaller than removed work.

## Invariants

- Five logical GEMMs and M64/N128/D128 remain the canonical algorithm.
- The optimization removes only mathematically invalid causal work.
- No new LDS page, ABarrier token, global transaction or output owner.
- LDS remains 128 KiB; WDRA remains 16/204/204/88.
- Main matrix traffic remains MLS/BPS + `ds_read_matrix` + MMAC.
- C1 zero-front, dS native publication, dQ writer order and stores are unchanged.
- Noncausal execution must be byte-for-byte equivalent at the source path.

## Expected Pipeline

```text
time0  producer: publish diagonal Raw0
       C0 owner0: normal M0 work
       C0 owner1-3: publish zero state for invalid M0
       C1: accepted whole-group zero-front path

time1  C0 owner0-1: normal M1 work
       C0 owner2-3: publish zero state for invalid M1
       writer: consume C1 zero dS while C0 continues

time2  C0 owner0-2: normal M2 work
       C0 owner3: publish zero state for invalid M2

time3  all C0 owners: normal M3 work, then canonical dV/dK and ownership release
```

The desired outcome is less C0 MMAC/VALU contention during the diagonal
startup without changing the steady q-loop conveyor.

## Admission

1. Build and all fused5 static gates pass.
2. H1/S128 causal and noncausal correctness pass.
3. H1/S1024 causal lifecycle correctness passes with no NaN/Inf.
4. Private segment, spill and scratch remain zero; LDS bank conflicts remain zero.
5. Dynamic MMOP falls by approximately 768 at S1024.
6. Three paired H1/S1024 runs improve fused and lifecycle ticks. A positive
   result is accepted even if raw MMAC-active percentage falls, because dead
   MMOP is intentionally removed.
7. If accepted, validate H1/S2048 scaling and capture one fullperf/xcu trace.

## Failure Boundary

Reject and restore if the owner-dependent branch or zero-state setup increases
VALU/SCA enough to erase the removed work. Do not immediately extend the
experiment to dV/dK or dQ-writer pruning; that would change accumulator seeding
or writer source selection and is a separate hypothesis.

## Result

- Formula and semantic correctness are confirmed. H1/S128 causal/noncausal
  and H1/S1024 causal all match the complete CPU golden; bank conflicts are
  zero and static metadata remains SGPR71/VGPR128 with no spill, scratch or
  private segment.
- The expected work deletion occurs exactly at S1024:
  `MMOP 88,064 -> 87,296` (`-768`), `VALU 92,496 -> 91,144`, and
  `LDS 61,056 -> 60,720`. The observed fused tick is `40,624,220`, about
  `1.77%` below the accepted three-run mean, but this is not an admitted
  performance comparison because the PMD warning gate fails.
- PMD reports `read vgpr VirId93 PhyId197 before writing` twice at S128 and
  sixteen times at S1024. ASM shows compiler-generated PHI copies such as
  `v_mov_b32 v68, v93` before the zero branch defines the register.
- Three source forms were tested: shared-zero copy, direct branch-local zero,
  and first-tile predefinition protected from DSE. All preserve numerical
  results but retain the same register-init warning family.
- Decision: reject the source from canonical and restore byte-exact commit
  `2f73cab`. Keep the remote experiment as a compiler/PMD minimalization lead.
  Reopen only with a branch-free owner specialization that does not multiply
  the hot instruction body, or after compiler/PMD register-init tracking is
  fixed.

Evidence:

- `/zys/sb/runs/f5c0zero_correctness_20260823`
- `/zys/sb/runs/f5c0zero_correctness2_20260823`
- `/zys/sb/runs/f5c0zero_correctness3_20260823`
- `/zys/sb/runs/f5c0zero_correctness4_20260823`
- remote source repro:
  `/zys/sb/experiments/fused5_causal_c0_zero_panels_20260823_c75`
