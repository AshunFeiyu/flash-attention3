# Fused5 Causal C1 Zero-Front Prune Design

Status: `ACCEPT_CAUSAL_ZERO_WORK_PRUNE / MMAC50_OPEN`.

## Algebraic Proof

For every causal CTA, the first retained Q tile and the two dKV consumer
groups own:

```text
Q0 rows  = [k_base + 0,  k_base + 63]
C0 K rows = [k_base + 0,  k_base + 63]
C1 K rows = [k_base + 64, k_base + 127]
```

The causal predicate is `k_row <= q_row`. Therefore every C1 element in Q0 is
invalid:

```text
min(C1 K) = k_base + 64 > max(Q0) = k_base + 63
```

So C1's first-tile `score`, `P`, `dP`, `dS`, `dV` and `dK` contributions are
all exactly zero. C0 remains a true diagonal boundary and is unchanged.

## One Hypothesis

On `causal && Group==1 && qi==0`, replace the four-GEMM C1 consumer body with
a zero publication path:

1. wait for RawFilled0 so publication/phase semantics remain exact;
2. publish four native zero dS panels for the existing dQ writer;
3. arrive the existing BatchDsFilled1, RawUsed0 and DqDone1 tokens;
4. perform no score/dP/dV/dK matrix reads, VALU or MMAC for that zero region.

The dQ writer remains unchanged and consumes zero dS through the native
`ds_write_matrix -> ds_read_matrix -> MMAC` path. This first experiment does
not change barrier IDs, page ownership, producer traffic, writer work, stores
or API.

## Accumulator Correctness

C1 dV is already explicitly zero-initialized. Because the first real dK MMAC
moves to `qi=1`, C1 dK is also explicitly initialized once, then every C1
MMAC uses normal accumulation. This adds 16 `v_mov_b64`-class zero operations
per C1 wave per CTA, but removes an entire first-tile C1 four-GEMM body. C0
retains compile-time zero seeding and is untouched.

## Work And Resource Budget

| Item | Accepted | Candidate causal delta |
|---|---:|---:|
| tile / roles | M64/N128/D128, 4P+4C0+4C1+4W | exact |
| logical GEMMs | five formulas | exact with proven-zero region pruned |
| C1 first-tile dKV MMAC | 4 waves x 128 = 512 | -512 per CTA |
| writer first-tile dQ MMAC | 128 | unchanged in this phase |
| producer raw BPS | unchanged | unchanged |
| C1 first-tile matrix reads | score/dP/dV/dK packets | removed |
| zero dS matrix writes | four per C1 wave | same publication count |
| ABarrier IDs / dynamic events | 12 / existing cadence | exact |
| LDS | 128KiB | exact |
| WDRA | 16/204/204/88 | exact; no new live fragment |
| C1 dK initialization | seeded by first MMAC | +16 zero moves/wave/CTA |

For H1/S1024 causal, there are eight K-tile CTAs. The prune removes 4,096
consumer MMAC issues before scheduler replication while preserving every
nonzero formula contribution. MMAC active may move either way because total
MMOP falls; paired fused/lifecycle ticks are the primary decision metric.

## Expected Pipeline

```text
time0  producer: publish Raw0
       C0: diagonal score -> mask/P -> dP/dS -> dV/dK
       C1: wait Raw0 -> publish zero dS -> release Raw0 side early
       writer: consume zero C1 dS while C0 computes

time1  producer: publish Raw1 when C0 releases Raw0
       C0/C1: both execute the first real steady work
       writer: normal G1-first then G0 ownership order
```

Expected effect: C1's first epoch becomes useful synchronization/publication
work instead of zero GEMMs; C0 remains the RawUsed pace setter, but peer SIMD
MMAC/VALU contention and CTA triangular work shrink.

## Admission

1. Static code retains all five formula declarations and canonical main
   matrix path; no new token/page/fallback/wrong-layout path.
2. Full golden S128 causal/noncausal, S1024 causal and S2048 causal pass with
   no spill/private/scratch, warning0 and bank0.
3. Dynamic causal MMOP/read work must fall; noncausal counts must remain the
   accepted value.
4. Three S1024 pairs must lower fused and lifecycle ticks. Two S2048 pairs
   must not regress.
5. Fullperf/xcu must show fewer C1 first-epoch instructions and no producer or
   writer ownership deadlock. A lower MMAC-active percentage is acceptable
   only if normalized useful-work ticks improve and no nonzero GEMM is lost.

Workbook: section71 in the 2026-08-23 fused5 design workbook.

## Result

- Static gates pass at role use `9/173/87/162`, SGPR71/VGPR128,
  LDS128KiB, private/spill/scratch0 and bank0. The symbol retains exactly five
  formula declarations and the native matrix path.
- Full CPU-golden correctness passes S128 causal/noncausal, S1024 causal and
  S2048 causal with warning0. The noncausal path keeps the canonical work.
- Three paired S1024 runs improve fused mean
  `42,148,773 -> 41,357,377` ticks (`-1.878%`) and lifecycle mean
  `46,251,660 -> 45,538,675` (`-1.542%`), with three wins from three pairs.
- Two paired S2048 runs improve fused mean
  `78,241,800 -> 76,904,783` (`-1.709%`) and lifecycle mean
  `85,742,020 -> 84,307,405` (`-1.673%`).
- S1024 fullperf improves fused ticks
  `42,371,875 -> 41,497,820` (`-2.063%`). Dynamic work falls from
  `MMOP/VALU/LDS = 92,160/98,032/63,872` to
  `88,064/92,496/61,056`; no nonzero formula contribution is removed.
- Raw MMAC active falls `36.659% -> 36.133%` because the numerator no longer
  includes 4,096 provably zero MMOP issues. This is an effective-work/ticks
  win, not an MMAC-active promotion.
- XCU dispatch issues fall `342,752 -> 327,712` (`-4.388%`) and duration
  `93,128 -> 91,204` (`-2.066%`). The representative C1 wave loses 432
  instructions and 64 MMAC issues. Aggregate ABarrier, transpose-read wait
  and terminal ebarrier gaps fall `8.55%`, `16.88%` and `12.40%`.
- The next debt is not the removed zero region: global-load wait and
  global-store gaps rose in this capture. Any follow-up must target those
  existing readiness/tail edges without restoring invalid MMAC work or adding
  ownership objects.

Evidence:

- `/zys/sb/runs/f5c1zero_correctness_20260823`
- `/zys/sb/runs/f5c1zero_ab_20260823`
- `/zys/sb/runs/f5c1zero_s2048_ab_20260823`
- `/zys/sb/runs/f5c1zero_fullperf_20260823`
- workbook sections71-72
