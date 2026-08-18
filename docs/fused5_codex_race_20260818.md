# Fused5 Codex Race

## Baseline

- Source: `e8a629e` equivalent on `race/codex-fused5-20260818`.
- Target: `B1/H1/S1024/D128`, causal, `GPU_CHIP=sb`,
  `GPU_ARGS=['--SQCIPfLines=7']`.
- Fused ticks: `47,334,105 / 46,688,915 / 47,146,190`; median
  `47,146,190`.
- MMAC active: `33.561870 / 33.747939 / 33.634715%`.
- Exact work: five GEMMs, `1,280` MMAC per CTA tile and dynamic
  `MMOP=92,160` for the target case.
- Resources: producer/consumer0/consumer1/dQ-writer VGPR
  `9/161/163/86`; SGPR `60`; LDS `131,072B`; no private, spill, or scratch;
  LDS bank conflict `0`.

The fixed DAG is:

```text
score = Q @ K^T -> P
dP = dO @ V^T
dV += P^T @ dO
dS = P * (dP - delta)
dK += dS^T @ Q
dQ += dS @ K
```

## SQTT Diagnosis

The baseline fused helper perf is under
`/zys/sb/race_codex/fullperf/b1_h1_s1024_d128_c1_fullperf_20260818_161000`.
XCU reports:

- `s_abarrier_try_wait -> s_xor_b32`: `27.66%` of issue-gap duration.
- trans matrix read to first-use wait: `11.01%`.
- normal matrix read to first-use wait: `5.47%`.
- terminal ebarrier: `7.65%`.
- MMAC-to-MMAC gaps: `5.56%`.

Barrier tomography separates visible wait from critical-path wait:

- Producer `RawUsed0/1` waits are about `1.01M` aggregate cycles, but dKV
  consumers wait only about `1-1.8K` cycles per `RawFilled` edge. The raw
  double buffer is therefore generally ready in time; producer spin is not
  the first experiment.
- dQ writer waits `467,785` aggregate cycles on `BatchDsFilled0` and
  `208,798` on `BatchDsFilled1` in the top-5000 bubble sample.
- Consumer1 and dQ writer are among the last roles to finish. Consumer MMAC
  windows are still mostly aligned: 103 sampled bins are MMAC-vs-MMAC and
  only 24 are MMAC-vs-VALU.

## H2: Early-Ready dS Group First

Consumer0 publishes dS after `score -> softmax -> dV -> dP -> dS`.
Consumer1 publishes dS after `dP -> score -> softmax -> dS`, then performs
dV. The measured writer wait confirms group1 is normally ready earlier, but
the canonical writer waits for group0 first.

Change only the dQ-writer ownership cadence:

```text
before: wait DsFilled0 -> 32 dQ MMAC -> DqDone0
        wait DsFilled1 -> 32 dQ MMAC -> DqDone1 -> store

H2:     wait DsFilled1 -> 32 dQ MMAC -> DqDone1
        wait DsFilled0 -> 32 dQ MMAC -> DqDone0 -> store
```

Expected effect: release consumer1's single dS page earlier and use its dQ
MMAC island to cover consumer0's later dS production. This is useful-work
staggering, not an inserted delay. Formula, MMAC count, LDS layout, barrier
count, output ownership, and main matrix path do not change.

Admission gates:

1. Static metadata and exact-MMOP gates pass with no new resources.
2. Full S128 and S1024 correctness pass; no private/spill/scratch and bank0.
3. Same-build repeated S1024 fused ticks improve beyond model noise.
4. If promoted, XCU must show lower combined writer ID6/ID7 wait and/or more
   MMAC-vs-VALU bins without moving the delay to a later critical edge.

Result: `REJECT_TICKS_AND_ACTIVE_CANONICAL_RESTORED`.

- S128 and three S1024 full-lifecycle correctness runs passed; exact dynamic
  instruction counts and bank0 were preserved.
- S1024 fused ticks were `48,540,310 / 48,550,775 / 48,373,325`, median
  `48,540,310`, or `+2.96%` versus the race baseline median.
- MMAC active fell to `32.533945 / 32.958656 / 32.812374%`; barrier share
  rose to `15.46-15.58%`.
- Swapping the ready-group order moved the rendezvous and damaged the useful
  existing stagger. It did not reduce the total critical ownership path.

Do not retry writer group-order swaps. The canonical group0-then-group1 path
is restored before the next hypothesis.

## H3: Group0 dS Two-Page Ownership

The current consumer0 computes dS for tile `t`, then waits for the dQ writer
to finish tile `t-1` before it can publish. This predecessor edge delays
`BatchDsFilled0`, which is the writer's larger measured wait. A prior two-page
attempt reused one phase/token pair and timed out beyond S128 because one
ABarrier cannot represent two live physical generations.

The released resident-V interval has enough non-overlapping space after the
8KB P scratch and 1.5KB sidecar for one additional 16KB group0 dS page.
H3 adds exactly that page and a separate Filled/Done token pair:

```text
consumer0(t): compute dS -> wait Done[page=t%2] only when t>=2
              publish dS[page] -> Filled[page] -> dK -> Done[page]
dQ writer(t): wait Filled[page] -> dQ group0 -> Done[page]
```

Group1 remains single-page. The DAG, five-GEMM/MMAC count, output ownership,
raw Q/dO double buffer, and matrix instruction path remain unchanged. Planned
physical LDS remains 128KB because the new page overlays already released V.
The hypothesis is admitted only if S384 and S1024 complete; this is the direct
regression test for the old shared-token deadlock.

Result: `REJECT_TICKS_AND_CONTROL_COST_CANONICAL_RESTORED`.

- Separate physical pages/tokens fixed the old lifecycle failure: S128, S384,
  and all S1024 runs passed correctness with no spill/scratch and bank0.
- Candidate-only S1024 median was `46,832,695`, but an interleaved saved-binary
  A/B reversed the apparent win: canonical median `47,027,890`, H3 median
  `47,405,995` (`+0.80%`). Candidate fullperf was also about `1.0%` slower.
- Dynamic SCA rose `58,368 -> 72,208`; MMAC active fell from roughly
  `33.63%` to `33.18-33.25%`. XCU ABarrier share improved only
  `27.66 -> 27.25%`, while trans-read wait rose `11.01 -> 11.22%`.

The independent-token lifetime is a verified probe fact, but the runtime
page/token selection costs more than the removed predecessor edge. Do not
promote the extra page into the canonical kernel.
