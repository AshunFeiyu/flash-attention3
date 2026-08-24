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

## H4: dK Owned By The dQ Writer

The canonical role ledger is structurally imbalanced on every SIMD:

```text
producer / consumer0 / consumer1 / dQ-writer = 0 / 128 / 128 / 64 MMAC
```

H4 moves the complete `dK=dS^T@Q` logical GEMM from the eight N16 dKV
consumer waves to the four D32 dQ-writer waves. It does not split or repeat a
logical GEMM. The new exact-work ledger is:

```text
dKV consumer: score + dP + dV = 96 MMAC/wave
dQ writer:    dQ + dK         = 128 MMAC/wave
total:        8*96 + 4*128    = 1280 MMAC/tile
```

The writer already consumes every native dS page and owns one D32 slice. It
can read the same dS page through transposed view for dQ and normal view for
dK, then read only its D32 Q fragment from the raw page. This creates one
`matrix-read -> wait -> 16 MMAC` group island: eight dQ MMAC plus eight dK
MMAC, with no scalar matrix read, gather, permute, atomic, or wrong layout.

Ownership also becomes transitive. A writer can observe `BatchDsFilled` only
after all four matching consumers have finished their raw reads and published
dS. Therefore the four writers alone close both reuse edges:

```text
BatchDsFilled(group, count=4) -> writer dQ+dK
writer arrives DqDone(group, count=4)
writer arrives RawUsed(page, count=4) after both groups
```

Consumers no longer arrive at `RawUsed` or `DqDone`. They wait on the matching
four-writer `DqDone` only before overwriting the single group-local dS page.
The physical LDS map and ten barrier IDs remain unchanged. The tentative WDRA
ledger is `16/144/144/208=512`; compilation must prove the role-local use
rather than relying on this estimate.

Admission order is static exact-work/resource gates, S128 correctness,
S1024 correctness, repeated same-binary A/B stats, then fullperf/XCU. Any
spill, bank conflict, duplicate GEMM, layout workaround, or incorrect
transitive reuse edge rejects H4 immediately.

Result: `REJECT_LAYOUT_INTEGRATION_CANONICAL_RESTORED`.

- The rebalance compiled as exact five-GEMM work with role usage
  `9/129/155/131`, metadata `VGPR=128, SGPR=81`, no private segment,
  spill, scratch, or LDS bank conflict.
- H1/S128 causal kept dV and dQ correct, but dK failed with
  `rel_l2=0.523375` and `cosine_error=0.132246`.
- Replacing writer-side Q D32 base-address shifts with a fixed panel base and
  compile-time matrix-read immediates emitted the intended
  `offset:0x800/0x1000/0x1800` instructions, but produced bit-identical dK
  errors. Splitting dQ/dK stages, changing MMAC wrappers, and keeping the dK
  accumulator live also left the error unchanged.
- Zeroing dK immediately before its final store produced exact zero output,
  and block correlation kept every D32 block mapped to itself. The final
  store owner is therefore sound; the unresolved contract is the writer-role
  Q fragment as the right operand of `dS^T @ Q`.

H4 is closed at the integration layer. Do not add another layout workaround
to the canonical kernel. A future retry requires a focused dense-MMAC oracle
for the exact tuple `raw-Q MLS writer -> normal D32 reader in dQ-writer role
-> dS-normal lhs -> dK output`, including lane-level fragment evidence.

## H5: Native dK/dV Matrix-Store Epilogue

User-provided ISA rule: `mmac_4interleave` keeps the C-fragment layout of its
left matrix. Therefore the production experiment preserves that layout
directly instead of searching for a generic FP32-C conversion:

```text
dS^T/P^T left-layout FP32 accumulators
  -> pair adjacent D16 accumulators in source-slot order
  -> cvt_pk FP16 concat
  -> ds_write_matrix_format_f16(trans)
  -> released LDS page
  -> matrix_store_32x16_b16(T=1,R=0,stride=128 elements)
```

Each of the eight dKV waves still owns one N16 row tile and all D128 columns.
The four D32 dK tiles and four D32 dV tiles are unique; GEMM work and output
ownership do not change. A CTA-wide ownership barrier after all q-loop work
releases resident K/V, raw Q/dO, sidecar, and final dS pages before the
epilogue reuses V-LDS. Each wave uses two private 1KB writer pages and issues
one dK plus one dV matrix store per D32 block, then reuses the pages.

This branch intentionally changes fused5 dK/dV output storage to FP16, which
matches the requested matrix-store contract; the harness converts it to FP32
only for comparison with the existing CPU golden. Admission is S128 full
correctness, no spill/private/scratch, bank0, exact matrix-store ISA, then an
S1024 same-environment tick comparison. Failure restores the canonical direct
FP32 global-store path rather than adding a layout workaround.

Result: `REJECT_CURRENT_N16_OUTPUT_OWNERSHIP`.

- The experiment compiled with `SGPR=62`, `VGPR=128`, no private segment or
  spill, and ran without PMD panic or LDS bank conflict.
- Both 32x16-per-wave and paired-wave 32x32 matrix-store epilogues wrote all
  elements, but H1/S128 dK/dV correctness failed. Delta and dQ stayed correct.
- A dumped actual-to-golden map proved that values were preserved but emitted
  in a fixed source-slot swizzle. With the trans writer, rows were correct while
  columns followed `0,16,4,20,...`; store T/R flags did not change the map.
- This does not reject the `mmac_4interleave` left-layout rule. It rejects the
  assumption that two waves, each owning N16xD32, can compose the same native
  writer source contract as one wave owning a complete N32xD32 C tile.
- The next legal production experiment must give one store owner the complete
  N32xD32 result tile. Do not add lane gather, permute, or another pack sweep to
  the current N16 ownership path.
- After removing the rejected epilogue, the restored canonical FP32-store
  source rebuilt at SGPR60/VGPR128 with no spill/private segment. H1/S128 full
  backward passed delta/dK/dV/dQ with bank0; fused5 ticks were 11,380,005 and
  total three-dispatch ticks were 14,838,005 on the same locked environment.

## H6: FP16-Output dK/dV Native Matrix Store

The H5 ownership attribution was too broad. Two N16 waves can cooperatively
publish one N32 page; the exact two-wave probe had already proved that. The
failed production tuple was specifically FP32 C followed by lane-local FP16
packing. H6 keeps the existing N16 ownership and uses the native exact tuple:

```text
FP16-output MMAC lit0/lts0
  -> adjacent D16 concat
  -> trans ds_write_matrix, one N16 half per wave
  -> matrix_store_32x32_b16 by the even wave
```

Result: `ACCEPT_NATIVE_FP16_DKV_MATRIX_STORE`.

- H1/S128 and H1/S1024 full backward pass delta/dK/dV/dQ with bank0.
- S1024 dK/dV relative L2 errors are 0.00904349 and 0.00169241.
- Metadata is SGPR62/VGPR128 with no private segment or spill. Dynamic MMOP
  remains 92,160; dK/dV direct FLAT stores are gone.
- S128 fused ticks improve 11,380,005 -> 10,338,055 (-9.16%).
- S1024 stats ticks improve 47,577,985 -> 46,465,055 (-2.34%).
- S1024 fullperf ticks improve 47,024,250 -> 46,384,975 (-1.36%), while
  MMAC active improves 33.578961% -> 34.593226%.
- XCU still identifies RawUsed ABarrier as the dominant issue-gap (28.35%).
  The matrix-store tail is not the new critical path.
