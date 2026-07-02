# Client

## Mission

Build and optimize a clean Shaobo FA3 BWD dKV WASP kernel from scratch.  The
primary optimization target is MMAC active share, using same-run FA3 forward as
the benchmark.  dQ is frozen.

## Code Plan

Keep the repo small and modular:

- `include/dkv_contract.h`: tile, role, resource, and target constants.
- `src/dkv_kernel.cpp`: one cohesive FA3 BWD dKV implementation.
- `scripts/`: build/run/check helpers only.
- `docs/design_contract.md`: algorithm, resources, expected pipeline, gates.

Do not port the old phase stack.  Port only proven pieces, one block at a time:

1. producer A/B packet publishers
2. consumer score+dP MMAC island
3. softmax+dS VALU island
4. dV+dK MMAC island
5. store epilogue

## Design Rules

- First eliminate duplicate score/dP and wrong output ownership.
- Main matrix path must be MLS/BPS + `ds_read_matrix` + `v_mmac_*lit`.
- `Q^T` and `dO^T` come from source-layout ABI, not LDS raw-to-trans scatter.
- Producer MLS/BPS publication should not be followed by local
  `wait_lgkm(0)`; ABarrier is the ownership fence to consumer
  `ds_read_matrix`.  Keep waits near first use or true overwrite/reuse points.
- Current W12 LDS plan is already full 128 KB:
  `Q 16KB + dO 16KB + K 32KB + V 32KB + Q^T 16KB + dO^T 16KB`.
  Do not add a dedicated LDS sidecar/scratch page unless another page/lifetime
  is removed or reused.  Source-layout operands must come from MLS/BPS pages
  already in this budget, or a workbook row must show which existing bytes are
  freed before code changes.
- Consumer work should be balanced: score, dP, dV, dK are each 16 MMAC per
  consumer wave per q tile.
- Producers must have recurring work after K/V startup; avoid thin producer
  designs unless perf proves they win.

## Optimization Loop

1. Update the shared workbook before major code changes.
2. Implement one hypothesis per commit.
3. Run static gates: clean repo, build metadata, no spill/scratch, no bank
   conflict, expected instructions.
4. Run correctness on `H1/S1024,D128`.
5. Capture perf only after correctness/resource gates pass.
6. Analyze SQTT with `xcu` CLI; GUI Wavefronts is only a fallback.
7. Judge by MMAC active share first, then dKV ticks, wait/barrier, coissue,
   SQTT pipeline/SIMD evidence, and any GUI observations.
8. Commit accepted changes with evidence; remove or document rejected code.

## SQTT Evidence

Use `scripts/xcu_preflight.sh` before relying on `.perf` evidence.  It should
find `xcu` or unpack the sidecar package into `${SHAOBO_RUN_ROOT}`.

Use `scripts/analyze_sqtt_perf.sh` for every promoted perf run:

```bash
scripts/analyze_sqtt_perf.sh --perf case.perf --dispatch 1
scripts/analyze_sqtt_perf.sh --perf case.perf --dispatch 1 \
  --time-range 4148:10828 \
  --location xcd=0,se=0,cu=6,simd=1,wave=1
```

Required evidence files live outside the repo: `detail.txt`,
`wavefronts_bubbles.txt`, pipeline CSV, SIMD CSV, and `manifest.md`.
Conclusions must cite these files when explaining MMAC active share, bubbles,
waits, VALU/MMAC overlap, or SIMD imbalance.

Current FWD/BWD reference analysis:

```text
docs/sqtt_fwd_bwd_gap_20260701.md
```

The key lesson is that BWD already has MMAC and `ds_read_matrix`; the gap is
barrier/control serialization and matrix-read latency not hidden under peer
work.  Optimize the packet conveyor before chasing more MMAC instructions.

## Artifact Rules

Keep generated files out of git:

- no `m5out*`
- no `.perf`
- no full PMD logs
- no temporary CSV exports
- no `xcu` SQTT output directories

Use `${SHAOBO_RUN_ROOT}` for runs.  On liuchang:

```bash
export SHAOBO_RUN_ROOT=/zys/shaobo_runs/fa3_bwd_wasp_clean
```

Shared review artifacts go under:

```text
/Volumes/172.20.68.76/共享/shaobo
/Volumes/172.20.68.76/共享/shaobo/perf
```

Current FA3 BWD clean design workbook:

```text
/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx
```

2026-07-02 FWD-style dKV goal update:

- Added sheets `FWD目标`, `算法DAG`, `资源预算`, `流水设计`, `指标门禁`,
  and `实验记录`.
- The workbook now records the hard target: full dK/dV WASP path, no duplicate
  score/dP, no wrong output ownership, no spill/scratch, `ldsBankConflict=0`,
  main matrix path through MLS/BPS + `ds_read_matrix` + MMAC, and MMAC active
  share `>=60%` on the steady diagnostic shape.
- The expected pipeline is written as `T0..T5`: producer packet publication,
  score/dP MMAC, peer softmax/dS VALU overlap, dV/dK MMAC, and store epilogue.

## Current Next Step

Current promotion baseline:

- The best clean W12 evidence remains the zero-seed/read4x2 line, with
  H1/S1024 causal `kernel_ticks=70604625` and MMAC active avg `21.7988%`.
- Mq64 seed-fix and raw-page sidecar overlay are correctness/resource-clean
  diagnostics, not promoted performance baselines:
  - Mq64 seed-fix: `kernel_ticks=79595880`, active `19.8279%`.
  - raw-page sidecar overlay: `kernel_ticks=73113950`, active `18.9185%`.
  - score/dP read2x brick: `kernel_ticks=70801640`, active `21.5465%`.
  - Mq64 semantic-page conveyor: same-build `kernel_ticks=73320065`,
    active `21.7509%`, versus W12 baseline `kernel_ticks=70974995`.
  - causal whole-tile skip: `kernel_ticks=72881900`, active `16.7128%`,
    versus same-build W12 baseline `kernel_ticks=71006845`, active
    `21.6777%`; it removes MMOP but worsens tail/SIMD balance on H1/S1024.
  - mixed score/dP brick: same-build `kernel_ticks=71663865`, active
    `21.1732%`, versus W12 baseline `kernel_ticks=71312605`, active
    `21.5931%`; it raises coissue but lowers active share and regresses ticks.
- The next FWD-style redesign must raise MMAC active share by reducing
  ABarrier/control and exposed matrix-read/sidecar latency.  More coissue
  count is not enough unless active share and same-shape ticks move with it.
  The semantic-page negative result specifically says not to add another
  raw/source ABarrier generation unless it replaces an existing one.
  The causal-skip negative result additionally says not to judge by MMOP count
  alone: deleting upper-triangle work can make the conveyor thinner and reduce
  MMAC active share on the diagnostic shape.
  The mixed-score negative result adds that local consumer schedule asymmetry
  is not enough; do not combine already-rejected score/dP bricks just to make
  Wavefronts look less synchronized.
  The dedicated-LDS-sidecar design was rejected at resource gate: current W12
  LDS is already 128KB, so sidecar-in-LDS must replace an existing page/lifetime
  rather than append another 768B page.
- The source-score layout probe was rejected at correctness gate:
  `Q^T/dO^T` source-layout pages are not raw `Q/dO` drop-in replacements for
  score/dP with the current `ds_read_matrix` mapping.  Keep raw `Q/dO` pages
  for score/dP unless a smaller instruction-layout probe proves a different
  mapping.
- The raw-dVdK layout probe was also rejected at correctness gate: raw `Q/dO`
  pages are not drop-in replacements for source-layout `dO^T/Q^T` operands in
  the current dV/dK `ds_read_matrix` mapping.
- The early RawUsed release path is correct/resource-clean and slightly
  improves same-build H1/S1024 ticks and mid-window xcu bubbles, but it remains
  a micro observation: full-perf MMAC active share is still about `21.74%`, far
  from the `>=60%` FWD-style target, and dispatch-level xcu still reports
  `abarrier -> salu_32` plus `flat_rd -> immed` as dominant gaps.
- Next structural target: reduce page ownership/control debt or sidecar/global
  read debt.  Do not stack more consumer-order micro patches unless the
  workbook first shows how they shorten producer waits or expose a longer MMAC
  island.

Current state has two paths:

Reference correctness path `PASS`:

- enabled by `params.dkv_path = kDkvPathReferenceCorrectness` or standalone
  `--check=1`
- computes `P`, `delta`, `dP`, then writes float `dK/dV`
- compares against host CPU golden in standalone
- latest PMD S128 evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_correctness_20260702_003625`

WASP stream q-loop probe `OBSERVE_PIPELINE`:

- producer0 publishes resident K once, then streams double-buffered Q raw pages
- producer1 publishes resident V once, then streams double-buffered dO raw pages
- two consumer groups wait resident K/V once, then stream 32 `Mq=32` pages
  through a score+dP `ds_read_matrix + v_mmac_*lit` probe
- standalone now allocates real Q/K/V/dO buffers before running PMD

WASP softmax/dS sidecar `PASS`:

- enabled by `params.dkv_path = kDkvPathWaspSoftmaxDsSidecar` or standalone
  `--probe-check=1`
- computes one scalar `(P,dS)` diagnostic per consumer wave inside the WASP
  consumer role
- compares against host CPU golden
- latest PMD S128 evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_sidecar_correctness_20260702_005025`

WASP fragment sidecar `PASS`:

- enabled by `params.dkv_path = kDkvPathWaspFragmentSidecar` or standalone
  `--fragment-check=1`
- producer A publishes row sidecar into double-buffered shared sidecar pages
  together with Q raw pages
- consumer score/dP now keeps four MMAC fragments and computes fragment-local
  `P/dS` from sidecar max/sum/delta
- first MMAC in each fragment uses the FWD-style `mmac_zeros` seed instead of
  pre-zeroing every accumulator
- latest PMD S128 evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_fragment_sidecar_correctness_20260702_012111`
  with `p_max_abs=6.34603e-06`, `ds_max_abs=2.66919e-08`, `pass=1`

Full dK/dV MMAC baseline `PASS_BUT_LOW_ACTIVE`:

- enabled by `params.dkv_path = kDkvPathWaspDkvMmac` or standalone
  `--dkv-mmac-check=1`
- computes score, dP, softmax/dS, dV MMAC, dK MMAC, and float stores
- no duplicate score/dP across D halves; each consumer wave owns `Nk=16,D=128`
- source-layout `Q^T/dO^T` comes from the host ABI, not LDS transpose
- latest H1/S1024 causal=true evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_022300`
  with `pass=1`, `kernel_ticks=83290480`, MMAC active avg `18.7606%`,
  coissue `36070/20048`, and `ldsBankConflict=0`
- rejected negative result:
  splitting raw and source ownership regressed to MMAC active `18.1123%` and
  `kernel_ticks=86912280`, so keep coarse page ownership for now

12-wave single-producer candidate `ACCEPT_CANDIDATE_BUT_LOW_ACTIVE`:

- enabled by `params.dkv_path = kDkvPathWaspDkvMmac12Wave` or standalone
  `--dkv-mmac12-check=1`
- smoke command:
  `SKIP_BUILD=1 GPU_CHIP=sb GPU_ARGS="['--SQCIPfLines=7']" B=1 H=1 S=1024 D=128 CAUSAL=1 WAVES=12 scripts/run_dkv_mmac_correctness.sh`
- latest H1/S1024 causal=true evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_024903`
  with `pass=1`, `kernel_ticks=78751400`, MMAC active avg `20.2578%`,
  coissue `23301/12740`, and `ldsBankConflict=0`
- improvement over 16-wave baseline:
  `kernel_ticks` down `5.45%`, MMAC active up from `18.7606%` to `20.2578%`
- perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_025324_clean_w12_dkv_mmac12_h1s1024_sqc7`
- xcu conclusion:
  MMAC is present but the pipeline is still dominated by
  `abarrier -> salu_32` bubble (`38.85%`) and large `abarrier -> immed` gaps
  around `15.8k` cycles.  Producer waves remain thin, just fewer than before.

Latest evidence archive:

```text
/Volumes/172.20.68.76/共享/shaobo/perf/20260702_025324_clean_w12_dkv_mmac12_h1s1024_sqc7
```

W12 sidecar-address micro baseline `ACCEPT_MICRO`:

- current source keeps W12 and hoists the global sidecar q-tile base out of
  the inner softmax/dS vector loop
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_031634`
- full-perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_031634_clean_w12_sidecar_addr_h1s1024_sqc7`
- same full-perf W12 comparison:
  `kernel_ticks` down from `78625365` to `75964525`;
  MMAC active up from `19.9522%` to `20.3523%`
- xcu conclusion:
  the top bubble is still `abarrier -> salu_32` (`38.33%`), and
  `flat_rd -> immed` did not materially fall, so this is a micro cleanup, not
  the FWD-style pipeline answer

Next step: keep W12 sidecar-address as the current clean baseline and redesign
the ABarrier/control protocol toward the FA3 FWD pattern.  The target is still
MMAC active `>=60%`; the next implementation must create longer continuous
MMAC islands and reduce ownership/control bubbles, not just improve local
address arithmetic.

Late-source conveyor negative:

- tried to reuse raw pages for source-layout `Q^T/dO^T` after score/dP, so
  producer MLS could overlap consumer softmax/dS
- H1/S128 and H1/S1024 correctness both PASS, no spill, no LDS conflict
- H1/S1024 regressed to `kernel_ticks=81165175`, MMAC active `19.1939%`
  versus W12 sidecar baseline `75964525` and `20.3523%`
- decision: `REJECT_PERF`; code reverted, lesson retained in workbook/log

Next design rule: do not add another per-page raw/source ownership epoch unless
the workbook proves that the extra wait is hidden by a larger useful compute
island.  Producer usefulness is necessary, but not sufficient.

Producer early-exit negative:

- tried to remove the long thin producer tail by letting producer waves return
  after `producer_all_loop`
- producer VGPR `80` failed WDRA branch-average granularity; producer VGPR
  `76` passed static metadata
- H1/S128 PMD then aborted with `vgpr81 is not init or has been freed` during
  MMAC
- decision: `REJECT_RUNTIME_PANIC`; code reverted

Rule: keep producer waves alive through the existing tail until a focused probe
proves an early-exit/cleanup ABI that PMD supports.

Full-valid softmax fast-path negative:

- tried to skip per-element causal/predicate checks in
  `softmax_ds_owner16_from_global_sidecar` when an owner16 tile is fully valid
- early-return and `if/else` variants both passed static metadata but failed
  H1/S128 causal correctness
- representative result:
  `dK rel_l2=0.000361379`, `dV rel_l2=14.7566`
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_035820`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_035954`
- decision: `REJECT_CORRECTNESS`; code reverted

Rule: do not optimize away owner16/global-sidecar mask work by local branching
unless a focused sidecar probe proves the `P`/dV fragment mapping first.

W12 dV/dK read-all micro baseline `ACCEPT_MICRO`:

- current source batches all eight source-layout `dO^T/Q^T` operand fragments
  for the owner16 dV/dK island, then does one `wait_lgkm(0)` before the longer
  MMAC island
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_041545`
- full-perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_041545_clean_w12_dvdk_readall_h1s1024_sqc7`
- same-shape comparison against W12 sidecar-address:
  `kernel_ticks` down from `75964525` to `72499700`; MMAC active avg up from
  `20.3523%` to `21.3054%`; `ldsBankConflict=0`
- xcu conclusion:
  `lds_matrix -> immed` falls from `9.38%` to `5.53%`, so the local read-use
  gap improved; top bubble remains `abarrier -> salu_32` at `38.64%`, so this
  is not the 60% MMAC active solution

Next step: keep W12 dV/dK read-all as the current clean baseline.  The next
structural change should be workbook-first and focus on the ABarrier/control
ledger and producer/consumer phase alignment, because local dV/dK read
batching is now proven but insufficient.

Pre-softmax dV/dK read negative:

- tried to issue all eight `dO^T/Q^T` source operand reads before softmax/dS,
  then wait only before dV/dK MMAC
- build passed but metadata failed:
  `private_segment_fixed_size=24`, `vgpr_spill_count=10`
- decision: `REJECT_RESOURCE`; code reverted

Rule: read-early/wait-late is still the right latency-hiding pattern, but this
kernel cannot keep all source operands live across softmax/dS.  Future retries
must use a smaller 4+4 source grouping or reduce softmax live range first.

W12 dV/dK read4x2 micro baseline `ACCEPT_MICRO`:

- current source keeps only the low half of `dO^T/Q^T` source operands live
  across softmax/dS, then issues the high half before the low dV/dK MMAC group
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_043641`
- full-perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_043641_clean_w12_dvdk_read4x2_h1s1024_sqc7`
- same-shape comparison against read-all:
  `kernel_ticks` down from `72499700` to `71508255`; MMAC active avg up from
  `21.3054%` to `21.5678%`; `lds_matrix -> immed` down from `5.53%` to `2.46%`
- xcu conclusion:
  matrix-read latency is now much less exposed; top bubble remains
  `abarrier -> salu_32` at `39.00%`

Rejected consumer turnstile retry:

- hypothesis: add a FWD-like `ValuExec0` token so consumer1 reaches softmax
  first and consumer0 waits before softmax, hoping to create a useful VALU/MMAC
  phase offset
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_044441`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_044502`
- result: H1/S128 and H1/S1024 correctness pass, metadata clean, but
  `kernel_ticks` regressed from `71508255` to `73908835`, MMAC active avg fell
  from `21.5678%` to `20.8817%`, and coissue also fell
- decision: `REJECT_PERF`; code reverted

Next step: do not keep polishing dV/dK read granularity and do not add pure
turnstile waits.  The next workbook proposal must reduce ABarrier/control
serialization through useful independent work, fewer ownership turns, or a
different role split that preserves no-duplicate score/dP.

Rejected raw/source ownership split:

- current over-sync: one raw packet token covers both raw `Q/dO` and
  source-layout `Q^T/dO^T`, so consumers cannot start score/dP until all four
  MLS groups are published
- candidate tried: publish raw `Q/dO` first and release `RawFilled`; publish
  `Q^T/dO^T` under a separate `SourceFilled` token while consumers do
  score/dP; consumers release `RawUsed` immediately after score/dP and
  `SourceUsed` after dV/dK
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_045658`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_045719`
- result: correctness PASS and no spill, but H1/S1024 `kernel_ticks`
  regressed from `71508255` to `75607805`, MMAC active avg fell from
  `21.5678%` to `20.5505%`, and failed coissue rose from `20971` to `23826`
- decision: `REJECT_PERF`; code reverted

Rule: splitting raw/source ownership is architecturally cleaner, but on this
W12 topology it adds more ABarrier/SCA cost than it hides.  Do not add more
tokens unless the design removes another ownership turn or moves substantial
producer work into a proven critical window.

Consumer-group template cleanup `ACCEPT_MICRO`:

- change: make `consumer_dkv_mmac_loop` a `ConsumerGroup` template and call
  `<0>` / `<1>` from the two consumer branches
- reason: align with FWD-style branch-local specialization and remove a small
  runtime `consumer_group` address/control dependency
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_050701`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_050727`
- result: H1/S1024 `kernel_ticks` improved from `71508255` to `71412705`;
  MMAC active avg is essentially flat, `21.5678%` to `21.5708%`
- decision: keep as a micro codegen/style cleanup, not a pipeline solution

dV/dK zero-seed cleanup `ACCEPT_MICRO`:

- change: only call `zero_f16x8` inside dV/dK MMAC helpers when
  `FirstQTile=true`; for later q tiles the accumulator is already live and the
  zero operand is unused
- reason: this directly removes volatile `v_mov` zeroing before dV/dK MMAC,
  matching the FWD `mmac_zeros` lesson
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_051325`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_051343`
- full-perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_051513_clean_w12_zero_seed_h1s1024_sqc7`
- result: stats-only H1/S1024 `kernel_ticks` improved from `71412705` to
  `70604625`; MMAC active avg rose from `21.5708%` to `21.7988%`
- xcu detail: `valu_32` hits dropped to `151648` and MMAC latency share rose
  to `23.68%`; top bubbles remain `abarrier -> salu_32` and
  `flat_rd -> immed`
- decision: keep; this is a real cleanup but still far from the 60% active goal

Sidecar prefetch attempt `REJECT_PERF_STATS_ONLY`:

- tried to prefetch all 8 q-row sidecar triplets into consumer registers before
  `RawFilled`/score-dP to hide the xcu `flat_rd -> immed` bubble
- correctness and metadata passed, but consumer branch pressure reached
  `158/160`; H1/S1024 regressed to `kernel_ticks=75394410`, MMAC active avg
  `18.4182%`
- archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_052900_clean_w12_sidecar_prefetch_reject_h1s1024_sqc7`
- decision: code reverted; do not repeat "prefetch all sidecar fields into
  long-lived registers" without a new resource proof

Next step: sidecar/global-read latency remains a target, but the next design
must either shrink sidecar live state or move useful producer/control work
without consuming the last consumer VGPR slack.  A broader path is to revisit
the algorithm/resource workbook for a sidecar representation that avoids both
global-use-point latency and 24-float live ranges.

Noncausal diagnostic boundary:

- after rebuilding the reverted zero-seed baseline, `CAUSAL=0` H1/S128 passed
  correctness, but H1/S1024 failed numerical comparison
- failing run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_053811`
- decision: do not use `CAUSAL=0` H1/S1024 as a perf diagnosis path until its
  correctness/tolerance is resolved; keep the active mainline on `causal=true`

Sidecar pair-prefetch boundary:

- a smaller two-row sidecar batching attempt stayed resource clean
  (`private=0`, `sgpr_count=82`, `vgpr_count=112`, consumer branch `144/160`)
  but failed H1/S128 correctness before stats/perf
- failing run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_055216`
- signal:
  `dk_rel_l2=8244.1`, `dv_rel_l2=30.6025`, `pass=0`, with PMD
  `read vgpr111 before writing`
- decision: `REJECT_CORRECTNESS`; code reverted

Rule: do not continue sidecar batching in full dKV without a focused
sidecar-fragment correctness probe.  The active code remains the zero-seed
baseline while the next MMAC-active push should look at structural ABarrier /
producer-thinness / FWD-style pipeline gaps rather than more sidecar preload.

Mq64 single-buffer boundary:

- tried a structural `Mq=64` W12 path to double the consumer MMAC island and
  halve q-loop control/barrier turns
- unspecialized forms spilled; the final S1024/causal specialization became
  resource-clean with `private=0`, `sgpr=100`, `vgpr=144`, no spill, and
  consumer branch `171/208`
- H1/S1024 correctness failed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_062758`
- failure signal:
  `dk_rel_l2=5680.1`, `dv_rel_l2=20.0452`, `pass=0`, PMD
  `read vgpr268 before writing`
- decision: `REJECT_CORRECTNESS`; full implementation is not a perf candidate

Rule: a future Mq64 attempt must start as a focused layout/correctness probe,
not as another full-kernel integration.  Prove score/dP, sidecar row mapping,
source-layout `Q^T/dO^T`, dV/dK accumulation, and store ownership first.

Post-revert baseline check:

- remote build/static gate PASS after removing the Mq64 implementation
- W12 metadata PASS with `private=0`, `sgpr=84`, `vgpr=112`, no spill
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_063709`

Mq64 seed-fix diagnostic:

- workbook-first row: `Mq64 seed-fix reattempt`
- fixed the specific high D-block accumulator bug from the rejected Mq64 diff:
  `dv_acc[4..7]`/`dk_acc[4..7]` now seed on the first q tile instead of using
  an unconditional accumulate path
- H1/S1024 causal correctness now PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_064930`
- resource gate clean:
  `private=0`, `sgpr=100`, `vgpr=144`, no spill, consumer branch `171/208`
- performance is not promoted:
  `kernel_ticks=79595880`, MMAC active avg `19.8279%`, worse than the
  zero-seed W12 baseline `70604625` and `21.7988%`
- decision: `OBSERVE_CORRECTNESS_REJECT_PERF`; keep only as an opt-in
  diagnostic/future Mq64 basis

Rule: a larger MMAC island by itself is not enough.  The exact-128KB
single-buffer Mq64 topology fixes redundant q-loop control but appears to lose
too much producer/consumer overlap.  The next FWD-style design should preserve
the high-D seed lesson while recovering LDS slack or double-buffered overlap.

Raw/source layout swap boundary:

- source-score probe failed H1/S128 correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_092729`
- raw-dVdK probe failed H1/S128 correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_094838`
- the two probes together close the direct 32KB LDS freeing route: raw
  `Q/dO` pages are required for score/dP, and source-layout `Q^T/dO^T` pages
  are required for dV/dK under the current operand mapping
- do not reintroduce full-kernel raw/source swaps without a smaller
  instruction-layout proof that names the exact `matrix_load` and
  `ds_read_matrix` pairing
