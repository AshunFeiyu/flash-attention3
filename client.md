# Client

## Mission

Build and optimize a clean Shaobo FA3 BWD dKV WASP kernel from scratch.  The
primary optimization target is MMAC active share, using same-run FA3 forward as
the benchmark.  dQ is frozen.

## Current Evidence

- Best active route is the W12 canonical kernel with accepted
  H22/H23/H27/H28/H30 cleanups:
  `fa3_bwd_dkv_kernel`, `Mq=32,Nk=128,D=128`, one producer group and
  two consumer groups.  Reference correctness remains available through
  `--check=1`; rejected historical kernels are no longer public launch routes.
- Latest accepted H30 H1/S1024 full-perf stats:
  `kernel_ticks=66321255`, `MMOP=131072`, `ldsBankConflict=0`,
  `MMAC active share=28.3952%`.
- H30 shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_063614_clean_w12_h30_future_sidecar_prefetch_h1s1024_sqc7_fullperf`.
- xcu top1000 ABarrier audit shows producer-side `RawUsed` wait dominates:
  `RawUsed0 id=2` 1.629M cycles, `RawUsed1 id=4` 1.669M cycles, no
  `RawFilled` in top1000.  Producer is not starving consumers; consumers hold
  pages too long.
- H18A packed-sidecar `dwordx4` probe is rejected:
  correctness/resource clean and asm changed from `global_load_dwordx3` to
  `global_load_dwordx4`, but H1/S1024 regressed slightly to
  `kernel_ticks=70656495` and VALU cycles were unchanged.
- H19A pre-read-all-source-before-softmax is rejected at resource gate:
  build succeeds, but metadata reports `private_segment_fixed_size=24` and
  `vgpr_spill_count=10` because low+high `dO^T/Q^T` source fragments stay live
  across softmax/dS inside the 160 VGPR consumer window.
- H19B proves a 208 VGPR consumer window can make that schedule resource-clean
  and correct (`branch_consumer=164/208`), but H1/S1024 regresses to
  `kernel_ticks=77708085` versus restored canonical `70444920`.  Do not widen
  VGPR only to release `RawUsed` earlier.
- H20A owner16 full-valid softmax split is rejected at correctness gate:
  static/resource clean, but H1/S128 dV fails with `dv_rel_l2=14.4712`, matching
  the older full-valid fastpath negative.  Do not retry this helper split in the
  main route without a focused owner16 fragment/codegen probe.
- Next high-value work should reduce consumer page lifetime by removing or
  hiding duplicated sidecar/mask/softmax work, or by changing page ownership;
  do not spend time making producers faster until xcu shows `RawFilled` wait.
- H27 proves early `RawUsed` release after high-source read issue is safe and
  slightly useful, but not structural: xcu still shows RawUsed/ABarrier and
  sidecar global-load waits as the main debt.  Do not stack more tiny local
  reorder patches unless the workbook explains how they shorten those waits or
  extend a useful MMAC island.
- H28 producer-side sidecar cache-warm is accepted: it raises MMAC active into
  the high-20% band and xcu shows RawUsed/sidecar waits falling.  It also adds
  a producer `flat_load_dword -> s_waitcnt` bubble, so the next step should
  hide/batch that prefetch or convert it into a cleaner producer-helper
  protocol without adding LDS bytes or another ownership generation.
- H29 fire-and-forget sidecar prefetch is rejected for promotion: full perf
  raises MMAC active only from `27.9004%` to `27.9272%` while regressing
  `kernel_ticks` from `66630200` to `66690260`.  The active code is restored to
  the H28 explicit sidecar prefetch form.
- H30 future sidecar prefetch placement is accepted and is the current best:
  moving the prefetch to `q_tile+2` after `RawFilled` gives
  `kernel_ticks=66321255`, `MMAC active=28.3952%`.  xcu shows RawUsed bubble
  improves to `25.89%` and sidecar wait to `11.38%`, while producer prefetch
  wait grows to `1.79%`.
- After single-kernel convergence and archived-kernel cleanup, the current
  canonical full-perf rebaseline is clean and essentially matches H30 when
  both are normalized with the same stats formula:
  `kernel_ticks=66411800`, `MMAC active=23.4288%`,
  `VOP active=21.3239%`, `coissue=23057/18211`, `ldsBankConflict=0`.
  H30 normalized by `sum(mmopRunTimeCounter)/sum(activeTimeCounter)` is
  `23.4386%`; the older `28.3952%` row used a different/non-normalized active
  metric and should not be compared directly.
  Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_093932_clean_canonical_after_convergence_h1s1024_sqc7_fullperf`.
  Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `28_current_xcu_rebaseline`.
- Current xcu top bubbles are
  `s_abarrier_try_wait -> s_xor_b32` `25.95%`,
  `global_load_dwordx3 -> s_waitcnt` `11.28%`,
  `v_mmac -> s_waitcnt` `6.35%`, and
  `ds_read_matrix -> s_waitcnt` only `2.38%`.  The next performance edit
  should attack raw-page or sidecar exposure; batching `ds_read_matrix` alone
  is not a credible path to `60%`.

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
- The accepted H27 early RawUsed release path is correct/resource-clean and
  slightly improves same-shape H1/S1024 full-perf metrics:
  `kernel_ticks=66892280`, `MMAC active share=23.3787%`, versus H23
  `kernel_ticks=67246725`, `MMAC active share=23.2228%`.  It remains a micro
  observation, not a path to `>=60%` by itself; dispatch-level xcu still
  reports `RawUsed/ABarrier` plus sidecar global-read wait as dominant gaps.
- H30 adds future producer-side sidecar cache-warm placement and is now the
  best accepted candidate:
  `kernel_ticks=66321255`, `MMAC active share=28.3952%`,
  `ldsBankConflict=0`.  xcu explains the win: RawUsed
  `s_abarrier_try_wait -> s_xor_b32` drops to `25.89%`, sidecar
  `global_load_dwordx3 -> s_waitcnt` drops to `11.38%`, but producer prefetch
  `flat_load_dword -> s_waitcnt` rises to `1.79%`.
- The post-convergence full-perf rebaseline is:
  `kernel_ticks=66411800`, `MMAC active share=23.4288%`,
  `ldsBankConflict=0`, with top xcu bubbles RawUsed/control `25.95%` and
  sidecar wait `11.28%`.  With the same stats formula, H30 is `23.4386%`, so
  treat the current source as equivalent to H30 rather than a real regression.
- The sidecar lane-broadcast idea was rejected even though correctness and
  resource gates passed: `lane_n==0 + __shfl` regressed H1/S1024 ticks from
  `71209320` to `86765770` and MMAC active from `21.5636%` to `15.8550%`.
  Do not use wave shuffle/bpermute as the main sidecar latency fix.
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

Current active route, 2026-07-03:

- canonical kernel: `fa3_bwd_dkv_mmac12_kernel`
- active micro change: consumer groups use early RawUsed release
  (`consumer_dkv_mmac_loop<Tile, Bar, group, true>`)
- latest accepted H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_235606`
  with `kernel_ticks=70658770`, MMAC active avg `21.8783%`,
  `coissue=31248/20588`, `ldsBankConflict=0`
- unresolved main bottleneck: xcu still reports
  `s_abarrier_try_wait -> s_xor_b32` RawUsed bubble at about `28.66%`
- rejected main-bottleneck retry: splitting raw/source ABarrier token families
  passed correctness but regressed H1/S1024 to `kernel_ticks=75855325` and
  MMAC active avg `21.0489%`; the code was removed
- next work should reduce token turns or create useful producer/consumer work
  that hides RawUsed; do not add another source/raw page-generation token
  without workbook reasoning and xcu proof

Current top-down redesign note, 2026-07-03:

- shared workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- repo ledger:
  `results/tile_ledger_20260703.md`
- current mainline tile:
  `Mq=32,Nk=128,D=128,W12`
- per consumer wave per q tile:
  `score=16`, `dP=16`, `dV=16`, `dK=16`, total `64` MMAC
- `S1024` dispatch total:
  `8 K CTAs * 32 q tiles * 512 CTA-MMAC/q = 131072` MMAC
- LDS is exactly 128KB, so any sidecar/scratch or larger raw/source plan must
  replace an existing lifetime instead of appending bytes
- design hypothesis for the next mainline change:
  improve MMAC active by increasing effective MMAC-island length or hiding
  RawUsed, not by adding more ABarrier token families or duplicating score/dP

W16 WG-local semantic negative, 2026-07-03:

- workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `17_wg_local_nk64_design`
- legal LDS version must use semantic pages:
  naive private raw double pages plus private source double pages would be
  `192KB`; the tested version used `K/V64 32KB + two 16KB semantic pages`
  per WG, exactly `128KB` for two WGs
- implementation was resource/correctness-clean on the existing W16
  `fa3_bwd_dkv_mmac_kernel` route:
  `private=0`, `sgpr=86`, `vgpr=88`, no spill, H1/S128 and H1/S1024 pass
- H1/S1024 result:
  `kernel_ticks=80790710`, `MMOP=131072`, `ldsBankConflict=0`,
  MMAC active `19.1856%`, worse than W12 canonical `~21.8783%`
- decision: `REJECT_PERF_STATS_ONLY`; experiment code was removed from the
  live route after logging

Rule: do not assume FWD-style two-WG topology automatically fixes BWD.  If the
price is duplicated Q/dO/source loads plus raw/source semantic epochs, the
extra producer independence can lose to source-epoch control cost.  The next
60% attempt should preserve shared double-buffering or create a longer
consumer MMAC island without source-epoch serialization.

H21A q-pair control-only boundary:

- design workbook sheet `19_qpair_design` now records the q-pair/Mq64-equivalent
  idea using the existing W12 page0/page1 LDS pages
- direct implementation was rejected before PMD:
  helper form gave `sgpr_spill_count=39`; local macro form still gave
  `sgpr_spill_count=38`
- after revert, canonical W12 metadata is clean again:
  `private=0`, `sgpr=84`, `vgpr=112`, no spill/scratch
- lesson: do not use a broad `q_tile += 2` body duplication as the next 60%
  active-share route; it expands consumer SGPR live ranges without changing the
  real operand lifetime problem

Next design direction:

- keep the workbook-first H21 sheet as the design anchor
- before trying H21B stagger, shrink the consumer body or split address/sidecar
  state so the compiler does not spill SGPRs
- alternatively choose a topology that lengthens the useful MMAC island without
  duplicating the complete q-tile control body in one branch

Current canonical baseline after H22:

- active source: `fa3_bwd_dkv_mmac12_kernel`
- accepted micro change: peel `q_tile=0` out of `consumer_dkv_mmac_loop`, then
  run the steady loop from `q_tile=1`
- resource gate: `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch
- H1/S1024 full perf:
  `kernel_ticks=67665325`, `MMOP=131072`, `ldsBankConflict=0`,
  `MMAC active share=23.0485%`, coissue `23064/18083`
- shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_042626_clean_w12_h22_first_tile_peel_h1s1024_sqc7_fullperf`
- current blocker to 60%:
  xcu still shows `s_abarrier_try_wait -> s_xor_b32` around `28.44%` and
  `global_load_dwordx3 -> s_waitcnt` around `11.59%`

Next implementation constraint:

- keep H22 as the clean canonical baseline
- attack exposed RawUsed/sidecar wait or increase useful MMAC-island length
- do not add source/raw token families or duplicate the whole q-tile body,
  because both already have evidence-backed negative results

Current canonical baseline after H23:

- active source: `fa3_bwd_dkv_mmac12_kernel`
- accepted micro changes:
  - H22 first-tile peel keeps the steady q-loop free of the runtime
    first-packet accumulator branch
  - H23 removes the fixed leading `s_nop 0` from
    `ds_read_matrix_trans_pair`
- resource gate: `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch
- H1/S1024 full perf:
  `kernel_ticks=67246725`, `MMOP=131072`, `ldsBankConflict=0`,
  `MMAC active share=23.2228%`, coissue `22768/18808`
- shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_044220_clean_w12_h23_no_dsread_snop_h1s1024_sqc7_fullperf`
- useful xcu delta:
  issue count drops `897096 -> 847944`, `ds_read_matrix` latency drops
  `557792 -> 475420`, and the `s_nop` row disappears
- current blocker to 60%:
  xcu still shows RawUsed/ABarrier around `28.48%` and sidecar
  `global_load_dwordx3 -> s_waitcnt` around `11.54%`

Next implementation constraint:

- keep H23 as the clean canonical baseline
- do not chase more read-scheduling micro-nops before reducing exposed
  ABarrier/sidecar waits or lengthening a useful MMAC island
- do not restore q-pair body duplication, raw/source token split, source
  preread-all, W16 WG-local semantic pages, or sidecar shuffle/packing routes;
  each already has negative evidence in the ledger

H24 raw ABarrier wait boundary:

- workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `20_abarrier_wait_design`
- H24A changed both raw Filled/Used waits from asm wrapper
  `abarrier_try_wait<true>` to builtin wrapper `abarrier_try_wait<false>`;
  static metadata failed with `private_segment_fixed_size=12`,
  `sgpr=80`, `vgpr=112`
- H24B changed only raw Used waits to builtin; static metadata still failed
  with `private_segment_fixed_size=12`, `sgpr=82`, `vgpr=112`
- source was restored to H23 and rechecked:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch
- lesson:
  do not try to solve the top `s_abarrier_try_wait -> s_xor_b32` bubble by
  a raw-wait builtin swap inside the active dKV q-loop.  The next ABarrier
  attempt must reduce page lifetime/turns or hide the wait with useful work.

H25 RawUsed lifecycle boundary:

- workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `21_rawused_release_design`
- tested order:
  `wait low -> read high -> wait high -> release RawUsed -> low dV/dK MMAC -> high dV/dK MMAC`
- result:
  correctness/resource clean, but H1/S1024 regressed to
  `kernel_ticks=68373305`, MMAC active `22.8899%`
  versus H23 `67246725`, `23.2228%`
- lesson:
  do not move high-source wait before the dV/dK MMAC island.  H23's current
  order is better because low dV/dK MMAC hides high source-read latency.
  Future RawUsed work must keep that hiding while giving producer a useful
  window, or attack sidecar/softmax work instead.

H26 causal sidecar boundary:

- workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `22_causal_sidecar_design`
- tested change:
  split a causal=true helper for `softmax_ds_owner16_from_global_sidecar` to
  remove the inner runtime `(!causal || ...)` predicate term
- result:
  correctness/resource clean, but H1/S1024 regressed to
  `kernel_ticks=70504980`, MMAC active `22.5343%`, `VALU=230108`
  versus H23 `67246725`, `23.2228%`, `VALU=213208`
- source state:
  H26 code reverted; restored metadata is `private=0`, `sgpr=78`,
  `vgpr=112`, no spill/scratch
- lesson:
  causal predicate specialization is not a useful route toward 60% in the
  current code shape.  Prioritize structural RawUsed/ABarrier lifetime,
  sidecar data path, or larger useful MMAC island work.

Physical code convergence:

- status: `CODE_GOVERNANCE_ACCEPT`
- live dKV performance route is now only `fa3_bwd_dkv_kernel`
- reference correctness kernels remain behind `kDkvPathReferenceCorrectness`
- removed from live source: archived `#if 0` kernels, old Wasp path constants,
  Mq64/semantic/sidecar-overlay/causal-skip helpers, and probe/fragment
  standalone scripts
- static gate now checks the active canonical route directly and forbids stale
  experiment route symbols in `src/include`
- remote verification:
  build PASS, dKV kernel gate PASS, symbol metadata PASS
  `private=0`, `sgpr=78`, `vgpr=112`, no spills
- H1/S128 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_111624`
- H1/S128 stats:
  `simTicks=17781855`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=17781855`, `MMOP=2048`, `ldsBankConflict=0`,
  `coissue=351/238`

Rule going forward:

- do not add another dKV performance path
- modify `fa3_bwd_dkv_kernel` in place
- keep experiment history in git, workbook, ledger, and optimization log
