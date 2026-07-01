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
- LDS budget target is 98816 B, leaving about 28 KB slack under 128 KB.
  Do not reserve a separate LDS raw-to-trans scratch in the main path unless a
  workbook row proves it.  Source-layout operands must come from MLS/BPS pages
  that are reused after raw-page ownership is released, or be added explicitly
  to the resource budget before code changes.
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
