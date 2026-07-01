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
- LDS budget target is 115456 B with 15616 B slack under 128 KB.
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

Latest evidence archive:

```text
/Volumes/172.20.68.76/共享/shaobo/perf/20260701_235316_clean_stream_qloop_probe
```

Next step: use the reference path as the correctness oracle while moving the
math into the WASP path in this order: softmax/dS sidecar, dV MMAC accumulation,
dK MMAC accumulation, store epilogue.  Keep batching same-family
`ds_read_matrix` reads and delaying `s_waitcnt` until true first use.
