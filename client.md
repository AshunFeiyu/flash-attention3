# Client

## Mission

Build a clean FA3 BWD dKV kernel in FA3 FWD style.  The primary optimization
target is MMAC active share, using same-run FA3 FWD as the benchmark.  dQ is
frozen.

## Code Plan

Keep the repo small and modular:

- `include/stage61_dkv_contract.h`: tile, role, resource, and target constants.
- `src/stage61_dkv_fwdstyle.cpp`: one cohesive Stage61 dKV implementation.
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

## Artifact Rules

Keep generated files out of git:

- no `m5out*`
- no `.perf`
- no full PMD logs
- no temporary CSV exports
- no `xcu` SQTT output directories

Use `${SHAOBO_RUN_ROOT}` for runs.  On liuchang:

```bash
export SHAOBO_RUN_ROOT=/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean
```

Shared review artifacts go under:

```text
/Volumes/172.20.68.76/共享/shaobo
/Volumes/172.20.68.76/共享/shaobo/perf
```

Current Stage61 S0 design workbook:

```text
/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_stage61_cleanrepo_s0_20260701.xlsx
```

## Current Next Step

Current state is S2 `BRINGUP_ONLY`:

- producer0 publishes Q + K with MLS/BPS
- producer1 publishes dO + V with MLS/BPS
- two consumer groups wait packet ownership tokens and execute a score+dP
  `ds_read_matrix + v_mmac_*lit` probe
- standalone now allocates real Q/K/V/dO buffers before running PMD

Next step: replace the probe with a real q-loop body, add sidecar loading, then
connect softmax+dS before adding dV/dK accumulation and stores.
