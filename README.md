# Shaobo FA3 BWD dKV WASP

This repo is the clean rewrite lane for Shaobo FA3 backward dKV.  It exists to
avoid continuing the historical phase stack in `fa3_bwd_wasp.cpp`.

## Goal

- Build the dKV kernel from first principles with a readable WASP structure.
- Optimize FA3 BWD dKV only; dQ remains frozen.
- Primary performance target: MMAC active share, with same-run FA3 forward as
  the hard reference.
- Keep algorithm design, resources, pipeline, and profiler evidence readable.

## Repo Rules

- No `m5out`, `.perf`, logs, full PMD output, or temporary trace files in git.
- Run outputs go under `${SHAOBO_RUN_ROOT}`.  On liuchang use:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean`.
- Each accepted optimization is a git commit with a short evidence note.
- Rejected experiments are documented, not kept as permanent phase branches.
- Major changes update the shared workbook before implementation and update
  perf ledger after profiling.

## Starting Point

This clean lane starts from the latest learned constraints:

- `GPU_CHIP=sb`
- `GPU_ARGS=['--SQCIPfLines=7']`
- Target diagnosis shape: `B=1,H=1,S=1024,D=128,causal=true,fp16`
- Target steady shape: `B=1,H=4,S=2048,D=128,causal=true,fp16`
- Hard gates: correctness pass, no scratch/spill, LDS <= 128KB,
  `ldsBankConflict=0`, main matrix path uses MLS/BPS + `ds_read_matrix` +
  `v_mmac_*lit`.

## Current State

The repo now contains a buildable clean WASP FA3 BWD dKV probe.  It has a real
HIP kernel, standalone launcher, four explicit WDRA role branches, ABarrier
ownership gates, MLS/BPS packet publication, and a score+dP MMAC island.  It is
`BRINGUP_ONLY`: it does not compute dV/dK yet.

The next implementation step is to add the real q-loop, sidecar loading,
softmax+dS, and dV/dK accumulation while preserving the clean role topology and
evidence chain.
