# Shaobo FA3 BWD dKV FWD-Style Clean Repo

This repo is the clean rewrite lane for Shaobo FA3 backward dKV.  It exists to
avoid continuing the historical phase-stack in `fa3_bwd_wasp.cpp`.

## Goal

- Match FA3 FWD coding style and pipeline discipline.
- Optimize Stage61 dKV only; dQ remains frozen.
- Primary performance target: MMAC active share, with FA3 FWD as the hard
  reference.
- Keep algorithm design, resources, pipeline, and profiler evidence readable.

## Repo Rules

- No `m5out`, `.perf`, logs, full PMD output, or temporary trace files in git.
- Run outputs go under `${SHAOBO_RUN_ROOT}`.  On liuchang use:
  `/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean`.
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

The repo currently contains the FWD-style design scaffold.  The next step is to
port the proven C125C semantics into this structure without copying the
historical phase stack.

