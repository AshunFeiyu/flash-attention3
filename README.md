# Shaobo FA3 BWD 7-GEMM

This is the clean Shaobo FA3 backward lane. It contains separate canonical
dKV and dQ kernels plus an end-to-end correctness harness. Historical phase
stacks and PMD output stay outside this repository.

## Goal

- Keep dKV and dQ as readable 7-GEMM kernels with explicit WASP ownership.
- Preserve exact math and output ownership before instruction scheduling.
- Use same-shape ticks as the primary metric and MMAC active as the main
  pipeline-quality metric.
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

- `src/dkv_kernel.cpp`: canonical four-GEMM dKV path.
- `src/dq_kernel.cpp`: canonical three-GEMM dQ path.
- `src/dot_do_o_kernel.cpp`: correctness-chain delta and packed-sidecar stage.
- `src/full_bwd_correctness.cpp`: cached-golden full lifecycle harness.

Build and run the full lifecycle on PMD:

```bash
scripts/build_full_bwd_correctness.sh
S=128 scripts/run_full_bwd_correctness.sh
S=1024 SKIP_BUILD=1 scripts/run_full_bwd_correctness.sh
```

The first run for a shape generates a CPU golden under
`${SHAOBO_GOLDEN_ROOT}`. Later runs validate and reuse the same cache. See
`docs/full_bwd_correctness.md` for the data contract and evidence gates.
