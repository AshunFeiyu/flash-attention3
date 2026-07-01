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
6. Judge by MMAC active share first, then dKV ticks, wait/barrier, coissue,
   Source CSV, and Wavefronts.
7. Commit accepted changes with evidence; remove or document rejected code.

## Artifact Rules

Keep generated files out of git:

- no `m5out*`
- no `.perf`
- no full PMD logs
- no temporary CSV exports

Use `${SHAOBO_RUN_ROOT}` for runs.  On liuchang:

```bash
export SHAOBO_RUN_ROOT=/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean
```

Shared review artifacts go under:

```text
/Volumes/172.20.68.76/共享/shaobo
/Volumes/172.20.68.76/共享/shaobo/perf
```

## Current Next Step

Create the first real kernel cut by porting C125C semantics into this clean
structure without copying C125C's phase-stack plumbing.  The first promotion
gate is not speed; it is a readable, buildable, no-spill kernel whose pipeline
can be inspected.

