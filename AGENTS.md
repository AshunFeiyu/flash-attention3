# Shaobo FA3 BWD Repository Rules

## Performance Baseline

- Read `results/best_baseline.json` before creating an optimization branch.
- Incremental optimization branches must start from the recorded best commit.
- A local A/B win against an older parent is not a global performance win.
- Do not move the canonical branch or create a new `best/*` tag until
  `scripts/check_best_baseline_gate.py` passes against the recorded best.
- Compare the same shape, compiler, PMD, `GPU_CHIP`, and `GPU_ARGS`. Archive
  the correctness result, stats, and helper `.perf` before promotion.
- Preserve the prior best commit and perf archive. Promotion updates the
  registry only after correctness, resource, bank-conflict, ticks, and MMAC
  active gates pass.

## Kernel Changes

- Keep one canonical implementation. Failed experiments stay in Git history
  and the evidence ledger, not behind production phase switches.
- Make one primary performance hypothesis per branch and commit.
- Run static/resource gates and correctness before PMD performance capture.
- Record `ACCEPT`, `REJECT`, or `OBSERVE` with exact commit and evidence paths.

