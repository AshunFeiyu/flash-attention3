# Client Notes

## Code Governance

- This repo is the clean FWD-style rewrite lane.
- Historical code remains in `/zys/shaobo/fa3_bwd_wasp` and
  `remote_src/fa3_bwd_wasp`.
- Do not place PMD outputs, `.perf`, `m5out`, or large logs in this repo.
- Use git commits for accepted optimization steps.
- Rejected experiments should be described in docs or commit messages, then
  removed from code unless they are useful guarded diagnostics.

## Current Development Contract

- Port from old repo only when the block has a clear owner and resource budget.
- Every implementation round starts from the workbook/design contract.
- Every perf round records MMAC active share, dKV ticks, waits/barriers,
  coissue, bank conflicts, and Source/Wavefronts explanation.

