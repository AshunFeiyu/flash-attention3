# Fused5 dK Same-Group Pair Island

Date: 2026-08-12

## Hypothesis

The current dK lag-one path issues one Q/dS panel read set per M16 panel. Keep
consumer groups independent, but pair adjacent panels inside each group:

```text
current:  5 reads -> wait -> 8 MMAC, with next-panel overlap
candidate: 10 reads -> wait -> 16 MMAC
```

The pair is applied to panels `0+1` and `2+3`. No cross-group wait, barrier,
formula, output ownership or GEMM count changes.

## Resource And Lifetime

The helper already owns two Q/dS slots. The candidate fills both slots before
the wait and reuses them after the two dK islands, so its static slot count is
unchanged. Q and dS remain valid until the pair's two MMAC islands finish;
`RawUsed` and `DqDone` remain at their existing positions.

## Admission

Require H1/S128 correctness, then H1/S1024 correctness/stats, exact
MMOP92,160, no private/spill/scratch, bank0, and same-shape ticks not worse
than the wait-pruned canonical. If the longer read island exposes first-use
wait or loses the existing lag-one overlap, restore the canonical helper.
