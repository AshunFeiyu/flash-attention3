# Fused5 C0 Early-dS-Publish Design

## Evidence

The accepted `a427be9` writer SQTT has one startup `ResidentFilled` wait, then
two dS waits per q tile.  After startup, the first dS wait is consistently the
long wait (`1.4--2.7k` cycles, `4.8k` on the first tile), while the second is
normally satisfied in about three cycles.  Therefore group0, not group1, is
the dQ-writer pacing input.

Group0 currently executes each panel as:

```text
score -> softmax/P -> dV -> dP -> dS
```

It cannot publish the four dS panels until all four intervening dV islands
finish.  Group1 already moves dV after dS and reaches its publication token
before the writer needs it.

## Single Hypothesis

Retain only the four fp16 P fragments in group0, finish all four
`score -> softmax/P -> dP -> dS` chains, publish the complete dS batch, then
execute the four dV islands followed by dK.  This keeps the exact five-GEMM
count and moves existing useful work behind the dS ownership edge:

```text
time0  C0: score/P/dP/dS x4 -> publish dS0
time1  W : dQ(group0) MMAC
       C0: dV x4
       C1: dV / next useful work
time2  W : dQ(group1) MMAC -> store partial dQ
       C0: dK
```

No delay, extra GEMM, extra LDS page, token, or layout transform is added.
The raw-page, dS-page, writer and producer ownership contracts are unchanged.

## Resource Budget

- Accepted C0 branch: 178 VGPR in a 204-VGPR WDRA pool.
- Four fp16 P fragments cost at most 16 VGPR; one was already live in the
  panel body, so expected branch use is no more than about 194 VGPR.
- LDS remains 128 KiB; SGPR and all four WDRA pool sizes remain unchanged.
- dK/dV accumulators remain resident exactly as before.

## Gates

1. Generated metadata: private/spill/scratch0 and C0 <= 204 VGPR.
2. Exact MMOP count unchanged; no ordinary matrix-path DS read.
3. S128 causal/noncausal and S1024 causal full correctness PASS; bank0.
4. Paired S1024 fused ticks must improve.  SQTT should reduce the writer's
   first dS-token bubble without increasing producer/raw-page ownership debt.
5. If the writer bubble merely moves to group1 or terminal synchronization,
   reject and restore `a427be9`.

## Result

Status: `ACCEPT_TICKS_ACTIVE_NEW_BEST`.

Static gates pass with roles `9/172/179/86`, SGPR62/VGPR128 and
private/spill/scratch0.  S128 causal/noncausal and S1024 causal full backward
correctness pass with bank0.  Two paired S1024 runs improve mean fused ticks
from `45,947,038` to `45,021,340` (`-2.015%`).  The candidate fullperf records
`44,794,750` fused ticks and `35.094969%` MMAC active, versus accepted a427
fullperf `45,496,815` and `34.330377%`.

For the same representative dQ-writer wave, xcu reports total
`s_abarrier_try_wait -> s_xor_b32` bubbles falling `34,831 -> 30,943` cycles
and `ds_read_matrix_trans -> s_waitcnt` bubbles falling `22,024 -> 19,832`.
This validates the mechanism: existing dV work moved behind the dS ownership
edge and overlaps writer dQ progress; no extra arithmetic or token was added.
