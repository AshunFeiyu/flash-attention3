# Fused5 C1 dO Lag-One

## Design

Consumer group1 keeps the five-GEMM DAG and all ownership tokens unchanged.
For panel `m`, its dV operand packet is issued as:

```text
P-normal(m) + dO-normal(m) + dO-trans(m+1)
wait lgkmcnt(4)
dV(m) MMAC x8
```

The four future dO-trans reads remain outstanding across the current dV
island. At panel `m+1`, four Q-trans reads are issued, the older dO packet is
retired with `lgkmcnt(4)`, and C1 executes dP before retiring Q for score.
There is no new LDS page, ABarrier, GEMM, or global transaction.

## Gates

- Generated ISA proves next dO reads precede the partial wait and full dV
  island, followed by next Q reads and the dP/score selective-wait pair.
- Branch VGPR use: `9/178/179/86` inside `16/204/204/88`.
- Metadata: SGPR60/VGPR128, private/spill/scratch 0.
- H1/S128 causal and noncausal: PASS.
- H1/S1024 causal: PASS.
- Exact MMOP 92,160; LDS bank conflict 0.

## Performance

- Stats-only candidate: 45,600,100 and 45,280,235 fused ticks.
- Same-round control/candidate: 46,290,335 -> 45,280,235 (-2.182%).
- Fullperf control/candidate: 46,149,740 -> 45,496,815 (-1.414%).
- MMAC active: 33.794843% -> 34.330377% (+0.535534 pp).
- Coissue success/fail: 21,933/25,812 -> 23,313/24,165.
- Barrier share: 14.431084% -> 14.000763%.
- XCU normal matrix-read-to-wait: 5.68% -> 4.02%.
- XCU FP32 MMAC-to-MMAC gap: 5.95% -> 5.68%.

The aggregate wait-LGKM share rises slightly because the remaining wait owns
more retired work, but elapsed ticks, MMAC active, coissue, and the relevant
normal-read gap all improve. The result is therefore an explained pipeline
win rather than a counter-only win.

## Decision

`ACCEPT_TICKS_AND_ACTIVE_NEW_BEST`. Continue from this commit. The next
hypothesis must target another measured critical-chain window; do not add
ABarrier refinements or overwrite the accepted dO/Q packet schedule.

