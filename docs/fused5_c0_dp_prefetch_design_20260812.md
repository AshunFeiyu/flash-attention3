# Fused5 Consumer0 dP Operand Prefetch

## Hypothesis

The rejected consumer1 read8 experiment proved that matrix first-use latency
can move, but adjacent dP/score MMAC islands lose useful MMAC/VALU coissue.
Consumer0 already has independent work between score and dP:

```text
score MMAC -> softmax/P -> dV -> dP MMAC -> dS
```

Issue only the dP lhs reads after sidecar values are latched and before
softmax/P.  Keep the fragments live while softmax and dV execute, then wait at
dP first use:

```text
score MMAC -> sidecar read/wait -> issue dP[4]
           -> softmax/P -> dV -> wait0 -> dP MMAC -> dS
```

This uses existing useful work to cover LDS latency and preserves the
consumer0/consumer1 cadence.  It does not pair two MMAC islands.

## Invariants And Budget

- Exact five-GEMM DAG, MMAC count, output ownership, and causal mask are
  unchanged.
- Consumer1, producer, and dQ-writer schedules are unchanged.
- LDS115,456B and all seven ABarrier lifetimes are unchanged.
- Sidecar still uses three ordinary 32-bit LDS reads; only their issue order is
  made explicit so their wait cannot capture the younger dP matrix reads.
- Consumer0 keeps four dP fragments live across softmax+dV. Expected pressure
  increase is about eight VGPR; reject on spill/scratch or WDRA overflow.

## Gates

1. ASM must show sidecar read3/wait, dP matrix read4, useful VALU/dV work, then
   dP MMAC without a long first-use gap.
2. H1/S128 causal/noncausal and H1/S1024 causal correctness must pass.
3. No private segment, spill, scratch, or LDS bank conflict.
4. Promote only if same-mode ticks improve; use MMAC active, coissue and XCU
   first-use gaps to explain the result.

## Result

Status: `REJECT_STATS_TICKS_AND_PIPELINE / SOURCE_RESTORED`.

- The intended operand lifetime appeared in ASM. Consumer0 used 181/200 VGPR;
  metadata stayed SGPR60/VGPR124 with no private segment, spill or scratch.
- H1/S128 causal/noncausal and H1/S1024 causal passed with bank0 and exact
  MMOP92,160.
- S1024 fused ticks were `72,171,645`, versus the accepted stats mean
  `72,048,113` (+0.171%). MMAC active was 22.578767%, wait-LGKM 7.413218%,
  and barrier share 27.839326%; none improves the accepted pipeline evidence.
- Dynamic VALU increased by 32 and the explicit early sidecar placement did
  not convert into a lower first-use or elapsed-time path.

Stop instruction-schedule tuning after this second same-tier failure. The next
experiment must shorten raw-page ownership itself, while retaining exact work
and the accepted consumer cadence.

Evidence: `/zys/shaobo_runs/fused5_c0_dp_prefetch_20260812/`.
