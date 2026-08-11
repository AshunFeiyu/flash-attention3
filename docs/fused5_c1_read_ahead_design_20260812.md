# Fused5 Consumer1 Operand Read-Ahead

## Hypothesis

The accepted `b28e73d` SQTT shows that each heavy consumer repeatedly executes
four matrix reads, a full `lgkmcnt(0)` wait, then eight FP16 MMAC issues.  In a
representative steady SIMD, the final matrix read to the wait issue averages
96.8 cycles for consumer0 and 100.4 cycles for consumer1.  Both consumers often
pay this first-use gap together.

This experiment changes only consumer1's first two logical GEMMs:

```text
current:  read dP[4] -> wait0 -> MMAC dP[8]
          read QK[4] -> wait0 -> MMAC QK[8]

candidate: read dP[4] -> read QK[4] -> wait4 -> MMAC dP[8]
           -> wait0 -> MMAC QK[8]
```

`lgkmcnt(4)` retires the older dP operand reads while the four younger score
reads remain outstanding.  The dP MMAC island should cover score-read latency,
then consumer1's score MMAC should overlap consumer0's softmax/VALU work.

## Invariants And Budget

- Exact five-GEMM DAG and MMAC count are unchanged.
- Consumer0 schedule is unchanged and remains the cadence anchor.
- LDS layout, all seven ABarrier tokens, packet ownership, stores, and atomics
  are unchanged.
- The main matrix path remains `matrix_load BPS -> ds_read_matrix -> MMAC`.
- Consumer1 temporarily holds two four-fragment lhs batches.  Expected extra
  pressure is about eight VGPR relative to the sequential reuse schedule;
  reject on spill/scratch or an unsafe WDRA window.

## Gates

1. ASM must contain an eight-read island followed by `lgkmcnt(4)`, eight dP
   MMAC issues, `lgkmcnt(0)`, and eight score MMAC issues in consumer1.
2. H1/S128 causal and noncausal plus H1/S1024 causal correctness must pass.
3. LDS bank conflicts, private segment, and all spills must remain zero.
4. Compare same-build H1/S1024 ticks, MMAC active, wait/barrier shares, and XCU
   pipeline evidence against `b28e73d`.

## Result

Status: `REJECT_TICKS_REGRESSION / SOURCE_RESTORED`.

- The intended ISA and every hard gate passed. Consumer1 used 178/200 VGPR;
  metadata remained SGPR60/VGPR124 with no private segment, spill, scratch, or
  bank conflict. H1/S128 causal/noncausal and H1/S1024 causal were correct.
- Two stats-only fused ticks were `72,377,305` and `71,923,215`. Fullperf was
  `72,120,230` versus accepted `71,950,060`, a 0.2365% regression.
- MMAC active improved `22.725077% -> 23.055142%`, but coissue success fell
  `17,378 -> 11,699` and wait-LGKM share rose `7.344969% -> 7.591733%`.
- XCU confirms the trade: transposed matrix-read first-use gaps fell
  `654,436 -> 618,272` cycles, while normal-read gaps rose
  `624,324 -> 635,576`; ABarrier and intra-MMAC gaps also grew slightly.

The read-ahead hid part of the LDS dependency but joined two MMAC islands,
removing useful peer-VALU overlap. Do not retry read8 as adjacent dP/score
MMAC. The next schedule experiment must put independent useful VALU/MMAC work,
not another MMAC island, between operand issue and first use.

Evidence: `/zys/shaobo_runs/fused5_c1_read_ahead_20260812/`.
