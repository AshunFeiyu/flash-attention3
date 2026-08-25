# Fused5 C1 Early dS Publication

Status: `DESIGN_READY_FOR_ONE_HYPOTHESIS_EXPERIMENT`.

## Evidence

C85 H1/S1024 causal XCU token-sequence tomography assigns all 2,048
`s_abarrier_try_wait -> s_xor_b32` gaps without changing the kernel:

- producer `RawUsed`: 782,412 cycles; producer-side and mostly overlappable;
- dQ-writer `BatchDsFilled1`: 293,592 cycles, 68.51% of writer wait time;
- consumer1 `RawFilled0/1`: 190,172 cycles;
- all `DqDone` waits: 1,548 cycles.

The consumer-critical chain is therefore producer -> consumer1 -> dQ writer.
`DqDone` reuse pressure is not the current limiter.

## Formula And Ownership

The five GEMMs and ownership do not change:

1. `S = Q @ K^T`
2. `dP = dO @ V^T`
3. `dV += P^T @ dO`
4. `dK += dS^T @ Q`
5. `dQ_partial = dS @ K`

Consumer1 still owns the same N16 dK/dV slice. It computes every `P` and `dS`
fragment once. The dQ-writer remains the sole reader of the published dS page.

## Schedule Change

Current consumer1 order per M16 panel is:

`score -> dP -> softmax/dS -> dV`, followed by batch dS publication and dK.

Candidate order is:

`score -> dP -> softmax/P -> issue dV operands -> dS` for all four panels,
publish the complete dS batch, then consume the already-issued dV operands and
run dK. The four dV packets stay live across dS publication so the publication
does not expose a new `ds_read_matrix -> wait -> dV MMAC` chain.

Expected steady overlap:

| Time | Consumer0 | Consumer1 | dQ writer |
| --- | --- | --- | --- |
| t0 | score/dP/dS | score/dP, issue dV operands, dS | wait group1 dS |
| t1 | dV/dK | publish group1 dS, consume ready dV | group1 dQ MMAC |
| t2 | next packet or dV/dK | dV/dK | group0 dQ MMAC / store |

This is useful-work staggering: dQ MMAC covers consumer1 dV/dK. No empty delay,
new barrier, extra GEMM, or layout conversion is introduced.

## Resource Budget

- LDS remains 128 KiB with the same raw, resident, dS, and sidecar regions.
- Each dV packet holds one P fragment and four dO fragments, about 20 VGPRs.
- Four live packets require about 80 VGPRs, roughly 60 more than the previous
  one-packet schedule. Other C1 temporaries remain panel-local.
- WDRA targets remain `16/204/204/88`; any compiled role overflow rejects the
  candidate before PMD.

## Admission Gates

- Exact MMOP remains 88,064 for H1/S1024 causal.
- Correctness passes S128/S1024, noncausal S128, and GQA Hq4/Hkv2/S128.
- No private segment, scratch, SGPR/VGPR spill, or LDS bank conflict.
- Same-build stats-only ticks must improve over C85.
- A winning fullperf must reduce `BatchDsFilled1` wait and total ticks. Higher
  MMAC active without lower ticks is not sufficient.
