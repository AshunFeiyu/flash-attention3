# Fused5 C0 dO Early-Prefetch Design

## Evidence

The accepted `a427be9` H1/S1024 fullperf reports `34.330377%` MMAC active.
XCU pipeline evidence for one heavy C0 wave attributes about 10.8k cycles to
`ds_read_matrix_format -> s_waitcnt` and another 10.8k to
`ds_read_matrix_trans_format -> s_waitcnt`. C1's accepted dO lag-one schedule
reduces its corresponding trans-read wait to about 8.7k cycles.

## Hypothesis

C0 currently reads dO-trans only after score, softmax and dV have completed,
then waits before dP. Issue the current dO-trans packet before score, together
with the already accepted next-Q packet. A count-controlled wait admits the
current Q while leaving dO and next Q outstanding. The existing dV readiness
wait retires those reads, so dP consumes a ready register packet.

```text
before: Qnext read -> wait Qcur -> score -> softmax -> dV -> dO read -> wait -> dP
after:  dOcur read + Qnext read -> wait Qcur -> score -> softmax -> dV -> dP
```

The algorithm DAG, M64/N128 tile, five-GEMM count, dK/dV/dQ ownership, LDS
layout and all ABarrier generations remain unchanged. C0 has 26 VGPRs of
branch headroom (`178/204`), and the added dO packet costs 16 VGPRs.

## Gates

1. C0 branch must remain at or below 204 VGPR with private/spill/scratch0.
2. S128 causal/noncausal and S1024 causal full correctness must pass, bank0.
3. Generated ISA must show the dO/Q read island before score and no second dO
   trans read before dP.
4. Promote only if paired S1024 fused ticks improve; MMAC active and XCU must
   confirm reduced C0 matrix-read wait rather than shifted barrier debt.

## Result

Status: `REJECT_SHIFTED_TO_BARRIER`.

The candidate passes S128 c0/c1 and S1024 full correctness, bank0, and uses
roles `9/194/179/86` with private/spill/scratch0. Two candidate S1024 runs are
`47,283,600` and `46,151,105` fused ticks versus the paired a427 control
`45,955,455`. Wait-LGKM improves from `9.1138%` to `8.06--8.11%`, but barrier
share rises from `14.0511%` to `14.90--15.33%`, MMAC active falls from
`34.2706%` to `33.58--33.85%`, and failed coissue rises sharply. The dO wait
was hidden, but C0/C1 reached the dS ownership edge in a less useful cadence.
The source is restored to a427.
