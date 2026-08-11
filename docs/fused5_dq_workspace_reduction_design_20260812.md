# Fused5 dQ Workspace Reduction Design

Date: 2026-08-12

Status: `ACCEPT_CANONICAL / NEXT BOTTLENECK ABARRIER`.

## Measured Trigger

Accepted baseline `b28e73d` at H1/S1024/D128 causal:

- fused ticks: 71,950,060
- MMAC active: 22.725077%
- exact MMOP: 92,160
- ABarrier issue-gap share: 29.58%
- dQ atomic issue-gap share: about 15.74%
- matrix-read first-use share: about 10.79%

Atomic removal alone cannot reach 50% MMAC active. Treating the issue-gap
share as an optimistic bound gives roughly `22.725% / (1 - 15.74%) = 26.97%`.
This experiment is admitted because atomic output is an independent measured
boundary and is also unsafe as a future cross-die accumulation contract.

## Formula And Ownership

The five logical GEMMs remain unchanged:

```text
score = Q @ K^T
dP    = dO @ V^T
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

Current dQ ownership uses one CTA per K128 tile and FP32 atomic adds into the
final dQ. The candidate changes only the reduction boundary:

```text
compute CTA(k_tile): dQ_partial[k_tile, q, d] = dS @ K
reduction owner:     dQ[q, d] = sum_valid_k_tile dQ_partial[k_tile, q, d]
```

Each partial element has one writer. Each final dQ element has one reduction
writer. Score/dP/dV/dK/dQ MMAC work remains exact; no GEMM is repeated.

For causal H1/S1024, eight K tiles produce 589,824 FP32 partial elements.
The workspace allocation is dense and simple:

```text
B * H * (S / 128) * S * D * sizeof(float) = 4,194,304 bytes
```

The reduction skips invalid causal K tiles, so unwritten workspace rows are
never read and the workspace does not require a memset.

## Resource And Traffic Budget

| Item | Baseline | Candidate |
|---|---:|---:|
| Fused LDS | 115,456 B | 115,456 B |
| Fused WDRA windows | 8/200/200/88 | unchanged |
| Fused MMOP | 92,160 | 92,160 |
| dQ output instructions | 9,216 scalar atomics | 2,304 vector stores |
| Causal partial-store bytes | atomic RMW contract | 2,359,296 B |
| Reduction reads | 0 | 2,359,296 B |
| Final dQ writes | atomic destination | 524,288 B |
| Workspace capacity | 0 | 4,194,304 B |

The extra reduction dispatch is charged in full. A candidate is not promoted
because the compute dispatch alone becomes shorter or its MMAC active rises.

## Translation-Unit Boundary

The WDRA compute kernel remains in `src/fused_bwd_kernel.cpp` and keeps
local-wave/run-on-model plus entry `__builtin_hcu_wdra_init`.

The plain vector reduction kernel lives in a separate translation unit built
without local-wave/WDRA flags. This prevents an ordinary kernel from receiving
an implicit `s_set_vgpr_size` contract without WDRA initialization.

## Expected Pipeline

```text
time0  P0: publish raw Q/dO
       C0/C1: five-GEMM consumer pipeline
       WQ: dQ MMAC -> vector partial store

time1  next q tile continues with the same ABarrier ownership
       previous atomic serialization is absent

time2  after all compute CTAs complete, reduction CTAs
       vector-load valid K-tile partials -> FP32 sum -> vector final store
```

This deliberately does not modify ABarrier or matrix-read scheduling. It tests
one hypothesis: whether replacing serialized atomic output with an explicit
unique-owner reduction lowers complete-lifecycle ticks.

## Gates

1. H1/S128 causal and noncausal correctness PASS for dQ/dK/dV.
2. H1/S1024 causal correctness PASS.
3. Compute MMOP exactly 92,160; reduction MMOP zero.
4. Both kernels private/spill/scratch0; compute LDS115,456 B and bank0.
5. Compute main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
6. Same-runtime repeated total ticks are
   `compute_partial + dq_reduce < 71,950,060` fullperf baseline.
7. SQTT, if promoted, must show the atomic gap removed without an equivalent
   VMEM/store/reduction tail.

If total ticks do not improve, remove the candidate source and retain only
this evidence. Do not tune reduction block size before the first complete
result unless resource or correctness gates require it.

## Measured Result

The complete two-dispatch lifecycle passes H1/S128 causal and noncausal plus
H1/S1024 causal CPU golden. Both kernels are private/spill/scratch0 and bank0.
The fused compute retains MMOP92,160 and the reduction has MMOP0.

Repeated H1/S1024 stats:

| Metric | Atomic baseline | Workspace reduction |
|---|---:|---:|
| Mean complete ticks | 72,048,112.5 | 58,696,137.5 |
| Mean fused-compute ticks | 72,048,112.5 | 56,003,902.5 |
| Mean reduction ticks | 0 | 2,692,235.0 |
| Complete improvement | - | 18.532% |
| Fused MMAC active | 22.714850% | 28.897238% |
| Fused barrier share | 27.5236% | 22.8141% |

Fullperf total ticks are 58,948,890, 18.070% below the 71,950,060 baseline.
XCU duration falls from 158,132 to 123,436 cycles. The old atomic issue gaps
(12.60% atomic-to-atomic plus 3.14% address-to-atomic) disappear. The terminal
ebarrier edge falls from 11.49% to 6.12%.

The inner consumer schedule is not materially improved: representative C0/C1
classification changes from `158 MMAC-vs-MMAC / 135 MMAC-vs-VALU` to
`162 / 139`. The candidate mainly deletes the exposed atomic tail. The next
structural target is the remaining `s_abarrier_try_wait -> s_xor_b32` edge,
which occupies 40.72% of candidate issue-gap cycles.
