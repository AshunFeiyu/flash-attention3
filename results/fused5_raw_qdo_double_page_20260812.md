# Fused5 Full Raw Q/dO Double-Page Result

Date: 2026-08-12

Decision: `ACCEPT_CANONICAL_STRUCTURAL_WIN`.

## Hypothesis

The accepted producer could not refill Q/dO until both consumers finished dK
and released one raw page. Use two physically disjoint 32KB Q+dO pages and
page-local Filled/Used tokens so packet `t+1` can be loaded while packet `t`
is consumed. The five GEMMs, M64/N128/D128 tile, output ownership, and dQ
workspace reduction remain unchanged.

## Gates

- H1/S128 causal: dQ/dK/dV CPU golden PASS.
- H1/S128 noncausal: dQ/dK/dV CPU golden PASS.
- H1/S1024 causal, two stats runs and fullperf: PASS.
- WDRA window `16/204/204/88`; actual role use `9/161/163/86`.
- Compute SGPR60/VGPR128/LDS131,072B; reduction SGPR26/VGPR36/LDS0.
- Both kernels: private/spill/scratch0 and bank0.
- Compute MMOP92,160; reduction MMOP0.

## PMD Result

Repeated complete lifecycle ticks:

| Run | Compute | Reduction | Total |
|---|---:|---:|---:|
| 1 | 50,374,415 | 2,719,080 | 53,093,495 |
| 2 | 50,206,520 | 3,114,475 | 53,320,995 |
| Mean | 50,290,467.5 | 2,916,777.5 | 53,207,245.0 |

Against the accepted dQ-workspace baseline, complete ticks improve 9.351%
and compute ticks improve 10.202%. Fullperf is
50,863,995 + 2,850,575 = 53,714,570 ticks, 8.879% lower than 58,948,890.

Repeated stats MMAC active rises 28.897238% -> 31.476233%, while barrier
share falls 22.814064% -> 15.940946%. Dynamic MMOP and VMEM are unchanged;
SCA rises 38,192 -> 58,336 because runtime page selection remains in both
producer and consumer loops.

## XCU Attribution

- Fused compute duration falls 123,436 -> 111,792 cycles, a 9.433% reduction.
- Dominant ABarrier-following issue gap falls 40.72% -> 29.23%.
- Trans/normal matrix-read first-use gaps remain 9.45%/7.88%; this ownership
  win does not yet solve LDS first-use latency.
- C0/C1 each execute 2,048 MMAC. MMAC paired with peer vector work is
  395/2,048 for C0 and 641/2,048 for C1; 159 bins are MMAC-vs-MMAC and 160
  are MMAC-vs-VALU. Useful staggering remains asymmetric.
- Terminal ebarrier is 6.12%; global-store consecutive gap is 3.42%.

The measurements prove that a complete raw Q+dO second page shortens the
producer ownership cycle. The next experiment is compile-time page
specialization to recover SCA/control cost without changing pages, tokens,
MMOP, arithmetic order, or output ownership.

## Evidence

- Stats:
  `/zys/shaobo_runs/fused5_raw_qdo_double_20260812/stats`
- Fullperf:
  `/zys/shaobo_runs/fused5_raw_qdo_double_20260812/fullperf/5gemm_owner_s1024_c1_fullperf_20260812_055041`
- XCU:
  `/zys/shaobo_runs/fused5_raw_qdo_double_20260812/xcu`
- Compiler: LLVM `e0f10535`; PMD HEAD1694; `GPU_CHIP=sb`; SQC prefetch7.
