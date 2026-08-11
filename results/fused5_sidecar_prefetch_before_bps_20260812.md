# Fused5 Sidecar Prefetch Before BPS Result

Date: 2026-08-12

Decision: `REJECT_TICKS_REGRESSION_CANONICAL_RESTORED`.

## Hypothesis

Issue the producer's three-value sidecar global load before four Q/dO BPS
matrix loads, then publish those values to LDS. This targets the accepted
kernel's exposed sidecar-load gap without changing the five GEMMs, tile,
barriers, raw pages, output ownership, or matrix path.

## Gates

- ISA order: `global_load_dwordx3 -> matrix_load BPS x4 -> vmcnt(0) ->
  ds_write_b32 x3`.
- H1/S128 causal and noncausal dQ/dK/dV CPU golden: PASS.
- Two H1/S1024 causal complete runs: PASS.
- Role use `12/161/163/86` in `16/204/204/88`.
- SGPR60/VGPR128/LDS131,072B; private/spill/scratch0; bank0.
- Exact compute MMOP92,160; reduction MMOP0.

## PMD Comparison

| Metric | Accepted double page | Candidate | Delta |
|---|---:|---:|---:|
| compute ticks mean | 50,290,467.5 | 51,558,552.5 | +2.522% |
| reduction ticks mean | 2,916,777.5 | 2,721,355.0 | -6.699% |
| complete ticks mean | 53,207,245.0 | 54,279,907.5 | +2.016% |
| MMAC active | 31.476233% | 31.059025% | -0.417 pp |
| wait-VM | 2.589046% | 1.632839% | -0.956 pp |
| wait-LGKM | 11.572147% | 12.402483% | +0.830 pp |
| barrier | 15.940946% | 16.137597% | +0.197 pp |
| VALU | 127,352 | 128,280 | +928 |
| SCA | 58,336 | 59,424 | +1,088 |

The source-level split duplicates lane predicate/exec-mask and address setup,
increases producer live state from 9 to 12 VGPR, and grows the static kernel
from 2,076 to 2,114 instruction lines. The VM request ages as intended, but
extra control plus higher LGKM pressure dominates the saved VM wait.

## Evidence

- Build: `/zys/shaobo/fa3_bwd_5gemm_toolkit/build/fused5_sidecar_prefetch`
- Runs: `/zys/shaobo_runs/fused5_sidecar_prefetch_20260812`
- Compiler: LLVM `e0f10535`; PMD HEAD1694; `GPU_CHIP=sb`; SQC prefetch7.

No fullperf/xcu was captured because repeated same-shape ticks failed the A5
promotion gate. Production source is restored byte-for-byte to the accepted
double-page implementation at `d62c645`.
