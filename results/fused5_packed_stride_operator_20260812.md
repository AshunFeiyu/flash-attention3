# Fused5 Packed P/dS Stride Operator Result

Date: 2026-08-12

Decision: `OBSERVE_A5_PASS_ENABLER_NOT_PROMOTED`.

## Change

Keep raw Q/dO and resident K/V matrix panels at their existing 2KB spacing,
but pack native `ds_write_matrix_32x16_trans_f16` P/dS pages at the verified
1KB writer footprint. This reduces P scratch from 16KB to 8KB and four-panel
batch dS storage from 64KB to 32KB.

## Gates

- H1/S128 causal: dQ/dK/dV PASS.
- H1/S128 noncausal: dQ/dK/dV PASS.
- H1/S1024 causal: dQ/dK/dV PASS.
- Compute role VGPR: `8/163/166/86`; SGPR60/VGPR124/LDS107,264B.
- Reduction: SGPR26/VGPR36/LDS0.
- Both kernels private/spill/scratch0; `ldsBankConflict=0`.
- Dynamic compute MMOP remains exactly 92,160.

## Performance

| Run | Compute ticks | Reduction ticks | Total ticks |
|---|---:|---:|---:|
| 1 | 56,238,455 | 2,690,870 | 58,929,325 |
| 2 | 56,366,310 | 3,001,180 | 59,367,490 |
| Mean | 56,302,382.5 | 2,846,025.0 | 59,148,407.5 |

The accepted canonical mean is 58,696,137.5 ticks. Compact stride alone is
0.771% slower end to end and 0.533% slower in the compute dispatch. Compute
MMAC active averages 28.760%, versus 28.897% for the paired canonical stats.
The dynamic ISA mix is otherwise unchanged except SCA falling by 96.

The result is not promoted as a speedup. It is retained as an operator-proven
resource enabler for the next independent hypothesis: use the released LDS for
a complete Q+dO raw page1 and test whether that shortens the dominant
`RawUsed(t) -> RawFilled(t+1)` ownership edge.

Evidence root:
`/zys/shaobo_runs/fused5_packed_stride_20260812`.
