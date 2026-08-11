# ds_write_matrix Packed-Stride Probe Result

Date: 2026-08-12

Decision: `PASS_A0_A4 / OPERATOR_INTEGRATION_PENDING`.

## Contract

Two waves write deterministic FP16 fragments at packed bases `0/1024` and at
known-safe control bases `4096/6144`. Two reader waves compare both native
normal/trans fragment views and a dense MMAC result.

## Result

| Writer | Normal mismatch | Trans mismatch | MMAC mismatch |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |

Metadata is SGPR18/VGPR26 with private/spill/scratch0. ASM contains two
`ds_write_matrix_format`, two normal `ds_read_matrix_format`, two
`ds_read_matrix_trans_format`, and two MMAC instructions.

This proves a 1024-byte writer stride is equivalent to the 2048-byte control
for the fragment views consumed by the fused backward kernel. It does not yet
prove whole-operator correctness or performance.

Evidence:
`/zys/shaobo_runs/fused5_packed_stride_probe_20260812/ds_write_matrix_packed_stride_20260812_045436`.
