# dQ FP32 DS-Write To Matrix-Store Probe

Status: `ACCEPT_NATIVE_LIT0_WRITER_STORE_ABI / PRODUCTION_REJECTED`.

## Contract

The probe tests the exact dQ epilogue chain with a dense asymmetric 16x16
CPU oracle:

```text
MLS/BPS -> ds_read_matrix -> FP32 MMAC C
        -> ds_write_matrix_format_f32 -> LDS
        -> matrix_store_16x16_b32 -> global
```

It also includes a direct
`matrix_load_16x16_b32 -> matrix_store_16x16_b32` control and an
MLS-read-to-DS-write replay. No scalar gather, permutation, ordinary matrix
DS read, or direct VGPR matrix-store workaround is present.

## Result

- Compiler: `e0f10535`; PMD: `CoreArch:HEAD_1694`.
- Control MLS-to-store is exact.
- Two MMAC/writer tuples are exact against the CPU dense GEMM:
  - `lit0/lts0 -> trans FP32 writer -> B32 matrix store`.
  - `lit0/lts1 -> normal FP32 writer -> B32 matrix store`.
- FWD `mmac_4interleave`, which is exactly `lit1/lts0`, does not match either
  FP32 writer orientation.
- Static evidence: B16 MLS2, B32 MLS2, matrix-read4, MMAC4, FP32 writer4,
  LDS-source B32 matrix-store1, ordinary DS read0, permutation0, trap0.
- Metadata: SGPR24/VGPR12, private/spill/scratch0; PMD bank conflict0.
- BPS readiness is part of the contract. Omitting `s_waitcnt_vbcnt 0` makes
  the MMAC source invalid and must not be misattributed to the FP32 writer.

Evidence:

`/zys/sb/dq_f32_writer_store_test/layout_probes/`
`dq_f32_dswrite_store_20260825_112154`

## Integration Decision

The canonical dQ GEMM uses the same `lit1/lts0` operation as FWD
`mmac_4interleave`. The production candidate changed dQ only to `lit0/lts0`,
then used the transposed FP32 writer and `matrix_store_16x16_b32`. It passed
full correctness and resources but regressed H1/S1024 lifecycle ticks by
`10.954%`, so production remains on the accepted interleave plus FP16 direct
partial-store path. See
`results/dq_f32_native_matrix_store_operator_ab_20260825.md`.
