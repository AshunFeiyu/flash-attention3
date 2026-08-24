# Matrix Global Roundtrip Probe

> **SUPERSEDED (2026-08-24):** This probe treated `32x16` as a row-major
> `32 rows x 16 columns` tile with stride 16.  The corrected row-major
> contract is `16 rows x 32 columns` with stride 32.  On PMD HEAD1734 the
> corrected direct `matrix_load_32x16_b16 -> matrix_store_32x16_b16` chain
> is exact for all 512 elements.  See
> `results/matrix_store_shape_correction_20260824.md`.  The truncation and
> PMD-defect interpretation below is retained only as historical evidence
> of the invalid probe contract.

## Question

Does a `32x16` FP16 matrix survive this native path bit-for-bit?

```text
A-global -> matrix_load_32x16_b16 BPS -> LDS
         -> ds_read_matrix -> A-reg
         -> ds_write_matrix -> LDS
         -> matrix_store_32x16_b16 -> A1-global
```

## Coverage

- Input: 512 unique finite FP16 bit patterns in row-major `32x16` storage.
- Full chain: four MLS `t/r` modes, three matching m32 readers, four f16
  writer modes, and four matrix-store `t/r` modes: 192 combinations.
- Direct control: four MLS modes times four matrix-store modes, bypassing
  `ds_read_matrix` and `ds_write_matrix`: 16 combinations.
- Every full-chain candidate has a private 2 KiB LDS writer page.
- No MMAC, scalar matrix read, gather, permutation, or WDRA is present.

## Environment

- LLVM commit `e0f10535...`, clang SHA256 `334cb561...`.
- PMD HEAD1694, `GPU_CHIP=sb`, SQ7.
- Run:
  `/zys/sb/fa3b/layout_probes/matrix_global_roundtrip_20260722_202711`.

## Result

- Static gates: SGPR18/VGPR4, private/spill/scratch0, LDS 100,352 bytes.
- ASM: MLS4, normal reader2, transpose reader1, writer4, matrix-store8,
  MMAC0, scalar-read0, permutation0.
- PMD transport passes with `ldsBankConflict=0` and `simTicks=20,328,035`.
- Full-chain exact pairs: `0/192`; complete permutation pairs: `0/192`.
- Direct MLS-to-matrix-store exact pairs: `0/16`; complete permutation pairs:
  `0/16`.
- Every direct control has exactly 240 mismatches. Elements in rows 0..16
  are correct; row17/col0 is the first failure and contains `0xfefe`. Thus
  only `17x16=272` of 512 FP16 values reach global memory.
- Matrix-store `t/r` flags do not change this boundary on the current PMD.

Representative direct result:

```text
row_mismatch=240 unmapped=240 duplicate=0 missing=240
first_row=17 first_col=0
first_expected=0x3110 first_actual=0xfefe
```

## Interpretation

The requested end-to-end chain is not equal on the current PMD, but the
direct control proves that the first blocking point is
`matrix_store_32x16_b16`: it already truncates an MLS-created LDS page before
any DS register roundtrip is introduced. Therefore the 192 full-chain
failures cannot be used to select or reject a DS reader/writer semantic pair.

This reproduces PMD-005 with a broader, byte-correct builtin sweep. Keep native
matrix-store out of the canonical dKV/dQ epilogue until the PMD/compiler owner
provides the required descriptor/source-layout contract or fixes the model.

## Reproduce

```bash
cd /zys/shaobo/fa3_bwd_wasp_clean
GPU_CHIP=sb GPU_ARGS="['--SQCIPfLines=7']" \
  scripts/run_matrix_global_roundtrip_probe.sh
```
