# Matrix Store Shape Correction

## Corrected Contract

The instruction suffix does not describe the row-major global tensor as
`rows x columns` in the way the old probes assumed.

| Instruction | Correct row-major probe | Elements | Working mode |
|---|---:|---:|---|
| `matrix_load/store_32x16_b16` | 16 rows x 32 columns, stride 32 | 512 | `T=1,R=0` |
| `matrix_load/store_64x16_b16` | 16 rows x 64 columns, stride 64 | 1024 | `T=1,R=0` |

The 64x16 MLS path also requires an LDS reservation large enough for its
swizzled footprint.  The focused probe reserves 8 KiB.  Sweeping arbitrary
`R` modes is not a valid capability test because some combinations violate
the BPS alignment contract.

## Evidence

Environment: gfx946, `GPU_CHIP=sb`, SQ7, compiler `e0f10535...`, PMD
HEAD1734.

- Corrected 32x16 direct chain: `0/512` mismatches.
  Run: `/zys/sb/matrix_store_32x16_head1734/fixed_20260824_170530`.
- Corrected 64x16 direct chain: `0/1024` mismatches.
  Run: `/zys/sb/matrix_store_64x16_direct_probe/run_20260824_171452`.
- Corrected 32x16 sweep: all 16 direct MLS/store combinations are complete
  permutations and exact.  The 192 generic
  `MLS -> ds_read_matrix -> ds_write_matrix -> matrix_store` combinations
  are complete permutations but none is exact.
  Run: `/zys/sb/matrix_store_32x16_head1734/sweep_20260824_170709`.

The old 32x16 result had exactly 240 mismatches because a 32x16 logical
matrix was written with stride 16.  The resulting 32 overlapping 16-wide
rows cover only 272 unique locations, so `512 - 272 = 240` values remain
poisoned.  This was a shape/stride error, not matrix-store truncation.

## Conclusion

`matrix_store_32x16_b16` and `matrix_store_64x16_b16` work on the current
PMD for the corrected row-major contracts.  PMD-005 is not reproduced by
these focused direct tests.

The remaining open problem is narrower: the destination slots of a generic
`ds_read_matrix` fragment are not automatically the source slots expected
by `ds_write_matrix`.  For the dK/dV epilogue, the exact native contract from
MMAC FP32 C-fragments through FP16 packing into the matrix writer must still
be proven.  Do not infer that contract from a direct MLS/store pass.

## Reproduce

```bash
GPU_CHIP=sb GPU_ARGS="['--SQCIPfLines=7']" \
  scripts/run_matrix_store_64x16_direct_probe.sh
```

The corrected 32x16 coverage is exercised by
`scripts/run_matrix_global_roundtrip_probe.sh` and the focused dKV probe.
