# Matrix Stride Probe Audit

## Revalidated

| Probe | Previous state | Corrected result | Decision |
|---|---|---|---|
| 32x16 direct MLS/store | 240/512 mismatch, suspected PMD | 16x32/stride32: 512/512 exact | ACCEPT |
| 64x16 direct MLS/store | 1024 mismatch, suspected PMD | 16x64/stride64, 8 KiB LDS: 1024/1024 exact | ACCEPT |
| 32x16 completion policies | All policies reproduced truncation | vmcnt, ABarrier, GLC, SLC, GLC+SLC, cache invalidate: 6/6 exact | ACCEPT |
| 32x16 `_rtn` | Mixed into completion failure | Corrected stride still fails; PMD warns return VGPR read-before-init | OBSERVE_RTN_ABI |
| 32x16 reader/writer sweep | Could not attribute because direct failed | Direct 16/16 exact; 192 reader/writer chains are complete permutations but 0 exact | OPEN_SOURCE_SLOTS |
| dKV packed-B16 writer chain | Direct control failed | Direct exact; ds-write chain 510/512 mismatch | OPEN_SOURCE_SLOTS |

## Superseded Artifacts

The following historical copies use 32 rows x 16 columns with stride16 and
must not be replayed as current evidence:

- `artifacts/matrix_store_attribution_20260823_v1/`
- `artifacts/matrix_store_b16_32x16_64x16_repro_20260824/`
- `artifacts/matrix_store_b16_teacher_repro_20260824/`
- `probes/matrix_store_completion_probe.cpp` and
  `probes/matrix_store_shape_family_probe.cpp` in the historical
  `fa3_bwd_fused5_two_global_writers` worktree.
- Uncorrected copies of `dkv_b16_matrix_store_probe.cpp` and
  `matrix_global_roundtrip_probe.cpp` in old repos/worktrees.

Do not bulk-edit historical branches. Use the corrected probes in the current
clean repo as the canonical replay source.

## Not Invalidated By This Correction

- Main operator MLS descriptors using the real tensor leading dimension such
  as D=128 or Nk=32/64.
- Same-LDS normal/trans reader proofs whose direct dense-MMAC oracle already
  passed with the intended tensor stride.
- 32x32 dense MMAC writer/store evidence; its logical shape is symmetric and
  its direct path already passes.
- FP32 MMAC-C to packed-FP16 writer source-slot failures. Correct stride
  removes the transport blocker but does not convert fragment ownership.

## Completed FWD-Native Follow-up

The high-value native probe is now complete:

```text
FWD-exact MLS + ds_read_matrix
  -> v_mmac_f32_16x16x16_f16_4interleave
  -> FWD-style cvt_pk pairing
  -> ds_write_matrix_format_f16
  -> matrix_store_32x16_b16
  -> dense CPU GEMM oracle
```

The direct FWD-style global control is exact for all eight repeated paths. The
normal/trans writer crossed with all four matrix-store T/R modes is complete
but `0/8` exact. Therefore the remaining mismatch is the FP32 C-fragment's
natural FWD pack versus the supported F16 writer source-slot ABI, not stride,
matrix-store completion, or store-view selection. Evidence:
`results/fwd_native_mmac_writer_chain_20260824.md`.
