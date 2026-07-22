# DS Matrix Register ABI Roundtrip Probe

## Question

Do `ds_write_matrix_format_f16` and matching m32 matrix readers share a
compatible LDS swizzle, even though their register lane/word layouts differ?

```text
A-reg -> ds_write_matrix_format_f16 -> LDS
      -> ds_read_matrix[_trans]_format_f16 -> A1-reg
```

The probe contains no MMAC, ABarrier, cross-wave handoff, ordinary
`ds_read_b*`, permutation, gather, or production FA code.

## Method

- Source: one wave, 64 lanes, 8 unique finite FP16 bit patterns per lane.
- Writers: normal/trans times alt0/alt1.
- Readers: matching normal m32 alt0/alt1 and trans m32 alt0.
- Phase 1 tags every source lane/word and measures the complete register-slot
  permutation for all 12 writer/reader pairs.
- Phase 2 CPU-inverts each measured permutation, prepares the source for that
  writer ABI, and requires strict lane/word identity after the native chain.
- Toolchain: LLVM `e0f10535...`, PMD HEAD1694, `GPU_CHIP=sb`, SQ7.

## Result

Run: `/zys/sb/fa3b/layout_probes/`
`ds_matrix_reg_roundtrip_20260722_220758`.

- Transport passes with SGPR14/VGPR7, no private segment or spill, and
  `ldsBankConflict=0`.
- ASM has four matrix writers, two normal readers and one transpose reader;
  MMAC0, scalar-matrix-read0 and permutation0.
- A lane-linear source is not identity for any pair: `identity_pairs=0/12`.
- All pairs are complete 512-word bijections: `permutation_pairs=12/12`.
- After CPU inverse-packing for the writer source ABI, every pair is exact:
  `replay_identity_pairs=12/12`, with `mismatch=0` for every pair.
- Full slot map: `ds_matrix_slot_map.csv`, SHA256
  `c40636535369ea0f577a4ee8ea036d85e3b2bad6ae3f86b4e2accfd026f27f4d`.

Representative calibration:

```text
normal_alt0 -> normal_m32_alt0: identity_mismatch=510
normal_alt0 -> trans_m32_alt0:  identity_mismatch=504
all pairs: unmapped=0 duplicate=0 missing=0
all inverse replays: mismatch=0
```

For `normal_alt0 -> normal_m32_alt0`, destination lane0 words0..7 come
from source lanes `[0,16,32,48,8,24,40,56]`, all at source word0. For
`normal_alt0 -> trans_m32_alt0`, destination lane0 comes from source slots
`(0,0),(0,1),(1,0),(1,1),(8,0),(8,1),(9,0),(9,1)`.

## Interpretation

The compiler guidance is correct: writer and readers use a compatible unified
LDS swizzle layout. The apparent mismatch came from comparing two different
register ABIs as lane-linear identity. `ds_write_matrix` consumes a producer
fragment layout, while normal/trans readers emit different consumer MMAC
operand layouts. The LDS transport is lossless and bijective.

The inverse replay is proof, not a production implementation. It prepares the
source on the CPU. A real kernel still needs its MMAC/VALU producer to
naturally generate the required writer source slots; otherwise a runtime
cross-lane conversion would still be needed.

An earlier run passed LDS byte offset 16 and skipped eight FP16 words. The
correct builtin byte offset is zero when the pointer already names the page
base.

Do not use raw register identity to reject a documented pairing. First prove
the writer/reader slot ABI, then validate the actual MMAC-output-to-next-MMAC
semantic chain with a dense CPU oracle.

## Reproduce

```bash
cd /zys/shaobo/fa3_bwd_wasp_clean
GPU_CHIP=sb GPU_ARGS="['--SQCIPfLines=7']" \
  scripts/run_ds_matrix_reg_roundtrip_probe.sh
```
