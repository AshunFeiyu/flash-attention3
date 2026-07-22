# DS Matrix Register Roundtrip Probe

## Question

Does a raw FP16 fragment survive this chain bit-for-bit in the same lane and
word position?

```text
A-reg -> ds_write_matrix_format_f16 -> LDS
      -> ds_read_matrix[_trans]_format_f16 -> A1-reg
```

The probe intentionally contains no MMAC, ABarrier, cross-wave handoff,
ordinary `ds_read_b*`, permutation, gather, or production FA code.

## Matrix

- Source: one wave, 64 lanes, 8 unique finite FP16 bit patterns per lane.
- Writers: normal/trans times alt0/alt1.
- Readers: normal m32 alt0/alt1, trans m32 alt0, trans m16 alt0/alt1.
- Comparison: all 20 writer/reader pairs, 512 words per pair.
- Toolchain: LLVM `e0f10535...`, PMD HEAD1694, `GPU_CHIP=sb`, SQ7.

## Result

Corrected run:
`/zys/sb/fa3b/layout_probes/ds_matrix_reg_roundtrip_20260722_201613`.

- Transport passes: PMD exits normally, metadata is SGPR16/VGPR13 with no
  private segment or spill, and `ldsBankConflict=0`.
- ASM has exactly four matrix writers, two normal readers, three transpose
  readers, no MMAC, no scalar matrix read, and no permutation instruction.
- No pair is a same-lane/same-word identity: `identity_pairs=0/20`.
- All 12 combinations using a matching m32 reader are complete 512-word
  permutations: `permutation_pairs=12/20`.
- The m16 readers expose only 256 source words for this m32 source shape and
  return poison in the other 256 slots, so they are shape-mismatched readers.
- PMD prints `ds_write_matrix : testing` and
  `ds_read_matrix_trans : testing` warnings.

Representative normal/normal result:

```text
identity_mismatch=510 identity_pass=0
unmapped=0 duplicate=0 missing=0 permutation_pass=1
```

## Interpretation

`ds_write_matrix` does not promise that an arbitrary lane-linear register
fragment will be returned unchanged by an arbitrary `ds_read_matrix` form.
The writer source contract and reader destination contract are matrix-layout
contracts, normally tied to MMAC output/operand layouts. Most recovered words
show a deterministic lane/word redistribution, not random data corruption.

An earlier run incorrectly passed `16` as the builtin LDS byte offset. Its ASM
contained `offset:16`, skipped eight FP16 words and returned eight `0xfefe`
poison slots. With the required byte offset `0`, all matching m32 paths are
complete permutations. This correction is a probe bug fix, not a hardware or
PMD behavior change.

Do not use raw register identity to reject a documented MMAC-output
writer/reader pairing; validate that exact semantic chain with a dense CPU
oracle.

## Reproduce

```bash
cd /zys/shaobo/fa3_bwd_wasp_clean
GPU_CHIP=sb GPU_ARGS="['--SQCIPfLines=7']" \
  scripts/run_ds_matrix_reg_roundtrip_probe.sh
```
