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

Run: `/zys/sb/fa3b/layout_probes/ds_matrix_reg_roundtrip_20260722_200230`.

- Transport passes: PMD exits normally, metadata is SGPR16/VGPR13 with no
  private segment or spill, and `ldsBankConflict=0`.
- ASM has exactly four matrix writers, two normal readers, three transpose
  readers, no MMAC, no scalar matrix read, and no permutation instruction.
- No pair is a same-lane/same-word identity: `identity_pairs=0/20`.
- No pair is a complete 512-word bijection on this PMD:
  `permutation_pairs=0/20`.
- The m32 readers recover 504 unique source words and return `0xfefe` poison
  for eight destination slots. The m16 readers expose only 256 source words
  for this m32 source shape and return poison in the other 256 slots.
- PMD prints `ds_write_matrix : testing` and
  `ds_read_matrix_trans : testing` warnings.

Representative normal/normal result:

```text
identity_mismatch=511 identity_pass=0
unmapped=8 duplicate=0 missing=8 permutation_pass=0
lane56..63 word7 include PMD poison 0xfefe
```

## Interpretation

`ds_write_matrix` does not promise that an arbitrary lane-linear register
fragment will be returned unchanged by an arbitrary `ds_read_matrix` form.
The writer source contract and reader destination contract are matrix-layout
contracts, normally tied to MMAC output/operand layouts. Most recovered words
show a deterministic lane/word redistribution, not random data corruption.

The eight poison words mean this PMD run cannot prove a complete native
roundtrip even as a permutation. Treat that as a compiler/PMD contract question
because the model itself marks these instructions as testing. Do not use this
probe alone to reject a documented MMAC-output writer/reader pairing; validate
that exact semantic chain with a dense CPU oracle.

## Reproduce

```bash
cd /zys/shaobo/fa3_bwd_wasp_clean
GPU_CHIP=sb GPU_ARGS="['--SQCIPfLines=7']" \
  scripts/run_ds_matrix_reg_roundtrip_probe.sh
```

