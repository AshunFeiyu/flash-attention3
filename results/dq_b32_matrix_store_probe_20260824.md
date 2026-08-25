# dQ FP32 B32 Matrix-Store Probe

Status: `OBSERVE_SOURCE_FRAGMENT_ABI_OPEN`.

The earlier `REJECT_DIRECT_CANONICAL_LAYOUT` result is superseded. Its first
oracle manually constructed a lane-linear source fragment and the historical
FP32 roundtrip runner contained three independent errors: page-relative DS
offset `16` instead of `0`, a comment counted as a real `s_trap`, and WDRA
flags on a one-wave probe.

## Question

Can the canonical FP32 dQ MMAC accumulator be stored without downcast or
lane-wise global stores through either native B32 path?

```text
MMAC C -> matrix_store_16x16_b32 VGPR -> global
MMAC C -> ds_write_matrix_f32/u32 -> LDS -> matrix_store_16x16_b32
```

## Corrected Evidence

- Compiler: `e0f10535`.
- PMD: `CoreArch:HEAD_1694`; native FP32 writer was also rechecked on
  `CoreArch:HEAD_1734`.
- ISA defines the matrix resource stride in matrix elements. The production
  dQ test therefore uses `stride=128`, not `16` or bytes.
- Direct VGPR B32 matrix store compiles and executes with SGPR14/VGPR20,
  private/spill/scratch0, no DS fallback and no permute.
- The exact D128 test leaves every padding sentinel intact (`guard=0`), so the
  failure is not an out-of-bounds or row-stride error.
- LIT/LTS, store T/R and descriptor `mfmt={0,1,2}` were crossed for 48 modes.
  All execute, but none matches the current direct dQ row/component ownership;
  every mode reports 252/256 dense mismatches.
- Raw code-object disassembly proves the earlier `0xd38b5008` failure was
  `v_pk_sub_u16` in the probe's modulo-based test-data generator, not a DS
  instruction. The actual first FP32 writer encodes as
  `D9DE0000 08010309`.
- After replacing that unrelated setup arithmetic, PMD HEAD1694 executes all
  four FP32 writers and eight FP32 readers. All four writer/read combinations
  are complete 256-slot permutations, with no panic, invalid opcode, spill,
  scratch or LDS bank conflict.
- None of the four combinations is lane-linear identity and the current
  direct-fragment semantic comparison reports `0/4`. That oracle does not yet
  reconstruct the measured slot permutation, so it leaves the MMAC-C source
  ABI open; it is not evidence that the writer or PMD is broken.

## Conclusion

`matrix_store_16x16_b32`, its descriptor stride, and the direct VGPR form are
not broken by this evidence. The open issue is the producer register ABI: the
current dQ MMAC C fragment is not arranged as the B32 matrix-store source
fragment expected by the tested direct-store forms. The next probe must
reconstruct the measured writer/read permutation and validate the downstream
dense MMAC. `DS_MATRIX_TRANSPOSE_4V` or another compiler-provided output
contract is only needed if that native chain still cannot satisfy the consumer
ABI; do not add scalar gather/permute code first.

The production path was not modified. It currently stores each dQ partial as
FP16 before the reduction, despite older notes calling it FP32. Preserving dQ
partial precision is a separate change: first establish an FP32 direct-store
workspace baseline, then compare any native B32 epilogue against it.

## Reproduction

- Source: `probes/dq_b32_matrix_store_probe.cpp`
- Runner: `scripts/run_dq_b32_matrix_store_probe.sh`
- Device ISA extractor: `scripts/extract_device_isa.sh`
- Direct VGPR D128/mfmt sweep:
  `/zys/sb/dq_b32_vgpr_store_mfmt/layout_probes/dq_b32_matrix_store_20260825_000145`
- Corrected FP32 writer run:
  `/zys/sb/dq_f32_writer_opcode_clean/layout_probes/dkv_pds_f32_roundtrip_probe_20260825_104710`
- Superseded pre-writer failures:
  `/zys/sb/dq_f32_writer_offset0_test/layout_probes/dkv_pds_f32_roundtrip_probe_20260824_234613`
  and `/zys/sb/f32pmd1734/run_nowa_20260824_224337`

The runner intentionally returns failure until at least one direct source ABI
combination passes.
