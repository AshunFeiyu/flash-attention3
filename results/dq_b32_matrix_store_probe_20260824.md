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
- The corrected native FP32 DS writer emits the documented
  `element:3,row:1,col:1,offset:0` form with no real `s_trap`. Both PMD
  HEAD1694 and HEAD1734 stop at
  `Invalid opcode encountered: 0xd38b5008`, before a semantic result.

## Conclusion

`matrix_store_16x16_b32`, its descriptor stride, and the direct VGPR form are
not broken by this evidence. The open issue is the producer register ABI: the
current dQ MMAC C fragment is not arranged as the B32 matrix-store source
fragment expected by the tested native forms. The ISA-documented
`DS_MATRIX_TRANSPOSE_4V` or another compiler-provided output-layout contract
must be probed before production integration; do not call this an instruction
failure and do not add scalar gather/permute code.

The production path was not modified. It currently stores each dQ partial as
FP16 before the reduction, despite older notes calling it FP32. Preserving dQ
partial precision is a separate change: first establish an FP32 direct-store
workspace baseline, then compare any native B32 epilogue against it.

## Reproduction

- Source: `probes/dq_b32_matrix_store_probe.cpp`
- Runner: `scripts/run_dq_b32_matrix_store_probe.sh`
- Direct VGPR D128/mfmt sweep:
  `/zys/sb/dq_b32_vgpr_store_mfmt/layout_probes/dq_b32_matrix_store_20260825_000145`
- Corrected FP32 writer run:
  `/zys/sb/dq_f32_writer_offset0_test/layout_probes/dkv_pds_f32_roundtrip_probe_20260824_234613`
- HEAD1734 no-WDRA cross-check:
  `/zys/sb/f32pmd1734/run_nowa_20260824_224337`

The runner intentionally returns failure until at least one direct source ABI
combination passes.
