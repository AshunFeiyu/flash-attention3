# FWD-Native MMAC Writer Chain

Date: 2026-08-24

Decision: `ACCEPT_FWD_LAYOUT_CONTROL / REJECT_NATURAL_FP32_PACK_TO_WRITER`.

## Contract

Reproduce the exact FWD layout lifecycle before testing the B16 matrix-store
epilogue:

```text
Q/K/V global
  -> matrix_load_32x32_b16 BPS+LDS
  -> FWD normal/trans ds_read_matrix operands
  -> QK^T FP32 MMAC lit1/lts0
  -> FWD internal cvt_pk score-to-P
  -> PV FP32 MMAC lit1/lts0
  -> FWD output cvt_pk across the two output-D halves
  -> direct global control
  -> ds_write_matrix_format_f16
  -> matrix_store_32x16_b16
```

The dense integer inputs make both FP32 reductions and FP16 casts exact. The
CPU oracle covers the complete `16x32x32` QK^T followed by `16x32x32` PV
chain, rather than a coordinate tag or a single GEMM.

## Evidence

- Source: `probes/fwd_native_mmac_writer_chain_probe.cpp`
- Runner: `scripts/run_fwd_native_mmac_writer_chain_probe.sh`
- Source commit: `a3e1533`
- PMD run: `/zys/sb/fwd_native_mmac_writer_chain_probe/run_20260824_183413`
- Local result: `artifacts/fwd_native_mmac_writer_chain_20260824/result.txt`
- Compiler: LLVM `e0f10535a0d681bcf3885ea2c398cc494bf6e332`
- PMD: `CoreArch:HEAD_1734`, `GPU_CHIP=sb`, `SQCIPfLines=7`

Static gate:

```text
SGPR=22, VGPR=33, private/spill=0
MLS=3, ds_read_matrix=5, MMAC=8, cvt_pk=8
ds_write_matrix=2, matrix_store_32x16=4
ordinary ds_read=0, permute=0, permlane=0
ldsBankConflict=0
```

Semantic result:

```text
direct FWD pack: 8/8 exact, no poison
normal writer x four store T/R modes: 408/512 mismatch each
trans writer x four store T/R modes: 384/512 mismatch each
matrix-store paths: 0/8 exact, no poison
```

Changing the matrix-store `T/R` view does not change either mismatch count.
The store receives a complete LDS page in every case.

## Interpretation

- The exact FWD MLS/reader/MMAC/pack chain is now a positive dense oracle.
- Correct element stride, matrix-store completion, cache policy, and store
  `T/R` selection are no longer plausible causes of this mismatch.
- FWD's natural lane-local FP32-to-FP16 output pack is not the input register
  ABI expected by the supported F16 matrix writer.
- This is consistent with the earlier one-GEMM FP32 source-slot sweep and the
  corrected dKV packed-B16 writer result. It is not evidence that PMD loses
  values or that matrix store is broken.

The result does not prove that no native compiler/ISA solution exists. It
only rejects the tested FWD pack with the supported `row=2,col=1` normal/trans
writers and all four matrix-store views. A future retry needs a documented
FP32-C-to-F16-writer conversion, a different native C-fragment mode, or a new
compiler-provided builtin contract. Do not add a hot-path gather or lane
permute to force this epilogue into dK/dV.
