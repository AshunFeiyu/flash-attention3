# 32x32 MMAC to Matrix Store Probe

Date: 2026-08-24

Decision: `ACCEPT_32X32_FP16_NATIVE_CHAIN / FP32_SOURCE_ABI_OPEN`.

## Contract

Validate the native 32x32 B16 epilogue without `ds_read_b32`, lane permute,
gather, or a software transpose:

```text
dense 32x32x32 GEMM
  -> MMAC result registers
  -> ds_write_matrix_format_f16
  -> swizzled LDS
  -> matrix_store_32x32_b16
  -> row-major FP16 output
```

The probe covers FP32-output and FP16-output MMAC, all four LIT/LTS modes,
normal/trans writers, adjacent-N/adjacent-M fragment pairs, and concat/FWD
interleave packing. Inputs are dense asymmetric integers, so both FP32 and
FP16 accumulation have an exact CPU oracle.

## Evidence

- Source: `probes/dkv_b16_matrix_store_32x32_mmac_probe.cpp`
- Runner: `scripts/run_dkv_b16_matrix_store_32x32_mmac_probe.sh`
- PMD run: `/zys/sb/matrix_store_32x32_mmac_probe/run_20260824_162517`
- Local archive: `artifacts/matrix_store_32x32_probe_20260824`
- Compiler LLVM: `e0f10535a0d681bcf3885ea2c398cc494bf6e332`
- PMD: `CoreArch:HEAD_1734`, `GPU_CHIP=sb`, `SQCIPfLines=7`

Static gate:

```text
private=0, SGPR=24, VGPR=40, spill=0
matrix_load=2, ds_read_matrix=4, ds_write_matrix=16
MMAC-f32=32, MMAC-f16=32, matrix_store_32x32=1
ordinary ds_read=0, permute=0, permlane=0
ldsBankConflict=0
```

Semantic result:

```text
FP16 MMAC lit0/lts0
  -> trans writer
  -> adjacent-N concat
  -> matrix_store_32x32_b16
  -> 0/1024 mismatch
```

This is the only exact native tuple in the 64-candidate sweep. All 64 paths
commit all 1,024 elements; no output remains poisoned.

All 32 FP32-accumulator paths are non-exact. The production-like
`lit1/lts0 -> trans writer -> adjacent-N concat` path has 765/1024
mismatches. Therefore a lane-local FP32-to-FP16 cast plus the tested natural
pair/pack choices does not produce the writer source-slot ABI.

The earlier run at `run_20260824_155916` is invalid: its input formulas made
all A rows and all B rows identical, so permutations were invisible. It is
excluded from evidence.

## Interpretation

- HEAD1734 supports the complete 32x32 writer/store transport. This is not a
  general PMD value-loss problem.
- FP16-output MMAC has a native no-permute 32x32 store path.
- dK/dV still require FP32 accumulation over `seqlen_q`. Their current
  lane-local downcast does not match the B16 writer source ownership.
- Do not integrate the 32x32 epilogue into canonical dK/dV until a native
  FP32-C-to-B16-writer contract is found. The accepted direct-global FP16
  store remains the production control.

Next: inspect compiler HCU tests for an intended FP32 accumulator pack or a
different B16 writer shape. If none exists, send this source and result to the
compiler/PMD owners as an ABI question rather than reporting a PMD failure.

## Writer Shape Boundary

The natural next candidate was an F16 writer with `row=1,col=2`, matching a
logical 16x32 fragment. It is unavailable in the locked gfx946 toolchain:

```text
__builtin_hcu_ds_write_matrix_format_f16(..., row=1, col=2, ...)
  -> clang: unsupported ds matrix format modifiers' combination

ds_write_matrix_format v[0:3], s0 element:2 row:1 col:2
  -> llvm-mc: unsupport ds write matrix format on this GPU
```

Normal and transpose forms fail at both levels. The supported control
`element:2 row:2 col:1` assembles successfully. The GFX946 HCU tests contain
16x32 reader forms but no matching F16 writer example. Do not carry
`row=1,col=2` into operator code under this compiler.

## Two-Wave Ownership Check

A minimal two-wave probe validates the accepted FP16 tuple under cooperative
ownership:

```text
wave0: produce A/B; compute and publish output rows 0..15
wave1:            compute and publish output rows 16..31
wave0: matrix_store_32x32_b16 the shared result page
```

- Source: `probes/dkv_b16_matrix_store_32x32_two_wave_probe.cpp`
- Runner: `scripts/run_dkv_b16_matrix_store_32x32_two_wave_probe.sh`
- PMD run: `/zys/sb/matrix_store_32x32_two_wave_probe/run_20260824_161843`
- Result: `mismatches=0/1024`, `max_abs=0`, `ldsBankConflict=0`
- Resources: SGPR20, VGPR16, private/spill0
- ISA: MLS2, matrix-read4, FP16-MMAC8, writer2, matrix-store1;
  ordinary DS read and permute0

This closes the A4 resource/lifecycle gate for the FP16 32x32 chain. It does
not close the FP32-to-B16 source-slot contract required by long dK/dV
accumulation.
