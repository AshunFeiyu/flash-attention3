# Five-GEMM Native dS Gate

Date: 2026-07-22

Status: `CLOSED_FAIL_CURRENT_TOOLCHAIN`

## Why This Gate Exists

The fused backward algorithm has exactly five GEMMs:

```text
score = Q @ K^T
dP    = dO @ V^T
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

`score` and `dP` must be computed once.  The same `P/dS` values must feed both
the transposed dKV GEMMs and the non-transposed dQ GEMM.  A correct five-GEMM
kernel therefore needs a native ownership conversion or a natural fragment
layout accepted by both sides.

## Closed Search Space

The prior branch already tested the direct register surface across operand
order, LIT/LTS and output maps, and tested the f16 matrix writer/reader surface.
No natural mode completed all five GEMMs.  Some modes make dV/dK correct while
others make dQ correct; none make both correct without a layout bridge.

Current e0f10535 rechecks show:

- `dq_source_slot_coordinate_probe`: no direct source-slot match in the tested
  normal/trans reader and LIT/LTS combinations.
- adjacent-M writer/reader sweep: no exact match; best candidate still has 448
  downstream mismatches.
- f32 `ds_write_matrix_format` remains unavailable in PMD.
- `DS_MATRIX_TRANSPOSE_4V` is documented but has no current compiler/PMD
  entry; see `docs/perf_model_pmd_compiler_issues.md` COMP-004.

Do not repeat broad LIT/LTS, reader, writer or store-map sweeps.  They are not
new architectural hypotheses.

## Final Differential Probe

One final probe is admissible before closing this gate.  It must use:

1. Actual MLS-loaded, non-symmetric Q and K data.
2. A CPU coordinate oracle for a fixed canonical `M16xN32xD32` product.
3. One mathematically verified upstream score/dS orientation.
4. FP32-to-FP16 conversion followed by
   `ds_write_matrix_format_f16(t=1,alt0)`.
5. Both downstream consumers from the same published fragment:
   transposed read for dQ and normal read for dK.
6. Full-coordinate comparison for both dQ and dK, not only a synthetic MMAC
   fingerprint.
7. No scalar matrix LDS read, permute, gather or manual source-slot scatter.

Pass requires both dQ and dK exact within FP16/FP32 tolerance, bank conflict
zero, and no private segment or spill.  Failure closes the native f16 writer
route for the current toolchain.

## Canonical Consequence

Until the final differential probe passes, the production five-GEMM kernel is
not allowed to use a wrong-layout path, `ds_mpermute`, `bpermute`, gather,
ordinary `ds_read_b32`, duplicate score/dP, or a second layout-conversion
kernel.  The top-level workbook may continue, but code implementation remains
behind this hard gate.

## Final Dense Differential Result

The final probe is `probes/dq_native_ds_write_dense_probe.cpp`.  It uses
non-symmetric MLS-loaded `Q/K/V/dO`, verifies the upstream `score`, `dP` and
FP16 `dS` coordinates against a CPU oracle, then sends the same natural dS
fragment through two paths:

```text
direct dS fragment -> dQ/dK MMAC
dS fragment -> f16 ds_write_matrix(t=1,alt0) -> trans/normal reader -> dQ/dK MMAC
```

Locked environment:

```text
compiler LLVM e0f10535, clang SHA256 334cb561...
PMD HEAD1694, GPU_CHIP=sb, SQCIPfLines=7
run /zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_192016
```

Result:

```text
score                 max_abs=0          PASS
dP                    max_abs=0          PASS
dS                    max_abs=0          PASS
dK direct             max_abs=0          PASS
dQ direct             max_abs=0.0597854  FAIL, 458 mismatches
dQ after trans read   max_abs=0.0586505  FAIL, 462 mismatches
dK after normal read  max_abs=0.0304303  FAIL, 887 mismatches
metadata              SGPR40/VGPR48, private/spill=0
LDS bank conflict     0
```

This differential removes the output-store-map ambiguity: the direct dK path
uses the same store coordinates and is exact.  The natural score-owned dS
fragment is dK-compatible but not dQ-compatible.  The tested native f16
writer/read pair does not convert it into a dQ-compatible view and also loses
the original dK-compatible view.  Together with the previously closed direct
64-mode and writer/read 80-mode surfaces, the current compiler/PMD contract
does not expose a legal no-permute ownership switch for all five GEMMs.

The later ABI-calibrated register probe narrows this conclusion. The matrix
writer and matching m32 readers are lossless and use a compatible LDS
swizzle: all 12 pairs are complete permutations, and CPU inverse-packing of
the writer source makes all 12 pairs strict identities. Therefore the blocker
is not DS writer/reader transport. It is that the natural score-owned dS MMAC
output is not already in the writer source-slot ABI required by the desired
dQ and dK consumer views. CPU inverse-packing proves existence but is not a
legal runtime implementation.

Do not start the fused canonical kernel on this toolchain.  Re-open the gate
only when one of these is available and passes this same dense oracle:

1. the documented final builtin/encoding plus PMD support for
   `DS_MATRIX_TRANSPOSE_4V`;
2. a compiler-owner-provided `ds_write_matrix` source-layout contract that
   serves both downstream views;
3. another documented native MMAC/writer/reader combination outside the
   already closed mode surfaces.
