# Five-GEMM Native dS Gate

Date: 2026-07-22

Status: `NATIVE_D32_D128_CHAIN_PASS / REAL_FA_CORRECTNESS_PASS / PERF_PENDING`

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

## Historical Search Space

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

## Dense Promotion Requirements

Any native tuple must be promoted with:

1. Actual MLS-loaded, non-symmetric Q and K data.
2. A CPU coordinate oracle for a fixed canonical `M16xN32xD32` product.
3. One mathematically verified upstream score/dS orientation.
4. The actual MMAC output ABI and source-slot-local dS arithmetic followed by
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

The D32 and D128 ABI/numeric gates now pass, but production remains behind
real softmax/causal and full-FA correctness gates. A wrong-layout path,
`ds_mpermute`, `bpermute`,
gather, ordinary `ds_read_b32`, duplicate score/dP, or a second layout-
conversion kernel remains forbidden.

## Superseded Dense Differential Result

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

The result above was produced by an invalid differential probe and is retained
only as history. Do not use it to classify the native chain. The corrected
result below supersedes it.

## Superseded FP16-Output MMAC Recheck

The HCU FP16-output MMAC initially produced an apparent source-slot match on a
sparse coordinate-label input. A dense non-symmetric rerun rejects that
promotion:

```text
FP16 MMAC LTS0 -> writer(t=1,offset=0) -> trans/normal readers
  tensor mismatches = 496/512 and 496/512
  dQ/dK mismatches  = 493 and 911

FP16 MMAC LTS1 -> writer(t=1,offset=0) -> trans/normal readers
  tensor mismatches = 488/512 and 488/512
  dQ/dK mismatches  = 488 and 882
```

Runs:

- `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_232115`
- `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_232230`

These runs were invalid for two independent reasons: the reader pointer already
named the dS page but the builtin also received byte offset 16, and the dQ
control paired the N-half and D-half fragments incorrectly. They are negative
probe-development evidence, not instruction evidence.

## Corrected Dense Native dS Result

The corrected probe uses this exact native chain:

```text
FP16-output score MMAC + FP16-output dP MMAC
  -> source-slot-local dS VALU (no lane movement)
  -> ds_write_matrix_format_f16(t=1, alt0, offset=0)
  -> same LDS page
     -> trans-m32-alt0 reader -> dQ MMAC
     -> normal-m32-alt0 reader -> dK MMAC
```

The source-slot-local dS code uses the inverse of the measured writer-to-trans
reader bit permutation only to select the logical q/k sidecar coordinate. It
does not move register values between lanes or words.

Locked PMD result:

```text
run /zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260723_000722
score/dP/dS             max_abs=0, PASS
trans/normal tensor     max_abs=0, PASS
dQ/dK downstream MMAC   max_abs=0, PASS
metadata                SGPR44/VGPR68, private/spill=0
LDS bank conflict       0
ASM                     FP16 MMAC=8, scalar matrix read=0, permute=0
```

An earlier corrected score-only control also passes both consumers:
`/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_235908`.

The same probe now streams four D32 slices for D128. Locked run
`/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260723_002608` has exact
score/dP/dS controls, both reader tensors and downstream dQ/dK. FP16-output
accumulation versus the FP32 oracle is `max_abs=0, rel_l2=0`. Metadata is
`SGPR40/VGPR53`, private/spill `0`, bank conflict `0`; ASM has 16 MLS, 32
FP16-output MMAC, no scalar matrix read and no permutation instruction. D32
regression also remains exact at `dq_native_ds_dense_20260723_002657`.

The native D128 handoff is now integrated into the single canonical fused
kernel. Real softmax/LSE/delta semantics pass for causal H1/S128 and H1/S1024,
and non-causal H1/S128. All three outputs pass with relative-L2 below
`7.9e-4`; metadata is SGPR74/VGPR168 with branch usage 8/123/123 and no
private segment or spill. Performance remains pending: the initial path uses
FP32 atomic partial stores for all three outputs and is a correctness baseline,
not a promoted schedule.
