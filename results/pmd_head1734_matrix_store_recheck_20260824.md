# HEAD1734 Matrix-Store Recheck

> **SUPERSEDED (2026-08-24):** The 32x16/64x16 failures below used reversed
> row-major shapes and stride16. Corrected 16x32/stride32 and
> 16x64/stride64 direct paths pass exactly. See
> `results/matrix_store_shape_correction_20260824.md`.

Date: 2026-08-24

Historical decision: `HEAD1734_RUNNABLE / INVALID_32X16_64X16_CONTRACT`.

## Question

Does the 2026-08-24 PMD core package fix the isolated LDS-source B16
matrix-store failures, and is the failure common to every matrix-store shape?

## Package And Installation

```text
URL: http://172.19.22.214/files/core/pmd.tar.gz
Last-Modified: Mon, 24 Aug 2026 03:40:46 GMT
tar SHA256: 06376f031f90420e5e2ab7be850dc30a0f1cef710d56cc0cb30026b1b7128fc1
runtime: CoreArch:HEAD_1734(lib_ini_opt)
gem5.opt SHA256: d2fb7078b28da9ecdb1f6ebaa3163befaed3ed9512b3721898737ec451cbe00f
lib SHA256: f1dc80e9358ace4640ec69dbb6ff95e57c943489bf157844d02b0337d81b3cc3
install root: /zys/shaobo/toolchains/pmd_20260824
```

The existing HEAD1694 installation was left untouched. The Shaobo C0 SOC is
the unchanged audited package from `/zys/shaobo/toolchains/pmd_20260717/soc`.

The first sidecar launch incorrectly resolved `/opt/rocm/bin/gem5` and produced
the misleading `GPUDispatcher.cp_prefetch` error. `strace -f -e execve` exposed
the wrong executable. Adding the split-package symlinks and prefixing the new
`bin` in `PATH` fixed generation without source or config-schema patches:

```text
bin/gem5        -> ../core/gem5.opt
bin/aqlReplayer -> ../core/aqlReplayer
lib/libgem5.so  -> ../core/libgem5_opt.so
bin/gem5_soc    -> ../soc/gem5.opt
```

The coherent generator reports `CoreArch:HEAD:1734(exec_ini_opt)` and runtime
reports `CoreArch:HEAD_1734(lib_ini_opt)`. Generated config:

```text
/zys/sb/ms1734_coherent/mode0/m5out/config.ini
SHA256 d94d10bb7d01a322ec073c2d32f08f8913626995058f33d656f5dc3eda2a29af
```

## Completion-Policy Sweep

The unchanged direct 32x16 probe was run under modes `0,2,6,7,8,9`:

```text
mode,status,mismatches,poison,first_row,first_col,pass
0,0,240,240,17,0,0
2,0,240,240,17,0,0
6,0,240,240,17,0,0
7,0,240,240,17,0,0
8,0,240,240,17,0,0
9,0,240,240,17,0,0
```

Wait, ABarrier, and cache-policy variants do not change the fixed truncation.

## Shape-Family Sweep

```text
mode,status,shape,mismatches,poison,first_row,first_col,pass
0,0,32x16,240,240,17,0,0
1,0,64x16,1024,992,0,0,0
2,0,32x32,0,0,-1,-1,1
```

The direct 32x32 chain is exact:

```text
A-global[32x32]
  -> matrix_load_32x32_b16
  -> swizzled LDS
  -> matrix_store_32x32_b16
  -> A1-global bitwise oracle
```

Run root: `/zys/sb/ms1734_sweep_20260824`.

The unchanged HEAD1694 control still reproduces the historical 32x16
signature at `/zys/sb/ms1694_recheck_20260824/mode0`.

## Interpretation

- HEAD1734 is runnable; the earlier config error was a mixed-PATH sidecar
  installation mistake, not a package ABI defect.
- B16 matrix-store is not universally broken in PMD: direct 32x32 passes.
- Direct 32x16 and 64x16 remain incorrect, so the unresolved boundary is
  shape-specific PMD support or an undocumented source-layout/descriptor ABI.
- Writer-only `ds_write_matrix_format_f16` remains independently proven to
  preserve all 512 input labels in LDS on HEAD1694.
- The passing 32x32 direct control does not prove the desired dKV epilogue.
  Before canonical integration, separately validate:
  `FP32 accumulator -> FP16 pack -> ds_write_matrix -> LDS ->
  matrix_store_32x32_b16`.

No canonical FA kernel was changed by this recheck.

## 32x32 MMAC Source Follow-up

The required follow-up was completed at
`/zys/sb/matrix_store_32x32_mmac_probe/run_20260824_163529`.

- Dense FP16-output MMAC has one exact native chain:
  `lit0/lts0 -> trans writer -> adjacent-N concat -> matrix_store_32x32`.
- All 128 candidates commit all 1,024 values, so writer/store transport is
  complete.
- None of the 64 FP32-accumulator lane-local pack candidates is exact. The
  production-like `lit1/lts0` path is a source-slot mismatch, not data loss.
- Native `__builtin_hcu_cvt_pk_f16_f32` concat/interleave has the same
  mismatch counts as scalar conversion. It is not a source-layout transform.
- Static resources are SGPR24/VGPR40, private/spill0, bank0, with no scalar
  matrix reads or permutation instructions.

Detailed evidence: `results/matrix_store_32x32_mmac_source_20260824.md`.

## Two-Wave 32x32 Follow-up

The accepted FP16 tuple also passes when two waves cooperatively own one
32x32 LDS result page. Wave0 and wave1 publish separate 16x32 row halves;
wave0 then issues the 32x32 matrix store after a CTA synchronization.

```text
run: /zys/sb/matrix_store_32x32_two_wave_probe/run_20260824_161843
result: 0/1024 mismatch, max_abs=0, bank conflict=0
resource: SGPR20, VGPR16, private/spill=0
```

Current compiler and `llvm-mc` both reject an F16 writer encoded as
`element:2 row:1 col:2`, including its transpose form. The available writer
shape remains `element:2 row:2 col:1`.
