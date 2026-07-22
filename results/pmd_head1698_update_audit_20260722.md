# PMD HEAD1698 Update Audit

Date: 2026-07-22

Decision: `REJECT_TOOLCHAIN_PROMOTION / CONFIRMED_PACKAGE_COMPATIBILITY`.

## What Was Tested

- Downloaded the current fixed core package from
  `http://172.19.22.214/files/core/pmd.tar.gz` through `hedr-common`.
- Extracted it side by side at `/zys/shaobo/toolchains/pmd_20260722`.
- Reused the unchanged Shaobo C0 SOC because its HTTP fingerprint and
  `gem5.opt` SHA256 are unchanged.
- Kept compiler, probe binary, `GPU_CHIP=sb`, SQ7, and the HEAD1694 config seed
  fixed while attempting the update.
- Re-ran `scripts/run_matrix_global_roundtrip_probe.sh` on locked HEAD1694 as
  the control.

## Fingerprints

| Component | Identity |
| --- | --- |
| Core package | Last-Modified `2026-07-21 09:35:48 GMT`, SHA256 `fef22f48080e48f893fb66d81736c11a42453a50d12d0ea79901821b10ddf470` |
| Core readme | commit `029b9d17d7f0e1078fea6e9e5d8aebc7f6d95bcc` |
| Runtime banner | `CoreArch:HEAD_1698(lib_ini_opt)`, compiled `2026-07-21 09:34:22` |
| Core `gem5.opt` | SHA256 `a0f1b681ac8c1271731465ce1bae209d675a7a176be77923350a5a6ff2968b5c` |
| Core library | SHA256 `aca312fd740607d3eb80750bd1557527c682109cc1e7dfd37bc07e7a31dc6933` |
| C0 SOC | SHA256 `d0c03538753a4b91c2aa3e110cb12f1302b66c891c3ab2d446c85de99fe24524` |

## Failure Chain

1. Official HEAD1698 config generation fails because packaged `gpu.py` passes
   `GPUDispatcher.cp_prefetch`, which packaged `gem5.opt` does not expose.
2. Falling back to the locked HEAD1694 seed fails because the HEAD1698 library
   requires new CP-prefetch topology and config fields.
3. An isolated, non-production copy removed only the generator-incompatible
   fields. Its config-generator banner is `HEAD:1668(exec_ini_opt)`, proving
   the core package combines a 1668 config generator with a 1698 runtime.
4. The generated config is then rejected by HEAD1698 at
   `TcaHoleArbAge`; further hand-editing would no longer be trustworthy.

The official package never dispatches the probe, so it cannot answer whether
PMD-005 is fixed.

## Control

Locked HEAD1694 was rebuilt and rerun unchanged:

```text
run: /zys/sb/fa3b/layout_probes/matrix_global_roundtrip_20260722_211218
transport: PASS
panic: 0
ldsBankConflict: 0
exact chain/direct pairs: 0/0
direct mismatch: 240/512, first row17 col0
```

This confirms the project environment was not damaged by the side-by-side
audit and reproduces PMD-005 exactly.

## Action

- Keep `scripts/toolchain_lock.sh` on audited HEAD1694.
- Preserve official HEAD1698 sidecar for provider inspection, but do not use it
  for correctness, performance, or SQTT evidence.
- Re-run the unchanged global roundtrip probe only after receiving a matched
  HEAD1698 package/config seed.
