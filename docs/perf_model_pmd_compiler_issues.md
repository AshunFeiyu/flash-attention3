# Shaobo Perf Model PMD And Compiler Issue Registry

Last updated: 2026-07-20

This document tracks possible PMD, compiler, and compiler/PMD compatibility
issues found while validating Shaobo kernels. It is an issue registry, not a
kernel experiment log. Performance experiments remain in
`results/perf_ledger.csv` and `results/optimization_log.md`.

## Classification Rules

Use these states:

- `CONFIRMED`: isolated by a focused reproducer and assigned to one component.
- `SUSPECTED`: repeatable, but PMD behavior and an undocumented ABI constraint
  cannot yet be distinguished.
- `COMPATIBILITY`: each component may work separately, but the selected
  compiler and PMD versions do not work together.
- `ENVIRONMENT`: host runtime, profiler helper, path, or launch infrastructure.
- `DEFERRED`: evidence is preserved, but the required PMD/compiler capability
  is unavailable.
- `RESOLVED`: a validated version or invocation has removed the failure.

Do not label a failure as a PMD or compiler bug solely because static metadata
passes. First preserve final ASM, build and PMD fingerprints, reduce the case,
and compare against a known-good artifact or compiler.

## Current Validated Fingerprint

Current baseline environment on 2026-07-17:

```text
host/container: liuchang / zys1
repo: /zys/shaobo/fa3_bwd_wasp_7gemm_tomography_20260716
GPU_CHIP: sb
compiler: clang 18.0.0, llvm a6a6eb6616abdd98b6dd72074afad281b47c8c6a
compiler path: /opt/rocm-6.3.3/llvm/bin/clang++
PMD layout: /opt/rocm-6.3.3/pmd, flat layout
PMD run.py timestamp: 2026-06-16 11:00:48 +0800
PMD run.py sha256: 9c39779bfe9db7e0a1d94d9a13926c3e10301c873e8e4a3184e52179a80ce14a
PMD libgem5_opt.so sha256: b77e48f1e4fc4b0112e93022180772cd8d1113501897607c9ff38fdd1ed0628f
PMD gem5.opt sha256: b517f4b384a8641549f385a75c3f6677b353bec95783f520fe2fadf2e0045957
```

Refresh this block after every PMD or compiler update. Never compare compiler
or PMD behavior across fingerprints without saying so explicitly.

### Current Side-By-Side HEAD1694 Launch Contract

The side-by-side PMD root is `/zys/shaobo/toolchains/pmd_20260717`.  For a
split installation, `ROCM_PATH` during PMD execution must be that root, not
the compiler/runtime root `/opt/rocm-6.3.3`.  Its `lib/libgem5.so` symlink
points to the matching `core/libgem5_opt.so` SHA256
`29fa2020e6bfb399225e206cf7c589ba838ad56b891cb07c97e88029e954bfa5`.

If `run.py` uses new HEAD1694 configs but loads
`/opt/rocm-6.3.3/lib/libgem5.so` from HEAD1668, config generation fails at
`ASTCA has no parameter num_phase`.  This is an environment mix, not a kernel
or instruction failure.  Use this runtime ordering:

```bash
export ROCM_PATH=/zys/shaobo/toolchains/pmd_20260717
export PMD_PATH=${ROCM_PATH}/core
export SOC_PATH=${ROCM_PATH}/soc
export LD_LIBRARY_PATH=${SOC_PATH}/libs
export RPY_LIB_PATH=${ROCM_PATH}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs
```

HEAD1694 currently still needs a known-good `config.ini` seed when its config
generator hits the `num_phase` mismatch; the model can then consume the seed
and run the target binary with the matching HEAD1694 library.  Evidence:
`/zys/shaobo_runs/dkv_owner16_4c_resource_probe_20260720_045244`.

### ENV-003: Fullperf Run-Root Path-Length Overflow

Status: `ENVIRONMENT`; short-path workaround validated.

Two M48 helper/fullperf launches under long roots beginning with
`/zys/shaobo_runs/dkv_m48_head_lookahead_s768_fullperf...` abort before the
kernel with `*** buffer overflow detected ***` in `__sprintf_chk` from the
`hsaKmtOpenKFD` path. The same binary, PMD, compiler, flags, and seeded config
run successfully under `/zys/sb/m48fp`, producing correctness PASS, stats and
a valid `.perf`.

Interpret this as a PMD/helper fixed-path-buffer boundary, not a kernel or
Shaobo instruction failure. Use a short `${SHAOBO_RUN_ROOT}` such as
`/zys/sb/<case>` for fullperf capture, then archive and rename the completed
artifacts. Preserve the binary and toolchain fingerprints across the retry.

## PMD Issues

### PMD-001: F32 `ds_write_matrix_format` Is Not Decoded

Status: `DEFERRED`, component assignment pending PMD owner confirmation.

Symptom:

```text
fatal: Invalid opcode encountered: 0xd38b5007
```

The source builtin is:

```cpp
__builtin_hcu_ds_write_matrix_format_f32(
    value, ptr, 16, 1, 1, 0, transpose);
```

The first generated writer is:

```asm
ds_write_matrix_format v[1:4], s8 offset:16 element:3 row:1 col:1
```

`element:3` is the f32 form. The compiler also emits the transpose writer and
normal/transpose f32 readers, but PMD aborts at the first non-transpose writer,
before any reader or numerical oracle runs.

Evidence:

- Source: `probes/dkv_pds_f32_roundtrip_probe.cpp`.
- Runner: `scripts/run_dkv_pds_f32_roundtrip_probe.sh`.
- Remote run:
  `/zys/shaobo/runs/dkv_pds_f32_roundtrip_probe_20260716_215919`.
- Static result: `private=0`, `sgpr=18`, `vgpr=45`, no spill, scratch, or trap.

Current interpretation:

- The compiler accepts the builtin and emits a concrete instruction.
- The current PMD does not decode that encoded f32 writer variant.
- This does not prove that Shaobo hardware lacks the instruction.
- A compiler/PMD ISA-version mismatch remains possible; do not assign this
  solely to PMD until the owners compare the encoding tables.

Impact and workaround:

- Native f32 `MMAC -> ds_write_matrix -> ds_read_matrix -> MMAC` validation is
  blocked.
- Keep the probe isolated. Do not connect this path to the canonical FA kernel.
- Continue with the correct direct-register P/dS path until a supported PMD is
  available.

Owner question:

> Does the selected sb PMD implement the f32 `ds_write_matrix_format` encoding
> emitted as `element:3 row:1 col:1` / opcode `0xd38b5007`, and which compiler
> commit is expected to match that PMD?

### PMD-002: WDRA VGPR Init/Free Tracking Or Hidden Role-Exit ABI

Status: `SUSPECTED`.

Typical symptoms:

```text
vgpr16 is not init or has been freed
vgpr47 is not init or has been freed
vgpr80/vgpr81 is not init or has been freed
read vgprX before writing
```

Repeatable trigger families:

- a WDRA role exits before all other roles reach terminal convergence;
- terminal CTA convergence or ABarrier invalidation is removed;
- an operand wait is collapsed across a role-local VGPR readiness boundary;
- historical addtid/global-store probes combine role branches,
  `s_set_vgpr_size`, `ds_read_addtid_b32`, and a nearby vector/store path.

Evidence:

- `results/perf_ledger.csv`: `dkv_owner_teardown_wave0_wait`,
  `dq_owner_teardown_wave0_wait`, `dq_terminal_cleanup_removed`, and
  `dq_dqgemm_batch8_wait0`.
- Historical focused role-uniformity probe:
  `/Users/zhangyushun/soul/算子工程师/shaobo_instruction_lab/repros/`
  `wdra_role_uniform_store_probe_20260609`.
- Known-good FA3 FWD and GEMM binaries contain the same broad
  `addtid -> s_set_vgpr_size -> vector/MMAC/store` mechanism and pass PMD.

Current interpretation:

- `threadIdx.x & 63` versus `% 64` is not the root cause; both lower to the
  same `v_and_b32` in the tested compiler.
- Same-role uniform execution is good discipline, but the focused probe shows
  it is not sufficient to prevent the failure.
- The remaining ambiguity is incomplete PMD register-init/free state merging
  versus an undocumented WDRA role-exit/compiler ABI requirement.

Canonical workaround:

- Keep lane/index setup branch-local after role-local `s_set_vgpr_size`.
- Keep every role alive through `AllDone`/terminal convergence.
- Keep the proven terminal sync and ABarrier invalidation sequence.
- Preserve proven `lgkmcnt` readiness boundaries unless a focused probe passes.

Required closure evidence:

- A minimal binary pair differing only in role-exit/convergence behavior.
- Final ASM and metadata identity for the unaffected mainloop.
- PMD owner confirmation of VGPR lifetime semantics across WDRA branches.

### PMD-003: Fullperf Helper Can Abort Before Dispatch

Status: `ENVIRONMENT`.

Symptom:

```text
*** buffer overflow detected ***
```

It occurs in `libhsakmt`/fullperf-helper startup before the target kernel is
dispatched. It is not evidence of kernel correctness or performance failure.
Long nested output paths have made it more likely.

Workaround:

- Use a short run root such as `/zys/shaobo_runs/t3c_<time>`.
- Retry the identical binary before changing source.
- Use stats-only to continue functional work and mark SQTT as pending.
- Accept fullperf evidence only when the helper exits zero and the harness
  reports success.

### PMD-004: Foreground SSH Completion Is Not PMD Completion

Status: `ENVIRONMENT`.

Long CPU-simulated runs can outlive an SSH/command wrapper. A truncated client
session is not a PMD hang. Launch long S1024/fullperf runs detached, persist
`driver.log` and `exit_code`, and require both `exit_code=0` and harness
`status=success` before parsing the result.

### PMD-005: B16 Matrix Store Writes Only 17 Of 32 Rows

Status: `SUSPECTED PMD/UNDOCUMENTED CONTRACT`; canonical integration blocked.

Environment fingerprint:

```text
compiler: clang 18.0.0, llvm 7b796991375a79111716e29e6050bd719f46de94
clang++ sha256: c859dae0b4573361b728a558607ba9e0735d19540670f9443ecc8ea0335de0b0
PMD: pmd_20260717/core, HEAD1694 family
libgem5_opt.so sha256: 29fa2020e6bfb399225e206cf7c589ba838ad56b891cb07c97e88029e954bfa5
```

Minimal trigger:

- one active wave in a 16-wave CTA;
- one official `__builtin_hcu_matrix_store_32x16_b16` call with `t=1,r=0`;
- source page produced by matching `matrix_load_32x16_b16`;
- one ABarrier slot with the documented lifecycle:
  `init -> whole-CTA ebarrier -> seq -> matrix_store -> arrive -> try_wait ->
  whole-CTA ebarrier -> inv`;
- no WDRA flags, no `s_trap`, no spill/private segment, and zero LDS bank
  conflicts.

Expected behavior: all `32x16=512` FP16 values reach the row-major output.

Observed behavior: rows 0 through 16 are correct and rows 17 through 31 remain
zero, giving exactly 240 mismatches. The result is unchanged from the
multi-store probe, so consecutive stores overwriting one another and missing
ABarrier initialization are both ruled out. The separate
`ds_write_matrix_format_f16 -> matrix_store` path also fails, with 503
mismatches beginning at row 0.

Evidence:

- Source: `probes/dkv_b16_matrix_store_probe.cpp`.
- Runner: `scripts/run_dkv_b16_matrix_store_probe.sh`.
- Minimal PMD run:
  `/zys/shaobo_runs/dkv_b16_matrix_store_probe_builtin_single_reclass/`
  `run_20260719_103901`.
- Static ASM: `matrix_store_b16=2`, `abarrier_init/seq/arrive/wait/inv=
  1/2/2/2/1`, `ebarrier_sync=3`, `s_trap=0`.
- Runner verdict: `FAIL_MATRIX_STORE_CONTROL`, with PMD exit status 0.

Current interpretation:

- `s_abarrier_init` at kernel entry is required by the documented ABarrier
  contract, but it is not sufficient to make this matrix store correct.
- The remaining boundary is PMD's `matrix_store_32x16_b16` implementation
  versus an undocumented descriptor/source-layout requirement. Do not call it
  a hardware or compiler bug until the PMD/compiler owner confirms the ABI.
- Keep C2 (`FP16 pack -> ds_write_matrix -> matrix_store`) out of the canonical
  dKV epilogue. The C1 packed-FP16 direct-global focused control passes; use it
  as the next canonical performance A/B.

C1 known-good comparison:

- exact dKV owner16 coordinates, `16x128` FP16 output;
- 2,048/2,048 values correct, no spill/private segment, no trap, bank0;
- generated hot path contains eight `global_store_dwordx2`, zero
  `global_store_dwordx4`, and 32 FP32-to-FP16 conversions;
- run: `/zys/shaobo_runs/dkv_b16_direct_store_probe_builtin/`
  `run_20260719_104409`.

Owner question:

> With compiler llvm `7b796991`, what descriptor stride and LDS source-layout
> contract does PMD HEAD1694 require for `matrix_store_32x16_b16`? Why does a
> single official-builtin store with the documented ABarrier transaction
> lifecycle commit only the first 17 rows?

## Compiler And Compiler/PMD Issues

### COMP-001: WDRA Codegen Is Version-Coupled

Status: `COMPATIBILITY`, with validated workarounds.

Observed compiler behavior:

| Compiler | Source/flags | Generated WDRA form | PMD result |
|---|---|---|---|
| llvm `a6a6eb66` on liuchang | current clean FA source, no explicit `wdra_init` | role-local `s_set_vgpr_size`, no `s_trap` | PASS |
| llvm `279dd2bb` container default | same known-good FWD source | `s_trap 0xff`, zero-sized prologue resize | PMD FAIL |
| llvm `7940bbec` overlay | same FWD/FA source | role-local resize, no trap | PASS |
| llvm `0b52cc28` / 8435 GEMM | explicit `__builtin_hcu_wdra_init`, local-wave, `-run-on-model=true` | valid 24/72 resize, no trap | GEMM PASS only |

Rules:

- `__builtin_hcu_wdra_init(...)` is a versioned compiler contract, not a
  universally required builtin.
- Do not add or remove it without a same-source compiler A/B, ASM gate, and PMD
  correctness run.
- Reject a build before PMD if canonical WDRA code contains `s_trap` or loses
  role-local `s_set_vgpr_size`.
- Record `clang++ --version`, full flags, and ASM counts in every compiler
  comparison.

### COMP-002: Natural MMAC Output Does Not Match The Tested F16 Writer Source Slots

Status: `SUSPECTED ISA/compiler-contract gap`, not a confirmed compiler bug.

The f16 matrix writer/readers transport data correctly, including across WDRA
roles, but a non-degenerate oracle shows that natural score/dP MMAC output is
permuted relative to the writer source-slot ABI. Writer transpose flags,
normal/transpose readers, lane-local pack orders, operand order, and tested
LIT/LTS combinations do not recover semantic equivalence.

Do not report this as a broken `ds_write_matrix` implementation: transport is
correct. The unresolved question is which native MMAC output/layout and writer
format are intended to be paired. The f32 writer is the remaining no-permute
candidate, currently blocked by PMD-001.

### COMP-003: Read/MMAC Scheduling And Long-Lived Zero Codegen Quality

Status: `OBSERVE`, performance quality only.

Two recurring observations are not correctness bugs:

- source-level read batching does not guarantee useful ping-pong; an
  `8 reads -> wait -> MMAC` rewrite made ASM visually regular but increased
  exposed first-use wait and regressed fullperf;
- keeping one zero seed live across score/dP, softmax/dS, ABarrier, and output
  MMAC can replace explicit zero moves with additional copy moves and longer
  live ranges.

Always inspect generated ASM and SQTT. Escalate to the compiler owner only with
a minimal hot-island A/B proving identical math and operands but avoidable
instruction or scheduling debt.

## Do Not Misclassify These As PMD/Compiler Bugs

- `private_segment`, scratch, SGPR spill, or VGPR spill caused by an over-budget
  tile or long live range.
- Numerical mismatch caused by a guessed MLS/`ds_read_matrix`/MMAC layout.
- LDS bank conflict caused by an unverified address or writer layout.
- ABarrier deadlock caused by incorrect arrival counts, phases, or page reuse.
- Stats-only and fullperf variance without a repeated same-binary comparison.
- Lower MMAC active after removing valid causal work; the MMOP numerator also
  changed.
- PMD wall time. CPU simulation wall time is not simulated kernel time.

## New Issue Template

Add one entry rather than appending free-form history:

```text
ID / title:
Status:
First seen:
Environment fingerprint:
Minimal trigger:
Expected behavior:
Observed behavior:
Final ASM evidence:
Static metadata:
Correctness result:
PMD log/run root:
Known-good comparison:
Current attribution and confidence:
Workaround:
Owner question / next action:
```

An issue may be promoted to `CONFIRMED` only after the reproducer isolates the
component or the relevant owner confirms the contract.

## 2026-07-21 Rolling Perf-Model Compiler Snapshot

Status: `VALIDATED_SIDE_BY_SIDE`, not a global compiler promotion.

Source index:

```text
http://10.65.42.71/build2/package/perf_model_latest_6.3_ubuntu-22.04
Packages.gz Last-Modified: Tue, 21 Jul 2026 03:28:43 GMT
Packages.gz ETag: 6a5ee76b-4905
```

Installed snapshot:

```text
package root: /zys/shaobo/toolchains/compiler_perf_model_latest_20260721_packages
extract root: /zys/shaobo/toolchains/compiler_perf_model_latest_20260721_root
compiler: clang 18.0.0, llvm 47a7d59a80a4313d0c33d4667c3c8573604d0dbc
clang-18 sha256: fddad9d6b6a0bc2264d815e97bbc7679fba9268e8f0b71d145acfa466da3b395
```

The snapshot contains one SHA-verified package set: `rocm-llvm`,
`rocm-device-libs`, `comgr`, `hipcc`, and `hip-dev`. Keep these components
together; do not copy only clang into another ROCm root.

| Package | SHA256 |
|---|---|
| `rocm-llvm_18.0.0.dev_amd64.deb` | `757d7daf2ae81d549174b7780108f8f3ae62ad24e25b7441612e62d0c2213e54` |
| `rocm-device-libs_1.0.0.99999-local_amd64.deb` | `984f92068d00730c1b308ece48754320a1955d7e497115a06b9bb4dac1f7a8cf` |
| `comgr_2.8.0.99999-local_amd64.deb` | `da5aea9c73cc5646da33fddb043ea855e176e0f4f981a543b0de5bc4e72432ca` |
| `hipcc_1.1.1.99999-local_amd64.deb` | `49552525341bb018e85900bf69d3d288d36e42a61c1f9010b90b1ebdc3f74975` |
| `hip-dev_6.3.42133-88df948-local_amd64.deb` | `30947d16af11b4a05ef555dacdc67e1a84cc9cb1de284aba3c5f106b5cf666ba` |

Validation:

- dQ commit `d97684f`: static gates PASS, trap0, private/spill/scratch0,
  H1/S128 correctness PASS. Target-kernel instructions exactly match the
  accepted LLVM47a7 build.
- dKV commit `f57714f`: static gates PASS, trap0, private/spill/scratch0,
  H1/S128 correctness PASS. Target-kernel instructions exactly match the
  already measured LLVM47a7 dKV build.
- Therefore the refresh changes no kernel codegen relative to the Jul20
  LLVM47a7 snapshot. Keep dQ on LLVM47a7, but keep dKV on LLVM7b796991 because
  its prior LLVM47a7 S1024 A/B regressed ticks and MMAC active.

Rule: "latest compiler installed" and "compiler promoted for every kernel"
are separate decisions. Promotion requires a same-source, same-PMD,
same-shape ASM/correctness/performance A/B for each canonical kernel.
