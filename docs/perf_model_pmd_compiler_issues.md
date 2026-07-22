# Shaobo Perf Model PMD And Compiler Issue Registry

Last updated: 2026-07-22

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

Current mandatory environment on 2026-07-21:

```text
host/container: liuchang / zys1
repo: /zys/shaobo/fa3_bwd_wasp_7gemm_consumer_conveyor_20260717
GPU_CHIP: sb
GPU_ARGS: ['--SQCIPfLines=7']
compiler: clang 18.0.0, llvm 47a7d59a80a4313d0c33d4667c3c8573604d0dbc
compiler path: /zys/shaobo/toolchains/compiler_perf_model_latest_20260721_root/opt/rocm-6.3.3/llvm/bin/clang++
compiler sha256: fddad9d6b6a0bc2264d815e97bbc7679fba9268e8f0b71d145acfa466da3b395
PMD root: /zys/shaobo/toolchains/pmd_20260717
PMD core gem5.opt sha256: 4748d40d99414c7be6ab3d2b62bca1f134d3454edec711a6321bdafa237be1e9
PMD lib/libgem5.so sha256: 29fa2020e6bfb399225e206cf7c589ba838ad56b891cb07c97e88029e954bfa5
PMD soc gem5.opt sha256: d0c03538753a4b91c2aa3e110cb12f1302b66c891c3ab2d446c85de99fe24524
WDRA: explicit __builtin_hcu_wdra_init + -mllvm -turn-off-wdra-trap-handler=no-pad
```

Refresh this block after every PMD or compiler update. Never compare compiler
or PMD behavior across fingerprints without saying so explicitly.

The executable gate is `scripts/toolchain_preflight.sh`. A fullperf captured
with LLVM47a7 but the old default HEAD1668 PMD (`2957138_fa3_bwd_dkv.perf`)
is classified `ENVIRONMENT` and excluded. Valid replacements are
`2957276_fa3_bwd_dkv.perf` and `2957578_fa3_bwd_dq.perf`, both generated with
the fingerprint above.

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

Latest locked-toolchain recheck (`2026-07-21`): the failure remains. LLVM
`47a7d59a` emits the same `element:3 row:1 col:1` f32 writer, encoded as
`0xd38b5008` in this compiler snapshot. PMD HEAD1694 aborts at that first
writer with `fatal: Invalid opcode encountered: 0xd38b5008`. A first build
with local-wave WDRA stopped earlier at `vgpr_alloc_mode isn't 1 when
s_set_vgpr_size`; rebuilding this uniform one-wave probe with
`SHAOBO_DISABLE_WDRA_FLAGS=1` removes every `s_set_vgpr_size` and isolates the
f32 opcode failure. Evidence is under `/zys/sb/u47_pds_f32_probe_nowedra`
(build: `/zys/sb/u47_pds_f32_probe_build_nowedra`). This strengthens the
PMD/compiler-encoding attribution and does not change the canonical fallback.

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

2026-07-22 exhaustive roundtrip confirmation:

- New focused source: `probes/matrix_global_roundtrip_probe.cpp`.
- Run:
  `/zys/sb/fa3b/layout_probes/matrix_global_roundtrip_20260722_202711`.
- Swept all four MLS `t/r` modes against all four matrix-store `t/r` modes.
  All 16 direct controls reproduce exactly 240/512 mismatches, first at
  row17/col0 with `0xfefe`; only rows0..16 are committed.
- A further 192 combinations insert matching m32 `ds_read_matrix` plus all
  four f16 `ds_write_matrix` modes. None is exact or a complete permutation,
  but these results are downstream of the already-failing direct control.
- The test uses LDS byte offset zero, the full documented ABarrier lifecycle,
  SGPR18/VGPR4, private/spill0, 100,352-byte LDS and bank0. This rules out the
  earlier offset misuse and any single `t/r` choice as explanations.

2026-07-22 writer-only isolation:

- `probes/ds_matrix_write_lds_dump_probe.cpp` removes MLS, matrix readers and
  matrix-store. It writes 512 unique FP16 values with each of the four writer
  modes, scans the complete 2 KiB page using ordinary `ds_read_b128`, and
  returns the bits with global stores.
- All four modes preserve all 512 labels exactly once, with 512 poison slots
  left untouched. Metadata is SGPR14/VGPR10 with no private segment or spill.
- This clears `ds_write_matrix_format_f16` as the value-loss point on the
  current model. PMD-005 remains specifically a matrix-store/descriptor
  boundary, while MMAC-output-to-writer source-layout semantics remain a
  separate compiler contract question.
- Evidence: `/zys/sb/fa3b/layout_probes/`
  `ds_matrix_write_lds_dump_20260722_214911` and
  `results/ds_matrix_write_lds_dump_20260722.md`.

### PMD-006: HEAD1698 Core Package Has An Internal Config ABI Mismatch

Status: `CONFIRMED PMD PACKAGE COMPATIBILITY / NOT PROMOTED`.

The fixed core package URL changed on 2026-07-21 and now contains runtime
`CoreArch:HEAD_1698(lib_ini_opt)`. The package cannot pass config generation or
consume the locked HEAD1694 seed:

- packaged `configs/gpu.py` passes `cp_prefetch`, `prefetch_obj_size`, and
  `prefetch_args_size`, but the packaged config-generator `gem5.opt` rejects
  `GPUDispatcher.cp_prefetch`;
- the old seed then fails successively on new runtime requirements including
  `cp_client_id`, `clientId`, `cp_prefetch`, and CP topology elements;
- in an isolated diagnostic copy, removing only generator-incompatible fields
  produces a config, but that generator identifies itself as
  `CoreArch:HEAD:1668(exec_ini_opt)` while the loaded runtime is HEAD1698;
- the generated config is immediately rejected by HEAD1698 for another missing
  field, `TcaHoleArbAge`.

Package and component fingerprints:

```text
core URL Last-Modified: 2026-07-21 09:35:48 GMT
core tar SHA256: fef22f48080e48f893fb66d81736c11a42453a50d12d0ea79901821b10ddf470
core readme commit: 029b9d17d7f0e1078fea6e9e5d8aebc7f6d95bcc
HEAD1698 gem5.opt SHA256: a0f1b681ac8c1271731465ce1bae209d675a7a176be77923350a5a6ff2968b5c
HEAD1698 lib SHA256: aca312fd740607d3eb80750bd1557527c682109cc1e7dfd37bc07e7a31dc6933
Shaobo C0 SOC SHA256: d0c03538753a4b91c2aa3e110cb12f1302b66c891c3ab2d446c85de99fe24524
```

The C0 SOC fixed URL is unchanged from 2026-04-14 and its binary is identical
to the current locked SOC. The top-level and b0 packages are older than c0.
Therefore selecting another published Shaobo SOC package is not a valid fix.

Impact and rule:

- do not promote `/zys/shaobo/toolchains/pmd_20260722` or change the executable
  lock until the provider publishes a matching config-generator/runtime pair or
  a known-good HEAD1698 config;
- do not use the diagnostic hotfix as performance or correctness evidence;
- PMD-005 remains unresolved because HEAD1698 never reaches the target kernel;
- canonical HEAD1694 remains healthy and re-runs the unchanged global
  roundtrip probe with transport PASS, bank0, and the same 240/512 direct-store
  mismatch.

Evidence:

- audit: `results/pmd_head1698_update_audit_20260722.md`;
- failed official launch:
  `/zys/sb/fa3b/pmd_update_audit/global_roundtrip_pmd_20260722_20260722_205355`;
- isolated generator proof:
  `/zys/sb/fa3b/pmd_update_audit/configgen_probe5_20260722_211152`;
- HEAD1694 control:
  `/zys/sb/fa3b/layout_probes/matrix_global_roundtrip_20260722_211218`.

Owner question:

> Can the PMD provider publish a HEAD1698 core tarball whose `gem5.opt`,
> `libgem5_opt.so`, Python configs, and config schema are built from one
> revision, or provide the matching Shaobo C0 `config.ini` seed?

## Compiler And Compiler/PMD Issues

### COMP-004: `DS_MATRIX_TRANSPOSE_4V` Has No Current Toolchain Entry

Status: `DEFERRED_COMPILER_AND_PMD_SURFACE_MISSING`.

Environment fingerprint:

```text
compiler: LLVM e0f10535a0d681bcf3885ea2c398cc494bf6e332
clang SHA256: 334cb561ceeaf1499039f6ff2a146e71e6b55b83b80d8d407a77ed27155f6f34
PMD: HEAD1694, compiled 2026-07-17
HCU source branch: rocm-llvm-dtkenv @ 351b875334eff96873f0ec62ddde6436d8aade08
XCompute Light: 4.6.3 SQTT CLI package
```

The public ISA Delta dated 2025-06-10 describes
`DS_MATRIX_TRANSPOSE_4V`: a 16x16 b32 / 32x16 b16 / 64x16 b8 / 128x16 b4
matrix in four adjacent VGPRs is transposed through the LDS pipeline and
returned to VGPRs without writing LDS.  Its table row repeats opcode `226`,
which is also assigned to `DS_READ_MT8X32_B32`.

Current tool evidence does not expose an independent instruction contract:

- LLVM e0f10535 rejects a `ds_matrix_transpose_4v` mnemonic and has no
  matching builtin or HCU test.
- The latest internal `rocm-llvm-dtkenv` branch has no matching mnemonic,
  builtin, intrinsic, or test.
- XCompute 4.6.3 maps DS opcode `226` only to `ds_read_mt8x32_b32`.
- PMD has no `matrix_transpose_4v` symbol or disassembly string.
- Raw opcode `227` is not this instruction. PMD decodes it as `DS_READ_CV`
  and rejects an invalid `opctrl` (`W == 0`).

Do not infer a raw encoding from adjacency in the ISA table and do not place
`.long` encodings in a production kernel.  The two raw attempts are retained
only in remote run evidence:

```text
/zys/sb/fa3b/layout_probes/ds_transpose4v_20260722_181852
/zys/sb/fa3b/layout_probes/ds_transpose4v_20260722_183028
```

Impact:

- This instruction could be a native way to convert the natural dS fragment
  between dKV-friendly and dQ-friendly ownership without a `ds_mpermute`
  bridge.
- It cannot be used or performance-tested with the current compiler/PMD
  contract, so it is not a valid dependency for the canonical five-GEMM
  kernel.
- Continue only with a documented builtin/mnemonic plus a focused numerical
  probe, or with a separately proven native MMAC/writer/reader layout.

The final dense differential probe strengthens this from a speculative impact
to an active five-GEMM blocker.  Under the locked e0f10535/HEAD1694 toolchain,
the natural score-owned dS fragment gives exact direct dK but incorrect direct
dQ; `ds_write_matrix_format_f16(t=1,alt0)` followed by trans/normal readers
gives incorrect dQ and dK.  Upstream score/dP/dS are exact, metadata is
spill/private-free, and `ldsBankConflict=0`, so this is not an algorithm,
store-index, resource, or bank-conflict failure.  Evidence:
`/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_192016` and
`docs/5gemm_native_ds_gate_20260722.md`.

Owner question:

> What is the final encoding and compiler builtin for
> `DS_MATRIX_TRANSPOSE_4V` on gfx946, and which PMD build implements it?  Is
> the repeated opcode `226` in the 2025-06-10 ISA Delta intentional or a
> documentation placeholder?

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

## PMD-NONZERO-VBCNT-WARNING

Status: `OBSERVED_USABLE_IN_FOCUSED_PROBE / CONTRACT_UNCONFIRMED`.

- First seen: `2026-07-21`.
- Environment fingerprint: LLVM `47a7d59a`, clang SHA256 `fddad9d6...`, PMD
  HEAD1694 locked core/lib/SOC hashes.
- Minimal trigger: one wave issues 32 BPS requests, executes
  `s_waitcnt_vbcnt 4`, publishes A to a consumer through ABarrier, then uses
  `vbcnt 0` before publishing B.
- Expected behavior: wait until at most four BPS requests remain outstanding,
  without draining the final four.
- Observed behavior: PMD prints `S_WAITCNT_VBCNT: vbcnt isn't 0`, but completes.
  Candidate A/B data is exact and bank0.
- Final ASM evidence: exact BPS32, matrix-read8, wait4 once, wait0 once; no
  WDRA resize or trap in the focused kernel.
- Static metadata: SGPR16/VGPR7, private/spill/scratch0.
- PMD log/run roots:
  `/zys/sb/vbcnt4_stress_candidate/bps_vbcnt_threshold/run_20260721_195840`,
  `/zys/sb/vbcnt4_stress_control/bps_vbcnt_threshold/run_20260721_195935`,
  `/zys/sb/vbcnt4_stress_fullwait/bps_vbcnt_threshold/run_20260721_200505`.
- Known-good comparison: wait4 kernel ticks `2,704,520`, no-wait `2,700,880`,
  full-wait `2,842,840`. The `4.865%` gap from full-wait proves PMD does not
  clamp nonzero VBCNT to zero.
- Current attribution and confidence: nonzero threshold execution works in
  this PMD path with medium-high confidence; FIFO retirement semantics remain
  unconfirmed because no-wait also passes.
- Workaround: use only in a reversible, exact-count same-wave experiment and
  retain full correctness/performance/SQTT gates.
- Owner question / next action: confirm whether BPS completion accounting is
  FIFO for one wave and whether the warning is informational or indicates an
  unsupported hardware contract.

## 2026-07-21 Unified Latest-Compiler Resolution

Status: `RESOLVED_FOR_CANONICAL_BUILD`; this supersedes the prior per-kernel
compiler exception by explicit user policy.

- Locked compiler: LLVM47a7d59a, clang SHA256
  `fddad9d6b6a0bc2264d815e97bbc7679fba9268e8f0b71d145acfa466da3b395`.
  `build.sh` and preflight reject any other hash.
- The extracted package root is not a complete runtime: invoking its hipcc
  fails at link with `unable to find library -lamdhip64`. Use latest overlay
  clang but `/opt/rocm-6.3.3/bin/hipcc`, headers and libraries.
- LLVM47a7 rejects `-mllvm -run-on-model=true`. Its verified WDRA contract is
  explicit `__builtin_hcu_wdra_init(...)` plus
  `-mllvm -turn-off-wdra-trap-handler=no-pad`; both dKV and dQ then emit real
  `s_trap=0`, preserve role-local `s_set_vgpr_size`, pass metadata gates and
  pass PMD correctness.
- PMD HEAD1694 still prints `ASTCA has no parameter num_phase` while trying to
  generate a fresh config. With a SHA-locked `PMD_CONFIG_SEED`, run.py falls
  back to that existing config and both kernels complete successfully. Treat
  this as a PMD config-generation ABI issue, not a kernel failure. A non-empty
  seed remains mandatory in `scripts/toolchain_preflight.sh`.
- Default run root is now `/zys/sb/fa3b`; longer nested paths can overflow the
  PMD fake-device path before dispatch.

## 2026-07-22 Rolling Compiler Supersedes LLVM47a7

Status: `ACCEPT_TOOLCHAIN_BASELINE_RESET`; this section supersedes the
2026-07-21 compiler identity while retaining its runtime and PMD rules.

Latest audited package identity:

```text
index: perf_model_latest_6.3_ubuntu-22.04/Packages.gz
Last-Modified: 2026-07-22T03:33:27Z
index SHA256: 351d4166e3ff6dded100f205033af99d1ab42f222fe1a807a6a1fc349033ea1a
rocm-llvm deb SHA256: bb880ba1477b579a25b9f36772796893dad914a5175fe9278a4616e0bc62808c
LLVM commit: e0f10535a0d681bcf3885ea2c398cc494bf6e332
clang SHA256: 334cb561ceeaf1499039f6ff2a146e71e6b55b83b80d8d407a77ed27155f6f34
extract root: /zys/shaobo/toolchains/compiler_perf_model_latest_20260722_root
```

Rules:

- Use the installed `/opt/rocm-6.3.3/bin/hipcc` as the runtime wrapper with
  `HIP_CLANG_PATH` directed to the e0f10535 overlay. The side-by-side package
  remains a compiler overlay, not a standalone ROCm runtime.
- `scripts/toolchain_lock.sh` is the source of truth. `build.sh` records the
  index hash, package hash and timestamp in addition to compiler identity.
- PMD remains HEAD1694 with the audited config seed. PMD still tries fresh
  config generation; the known `ASTCA ... num_phase` failure triggers the
  SHA-checked seed fallback.
- Same-source H1/S1024 dKV/dQ A/B is mandatory when this rolling index changes.
  The e0f10535 reset regressed median ticks by `0.6815%` and `1.7820%`
  respectively, but is accepted by the policy requiring the latest compiler.
- Do not compare a new kernel candidate on e0f10535 against a control built by
  LLVM47a7. Historical performance remains useful for diagnosis only.

Validation evidence: workbook `218_LatestCompiler_e0f` and shared archive
`shaobo/perf/20260722_061614_latest_e0f10535_h1s1024_dkv_dq_fullperf`.

## 2026-07-22 Multi-Dispatch Full-Lifecycle Helper Emits No Perf

Status: `OBSERVE / SINGLE-DISPATCH FOLLOW-UP ONLY`.

- The full backward harness loads three kernel objects and dispatches
  `dot_do_o -> dKV -> dQ` in one process. With helper 2.2.0,
  `HSA_TOOLS_LIB=/opt/rocm-6.3.3/lib/xprofiler/libperf_gen_helper.so` and
  `GPU_DFLAGS=['StatLog','SQAbar','SQEbar','MMUCheck','TT','Perf']`, PMD runs all
  three dispatches correctly and writes complete `stats.txt` files, but emits
  neither the helper `<pid>_<binary>.perf` nor `stats_xcd0.perf`.
- Evidence root:
  `/zys/sb/dotfp/full_bwd_correctness_20260722_124146`. The run passes all
  numerical/resource/bank gates and measures dot `2,443,805` ticks, so this is
  not a kernel abort or correctness issue.
- Do not claim an xcu result from this run. Use stats for the exact lifecycle
  share and active-CU/SIMD proof. If instruction-level dot attribution becomes
  necessary, create a single-dispatch harness before escalating the helper;
  standalone dKV/dQ helper capture remains known-good in the same environment.
