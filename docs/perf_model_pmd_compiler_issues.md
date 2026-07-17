# Shaobo Perf Model PMD And Compiler Issue Registry

Last updated: 2026-07-17

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
