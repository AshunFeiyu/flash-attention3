---
title: "WASP dKV ABarrier/BPS 指令分析"
date: 2026-07-11
status: OBSERVE
scope: Shaobo FA3 BWD dKV, WASP/WDRA path
---

# WASP dKV ABarrier/BPS 指令分析

## Trigger

FA2-style no-WDRA 路线暂缓，回到 Shaobo WASP/WDRA 版本。DCU wiki 新增
Shaobo ABarrier、DS/TLS、MLS/BPS 资料后，重新检查 dKV 是否遗漏了
Shaobo 原生同步语义。

## Knowledge Imported

- ABarrier 是 transaction-aware barrier。一个 round 完成需要
  `pending_arrival_count == 0` 且 `transaction_count == 0`。
- `s_abarrier_seq` 标记后续 memory transaction tracking interval；
  `s_abarrier_arrive*` 关闭该 interval。
- ABarrier 指令本身 non-blocking，等待必须通过 `test_wait/try_wait`
  和 phase identity 证明。
- BPS bypass-L1 path 正确消费前需要 `s_waitcnt_vbcnt 0`，普通
  `vmcnt/lgkmcnt` 不是 BPS 完成证明。
- `s_abarrier_init` 后、`s_abarrier_inv` 前后都需要 full/appropriate
  sync；当前代码 init 后用 `s_ebarrier_sync(0)`，inv 前用 AllDone +
  `__syncthreads()`，生命周期大体符合资料。

## Code Evidence

Current dKV:

- Source: `src/dkv_kernel.cpp`, `include/shaobo_instr.h`
- Accepted asm snapshot:
  `work/asm/dkv_qused_accepted_20260710/fa3_bwd_wasp_clean.asm`

Observed source pattern:

```text
s_abarrier_seq(Filled)
matrix_load_* bps lds
s_abarrier_arrive(Filled)
consumer s_abarrier_try_wait(Filled)
consumer ds_read_matrix + s_waitcnt lgkmcnt
consumer s_abarrier_arrive(Used)
producer s_abarrier_try_wait(Used)
```

Important absence:

- no explicit `s_abarrier_expect_tx`
- no explicit `s_abarrier_complete_tx`
- no explicit `s_waitcnt_vbcnt`

FWD 0310 control asm has the same important absence: no explicit
`s_waitcnt_vbcnt/expect_tx/complete_tx`, while also using
`matrix_load_* bps lds` inside `s_abarrier_seq -> arrive` regions. Therefore
missing `vbcnt` is not automatically a BWD bug. It is either:

1. `seq + matrix_load bps + arrive` already lets hardware/PMD track BPS
   transactions through ABarrier; or
2. both FWD and BWD rely on an undocumented compiler/PMD convention.

This must be tested with focused probes before changing canonical dKV.

## Static ASM Counts

Approximate static counts from accepted dKV and FWD 0310 asm:

| Instruction class | dKV accepted | FWD 0310 |
|---|---:|---:|
| `s_abarrier_seq` | 6 | 24 |
| `s_abarrier_try_wait` | 25 | 88 |
| `s_abarrier_arrive` | 28 | 95 |
| `matrix_load_32x32_b16` | 10 | 94 |
| `matrix_load_32x16_b16` | 18 | 0 |
| `ds_read_matrix_format` | 256 | 640 |
| `ds_read_matrix_trans_format` | 288 | 704 |
| `s_waitcnt_vbcnt` | 0 | 0 |
| `s_waitcnt lgkmcnt` | 395 | 342 |
| `v_mmac` | 1028 | 5120 |

Interpretation:

- FWD has more barrier operations in static asm, but far larger MMAC/read
  islands per source region and no 32x16 MLS load path.
- dKV has many more wait/control instructions per static MMAC and uses smaller
  half-page MLS packets, so ABarrier overhead is exposed more easily.
- The performance gap is not explained by "FWD uses explicit vbcnt and BWD
  does not"; neither accepted asm shows explicit vbcnt.

## Current Bottleneck Re-read

Existing SQTT evidence still stands:

- dKV accepted H1/S1024: `MMAC active ~= 33.24%`, `MMOP runtime share ~= 59.47%`.
- Top bubble: `s_abarrier_try_wait -> s_xor_b32`, around 40%.
- Secondary: `s_abarrier_try_wait -> s_waitcnt`, matrix-read wait, sidecar
  readiness.
- FWD reaches much better active behavior because ownership epochs carry larger
  useful MMAC/VALU islands and consumer/producer roles do not stall together
  as often.

New interpretation from ABarrier docs:

- The hot ABarrier row is not merely "too many barriers"; it means page/packet
  ownership reaches a phase wait while no peer role has enough independent
  useful work to hide it.
- Since ABarrier already has transaction tracking semantics, blindly adding
  `s_waitcnt_vbcnt 0` before every `arrive` may serialize producer and reduce
  overlap. It is a probe candidate, not a direct fix.

## Missed Or Underused Possibilities

1. **ABarrier transaction proof probe**
   - Use `s_abarrier_slot_rd` in an isolated probe to observe whether
     `transaction_count` changes for `seq + matrix_load bps + arrive`.
   - If supported, test explicit `expect_tx` around MLS packets.

2. **BPS completion proof probe**
   - Compare current `seq + arrive` against `seq + matrix_load + vbcnt + arrive`
     on a tiny LDS matrix read correctness/perf probe.
   - Do not patch canonical dKV unless SQTT shows lower wait without worse
     producer serialization.

3. **FWD-style larger packet/island**
   - More promising than vbcnt: increase useful MMAC/read work per ownership
     epoch.
   - Candidate: replace 32x16 source packet reads with 32x128-style read
     bricks where VGPR budget allows, keeping one canonical dKV path.

4. **ValuExec-style useful stagger**
   - FWD has an explicit `ValuExec0` token to let causal/VALU work of one
     consumer gate or offset the other consumer.
   - dKV currently has no equivalent useful-work token; the two consumers often
     reach ownership/readiness waits together.
   - A dKV version must only gate real softmax/dS/address work, not insert
     empty delay.

5. **EBarrier is not the main fix**
   - EBarrier can simplify partial-wave sync or tail lifecycle, but current
     dominant tokens protect producer/consumer LDS ownership and BPS
     transactions. Those still fit ABarrier better.

## Next Experiments

P0, no canonical kernel change:

1. `abarrier_tx_slot_probe`
   - `seq -> matrix_load bps lds -> arrive`
   - read slot state if possible
   - verify whether transaction tracking is active without explicit
     `expect_tx`

2. `bps_vbcnt_completion_probe`
   - current vs `s_waitcnt_vbcnt 0` before `arrive`
   - correctness plus SQTT: ownership wait, matrix-read wait, producer bubble

P1, if probe supports it:

3. Add the minimum helper wrapper for transaction-aware Filled tokens, but keep
   independent Used tokens. Do not merge Used unless source evidence changes.

P2, likely higher ROI:

4. Move dKV toward FWD-style large read/MMAC islands and useful consumer
   stagger. Measure with xcu, not just PMD counters.

## Decision

Current state is `OBSERVE`.

Do not implement FA2-style no-WDRA now. Return to WASP dKV, but the next change
should be a focused ABarrier/BPS probe or a larger-island dKV restructuring
that has workbook/pseudocode first.

## 2026-07-11 BPS vbcnt Probe Result

Decision: `OBSERVE_MICRO_WIN_NEEDS_XCU`

Probe:

- Added opt-in macro `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE`.
- When enabled, producer inserts `s_waitcnt_vbcnt 0` before BPS-published
  Filled-token `arrive`.
- Default build keeps the macro off, so the canonical path is unchanged.

Static/resource evidence:

- Default and vbcnt variant both compile and pass dKV evidence gate plus symbol
  metadata gate.
- Branch windows remain `14/16`, `222/240`, `222/240`, `8/16`.
- Metadata remains `private_segment=0`, `sgpr=99`, `vgpr=128`,
  `sgpr_spill=0`, `vgpr_spill=0`.
- Default asm has `0` `s_waitcnt_vbcnt`; vbcnt asm has `6`.

Correctness evidence:

- H1/S128 default and vbcnt both PASS.
- H1/S1024 default and vbcnt both PASS.
- H1/S1024 numerical output is unchanged at the harness level:
  `dk_rel_l2=0.0025563`, `dv_rel_l2=0.000337571`, `bad=0`.

Stats-only result on liuchang/zys1, same PMD/compiler/env
`GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`:

| Case | simTicks | kernel_ticks | MMOP | VALU | SCA | LDS | coissue success/fail | ldsBankConflict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default H1/S1024 | 47,136,635 | 43,523,025 | 131,072 | 168,514 | 114,520 | 79,360 | 36,078/25,327 | 0 |
| vbcnt H1/S1024 | 46,609,290 | 42,995,680 | 131,072 | 168,514 | 114,520 | 79,360 | 36,749/26,029 | 0 |

Interpretation:

- Explicit `vbcnt` before Filled `arrive` is not obviously harmful here; it
  improves same-shape stats-only `kernel_ticks` by about `1.21%` and
  `simTicks` by about `1.12%`.
- Instruction class counts do not change, bank conflict remains zero, and no
  spill/scratch appears.
- Coissue count rises but coissue rate does not clearly improve, so this is
  not yet a coissue proof.
- Treat as a focused micro-win candidate only. Promotion needs repeat/xcu
  evidence showing the ownership or matrix-read bubble actually shrinks.

Artifacts:

- Run root:
  `/zys/shaobo_runs/bps_vbcnt_probe_20260711_105224`.
- Default H1/S1024:
  `/zys/shaobo_runs/bps_vbcnt_probe_20260711_105224/dkv_mmac_correctness_20260711_110121`.
- Vbcnt H1/S1024:
  `/zys/shaobo_runs/bps_vbcnt_probe_20260711_105224/dkv_vbcnt_h1s1024_20260711_110217`.

## 2026-07-11 Default Enable Decision

Decision: `ACCEPT_DEFAULT_STATS_WIN_PENDING_XCU`

The probe was promoted to default because the default-enabled rebuild preserved
resources and correctness, and the H1/S1024 rerun stayed in the improved band:

- default asm now has `6` `s_waitcnt_vbcnt`.
- evidence gate and symbol metadata gate PASS.
- H1/S128 and H1/S1024 correctness PASS.
- H1/S1024: `simTicks=46,554,690`,
  `kernel_ticks=42,941,080`, `MMOP=131,072`, `VALU=168,514`,
  `SCA=114,520`, `LDS=79,360`, coissue `37,689/26,615`,
  `ldsBankConflict=0`.

Boundary:

- This is a local BPS readiness fix. It does not solve the fundamental dKV
  fragmentation problem by itself.
- Keep the macro override for A/B testing:
  `EXTRA_CXXFLAGS="-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0"`.
  Full-perf xcu comparison is still useful before treating the effect as a
  stable architectural rule.
