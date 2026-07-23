# Fused5 Native Lag-One Canonical Integration

Date: 2026-07-23

Status: `ACCEPT_CANONICAL_NATIVE_LAGONE_16WAVE / MMAC50_OPEN`

## Design

- Tile: `M64/N128/D128`, exactly five logical GEMMs and 1,280 useful MMAC
  per tile.
- Roles: P0 waves0-3, dKV C0 waves4-7, dKV C1 waves8-11, dQ writers
  waves12-15.
- Each dKV wave owns one N16 dK/dV slice and executes 128 MMAC.
- Each dQ writer owns one D32 partial and executes 64 MMAC.
- dS is published once into the former K/V LDS region. dK reads the normal
  view and dQ reads the trans view.
- The dQ writers latch their K normal fragments before K/V LDS is overwritten.
- Q is read one M16 panel at a time for dK. A whole four-panel Q batch is not
  kept live.

## Resource Sweep

| WDRA windows | Result | Conclusion |
|---|---|---|
| `32/176/176/96` | private 260 B, 93 VGPR spills | Initial role estimate is infeasible. |
| `8/200/200/88` | private 116 B, 40 spills | Reallocating windows is insufficient. |
| `8/208/208/88` | private 40 B, 16 spills | Spending the full 512-VGPR budget is still insufficient. |
| `8/200/200/88`, panel-streamed Q | private/spill/scratch 0 | The blocker was whole-batch Q liveness, not only the quota. |

The accepted build uses role VGPRs `8/168/169/84`, metadata SGPR82/VGPR124,
LDS 115,456 B, and no private segment, scratch, SGPR spill, or VGPR spill.

## Correctness

S128 causal:

- dK: max_abs `5.47478e-06`, rel_l2 `2.30611e-04`
- dV: max_abs `1.89424e-04`, rel_l2 `3.84497e-04`
- MMOP `2,560`, bank conflict 0

S128 noncausal:

- dK: max_abs `1.20397e-06`, rel_l2 `6.26655e-04`
- dV: max_abs `1.49664e-05`, rel_l2 `7.81275e-04`
- MMOP `2,560`, bank conflict 0

S1024 causal:

- dK: max_abs `5.55674e-06`, rel_l2 `2.77289e-04`
- dV: max_abs `1.94430e-04`, rel_l2 `4.61692e-04`
- MMOP `92,160`, bank conflict 0

## Same-Build Performance

Compiler: `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`,
`GPU_ARGS=['--SQCIPfLines=7']`.

| Metric | Q-latch control | Native lag-one | Change |
|---|---:|---:|---:|
| kernel ticks | 100,667,385 | 73,271,835 | -27.214% |
| MMAC active | 17.016302% | 21.785506% | +4.769204 pp |
| LDS instructions | 73,504 | 64,096 | -12.799% |
| VALU instructions | 181,856 | 176,896 | -2.728% |
| waitLgkm share | 11.584188% | 8.327149% | -3.257039 pp |
| ABarrier share | 27.082444% | 22.471720% | -4.610724 pp |
| coissue success/fail | 6,945 / 6,889 | 14,474 / 15,170 | success +108.41% |

The promoted fullperf run is 73,280,025 kernel ticks and 21.809889% MMAC
active. It preserves MMOP 92,160, bank conflict 0, and correctness.

## SQTT Findings

The 16-wave role split is active, but useful staggering is incomplete.

- C0/C1 256-cycle bins: 216 MMAC-vs-MMAC, 121 MMAC-vs-VALU, 55 MMAC with
  no peer VALU.
- C0/P1 bins: 15 MMAC-vs-MMAC, 50 MMAC-vs-VALU, 339 MMAC with no peer VALU.
- C1/P1 bins: 47 MMAC-vs-MMAC, 38 MMAC-vs-VALU, 286 MMAC with no peer VALU.
- The top issue edge is `s_abarrier_try_wait -> s_xor`, 28.59% of sampled
  latency. This is ownership waiting attributed to the following instruction,
  not an `s_xor` arithmetic bottleneck.
- Atomic-to-atomic gaps account for 13.01%; the dQ atomic tail remains exposed.
- Matrix-read-to-wait gaps remain visible: trans 6.58%, normal 4.67%.

Therefore the accepted change removes real ownership work, but it does not
yet create the intended FWD-style conveyor. C0 and C1 still spend too many
windows issuing MMAC together, and P1 dQ MMAC often overlaps peer waiting
rather than peer VALU.

## Evidence

- S128 causal:
  `/zys/sb/fa3b/fused5_native_lagone_correctness/5gemm_symmetric_s128_c1_20260723_180748`
- S128 noncausal:
  `/zys/sb/fa3b/fused5_native_lagone_correctness/5gemm_symmetric_s128_c0_20260723_180825`
- S1024 stats:
  `/zys/sb/fa3b/fused5_native_lagone_stats/5gemm_symmetric_s1024_c1_20260723_180921`
- Same-build control:
  `/zys/sb/fa3b/fused5_native_lagone_control_stats/5gemm_symmetric_s1024_c1_20260723_181216`
- Fullperf:
  `/zys/sb/fa3b/fused5_native_lagone_fullperf/5gemm_owner_s1024_c1_fullperf_20260723_181404`
- Commit-audit rebuild and rerun:
  `/zys/sb/fa3b/build_fused5_native_lagone_audit_20260723`,
  `/zys/sb/fa3b/fused5_native_lagone_commit_audit/5gemm_symmetric_s128_c1_20260723_184449`,
  `/zys/sb/fa3b/fused5_native_lagone_commit_audit/5gemm_symmetric_s1024_c1_20260723_184449`
- XCU:
  `/zys/sb/fa3b/xcu_outputs/fused5_native_lagone_s1024_20260723`
- Shared archive:
  `/共享/shaobo/perf/20260723_181404_fused5_native_lagone_16wave_h1s1024_sqc7_fullperf`
- Perf SHA256:
  `717c89800a3f4f187760923a1903a07fdb5a0866275ecb19359e4d44eb5e4fbe`

## Next Architecture Gate

Preserve this topology and exact-work ledger. Split score/P, dP, dV, and dS
into explicit schedule islands so C0 and C1 can use different legal orders:
one group starts `score -> softmax/P -> dV`, while the other starts `dP`.
This must create peer MMAC/VALU overlap with useful work, not an empty delay.
Only after that gate should dQ atomic handoff be reconsidered.
