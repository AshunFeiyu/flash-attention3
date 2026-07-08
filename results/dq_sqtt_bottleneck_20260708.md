# dQ SQTT Bottleneck Note, 2026-07-08

## Question

dQ uses a FWD-sized `Mq=128` tile and has three GEMM islands per K/V tile:
`QK`, `dO*V`, and `dS*K`.  The open question is why its MMAC active is still
far below the FWD reference.

## Compared Artifacts

- FWD H4/S1024 SQTT:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_193535_fwd_bwd_sqtt_h4s1024_sqc7_xcu/remote_tree/fwd_h4s1024_sqc7/xcu_dispatch0`
- FWD H4/S2048 stats:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_004810_fwd_reference_h4s2048_sqc7/fwd_h4s2048_sqc7_perf_capture_metrics.json`
- Current dQ H1/S1024 SQTT:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_113438_dq_sidecar_soa_vec4_h1s1024_sqc7_fullperf/xcu`
- Current dQ H1/S1024 stats:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_113438_dq_sidecar_soa_vec4_h1s1024_sqc7_fullperf/stats.txt`

## Metrics

| Metric | FWD H4/S1024 SQTT | FWD H4/S2048 stats | dQ H1/S1024 current |
|---|---:|---:|---:|
| CUs / waves | 16 / 256 | 32 active CUs | 8 / 128 |
| Inst issues | 549,136 | n/a | 337,168 |
| SQTT MMAC latency share | 45.96% | n/a | 8.85% |
| PMD MMAC active share | n/a | 45.02% | 25.35% |
| PMD MMOP runtime share | n/a | 58.12% | 41.86% |
| MMOP instructions | 163,840 SQTT hits | 589,824 PMD | 55,296 PMD |
| VALU instructions | 237,952 SQTT hits | 758,656 PMD | 138,208 PMD |
| SCA instructions | 52,064 SQTT hits | 224,784 PMD | 87,176 PMD |
| LDS matrix/read pressure | 5.89% SQTT | 150,528 LDS | 28,656 LDS |
| LDS bank conflict | n/a | 0 | 0 |

## Top SQTT Differences

FWD:

- `mmop_fp16` is the dominant hot instruction: `45.96%`.
- `salu_32` is only `7.30%`.
- `lds_matrix -> mmop_fp16` bubble is only `1.18%`.
- The top barrier bubble is large, but it tends to move into useful scalar work
  (`abarrier -> salu_32`) rather than a pure wait chain.

dQ:

- Top hot rows are `s_xor_b32 35.36%` and `s_waitcnt 16.44%`; `v_mmac` is only
  third at `8.85%`.
- Top issue bubbles are `s_abarrier_try_wait -> s_xor_b32 37.26%` and
  `s_abarrier_try_wait -> s_waitcnt 10.23%`.
- `ds_read_matrix -> s_waitcnt` is only `1.51%`, and
  `s_waitcnt -> v_mmac` is `1.04%`; matrix-read latency exists but is not the
  first-order limiter in this capture.
- PMD totals agree: barrier/wait/control is large
  (`barrierCounter=54,562`, `waitLgkm=15,865`, `waitVm=11,727`,
  `emptyBuffer=26,965`), while `ldsBankConflict=0`.

## Code Mapping

Current dQ is 16 waves:

- waves0-3: load group0 `Q/dO/sidecar`, then K pages.
- waves4-7: consumer group0.
- waves8-11: consumer group1.
- waves12-15: load group1 `Q/dO/sidecar`, then V pages.

The active LDS plan uses `Q+dO + one K/V page + sidecar`.  Page1 reuses the
old Q/dO area after `QDoLatched`, so the main loop is controlled by
`PageFilled`, `PageUsed`, and `QDoLatched` ownership tokens.  The SQTT hotspot
lands on those tokens, especially the PageUsed path, not on missing MMAC or LDS
bank conflicts.

## Diagnosis

The dQ ceiling is currently not limited by GEMM count.  It is limited by useful
MMAC work per ownership epoch and by the ABarrier page ledger that serializes
producer and consumer progress.  FWD has enough long, regular MMAC and VALU
islands after each barrier to make the barrier protocol look like part of a
conveyor.  Current dQ repeatedly reaches `s_abarrier_try_wait` before the next
useful island is ready, so active time is filled by wait/control instead of
MMAC.

The immediate optimization target should therefore be:

1. Increase useful MMAC per PageFilled/PageUsed token without multiplying
   barrier tokens.
2. Reduce `PageUsed` wait exposure or change the LDS lifetime so PageUsed is not
   on the critical path.
3. Keep matrix paths native (`MLS/BPS + ds_read_matrix + MMAC`) and avoid
   ordinary DS reads on the main matrix path.
4. Treat `v_mov`, causal mask, and `wait_lgkm` cleanup as second-order work
   after the ABarrier ownership cliff is improved.

## Evidence Gap

This note uses H1/S1024 for the current dQ diagnostic but H4/S1024/H4/S2048 for
FWD references.  The bottleneck class is still clear, but final FWD-style
acceptance needs a same-shape dQ H4/S1024 SQTT capture with the same
`GPU_CHIP=sb` and `GPU_ARGS=['--SQCIPfLines=7']`.
