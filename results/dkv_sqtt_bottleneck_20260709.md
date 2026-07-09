# dKV SQTT Bottleneck Note, 2026-07-09

## Artifacts

- Current dKV commit: `a9666a4 Optimize dKV QUsed release before softmax`.
- Current dKV perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_033115_dkv_qused_before_softmax_h1s1024_sqc7_fullperf`.
- Previous dKV baseline:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_003152_dkv_splitwait_h1s1024_sqc7_fullperf`.
- FWD H4/S1024 SQTT reference:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_193535_fwd_bwd_sqtt_h4s1024_sqc7_xcu/remote_tree/fwd_h4s1024_sqc7/xcu_dispatch0`.
- FWD H4/S2048 stats reference:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_004810_fwd_reference_h4s2048_sqc7/fwd_h4s2048_sqc7_perf_capture_metrics.json`.

## Summary

The current dKV kernel is not primarily blocked by missing MMAC, LDS bank
conflict, or global/cache bandwidth.  It is blocked by packet ownership and
wait/control exposure:

- `s_abarrier_try_wait -> s_xor_b32` remains the largest SQTT bubble.
- `s_waitcnt` remains the second largest hot instruction class.
- Consumer windows still show high bubble percentage between useful MMAC
  issues.
- FWD has much higher MMAC share because each ownership epoch contains longer
  useful MMAC/VALU islands and less scalar/control serialization.

## Metrics

| Metric | FWD H4/S1024 SQTT | FWD H4/S2048 stats | dKV splitwait H1/S1024 | dKV QUsed-before-softmax H1/S1024 |
|---|---:|---:|---:|---:|
| SQTT duration | 60,608 | n/a | 96,420 | 94,728 |
| SQTT inst issues | 549,136 | n/a | 567,706 | 565,274 |
| SQTT MMAC latency share | 45.96% | n/a | 10.91% | 10.79% |
| PMD MMAC active share | n/a | 45.02% | 32.9468% | 33.2391% |
| MMOP instructions | 163,840 SQTT hits | 589,824 PMD | 131,072 PMD | 131,072 PMD |
| VALU instructions | 237,952 SQTT hits | 758,656 PMD | 168,514 PMD | 168,514 PMD |
| SCA instructions | 52,064 SQTT hits | 224,784 PMD | 115,544 PMD | 114,520 PMD |
| LDS instructions | 41,984 SQTT hits | 150,528 PMD | 79,360 PMD | 79,360 PMD |
| LDS bank conflict | n/a | 0 | 0 | 0 |
| TCC hit / sector reuse | n/a | 0.9329 / 0.7317 | 0.9107 / 0.6427 | 0.9107 / 0.6427 |

## Top SQTT Bottlenecks

Current dKV `QUsed-before-softmax` top global bubbles:

| Bubble | Share | Meaning |
|---|---:|---|
| `s_abarrier_try_wait -> s_xor_b32` | 40.55% | ABarrier ownership cliff, not ordinary XOR cost. |
| `s_abarrier_try_wait -> s_waitcnt` | 8.59% | Ownership wait followed by readiness wait. |
| `v_mmac -> v_mmac` | 8.19% | MMAC islands exist but are not enough to dominate runtime. |
| `ds_read_matrix_format -> s_waitcnt` | 3.90% | Read-to-use latency is still exposed. |
| `ds_read_matrix_trans_format -> s_waitcnt` | 2.89% | Trans matrix read wait is also exposed. |
| `s_cbranch_execz -> s_or_b64` | 2.10% | Causal/exec-mask control overhead. |

Compared with the previous `splitwait` baseline, `QUsed-before-softmax`
slightly improves ownership and elapsed time:

- SQTT duration `96,420 -> 94,728`.
- `s_abarrier_try_wait -> s_xor_b32` `41.38% -> 40.55%`.
- PMD `barrierCounter` sum `161,969 -> 157,259`.
- PMD MMAC active `32.9468% -> 33.2391%`.

But it also shifts some cost to source-read readiness:

- `ds_read_matrix_format -> s_waitcnt` `3.26% -> 3.90%`.
- `ds_read_matrix_trans_format -> s_waitcnt` `2.72% -> 2.89%`.
- Consumer branch window grows `189/240 -> 222/240`, leaving little VGPR
  scheduling headroom.

## Code Mapping

- `include/shaobo_instr.h:266`: inline `s_abarrier_try_wait` followed by
  phase `s_xor_b32`.  The hot `s_xor_b32` row is therefore the ABarrier
  ownership wait wrapper, not a standalone ALU problem.
- `include/shaobo_instr.h:278`: builtin ABarrier wait route seen in
  `s_abarrier_try_wait -> s_waitcnt` bubbles.
- `src/dkv_kernel.cpp:1079`: causal `valid_pair` branch inside
  softmax/dS.  It contributes branch/control bubbles, but is not the first
  order limiter.
- `src/dkv_kernel.cpp:1601-1623`: current ReleasePage order:
  dO source read, `DoutUsed`, Q source read, `QUsed`, softmax/dS, then dV/dK
  MMAC.  This is the accepted micro-win path.

## Diagnosis

The current dKV path is a correct, native-matrixized implementation, but the
pipeline is still ownership-bound:

1. Main matrix path is present: `ds_read_matrix` and `v_mmac` are hot, and
   `ldsBankConflict=0`.
2. Bandwidth/cache is not the top limiter for this case: VMEM is tiny, and
   TCC hit/reuse are healthy enough to rule out global-memory-first diagnosis.
3. The hot ABarrier rows mean producer/consumer groups frequently reach packet
   ownership fences before enough useful peer work is available to hide the
   wait.
4. The selected dKV consumer-wave window has `Bubble %=90.89`, while the
   comparable FWD window is `47.92`.  This is the clearest pipeline-quality
   gap.
5. Coissue exists, but much of the coissued VALU is `v_mov_b32` or small
   control work.  It is not the FWD-style useful softmax/MMAC conveyor yet.

## Optimization Implications

Do next:

- Reduce ownership/control per useful MMAC epoch, or increase useful MMAC per
  existing ownership epoch.
- Redesign packet lifetime before more wait-moving.  Isolated wait movement
  has repeatedly shifted cost between `s_waitcnt` and ABarrier without closing
  the FWD gap.
- Keep sidecar LDS-local and main matrices on MLS/BPS + `ds_read_matrix` +
  MMAC.
- Treat causal mask/control simplification as second-order unless it also
  lowers the ABarrier cliff.

Do not prioritize next:

- More source-reg lifetime stretching: `222/240` consumer window is already
  tight.
- Blind ABarrier count splitting: previous finer token/page experiments
  increased scalar/control cadence.
- Pure coissue-count improvements: coissued work must be useful and must lower
  ticks or raise MMAC active.
- Bandwidth-first work: current evidence does not point to global/TCC/LDS
  conflict as the primary bottleneck.

