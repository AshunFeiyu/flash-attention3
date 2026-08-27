# Fused5 Raw/Sidecar Readiness Split

Status: `ACCEPT_TICKS_AND_ACTIVE_MMAC50_OPEN`.

## Change

C85 held `RawFilled` until Q/dO BPS and sidecar global-to-LDS publication
both completed. Commit `07e224d` splits that contract:

- `RawFilled0/1` publishes Q/dO readiness;
- `SidecarFilled0/1` publishes sidecar readiness;
- `RawUsed0/1` remains the only page-reuse guard;
- consumer0 waits for sidecar after score MMAC;
- consumer1 waits after dP and score MMAC.

No formula, tile, matrix path, LDS allocation, output owner or MMOP count
changes. The producer still uses 9/16 VGPR; all symbols remain
private/spill/scratch free.

## Evidence

Same compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7,
H1/S1024/D128 causal:

| Metric | C85 | C109 | Delta |
| --- | ---: | ---: | ---: |
| stats-only fused mean | 39,690,332.5 | 38,729,600 | -2.421% |
| stats-only lifecycle mean | 43,758,715 | 42,822,173.3 | -2.140% |
| fullperf fused | 39,308,360 | 38,771,915 | -1.365% |
| fullperf lifecycle | 43,553,965 | 42,819,140 | -1.687% |
| MMAC active | 39.054060% | 39.563669% | +0.509609 pp |
| XCU `s_waitcnt` latency share | 28.46% | 23.90% | -4.56 pp |
| XCU post-ABarrier `s_xor` share | 22.59% | 20.37% | -2.22 pp |

S128 causal/noncausal and S1024 causal complete correctness pass; bank
conflict, PMD warning and panic counts are zero. MMOP/VALU/LDS/VMEM/FLAT are
unchanged. SCA and failed coissue increase because of the two new tokens, but
the readiness critical path shortens enough to improve both fused and
lifecycle ticks.

Evidence:

- `/zys/sb/fa3b/c109_sidecar_ready_runs`;
- `/zys/sb/fa3b/c109_ab_runs`;
- `/zys/sb/fa3b/c109_sidecar_ready_fullperf/fused5_full/`
  `b1_hq1_hkv1_s1024_d128_c1_fullperf_20260827_102500`;
- `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260827_102500_C109_sidecar_ready_split_H1S1024_causal_SQ7`.

The next optimization must preserve this split and audit the remaining exact
LGKM waits. It must not merge the tokens or move sidecar values across BPS
again.
