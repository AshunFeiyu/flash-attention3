# Fused 5-GEMM Q-Latch Early Raw Release

## Decision

Status: `ACCEPT_CANONICAL_Q_LATCH_EARLY_RAW_RELEASE / MMAC50_OPEN`.

Commit `d7308d4` latches all four Q normal panels into consumer VGPRs after
dV has consumed dO. The consumer then releases the combined raw Q/dO page
before final dS publication, dK, dQ and dQ atomics. The producer can therefore
publish the next raw packet while the consumers finish useful work from the
current iteration.

This is a lifetime/schedule change, not a mathematical change:

- score, dP, dV, dK and dQ remain the only five GEMMs;
- dynamic useful MMOP remains exactly `92,160`;
- score and dP are not recomputed;
- dS is still published once for normal dK and transposed dQ readers;
- the P scratch bridge remains open debt.

## Gates

- H1/S128 causal and noncausal: PASS.
- H1/S1024 causal: PASS.
- dK: `max_abs=5.55674e-06`, `rel_l2=2.77289e-4`.
- dV: `max_abs=1.94430e-4`, `rel_l2=4.61692e-4`.
- Metadata: SGPR100, VGPR168, LDS115,456 B.
- Private segment, scratch, SGPR spill and VGPR spill: zero.
- PMD panic and uninitialized-VGPR warning: zero.
- `ldsBankConflict=0`.
- Main matrix path remains MLS/BPS, `ds_read_matrix` and MMAC.
- The 24 generated `ds_read_b32` instructions map only to the three scalar
  sidecar fields at `src/fused_bwd_kernel.cpp:440-442`; none feeds a matrix
  operand.

## Performance

Locked environment:

```text
shape       B1/H1/S1024/D128 causal
GPU_CHIP    sb
GPU_ARGS    ['--SQCIPfLines=7']
compiler    LLVM e0f10535a0d681bcf3885ea2c398cc494bf6e332
PMD         HEAD1694, config seed sha256 c22d6a42...
```

Three paired stats-only runs:

```text
median control ticks     101,732,540
median candidate ticks   100,594,585
delta                    -1.119%
median control active    16.819793%
median candidate active  17.017960%
```

Fullperf:

```text
metric                 control       candidate      delta
kernel ticks           102,105,640   101,053,680    -1.030%
MMAC active            16.817606%    16.978666%     +0.161060 pp
MMOP                    92,160        92,160         0
s_waitcnt issues       30,656        29,504         -1,152
wait VM share          3.374652%     1.808978%      -1.565674 pp
wait LGKM share        11.752119%    11.586903%     -0.165216 pp
ABarrier share         23.536976%    27.279382%     +3.742406 pp
SQTT duration          224,408       222,096        -1.030%
```

## SQTT Interpretation

The top producer bubble remains `RawUsed`, barrier id4. It grows from
`11,687` to `12,099` cycles because the producer reaches the next-generation
wait earlier, not because the consumers stop progressing.

In the selected same-SIMD window:

```text
control:   1,325 issued instruction cost, 163 MMAC, 96.23% bubble
candidate: 1,407 issued instruction cost, 186 MMAC, 96.12% bubble
```

Thus the longer producer wait overlaps more consumer MMAC/VALU work, while
whole-kernel ticks fall. The higher aggregate barrier share is not itself a
critical-path regression. The batch schedule also removes 1,152 first-use
waits by issuing the Q and final-dS matrix-read families before their common
waits.

This remains far from the 50% useful-MMAC target. The producer is still thin,
the raw protocol has only one physical page, dQ atomics remain a 17-19% SQTT
latency family, and direct P-to-dV/dS-to-dK register chaining is still blocked
by the current q-owned fragment ABI.

## Evidence

- Fullperf:
  `/zys/sb/fa3b/q_latch_fullperf_20260723/5gemm_owner_s1024_c1_fullperf_20260723_161812`
- XCU:
  `/zys/sb/fa3b/xcu_outputs/5gemm_q_latch_s1024_20260723`
- Shared archive:
  `/共享/shaobo/perf/20260723_161812_fused5_q_latch_early_raw_release_h1s1024_sqc7_fullperf`
- Helper perf SHA256:
  `ccb045d84d37232b2ba29e28e802f51487a569c7ecc4c2891c29f28f58602c14`

## Next Gate

Preserve this early release and attack the remaining structural gap. Do not
add another Q/dO LDS page while the P scratch bridge keeps the budget above
128 KiB. The next design must either:

1. produce a native fragment ownership that directly feeds P to dV and dS to
   dK while retaining one legal dS publication for dQ, or
2. prove a producer-group dQ writer/lag-one schedule that hides required
   atomics without adding duplicate score/dP or extra ownership epochs.
