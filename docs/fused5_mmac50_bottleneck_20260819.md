# Fused5 MMAC50 Bottleneck Checkpoint

## Measured Baseline

Environment: compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7,
H1/S1024/D128 causal. The accepted FP16 pair-load fullperf reports:

- fused ticks: `45,663,345`;
- MMOP: `92,160` exactly;
- MMAC active: `34.586975%`;
- wait-VM / wait-LGKM / barrier: `2.291% / 7.505% / 13.447%`;
- coissue success/fail: `20,190 / 21,386`;
- LDS bank conflicts: zero.

At fixed useful MMOP, reaching 50% MMAC active requires the effective active
window to shrink to about `34.587 / 50 = 69.17%` of its current size, or a
roughly 30.8% reduction. This cannot come from reducer or store micro-tuning.

## XCU Evidence

Dispatch-1 detail attributes the largest issue gaps to:

| Chain | Issue-gap share |
|---|---:|
| `s_abarrier_try_wait -> s_xor_b32` | 26.75% |
| trans matrix read -> wait | 10.41% |
| terminal ebarrier -> branch | 6.98% |
| FP32 MMAC -> FP32 MMAC | 5.43% |
| normal matrix read -> wait | 3.96% |

One representative SIMD maps wave slots to producer, consumer0, consumer1,
and dQ writer. The trace shows:

- consumer startup resident waits of about 0.86--1.00k cycles;
- first raw-page waits of about 0.32--0.51k cycles;
- dQ writer's first dS wait of about 3.76k cycles;
- writer and producer carry far fewer issued instructions than the two heavy
  consumers, but their late waits/stores extend the CTA terminal window.

The aggregate ABarrier row mixes resident, raw and dS/dQ ownership waits; it
must be split by role and generation before changing token topology.

## Consequence

The dQ reduction path is now about 3.3% of H1/S1024 and 4.6% of H1/S2048.
Further reducer tuning cannot raise fused MMAC active. The next main-kernel
candidate must remove a lifecycle edge or overlap it with useful peer work.

Closed directions remain closed unless a new mechanism changes their premise:

- M128 lexical halves: two ownership-tier failures;
- split/merged dS token cadence: moved waits onto the pace-setting consumer;
- dV/writer read-ahead: lowered local wait but regressed CTA ticks/coissue;
- synchronous read-count permutations without a longer useful overlap window.

## Next Admission

Before the next source change, annotate or isolate dynamic ABarrier IDs in
SQTT and total exposed cycles for `ResidentFilled`, `RawFilled/RawUsed`,
`BatchDsFilled`, `DqDone`, and terminal ebarrier by wave role. Select exactly
one dominant ownership edge. A candidate is admitted only if its timeline
shows useful MMAC/VALU/MLS work occupying that edge; empty staggering and
token-count-only changes are forbidden.

H100 MFU and Shaobo MMAC active are complementary but not identical metrics.
The competitive goal is valid, but a reported 60% H100 MFU does not prove a
60% profiler tensor-core-active counter without matching definitions, FLOP
accounting, clocks and causal treatment.
