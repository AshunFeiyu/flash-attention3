# Fused5 C84 Remaining Optimization Space

Status: `ANALYSIS_ONLY_CANONICAL_LOCKED`.

## Locked Baseline

- Commit: `3388f47d95cc9a9fa543338782fb46304ebbdd14`.
- Tag: `best/fused5-native-dkv-matrix-store-20260824`.
- Case: B1/H1/S1024/D128, FP16, causal, `GPU_CHIP=sb`, SQ7.
- Compiler/PMD: `e0f10535` / `HEAD1694`.
- Fullperf fused/lifecycle ticks: `39,706,485 / 43,763,265`.
- MMAC active: `38.403324%`; dynamic MMOP: `88,064`.
- Correctness PASS, no private/spill/scratch, `ldsBankConflict=0`.

The fused kernel is about 90.9% of lifecycle ticks. `dot_do_o` and dQ reduce
together are about 9.1%, so the fused kernel remains the primary target.

## Evidence

XCU dispatch 1 covers 128 complete waves. The two heavy consumer slots issue
roughly 1.4K-2.0K MMAC and 1.4K-2.1K VALU instructions per sampled wave, but
only about 14.3%-18.3% of all issue events are MMAC+VALU coissues. This leaves
substantial scheduling headroom even though both instruction classes exist.

Top SQTT issue-gap attribution:

| Gap | Share | Interpretation |
| --- | ---: | --- |
| `s_abarrier_try_wait -> s_xor_b32` | 23.08% | Aggregate ownership waits; `s_xor` is only the first post-wait instruction. |
| trans matrix read -> wait | 7.37% | Operand readiness is exposed before first use. |
| normal matrix read -> wait | 4.31% | Same readiness debt for the normal view. |
| FP16 MMAC -> FP16 MMAC | 7.12% | MMAC islands still contain issue gaps or peer-role phase alignment. |
| sidecar global load -> wait | 5.41% | Producer sidecar publication is serialized by explicit VMEM readiness. |
| terminal ebarrier -> ebarrier | 4.07% | Native dK/dV store reuses one page per owner pair and synchronizes twice per D block. |

The largest ABarrier samples on producer waves map to the final
`RawUsed0/RawUsed1` waits after all raw packets have been published. They show
that the producer becomes idle early, but they are not automatically reclaimable
kernel ticks: consumer waves on the same SIMD can still issue useful work.
Token-specific latency must therefore be measured before changing the mainloop.

## Ranked Hypotheses

### H1: Batch the native dK/dV store pages

Current code performs two CTA-wide ebarriers for each of four D blocks. Reuse
more of the released V region so two D blocks can be published before their
matrix stores, reducing terminal synchronization generations without changing
the five-GEMM mainloop or output ownership.

- Expected scope: bounded epilogue-only change.
- Plausible gain: about 1%-3% fused ticks; not enough alone for 50% MMAC active.
- Gate: exact dK/dV, bank0, unchanged MMOP, no new spill, fewer dynamic
  `s_ebarrier_sync` and lower terminal issue-gap duration.

### H2: Token tomography before ownership redesign

Instrument or isolate the 12 ABarrier IDs to separate `RawFilled/RawUsed` from
`BatchDsFilled/DqDone`. Do not optimize the aggregate 23.08% number directly.
The redesign should target the token with the largest consumer-critical wait,
not producer epilogue idleness.

- If `RawFilled` dominates consumers: move one useful next-packet publication
  earlier without adding a third raw page.
- If `DqDone` dominates dKV consumers: lengthen the dS/dQ conveyor or reduce
  reuse generations while preserving group-local ownership.
- If only `RawUsed` dominates producers: no structural change is justified
  unless SIMD pipeline data also shows consumer starvation.

### H3: One-family matrix-read lookahead

Trans and normal reader readiness account for 11.68% of issue-gap duration.
Issue one complete operand family before independent softmax/dS work, then
wait only at first MMAC use. Apply to one consumer group first to create useful
phase offset and avoid increasing both groups' live ranges together.

- Expected scope: one read helper and one consumer schedule.
- Plausible gain: about 1%-4%, based on earlier accepted read-ahead results.
- Stop condition: VGPR role growth, wait reduction without ticks reduction, or
  worse MMAC+VALU coissue.

### H4: Sidecar load latency hiding

Sidecar data is already loaded by producer into LDS, so moving it from consumer
global reads is not applicable. The remaining 5.41% gap is the producer's
`global_load_dwordx3 -> wait` before publishing a raw page. Only overlap that
VMEM readiness with already useful MLS or ownership work; do not add another
sidecar path or consumer global loads.

- Plausible gain: below 2% unless token tomography proves this wait delays
  `RawFilled` on the critical consumer path.

## Closed Directions

- FP32 dQ DS-write/matrix-store: correct native ABI, but about 11% slower.
- More raw buffers without a lifetime proof: LDS is already 128 KiB and earlier
  buffer growth increased ABarrier/live-range cost.
- Blind wait deletion: matrix reader first-use waits and BPS `vbcnt` are proven
  readiness requirements.
- Optimizing `s_xor_b32`: it is attribution after ABarrier, not the cause.
- More MMAC work or smaller tiles: the five-GEMM ledger is exact at 1,280 MMAC
  per tile; adding work would inflate active share without improving time.

## Ceiling Assessment

At fixed MMOP, moving MMAC active from 38.40% to 50% requires the measured
critical duration to fall to about 76.8% of its current value, roughly a 23.2%
reduction. No single instruction-level edit has that range. Reaching 50%
requires a structural reduction in consumer-critical ownership and matrix-read
gaps, while the terminal-store and sidecar changes are supporting gains.

The next experiment should be H1 because it is bounded and directly exposed by
C84. In parallel, collect token-specific ABarrier evidence for H2; H2 is the
only current hypothesis with enough potential to change the MMAC-active ceiling.
