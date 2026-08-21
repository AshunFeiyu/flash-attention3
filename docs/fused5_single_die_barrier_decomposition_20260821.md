# Fused5 Single-Die Barrier Decomposition

## Baseline

- Shape: B1/H1/S1024/D128/causal, `GPU_CHIP=sb`, SQ7.
- Fused ticks: 44,988,125; MMAC active: 33.9132%.
- Exact work: MMOP92,160; correctness PASS; bank0; no spill/scratch.
- XCU representative SIMD: slot0 producer, slot1 C0, slot2 C1, slot3 dQ
  writer. Dispatch has zero no-wave idle cycles.

## Role-Local ABarrier Waits

| Role | Startup | Steady pattern | Interpretation |
|---|---:|---:|---|
| producer | 315 cycles | 16 waits, about 3.9--4.9K each | waits for RawUsed, but peer heavy waves keep the SIMD active |
| C0 | 1,471 then 639 | later waits are normally 3 cycles | raw/dS ownership is ready in steady state |
| C1 | 819 then 519 | later waits are normally 3--52 cycles | not a steady ownership blocker |
| dQ writer | 827 then 4,175 | alternates about 3--400 and 1.0--1.7K | G0 MMAC covers G1 publication; second short wait is earned overlap |

The aggregate `s_abarrier_try_wait -> s_xor_b32` share of 25.94% therefore
cannot be read as 25.94% removable wall time. Most producer wait is hidden by
other roles, and the writer's short G1 waits depend on executing G0 first.

## Closed Inferences

1. Two-q-tile writer reordering delays ready G1 behind G0-next and regresses
   fused ticks 8.17%.
2. Reversing complete writer order to G1-first exposes G1 publication and
   regresses fused ticks 1.25%.
3. Covering writer startup zeroing is only a 0.29% unstable observation.
4. Pre-latching a q tile of probability sidecar values raises wait-LGKM and
   regresses fused ticks 0.79%.

## Next Evidence Boundary

Do not optimize aggregate ABarrier percentage directly. Preserve the G0-first
writer stagger and the two-generation C0 page. The next candidate must target
one repeated matrix first-use chain or a proven CTA-tail store dependency,
and must show how useful work occupies that exact interval before source is
changed.

