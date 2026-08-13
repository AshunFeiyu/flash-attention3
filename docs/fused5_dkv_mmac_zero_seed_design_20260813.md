# Fused5 dKV MMAC Zero Seed

Status: `HYPOTHESIS_OPEN`

## Evidence

Canonical fused5 SQTT reports `v_mov_b64_e32=6208` and the generated source
still explicitly clears `dv_acc[8]` and `dk_acc[8]` before the q-loop. The
same compiler/runtime has accepted dKV zero-seed specialization in the
historical owner16 kernel.

## Hypothesis

The first q tile is guaranteed to contribute to every dV/dK accumulator. Use
the already-created `mmac_zero` FP16 fragment as the accumulator input for
that first dV and dK MMAC island, then use the accumulator registers for all
later q tiles. This should remove the standalone FP32 accumulator zeroing
island without changing the five GEMMs or numerical operation order.

## Implementation Boundary

- Specialize only the first q tile with `FirstAccum=true`.
- Keep the remaining q tiles on the existing accumulator path.
- Do not add a runtime first-update branch inside a long-lived MMAC loop.
- Do not change score/dP zero-seeding, tile shape, roles, LDS, or ABarrier.
- Keep main matrix movement MLS/BPS + `ds_read_matrix` + MMAC.

## Expected Schedule

```text
first q tile:  score/dP -> softmax/dS -> dV(MMAC zero seed) -> dK(MMAC zero seed)
later q tiles: score/dP -> softmax/dS -> dV(accumulate) -> dK(accumulate)
```

## Gates

Static/resource, H1/S128 full lifecycle, H1/S1024 full lifecycle, and bank
conflict zero are required. Promotion requires lower repeated H1/S1024 fused
ticks and fewer zero moves without compensating copies, extra MMAC, or spills.

## First Result: Runtime-Branch Rejected

The first implementation removed the explicit dV/dK accumulator clears and
selected `FirstAccum` with `qi == 0 && m_block == 0`. It passed static/resource
gates and H1/S128 plus H1/S1024 correctness with bank0, but generated code
grew the dKV consumers from `161/163` to `163/165` VGPRs and increased the
MMAC instruction body from `320` to `400` static instructions. H1/S1024
fused ticks were `47,946,080` versus canonical `46,637,955` (`+2.8%`).

Classification: `REJECT_TICKS_CANONICAL_RESTORED`. This is a counterexample
to runtime first-update branching. A future zero-seed attempt must peel the
first q tile at compile time or remain rejected.
