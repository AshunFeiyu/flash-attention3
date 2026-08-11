# Fused5 Producer Sidecar Load-Ahead

Status: `REJECT_PERF / CANONICAL_SOURCE_RESTORED`

## Hypothesis

Canonical SQTT maps a 3.03% issue-gap family to producer
`global_load_dwordx3 -> s_waitcnt` in `producer_load_raw`. The load currently
issues after four Q/dO BPS matrix loads. Issue the same three sidecar values
first, retain them while the four matrix loads issue, then write the unchanged
LDS sidecar fields. VMEM latency can age behind useful BPS publication.

## Invariants And Gates

- Same global/LDS bytes, addresses, sidecar format and RawFilled ownership.
- Same five GEMMs, MMOP, tile, wave roles and barriers.
- No consumer change and no new matrix path.
- Producer must remain inside its 8-VGPR window; otherwise reject rather than
  enlarge WDRA in this experiment.
- H1/S128 causal/noncausal and H1/S1024 causal correctness, bank0, no spill.
- ASM must place `global_load_dwordx3` before the Q/dO MLS family.
- Promotion requires lower repeated/fullperf ticks and a shorter mapped
  global-load-to-wait SQTT edge.

## Result

- Compiler emitted the intended order:
  `global_load_dwordx3 -> four MLS -> vmcnt(0) -> three ds_write_b32`.
- Producer remained at 8 VGPR; heavy roles stayed 163/166; no spill/scratch.
- H1/S128 and H1/S1024 causal correctness pass, MMOP and bank0 unchanged.
- H1/S1024 fused ticks are `72,306,780`, about 0.36% slower than the accepted
  consumer-zero-hoist stats mean (`72,048,113`).

The three sidecar values remain live across the MLS/exec-mask region. Hiding
part of VMEM readiness does not repay that longer producer path. Reject without
fullperf and restore the accepted consumer-zero-hoist source.
