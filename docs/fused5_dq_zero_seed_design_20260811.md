# Fused5 dQ First-MMAC Zero Seed

Status: `OBSERVE_INSTRUCTION_WIN / PERF_NEUTRAL / CANONICAL_RESTORED`

## Hypothesis

The dQ writer currently clears eight FP32 accumulator vectors for every
q-tile, then immediately overwrites each vector through its first dQ MMAC.
Keep one branch-local four-VGPR zero value and use it only as the accumulator
operand of each fixed first MMAC. All later MMACs remain in-place.

For H1/S1024 causal, the eight K CTAs execute 72 q-tile iterations in total.
Four dQ writer waves currently issue about `72 * 4 * 8 * 2 = 4608`
`v_mov_b64` instructions for per-tile accumulator clearing. The replacement
initializes one zero vector per writer wave at role entry.

## Invariants

- Exactly five logical GEMMs and dynamic MMOP92,160.
- Same M64/N128/D128 tile, 16-wave role map and output ownership.
- Same LDS115,456B and ABarrier ledger.
- No new branch in the q-loop after compile-time unrolling.
- No ordinary matrix DS read, gather, permute or extra memory traffic.

## Gates

1. ASM must show lower `v_mov_b64` count and unchanged MMAC count.
2. Role usage must remain inside `8/200/200/88`; private/spill/scratch stay 0.
3. Full lifecycle H1/S128 and H1/S1024 causal correctness must pass, bank0.
4. Compare same-build S1024 ticks and PMD instruction counts against restored
   canonical `72,254,455` ticks and `VALU169,376`.
5. Promote any repeatable tick reduction; otherwise reject and restore.

Only after A1-A6 operator evidence may the zero-seed helper be proposed for
the gfx946 probe/header library. This experiment does not modify the admitted
MLS32 dual-view contract.

## Revision B

Revision A reduced dynamic VALU by 4,544 but held the zero vector across the
whole q-loop, raising dQ role use from 84 to the 88-VGPR window limit. Two
paired S1024 comparisons were performance-neutral (mean delta about +0.007%).
Revision B scopes the zero to one q-tile. It should retain most move reduction
while shortening the four-VGPR live range. If role usage or ticks do not
improve, close the hypothesis and restore canonical source.

## Result

- Revision A: static `v_mov_b64 208 -> 194`, dynamic VALU
  `169376 -> 164832`, but dQ role use rose `84 -> 88`. Two paired runs were
  performance-neutral.
- Revision B restored dQ role use to 84 and passed H1/S128 plus H1/S1024.
  Two stats runs were `72,117,955` and `72,092,020` ticks versus canonical
  median `72,254,455` (`-0.21%`).
- Fullperf did not confirm a tick win: `72,132,060 -> 72,138,430`
  (`+0.009%`). It reduced `v_mov_b64 43,520 -> 39,488`, reduced total VALU
  `169,376 -> 165,344`, and raised MMAC active
  `22.106908% -> 22.141693%`, but atomic issue-gap share rose
  `14.06% -> 14.57%` and absorbed the local saving.

This is useful instruction evidence but not a canonical performance win.
Restore the source exactly to `c58272f`. Do not admit a generic gfx946
zero-seed helper to the shared header library from this result; A6 is neutral.
