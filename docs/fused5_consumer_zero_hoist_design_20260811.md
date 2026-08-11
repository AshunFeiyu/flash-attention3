# Fused5 Consumer MMAC Zero Hoist

Status: `ACCEPT_MICRO_TICKS / CANONICAL_CANDIDATE`

## Hypothesis

Every score/dP `f16_mmac_single` call currently creates a fresh four-VGPR zero
fragment. SQTT attributes 13,824 `MMAC -> v_mov_b64` and 13,824
`v_mov_b64 -> MMAC` issue-gap edges to this fixed island. Initialize one zero
fragment per heavy consumer role and reuse it as the first-MMAC seed for all
score/dP panels.

The heavy roles use 165/168 VGPR inside 200-VGPR windows, so this four-VGPR
live value has static headroom. It does not change the physical WDRA windows.

## Invariants And Gates

- Five GEMMs, MMOP92,160, M64/N128/D128 and the 16-wave roles are unchanged.
- LDS115,456B, ABarrier ownership and output stores/atomics are unchanged.
- Static MMAC count remains 320; `v_mov_b64` must fall.
- Heavy roles remain under 200 VGPR; no private/spill/scratch and bank0.
- H1/S128 and H1/S1024 complete lifecycle correctness pass.
- Promotion requires lower repeated S1024 ticks and fullperf confirmation.

If admitted, record the reuse pattern as a gfx946 knowledge/probe candidate.
Do not add an operator-independent header unless A6 demonstrates a tick win.

## Result

- Static `v_mov_b64 208 -> 84`; static MMAC remains 320.
- Heavy-role VGPR improves `165/168 -> 163/166`, rather than increasing.
- H1/S128 causal/noncausal and H1/S1024 causal complete lifecycle PASS;
  MMOP92,160, LDS115,456B, bank0, no private/spill/scratch.
- Two stats S1024 runs are `72,208,045` and `71,888,180` ticks.
- Fullperf improves `72,132,060 -> 71,950,060` (`-0.2523%`).
- MMAC active improves `22.106908% -> 22.725077%`; VALU falls
  `169,376 -> 120,800`; XCU issued instructions fall
  `429,136 -> 375,952`.
- Coissue success falls because removed moves no longer coissue with MMAC. This
  is not a useful-overlap regression: exact MMOP is unchanged and total ticks
  decline. Atomic issue-gap share also falls `14.06% -> 12.60%`.

Promote this source change. The next bottleneck remains ABarrier ownership
(`29.58%`) plus matrix-read first-use waits and terminal convergence; do not
continue deleting moves without a mapped source edge.
