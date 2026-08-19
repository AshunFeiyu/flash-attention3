# Fused5 Writer Lag-One After C0 Gen2

## Evidence

The accepted C0 two-generation baseline moves the representative CTA tail to
the dQ writer: producer `89092`, C0 `91436`, C1 `95136`, writer `96028`.
Within the writer, `ds_read_matrix_trans -> wait` contributes `23164` cycles,
the largest non-ABarrier local debt. Writer ABarrier debt is already reduced
from `33219` to `21983` cycles by the accepted ownership change.

The earlier writer lag-one experiment reduced wait-LGKM but regressed ticks
because the writer was not the CTA pace setter. That premise has changed.

## Single Hypothesis

Within each source group, keep two four-fragment dS packets. Issue panel `m+1`
before the eight-MMAC island for panel `m`, and wait with `lgkmcnt(4)` so the
current packet is ready while the next packet remains in flight. Preserve the
existing group0-then-group1 order, ownership tokens, arithmetic, and stores.

The writer needs one additional packet. Rebudget WDRA from `16/204/204/88` to
`16/196/196/104`; the accepted measured roles are approximately
producer/C0/C1/writer `9/187/182/87`, and the physical sum remains 512.

## Gates

- Exact five-GEMM MMOP count and numerical correctness are unchanged.
- No private segment, spill, scratch, or LDS bank conflict.
- Same-build S1024 fused ticks must improve against the accepted C0-gen2
  binary; S2048 must confirm scaling before promotion.
- SQTT must show lower writer trans-read wait without increasing ownership or
  MMAC contention enough to erase the tick gain.

## Result

Status: `REJECT_LDS_MMAC_CONTENTION_CANONICAL_RESTORED`.

The intended ASM is present: four next-panel transposed matrix reads,
`lgkmcnt(4)`, then eight contiguous dQ MMAC. Actual roles are
producer/C0/writer/C1 `9/187/101/182` in `16/196/104/196` windows; metadata is
SGPR76/VGPR128 with private/spill/scratch0. S128 causal/noncausal and all three
S1024 pairs pass complete correctness with bank0.

Across three alternating S1024 pairs, fused ticks regress
`44,647,482 -> 45,217,445` (`+1.277%`) and lifecycle ticks regress
`48,796,930 -> 49,350,362` (`+1.134%`). The changed critical-path premise is
not sufficient: the additional in-flight writer LDS reads compete with the
two heavy consumers and increase CTA completion time. Restore the accepted C0
two-generation source and close writer panel read-ahead on this topology.
