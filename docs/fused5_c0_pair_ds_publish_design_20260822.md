# C0 pair-level dS publication

## Hypothesis

The canonical C0 consumer computes all four M16 panels before publishing any
dS to the dQ writer. The writer therefore cannot begin its C0 dQ MMAC until the
entire C0 score/dP/probability/dS chain completes. Split only the C0 Filled
event into low and high panel pairs so the writer consumes panels 0-1 while C0
computes panels 2-3.

This is a structural publication-boundary experiment, not another matrix-read
or wait-placement experiment. The canonical Q-next lookahead remains intact.

## Exact work and ownership

- Tile remains `M64/N128/D128`, 16 waves and exactly five GEMMs.
- C0/C1 dKV waves retain unique N16 dK/dV ownership.
- dQ writer waves retain unique D32 ownership.
- Each C0 panel costs 8 score plus 8 dP MMAC per C0 wave.
- Each two-panel writer island costs 16 dQ MMAC per writer wave.
- Dynamic MMOP must remain exactly 92,160 for H1/S1024 causal.
- LDS addresses, bytes and native `ds_write_matrix -> ds_read_matrix` layout
  remain unchanged.

## ABarrier revision

Add only two tokens:

- `C0LowFilled generation0`, count 4.
- `C0LowFilled generation1`, count 4.

The existing C0 Filled tokens become the high-pair events. Existing DqDone
tokens remain full-generation count-8 rendezvous events: four C0 waves arrive
after dK, and four writer waves arrive after consuming the high pair. C0 waits
that full event before overwriting the low half on generation reuse. Because
page0/page1 generations alternate, a generation is reused only every two q
tiles.

Pair-level DqDone was rejected during design because it adds four tokens and
fragments ownership without creating an earlier first-use event.

## Expected steady schedule

1. C0 computes panels 0-1 while the writer finishes prior work.
2. On reuse, C0 waits the already-aging full DqDone, writes low dS, and signals
   LowFilled.
3. Writer reads low dS and performs 16 dQ MMAC per wave.
4. In parallel, C0 performs 32 score/dP MMAC per wave plus probability/dS VALU
   for panels 2-3.
5. C0 writes high dS and signals the existing Filled event.
6. Writer consumes high dS while C0 runs dV/dK, then both sides arrive at the
   existing DqDone event.

The useful target is C0 MMAC/VALU overlapping writer MMAC on the same SIMD.
No artificial delay is admitted.

## Admission and stop gates

- Static: one canonical fused symbol, 14 contiguous barrier IDs, exact MMAC,
  no main-path ordinary DS read, no scratch/private/spill.
- Correctness: S128 causal/non-causal, then S1024 causal CPU golden.
- Runtime: bank conflict zero and dynamic MMOP 92,160.
- Performance: same-build paired H1/S1024 fused ticks must improve. Higher
  coissue without lower ticks is a rejection.
- SQTT: capture only after the same-shape ticks gate passes; it must show the
  low writer MMAC island overlapping C0 high-panel work.

Workbook design: sheet `40 C0 Pair dS Publish`.

## Pre-implementation de-dup result

Status: `REJECT_DUPLICATE_HISTORICAL_EVIDENCE_NO_RUN`.

Before compilation, the structural fingerprint matched the 2026-08-19
`fused5_c0_ds_half_publish` experiment exactly: M64/N128/D128, 16 waves,
pair-level Filled events, one full-generation Done event, 14 ABarrier IDs,
unchanged LDS and five-GEMM work. Reversing which token is named low/high does
not change that lifecycle.

That experiment already passed full correctness, bank0 and resource gates at
C0 169/204 VGPR, SGPR76/VGPR128 and no spill/scratch. Three paired S1024 runs
regressed fused mean ticks `44,718,765 -> 45,857,327` (`+2.546%`) and MMAC
active `34.211% -> 33.949%`. Barrier share improved, but wait-LGKM and wait-VM
rose because the early writer island competed with C0/C1 for LDS and MMAC issue
slots.

The uncompiled candidate source was removed and canonical checksums restored.
Do not split C0 dS below the four-panel batch on this 16-wave topology.
