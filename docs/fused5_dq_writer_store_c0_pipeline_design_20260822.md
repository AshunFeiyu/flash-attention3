# Fused dQ writer store-to-next-C0 pipeline

## Hypothesis

The dQ writer was the latest sampled role. Its representative SQTT interval
contained about 8,355 cycles attributed to global-store issue gaps and exposed
waits of about 4,099 and 1,991 cycles on the next C0 generation. The candidate
therefore tried to store two current-tile accumulator panels, wait for the next
C0 generation, and reuse each remaining panel immediately after its current
store issued.

The experiment kept exactly five GEMMs, the same LDS pages and ABarrier IDs,
and the same dynamic dQ store count. It did not stack the rejected writer zero
seed. The paired control and candidate both used WDRA `16/196/196/104`.

## Result

Decision: `REJECT_SHORTENS_READY_AGING_AND_EXPANDS_WRITER_CONTROL_CANONICAL_RESTORED`.

All S128 causal/non-causal and both paired S1024 runs pass the complete cached
CPU-golden lifecycle. Metadata is SGPR82/VGPR128, private/spill/scratch zero,
and LDS bank conflict remains zero. Dynamic fused MMOP remains exactly 92,160,
so no GEMM was duplicated or removed.

| Pair | Same-WDRA control fused ticks | Candidate fused ticks | Delta |
| --- | ---: | ---: | ---: |
| 1 | 44,787,925 | 45,292,520 | +1.127% |
| 2 | 44,618,210 | 45,220,175 | +1.349% |
| Mean | 44,703,068 | 45,256,348 | +1.238% |

Full-lifecycle mean regresses from 48,824,458 to 49,409,588 ticks
(`+1.198%`). In the first pair, dynamic MMOP/LDS/VMEM/FLAT are unchanged,
while the candidate adds 700 SCA and 1,336 VALU instructions and raises
`noVALUready` by 6,073 cycles. Coissue success increases, but ticks regress;
the extra coissue is not useful throughput.

## Corrected pipeline interpretation

Canonical already executes all eight current-tile dQ partial stores before it
waits for the next C0 generation. That entire store interval ages the next
`BatchDsFilled0` event. The candidate waits after only the first two panel
stores, shortening rather than extending the readiness cover. Reusing panels
inside the transition also duplicates static generation-specific control:
the compiled symbol grows from 832 to 896 static MMAC sites, 840 to 872 matrix
reads, 31 to 33 ABarrier sites, and 24 to 32 `global_store_dwordx2` sites even
though dynamic mathematical work is unchanged.

This is not a PMD defect. The candidate changed the critical recurrence in a
measurably harmful way. Canonical source and WDRA are restored. Do not retry
store splitting unless SQTT proves a readiness event is already signaled
before the canonical writer begins its store island.

Evidence root:
`/zys/sb/fa3b/writer_store_c0_20260822/paired`.
