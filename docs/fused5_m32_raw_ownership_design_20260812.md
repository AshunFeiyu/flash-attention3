# Fused5 M32 Raw Ownership

Status: `REJECT_ABARRIER_GENERATION_CANONICAL_RESTORED`.

## Hypothesis

Keep the canonical M64/N128/D128 tile and 16-wave roles, but split each raw
M64 packet into two independently owned M32 subblocks. A `matrix_load_32x32`
already covers two M16 panels, so M32 is the smallest producer granularity
without changing the native MLS layout.

```text
current M32: score/dP -> softmax/dV/dS -> dK(two M16 panels)
             -> RawUsed(subblock)
producer:    refill the same subblock for q+2 while the other subblock runs
```

Only the Q/dO raw page ownership cadence changes. The exact five GEMMs,
resident K/V, dS publication and unique output owners remain unchanged.

## Barrier And Resource Ledger

Reuse existing page tokens, but advance each page token twice per q tile:

```text
RawFilled(page): producer publishes M32-0, then M32-1
RawUsed(page):   dKV releases M32-0, then M32-1
BatchDsFilled:   dKV publishes two dS subblocks, dQ consumes two waits
DqDone:          dQ releases two dS subblocks, dKV consumes two waits
```

No new barrier IDs, no new LDS bytes, no duplicate GEMM. Static resource
windows are unchanged. The main risk is additional ABarrier control and
phase bookkeeping; the candidate is useful only if the producer RawUsed gap
shrinks more than this control cost.

## Expected Pipeline

```text
time0: P publishes raw M32-0; C0/C1 consume it
time1: C0/C1 dK(M32-0); P waits RawUsed(M32-0), loads next q M32-0
time2: C0/C1 dK(M32-1); P waits RawUsed(M32-1), loads next q M32-1
time3: C0/C1 consume the next page while P ages the following packet
```

## Gates

Static exact-work/resource gates, H1/S128 causal/noncausal correctness, then
H1/S1024 correctness. Promotion requires lower repeated fused ticks and a
lower or hidden producer RawUsed bubble; a higher MMAC active value alone is
not sufficient.

## Result

The first implementation called `RawFilled(page)` twice before the matching
consumer release. PMD did not queue two unconsumed generations and the S128
smoke hung in the second generation. A strict repair added
`subblock0: Filled -> Used -> subblock1: Filled -> Used`, but still required a
full barrier round-trip between the two halves, so it could not realize the
intended producer overlap. The implementation was removed before timing or
correctness promotion. This closes the single-token M32 shape: without two
independent token IDs, it adds control epochs rather than hiding them.

Decision: `REJECT_ABARRIER_GENERATION_CANONICAL_RESTORED`. Keep the canonical
M64 raw ownership and target a topology that amortizes an existing token
instead of issuing two generations on one token.
