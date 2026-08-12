# Fused5 Q Latch Before RawUsed

Status: `REJECT_TICKS_CANONICAL_RESTORED`.

## Hypothesis

The accepted fused5 path keeps the raw Q/dO page owned until the dK MMAC
island finishes:

```text
publish dS -> read Q/dS -> dK MMAC -> RawUsed -> producer refills raw page
```

Only Q is still needed by dK. dO and sidecar are dead after dV/dS, and dS is
in the released K/V scratch region. Read all four Q panels into the consumer
VGPR window, wait for those LDS reads, and release `RawUsed` before the dK
MMAC island:

```text
publish dS -> read all Q panels -> wait lgkm -> RawUsed
                                      | producer loads next Q/dO/sidecar
                                      + dK reads dS and executes MMAC
```

This changes only raw-page ownership cadence. It does not duplicate score or
dP, add a GEMM, change output ownership, or introduce a new barrier token.

## Resource Ledger

Current heavy consumer use is `163/165` VGPR. The existing dK read-ahead keeps
8 Q fragments and 2 dS fragments live. The candidate keeps 16 Q fragments and
2 dS fragments, an estimated increase of 32 VGPR:

```text
consumer0: 163 + 32 = 195 <= 204
consumer1: 165 + 32 = 197 <= 204
producer/dQ writer: unchanged 9/86
LDS: 131072 B, bank contract unchanged
MMOP: 92160 at H1/S1024, exact five GEMMs
```

The generated metadata is authoritative. Any spill, private segment, or
heavy role above 204 rejects the candidate before PMD.

## Ownership Proof

1. `RawFilled(page)` guarantees the producer's Q/dO page is visible.
2. score, softmax, dV, dP and dS consume dO/sidecar before the Q latch.
3. all Q matrix reads are issued and `lgkmcnt(0)` completes before `RawUsed`.
4. after `RawUsed`, no instruction in the consumer reads the raw page again;
   dK reads only the dS scratch page and the latched Q fragments.
5. producer may therefore overwrite the raw page while dK MMAC runs.
6. `BatchDsFilled`/`DqDone` remain unchanged for dS ownership.

## Gates

- static fused kernel and symbol metadata gates pass;
- H1/S128 causal and noncausal correctness pass;
- H1/S1024 causal correctness pass;
- exact MMOP, no spill/private/scratch, LDS bank conflict zero;
- promotion only if repeated H1/S1024 fused ticks decrease and the SQTT/stats
  explain a lower RawUsed/RawFilled barrier interval.

## Result

Static and resource gates passed at role use `9/186/186/86`, with no private
segment, spill, scratch, or bank conflict. H1/S128 and H1/S1024 full
correctness passed. The S1024 candidate nevertheless regressed:

```text
canonical: 47,757,255 ticks, MMAC active 33.152365%
candidate: 47,875,100 ticks, MMAC active 33.069957%
delta:     +0.2467% ticks, -0.0824 percentage points active
```

The candidate raised `waitLgkm` from `10.134689%` to `11.186619%`, barrier
share from `14.686485%` to `15.238795%`, and reduced coissue success from
`21,354` to `20,985`. Reading all Q panels before the release fence creates a
full `lgkmcnt(0)` dependency that the existing lag-one read-ahead hides more
effectively. Restore the canonical two-slot Q/dS read-ahead and do not repeat
this full-Q latch shape.

Decision: `REJECT_TICKS_CANONICAL_RESTORED`.
