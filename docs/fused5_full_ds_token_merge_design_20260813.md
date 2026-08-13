# Fused5 Full-dS Ownership Token Probe

Date: 2026-08-13

## Hypothesis

The canonical kernel publishes dS through two group-local tokens and makes the
dQ writer wait and signal twice per q tile. XCU shows ABarrier wait/xor as the
largest issue-gap family. A full-tile token may remove one writer-side wait and
one writer-side signal without changing the mathematical work or LDS layout.

## Change Boundary

The probe keeps the canonical 16-wave map, M64/N128/D128 tile, five logical
GEMMs, resident K/V, raw Q/dO double pages, and packed dS pages. The only
changed ownership edges are:

```text
consumer0 + consumer1 -> BatchDsFilled(full tile)
dQ writer              -> consumes group0 then group1
consumer0 + consumer1 + dQ writer -> DqDone(full tile)
```

The full token arrival counts are eight dKV waves for `BatchDsFilled` and four
dKV plus four writer waves for `DqDone`. No new token or LDS bytes are added.

## Expected Schedule

```text
time t:   C0/C1 compute and publish their four dS panels
time t+1: dQ writer waits once, consumes C0 then C1 dS panels
time t+2: C0/C1 continue dK and wait for one full DqDone before raw-page reuse
```

## Result

Static/resource gates and H1/S128 plus H1/S1024 full lifecycle correctness
passed. The candidate then produced H1/S1024 fused ticks of `48,544,405` and
`48,721,400`, versus the canonical checkpoint near `47,622,850` to
`47,775,455`. This is a `+1.6%` to `+2.3%` regression. The full-tile wait
removed an early group-local dQ writer start and did not reduce the critical
path.

Decision: `REJECT_TICKS_CANONICAL_RESTORED`. The source is restored to
`2d9bf33`; retain the document as lifecycle evidence only.

## Acceptance

The probe is accepted only if H1/S128 and H1/S1024 full lifecycle correctness,
static resource gates, no spill/private/scratch, and bank-conflict zero pass,
and repeated H1/S1024 ticks do not regress the canonical baseline. A higher
MMAC active value without a tick improvement is not a promotion.

## Rejection Signals

Reject if the full-tile wait delays the writer enough to increase ticks, if the
shared token causes a deadlock, or if any dQ/dK/dV golden check fails. If
rejected, restore canonical source and keep this document as evidence.
