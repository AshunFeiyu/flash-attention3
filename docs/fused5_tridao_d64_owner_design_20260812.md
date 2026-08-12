# Fused5 Tri Dao D64 Ownership Probe

## Hypothesis

Tri Dao's D128 backward mapping gives each heavy warpgroup the same useful
work: one N64 dK/dV owner and one D64 dQ owner. The current Shaobo path has
two heavy dKV groups plus a separate dQ writer group, so the dQ MMAC island is
thin and its completion token is outside the heavy groups.

This branch tests only the ownership remap. The math remains exactly:

```text
score = Q @ K^T
dP    = dO @ V^T
dV    = P^T @ dO
dK    = dS^T @ Q
dQ    = dS @ K
```

Each consumer group computes its N64 dK/dV slice and its D64 dQ slice. dS is
published once by each group in the existing normal/trans-compatible LDS
page. dK uses the normal view, dQ uses the trans view. No duplicate score/dP,
no scalar matrix reads, no bpermute/gather, and no new GEMM.

## Resource Ledger

```text
waves 0-3   producer: resident K/V + streamed Q/dO/sidecar
waves 4-7   consumer0: N0:64 dK/dV + D0:64 dQ
waves 8-11  consumer1: N64:128 dK/dV + D64:128 dQ
```

The two consumers retain dK/dV FP32 accumulators across the q loop and add a
D64 dQ accumulator. The expected role windows are a bounded `producer /
consumer0 / consumer1` ledger, with no fourth WDRA branch. LDS remains the
existing resident K/V, raw Q/dO, packed dS and sidecar layout until a compile
and correctness gate proves a smaller layout is needed.

## Expected Pipeline

```text
time0: producer publishes K/V and raw Q/dO packet
time1: C0/C1 score + dP MMAC, then softmax/dV
time2: C0/C1 publish dS; normal dK MMAC and trans dQ MMAC use the same page
time3: C0/C1 continue next q tile while dK/dV and dQ accumulators persist
tail:  each consumer stores unique dK/dV N64 and dQ D64 partial
```

The expected gain is not more arithmetic. It is equal useful work per heavy
group and removal of the thin fourth role. Promotion requires correctness,
no private/spill/scratch, bank conflict zero, exact MMOP, and lower same-shape
ticks with higher MMAC active. A compile-only success is not a promotion.

## Decision Boundary

The 12-wave ancestor that gave consumers full-D dQ and atomic output is not
this design and is not a valid comparison. Reject this branch if the D64 dQ
accumulator exceeds the consumer WDRA window, if dQ ownership needs a new
cross-group barrier, or if the PMD shows a correctness failure at S128.
