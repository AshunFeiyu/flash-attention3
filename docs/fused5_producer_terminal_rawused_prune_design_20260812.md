# Fused5 Producer Terminal RawUsed Prune

Date: 2026-08-12

## Hypothesis

After the producer publishes all raw Q/dO/sidecar pages, it currently waits
for both `RawUsed0` and `RawUsed1` even though it will not overwrite either
page again. The final CTA EBarrier already waits for all roles before barrier
invalidation. Remove only the terminal `RawUsed1` wait and retain the terminal
`RawUsed0` wait as a conservative control.

This is a lifecycle-only wait audit. It does not change any page reuse edge,
matrix instruction, GEMM count, output ownership, or dQ schedule.

## Admission

The candidate is useful only if H1/S128 and H1/S1024 correctness remain PASS,
resources and bank0 remain clean, and the S1024 fused/full ticks improve. A
tick win without a complete lifecycle correctness result is not admitted.
