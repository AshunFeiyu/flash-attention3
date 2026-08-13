# Fused5 Raw Q/dO Split-Lifetime Probe

Status: `PROBE_DESIGN_OPEN`

## Single Hypothesis

In one raw page, `dO` is dead after dP/dV while `Q` remains live until the
final dK read. Keep the physical layout unchanged (`Q=16KB`, `dO=16KB`) but
split ownership so the producer may refill the dO half before it refills Q.

## Expected Schedule

```text
t0: page p contains Q_i + dO_i; consumer computes score/dP/softmax/dV
t1: consumer arrives DoutUsed[p]; producer may load dO_(i+2) into page p
t2: consumer computes dK from Q_i; consumer arrives QUsed[p]
t3: producer loads Q_(i+2) and sidecar into page p, then publishes the pair
```

The two physical pages remain page0/page1. Only the ownership granularity
changes; no GEMM, LDS byte, or matrix layout changes.

## Probe Contract

- 16-wave CTA: Q publisher waves0-3, dKV consumers4-11, dO publisher
  waves12-15.
- Two pages and three generations with deterministic Q/dO values.
- Separate `QFilled[p]`, `DoutFilled[p]`, `QUsed[p]`, and `DoutUsed[p]`
  tokens. The probe prices the extra ABarrier state before integration.
- Consumers read dO, signal `DoutUsed`, then read Q and signal `QUsed`.
  Producers may therefore overwrite dO while Q is still live, but never the
  reverse.
- Main data movement remains MLS+BPS plus `ds_read_matrix`; no ordinary
  `ds_read_b32`, gather, bpermute, or layout workaround is allowed.
- Check every generation/consumer checksum, producer/consumer completion
  counts, PMD panic/VGPR warnings, and bank conflicts.

## Admission

No new LDS bytes and no GEMM are allowed. A passing probe is required before
canonical integration. Deadlock, stale generation, or an unpriced token
explosion is `REJECT_LIFECYCLE`; no performance claim is made by the probe.

## Probe Results

The two-publisher probe passed on PMD HEAD1694 with compiler `e0f10535`:
`q_errors=0`, `dout_errors=0`, no panic/VGPR warning, and bank conflict 0.
The single-producer probe also passed with the canonical role shape
`producer=waves0-3`, `consumers=waves4-11`, `dQ-writer=waves12-15`.

The direct two-publisher canonical integration is not admitted: it passed
correctness but regressed H1/S1024 to `68,606,720` ticks. The full Q/dO token
integration also passed correctness but measured `46,888,205` and
`50,151,010` ticks on repeats versus the clean `46,637,955` baseline. The
next canonical trial therefore keeps one `RawFilled` and one `RawUsed` token
per page and adds only `DoutUsed[p]` to prefetch dO after its last use.
