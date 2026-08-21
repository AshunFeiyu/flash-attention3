# Fused5 Consumer Resident-Read Zero Cover

Status: `REJECT_NOISE_REGRESSION_CANONICAL_RESTORED`

## Evidence

The canonical H1/S1024 SQTT shows a one-time consumer startup sequence:

```text
8 resident K/V ds_read_matrix
-> exposed first-use wait (about 187--319 cycles on representative waves)
-> dK/dV accumulator and MMAC-zero initialization
```

The accumulator initialization is independent of the resident K/V fragments.
It currently executes after the read-ready wait even though it can occupy part
of that latency interval.

## Single Hypothesis

Split resident K/V latching into issue and completion stages.  Issue all eight
matrix reads, initialize the long-lived dK/dV accumulators and the branch-local
MMAC zero fragment, then execute the unchanged `lgkmcnt(0)` before any resident
fragment is consumed or `VSidecarReady` is signalled.

Expected instruction shape:

```text
8 resident ds_read_matrix
-> existing zero moves
-> lgkmcnt(0)
-> unchanged q-loop
```

This is latency cover, not zero-seed.  The previously rejected runtime
first-update zero-seed path is not revived.

## Invariants

- Exactly five logical GEMMs and dynamic MMOP remain unchanged.
- M64/N128/D128, 16-wave roles, output ownership and WDRA windows are fixed.
- LDS remains 128 KiB; no page or ABarrier token is added.
- `VSidecarReady` is still emitted only after all K/V reads retire.
- Main matrices remain MLS/BPS + `ds_read_matrix` + MMAC.

## Admission Gates

1. Generated ISA contains resident reads, the existing zero block, then the
   first-use wait; no duplicated zero block or MMAC is allowed.
2. No private segment, scratch, SGPR spill or VGPR spill.
3. H1/S128 and H1/S1024 causal lifecycle correctness PASS; bank conflict zero.
4. Same-build paired H1/S1024 fused ticks must not regress.  MMAC active and
   startup matrix-read bubbles are explanatory metrics.

If LLVM sinks the zero block after the wait, duplicates it, or adds register
moves, classify that as a compiler scheduling/code-generation boundary and
restore the canonical source.

## Result

LLVM emitted the intended sequence: eight resident matrix reads, the existing
34 `v_mov_b64` zero operations, then `lgkmcnt(0)`. Static resources were
unchanged at SGPR82/VGPR128, role use `9/187/87/182`, and no private segment,
spill or scratch. H1/S128 and both H1/S1024 candidate runs passed complete
CPU-golden correctness with `ldsBankConflict=0`.

| Pair | Control fused ticks | Candidate fused ticks | Change |
|---|---:|---:|---:|
| 1 | 45,062,290 | 44,991,310 | -0.158% |
| 2 | 45,027,255 | 45,344,390 | +0.704% |
| mean | 45,044,773 | 45,167,850 | +0.273% |

The compiler did not block the schedule. The rejected mechanism is a
once-per-CTA startup cover that does not shorten the repeated q-loop ownership
path; its small local gap reduction is absorbed by normal PMD issue variation.
No candidate fullperf was admitted. Canonical source is restored.
