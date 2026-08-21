# Fused5 C1 Panel Sidecar Under dP

## Evidence And Hypothesis

The full-tile probability-sidecar latch is rejected because eight values stay
live across the whole q tile and raise wait-LGKM. C1 already has a dependency-
safe per-panel window:

```text
read Q(m) -> wait dO(m) -> dP MMAC -> wait Q(m) -> score MMAC
```

Issue only the current panel's three scalar sidecar reads immediately after
the Q packet. The existing dP MMAC can age both Q and sidecar before their
first use:

```text
read Q(m) -> read max/sum/delta(m) -> wait dO(m)
          -> dP MMAC -> wait Q+sidecar -> score/softmax/dS
```

The first wait leaves the four youngest LDS operations alive; dO is the oldest
four-read packet and is therefore ready. The later `lgkmcnt(0)` makes Q and
sidecar ready before score/softmax/dS.

## Invariants And Gates

- C0 is unchanged; C1 adds only three panel-local FP32 live values.
- Five GEMMs, MMOP, LDS, ABarrier and output ownership are unchanged.
- Generated ISA must show Q matrix reads, three scalar sidecar reads, the dO
  readiness wait, dP MMAC, then final LDS wait.
- No private/spill/scratch; H1/S128 and repeated H1/S1024 correctness PASS;
  bank conflict zero and same-mode fused ticks improve.

## Result

Status: `REJECT_LGKM_SCHEDULING_TIER_CLOSED_CANONICAL_RESTORED`.

The ISA exactly matches the hypothesis and resources remain identical to the
canonical kernel at branch use `9/187/87/182`, SGPR82/VGPR128, with no private
segment, spill or scratch. H1/S128 and H1/S1024 pass full correctness and
bank conflict remains zero.

The same-mode S1024 pair regresses fused ticks `44,846,620 -> 45,307,990`
(`+1.03%`) and lifecycle ticks `49,074,935 -> 49,455,315` (`+0.78%`).
Wait-LGKM rises `7.51% -> 7.85%` and MMAC active falls
`33.91% -> 33.50%`. Together with the rejected full-tile sidecar latch, this
is the second same-tier failure: stop sidecar/read-placement micro-tuning and
return to structural pipeline analysis.
