# Fused5 C0 One-Panel dV Phase Seed

## Evidence

Fresh accepted SQTT shows C0/C1 terminal times `91436/95136`, leaving C0 about
3700 cycles of CTA-tail slack. Across 256-cycle bins, C0 versus C1 contains
147 MMAC/MMAC bins, 129 MMAC/VALU bins, and 77 no-MMAC bins. Repeated steady
no-MMAC windows are about 512 cycles and contain C0 softmax/dS VALU while C1
is also in VALU/readiness work.

Current C0 computes `score -> P -> dP -> dS` per panel and batches all four dV
panels after dS publication. C1 computes `dP -> score -> P -> dS -> dV` per
panel. Their unequal per-panel MMAC cadence periodically returns to lockstep.

## Single Hypothesis

Use one existing C0 dV panel as a real-work phase seed:

```text
C0 panel0: score -> P -> dV -> dP -> dS
C0 panel1-3: score -> P -> dP -> dS
publish four dS panels
C0 tail: dV panel1-3 -> dK
```

No GEMM, matrix read, LDS byte, token, output, or ownership edge is added.
Only one already-required dV island moves before publication. The remaining
three dV panels retain the accepted early-publication benefit. C1 and the dQ
writer are unchanged.

## Gates

- Exact five-GEMM MMOP count, full correctness, bank0, and no spill/scratch.
- Same-build three-pair S1024 fused ticks improve against C0-gen2.
- If promoted, SQTT must show fewer all-heavy-role no-MMAC bins or a shorter
  C0/C1 lockstep recurrence without increasing ownership debt.

## Result

Status: `REJECT_DS_PUBLICATION_DELAY_CANONICAL_RESTORED`.

The compiler preserves exact work and lowers C0 use `187 -> 174` VGPR; all
correctness, metadata, and bank gates pass. Across three alternating S1024
pairs, fused ticks regress `44,814,770 -> 45,324,218` (`+1.137%`) and full
lifecycle ticks regress `48,898,698 -> 49,424,982` (`+1.076%`). Moving one dV
island before dS publication delays ownership more than its phase shift helps.
Restore the accepted C0-gen2 schedule and close per-panel dV phase seeding.
