# Fused5 C0 dP Read-Under-Score Design

## Evidence

The accepted fixed-pair dS conveyor reduces the representative dQ writer's
ABarrier bubbles from 33,219 to 21,983 cycles and improves fused ticks at both
S1024 and S2048. Fresh SQTT now shows consumer0's largest local bubbles are
`ds_read_matrix_trans_format -> s_waitcnt` (13,371 cycles) and
`ds_read_matrix_format -> s_waitcnt` (12,316 cycles). The trans-read debt is
the synchronous dO packet immediately before each dP MMAC island.

## Single Hypothesis

For each C0 M16 panel, issue the current dO-trans packet and the next
Q-trans packet before the current score MMAC:

```text
latch sidecar(m)     three scalar LDS reads, complete before matrix packet
read dO(m)           four matrix reads
read Q(m+1)          four matrix reads, when present
wait lgkmcnt(8/4)    current Q(m) ready; younger packets may remain
score MMAC           hides current dO and next Q latency
wait lgkmcnt(4/0)    current dO ready; next Q may remain
dP MMAC              hides remaining next-Q latency
softmax + dS
```

This changes only C0 operand-read scheduling. Formula, exact five GEMMs,
M64/N128/D128 tile, output ownership, dS generations, ABarrier ledger, LDS
layout, and dynamic MMOP count remain unchanged.

## Resource Gate

- One additional four-fragment dO packet costs up to 16 VGPR.
- Accepted C0 evidence uses about 187 of its 204-VGPR WDRA pool, so the
  candidate is admitted only if generated metadata remains within 204 with no
  private segment, spill, or scratch.
- No new LDS allocation or ordinary matrix-path DS read is allowed.

## Admission

1. Generated ASM must show the intended read/read/wait/MMAC/wait/MMAC order.
   A sidecar-generated `lgkmcnt(0)` between score and dP invalidates the test;
   sidecar is therefore latched before the matrix packet is issued.
2. Static/metadata gates, exact MMOP, bank0, and cached-golden correctness must
   pass at S128 and S1024.
3. Paired S1024 fused ticks must improve. SQTT must show lower C0 trans-read
   first-use bubbles without increasing writer ABarrier or terminal time.
4. If C0 exceeds its WDRA pool or ticks regress, remove the helper and retain
   `best/fused5-c0-ds-gen2-fixed-pair-20260819` unchanged.

## Result

Status: `REJECT_LOCAL_WAIT_TO_OWNERSHIP_DEBT`

Generated ASM reached the requested structure after sidecar values were
latched before the matrix packet: eight trans matrix reads, `lgkmcnt(8)`,
eight score MMAC, softmax, `lgkmcnt(4)`, then eight dP MMAC. C0 used 190/204
VGPR; kernel metadata remained SGPR76/VGPR128 with no private segment, spill,
or scratch. S128 causal/noncausal and all S1024 runs passed correctness and
bank0.

Three alternating S1024 pairs nevertheless regress fused mean ticks
`44,896,518 -> 45,939,530` (`+2.323%`) and lifecycle mean ticks
`49,016,543 -> 50,023,458` (`+2.054%`). Mean wait-LGKM falls
`7.518% -> 7.277%`, but barrier share rises `13.433% -> 13.829%` and MMAC
active falls `34.060% -> 33.771%`.

The local read schedule worked, but accelerated C0 into the next ownership
gate and disturbed the accepted cross-role cadence. The helper is removed from
the canonical source. Do not retry another C0-only read-count variant unless a
fresh SQTT critical-path trace first proves C0 is the final CTA pace setter.
