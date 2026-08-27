# Fused5 C1 Early dS Publication

Status: `REJECT_LOCKSTEP_MMAC_CONTENTION / CANONICAL_RESTORED`.

## Hypothesis

C111 publishes C0's four dS panels before C0 dV/dK, but C1 executes dV inside
each panel and publishes dS only after all four panels.  The candidate retains
C1 probability fragments, publishes all C1 dS first, then runs the unchanged
dV and dK GEMMs.  This should let the G1-first dQ writer start earlier and use
otherwise exposed ownership gaps.

The arithmetic, `M64/N128/D128` tile, 16-wave roles, LDS128KiB map, ABarrier
IDs, matrix traffic, BPS traffic and output ownership stay unchanged.  The
only additional live state is four FP16 P fragments. C111 C1 uses 130/204
branch VGPR; the generated candidate uses 135/204, still below its WDRA pool.

## Expected Pipeline

```text
time0  C0: score -> P -> dP -> dS x4
       C1: dP -> score -> P -> dS x4
       W : wait prior generation

time1  C0/C1: publish all dS
       W    : start G1 dQ

time2  C0/C1: dV x4
       W    : G1 dQ

time3  C0/C1: dK
       W    : finish G1, start G0

time4  producer: publish next raw page after unchanged RawUsed
       W       : finish G0/store
```

## Stress Test

The candidate can fail even when dS publication is earlier.  C1's current
per-panel dV is useful asymmetry against C0's VALU/dP work; deferring it may
make both consumers more lockstep.  Writer dQ also competes with dV/dK for the
same SIMD MMAC pipe, so MMAC active alone cannot promote the change.  Repeated
same-shape fused and lifecycle ticks decide.

## Gates

- Compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7.
- Static MMAC `1344`, matrix-read `768`, no new BPS/store/barrier sites.
- Private segment, scratch and SGPR/VGPR spill are all zero.
- H1/S128 causal and noncausal, then H1/S1024 causal golden PASS.
- `ldsBankConflict=0`.
- Three interleaved H1/S1024 control/candidate pairs before fullperf.
- On regression, restore C111 source and retain only this evidence.

Workbook: `F5_C1_EarlyDsPublish` in
`/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`.

## Result

The generated kernel preserves MMAC, matrix-read, BPS, LDS, ABarrier and
output work.  Final causal branch use is `9/142/87/135`; metadata is
SGPR72/VGPR128 with private/spill/scratch0.  An explicit paired row-delta
definition is required to avoid PMD reading the unused adjacent packed source
as uninitialized.  S128 causal/noncausal and S1024 causal full golden checks
pass with warning0 and bank0.

Three interleaved H1/S1024 comparisons reject the schedule:

| Pair | Control fused | Candidate fused | Control lifecycle | Candidate lifecycle |
|---:|---:|---:|---:|---:|
| 1 | 38,805,585 | 40,943,630 | 42,805,035 | 45,115,070 |
| 2 | 39,205,530 | 40,919,970 | 43,268,225 | 44,958,095 |
| 3 | 38,437,945 | 40,530,945 | 42,460,145 | 44,692,375 |
| mean | 38,816,353 | 40,798,182 | 42,844,468 | 44,921,847 |

Fused and lifecycle means regress `5.106%` and `4.849%`.  One matched stats
pair keeps MMOP/LDS/SCA/VMEM exact, but dynamic VALU rises
`89,040 -> 94,096`, successful coissue falls `17,357 -> 16,746`, no-VALU-ready
cycles rise `286,825 -> 302,302`, and SIMD active CV worsens
`0.4212 -> 0.4368`.  Earlier G1 publication therefore does not fill an idle
writer slot; it removes C1's useful per-panel VALU/MMAC stagger and creates
MMAC contention among C0, C1 and the writer.  No fullperf was admitted.

Evidence: `/zys/sb/ab_c1_early_ds_20260827`; candidate archive:
`/zys/fa3_bwd_c1_early_ds_20260827`.
