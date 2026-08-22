# Fused5 N32 Workspace 12-Wave Experiment

Status: `REJECT_SINGLE_HEAVY_GROUP_LOSES_PEER_OVERLAP_CANONICAL_RESTORED`.

## Hypothesis

Replace the canonical `P4 + C0(N16)4 + C1(N16)4 + W4` topology with
`P4 + C(N32)4 + W4`. Each N32 dKV wave owns a complete 32-row K/V fragment,
while the four dQ writers retain D32 ownership and FP16 workspace reduction.
The design keeps exactly five logical GEMMs and removes the physical invalid
half, half of dS writer traffic, one dS publication group and four RawUsed
arrivals.

The expected gain was a longer dKV MMAC island and two full dS generations,
removing the measured C1 single-page `DqDone` stalls.

## Resource Revision

The direct implementation held four P panels, four dS panels and dK
read-ahead fragments across the q tile. It compiled at the 248-VGPR ceiling
but spilled 302 VGPR values into a 520-byte private segment, so it was not run.

P and dS were then persisted panel-by-panel into their existing LDS pages,
and dK was narrowed from two-panel to one-panel reads. The admitted build is:

```text
roles            P4 + C4 + W4
WDRA windows     24 / 248 / 112 (sum 384, average 128)
actual role use  9 / 240 / 85
metadata         SGPR87, VGPR128, private0, spill0, scratch0
LDS              131072 bytes
ABarrier IDs     10
```

The compiler requires the average of the three WDRA branch windows to satisfy
the VGPR allocation granularity. `24+248+128=400` is illegal; the admitted
`24+248+112=384` is legal and the writer's actual use is only 85.

## Verification

- S128 causal and noncausal full CPU golden: PASS.
- S1024 causal full CPU golden: PASS.
- dK, dV, dQ and delta nonfinite count: zero.
- Dynamic MMOP: 92,160, identical to canonical exact-five-GEMM work.
- LDS bank conflicts: zero.

Same compiler, PMD, `GPU_CHIP=sb` and `SQCIPfLines=7` H1/S1024 comparison:

| Metric | Canonical 16-wave | N32 12-wave | Delta |
| --- | ---: | ---: | ---: |
| fused ticks | 45,052,735 | 50,706,110 | +12.55% |
| MMAC active | 34.737% | 32.063% | -2.674 pp |
| MMOP | 92,160 | 92,160 | 0 |
| VALU | 118,880 | 108,868 | -8.42% |
| LDS instructions | 63,872 | 32,704 | -48.80% |
| SCA | 46,744 | 38,488 | -17.66% |
| coissue success/fail | 20,536 / 22,388 | 6,361 / 5,590 | lower activity |
| barrier share | 13.932% | 20.419% | +6.487 pp |

## Conclusion

The candidate removes real transport and scalar work but slows the critical
path. The second independent heavy dKV group is a useful same-SIMD issuer;
the dQ writer is dependent on published dS and cannot replace it. Collapsing
the heavy groups converts the saved C1 reuse wait into a larger publication
barrier and lower coissue opportunity.

Keep the canonical two-heavy-group topology. Future ownership work must repair
C1 dS reuse without removing C0/C1 independence, adding dead arithmetic, or
introducing a CTA-wide dO-dead rendezvous.

Evidence:

- Candidate: `/zys/sb/fa3b/n32_ws12_20260822`.
- Control: `/zys/sb/fa3b/n32_ws12_baseline_20260822`.
- Build sandbox: `/zys/sb/experiments/fused5_n32_ws12`.
- Workbook: sheet `45 N32 WS 12W`.

