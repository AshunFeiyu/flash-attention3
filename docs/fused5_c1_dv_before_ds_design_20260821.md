# Fused5 C1 dV-Before-dS Useful Stagger

Status: `REJECT_NOISE_AND_LATE_PUBLICATION_CANONICAL_RESTORED`

## Evidence And DAG

The canonical repeated panel order is:

```text
C0: score -> P -> dP -> dS; publish all dS; dV -> dK
C1: dP -> score -> P -> dS -> dV; publish dS -> dK
```

For one panel, `dV=P^T@dO` depends on P and dO but not on dS.  Therefore C1
may legally execute dV before `dS=P*(dP-D)*scale` without duplicating a GEMM
or changing numerical order inside either operation.

## Single Hypothesis

Change only C1's repeated panel order to:

```text
dP MMAC -> score MMAC -> P VALU -> dV MMAC -> dS VALU
```

Expected cross-role pairing after the initial beat:

```text
C0 dS VALU / publication | C1 dV MMAC
C0 next useful MMAC       | C1 dS VALU
```

This tests useful MMAC/VALU staggering with real work. No empty delay,
read-ahead, page, token, global access, matrix operation or output changes.

## Risk And Gates

- `dp` and FP32 P live across the C1 dV island, so C1 VGPR use may rise. A
  spill/private segment or WDRA-window violation rejects the experiment before
  PMD.
- C1 dS publication is later. If canonical G0 writer MMAC no longer hides C1
  readiness, barrier ticks will rise and the candidate must be rejected.
- Required: exact MMOP92,160, static gates, H1/S128 and H1/S1024 correctness,
  bank0, paired lower fused ticks. Fullperf is admitted only after paired
  stats improve.

## Result

The generated ISA moved the C1 row-delta read and dS VALU after the existing
dV MMAC island. Static resources remained SGPR82/VGPR128 with role use
`9/187/87/182`, private/spill/scratch zero. H1/S128 and both H1/S1024 runs
passed full CPU-golden correctness with bank0 and exact MMOP92,160.

| Pair | Control fused ticks | Candidate fused ticks | Change |
|---|---:|---:|---:|
| 1 | 45,145,100 | 45,327,555 | +0.404% |
| 2 | 45,413,550 | 45,323,005 | -0.199% |
| mean | 45,279,325 | 45,325,280 | +0.101% |

Complete-lifecycle means differ by only `-0.076%` in the opposite direction.
No stable gain exists, and the candidate structurally delays C1 dS
publication. Fullperf was not admitted. Canonical source is restored; another
local operation-order permutation is not justified on this topology.
