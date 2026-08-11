# Fused5 M128 Half1 Useful Prelude

Status: `REJECT_STATS_TICKS / CANONICAL_RESTORED`.

## Evidence Before Code

Explicit pair-vector probability/dS restores packed ISA and lowers dynamic
VALU from the rejected M128 expression's 128,464 to 118,032. Two H1/S1024
runs average 72,462,617.5 fused ticks, still 0.575% slower than the accepted
M64 mean 72,048,112.5. MMAC active is 22.211146% versus 22.714850%.

The remaining regression is ownership-shaped:

| Metric | M64 accepted | M128 packed | Delta |
|---|---:|---:|---:|
| barrier share | 27.523564% | 28.720295% | +1.197 pp |
| waitLgkm share | 7.441189% | 7.691578% | +0.250 pp |
| VALU | 120,800 | 118,032 | -2.291% |
| SCA | 37,808 | 35,260 | -6.739% |

M64 defers the prior `KvDsUsed` wait until the next tile has completed useful
score/dP/softmax/dV/dS work. M128 currently waits immediately between half0
and half1, exposing the dQ half0 source-read latency and defeating the intended
two-half conveyor.

## One Hypothesis

Move only half1 work that does not overwrite the shared P/dS source region
before the half0 `KvDsUsed` wait. The wait remains mandatory and no barrier
count, page, formula, output owner or MMOP changes.

```text
half0:
  C0/C1  score,dP,softmax,dV,dS -> publish -> dK -> arrive KvDsUsed
  WQ     dQ half0 -> arrive KvDsUsed -> atomic half0

half1 prelude while WQ reads half0:
  C0     four score + probability panels
  C1     four dP + score + probability + dS panels

half1 after KvDsUsed:
  C0     dV + dP + dS
  C1     dV
  both   publish dS -> dK
  WQ     dQ half1 -> atomic half1
```

C0 must retain four `ProbabilityPanel` values; C1 retains four FP16 P and dS
fragments. This is admitted only if compiled role usage stays within
`8/200/200/88` with no private/spill/scratch.

## Invariants And Gates

1. M128/N128/D128, exact five GEMMs and H1/S1024 MMOP92,160.
2. One canonical kernel; no phase flag, alternate path, empty delay or
   duplicate score/dP.
3. LDS131,072B; unchanged seven ABarrier tokens and generation counts.
4. Main matrices remain MLS/BPS + `ds_read_matrix` + MMAC; bank0.
5. Generated ASM retains explicit packed probability/dS instructions.
6. H1/S128 causal/noncausal and H1/S1024 causal correctness PASS.
7. Promotion requires lower repeated H1/S1024 ticks than M64; otherwise
   restore `b28e73d` and close M128 on the current compiler/PMD.

## Result

The resource and correctness gates pass at role `8/169/169/84`, SGPR64,
VGPR124, LDS131,072 B, private/spill/scratch0, exact MMOP92,160 and bank0.
H1/S128 causal and noncausal both pass.

Two H1/S1024 runs are 72,787,715 and 72,917,845 fused ticks, averaging
72,852,780. This is 1.117% slower than the accepted M64 mean and 0.539%
slower than packed M128 without the prelude. MMAC active averages 22.063894%
and barrier share averages 29.019252%.

Moving legal half1 work before the wait therefore does not hide the ownership
edge. It extends live state and scheduling pressure while the same P/dS page
still requires a hard half0-to-half1 handoff. This is the second M128
ownership-tier failure, so M128 is closed on this compiler/PMD and production
source is restored to `b28e73d`.
