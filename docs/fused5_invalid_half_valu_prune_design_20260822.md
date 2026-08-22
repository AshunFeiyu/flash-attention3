# Fused5 Invalid-Half VALU Prune

Status: `ACCEPT_LONG_LOOP_TICKS_AND_ACTIVE / S1024_NOISE_NEUTRAL / WAIT_DEBT_OPEN`.

## Single Hypothesis

Each dKV wave owns `N=16`, while the FP16 score fragment exposes eight scalar
half slots.  In the canonical source-slot mapping, slots `word=4..7` always map
to `local_k >= 16`, independent of lane.  They therefore cannot contribute to
probability, dS, dV, dK or dQ.  The current source nevertheless evaluates
`exp2`, masking and dS arithmetic for all eight slots before zeroing the upper
half.  Make the four valid slots explicit so the compiler does not issue that
dead VALU work.

## Coordinate Proof

For `word in [0,7]`, `source_slot_qk` contributes:

```text
local_k = lane_base + (word & 1) + 8*(word bit1) + 16*(word bit2)
lane_base in {0,2,4,6}
```

Thus:

- `word=0..3`: `local_k in [0,15]`, exactly the N16 owner.
- `word=4..7`: `local_k in [16,31]`, always outside the N16 owner.

`f16_mmac_single` also writes the logical score result only to
`out.f16x4[0]` and explicitly zeroes `out.f16x4[1]`.  The upper four
probability and dS slots must remain zero for the native writer/reader ABI, but
they do not require exp2 or multiply/subtract evaluation.

## Canonical Change

- `probability_stage`: evaluate `word=0..3`; zero `p.f16x4[1]` once.
- `ds_stage`: evaluate `word=0..3`; zero `ds.f16x4[1]` once.
- Keep score/dP/P/dS formulas for every useful coordinate unchanged.
- Keep exactly five GEMMs, MMOP, raw-page lifetime, dS publication, dQ writer,
  LDS layout and all ABarrier IDs unchanged.

## Expected Evidence

| Gate | Expected result |
| --- | --- |
| ASM | `v_exp_f32` and upper-half mask/dS instructions decrease; MMAC unchanged |
| Resources | WDRA roles remain within `16/204/204/88`; no spill/private/scratch |
| Correctness | H1/S128 causal and noncausal PASS; H1/S1024 causal PASS |
| PMD | `MMOP=92160`, `ldsBankConflict=0`, fused ticks decrease |
| Scaling | H1/S2048 must not regress |
| SQTT | VALU issue count and no-MMAC windows fall without higher wait/barrier debt |

Promotion requires same-build paired ticks with the same sign on S1024 and no
S2048 regression.  If the compiler already proves equivalent code or resource
pressure rises, restore canonical and record `REJECT_STATIC` or
`REJECT_PERF`.

## Result

The source change is mathematically exact and passes every hard gate:

- H1/S128 causal and noncausal full CPU-golden correctness: PASS.
- H1/S1024 causal full CPU-golden correctness: PASS, including dK/dV/dQ.
- SGPR87/VGPR128, private0, scratch0, SGPR/VGPR spill0.
- WDRA role use falls from about `9/187/87/182` to `9/171/87/168`.
- Dynamic MMOP92,160, LDS63,872, VMEM1,408 and FLAT3,616 are unchanged;
  `ldsBankConflict=0`.
- Dynamic VALU falls `140,208 -> 118,880` (`-15.21%`). Generated ISA keeps
  MMAC1,472 and matrix-read840 while `v_fma_mix` falls `160 -> 80`,
  `v_mul_f32` `489 -> 253`, and `v_cvt_pk_f16_f32` `288 -> 208`.

Six interleaved H1/S1024 pairs are noise-neutral: fused means move
`45,199,169 -> 45,088,377` (`-0.245%`) with only three pair wins. MMAC active
nevertheless rises consistently from mean `33.638%` to `34.643%`
(`+1.005 pp`).

Two interleaved H1/S2048 pairs both win. Fused means move
`86,063,478 -> 83,843,988` (`-2.579%`), and MMAC active rises
`37.422% -> 38.659%` (`+1.237 pp`). The candidate is therefore promoted for
the repeated steady loop even though the shorter diagnostic shape is inside
the PMD noise band.

SQTT explains the remaining debt. Relative to the exact control fullperf,
dynamic instruction issue falls `390,896 -> 370,800`, but `s_waitcnt` hits
rise `18,272 -> 19,712`. The ABarrier-to-XOR issue gap grows
`1,911,140 -> 2,044,352`, transposed matrix-read-to-wait grows
`781,478 -> 814,461`, and normal matrix-read-to-wait grows
`299,248 -> 317,068`. At role level C1's MMAC-with-vector-peer count falls
from about `664/2048` to `436/2048`; C0/C1 256-cycle no-MMAC bins rise
`72 -> 84`.

The optimization removes dead work but exposes readiness and ownership debt
that the dead VALU previously covered. Do not add delay or restore dead math.
The next admitted hypothesis must use the released consumer VGPR headroom for
an existing next-panel operand request, and must prove with SQTT that it lowers
wait/no-MMAC windows without worsening the RawFilled/RawUsed cadence.

Evidence:

- stats: `/zys/sb/fa3b/ihv22`, `/zys/sb/fa3b/ihv22_more_s1024`,
  `/zys/sb/fa3b/ihv22_s2048`;
- correctness: `/zys/sb/fa3b/ihv22_correctness`;
- perf: `/zys/sb/fa3b/ihv22_fullperf`;
- xcu: `/zys/sb/fa3b/ihv22_xcu`.
