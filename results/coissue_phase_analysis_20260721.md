# dKV/dQ Role-Local Coissue Analysis (2026-07-21)

## Scope

- Shape: `B=1,H=1,S=2048,D=128,causal=true`.
- Model: `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`, PMD HEAD1694.
- Best dKV: `56,162,470 ticks`, `45.655405%` MMAC active.
- Best dQ: `38,870,195 ticks`, `45.356456%` MMAC active.
- Evidence: role-local `xcu pipeline/simd/coissue` CSV under
  `work/rolelocal_20260721/deep2` and the archived best `.perf` files.

## Executive Conclusion

The current kernels do create consumer skew, but it is not a stable reciprocal
conveyor.

- dQ has strong short-window staggering, but it is one-way: C0 VALU is usually
  covered by C1 MMAC, while C1 VALU is rarely covered by C0 MMAC.
- dKV has weaker useful staggering and both consumers return to the same MMAC
  phase more often.
- Both kernels repeatedly reacquire the same operand-read, first-use wait, and
  ownership boundaries. Those boundaries collapse local skew into macro
  lockstep.
- Reaching `50%` MMAC active at fixed exact MMOP requires roughly `8.7%` dKV
  and `9.3%` dQ reduction in active-time dead zones. Extra MMAC is not allowed.

## dQ Instruction-Flow Asymmetry

The source deliberately emits different orderings for the two consumers:

```text
C0: score/dP MMAC -> softmax/dS VALU -> K-normal read8 -> wait -> dQ MMAC
C1: score/dP MMAC -> K-normal read8 -> softmax/dS VALU -> wait -> dQ MMAC
```

This is not random compiler scheduling. `ConsumerGroup == 1` reads K-normal
before the softmax block, while `ConsumerGroup == 0` reads it after that block.

Representative role-local SQTT gives:

| Metric | C0 | C1 |
|---|---:|---:|
| K-normal read clusters | 62 | 64 |
| Reads per cluster | 8 | 8 |
| VALU between last read and first-use MMAC | 0 | about 36 |
| Last-read to wait, median | 158 cycles | 192 cycles including useful VALU |
| First VALU after last read, median | n/a | 4 cycles |
| Last VALU to wait, median | n/a | 4 cycles |
| VALU coissued with peer MMAC | 1,457 / 2,320 | 256 / 2,410 |

For C0, the `158-cycle` interval contains no independent instruction from that
wave; the LDS readiness dependency is exposed. For C1, the longer elapsed
interval is occupied by useful softmax/dS work, so the read latency is hidden.

Aligning all 62 C0 read-to-wait intervals to C1 on the same SIMD shows that
this is usually a SIMD-level MMAC hole, not merely a local-wave symptom:

- only 32/62 intervals contain any C1 MMAC issue;
- the median is one C1 MMAC issue per interval;
- approximate peer-MMAC coverage is only 8% on average and never reaches 50%;
- the median interval instead contains 26 C1 VALU issues.

Therefore C0 waits for LDS at nearly the same time that C1 executes softmax/
dS. Both heavy consumers have re-entered a non-MMAC phase together.

The coissue direction is correspondingly asymmetric:

- C0 `v_exp`: `468/496` coissued with peer MMAC.
- C1 `v_exp`: `11/512` coissued with peer MMAC.
- C0 packed add/mul softmax instructions are commonly covered by C1 MMAC;
  the reverse overlap is sparse.

Thus C0 pays a local LDS hole in exchange for a favorable peer-MMAC window.
The schedule improves overlap in one direction but does not build a balanced
two-way conveyor.

## Why Wavefronts Looks Alternating And Lockstep At Different Times

At a `256-cycle` window, dQ has both consumers issuing MMAC in `55.6%` of
active windows and only one consumer issuing MMAC in `41.4%`. This is visible
local staggering. At a `1024-cycle` window, both consumers contain MMAC in
`88.9%` of windows. The phases therefore separate locally but repeatedly
rejoin at read/wait/page boundaries.

dKV is more synchronized:

| Window metric | dKV | dQ |
|---|---:|---:|
| Both consumers MMAC, 256-cycle bins | 80.7% | 55.6% |
| Only one consumer MMAC, 256-cycle bins | 17.9% | 41.4% |
| Both consumers MMAC, 1024-cycle bins | 94.6% | 88.9% |

dKV useful VALU-with-peer-MMAC counts are also one-way but weaker:
`295/2,730` for C0 and `758/2,731` for C1. The accepted C1 sidecar-tail
prefetch creates some skew, but not a durable phase separation.

## Direct dKV Counterexample

A focused candidate moved only C1 D0/D1 operand reads from before softmax to
after softmax. Dynamic MMOP/VALU/SCA/LDS/VMEM/FLAT and all ownership/read/store
counts stayed exact. Correctness, bank0, and no-spill gates passed.

H1/S1024 control/candidate:

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| kernel ticks | 31,547,425 | 33,640,880 | +6.6359% |
| MMAC active | 38.443812% | 37.027105% | -1.4167pp |
| waitLgkm | 16,600.5 | 22,283.75 | +34.2354% |
| barrier | 68,262.75 | 75,391.75 | +10.4435% |
| no-V-or-M | 158,620 | 173,921 | +9.6463% |
| successful coissue | 12,396 | 9,644 | -22.2007% |

This proves that deliberately exposing a read dependency is not a viable way
to manufacture consumer skew.

## Rejected Simplifications

- Swapping the early role to C0 regressed dQ S1024 ticks `3.27%` and active
  `0.997pp`.
- Moving all K-normal reads before score/dP reduced local wait but extended
  VGPR lifetime and regressed ticks/active.
- Inserting reads inside the score/dP MMAC island fragmented that island and
  regressed ticks.
- Splitting the softmax block to place a read in its middle caused compiler
  CFG expansion, added `1,056` VALU and `1,056` SCA, and regressed ticks.
- A blocking peer-consumer phase token serialized runnable waves and regressed
  dQ ticks `4.09%`.

Therefore the next candidate may not simply make both roles identical, split
an existing MMAC/VALU island, add a token, or move one read late.

## Next Structural Hypothesis

A C0 rolling cross-`n_tile` K-normal ping/pong was audited and rejected before
implementation. Although an ideal `lgkmcnt(8)` ledger can leave eight next-read
requests outstanding, the fragment lives across the next score/dP and
softmax, adds about 32 VGPR peak, does not shorten PageUsed ownership, and is
substantially equivalent to prior cross-`n_tile` prefetch failures.

The remaining focused hypothesis is instead a C1-only whole-island pre-score
read:

```text
C0 unchanged:
  score/dP MMAC -> softmax/dS -> read8 -> wait -> dQ MMAC

C1 candidate:
  trans read16 + K-normal read8
  wait to first trans half -> score/dP D0/D1 MMAC
  wait to second trans half while K-normal remains in flight
  score/dP D2/D3 MMAC -> softmax/dS -> wait -> dQ MMAC
```

This does not try to remove C0's local hole. It attempts to advance C1's dQ
MMAC so that it covers that hole at the SIMD level. Unlike the rejected C1
dead-slot experiment, no read may split a score/dP MMAC island. Unlike the old
all-consumer pre-score experiment, only C1 moves, preserving role asymmetry.

Expected C1 branch use is about `187` versus the configured `216` window. The
candidate is legal only if ASM preserves the `read24 -> MMAC half -> MMAC half`
structure, exact dynamic work/read counts, no spill/scratch, correctness and
bank0. Promotion additionally requires a material rise from the current 8%
C1-MMAC coverage of C0 read holes, together with lower same-shape ticks and
higher MMAC active.

## Final Diagnosis

The user's Wavefronts observation is correct. The remaining performance gap is
not a lack of MMAC instructions. It is a scheduling problem with three coupled
parts:

1. one consumer exposes operand readiness to obtain phase skew;
2. useful MMAC/VALU overlap is strongly directional rather than reciprocal;
3. common read/wait/ownership boundaries repeatedly pull both consumers back
   into the same macro phase.

The next optimization must remove the exposed C0 readiness hole without
destroying the accepted inter-consumer phase relationship.
