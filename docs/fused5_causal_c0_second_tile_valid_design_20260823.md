# Fused5 Causal C0 Second-Tile Valid Specialization

Date: 2026-08-23

Status: `ACCEPT_CAUSAL_C0_SECOND_VALID_MMAC50_OPEN`

## Formula Proof

For one causal CTA:

```text
owned K tile: [k, k+127]
C0 K domain:  [k, k+63]
C1 K domain:  [k+64, k+127]
Q tile qi=1:  [k+64, k+127]
```

For every C0 element on `qi=1`:

```text
max(C0 K) = k+63 < k+64 = min(Q qi=1)
```

Therefore `krow <= qrow` is always true for C0's second retained Q tile.
C1 still spans the diagonal and must retain causal masking. This is a
group-local refinement of the already accepted steady-region mask elision.

## One Change

In `run_consumer_group<0>`, call the Page1/second-tile
`run_consumer_q_tile` with local causal mode `0`. Keep group1's Page1 call
unchanged. Later q tiles are already compiled unmasked.

No runtime branch or new helper is added. MMAC, matrix reads, LDS, global
traffic, output ownership and all ABarrier generations remain exact.

## Expected Pipeline

```text
time0  qi0: C0 diagonal mask; C1 accepted whole zero-front
time1  qi1: C0 score -> P -> dP -> dS without compare/cndmask
             C1 score -> causal predicate -> P -> dP -> dS
             writer consumes group-local dS in accepted G1-first order
time2+ steady: both groups already unmasked in canonical code
```

The useful stagger is preserved: C0 becomes shorter only in qi1 while C1
keeps its triangular VALU work. That should reduce lockstep rather than add an
empty delay.

## Fixed Budget

| Item | Gate |
|---|---|
| formula | exact five logical GEMMs |
| tile/roles | M64/N128/D128; 16 waves |
| resources | WDRA16/204/204/88; no spill/private/scratch |
| LDS | 128 KiB; bank0 |
| matrix path | MLS/BPS + ds_read_matrix + MMAC |
| tokens | all 12 ABarrier IDs and arrivals exact |
| outputs | dK/dV owners and dQ workspace unchanged |

## Admission

1. Static MMAC, matrix-read, ABarrier and traffic counts stay exact while
   causal compare/select instructions decrease.
2. Full golden passes H1/S128 causal/noncausal, H1/S1024 and H1/S2048 causal
   with warning0 and bank0.
3. Three paired H1/S1024 runs improve fused and lifecycle ticks; two paired
   H1/S2048 runs do not regress.
4. Fullperf/xcu must attribute the win to lower C0 qi1 VALU/control without a
   larger LGKM, ABarrier or store tail.

## Failure Boundary

Reject if compiler specialization duplicates the hot body, raises resource
use, or merely exposes more readiness debt than the removed predicate cost.
Do not combine this trial with the rejected owner-dependent zero-panel branch.

## Result

The one-line compile-time specialization is accepted as the new canonical
causal path.

- Static gates: role VGPR use `9/173/87/162`, SGPR71/VGPR128, LDS128KiB,
  private/spill/scratch0, exact MMOP88064 and bank0.
- Full CPU-golden correctness passes S128 causal/noncausal and causal
  S1024/S2048, with `vgpr_warning=0`.
- Three paired S1024 runs improve fused mean
  `41,823,297 -> 41,039,483` (`-1.874%`) and lifecycle mean
  `45,953,180 -> 45,281,752` (`-1.461%`).
- Two paired S2048 runs improve fused mean
  `77,043,103 -> 76,122,865` (`-1.194%`) and lifecycle mean
  `84,519,435 -> 83,613,985` (`-1.071%`).
- Fullperf keeps MMOP/LDS/VMEM/FLAT exact while reducing
  `VALU 92,496 -> 91,248` and `SCA 38,216 -> 37,544`. MMAC active rises
  `36.132798% -> 36.407955%`; barrier share falls
  `13.982952% -> 13.837914%`.
- XCU instruction issues fall `327,712 -> 325,952`. On representative
  `SE2/CU0/SIMD0`, instructions fall `11,340 -> 11,285` and bubble cycles
  fall `229,356 -> 227,011`. The top ABarrier issue gap falls
  `1,558,196 -> 1,535,724` but still consumes `22.81%` of issue-gap duration.

The single fullperf dispatch duration regresses `0.456%` while repeated
stats-only ticks improve at both sequence lengths. This is recorded as capture
perturbation, not hidden: wait-VM and wait-LGKM shares rise `0.070 pp` and
`0.153 pp`. The promotion is based on exact correctness/resources and all
paired same-shape stats-only wins; ABarrier ownership and matrix-read readiness
remain open rather than being declared solved.

Evidence:

- `/zys/sb/runs/f5c0second_correctness_20260823`
- `/zys/sb/runs/f5c0second_ab_20260823`
- `/zys/sb/runs/f5c0second_s2048_ab_20260823`
- `/zys/sb/runs/f5c0second_fullperf_20260823`
- workbook sections77-78

### Skill Candidate

- Trigger / applicable scenario: a compile-time causal tile boundary is fully
  valid for one ownership group but remains diagonal for its peer group.
- Rule / reusable rule: prove the integer tile-domain relation and specialize
  only the fully valid group at the call boundary; do not add a runtime
  per-owner branch inside the hot body.
- Evidence / evidence: exact MMOP88064, all correctness/resource gates,
  S1024 fused `-1.874%`, S2048 fused `-1.194%`, MMAC active `+0.275 pp`.
- Boundary / boundary: the proof depends on fixed group K ranges and causal
  ordering; arbitrary sparse/dropout masks or ragged tiles require a new proof.
- Counterexample / not applicable: partially valid owner panels compiled
  through a runtime branch caused PMD VGPR-init warnings and were rejected.
- Proposed Target / target: Shaobo reference material during the next skill
  consolidation; do not directly modify public skills from this experiment.
