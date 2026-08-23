# Fused5 Causal Writer Zero-Front Prune Design

Status: `REJECT_CODEGEN_CONTROL_EXPANSION / CANONICAL_RESTORED`.

## Algebraic Proof

The accepted C1 zero-front path proves that, for the first causal Q tile,

```text
Q rows       = [k_base + 0,  k_base + 63]
C1 K rows    = [k_base + 64, k_base + 127]
dS_C1(q, k)  = 0
```

The writer computes the C1 contribution to dQ as:

```text
dQ_C1 = dS_C1 @ K_C1 = 0
```

Therefore the first causal writer tile does not need C1 dS matrix reads or C1
dQ MMAC. C0 is partially valid on the diagonal and remains unchanged.

## One Hypothesis

Specialize only the writer's first causal tile:

1. initialize dQ accumulators exactly as canonical;
2. wait for `BatchDsFilled1` and arrive `DqDone1` to preserve the accepted
   C1 publication/release cadence;
3. skip `update_dq_writer_group<1>` for this one tile;
4. consume C0, store the same dQ partial, and keep every later tile canonical.

This first experiment deliberately retains C1 zero dS writes and the existing
ABarrier events. It changes one variable: zero writer matrix work. A later
experiment may remove the now-redundant zero payload/token only if this phase
passes and SQTT proves the handshake remains material.

## Work And Resource Budget

| Item | Accepted | Candidate causal delta |
|---|---:|---:|
| tile / roles | M64/N128/D128, 4P+4C0+4C1+4W | exact |
| logical GEMMs | five formulas | exact with zero dQ region pruned |
| C1 writer matrix reads | 16 per writer wave/CTA | -16 |
| C1 writer dQ MMAC | 32 per writer wave/CTA | -32 |
| CTA writer dQ MMAC | 128 | -128 |
| C1 zero dS writes | four per C1 wave | unchanged |
| BatchDsFilled1 / DqDone1 | existing cadence | unchanged |
| dQ accumulator / stores | canonical | unchanged |
| LDS | 128KiB | exact |
| WDRA | 16/204/88/162 accepted compiler order | no larger live range |

For H1/S1024 causal, eight CTAs remove 1,024 additional zero MMOP issues and
512 matrix reads before scheduler replication. For S2048, sixteen CTAs remove
twice that amount. Raw MMAC active may fall again because invalid MMOP is
removed; same-problem fused/lifecycle ticks remain the decision metric.

## Expected Pipeline

```text
time0  C1: publish zero dS + release raw side
       writer: wait Filled1, skip C1 reads/MMAC, arrive DqDone1
       C0: diagonal score/dP/dV/dK work

time1  writer: wait/read/MMAC C0 only, store dQ partial
       producer: raw reuse still paced by C0

time2  all roles resume canonical steady tiles
```

Expected effect: writer first-epoch matrix-read readiness and MMAC issue gaps
shrink, terminal role tail shortens, and no new barrier or register lifetime is
introduced.

## Admission

1. Static formula declarations, native matrix path, barrier IDs, LDS layout
   and output ownership remain exact; no failed phase or fallback is retained.
2. Full golden S128 causal/noncausal, S1024 and S2048 pass with warning0,
   private/spill/scratch0 and bank0.
3. Dynamic causal MMOP and matrix reads fall by the proved amount; noncausal
   dynamic work remains accepted.
4. Three paired S1024 runs lower fused/lifecycle ticks; two S2048 pairs do not
   regress.
5. XCU shows fewer writer first-epoch instructions and no increase in
   producer/C0 ownership deadlock. A raw MMAC-active drop is acceptable only
   with the exact zero-work proof and lower same-problem ticks.

Workbook: section73 in the 2026-08-23 fused5 design workbook.

## Result

Two code-generation forms were tested, then the source was restored exactly
to accepted commit `2f73cab`.

### A: Peeled First Two Writer Tiles

- Static correctness/resource gates pass at roles `9/173/87/162`, SGPR71,
  VGPR128, private/spill/scratch0 and bank0.
- S128 causal/noncausal and S1024 full golden correctness pass.
- Dynamic MMOP/LDS fall `88,064/61,056 -> 87,040/60,544`, but compiler
  control/address work rises `VALU/SCA 92,496/38,216 -> 93,668/39,016`.
- Three S1024 pairs regress fused mean
  `41,254,092 -> 41,338,570` (`+0.205%`) and lifecycle mean
  `45,385,492 -> 45,487,412` (`+0.225%`), with zero wins from three pairs.
- MMAC active falls `36.146% -> 35.700%`; wait-LGKM falls about `0.117 pp`,
  but barrier share rises about `0.314 pp`.

### B: Runtime Skip Inside Original Loop

- The original pair loop and address progression are preserved, but a
  `causal && qi==0` branch guards the C1 update.
- Static/resource and S128/S1024 correctness gates pass.
- Dynamic MMOP/LDS again fall to `87,040/60,544`, while compiler work expands
  further to `VALU/SCA 94,064/38,936`.
- The S1024 probe is `41,386,800` ticks and `35.464%` MMAC active, not better
  than the accepted paired mean. Per the two-fail rule, no more runtime writer
  specialization, S2048 run or fullperf is admitted.

## Decision

The algebraic zero dQ proof is valid, but compiler `e0f10535` cannot express
this conditional writer specialization cheaply in either tested form. The
saved writer reads/MMAC are off the critical path or serve as overlap, while
new control/address work delays ownership release. Close this source-level
route. Retry only if a compile-time causal specialization removes the branch
and preserves the original loop codegen, or if a future compiler emits no
extra VALU/SCA.

Evidence:

- `/zys/sb/runs/f5writerzero_correctness_20260823`
- `/zys/sb/runs/f5writerzero_ab_20260823`
- `/zys/sb/runs/f5writerzero_loop_correctness_20260823`
- `/zys/sb/runs/f5writerzero_loop_probe_20260823`
- workbook sections73-74
