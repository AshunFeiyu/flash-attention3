# Fused5 GQA KV-owned reduction

Status: `REJECT_RESOURCE`; successor `ACCEPT_GQA_WORKSPACE_REDUCTION`.

## Primary hypothesis

Shaobo native FP32 global atomics serialize the dK/dV epilogue and are not a
valid multi-die ownership contract. Replace the `(q_head,k_tile)` CTA grid with
`(kv_head,k_tile)` for GQA. A CTA loops the query heads mapped to its KV head,
retains one FP32 dK/dV accumulator set across the loop, and direct-stores once.

## Formula and ownership

For `G = Hq / Hkv` and `hq = kv * G + g`:

```text
for g in [0,G):
    P_hq  = softmax(Q_hq @ K_kv^T)
    dP_hq = dO_hq @ V_kv^T
    dS_hq = P_hq * (dP_hq - delta_hq) * scale
    dV_kv += P_hq^T @ dO_hq
    dK_kv += dS_hq^T @ Q_hq
    dQ_hq_partial[k_tile] = dS_hq @ K_kv
store dK_kv, dV_kv once
```

No score, dP, dV, dK, or dQ GEMM is duplicated. Score and dP are necessarily
per query head because Q/dO differ. dK/dV accumulation is the GQA reduction.

## Resource ledger

- Tile and roles stay `Mq64/Nk128/D128`, 16 waves: producer 0-3,
  consumers 4-7 and 8-11, dQ writer 12-15.
- Physical LDS stays 128 KiB: resident K/V 64 KiB plus two Q/dO raw pages
  64 KiB. Sidecar and dS continue to reuse released resident intervals.
- Consumer dK/dV accumulators are unchanged: eight FP32 fragments per output
  for each consumer wave. Their lifetime grows across query-head epochs but
  their count does not grow.
- dQ partial workspace remains indexed by physical Q head and physical
  `blockIdx.z`; the existing reduction kernel is unchanged.
- Useful CTA count becomes `B * Hkv * ceil(Sk/Nk)`. Initial performance
  admission requires at least 48 CTAs on single-die `sb`.

For `B1/Hq16/Hkv2/S8192/Nk128`, useful CTAs are `1*2*64=128`; the grid covers
48 CUs. For `S1024`, only 16 CTAs exist, so that shape is correctness evidence,
not the grouped-path MFU target.

## Token lifecycle

K/V are published and latched once. Raw Q/dO and dS ownership repeats for each
query head, but every ABarrier phase counter is continuous across head
boundaries:

```text
time0: resident K/V load -> ResidentFilled -> all non-producer roles latch
head g, tile q0: Raw0 Filled/Used -> dS generation -> dQ Done
head g, tile q1: Raw1 Filled/Used -> dS generation -> dQ Done
...
head g+1: continue the same token phases; do not reset them to zero
tail: consumers direct-store accumulated dK/dV once
```

Producer waits for the final RawUsed page before publishing the next head.
This first implementation preserves correctness and exact ownership; overlap
across head boundaries is a later hypothesis only if SQTT proves the boundary
is material.

## Admission and comparison

- Full CPU golden: `Hq4/Hkv2/S128`, then `Hq16/Hkv2/S1024`.
- Static: one fused symbol, no global atomic instruction, no private/spill/
  scratch, main matrices remain MLS/BPS + ds_read_matrix + MMAC, bank0.
- MHA `Hq=Hkv` must remain correct and stay in the frozen LPT performance band.
- Long performance: `B1/Hq16/Hkv2/S8192/D128/causal/SQ7` against the frozen
  atomic baseline with identical PMD/compiler.
- Primary metrics: full lifecycle and fused ticks. Explanation metrics: MMAC
  active, FLAT/global-credit stalls, active-CU balance, ABarrier and matrix-read
  bubbles.

## Resource result and redesign

The direct KV-owned CTA formulation preserves the mathematical work ledger,
but it is not compiler-resource admissible on the target toolchain. The outer
query-head epoch extends scalar/control lifetime across the full consumer and
writer bodies:

- initial build: private 20 B, SGPR104 with 10 spills, VGPR128 with 4 spills;
- kernel-ABI packing: private 196 B, SGPR104 with 9 spills, VGPR128 with 92
  spills;
- countdown/pointer-increment cleanup did not change the rejected resource
  class.

No PMD run was admitted for these binaries. The accepted topology therefore
keeps one `(q_head,k_tile)` CTA and writes uniquely owned FP32 dK/dV partials.
A separate `(batch,kv_head,row,D-vector)` reduction kernel sums the G query
heads and stores final FP32 dK/dV. This removes native global atomics without
putting the G loop inside the WDRA kernel.

```text
fused CTA(q_head,k_tile):
  five GEMMs once -> dQ partial[k_tile] + dK partial[q_head] + dV partial[q_head]

dKV reduce CTA(kv_head,row):
  sum g=0..G-1 partial[batch,kv_head*G+g,row,:]
  -> final dK/dV[batch,kv_head,row,:]
```

The reduction is linear in `B*Hq*S*D`, while fused attention is quadratic in
sequence length. It also retains `B*Hq*Ktiles` fused CTAs, avoiding the useful
CTA coverage loss of the rejected KV-owned grid.

## Accepted evidence

- Main fused metadata: SGPR82/VGPR128, private/spill/scratch0; branch usage
  `9/187/87/182`; main matrix path remains MLS/BPS + ds_read_matrix + MMAC.
- dKV reducer metadata: SGPR24/VGPR24, private/spill/scratch0.
- `B1/Hq4/Hkv2/S128/D128/causal`: full CPU golden PASS, bank0;
  fused `12,139,400`, dKV reduce `904,540`, lifecycle `16,321,305` ticks.
- Atomic control on the same shape: lifecycle `26,699,855` ticks. Workspace
  reduction improves end-to-end ticks by `38.87%`; reducer share is `5.54%`.
- `B1/Hq16/Hkv2/S128` (G=8): full CPU golden PASS, bank0; lifecycle
  `19,237,855`, dKV reduce `2,517,515` ticks (`13.09%` on this tiny shape).
- MHA `B1/H1/S1024`: correctness PASS and bank0; lifecycle `49,252,385`,
  within `0.58%` of the frozen `48,970,285` GQA-capable control.
- `B1/Hq16/Hkv2/S1024` (G=8): full CPU golden PASS and bank0; dot
  `14,165,060`, fused `79,428,440`, dQ reduce `15,730,260`, dKV reduce
  `11,165,245`, lifecycle `120,489,005` ticks. The atomic same-shape control
  is `256,707,360` ticks, so workspace reduction lowers lifecycle ticks by
  `53.06%`; the dKV reducer is `9.27%` of the accepted lifecycle.
- The scale case has dQ `max_abs=8.63e-6` and `RMSE=1.14e-6`. Its elevated
  relative L2 comes from a very small reference norm and is bit-for-bit the
  same as the atomic control, so the FP16 dQ gate accepts absolute error plus
  either relative L2 or RMSE. This does not weaken dK/dV checks.

Decision: promote workspace reduction as the GQA ownership contract. Treat the
small MHA delta as `OBSERVE`. The next GQA performance step is SQTT evidence at
scale; reducer tuning is admitted only if it lowers full lifecycle ticks.
