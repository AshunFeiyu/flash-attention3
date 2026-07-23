# Tri Dao FA3 / FA4 BWD Source Audit

Date: 2026-07-23

Pinned upstream commit:

```text
b54df166ebb69b896892826014759d09b9c3c9c6
```

This audit separates source facts from Shaobo transfer hypotheses. It does not
change the canonical kernel.

## Executive Correction

The current symmetric M64/N128/D128 ownership is closer to official FA3 and
FA4-SM90 than the proposed "C0 owns all dQ, C1 owns no dQ" redesign.

For D128, the official mapping uses two heavy MMA warpgroups:

- each warpgroup owns one N64 half of full-D dK/dV;
- each warpgroup owns one D64 half of dQ;
- score and dP are partitioned by N and computed exactly once;
- total useful work is balanced at 640 of the 1280 tile MMACs per group.

FA4-SM90 has an optional single-dQ-warpgroup mode, but the dispatch selects it
for head dimension up to 96, not D128. The asymmetric D128 workbook draft is
therefore superseded unless a new Shaobo-specific resource and SQTT proof
justifies it.

## FA3 D128 Facts

### Tile and role budget

The causal/local D128 launch selects:

```text
M=64, N=128, D=128
Q stages=2, dO stages=2, P/dS stages=2
1 load warpgroup + 2 MMA warpgroups
register target=24/240/240
```

Sources:

- `hopper/flash_bwd_launch_template.h:347-354`
- `hopper/flash_bwd_kernel_sm90.h:57-65`

This rejects the idea that M128 is required before the pipeline can be
efficient. The current Shaobo tile already matches the official D128 causal
tile.

### Producer warpgroup is internally specialized

The four-warp load group is not four duplicate loaders:

- warp 0 runs the K/V/Q/dO loader;
- warp 1 runs the dQacc shared-to-global reduction/store loop;
- warps 2 and 3 do not duplicate that work.

Source: `hopper/flash_bwd_kernel_sm90.h:211-240`.

This is materially different from the rejected Shaobo experiment where the
same producer waves both published raw input and executed atomics before the
next publication. A future dedicated dQ writer must be evaluated as its own
role and pipeline, not inferred from that negative result.

### One physical shared tensor supports normal and transposed views

The source explicitly builds Q/QT, dO/dOT, K/KT and P-dS/P-dST as view
compositions over the same swizzled shared storage. It calls out that the
transpose is in the view rather than in memory.

Source: `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:136-191`.

This agrees with the Shaobo native MLS plus normal/trans `ds_read_matrix`
direction. Scalar LDS transpose, gather and lane-permute workarounds must stay
out of the canonical matrix path.

### Q and dO have different logical lifetimes

FA3 creates independent Q and dO pipelines. The loader issues dO for the
current step and Q for the next step, and the consumers release dO after dV
while Q remains needed through dK.

Sources:

- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:242-245`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:510-557`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:941-949`

The current Shaobo fused kernel uses one combined `RawFilled/RawUsed` lifetime
for Q and dO. That is too coarse, but splitting the token is the second design
step, not the first: the direct register path below removes more work without
adding token families.

### The largest missed path: direct P and dS register chaining

For the D128 configuration, `Mma_dKV_is_RS` is true. The official mainloop:

1. converts score to FP16 P;
2. reinterprets that register fragment as the dV MMA A operand;
3. converts dS to FP16;
4. reinterprets that register fragment as the dK MMA A operand;
5. writes dS once to shared memory for dQ.

Sources:

- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:75`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:898-949`

The current Shaobo code instead performs:

```text
P  -> ds_write -> wait -> ds_read -> dV MMAC
dS -> ds_write -> wait -> ds_read -> dK MMAC
dS -> ds_write again -> dQ MMAC
```

Current source: `src/fused_bwd_kernel.cpp:429-465,468-476`.

This is a strong transfer hypothesis, but not an automatically avoidable
chain: Hopper's RS fragment converter is part of its MMA contract. Shaobo must
prove that its current score-owned register fragment already has the downstream
dV/dK operand ownership. The first two tests are therefore focused
native-fragment probes:

```text
score MMAC output -> FP16 P source fragment -> dV MMAC
dP/softmax output -> FP16 dS source fragment -> dK MMAC
                                  \-> one ds_write_matrix -> dQ MMAC
```

No `bpermute`, scalar `ds_read_b32`, gather or wrong-layout path is admitted.

### Shaobo direct-register transfer result

Both isolated full-kernel probes were run with the canonical
M64/N128/D128 ownership and exact five-GEMM work:

```text
P-reg -> dV:
  dK PASS
  dV FAIL, max_abs=0.62529, rel_l2=1.34211
  role use 8/175/174, SGPR98/VGPR168, spill/private0, bank0

dS-reg -> dK:
  dV PASS
  dK FAIL, max_abs=0.0321551, rel_l2=1.25076
  role use 8/179/177, SGPR98/VGPR168, spill/private0, bank0
```

Evidence:

- `/zys/sb/fa3b/direct_p_probe/5gemm_symmetric_s128_c1_20260723_124108`
- `/zys/sb/fa3b/direct_ds_probe/5gemm_symmetric_s128_c1_20260723_124252`

The clean single-output failures prove that the local matrix
write/read is currently an ownership conversion, not a redundant readiness
wait. Tri Dao's RS idea remains architecturally valuable, but the current
Shaobo `Q @ K^T` / `dO @ V^T` register orientation cannot consume it directly.
The repository already contains the first half of the answer:
`src/dkv_kernel.cpp:448-469,571-603,639-678` computes score/dP with K/V as the
left operands and feeds its P/dS fragments directly into dV/dK. Its accepted
H1/S128 and H1/S1024 runs prove dK/dV correctness, no spill and bank0. This is
the native dKV oracle; it should be reused rather than rediscovered.

The remaining focused question is narrower: can the same K/V-left dS ownership
be published once through `ds_write_matrix`, then consumed through the native
matrix-reader view by dQ? That publication gate must pass at D128 before the
five-GEMM production kernel changes orientation.

### dQ output is a separate pipeline

Each heavy warpgroup writes its D64 FP32 dQ accumulator to shared memory. A
dedicated producer-group warp waits on per-warpgroup full/empty barriers and
uses bulk reduce-add to global dQacc.

Sources:

- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:593-667`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:951-957`

Our heavy consumers currently execute native scalar FP32 atomics directly.
Moving this work is a valid later experiment only if SQTT still shows the
atomic tail on the critical path after the direct P/dS chain is integrated.

### The writer does not remove the mathematical dQ reduction

The official full backward lifecycle is:

```text
preprocess:
    clear FP32 dQaccum
    compute dPsum = sum(dO * O)

main:
    execute five GEMMs
    accumulate dK/dV across Q blocks
    emit one local dQ partial per K-tile CTA

postprocess:
    scale/convert FP32 dQaccum to final dQ
```

The dedicated writer moves the local dQ transport and bulk reduce-add away
from the heavy MMA warpgroups. It does not make one K-tile CTA own the final
dQ. Different K-tile CTAs still contribute to the same dQaccum.

The deterministic path additionally uses a semaphore to order those
contributions. Atomicity and determinism are therefore separate contracts.
Neither Hopper bulk reduce-add nor its semaphore protocol proves that a
cross-die `sbx4` reduction is legal. This preserves the leadership concern
behind the separate 7-GEMM route.

Sources:

- `hopper/flash_bwd_launch_template.h:49-75,176-248`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:593-667`

## FA4 Facts and Boundaries

### FA4-SM90 confirms the D128 mapping

The CuTe-DSL SM90 implementation preserves M64/N128, Q2/dO2/PdS2 and
symmetric D64 dQ ownership for D128. Its optional one-dQ-WG path assigns
256/224 registers and is selected for D<=96.

Sources:

- `flash_attn/cute/interface.py:175-212`
- `flash_attn/cute/flash_bwd_sm90.py:428-440`
- `flash_attn/cute/flash_bwd_sm90.py:1487-1644`

It also preserves the direct P-to-dV and dS-to-dK register paths.

### FA4-SM100 uses a true lag-one recurrence

The one-CTA path is:

```text
prologue: S0 -> dP0 -> dV0

steady iteration:
  S(next)
  dK(current)
  dQ(current)
  dP(next)
  dV(next)

tail:
  dK(last)
  dQ(last)
```

Source: `flash_attn/cute/flash_bwd_sm100.py:2648-2753`.

This is not merely reordering operations inside one panel. It keeps current
and previous generations alive so that the next score/dP/softmax work overlaps
the previous dK/dQ MMAs.

### Why it cannot be copied mechanically

FA4-SM100 relies on:

- one asynchronous UMMA issuer warp;
- tensor memory that aliases S/P, dP/dS and dP/dQ;
- separate S/P, dP, dKV, dQ and dS pipelines;
- Q stage 2 and dO stage 1 in the one-CTA path;
- dedicated compute and dQ-reduction warps;
- optional 2-CTA MMA plus cluster/DSMEM exchange.

Sources:

- `flash_attn/cute/flash_bwd_sm100.py:134-215`
- `flash_attn/cute/flash_bwd_sm100.py:1168-1298`
- `flash_attn/cute/flash_bwd_sm100.py:2520-2753`

Shaobo has no proven TMEM-equivalent accumulator store or cross-die 2-CTA
contract. The transferable idea is the dependency schedule and explicit
lifetime model, not the Blackwell role IDs or barrier graph.

## Revised Shaobo D128 Direction

Keep:

```text
M64/N128/D128
12 waves
waves0-3  producer group
waves4-7  consumer0: N0:64 dK/dV + D0:64 dQ
waves8-11 consumer1: N64:128 dK/dV + D64:128 dQ
24/240/240 WDRA target
exact 5 GEMMs, 1280 useful MMAC/tile
```

Change in this order:

1. Keep the current P/dS matrix roundtrips in the canonical kernel. Direct
   register use with the current score ownership is rejected by isolated dV
   and dK correctness failures.
2. Reuse the accepted `src/dkv_kernel.cpp` K/V-left direct dV/dK path as the
   dKV oracle. Build only the missing focused D128 gate:
   K/V-left dS fragment -> one native matrix publication -> dQ MMAC.
3. Only if the publication gate passes, integrate the orientation as one canonical
   ownership change. Otherwise retain the current native writer/read bridge.
4. Split Q and dO release lifetimes. A feasible conservative steady budget is
   Q stage2 32 KiB + dO stage1 16 KiB + current padded dS 64 KiB + sidecar
   about 1.5 KiB = 113.5 KiB.
5. Re-profile. Only if atomic/store remains critical, try a dedicated dQ
   writer wave with explicit LDS alias and backpressure proof.
6. Only after the ownership and lifetime gates pass, design a symmetric
   prologue/main/tail lag-one
   recurrence. It must preserve both D64 dQ owners and fit two live
   generations without spill.

## Promotion Gates

Every step must preserve:

- exact dynamic MMOP count;
- no duplicate score or dP;
- full CPU golden correctness;
- no private segment, scratch or spills;
- zero LDS bank conflicts;
- native MLS/BPS + `ds_read_matrix` + MMAC main path;
- same-shape ticks reduction;
- xcu proof that the removed LDS/wait chain actually left the critical path.

The current best tag remains the restore point:

```text
best/fused5-prio-m64-20260723
```

The source audit changes the next hypothesis, not the measured baseline.

## Upstream Sources

- FA3 paper: https://arxiv.org/abs/2407.08608
- FA4 paper: https://arxiv.org/abs/2603.05451
- FA3 D128 launch:
  https://github.com/Dao-AILab/flash-attention/blob/b54df166ebb69b896892826014759d09b9c3c9c6/hopper/flash_bwd_launch_template.h
- FA3 kernel roles:
  https://github.com/Dao-AILab/flash-attention/blob/b54df166ebb69b896892826014759d09b9c3c9c6/hopper/flash_bwd_kernel_sm90.h
- FA3 mainloop:
  https://github.com/Dao-AILab/flash-attention/blob/b54df166ebb69b896892826014759d09b9c3c9c6/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp
- FA4 SM90:
  https://github.com/Dao-AILab/flash-attention/blob/b54df166ebb69b896892826014759d09b9c3c9c6/flash_attn/cute/flash_bwd_sm90.py
- FA4 SM100:
  https://github.com/Dao-AILab/flash-attention/blob/b54df166ebb69b896892826014759d09b9c3c9c6/flash_attn/cute/flash_bwd_sm100.py
- FA4 config dispatch:
  https://github.com/Dao-AILab/flash-attention/blob/b54df166ebb69b896892826014759d09b9c3c9c6/flash_attn/cute/interface.py
