# Tri Dao FA3 / FA4 BWD Source Audit

Date: 2026-07-23

Pinned upstream commit:

```text
b54df166ebb69b896892826014759d09b9c3c9c6
```

Local read-only sparse clone:

```text
/Users/zhangyushun/Documents/Codex/2026-06-08/shaobo-hip-shaobo-demo/work/upstream/flash-attention-main
```

The clone contains `hopper`, `flash_attn/cute` and
`csrc/flash_attn/src`. Its `origin/main` and `HEAD` both resolve to the pinned
commit above.

This audit separates source facts from Shaobo transfer hypotheses. It does not
change the canonical kernel.

## Source-Derived Top-Level Design

### Complete backward lifecycle

The official path is not a single kernel:

| Dispatch | Source responsibility | Output contract |
| --- | --- | --- |
| preprocess | compute `dPsum_i = sum_d(dO_i,d * O_i,d)`, convert LSE to log2 form and clear FP32 `dQaccum` | FP32 sidecar and zeroed reduction buffer |
| main backward | one CTA owns one `N=128` K/V tile, loops over `M=64` Q/dO tiles, executes exactly five GEMMs, accumulates dK/dV and emits one dQ partial per K tile | final dK/dV tile plus FP32 dQ partial reduction |
| postprocess | multiply FP32 `dQaccum` by `softmax_scale`, convert to FP16/BF16 and store final dQ | final b16 dQ |

Sources:

- `hopper/flash_bwd_launch_template.h:49-75`
- `hopper/flash_bwd_preprocess_kernel.h:202-239`
- `hopper/flash_bwd_launch_template.h:176-248`
- `hopper/flash_bwd_postprocess_kernel.h:200-216`

### D128 tile and exact ownership

For causal/local D128, the launch arguments instantiate:

```text
M=64, N=128, D=128
Q stages=2, dO stages=2, dS stages=2
SdP_swapAB=true, dKV_swapAB=false, dQ_swapAB=false
NumMmaWarpGroups=2
AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1
```

The resulting two heavy warpgroups are symmetric:

| Work per heavy WG | Matrix shape | Shaobo-equivalent 16x16x16 MMAC count |
| --- | --- | ---: |
| score | `K[N64,D128] @ Q^T[D128,M64] -> S^T[N64,M64]` | 128 |
| dP | `V[N64,D128] @ dO^T[D128,M64] -> dP^T[N64,M64]` | 128 |
| dV | `P^T[N64,M64] @ dO[M64,D128] -> dV[N64,D128]` | 128 |
| dQ | `dS[M64,N128] @ K[N128,D64] -> dQ[M64,D64]` | 128 |
| dK | `dS^T[N64,M64] @ Q[M64,D128] -> dK[N64,D128]` | 128 |

Each heavy WG therefore owns 640 useful equivalent MMACs; the CTA owns 1280.
Score and dP are split in N and computed exactly once. dK/dV are split in N,
while dQ is split in D. This equal-volume mapping is the central D128 load
balance, not an incidental code-generation detail.

Sources:

- `hopper/flash_bwd_launch_template.h:293-304,347-354`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:69-134`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:742-774`

### Twelve-warp role map

The kernel has one load warpgroup and two MMA warpgroups, with
`24/240/240` target registers:

```text
warps 0-3   load / output warpgroup
  warp 0    TMA K/V once; Q/dO/LSE/dPsum streaming
  warp 1    dQ shared-to-global bulk reduce-add writer
  warp 2-3  no duplicate loader or heavy MMA body

warps 4-7   MMA WG0
  N0:64 dK/dV owner, D0:64 dQ owner

warps 8-11  MMA WG1
  N64:128 dK/dV owner, D64:128 dQ owner
```

The dQ writer is therefore a specialization inside the existing load
warpgroup. It is not a fourth heavy warpgroup and does not compute another dQ
GEMM.

Sources:

- `hopper/flash_bwd_kernel_sm90.h:57-65`
- `hopper/flash_bwd_kernel_sm90.h:211-266`

### Mainloop pipeline

The source implements this prologue/steady/tail schedule for one fixed K/V
tile:

```text
prologue:
  producer warp0: publish Q0 + LSE0
  producer warp0: publish K + V under one combined transaction barrier

steady producer recurrence for m:
  publish dO[m] + dPsum[m]
  publish Q[m+1] + LSE[m+1]

steady heavy-WG step for m:
  wait Q[m]
  enqueue score[m]
  wait dO[m]
  enqueue dP[m]
  wait until <=1 GMMA group remains
  softmax(P[m]) while dP may still be outstanding
  wait all score/dP GMMA
  form dS[m]
  convert P and dS to b16
  publish dS once to shared
  enqueue dV[m] directly from register P
  synchronize the two heavy WGs on the full dS tile
  enqueue dQ[m] from shared dS and resident K
  release dO[m] after its last use
  enqueue dK[m] directly from register dS and Q
  publish FP32 dQ[m] to the per-WG shared output slice
  wait all GMMA, release Q[m]

concurrent writer recurrence:
  wait dQFull[WG0], bulk reduce-add D0:64 to global dQaccum
  wait dQFull[WG1], bulk reduce-add D64:128 to global dQaccum
  signal dQEmpty[WG0/1]

tail:
  scale accumulated dK
  convert dK/dV to b16
  register-to-shared matrix store
  TMA store dK/dV once
```

The GMMA helper only executes `warpgroup_wait<N>` when `wg_wait >= 0`.
Consequently the `-1`, explicit `wait<1>`, elementwise work, then `wait<0>`
sequence is intentional latency overlap rather than compiler accident.

Sources:

- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:506-557`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:834-965`
- `hopper/utils.h:254-321`
- `hopper/epilogue_bwd.hpp:164-222`

### Shared-memory views and ownership synchronization

- K and V are loaded once for the CTA under one combined transaction barrier.
- Q and dO have independent two-stage TMA pipelines.
- One physical Q allocation exposes Q and Q-transpose views; dO and K use the
  same view-composition technique. No in-memory transpose is performed.
- P remains in registers for dV because `Mma_dKV_is_RS=true`.
- dS remains in registers for dK and is written once to shared for dQ.
- The `PdS` named barrier joins both heavy WGs because each D64 dQ owner needs
  the full N128 dS tile produced as two N64 halves.
- Each heavy WG has its own `dQEmpty/dQFull` handshake with the writer warp.
- dK/dV FP32 accumulators live across all M iterations and are stored only in
  the epilogue.

Sources:

- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:136-209`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:242-289`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:510-557`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:898-957`
- `hopper/named_barrier.hpp:61-70`

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
next publication. The official design gives loading and dQ reduction to
different warps inside the producer group, so the Shaobo transfer must preserve
that internal specialization.

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
current step and Q for the next step. dO's last mathematical use is dV; the
source schedules its pipeline release after issuing dQ. Q remains needed
through dK and is released only after the final GMMA wait.

Sources:

- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:242-245`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:510-557`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:927-1000`

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
The accepted single-dS SQTT now measures atomic latency at about 19% of the
selected source window, so moving this work is no longer a speculative
afterthought. It is the next source-backed structural experiment after the
already-built Q-latch candidate is classified.

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
5. Re-profile the Q-lifetime candidate, then specialize a warp inside the
   existing producer group as the dQ writer. Reuse the dead dS/output LDS
   lifetime where possible and prove the per-WG empty/full backpressure; do not
   add a fourth heavy warpgroup.
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
