---
title: "Tri Dao FA3 BWD 对照与 Shaobo dKV 设计差距"
date: 2026-07-11
status: DESIGN_REFERENCE
scope: Shaobo FA3 BWD dKV / full BWD top-level pipeline
---

# Tri Dao FA3 BWD 对照与 Shaobo dKV 设计差距

## Sources

- FlashAttention-3 blog:
  https://tridao.me/blog/2024/flash3/
- FlashAttention-3 paper:
  https://arxiv.org/abs/2407.08608
- Open-source Hopper BWD mainloop:
  https://raw.githubusercontent.com/Dao-AILab/flash-attention/refs/heads/main/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp
- Open-source Hopper BWD kernel wrapper:
  https://raw.githubusercontent.com/Dao-AILab/flash-attention/refs/heads/main/hopper/flash_bwd_kernel_sm90.h

## What Tri Dao BWD Does Structurally

The Hopper BWD path is not a collection of tiny independent fragments.  It is
a high-cohesion `bwd_step` DAG executed by MMA warpgroups while one load
warpgroup supplies shared-memory packets.

Observed structure:

- One load warpgroup and two or three MMA warpgroups.
- Register split is explicit: load WG uses a small register budget; MMA WG can
  use a large budget.
- Separate Q and dO pipelines, plus a K/V transaction barrier.
- The main step computes:
  1. `S = Q @ K^T`
  2. `dP = dO @ V^T`
  3. softmax and `dS`
  4. write `P` / `dS` to shared memory when needed
  5. `dV`
  6. `dQ`
  7. `dK`
- Q and dO are released only after their consumers have completed the relevant
  dependent work.
- `P/dS` are used as shared intermediate layouts to feed later GEMMs, rather
  than repeatedly reconstructing view-specific fragments with ad hoc permutes.

Key design point:

- Dependencies are not removed; they are grouped into large useful islands.
  Barriers exist at real dataflow boundaries, not around every small view.

## Why Their BWD Is Not "As Fragmented" As Ours

Tri Dao's BWD still has more dependencies than FWD, but it controls them with
larger ownership epochs:

- Q and dO pipelines are stage-level, not per small subview token.
- `S/dP -> softmax/dS -> dV/dQ/dK` is one coherent consumer step.
- Shared-memory intermediate `P/dS` lets different GEMMs consume the same
  logical result without forcing global reload or duplicated score/dP.
- Pingpong/warp-specialized overlap is applied at the level of large GEMM and
  softmax islands, not at individual `ds_read_matrix` crumbs.

Our current Shaobo dKV issue is the opposite:

- The algorithm is correct and matrixized, but Q/dO ownership is too fine.
- ABarrier waits dominate because producers and consumers reach page lifetime
  boundaries too often.
- Instruction-level fixes such as read batching, vector sidecar, wait pruning,
  and BPS `vbcnt` help, but they cannot by themselves change the DAG shape.

## Implication For Shaobo 7-GEMM / dKV-Focused Route

For now leadership prefers the 7-GEMM route to avoid cross-die atomic concerns.
That means we cannot blindly copy the fused 5-GEMM full BWD topology, but we
can copy the structural principles:

1. **One canonical dKV step**
   - Keep score and dP fused in one consumer step.
   - Compute `P/dS` once per Q/K tile and feed both `dV` and `dK`.
   - Do not split dV-only and dK-only consumers.

2. **Large ownership epoch**
   - Treat a Q/dO packet as a step-level packet, not many small logical views.
   - Minimize Filled/Used token count per useful MMAC.
   - Preserve independent Q and dO release only when it is proven to reduce a
     measured producer wait.

3. **Native layout only**
   - Hot path remains `matrix_load ... bps lds` + `ds_read_matrix` + MMAC.
   - Use verified same-LDS normal/trans reads where Shaobo supports them.
   - Avoid `ds_read_b32` or permute workaround on the main matrix path unless a
     focused probe proves it wins.

4. **Pipeline target**
   - Form large blocks: `score+dP MMAC island -> softmax/dS VALU island ->
     dV+dK MMAC island`.
   - Then use two consumer groups to offset these islands, so one group has
     VALU/softmax/address work while the peer group has MMAC work.
   - Do not create artificial delay.  Stagger must come from useful work.

5. **BPS readiness**
   - `s_waitcnt_vbcnt 0` before BPS-published Filled arrivals is now enabled by
     default because it improved same-env stats without changing resources.
   - This is a local readiness fix, not a replacement for larger ownership
     epoch design.

## Next Top-Level Experiment

Design a new workbook sheet before code:

- `DAG`: score/dP once, softmax/dS once, dV/dK back-to-back.
- `Tile`: choose M/N so each ownership epoch has much more MMAC than current
  half-page fragments, while staying within 240 consumer VGPR and 128KB LDS.
- `Tokens`: count exactly how many ABarrier waits per epoch and divide by MMOP.
- `Pipeline`: time0/time1/time2 with two consumer groups offset by useful
  softmax/dS or address/store work.
- `Acceptance`: H1/S1024 correctness PASS, no spill/scratch, bank conflict 0,
  lower ticks, and xcu shows lower ownership bubble or higher MMAC active.
