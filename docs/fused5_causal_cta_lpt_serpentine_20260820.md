# Fused5 causal CTA load-balanced order

## Hypothesis

For `GPU_CHIP=sb`, one fused5 CTA consumes the full 128KB LDS budget, so a CU
can host one CTA. The model dispatches one CTA to each SE's first CU, then each
SE's second CU, and continues until all 48 CUs have one CTA before starting the
next round. With grid `(H,B,K_tiles)`, `H` is the fastest dimension.

For causal attention, logical K tile `z` executes

```text
q_tiles(z) = S / Mq - z * Nk / Mq
```

and is monotonically lighter as `z` increases. The original logical order
`0,1,2,...` can leave fixed CU groups with different accumulated work.

## Per-shape LPT serpentine planner

The launch wrapper evaluates one bounded scheduling family for the actual
`B*H,K_tiles` shape under the 48-CU dispatch model:

1. `identity`: the canonical order.
2. `LPT serpentine`: causal K tiles remain in monotonically decreasing work
   bands, while consecutive width-sized bands alternate forward/reverse. This
   keeps dispatch close to Hopper BWD's LPT principle and compensates the
   partial rounds created when `B*H` does not divide 48.

Widths `1..min(K_tiles,48)` are evaluated. The planner selects the serpentine
candidate only when its modeled maximum per-CU q-tile work is strictly lower
than identity. If `causal=false`, `B*H` is a multiple of 48, or no candidate
improves the model, dispatch remains identity. This makes the optimization
shape-adaptive rather than tied to one benchmark.

The simulation accounts for full `B*H/48` cycles as common work and models
the remaining `B*H%48` CTA assignments exactly. It does not assume that
`B*H` divides 48.

For `B=1,H=16,S=8192,Mq=64,Nk=128`, the selected serpentine width is 3. The
modeled fixed-CU maximum falls from 1430 to 1390 q-tile units and utilization
rises from 96.97% to 99.76%.

## Ownership invariants

- Producer and both dKV consumers derive `k_base` from the same remapped
  logical K tile.
- The dQ writer derives `q_tile_begin` from that logical K tile.
- dQ partial workspace retains physical `blockIdx.z` as its unique slot. The
  reduction sums every physical slot; remapping this ownership index would
  create overlap.
- Formulas, five-GEMM count, MMAC count, LDS layout, WDRA windows, barriers,
  global tensor offsets, and launch dimensions do not change.
- The remap helper is called only after each explicit role branch and
  `s_set_vgpr_size`; no new pre-role VGPR state is introduced.

## FA2 precedent

The older FA2 forward path changes the grid to `(h,b,num_m_block)` and maps
causal `m_block` in reverse order. Its backward path uses
`se_balance_id/se_balance_cnt` to place equal sequence work from different
`b,h` owners together. The current planner keeps that load-balancing principle
but scores each shape against the measured Shaobo 48-CU/4-SE dispatch law.

## Admission gates

1. The host checker proves every candidate is a permutation, verifies modeled
   non-regression over representative `B*H,S` shapes, and checks the target
   H16/S8192 decision.
2. Build and fused5 static/resource gates pass with no spill, scratch, private
   segment, ordinary matrix-path DS reads, or bank conflict.
3. Correctness passes on both identity and non-identity shapes.
4. A performance promotion requires same-runtime wins on at least two causal
   shapes. A modeled improvement alone is `OBSERVE`, not `ACCEPT`.

## Validation status

- Host permutation/model gate: PASS over 112 representative shapes; 51 select
  serpentine and 61 retain identity, with no modeled final-max regression.
- `B1/H1/S128/D128/causal`: full lifecycle correctness PASS, three dispatches,
  no PMD panic or VGPR warning, `ldsBankConflict=0`.
- `B1/H16/S512/D128/causal`: non-identity correctness PASS, three dispatches,
  no PMD panic or VGPR warning, `ldsBankConflict=0`.
- Static fused kernel gate: PASS; private segment and SGPR/VGPR spills are 0.
- `B1/H16/S8192/D128/causal` MFU run uses frozen binary SHA256
  `ce24106c0ea4de6097754992ba508d3ae51f125d5ddd48797fdacea71a537a68`
  under `/zys/sb/mfu_lpt_820`; performance promotion remains pending.
