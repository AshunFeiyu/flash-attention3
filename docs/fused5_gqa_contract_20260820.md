# Fused5 GQA contract

## Algebra and ownership

For `group_size = Hq / Hkv`, query head `hq` consumes
`kv_head = hq / group_size`. The five GEMMs and dQ ownership remain per query
head. Gradients of a shared KV head require

```text
dK[b,kv] = sum_hq_in_group dK[b,hq]
dV[b,kv] = sum_hq_in_group dV[b,hq]
```

Therefore changing only the K/V read index is insufficient: direct dK/dV
stores from multiple query-head CTAs would race. The canonical fused kernel
uses separate Q and KV tensor bases and, only for `group_size > 1`, atomically
accumulates FP32 dK/dV into a pre-zeroed KV-head buffer. MHA keeps the existing
vector direct-store path.

## Interface lifecycle

The pure C/HIP wrapper owns the same boundary as Hopper FA3:

1. Validate `Hq % Hkv == 0`.
2. Zero FP32 dK/dV accumulation buffers for GQA.
3. Launch the same fused five-GEMM kernel on the existing `(Hq,B,Ktiles)` grid.
4. Reduce dQ partials as before.
5. A future framework binding may convert FP32 dK/dV accumulators to the
   requested output dtype in explicit postprocess kernels.

PyTorch is not part of PMD execution correctness. It may allocate tensors in a
framework integration, but the model-facing implementation remains pure HIP.

## Admission gates

- MHA `Hq=Hkv` must retain the current direct-store ISA and same-shape ticks.
- GQA correctness first targets `B1/Hq4/Hkv2/S128/D128/causal`.
- No private segment, SGPR/VGPR spill, matrix-path fallback, or LDS bank
  conflict.
- GQA atomic aggregation is accepted only with full dQ/dK/dV CPU golden; an
  index-only smoke is not sufficient.

## Atomic baseline evidence

- Compiler `e0f10535`, gfx946: fused symbol is SGPR84/VGPR128 with private
  segment, SGPR spill and VGPR spill all zero. The same symbol contains 32
  direct `global_store_dwordx4` instructions and 128 native
  `global_atomic_add_f32` instructions.
- `B1/Hq4/Hkv2/S128/D128/causal`: full delta/dQ/dK/dV CPU golden PASS,
  fused ticks `23,596,300`, full lifecycle ticks `26,699,855`,
  `ldsBankConflict=0`, no PMD panic or VGPR warning.
- MHA regression `B1/Hq1/Hkv1/S1024` PASS. After hoisting direct and atomic
  output islands and adding the `q_heads_per_kv == 1` fast path, the
  GQA-capable symbol is `44,944,900` fused ticks versus `44,524,025` for the
  frozen LPT-only control (`+0.945%`). Full lifecycle is `48,970,285` versus
  `48,725,495` (`+0.502%`). This is a correctness baseline, not a promoted
  performance path.

## Next ownership experiment

For a KV-owned CTA, launch on `(Hkv, B, Ktiles)`, load resident K/V once, and
iterate all `Hq/Hkv` query heads mapped to that KV head. Keep dK/dV FP32
accumulators live across the head loop and direct-store them once. Each query
head keeps its own Q/dO/sidecar and dQ-partial ownership. Raw-page and dS
ABarrier phases must continue across head boundaries; resetting phase state at
each head is invalid.

The grouped path is intended for shapes with enough useful CTAs, initially
`B * Hkv * Ktiles >= 48` on one `sb` die. A later hybrid may retain an expanded
partial-reduction path for smaller grids, but that fallback is not part of the
first ownership experiment.
