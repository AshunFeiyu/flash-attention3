# Fused5 dQ FP16 Reduction Epilogue

Status: `ACCEPT_MICRO_END_TO_END`

## Boundary

- The five-GEMM kernel continues to emit uniquely owned FP32 dQ partials into
  the workspace. Its formula, tile, wave roles, LDS ownership, and barriers do
  not change.
- The existing reduction accumulates all legal `k_tile` partials in FP32 and
  converts the final four values to FP16 immediately before one vector store.
- The public fused5 `dq` buffer is therefore FP16. dK and dV remain FP32 in
  this standalone prototype.

## Hypothesis

The H1/S1024 baseline spends `2,614,430` of `49,854,805` full-lifecycle ticks
in dQ reduction (5.24%). Replacing the final 16-byte FP32 store per vector with
an 8-byte FP16 store removes a later conversion dispatch and halves final dQ
write traffic. Reduction reads remain FP32 and dominate, so the expected local
speedup is bounded and the full-lifecycle result decides promotion.

## Gates

1. Generated reduction ISA has no private segment, spill, scratch, WDRA, or
   MMAC and writes FP16 output.
2. H1/S128 causal and non-causal, then H1/S1024 causal full backward pass the
   cached CPU golden check after host conversion of the FP16 result to FP32.
3. `ldsBankConflict=0` for the complete run.
4. Compare dot, fused5, reduce, and total ticks against commit `2b6efe5` under
   compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, and SQ7.

Promotion requires lower same-shape total ticks. A faster reducer with a slower
total or failed numerical/resource gate is rejected and removed from the
canonical path.

## Result

- ISA: two `v_cvt_pk_f16_f32` instructions followed by one
  `global_store_dwordx2`; reducer metadata is SGPR26/VGPR36 with no private
  segment, spill, or scratch.
- Correctness: H1/S128 causal and non-causal plus H1/S1024 causal all pass;
  S1024 dQ `max_abs=2.57598e-7`, `rel_l2=0.00262468`, bank conflict zero.
- Three alternating H1/S1024 A/B pairs:
  - reducer mean `2,785,965 -> 2,707,857` ticks (`-2.80%`);
  - full lifecycle mean `50,801,812 -> 50,706,868` ticks (`-0.19%`);
  - fused5 compute is noise-flat (`-0.05%`), so the gain is correctly
    attributed to the reduction epilogue.

The change is promoted because it closes the FP16 output contract, removes a
future standalone conversion dispatch, and improves same-runtime total ticks.
