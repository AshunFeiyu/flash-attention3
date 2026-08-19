# Fused5 dQ FP16 Partial Workspace

Status: `ACCEPT_END_TO_END`

## Boundary

- dQ writer MMAC accumulators remain FP32.
- Each uniquely owned dQ partial is converted once to FP16 for workspace
  storage.
- The reducer loads FP16 vectors, converts them to FP32, accumulates all legal
  `k_tile` partials in FP32, and writes final FP16 dQ.
- Five-GEMM arithmetic, tile, roles, LDS, ABarrier and ownership do not change.

## Motivation And Budget

The accepted 2D reducer still spends 44.95% Source latency in `s_waitcnt` and
29.91% issue-bubble duration between workspace loads and waits. Synchronous
two-load batching regressed because it raised VGPR25 to VGPR36.

Changing the workspace element type halves both sides of the boundary:

- allocation: FP32 `4 B/element` -> FP16 `2 B/element`;
- main-kernel dQ partial writes: 50% fewer bytes;
- reducer partial reads: 50% fewer bytes;
- no additional dispatch.

For H1/S1024 causal, 589,824 valid partial elements move about 2.36MB in each
direction as FP32; FP16 reduces each direction to about 1.18MB.

## Risks And Gates

1. Partial quantization occurs before cross-`k_tile` summation and may increase
   dQ error, especially at S2048. Correctness decides admission.
2. Main fused5 and reducer must remain private/spill/scratch0 and bank0.
3. ASM must show packed FP16 partial store and packed FP16 reducer load; no
   scalarized four-store workaround.
4. Run H1/S128 causal/non-causal, H1/S1024 causal, then H1/S2048 causal.
5. Three S1024 paired A/B runs compare fused, reducer and total ticks against
   `1678545`; promotion requires lower total and acceptable dQ error.

## Result

- ISA: the fused writer uses packed FP16 partial stores; the reducer uses
  `global_load_dwordx2`, FP16-to-FP32 conversion, FP32 accumulation, and one
  packed FP16 final store.
- Resources: fused SGPR62/VGPR128 and reducer SGPR20/VGPR13; both are
  private/spill/scratch0. LDS bank conflicts remain zero.
- Correctness: H1/S128 causal and non-causal, H1/S1024 causal, and H1/S2048
  causal pass the cached CPU golden lifecycle.
- Three alternating H1/S1024 pairs improve reducer mean ticks
  `2,334,453 -> 1,933,295` (`-17.18%`), fused mean
  `45,629,827 -> 45,396,108` (`-0.51%`), and total mean
  `50,415,668 -> 49,779,123` (`-1.26%`).
- The H1/S2048 scale check improves reducer `7,166,705 -> 6,386,835`
  (`-10.88%`) and total `95,125,030 -> 93,932,475` (`-1.25%`).

The fresh reducer SQTT still attributes 31.75% issue-bubble duration to
`global_load_dwordx2 -> s_waitcnt`; this is now the next reducer-local target.
The fused kernel remains the end-to-end target: ABarrier wait and LDS
read-to-wait bubbles dominate, so reducer work alone cannot raise fused MMAC
active from about 35% to 50%.
