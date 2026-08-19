# Fused5 dQ Reduction 2D Grid

Status: `ACCEPT_KERNEL_LOCAL_END_TO_END_OBSERVE`

## Evidence

H1/S1024 FP16 reducer SQTT has 512 waves and no no-wave idle, but spends:

- 32.49% Source latency in `s_waitcnt`;
- 12.97% issue-bubble time in the vectorized `bh/local_vec` division path;
- 5.45% latency in `v_mov_b32`, plus repeated 64-bit address arithmetic.

The linear grid hides the natural row and batch-head ownership from the
compiler. It emits reciprocal conversion and exec-mask control even though one
256-thread block always covers exactly eight complete D128 rows.

## Change

- `blockIdx.x`: one aligned eight-row group;
- `blockIdx.y`: one batch-head;
- `threadIdx.x / 32`: row within the block;
- `threadIdx.x % 32`: one four-element vector within the row;
- causal `last_k_tile`: scalar `blockIdx.x / 16`.

The workspace format, FP32 accumulation, FP16 output, launch wave count, and
global byte count do not change. This experiment changes only ownership/index
expression and is expected to remove vector divide and divergent exec-mask
work before any load-batching experiment.

## Gates

1. Main fused5 ASM and resources are unchanged.
2. Reducer has no private/spill/scratch and retains packed FP16 output.
3. Reducer ASM removes `v_rcp_iflag_f32` and `v_cvt_f32_u32`.
4. H1/S128 causal/non-causal and H1/S1024 causal correctness pass, bank0.
5. Three alternating S1024 A/B pairs decide promotion using reducer and full
   lifecycle ticks.

## Result

- Reducer resources improve from SGPR26/VGPR36 to SGPR25/VGPR25, with no
  private segment, spill, or scratch.
- `v_rcp_iflag_f32`, `v_cvt_f32_u32`, and `saveexec` disappear from generated
  ISA; packed FP16 output remains intact.
- H1/S128 causal/non-causal and H1/S1024 causal correctness pass, bank0.
- Three alternating S1024 pairs reduce reducer mean ticks
  `2,687,382 -> 2,308,670` (`-14.09%`), with all three pairs improving.
- Reducer share falls from about `5.33%` to `4.58%`. Full-lifecycle mean is
  noise-flat (`50,370,168 -> 50,360,007`, `-0.02%`) because fused-compute
  variation is larger than the saved epilogue time.

The 2D ownership is accepted as the canonical reducer topology. End-to-end
impact remains `OBSERVE`; the next hypothesis must attack the still-dominant
workspace-load wait and must not reintroduce vector division.
