# SQTT Bottleneck Reanalysis And Next Plan, 2026-07-13

## Scope

- Repo: `remote_src/fa3_bwd_wasp_clean`
- Current canonical code: dKV cleanup commit `d48b53d`, with later record-only
  commit `2466ead`.
- Target quick shape: `B=1,H=1,S=1024,D=128,causal=true`.
- Env: `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.
- Caveat: the main dKV SQTT artifact below predates the final cleanup refactor,
  but the cleanup preserved formula, tile, ownership, ABarrier lifecycle, and
  native matrix path.  Before code changes, capture a fresh current-canonical
  dKV fullperf/xcu point.

## dKV Evidence

Artifact:

- Perf:
  `/zys/shaobo_runs/dkv_curfp_20260713/dkv_mmac_correctness_20260713_132949/m5out/0/0/2800042_fa3_bwd_wasp_clean.perf`
- XCU:
  `/zys/shaobo_runs/dkv_curfp_20260713/xcu_outputs`

Top dispatch-level SQTT evidence:

- `s_abarrier_try_wait -> s_xor_b32`: `35.85%` bubble, `2048` hits.
- `s_abarrier_try_wait -> s_waitcnt`: `8.36%` bubble.
- `v_mmac -> v_mmac`: `8.21%` bubble.
- `ds_read_matrix_format -> s_waitcnt`: `3.94%` bubble.
- `ds_read_matrix_trans_format -> s_waitcnt`: `2.88%` bubble.
- `matrix_load_32x16_b16 -> s_waitcnt_vbcnt`: `2.19%` bubble.
- Hot opcodes: `s_xor_b32` `34.16%`, `s_waitcnt` `19.59%`,
  `v_mmac_f32_16x16x16_f16` `10.82%`.

Role-window evidence:

- Producer window has `98.53%` bubble; top bubble is
  `s_abarrier_try_wait -> s_xor_b32`.
- Consumer-selected SIMD window has `91.25%` bubble; top bubble is
  `v_mmac -> v_mmac`.
- Consumer wave slots with MMAC show only about `15%` MMAC+VALU coissue, and
  the top coissued VALU is `v_mov_b32_e32`, not useful softmax/control work.

Interpretation:

- The primary limiter is ABarrier ownership/lifetime, not missing MMAC
  instructions or global memory bandwidth.
- Matrix read readiness is secondary but still actionable.  The key gaps are
  the explicit `ds_read_matrix -> wait_lgkm -> first use` boundaries.
- More LDS buffering is not enough.  Q-only double buffering was correct and
  resource-clean but regressed H1/S1024 `46.376M -> 49.101M` simTicks because
  SCA/barrier bookkeeping increased.

## dQ Evidence

Artifact:

- Perf:
  `/zys/shaobo_runs/dq_boundary_page_split_fullperf_20260712_225237/dq_correctness_20260712_225237/m5out/0/0/2796314_fa3_bwd_dq_clean.perf`
- XCU:
  `/zys/shaobo_runs/dq_boundary_page_split_fullperf_20260712_225237/xcu_outputs`

Top dispatch-level SQTT evidence:

- `s_abarrier_try_wait -> s_xor_b32`: `22.62%` bubble.
- `s_barrier -> s_cbranch_vccnz`: `15.56%` bubble.
- `s_cmp_lg_u32 -> s_waitcnt_vbcnt`: `7.37%` bubble.
- `global_load_dword -> s_waitcnt`: `2.07%` bubble.
- Pipeline selected window: `50.55%` bubble, top opcode `mmop_fp16`,
  top bubble `lds_matrix -> immed`.
- Heavy consumer wave slots have high coissue count, but `MMAC+VALU` coissue
  is `0%` in the selected window; producer wave slots are thin.

Interpretation:

- dQ is closer to the quick target on S2048, but S1024 remains ownership and
  control dominated.
- The visible `s_barrier` bubble is real, but earlier attempts to replace
  terminal CTA sync with ABarrier invalidation failed.  Do not remove it without
  a proven two-phase invalidation protocol.
- Need fresh S2048 xcu for current canonical dQ before optimizing based on the
  older S1024 window.

## Next Optimization Plan

### Step 0: Fresh Current-Canonical Evidence

Capture fresh fullperf/xcu before editing:

- dKV H1/S1024, current canonical after cleanup.
- dKV H1/S2048 if H1/S1024 still matches the old bottleneck signature.
- dQ H1/S2048, because dQ S2048 is near the 40% target and is the better
  steady-state diagnostic.

Required outputs:

- `detail.txt`
- `wavefronts_bubbles.txt`
- `pipeline_summary.csv`
- `simd_mix.csv`
- `coissue_summary.csv`

### Step 1: dKV ReleasePage Q-Read Wait Hiding

Hypothesis:

- Current ReleasePage path does:
  `dO reads -> wait/release dO -> Q reads -> wait/release Q -> softmax/dS -> dV/dK MMAC`.
- Softmax/dS does not depend on the Q source reads for dK; it only needs
  `score`, `dp`, and sidecar.
- Try:
  `dO reads -> wait/release dO -> issue Q reads -> softmax/dS while Q reads are in flight -> wait/release Q -> dV/dK MMAC`.

Why this is first:

- It targets measured `ds_read_matrix -> s_waitcnt` bubbles without adding
  LDS pages or ABarrier tokens.
- It preserves the proven dO-first release order.
- It should not materially increase live ranges because Q source registers are
  already live across softmax in the current path after the wait.

Accept criteria:

- H1/S128 and H1/S1024 correctness PASS.
- no spill/scratch/private segment, `ldsBankConflict=0`.
- H1/S1024 same-shape ticks improve.
- xcu shows lower `ds_read_matrix -> s_waitcnt` or lower total `s_waitcnt`
  without increasing ABarrier/control enough to cancel the win.

Reject if:

- consumer VGPR exceeds 240, SGPR spills, or ticks regress.

### Step 2: dKV Ownership Token Audit, No New Buffers

Do not add pages.  Instead, count the actual wait/arrive frequency per q tile:

- `Q0Filled/Q0Used`
- `Q1Filled/Q1Used`
- `Dout0Used`
- `Dout1Used`
- `ResidentFilled/Used`

Only consider a token merge if the fresh xcu still shows the same
`s_abarrier_try_wait` dominance and the proposed merge removes an ownership
epoch rather than moving it.

Candidate to stress only after Step 1:

- whole-Mq Q/dO filled/used token versus current half-token conveyor.

Risk:

- It may delay half-tile consumer start.  This is a top-level pipeline tradeoff,
  not an instruction tweak; require workbook note before coding.

### Step 3: dKV Producer Useful Work While Waiting

If producer windows remain `>95%` bubble after Step 1:

- Move only independent next-packet work before `QUsed/DoutUsed` waits:
  source descriptor preparation, sidecar address/value preparation, and
  lightweight predicate metadata.
- Do not add new ABarrier tokens or LDS pages.
- Do not duplicate score/dP or add raw transpose/gather paths.

Accept only if producer bubble falls and same-shape ticks fall.  Prior dQ
source-lookahead was not enough, so treat this as a small bounded experiment.

### Step 4: dQ Evidence Refresh Before Code

For dQ, do not start with code:

- Capture fresh S2048 xcu for the current canonical dQ.
- Compare S1024 and S2048 bubbles.  If S2048 MMAC active remains near 40% and
  S1024 is dominated by startup/control, optimize for S2048 steady state, not
  S1024-only cosmetics.

Likely dQ work after evidence:

- Revisit terminal `s_barrier` only with a safe two-phase invalidation design.
- Audit `s_cmp_lg_u32 -> s_waitcnt_vbcnt` around producer sidecar/KV publish.
- Do a causal=false diagnostic run to separate mask/control cost from pipeline
  ownership cost.  Do not promote causal=false code; use it only to quantify.

## Non-Goals

- Do not add Q/dO pages just to buffer more.
- Do not chase `s_xor_b32` directly; it is the first instruction after
  ABarrier wait and mainly exposes ownership wait, not a standalone problem.
- Do not promote coissue count unless useful MMAC active and ticks improve.
- Do not reintroduce `ds_read_b32`, bpermute, gather, or wrong-layout
  workarounds in the main matrix path.
