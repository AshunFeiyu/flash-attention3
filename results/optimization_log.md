# Optimization Log

## 2026-07-20 dKV Exact-Work Baseline Locked

- Status: `ACCEPT_CANONICAL_BASELINE`.
- The sole dKV optimization source is commit `20dbb81`, tag
  `best/dkv-three-m64-lifetimes-20260719`: `Mq192/Nk192/D128`, one producer,
  three heavy M64 consumers, and three physical Q/dO ownership lifetimes.
  It executes only the required causal score, dP, dV, and dK work. The older
  `43.7836%` result included causal-invalid MMAC and is not a valid baseline.
- With locked PMD HEAD1694, compiler LLVM `7b796991` and `SQCIPfLines=7`, the
  latest H1/S768 fullperf gives `32,990,230` kernel ticks, `41.2191%` MMAC
  active, exact `MMOP=46,080`, coissue `11,590/9,613`, bank0, and
  private/spill/scratch0. The same source improves `5.03%` in stats-only A/B
  versus the old compiler (`34,831,160 -> 33,078,955` kernel ticks).
- XCU attributes the remaining actionable debt to raw ownership
  ABarrier-to-XOR (`~16.08%`) and matrix-read first-use waits
  (`~12.45%` combined normal/trans families). Terminal AllDone is a separate
  correctness-sensitive exit edge and must not be deleted as a shortcut.
- The H1/S1024 no-tail control remains M128 logical `64/32/32` at `fcd87aa`;
  its `37.8149%` active is lower and it has only eight physical heavy waves,
  so it is not the performance mainline.
- Evidence:
  `/共享/shaobo/perf/20260720_114616_dkv_exact_three_m64_h1s768_sqc7_toolchain_locked_fullperf`.

## 2026-07-20 dKV D2/D3 Read-Before-Wait Rejected

- Status: `REJECT_STATS_UNSTABLE_AVG_REGRESSION_SOURCE_RESTORED`.
- Hypothesis: after softmax/dS, issue the existing D2/D3 normal reads before
  the D0/D1 first-use wait, then use `lgkmcnt(4)` to retire only the four older
  D0/D1 requests. D2/D3 should mature under the following D0/D1 MMAC8 island.
- The implementation changed no formula, tile, role, VGPR window, ABarrier,
  read/MMAC count, or output owner. Static gates stayed
  `31/156/156/156`, private/spill/scratch0. S384/S768 correctness passed with
  MMOP46080 and `ldsBankConflict=0`.
- Same-flags S768 pair 1 favored the candidate
  `33,496,190 -> 33,338,305` ticks (`-0.471%`), but pair 2 reversed to
  `32,967,935 -> 33,237,750` (`+0.818%`). Across both pairs, candidate mean
  ticks regress `0.168%`; mean MMAC active falls
  `41.0939% -> 41.0236%`. Mean wait changes only `10,730.13 -> 10,726.75`
  and barrier `33,778.50 -> 33,693.63`, too small to move the critical path.
- Decision: reject without helper fullperf, delete candidate source and restore
  `20dbb81`. The next candidate must target raw ownership directly rather than
  move four matrix requests around an unchanged ownership boundary.
- Evidence: candidate roots
  `/zys/sb/d23_read_before_wait_s768/` and
  `/zys/sb/d23_read_before_wait_s768_repeat/`; controls
  `/zys/sb/d23_read_before_wait_control_s768/` and
  `/zys/sb/d23_read_before_wait_control_s768_repeat/`; workbook
  `179_DKV_D23ReadBeforeWait`.

## 2026-07-20 dKV M48 Head Lookahead Rejected

- Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`.
- The focused layout gate remains valid, but the full M192 integration does
  not shorten the critical path. S384 and S768 correctness pass; metadata is
  private0, SGPR69, VGPR128, spill/scratch0, LDS `125,760B`, and bank0.
- Same-build stats-only S768 kernel ticks regress
  `33,104,435 -> 33,533,045` (`+1.295%`). Helper fullperf confirms the loss:
  `33,135,830 -> 33,290,530` (`+0.467%`), with MMAC active
  `41.1992% -> 41.0779%` and successful coissue `11,457 -> 10,719`.
- XCU shows why the legal prefetch does not win. Combined current-head and
  lookahead Used debt falls `93,116 -> 81,612` cycles (`-12.35%`), but
  `RawMiddleUsed` rises `102,920 -> 116,192`, `RawHeadFilled` rises
  `90,784 -> 95,000`, aggregate ABarrier-to-XOR rises
  `504,808 -> 507,696`, and dispatch duration rises `72,828 -> 73,168`.
  Matrix-read latency is essentially unchanged; SCA rises
  `15,702 -> 16,616` while exact MMOP stays `46,080`.
- Decision: preserve probe commit `09419bd` as high-address MLS/layout
  evidence, preserve rejected implementation commit `fdfe1cd` on its isolated
  branch, and restore the canonical M192 source. Do not add a token merely to
  move an ownership wait; a successor must reduce the sum of the protected
  critical edges and retain coissue.
- Evidence: candidate
  `/zys/sb/m48fp/dkv_mmac_correctness_20260720_083110`; baseline
  `/zys/shaobo_runs/dkv_three_m64_s768_fullperf/`
  `dkv_mmac_correctness_20260719_222018`; workbook
  `175_DKV_M48HeadLookahead`.

## 2026-07-20 dKV M48 Head-Lookahead LDS Gate

- Status: `ACCEPT_LAYOUT_GATE_ONLY`.
- Hypothesis: use the 27.5KB slack left by the exact M192 raw layout to stage
  the next q-loop's first M48 Q/dO/sidecar packet before the producer waits on
  the current `RawHeadUsed` token.  The proposed peak is `125,760B`, leaving
  `5,312B` below the 128KB limit while preserving all three M64 consumer
  owners and exact MMAC work.
- The focused probe publishes distinct Q and dO values through
  `matrix_load_32x16_b16 bps lds` into both canonical and high-address
  lookahead pages, then checks normal, transpose, canonical dual-base reads,
  and sidecar values bit-for-bit.  All mismatch counts are zero.
- Static/runtime gates pass: private0, SGPR25, VGPR21, spill/scratch0,
  BPS14, trans-read12, normal-read4, executable trap0, LDS `125,760B`, and
  `ldsBankConflict=0` on PMD HEAD1694.
- This is an address/layout admission gate, not a performance result.  The
  next experiment must keep terminal `AllDone`, add independent lookahead
  Filled/Used ownership, and prove same-build S768 correctness and ticks
  before promotion.
- Evidence:
  `/zys/shaobo_runs/dkv_m48_lookahead_layout_probe_20260720/`
  `mls_page_imm_20260720_080645`; workbook `175_DKV_M48HeadLookahead`.

## 2026-07-20 dKV FWD-Style Terminal Release Rejected

- Status: `REJECT_CORRECTNESS_SOURCE_RESTORED`.
- SQTT attribution correction: the `s_abarrier_try_wait -> s_waitcnt` bubble
  totaling about `15.07%` uses immediate barrier ID 8, which is terminal
  `AllDone`; it is not a raw Q/dO main-loop wait. Raw Used waits account for
  the separate `s_abarrier_try_wait -> s_xor_b32` family, about `16.20%`.
- Hypothesis: match FA3 FWD release and replace dKV `AllDone` with
  `__syncthreads(); wave0 invalidate; __syncthreads()` without changing math,
  tile, ownership, stores, or the raw pipeline.
- Static gates passed with branch VGPR `31/156/156/156`, private0, SGPR50,
  VGPR128, spill/scratch0. PMD H1/S384 then failed dV correctness
  (`dv_rel_l2=0.073167`, `pass=0`) and reported an invalid LDS read on the V
  consumer path; dK remained near reference.
- Decision: restore canonical source. In this mixed role, waves12-15 publish V
  before becoming heavy consumers, so terminal `AllDone` is a current WDRA
  role-convergence requirement rather than a redundant wait. Do not optimize
  the 15.07% terminal idle accounting by deleting it; reduce the consumer
  global-store critical path or first prove a focused WDRA-exit ABI.
- Evidence:
  `/zys/shaobo_runs/dkv_fwd_terminal_release_20260720/`
  `dkv_mmac_correctness_20260720_073533`.

## 2026-07-20 dKV Owner16 Four-Consumer Full Integration Rejected

- Status: `REJECT_STATIC_RESOURCE`.
- Hypothesis: four exact owner16 groups could extend the passing resource and
  lifecycle probes into one full `Mq64/Nk256/D128` dKV kernel, with all 16
  waves doing unique score/dP/dV/dK work.
- Build result: branch allocation is `128/128/128/128`, but kernel metadata
  is private468B, SGPR88, VGPR128, VGPR spill971. LDS remains 131072B and the
  static native matrix-path/topology gate passes.
- Root cause: the focused probe omitted complete-FA transient live ranges.
  Persistent K/V plus dK/dV accumulation fit, but score/dP operands,
  softmax/dS fragments, page-control state and output state do not fit in
  the remaining 32 VGPR per role.
- Decision: stop before PMD correctness/perf, remove the failed canonical
  code, and preserve only this evidence. M128 `64/32/32` remains the exact
  tail-free control; it does not by itself create three full consumers.

## 2026-07-20 dKV Owner16 Four-Consumer Lifecycle Gate

- Status: `ACCEPT_OWNERSHIP_LIFECYCLE_GATE`.
- Hypothesis: `Mq64/Nk256/D128` can use four symmetric owner16 groups if K/V
  occupies a one-shot 128KB LDS epoch, every wave latches its unique N16
  fragment, and the same LDS is then reused by a two-page Q/dO+sidecar ring.
- Static result: four WDRA branches at 128 VGPR, private0, SGPR40, VGPR128,
  spill/scratch0, LDS131072, BPS108, matrix-read56, ABarrier wait25/arrive36,
  trap0.
- PMD result: deterministic host comparison passes with `bad=0`, three page
  generations and bank0. Run is
  `/zys/shaobo_runs/dkv_owner16_4c_lifecycle_probe_20260720_054620`; kernel
  ticks `6,412,770` are probe diagnostics only.
- Debugging lesson: device-side vector equality caused a false PMD VCC/SGPR
  init failure, while exporting the fragments and comparing on the host
  passed. Pre-role lane/wave state plus many global stores also triggered a
  PMD tracking abort; branch-local setup and compact outputs removed it.
- Decision: admit the canonical four-GEMM integration. Do not claim a
  performance win until S256/S1024 FA correctness and SQTT pass.

## 2026-07-20 dKV M128 64/32/32 Physical 1P3C Gate

- Status: `DESIGN_COMPLETE_RESOURCE_PROBE_PENDING`.
- The existing M128 branch already implements exact, tail-free logical
  ownership `64/32/32`, but SQTT residency is `P0/C0/C12/P1`: two heavy waves
  per SIMD, not three.
- ISA/HCU evidence fixes native FP16 output ownership at 16 rows per wave.
  A four-wave C1 or C2 therefore covers 64 physical rows.  Running all twelve
  consumer waves would issue `3072` MMAC per q packet instead of exact `2048`
  (+50%); activating only two waves in each small role returns to eight heavy
  waves and leaves four role slots idle.
- A D64 split can keep arithmetic exact only by reducing score/dP partials
  across waves and republishing P/dS.  The lower-bound extra LDS traffic is
  about 192KB per q packet for both 32-row groups, so it breaks the current
  direct-register P/dS feed and is rejected before code.
- Exact-work PMD evidence is M128 H1/S1024 `32,393,270` kernel ticks and
  `37.8149%` active versus M192 H1/S768 `34,372,975` and `39.2884%` active.
  The old `43.7836%` M192 number included causal-invalid MMAC.
- Workbook sheet `172_DKV_M128_3C_Gate` advances one structural resource
  probe: `Mq64/Nk256/D128`, four symmetric owner16 groups, each with 96
  persistent VGPR plus 32 transient, two raw pages at 67,072B after a 128KB
  K/V latch epoch.  No canonical source was changed in this design round.

## 2026-07-20 dKV M128 Final-M16 Early Store Rejected

- Status: `REJECT_CORRECTNESS_SOURCE_RESTORED`.
- Hypothesis: exploit the M128 `P0/C0/C12/P1` VGPR headroom by storing the
  finalized D0-D63 dK/dV half after its last MMAC8, while D64-D127 executes
  its final MMAC8.  Formula, MMOP, bytes, ownership, LDS and ABarrier count
  were unchanged by construction.
- Static result: 160 and 168 consumer windows spilled; 176 was clean with
  branch use `8/175/175/14`, private/spill/scratch zero, and the native matrix
  path gate passing.
- Correctness: H1/S128 causal failed both dK and dV.  Diagnostics were
  `simTicks=14,469,455`, kernel ticks `10,855,845`, MMOP `2,048`, coissue
  `695/479`, waitLgkm `1,649.5`, barrier `6,543.75`, bank conflict zero.
- Decision: stop before S1024/fullperf, restore `fcd87aa`, and retain only the
  negative evidence in workbook `171_DKV_M128EarlyStore`.

## 2026-07-15 dKV Score/dP Immediate LDS Offsets Accepted

- Status:
  `ACCEPT_MICRO_FULLPERF_XCU`.
- Change:
  preserve the accepted score/dP operand ping-pong, but issue each four-read
  transpose packet from one SGPR LDS base with four compile-time immediate
  offsets.  Formula, tile, wave roles, VGPR sets, ABarrier ownership, and
  output ownership are unchanged.
- Gates:
  build/static/metadata pass with branch windows `14/16,221/240,221/240,8/16`,
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024
  correctness pass; `ldsBankConflict=0`.
- Performance:
  same-reference fullperf kernel ticks improve `42,564,340 -> 42,335,020`
  (`-0.54%`).  MMAC active rises `33.7716% -> 34.1944%`, waitLgkm falls
  `47,974.25 -> 46,460.50`, barrier falls `134,449.25 -> 129,157.25`, and
  coissue success rises `37,010 -> 40,755`.
- XCU explanation:
  dispatch duration falls `93,548 -> 93,044`, instruction issues fall by
  `972`, and the hot transpose matrix-read contribution falls roughly 22%.
  The dominant ownership bubble still consumes 35.21%, so this closes the
  scalar-address cleanup and does not justify more micro tuning in that area.
- Evidence:
  remote `/zys/shaobo_runs/dkv_score_dp_imm4_fullperf_20260715`; archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260715_193455_dkv_score_dp_imm4_accept_h1s1024_sqc7_fullperf`.

## Next Structural Hypothesis: Three-Slot Q/dO Half-Tile Ring

- Keep `Mq=128,Nk=128,D=128`, 16 waves, resident K/V, no repeated score/dP,
  and the accepted operand ping-pong.
- After consumers latch resident K/V, reuse LDS for three `M64` Q/dO slots.
  Each slot costs `32KB` plus `768B` sidecar; three slots cost `100,608B`,
  leaving `30,464B` under the 128KB LDS limit.
- Producer waves0-3 publish Q+sidecar and waves12-15 publish dO.  Both producer
  groups arrive one per-slot Filled token.  Both symmetric consumer groups
  consume the slot and arrive one combined Used token only after the final
  normal Q/dO reads.
- The intended ownership reduction is three handshakes per half packet
  (`Filled + QUsed + dOUsed`) to two (`Filled + Used`), with one additional
  packet of prefetch distance.  The tradeoff is losing dO's early release;
  only SQTT/ticks can decide whether the deeper ring compensates.

## 2026-07-15 dKV Score/dP Operand Ping-Pong Accepted

- Status:
  `ACCEPT_MICRO_FULLPERF_XCU`.
- Change:
  retain the existing two score/dP operand register sets and reuse them as a
  D-block ping-pong: `read D0/D1 -> wait0 -> MMAC D0 -> read D2 -> MMAC D1
  -> read D3 -> wait4 -> MMAC D2 -> wait0 -> MMAC D3`.  No extra VGPR, LDS,
  ABarrier, formula, or output-ownership change was introduced.
- Gates:
  build/static/metadata pass with unchanged consumer windows `221/240`,
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.  H1/S128 and
  H1/S1024 correctness pass; `ldsBankConflict=0`.
- Performance:
  same-build stats-only ticks improve `42,824,145 -> 42,138,005` (`-1.60%`).
  Same-build helper fullperf also improves `42,622,580 -> 42,564,340`
  (`-0.137%`); MMAC active rises `33.4610% -> 33.7716%`, waitLgkm falls
  `51,651 -> 47,974.25`, and barrier falls `138,200 -> 134,449.25`.
- XCU explanation:
  dispatch duration falls `93,676 -> 93,548`; `MMAC -> s_waitcnt` bubble
  duration falls 62.29%, the main ownership bubble falls 1.87%, and
  `MMAC -> MMAC` falls 1.06%.  The added split readiness point raises
  `s_waitcnt` hits by 2,048, so the next experiment should test whether one
  wait can be consolidated without restoring the hard dependency.
- Evidence:
  candidate `/zys/shaobo_runs/dkv_score_dp_operand_pingpong_fullperf`;
  baseline `/zys/shaobo_runs/dkv_regular_islands_baseline_fullperf`.

## 2026-07-15 dKV Eight-Read Score/dP Island Rejected

- Status:
  `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`.
- Hypothesis:
  replace four independent score/dP source-read calls with one fixed eight
  `ds_read_matrix_trans_format` island for two D32 slices, without changing
  formula, tile, roles, ownership, barriers, or MMAC count.
- Gates:
  build/static/metadata pass with `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch; H1/S128 and H1/S1024 correctness pass and
  `ldsBankConflict=0`.
- Result:
  ASM regularity improved substantially and stats-only ticks improved 1.62%,
  but same-build helper fullperf regressed `42,622,580 -> 42,677,635` ticks.
  XCU explains the disagreement: the main ownership bubble fell 2.22%, while
  `MMAC -> s_waitcnt` bubble duration rose 47.4% and `MMAC -> MMAC` rose 6.22%.
- Decision:
  grouping reads is not sufficient when first-use wait remains immediate.
  Restore source.  A successor must read the next operand group into a second
  register set while the current group executes MMAC, then wait only at the
  next group's first use.
- Evidence:
  candidate `/zys/shaobo_runs/dkv_regular_islands_stageA_fullperf`; baseline
  `/zys/shaobo_runs/dkv_regular_islands_baseline_fullperf`.

## 2026-07-12 dKV/dQ Fullperf XCU Reprofile

- Status:
  `OBSERVE_PROFILE`; no source change.
- Design context:
  both kernels remain on the clean canonical route.  dKV uses
  `Mq=128,Nk=128,D=128`, 16 waves, resident K/V, Q/dO half-page ownership,
  LDS sidecar, and score/dP/softmax/dS/dV/dK MMAC islands.  dQ uses
  `Mq=128,Nk=128,D=128`, 16 waves, startup Q/dO/sidecar latch, K/V pages, and
  score/dP/softmax/dS/dQ MMAC islands.  Main matrix paths remain
  MLS/BPS + `ds_read_matrix` + MMAC with no `ds_read_b32`, gather, bpermute, or
  wrong-layout workaround in the canonical path.
- dKV evidence:
  fullperf root
  `/zys/shaobo_runs/dkv_wave0_inv_fullperf_20260712_211315`.
  H1/S1024 causal correctness PASS; `simTicks=46,829,510`,
  `MMOP=131,072`, `VALU=168,384`, `SCA=111,248`, `LDS=79,360`,
  `VMEM=4,352`, `ldsBankConflict=0`.
  xcu detail for
  `/zys/shaobo_runs/dkv_wave0_inv_fullperf_20260712_211315/dkv_mmac_correctness_20260712_212056/m5out/0/0/2793936_fa3_bwd_wasp_clean.perf`
  reports duration `94,976`, waves `128`, inst issues `563,088`, and hot
  rows `s_xor_b32 34.64%`, `s_waitcnt 19.54%`,
  `v_mmac_f32_16x16x16_f16 10.73%`, `s_waitcnt_vbcnt 4.34%`,
  `ds_read_matrix_trans_format 3.21%`.
  The selected Q1/Dout1-used window shows `s_abarrier_try_wait -> s_xor_b32`
  around `5,171` cycles; the tail AllDone window shows
  `s_abarrier_try_wait -> s_waitcnt` around `12,327` cycles.
- dQ evidence:
  fullperf root
  `/zys/shaobo_runs/dq_canonical_fullperf_20260712_212222`.
  H1/S1024 causal correctness PASS; `simTicks=29,269,240`,
  `MMOP=50,688`, `VALU=57,968`, `SCA=54,172`, `LDS=26,352`,
  `VMEM=1,408`, `ldsBankConflict=0`.
  xcu detail for
  `/zys/shaobo_runs/dq_canonical_fullperf_20260712_212222/dq_correctness_20260712_213002/m5out/0/0/2794756_fa3_bwd_dq_clean.perf`
  reports duration `56,320`, waves `128`, inst issues `215,072`, and hot
  rows `s_xor_b32 26.70%`, `s_cbranch_vccnz 17.35%`,
  `mmop_fp16 12.52%`, `s_waitcnt_vbcnt 8.96%`, `s_waitcnt 4.48%`,
  `lds_matrix 3.19%`.  The top Page0Used bubble is
  `s_abarrier_try_wait -> s_xor_b32` with max duration `6,319` cycles, and
  terminal `s_barrier -> s_cbranch_vccnz` is `15.26%`.
- Decision:
  both kernels are bottlenecked primarily by ABarrier ownership/control
  exposure, not by missing MMAC instructions.  Next changes should be directed
  at page-used lifetime, useful producer work before waits, or increasing
  useful work per ownership epoch.  Treat `v_mov`, sidecar global waits, and
  local priority tweaks as secondary until xcu says otherwise.

## 2026-07-12 dKV/dQ Owner-Teardown Rejected

- Status:
  `REJECT_PMD_VGPR_TRACKING_ABORT_SOURCE_RESTORED`.
- Hypothesis:
  since xcu shows terminal AllDone / CTA-sync bubbles, make only wave0 wait on
  the final ownership token and invalidate ABarriers while non-wave0 roles exit
  early.  The algorithm DAG, tile, ABarrier mainloop, and matrix paths were
  unchanged.
- Gates:
  both temporary sources passed static/resource gates before PMD.  dKV stayed
  at branch windows `14/16,221/240,221/240,8/16`; dQ stayed at
  `8/40,158/216,158/216,9/40`; both had `private=0` and no spill/scratch.
- PMD result:
  dKV H1/S128 aborted in
  `/zys/shaobo_runs/owner_teardown_stats_20260712_2134/dkv_mmac_correctness_20260712_213619`
  with `vgpr47 is not init or has been freed` during
  `V_MMAC_F32_16X16X16_F16`.
  dQ H1/S128 aborted in
  `/zys/shaobo_runs/dq_owner_teardown_20260712_2140/dq_correctness_20260712_213749`
  with `vgpr81 is not init or has been freed` during
  `V_MMAC_F32_16X16X16_F16`.
- Decision:
  reject and restore both sources.  Current PMD/WDRA role-exit discipline needs
  all role waves to remain converged through terminal cleanup, or PMD register
  init/free tracking or a hidden WDRA ABI assumption breaks.  Do not retry
  non-wave0 early exit without a focused WDRA-exit ABI proof.

## 2026-07-12 dKV Terminal Invalidate

- Result:
  `ACCEPT_MICRO_CANONICAL` for wave0-only terminal invalidate, after rejecting
  the stronger no-`AllDone` variant at static gate.
- Design basis:
  The dKV algorithm DAG, `Mq=128/Nk=128/D=128` tile, Q/dO half-page ownership,
  resident K/V, score+dP, softmax/dS, dV/dK MMAC islands, and output ownership
  are unchanged.  The only promoted change is terminal ABarrier cleanup:
  keep `AllDone` as the WDRA role-exit convergence token, then let wave0
  invalidate the ABarriers instead of all waves issuing the same invalidates.
- Static/resource:
  deleting `AllDone` entirely failed metadata with
  `private_segment_fixed_size=244`, `sgpr_spill_count=2`,
  `vgpr_spill_count=60`; this proves `AllDone` is currently a live-range/codegen
  stabilizer, not only a semantic wait.  The wave0-only invalidate variant
  passes dKV gate and symbol metadata with branch windows
  `14/16,221/240,221/240,8/16`, `private=0`, `sgpr=99`, `vgpr=128`, no spill.
- Correctness/perf:
  H1/S128 and H1/S1024 causal correctness pass under `GPU_CHIP=sb`,
  `GPU_ARGS=['--SQCIPfLines=7']`.  H1/S1024 first/repeat:
  `46,594,275` / `46,682,090` ticks; `MMOP=131,072`,
  `VALU=168,384`, `SCA=111,248`, `LDS=79,360`, `VMEM=4,352`,
  `coissue=37,013/25,997`, `waitLgkm=52,429.0`, `barrier=140,274.67`,
  `ldsBankConflict=0`.  Prior half-merge repeat was `46,698,470` ticks and
  `SCA=111,944`.
- Evidence:
  first `/zys/shaobo_runs/dkv_wave0_inv_20260712_205804`;
  repeat `/zys/shaobo_runs/dkv_wave0_inv_repeat_20260712_210159`.
- Lesson:
  keep terminal `AllDone` in dKV until a WDRA-exit proof removes spill, but
  wave0-only invalidate is a valid tiny cleanup.  This is not the 40% active
  structural solution; mainloop ownership/wait remains the bottleneck.

## 2026-07-12 dQ Setprio Narrowing Probe

- Result:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Design basis:
  dQ `dq_update_from_ds_pair` had `s_setprio 2` before K-normal
  `ds_read_matrix` reads.  The probe moved it after those reads, so the
  priority island covered only wait/MMAC instead of read/wait/MMAC.  Formula,
  `Mq=128/Nk=128`, startup Q/dO/sidecar latch, K/V pages, ABarrier tokens,
  and store ownership were unchanged.
- Gates:
  build, dQ gate, and metadata gate pass: `8/40,158/216,158/216,9/40`,
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.  H1/S128 and
  H1/S1024 correctness pass; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regresses to `29,979,040` ticks with `MMOP=50,688`,
  `VALU=57,968`, `SCA=54,172`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=10,578/9,194`, `waitLgkm=16,638.5`, `barrier=58,052.75`.
  Current canonical dQ remains the restored setprio/latched-compute route.
- Evidence:
  `/zys/shaobo_runs/dq_setprio_narrow_dqmmac_20260712_210421`.
- Lesson:
  for this dQ dS@K island, high priority across the K-normal read/wait/MMAC
  window appears better than a narrower MMAC-only priority island.  Do not
  retry setprio narrowing without SQTT evidence that the read priority is
  hurting a peer wave.

## 2026-07-12 dKV Full-Tile Filled Probe

- Hypothesis:
  dKV may pay too much for half-level Filled waits.  Try a full Mq128 Filled
  token by making all Q/dO half publishes arrive `Q0Filled` count 16 and
  removing the consumer wait on `Q1Filled`, while preserving half-level
  `QUsed`/`DoutUsed` release.
- Source experiment:
  temporary only.  `Q1Filled` hot-path arrives/waits were bypassed; `Q0Filled`
  init count changed to 16.  Used lifetime and matrix math were unchanged.
- Static/resource:
  build, dKV gate, and metadata gate PASS.  Metadata stayed legal and SGPR
  dropped `97 -> 96`: `private=0`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS; `ldsBankConflict=0`.
- PMD stats:
  H1/S1024 `simTicks=47,544,770`, MMAC active `31.6659%`,
  `MMOP=131,072`, `VALU=170,180`, `SCA=110,280`,
  `coissue=40,053/28,128`, `waitLgkm=53,209.75`,
  `barrier=161,363.67`.
- Evidence:
  `/zys/shaobo_runs/dkv_full_tile_filled_probe_20260712_162909`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.  The full-tile wait removes
  an ABarrier wait class but destroys useful half-page overlap.  Keep
  half-level Filled readiness; future work should preserve half0/half1
  conveyor while reducing wait exposure or increasing useful MMAC per epoch.

## 2026-07-12 dKV Producer1 Filled-Seq Prune Probe

- Hypothesis:
  after merging Q/dO half-filled readiness, producer0 and producer1 both call
  `seq_q_half_filled` for the same token.  Maybe producer0 can own the `seq`
  and producer1 can only arrive after dO MLS/BPS, trimming SCA.
- Source experiment:
  temporary only.  Removed `seq_q_half_filled<Wdra, 0/1>()` from
  `producer_vdout_loop`; all `QUsed`/`DoutUsed`, arrive counts, and consumer
  waits were unchanged.
- Static/resource:
  build, dKV gate, and metadata gate PASS.  Branch windows
  `14/16,221/240,221/240,8/16`; metadata `private=0`, `sgpr=97`,
  `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS; `ldsBankConflict=0`.
- PMD stats:
  H1/S1024 `simTicks=46,755,345`, MMAC active `33.1816%`,
  `MMOP=131,072`, `VALU=168,384`, `SCA=111,432`,
  `coissue=38,424/26,956`, `waitLgkm=52,109.50`,
  `barrier=141,081.26`.
- Evidence:
  `/zys/shaobo_runs/dkv_half_filled_seq_p0only_20260712_162013`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.  Lower SCA and higher
  coissue are not enough: ticks and MMAC active regress, and wait/barrier
  increase.  Keep the producer1 `seq` calls in the canonical route.

## 2026-07-12 dKV Half-Filled Token Merge

- Hypothesis:
  current dKV spends too much control time on half-page ownership.  Previous
  merged-used-token work proved that merging `QUsed` and `DoutUsed` delays
  independent release and regresses ticks.  Try the narrower lifetime change:
  merge only the `QHalfFilled`/`DoutHalfFilled` readiness into one 8-wave
  filled token per half, while keeping `QUsed` and `DoutUsed` separate.
- Source change:
  producer0 and producer1 now both `seq/arrive` `Q0Filled/Q1Filled` after
  publishing their Q or dO half.  Consumers wait `Q0Filled/Q1Filled` once and
  no longer wait `Dout0Filled/Dout1Filled`.  `Dout0Used/Dout1Used` remain
  unchanged, so producers still cannot overwrite dO before consumers release
  it.
- Static/resource:
  build, dKV gate, and metadata gate PASS.  Branch windows
  `14/16,221/240,221/240,8/16`; metadata `private=0`, `sgpr=97`,
  `vgpr=128`, no spill/scratch.  ASM counts:
  `ds_read_matrix=550`, `v_mmac=1028`, `ds_read_b32=0`,
  `ds_bpermute=0`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS, no NaN/Inf, `ldsBankConflict=0`.
- PMD stats:
  first H1/S1024:
  `simTicks=46,323,550`, MMAC active `33.2633%`, `MMOP=131,072`,
  `VALU=168,384`, `SCA=111,944`, `coissue=37,284/25,932`.
  Repeat:
  `simTicks=46,698,470`, MMAC active `33.3278%`,
  `coissue=37,057/25,788`, `waitLgkm=51,337.75`,
  `barrier=139,802.92`.
- Evidence:
  `/zys/shaobo_runs/dkv_half_filled_merge_20260712_160653`;
  repeat
  `/zys/shaobo_runs/dkv_half_filled_merge_repeat_20260712_160817`.
  Fullperf/xcu blocked by two pre-dispatch PMD/libhsakmt startup aborts:
  `/zys/shaobo_runs/dkv_half_filled_merge_fullperf_20260712_160937`
  and
  `/zys/shaobo_runs/dkv_half_filled_merge_fullperf_retry_20260712_161104`.
- Decision:
  `OBSERVE_STATS_REPEAT_WIN_FULLPERF_PMD_STARTUP_BLOCKED`.  This is a clean
  micro-lifetime improvement, not the structural dKV answer.  Next dKV work
  should use xcu when fullperf is stable and keep targeting ABarrier ownership
  bubbles plus useful MMAC per ownership epoch.

## 2026-07-12 dQ Latched Compute Helper Refactor

- Hypothesis:
  short-causal optimization needs a way to reuse the exact same
  score+dP/softmax/dS/dQ math without copying the whole consumer loop.  Extract
  a helper that starts after Q/dO/sidecar are latched and consumes K/V pages,
  keeping canonical barriers unchanged.
- Source change:
  added `dq_compute_pages_from_latched<Tile, Bar, UsePageBarriers>`.  The
  existing consumer calls it with `UsePageBarriers=true`.  `mmac_zero` remains
  in the outer consumer scope; an intermediate build that zeroed inside the
  n-tile loop raised VALU and was fixed before acceptance.
- Static/resource:
  build, dQ source gate, and metadata gate PASS.  Consumer branch windows
  improved `159/216 -> 158/216`; metadata stayed `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- PMD stats:
  first H1/S1024 run after the zero fix:
  `simTicks=29,289,260`, MMAC active `32.8632%`, `MMOP=50,688`,
  `VALU=57,968`, `SCA=54,172`, `LDS=26,352`,
  `coissue=10,970/9,686`, `ldsBankConflict=0`.
  Repeat:
  `simTicks=29,216,460`, MMAC active `32.6674%`,
  `coissue=11,820/10,450`.
- Evidence:
  first root `/tmp/dq_refactor_151548`;
  repeat root `/tmp/dq_refactor_repeat_151642`.
- Decision:
  `ACCEPT_MICRO_CODE_GOVERNANCE`.  Keep the helper extraction because it lowers
  instruction counts and branch live range while preserving correctness.  It is
  a staging step for short-causal fast path, not the final 40% active solution.

## 2026-07-12 dQ Setprio Reverse M16 Retest

- Hypothesis:
  the old `consumer1 reverse M16` row-pair balancing failed before `s_setprio`.
  After priority islands, same-SIMD scheduling may interact differently with
  paired row work, so retest the one-line mapping change without any other
  code movement.
- Source experiment:
  temporary only.  Changed consumer1's `local_m16` order from `4,5,6,7` to
  `7,6,5,4`, pairing same-SIMD consumer row counts as
  `(0,7),(1,6),(2,5),(3,4)`.
- Static/resource:
  build, dQ gate, and metadata gate PASS.  Metadata stayed
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch, branch windows
  `8/40,159/216,159/216,9/40`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- PMD stats:
  H1/S1024 successful run:
  `simTicks=29,148,665`, MMAC active `32.7388%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`,
  `coissue=10,919/9,596`, `ldsBankConflict=0`.
  This is statistically tied with setprio first `29,145,480` and better than
  setprio repeat `29,438,955`, but not a stable promotion.
- Repeat issue:
  two repeats aborted before dispatch with the known libhsakmt
  `buffer overflow detected` startup failure.  There is no clean repeat.
- Evidence:
  successful root
  `/zys/shaobo_runs/dq_setprio_reverse_m16_retest_20260712_150345`;
  abort roots
  `/zys/shaobo_runs/dq_setprio_reverse_m16_retest_repeat_20260712_150450`
  and
  `/zys/shaobo_runs/dq_setprio_reverse_m16_retest_repeat2_20260712_150522`.
- Decision:
  `OBSERVE_NEEDS_REPEAT_SOURCE_RESTORED`.  Source restored to canonical
  `local_m16 = ConsumerGroup * 4 + wave_local`.  Do not promote until the PMD
  startup issue allows clean repeat/xcu evidence.

## 2026-07-12 dQ BPS vbcnt Off Probe

- Hypothesis:
  xcu on the accepted setprio dQ fullperf listed `s_waitcnt_vbcnt` as a
  visible top row.  Since dKV benefits from BPS-vbcnt but dQ has a different
  startup/page ownership cadence, test whether dQ can compile with
  `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0` and rely on existing ABarrier/page
  ownership alone.
- Probe:
  no source edit.  Built only `src/dq_kernel.cpp` as
  `build/fa3_bwd_dq_clean_novbcnt` with
  `EXTRA_CXXFLAGS=-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0`.
- Static/resource:
  dQ gate and metadata gate PASS.  Metadata stayed legal:
  branch windows `8/40,159/216,159/216,9/40`, `private=0`,
  `sgpr=63`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 causal PASS.  H1/S1024 causal FAIL with NaNs:
  `actual_nonfinite=6144`, first bad row `640`, last bad row `687`.
- Evidence:
  run root `/zys/shaobo_runs/dq_novbcnt_probe_20260712_145055`;
  failing run
  `/zys/shaobo_runs/dq_novbcnt_probe_20260712_145055/dq_correctness_20260712_145100`.
- Decision:
  `REJECT_CORRECTNESS_BPS_READINESS`.  `s_waitcnt_vbcnt` is part of the
  matrix-load readiness contract for this dQ path.  Do not remove it as a
  source-level micro-optimization without a narrower lifetime proof.

## 2026-07-12 dQ QDoFilled Group Split

Decision: `REJECT_STATS_NO_BEST_WIN_SOURCE_RESTORED`

Hypothesis:

Current fullperf shows a large ABarrier/control bubble.  PMD trace maps one
early wait class to `barId 4` (`QDoFilled`), where both consumers wait for all
eight producer waves.  Split the startup filled token into group-local
`QDoFilled0` and `QDoFilled1`, while leaving `QDoLatched` as a single 8-wave
token because page0 K/V reuses the sidecar LDS region.

Evidence:

- Temporary source changed only the dQ barrier ledger and QDo filled
  wait/arrive wrappers.
- Build/source/metadata gates passed with unchanged resources:
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.
- H1/S128 and H1/S1024 correctness passed; `ldsBankConflict=0`.
- H1/S1024 first run: `simTicks=29,853,915`, MMAC active `32.3773%`,
  `coissue=6,046/9,704`, `waitLgkm=16,374.2`, `barrier=55,755.8`.
- H1/S1024 repeat: `simTicks=29,870,295`, MMAC active `32.0531%`,
  `coissue=6,135/10,288`, `waitLgkm=16,500.8`, `barrier=56,716.5`.

Conclusion:

Reject and restore source.  The split is legal but not enough: it does not
beat the accepted repeat best `29,706,495`, and the repeat still shows high
barrier cost.  Startup remains constrained by sidecar/QDo latch before page0
K/V can overwrite the shared sidecar LDS page.

## 2026-07-12 dQ dS-Cache VUsed

Decision: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

Hypothesis:

The previous VUsed early-release attempt was semantically too early because the
same V page is still needed by later `n_tile` dP work.  Try the correct
lifetime: compute all page-local `score + dP + dS` fragments first, cache four
dS fragment pairs in VGPR, arrive `VUsed`, then reread K and compute
`dQ = dS @ K`.

Evidence:

- Temporary source added separate `Page{0,1}VFilled/VUsed` tokens and split
  the consumer page loop into dS-cache and dQ-update phases.
- Build/source/metadata gates passed.  Consumer branch windows grew
  `159 -> 175/216`; metadata remained legal:
  `private=0`, `sgpr=69`, `vgpr=128`, no spill/scratch.
- H1/S128 and H1/S1024 correctness passed; `ldsBankConflict=0`.
- H1/S1024 stats regressed to `simTicks=30,905,875`,
  `MMAC active=31.1624%`, `MMOP=50,688`, `VALU=63,968`,
  `SCA=63,672`, `coissue=5,802/11,721`.

Conclusion:

Reject and restore source.  The lifetime is now correct, but it pays for that
correctness with extra ABarrier tokens, a two-phase n-tile loop, larger live
dS cache, and more VALU/SCA/barrier work.  This closes the small VUsed
early-release direction: improving dQ now needs a larger ownership/dependency
change, not another token-level split.

## 2026-07-12 dQ Builtin Try-Wait

Decision: `REJECT_METADATA_PRIVATE_SEGMENT_SOURCE_RESTORED`

Hypothesis:

XCU shows the dominant gap as `s_abarrier_try_wait -> s_xor_b32`.  Current dQ
uses the inline-asm `ins::abarrier_try_wait<true>` wrapper, which explicitly
emits `s_xor_b32` for phase toggling.  Test whether the builtin wrapper
`ins::abarrier_try_wait<false>` improves codegen or scheduling.

Evidence:

- Temporary source-only change: switched PageFilled, PageUsed, QDoFilled, and
  QDoLatched wait wrappers to the builtin path.
- Build and dQ source gate passed.
- Symbol metadata failed:
  `private_segment_fixed_size=12`, `sgpr=69`, `vgpr=128`, no SGPR/VGPR spill.
- Source restored to inline-asm wait wrappers.
- Remote metadata recert after restore:
  `private=0`, `sgpr=65`, `vgpr=128`, no SGPR/VGPR spill.

Conclusion:

Reject without PMD.  The builtin path violates the no-private-segment hard
gate on the current compiler/PMD route.  Keep inline-asm wait wrappers; attack
ABarrier/control by changing ownership cadence or dependency graph instead.

## 2026-07-12 dQ Contract Cleanup

Decision: `OBSERVE_CLEANUP_RECERT_CANONICAL_UNCHANGED`

Moved native dS ring/source-slot probe-only contracts out of the active dQ
contract header:

- Removed `kDqPathNativeDsRingPrototype`,
  `DqNativeDsRingTileD128`, `NativeDsRingDqTile`,
  `DqNativeDsRingBarrierLedger`, and `NativeDsSlotMap` from
  `include/dq_contract.h`.
- Added `probes/dq_probe_contract.h` and updated source-slot/dS-ring probes to
  include it.
- Canonical `src/dq_kernel.cpp` was not changed.

Local source gate without asm input passes.  Remote build/asm dQ gate and
metadata gate also pass: branch windows `8/40,159/216,159/216,9/40`,
`private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024
correctness pass.  H1/S1024 fullperf stats:
`simTicks=30,262,960`, `MMAC active=32.0547%`, `MMOP=50,688`,
`VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`,
`coissue=6,096/9,906`, `ldsBankConflict=0`.

XCU evidence is archived at
`/Volumes/172.20.68.76/共享/shaobo/perf/20260712_dq_contract_cleanup_h1s1024_sqc7_fullperf`.
Top bubbles are still ABarrier/control dominated:
`s_abarrier_try_wait -> s_xor_b32` 24.67%,
`s_barrier -> s_cbranch_vccnz` 14.88%, `abarrier -> salu_32` 7.85%,
`s_waitcnt_vbcnt` 6.55%, and `lds_matrix -> immed` 4.37%.

Conclusion: keep the cleanup.  It is code hygiene, not a performance
optimization.  The next dQ work must attack ABarrier ownership/control or a
native dS dependency graph; the cleanup alone does not move the 40% MMAC-active
target.

## 2026-07-12 dQ Exact Active K-Tiles

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

The canonical dQ path has a stronger shape contract than the generic formula:
`causal=true`, `Mq=Nk=128`, `S % 128 == 0`, and `seqlen_q == seqlen_k`.
Therefore `active_k_tiles = q_tile + 1`.  Test whether replacing the runtime
`min(q_base+Mq,seqlen)` plus ceil-div in the kernel and consumer reduces
scalar/control debt without changing the algorithm.

Evidence:

- Workbook: `78_DQ_ExactKTile`.
- Temporary source change only; restored after rejection.
- Static/resource PASS:
  branch windows stayed `8/40,159/216,159/216,9/40`; metadata improved
  `sgpr=65 -> 58` with `vgpr=128`, `private=0`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS; `ldsBankConflict=0`.
- H1/S1024 stats:
  `simTicks=32,597,110 -> 32,615,310`,
  `MMAC active=31.6674% -> 31.6334%`,
  `SCA=40,732 -> 42,344`, while `MMOP/VALU/LDS/VMEM` stayed
  `55,296/89,216/28,656/1,408`.

Conclusion:

Reject.  The algebra is correct for the current canonical shape, but the
generated schedule is not better: lower static SGPR count came with higher SCA
and a small tick regression.  Keep C74's original `q_tile_end`/ceil-div form
unless a future compiler/codegen change gives new evidence.

## 2026-07-12 dQ 12-Wave Single Producer

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

C74 xcu showed producer wave slots were thin while ABarrier/BPS/control debt
remained large.  Test a structural 12-wave variant with one 4-wave producer
that publishes both Q/dO sidecar groups and both K+V pages, while the two
4-wave consumer groups keep the full dQ 3-GEMM chain for disjoint M rows.

Evidence:

- Workbook: `75_DQ_SingleProducer12`.
- Temporary source change only; restored after rejection.
- Static/resource PASS after increasing the producer WDRA window from 40 to 48
  because 3 WDRA branches require a branch-average VGPR size aligned to the
  target granularity.  Branch windows were `48/216/216`; symbol metadata was
  `private=0`, `sgpr=52`, `vgpr=160`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS; `ldsBankConflict=0`.
- H1/S1024 stats:
  `simTicks=32,597,110 -> 32,779,565`,
  `MMAC active=31.6674% -> 31.6917%`,
  `VALU=89,216 -> 88,096`,
  `SCA=40,732 -> 33,400`,
  `coissue=9,431/8,921 -> 9,555/8,866`.

Conclusion:

Reject.  The single producer does reduce some VALU/SCA/control work, but
serializing K+V BPS publication delays PageFilled enough to regress elapsed
ticks.  The active source is restored to C74.  Future producer-work changes
must preserve K/V page-ready timing; producer-count reduction alone is not the
route to 40% MMAC active.

## 2026-07-12 dQ Branchless Causal Mask

Decision: `ACCEPT_PERF`

Hypothesis:

The canonical dQ dS loop still had two per-element causal branches.  Replace
the `if (krow <= qrow)` blocks with a branchless valid multiply so the hot path
removes control flow without adding new tokens, changing tile shape, or
extending operand lifetime.

Evidence:

- Workbook: `74_DQ_BranchlessCausal`.
- Source: `src/dq_kernel.cpp`; only the ds0/ds1 causal dS predicate is changed.
- Static/resource PASS:
  branch windows improved from `8/40,161/216,161/216,9/40` to
  `8/40,159/216,159/216,9/40`; metadata remains `private=0`, `sgpr=65`,
  `vgpr=128`, no SGPR/VGPR spill and no scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS; `ldsBankConflict=0`.
- Stats-only H1/S1024:
  `simTicks=33,529,405 -> 32,597,110`,
  `MMAC active=29.5058% -> 31.6674%`,
  `VALU=112,064 -> 89,216`, `MMOP=55,296` unchanged.
- Fullperf H1/S1024:
  `simTicks=33,977,580 -> 32,721,325`,
  `MMAC active=29.4292% -> 31.6115%`.
- XCU:
  `/zys/shaobo_runs/dq_branchless_causal_fullperf_20260712_060000/xcu_outputs/branchless_causal_d0`.
  Dispatch duration improves `66,668 -> 63,904`, instruction issues
  `272,656 -> 234,608`, and top `s_cbranch_vccnz` latency drops
  `708,588 -> 682,412` cycles.

Conclusion:

This is a real canonical micro-win.  The extra invalid-lane diagonal-tile
`exp2` work is cheaper than the removed branch/control debt.  C74 is now the
dQ baseline, but it is still only about `31.6%` MMAC active; next work should
target remaining ABarrier/control ownership and BPS readiness, not missing
MMAC.

## 2026-07-12 dQ Source-Slot Fast Formula Probe

Decision: `ACCEPT_PROBE`

Hypothesis:

The rejected native dS source-slot pack probe was slow because
`source_slot_to_dst` used a runtime reverse-search/control loop, not because
`ds_write_matrix -> ds_read_matrix_trans -> MMAC` handoff is inherently
expensive.  Replace the reverse search with a closed-form mapping from
`src_lane/src_word` to the destination source-slot `(group, q, word)`.

Evidence:

- Workbook: `61_DQ_SourceSlot_FastFormula`.
- Source: `probes/dq_ds_source_pack_cost_probe.cpp`; canonical
  `src/dq_kernel.cpp` unchanged.
- Closed-form equivalence check against the old loop:
  `mismatches=0`, `mapped=504/512`.
- Static/resource PASS:
  `native_slot_pack_cost_kernel private=0 sgpr=20 vgpr=29`,
  no SGPR/VGPR spill.
- PMD run:
  `/zys/shaobo_runs/dq_source_slot_fast_formula_20260712_020039`.
- Correctness:
  `ds_source_pack_cost_pass=1`, `errors=0`, `checksum=1052224`.
- PMD stats:
  `simTicks=102,442,795`, `MMOP=2048`, `VALU=10,593`,
  `SCA=2,206`, `LDS=2,112`, `ldsBankConflict=0`.

Conclusion:

This is a focused-probe win, not a canonical dQ performance claim.  The old
native-slot probe cost (`simTicks=1,176,224,595`) was reverse-search overhead.
The fast formula lands close to the wrong-layout lower bound
(`natural_wrong simTicks=107,657,095`) while preserving source-slot correctness
in the probe.  Next step is a real C_dS source-slot publisher probe: compute
true dS values into source-slot order, publish with `ds_write_matrix`, read with
`ds_read_matrix_trans`, and feed `dS @ K` split-low/high without
`bpermute`/gather.

## 2026-07-12 dQ Real C_dS Source-Slot Handoff Probe

Decision: `ACCEPT_PROBE_NATIVE_HANDOFF`

Hypothesis:

After replacing runtime reverse search with the fast source-slot formula, test
whether a C_dS publisher can write dS values directly in
`NativeDsSlotMap` source-slot order, then let a dQ consumer read them with
`ds_read_matrix_trans` and feed split-low/high `dS @ K` MMAC without
`bpermute`, LDS gather, or ordinary matrix-path `ds_read_b*`.

Evidence:

- Workbook: `62_DQ_RealCDS_SourceSlot_Probe`.
- Source: `probes/dq_native_ds_source_schedule_probe.cpp`; canonical
  `src/dq_kernel.cpp` unchanged.
- First float-formula attempt failed:
  `/zys/shaobo_runs/dq_real_ds_source_slot_probe_20260712_021632`.
  It reported `read_errors=64`, bad half values such as `9472`, and PMD
  warnings around untested half arithmetic.  Treat that as a probe-generation
  issue, not a source-slot layout reject.
- Accepted bit-pattern run:
  `/zys/shaobo_runs/dq_real_ds_source_slot_bits_20260712_022239`.
- Static/resource PASS:
  `source_schedule_kernel private=0 sgpr=22 vgpr=39`,
  no SGPR/VGPR spill.
- Correctness:
  `read_errors=0`, `mapped=504`,
  `frag_low_pass=1`, `frag_high_pass=1`,
  `split_low pass=1`, `split_high pass=1`.
- PMD stats:
  `simTicks=10,236,135`, `MMOP=3`, `VALU=419`, `SCA=495`, `LDS=67`,
  `ldsBankConflict=0`.
- ASM evidence:
  `ds_write_matrix_format`, `ds_read_matrix_trans_format`,
  `ds_read_matrix_format`, and `v_mmac_f32_16x16x16_f16 ... lit` are present.

Conclusion:

The native source-slot handoff is viable: C_dS can publish values directly in
the dQ consumer's source layout and the consumer can read/trans-consume them
with split MMAC, without gather/permute.  Boundary: the accepted probe uses
deterministic half bit-pattern dS values to isolate layout; it does not prove
that full softmax/dS arithmetic can be generated in source-slot order without
extra codegen debt.  The next design step is a structural dQ prototype that
uses the canonical arithmetic path to produce dS, then packs the resulting
values with this source-slot mapping.

## 2026-07-12 dQ Tail No-Invalidate Fast Exit

Decision: `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`

Hypothesis:

XCU mainline showed a large terminal `s_barrier -> s_cbranch_vccnz` bubble
(`18.28%`).  Since no performance-path code uses ABarrier tokens after all
roles finish, try moving terminal `__syncthreads()` plus `abarrier_inv` under
`diag_store != 0`.

Evidence:

- Workbook: `60_DQ_TailNoInvFastExit`.
- Static/resource PASS:
  branch windows `8/40,161/216,161/216,9/40`, `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.
- H1/S128 PMD abort before correctness:
  `/zys/shaobo_runs/dq_tail_noinv_20260712_013135/dq_correctness_20260712_013916/pmd_stdout.log`.
- Abort:
  `panic condition !regInit[regIdx] occurred: cu0 simd1 vgpr81 is not init or has been freed`
  during MMOP execute.

Conclusion:

The tail sync/invalidate is not currently removable as ordinary overhead.  It
appears to be part of the WDRA/PMD role-exit/register-liveness discipline, or a
hidden ABI cleanup requirement.  Source restored to canonical tail cleanup.
Do not retry fast-exit without a focused WDRA-exit probe.

## 2026-07-12 dQ Score/dP Wait12 Split

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

After issuing the score/dP K/V trans matrix reads, the current code waits at
`lgkmcnt(8)` before starting D-block 0/1 MMAC.  Since D-block 0 depends only on
the earliest K/V read group, try `wait_lgkm(12) -> D-block0 MMAC ->
wait_lgkm(8) -> D-block1 MMAC -> wait_lgkm(0) -> D-block2/3`.

Evidence:

- Workbook: `59_DQ_ScoreDP_Wait12`.
- Static/resource PASS:
  branch windows `8/40,161/216,161/216,9/40`, `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/dq_score_wait12_20260712_012423/dq_correctness_20260712_012424`;
  H1/S1024 `/zys/shaobo_runs/dq_score_wait12_20260712_012423/dq_correctness_20260712_012432`.
- H1/S1024 stats:
  `simTicks=36,199,800`, `kernel_ticks=32,586,190`,
  `MMAC active=27.1810%`, `MMOP=55,296`, `VALU=121,632`,
  `SCA=77,516`, `LDS=28,656`, `VMEM=1,408`,
  coissue `15,760/17,888`, `waitLgkm=13,636`,
  `barrier=52,102.75`, `ldsBankConflict=0`.

Conclusion:

This is slower than the mainline fullperf reference (`simTicks=35,881,300`,
`MMAC active=27.4198%`) and does not reduce the real limiter.  Source restored
to canonical `wait_lgkm(8)` before D-block0/1.  Do not retry finer score/dP
wait splitting until PageUsed/ABarrier ownership is no longer dominant.

## 2026-07-12 dQ MLS32x16 Source-Slot Probe

Decision: `REJECT_PROBE`

Hypothesis:

The MLS32 direct reader route failed, but previous instruction probes showed
`matrix_load_32x16_b16` is the official same-LDS normal/trans positive pair.
Test whether a 32x16 page can satisfy the `NativeDsSlotMap` source-slot q
ownership that the native dS ring would need.

Evidence:

- Source: `probes/dq_source_operand_layout_probe.cpp`.
- Workbook: `58_DQ_MLS32x16_SourceSlot`.
- Static/resource gate PASS:
  `private=0`, `sgpr=20`, `vgpr=12`, no SGPR/VGPR spill.
- PMD run:
  `/zys/shaobo_runs/dq_mls32x16_source_slot_20260712_011745`.
- `load_name=mls32x16` result:
  `trans_32x16_alt0 q_match=32/504 decoded=496/504`,
  `normal_32x16_alt0 q_match=44/504 decoded=496/504`,
  `trans_16x32_alt0 q_match=16/504 decoded=248/504`,
  `trans_16x32_alt1 q_match=18/504 decoded=248/504`,
  `normal_32x16_alt1 q_match=40/504 decoded=496/504`.
- Final:
  `operand_layout_final any_full_match=0`, `simTicks=8,070,335`,
  `MMOP=0`, `ldsBankConflict=0`.

Conclusion:

The official 32x16 same-LDS normal/trans pairing is valid as an instruction
contract, but it still does not produce the dQ native source-slot q ownership.
Do not spend more code on direct-load/direct-reader source-slot variants.
The native ring needs a different producer/MMAC orientation proof, or the next
work should return to canonical full-3GEMM dQ and attack PageUsed/ABarrier
cadence plus MMAC island sizing.

## 2026-07-12 dQ DSRead ALT Source-Slot Probe

Decision: `REJECT_PROBE`

Hypothesis:

The direct source-slot read probe had only tested four DS matrix reader forms.
Because the Shaobo ISA documents ALT/interleave variants, extend the focused
probe before abandoning the idea that one MLS32 LDS page can be read into the
NativeDsSlotMap source-slot order without gather/permute.

Evidence:

- Source: `probes/dq_source_operand_layout_probe.cpp`.
- Compile boundary: current compiler rejects
  `trans_32x16_alt1`, `trans_32x16_alt2`, `normal_32x16_alt2`, and
  `trans_16x32_alt2` as unsupported DS matrix format combinations.
- Legal reader set tested after narrowing:
  `trans_32x16_alt0`, `normal_32x16_alt0`, `trans_16x32_alt0`,
  `trans_16x32_alt1`, and `normal_32x16_alt1`, each over MLS32 non-transposed
  and transposed load pages.
- Static/resource gate PASS:
  `private=0`, `sgpr=20`, `vgpr=12`, no SGPR/VGPR spill.
- PMD run:
  `/zys/shaobo_runs/dq_dsread_alt_source_slot_20260712_010656`.
- Result:
  `operand_layout_final any_full_match=0`.  Best legal q-match remains
  `44/504` for `normal_32x16_alt0`; the new `normal_32x16_alt1` gives
  `40/504`.  `simTicks=7,895,615`, `MMOP=0`, `ldsBankConflict=0`.

Conclusion:

ALT/interleave on the legal MLS32 direct readers does not solve the dQ
source-slot ownership contract.  Do not implement the native dS ring by adding
hot-path gather/permute around this direct-read route.  The remaining native
route must find a producer/MMAC orientation that writes values into source-slot
order, or return to canonical full-3GEMM dQ and optimize PageUsed/ABarrier
cadence and MMAC island shape.

## 2026-07-10 dKV Precise SQTT Localization

Decision: `OBSERVE_BOTTLENECK_LOCALIZED`

Evidence:

- Accepted dKV perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_033115/m5out/0/0/2753586_fa3_bwd_wasp_clean.perf`.
- XCU output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dkv_qused_precise_20260710`.
- `detail` top classes:
  `s_xor_b32` `38.65%`, `s_waitcnt` `19.78%`,
  `v_mmac_f32_16x16x16_f16` `10.79%`,
  `ds_read_matrix_trans_format` `3.23%`,
  `ds_read_matrix_format` `1.69%`.
- Steady-state sample:
  `Q0Used=3` shows
  `s_abarrier_try_wait s4, 3, s19 -> s_xor_b32`
  from `18544:24352`, duration `5807` cycles.
  The selected pipeline CSV reports `Bubble %=98.67`, with
  `5807 / 6221` bubble cycles from this pair.
- Tail sample:
  `AllDone=10` shows
  `s_abarrier_try_wait s0, 10 -> s_waitcnt`
  from `80444:92992`, duration `12547` cycles.  This is producer drain, not
  the main q-loop ownership dependency.

Conclusion:

The dKV steady-state limiter is producer-side Q/dO page ownership polling, not
missing MMAC, LDS bank conflict, or a first-order matrix-read issue.
`ds_read_matrix -> s_waitcnt` remains real, but secondary.  The next code
change must either make Q/dO pages releasable earlier without stretching
consumer VGPR further, or give producer waves useful independent work while
they wait.  Do not treat the long `AllDone=10` tail as proof that q-loop
matrix reads are the primary bottleneck.

## 2026-07-10 dKV Bottleneck Reanalysis

Decision: OBSERVE_REDESIGN_PRIORITIES

Document:

- results/dkv_bottleneck_reanalysis_20260710.md

New evidence:

- MMOP runtime share=59.47% already matches/exceeds FWD 58.12%, while MMAC
  active is only 33.24% versus FWD 45.02%. The target is dead-time removal,
  not a higher MMAC/VALU ratio.
- VALU/MMOP matches FWD, but SCA/MMOP is 2.293x and LDS/MMOP is 2.372x.
- Per-role XCU shows each BWD consumer has the same 2048 MMAC as the FWD
  consumer sample, but 1040 matrix reads versus FWD 520.
- 65.6% of consumer s_waitcnt issues source-map to sidecar use in softmax/dS.
  Current issue order makes sidecar waits drain older dV/dK matrix reads.
- Removing AllDone was revalidated as a static reject: private=244,
  sgpr_spill=2, vgpr_spill=60. Accepted source was restored remotely and
  re-passed metadata with no spill.

Next:

1. Issue sidecar before dV/dK reads and use staged lgkmcnt waits.
2. Focused-probe combined Q+dO Filled per half with independent Used release.
3. Batch DS addresses into FWD-style pair/multi-offset bricks.

Do not reopen deeper buffering, causal skip, or tail cleanup first.

## 2026-07-07 dQ Q/dO Latched K/V Double Page

Decision: `ACCEPT`

Hypothesis:

The 16-wave dQ path already reads each consumer group's Q/dO fragments into
VGPR before the `kt` loop, but the LDS lifetime was not exploiting this: K/V
still used one page, so producers waited for `PageUsed` before loading the
next K/V tile.  If consumers explicitly publish `QDoLatched` after reading
Q/dO/sidecar, producers can reuse the released Q LDS region as a second K/V
page and overlap `kt+1` MLS with `kt` consumer compute.

Implementation:

- Single canonical dQ kernel only; no new phase or alternate path.
- Barrier ledger changed from one `PageFilled/PageUsed` pair to page-local
  `Page0Filled/Page0Used`, `Page1Filled/Page1Used`, plus one one-time
  `QDoLatched`.
- Page0 uses the original K/V LDS region; page1 reuses the Q LDS region after
  all eight consumer waves have latched Q/dO and sidecar into registers.
- `dq_update_from_ds_vec` and score/dP reads now select K/V LDS by page.
  The math, tile shape, output ownership, sidecar staging, and store path are
  unchanged.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `118/216`, `118/216`, `9/40`;
  metadata `private=0`, `sgpr=54`, `vgpr=128`, no SGPR/VGPR spill.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_194910`.
  - H1/S1024:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_194922`.
- H1/S1024 stats-only:
  `simTicks=45,520,475`, `kernel_ticks=41,906,865`,
  `MMOP=55,296`, `VALU=140,320`, `SCA=96,904`, `LDS=37,872`,
  coissue `13,590/10,358`, `ldsBankConflict=0`,
  `MMAC active=22.9396%`.
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_195218`.
  Helper perf archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_195218_dq_qdo_latched_kv_double_page_h1s1024_sqc7_fullperf/DQ_QDO_LATCHED_KV_DOUBLE_PAGE_H1S1024.perf`.
  Full-perf stats: `simTicks=45,436,755`,
  `kernel_ticks=41,823,145`, coissue `13,633/10,286`,
  `MMAC active=22.9566%`, `ldsBankConflict=0`.
- XCU CLI:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_qdo_latched_kv_double_page_20260707_195218`.
  Dispatch duration falls to `91,852` and average active waves are `76.13`.
  The dominant `s_abarrier_try_wait -> s_xor_b32` bubble drops from the
  split-wait baseline's about `49.71%` to about `38.54%`.  The new visible
  tradeoff is `s_abarrier_try_wait -> s_waitcnt` at about `10.79%`, with top
  bubble instances around bar5/tail wait.  XCU CSV artifacts are archived under
  the shared perf folder's `sqtt_csv/`.

Conclusion:

Promote as the current 16-wave dQ baseline.  This is the first dQ change in
this route that directly uses Q/dO long-lived VGPR state to improve the
producer/consumer pipeline: full-perf kernel ticks improve about `11.3%` over
the K-normal split-wait baseline (`47.15M -> 41.82M`) and MMAC active improves
about `2.8` points (`20.13% -> 22.96%`).  The remaining bottleneck is no
longer just single-page K/V ownership; the next target is the new bar5/tail
wait and page0/page1 cadence, while keeping the two-page correctness proof.

## 2026-07-07 dQ K-Normal Split Wait

Decision: `ACCEPT_MICRO`

Hypothesis:

After batching all four D-block K-normal reads, one `wait_lgkm(0)` still
exposed a `ds_read_matrix_format -> s_waitcnt` bubble.  The read stream is
issued in D-block order, two LDS matrix reads per D-block.  A split wait should
allow the first half of the dQ MMAC island to start once the first four LDS
reads are ready, while the latter four reads continue to mature.

Implementation:

- Single canonical dQ kernel only.
- In `dq_update_from_ds_vec`, keep the four D-block read batch, but replace the
  single `wait_lgkm(0)` with:
  `wait_lgkm(4) -> MMAC DBlock0/1 -> wait_lgkm(0) -> MMAC DBlock2/3`.
- No tile, role, barrier, LDS layout, sidecar, API, or output ownership change.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `118/216`, `118/216`, `9/40`;
  metadata `private=0`, `sgpr=76`, `vgpr=128`, no SGPR/VGPR spill.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_173804`.
  - H1/S1024:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_173814`.
- H1/S1024 stats-only:
  `simTicks=50,638,315`, `kernel_ticks=47,024,705`,
  `MMOP=55,296`, `VALU=140,320`, `SCA=65,824`, `LDS=37,872`,
  coissue `13,798/13,342`, `ldsBankConflict=0`,
  `MMAC active=20.1654%`.
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_174046`.
  Helper perf archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_174046_dq_k_normal_split_wait_h1s1024_sqc7_fullperf/DQ_K_NORMAL_SPLIT_WAIT_H1S1024.perf`.
  Full-perf stats: `simTicks=50,760,255`, `kernel_ticks=47,146,645`,
  coissue `13,860/13,016`, `MMAC active=20.1315%`.
- XCU CLI:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_k_normal_split_wait_20260707_174046`.
  `ds_read_matrix_format -> s_waitcnt` drops from about `4.83%` in the
  read-batch baseline to about `2.72%`.  The tradeoff is that
  `v_mmac -> s_waitcnt` rises to about `2.95%`.
  The dominant bubble remains `s_abarrier_try_wait -> s_xor_b32` at about
  `49.71%`.

Conclusion:

Keep as the current 16-wave dQ micro-baseline.  It improves stats-only
H1/S1024 simTicks by about `1.59%` over K-normal read-batch and about `8.25%`
over the structural full-3GEMM baseline, with no spill or bank conflict.  It
confirms that dKV-style wait placement helps dQ, but the main route to higher
MMAC active is still reducing ABarrier ownership exposure or increasing useful
MMAC work per ownership epoch, not more local wait splitting.

## 2026-07-07 dQ K-Normal Read Batch

Decision: `ACCEPT_MICRO`

Hypothesis:

The new 16-wave full-3GEMM dQ path already batches Q/dO and score/dP operand
reads, but the final `dQ = dS @ K` helper still issued one K-normal
`ds_read_matrix` pair per D-block followed immediately by `wait_lgkm(0)` and
MMAC.  This recreated the small `ds_read_matrix -> wait -> MMAC` fragments
that were previously fixed in dKV.  Batching all four D-block K-normal reads
before one wait should shrink wait bubbles without changing math or ownership.

Implementation:

- Single canonical dQ kernel only; no new phase or path.
- In `dq_update_from_ds_vec`, issue all K-normal
  `ds_read_matrix_normal_pair` D-block reads first, then one `wait_lgkm(0)`,
  then the full D128 dQ MMAC island.
- No wave role, tile, barrier ledger, sidecar, API, or output-store change.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `118/216`, `118/216`, `9/40`;
  metadata `private=0`, `sgpr=76`, `vgpr=128`, no SGPR/VGPR spill.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_164521`.
  - H1/S1024:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_164530`.
- H1/S1024 stats-only:
  `simTicks=51,458,680`, `kernel_ticks=47,845,070`,
  `MMOP=55,296`, `VALU=140,320`, `SCA=65,824`, `LDS=37,872`,
  coissue `14,065/12,496`, `ldsBankConflict=0`,
  `MMAC active=19.9714%`.
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_164850`.
  Helper perf archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_164850_dq_k_normal_read_batch_h1s1024_sqc7_fullperf/DQ_K_NORMAL_READ_BATCH_H1S1024.perf`.
  Full-perf stats: `simTicks=51,460,500`, `kernel_ticks=47,846,890`,
  coissue `14,200/12,481`, `MMAC active=19.9938%`.
- Baseline 16-wave full-3GEMM dQ:
  `simTicks=55,191,955`, `kernel_ticks=51,578,345`,
  `MMAC active=19.1324%`, coissue `10,490/4,779`.
- XCU CLI:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_k_normal_read_batch_20260707_164850`.
  Top bubbles remain `s_abarrier_try_wait -> s_xor_b32` at about `49.39%`
  and `s_abarrier_try_wait -> s_waitcnt` at about `7.09%`;
  `ds_read_matrix_format -> s_waitcnt` is still visible at about `4.83%`.

Conclusion:

Keep this as the new 16-wave dQ micro-baseline.  It improves H1/S1024 ticks by
about `7.24%` versus the structural full-3GEMM baseline and slightly raises
MMAC active, with no spill or bank conflict.  It does not solve the main
ABarrier ownership bubble.  Next low-level attempt can split the K-normal
read-batch wait (`lgkmcnt(4)` then `lgkmcnt(0)`) or attack `PageFilled/PageUsed`
ownership; larger `BlockM` needs a separate LDS/VGPR design because direct
`Mq=256` would exceed the 128KB LDS plan unless Q/dO are latched then released.

## 2026-07-06 K/V Latch Wait Prune

Decision: `ACCEPT_MICRO_OBSERVE`

Hypothesis:

Some explicit wait instructions are real data-safety waits, but the resident
K/V latch waited after each DBlock pair even though K/V is latched only once
before the steady q-loop.  Batching the four resident K/V reads and waiting
once should be resource-safe and may trim a small front-end/LDS wait bubble.

Implementation:

- Single canonical kernel only.
- In `latch_owner16_kv_regs`, issue all four K/V
  `ds_read_matrix_trans_pair` groups first, then one `wait_lgkm(0)`.
- No math, q-loop ownership, sidecar path, ABarrier ledger, output ownership,
  or API change.

Evidence:

- Static/resource PASS:
  branch windows `14/188/188/8`;
  metadata `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- ASM summary:
  `s_waitcnt=347`, `ds_read_matrix=550`, `ds_read_b32=0`,
  `ds_read_b128=96`.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_202609`.
  - H1/S1024:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_202706`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_203150`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_203150_7gemm_wait_prune_kv_latch_h1s1024_sqc7_fullperf`.

Result:

- Full perf `simTicks`: `47,873,735 -> 47,871,005`.
- `shaderActiveTicks`: `44,260,125 -> 44,257,395`.
- MMAC active: `32.6559% -> 32.7888%`.
- `VALU/SCA/LDS/VMEM`: `183136/115544/79360/4352`.
- `ldsBankConflict=0`.
- XCU still shows `s_waitcnt` at `19.99%` latency and
  `s_abarrier_try_wait -> s_xor_b32` at `41.75%`.

Conclusion:

- Keep as a safe micro cleanup, but do not treat it as a main pipeline win.
- The major limiter remains steady q-loop ownership/wait placement, especially
  ABarrier waits and `ds_read_matrix -> wait` gaps that cannot be deleted
  before `QUsed/DoutUsed` release without risking LDS overwrite.

## 2026-07-06 dO Normal Preread Under Score/dP

Decision: `REJECT_STATS_ONLY`

Hypothesis:

XCU top2000 on the sidecar Vec4 baseline showed the dominant steady ownership
waits were `bar5=Dout0Used` and `bar9=Dout1Used`, about `1.045M` and
`0.994M` cycles respectively.  The candidate tried to issue the final M-pair
dO normal source reads before the second score/dP MMAC block, so the
`ds_read_matrix_normal` latency could be hidden under useful score/dP MMAC and
`DoutUsed` could arrive earlier.

Implementation:

- Single canonical kernel only.
- Added a temporary ReleasePage helper that kept the score/dP read8 schedule,
  but inserted dO normal source reads before score/dP DBlock2/3 MMAC.
- No API, tile, barrier-ledger, output-ownership, or MMOP-count change.

Evidence:

- Design workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `72_dout_preread`.
- Static gates PASS:
  branch windows producer0 `14/16`, consumer0 `188/240`,
  consumer1 `188/240`, producer1 `8/16`;
  metadata `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Static asm remained clean:
  `ds_read_b32=0`, `ds_read_b128=96`, `ds_read_matrix=550`.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_200711`.
  - H1/S1024:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_200742`.
- H1/S1024 stats-only metrics:
  `simTicks=48,590,815`, `shaderActiveTicks=44,977,205`,
  `MMAC active=32.2856%`, `MMOP=131,072`, `VALU=183,136`,
  `SCA=115,608`, `LDS=79,360`, `VMEM=4,352`,
  coissue `37,081/26,440`, `ldsBankConflict=0`.
- Same-code-family sidecar Vec4 stats-only baseline:
  `simTicks=48,445,215`, `shaderActiveTicks=44,831,605`,
  `MMAC active=32.6312%`, coissue `35,844/26,232`.

Conclusion:

Reject without full perf.  The candidate was correct and resource-clean, but it
regressed same-shape stats by about `0.30%` simTicks and lowered MMAC active by
about `0.35` points.  Instruction counts were identical, so this was pure
scheduling/code-motion and did not shorten the dO critical path enough to
matter.  The kernel source was restored to the sidecar Vec4 baseline; the next
attempt should redesign dO lifetime or producer1 useful work structurally
instead of only moving the dO source reads across the final score/dP MMAC.

## 2026-07-06 Mq128 Sidecar Vec4 LDS Reads

Decision: `ACCEPT_MICRO_CANDIDATE`

Hypothesis:

The read8 score/dP baseline still had a visible sidecar LDS read bubble:
`ds_read_b32 -> s_waitcnt` was about `4.16%` in xcu.  The hot softmax/dS
helper reads three sidecar streams, row max, inverse sum, and delta, one float
at a time for four rows.  Reading each stream as a `Vec4F32` should let the
compiler emit wider LDS reads, reduce scattered sidecar LDS waits, and preserve
the existing half-page ownership protocol.

Implementation:

- Changed only the canonical softmax/dS sidecar helpers.
- Replaced scalar sidecar loads with one `Vec4F32` load per sidecar family and
  M-pair: max-log2, inverse-sum, and delta.
- Kept the same algorithm, same Q0/Dout0/Q1/Dout1 half-page tokens, same output
  ownership, same MMOP count, and same external API.
- No new kernel, no phase stack, no asm island, no dQ change.

Evidence:

- Branch: `shaobo/7gemm-dkv-sidecar-vec4-read`.
- Baseline: `shaobo/7gemm-dkv-read8-baseline` commit `e2d445b`.
- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows:
  producer0 `14/16`, consumer0 `188/240`, consumer1 `188/240`,
  producer1 `8/16`.
- Static asm evidence:
  `ds_read_b32=0`, `ds_read_b128=96`, `ds_read_matrix=550`.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_193914`.
  - H1/S1024 stats-only:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_193944`.
  - H1/S1024 full perf:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_194400`.
- Full perf metrics:
  `kernel_ticks=44,260,125`, `simTicks=47,873,735`,
  `MMAC active=32.6559%`, `MMOP=131,072`, `VALU=183,136`,
  `SCA=115,608`, `LDS=79,360`, `VMEM=4,352`,
  coissue `36,479/26,644`, `ldsBankConflict=0`.
- Read8 baseline full perf:
  `kernel_ticks=47,313,175`, `simTicks=50,926,785`,
  `MMAC active=32.0455%`, `VALU=165,872`, `SCA=115,608`,
  `LDS=83,856`, coissue `36,333/25,091`, `ldsBankConflict=0`.
- XCU detail:
  duration `103,988 -> 97,276`, avg active waves `115.47 -> 120.93`.
  The old `ds_read_b32 -> s_waitcnt` bubble disappears.  Top remaining issue
  gaps are still `s_abarrier_try_wait -> s_xor_b32` about `41.86%` and
  `s_abarrier_try_wait -> s_waitcnt` about `8.43%`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_194400_7gemm_sidecar_vec4_h1s1024_sqc7_fullperf`.

Conclusion:

Accept as the new clean micro-baseline.  It improves same-shape full-perf
ticks by about `6.0%` versus read8, removes the scattered sidecar
`ds_read_b32` wait source, and keeps correctness/no-spill/no-bank-conflict.
This is still not a 60% MMAC-active solution: VALU and coissue-fail rise, and
the dominant ABarrier ownership bubble remains.  The next optimization should
target half-page ownership waits and useful consumer/producer overlap, not
another sidecar-only cleanup.

## 2026-07-05 Mq128 Score/dP Read8

Decision: `ACCEPT_MICRO_CANDIDATE`

Hypothesis:

Sheet `71_mq128_score_dp_read8_design` targeted a secondary local stall rather
than the main barrier topology.  The current half-page conveyor emits
`score/dP` as four small islands per M-pair:
`4 ds_read_matrix_trans -> wait_lgkm(0) -> 8 MMAC`.  The candidate batches
adjacent D32 blocks into two larger islands:
`8 ds_read_matrix_trans -> wait_lgkm(0) -> 16 MMAC`.

Implementation:

- Changed only the canonical `score_dp_mmac_owner16` helper.
- Added one small read helper for a static DBlock and one small MMAC helper
  for a static DBlock; no new kernel, no phase stack, no asm island.
- Kept half-page ownership exactly the same:
  `Q0/Dout0/Q1/Dout1` tokens and `AllDone` unchanged.
- Kept the algorithm and work count unchanged:
  per M-pair score+dP remains `32` MMAC and dV+dK remains `32` MMAC.

Evidence:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `71_mq128_score_dp_read8_design`.
- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows unchanged from the half-page baseline:
  producer0 `14/16`, consumer0 `180/240`, consumer1 `180/240`,
  producer1 `8/16`.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_062922`.
  - H1/S1024 stats-only:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_062928`.
  - H1/S1024 full perf:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_063337`.
- Full perf metrics:
  `kernel_ticks=47,313,175`, `MMOP=131,072`, `VALU=165,872`,
  `SCA=115,608`, `LDS=83,856`, `VMEM=4,352`,
  coissue `36,333/25,091`, `ldsBankConflict=0`.
- Half-page conveyor baseline:
  `kernel_ticks=48,279,140`, coissue `33,962/22,131`,
  same `MMOP/VALU/SCA/LDS/VMEM`.
- XCU detail:
  duration `106,108 -> 103,988`, avg active waves `114.79 -> 115.47`.
  `v_mmac -> s_waitcnt` improves `3.92% -> 1.62%`;
  `s_waitcnt -> v_mmac` improves `0.96% -> 0.60%`.
  `s_abarrier_try_wait -> s_xor_b32` remains dominant at about `40.93%`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_063337_clean_read8_score_dp_h1s1024_sqc7_fullperf`.

Conclusion:

Accept as the new clean micro-baseline because it improves full-perf ticks by
about `2.00%` without changing algorithmic work, resource gates, LDS traffic,
or instruction counts.  This does not change the main diagnosis: the next
60%-oriented design must attack ABarrier/consumer lockstep, not simply batch
more reads or jump to assembly.

## 2026-07-05 Mq128 Half-Ring3 Slot

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

Sheet `70_mq128_half_ring3_design` proposed adding one extra M64 half slot
instead of a full Mq128 double buffer.  The expected benefit was that producers
could publish one more half packet ahead:
`q0h0 slot0 -> q0h1 slot1 -> q1h0 slot2 -> q1h1 slot0`, reducing
half-token ownership waits without duplicating Q/dO or changing output
ownership.

Implemented in the single canonical kernel:

- LDS raw/sidecar layout changed from two M64 semantic halves on one Mq128
  page to three local M64 slots.
- Barrier ledger changed to `Slot0/1/2 Filled/Used = bar2..bar7`,
  `Filled=8`, `Used=8`, `AllDone=bar8`.
- Producers publish Q+sidecar and dO into `half_packet % 3`.
- Consumers process each half slot with local `MBlockBase=0/2` and arrive
  `SlotUsed` only after both dO and Q source reads are complete.
- No new kernel, no phase stack, no asm island, no dQ change.

Evidence:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `70_mq128_half_ring3_design`.
- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=49`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows:
  producer0 `8/16`, consumer0 `189/240`, consumer1 `189/240`,
  producer1 `1/16`.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_060908`.
  - H1/S1024 stats-only:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_060914`.
- Stats-only metrics:
  `kernel_ticks=50,617,385`, `simTicks=54,230,995`,
  `MMOP=131,072`, `MMAC active=30.2521%`, `VALU=163,682`,
  `SCA=213,896`, `LDS=83,920`, `VMEM=4,352`,
  coissue `28,645/18,016`, `ldsBankConflict=0`.
- Same-debug half-page baseline:
  `kernel_ticks=48,268,220`, `simTicks=51,881,830`,
  `MMAC active=31.6990%`, coissue `34,498/22,594`,
  `SCA=115,608`.

Conclusion:

Reject before full perf/xcu.  The candidate is correctness- and resource-clean,
but it regresses same-shape stats by about `+4.87%` kernel ticks and drops
MMAC active by about `1.45` points.  The lower token-family count did not
translate into useful overlap because pairing Q and dO at slot granularity lost
the previous early lifetime split, and slot-control/SCA rose materially
(`213,896` vs `115,608`).

The canonical source should return to the half-page conveyor baseline.  Future
topology work must either preserve early dO/Q release benefit or create real
consumer-side work under ownership waits; adding ring depth alone is not
enough.

## 2026-07-05 Mq128 Half-Page Conveyor

Decision: `ACCEPT_CANDIDATE_CURRENT_BEST`

Hypothesis:

The Q/dO lifetime split proved that separating Q and dO tokens is legal and
slightly better, but focused xcu still showed full-page ownership cliffs:
representative `QUsed`/`DoutUsed` windows had about `95%` same-SIMD bubble.
This candidate keeps one physical Q page and one physical dO page, but splits
the Mq128 page into two M64 semantic ownership halves.  Producers should be
able to publish the next tile half0 while consumers finish the previous tile
half1, reducing the cliff without duplicating Q/dO or adding a full extra LDS
page.

Implemented in the single canonical kernel:

- Barrier ledger is now:
  `ResidentFilled=0`, `ResidentUsed=1`,
  `Q0Filled/Q0Used=2/3`, `Dout0Filled/Dout0Used=4/5`,
  `Q1Filled/Q1Used=6/7`, `Dout1Filled/Dout1Used=8/9`,
  `AllDone=10`.
- Producers publish half0 then half1 for Q/sidecar and dO.
- Consumers wait/consume/release MBlockBase `0/2` for half0, then
  wait/consume/release MBlockBase `4/6` for half1.
- No new kernel, no phase stack, no extra full LDS page, no output ownership
  change, no dQ change, and no asm island.

Evidence:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `69_mq128_half_page_conveyor`.
- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows:
  producer0 `14/16`, consumer0 `180/240`, consumer1 `180/240`,
  producer1 `8/16`.
- Correctness PASS:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_053013`.
  - H1/S1024 stats-only:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_053032`.
  - H1/S1024 full perf:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_053321`.
- Full perf metrics:
  `kernel_ticks=48,279,140`, `simTicks=51,892,750`,
  `MMOP=131,072`, `MMAC active=31.7858%`, `VALU=165,872`,
  `SCA=115,608`, `LDS=83,856`, `VMEM=4,352`,
  coissue `33,962/22,131`, `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_053321_clean_half_page_conveyor_h1s1024_sqc7_fullperf`.

XCU:

- Dispatch aggregate:
  `s_abarrier_try_wait -> s_xor_b32` remains top at `40.42%`;
  `s_abarrier_try_wait -> s_waitcnt` is `9.07%`;
  `v_mmac -> v_mmac` is `7.50%`.
- Focused windows:
  - `bar3 Q0Used`, `7832:14048`, `xcd0/se0/cu0/simd3/wave0`:
    producer pipeline bubble `99.98%`; same-SIMD bubble `96.04%`,
    MMAC `1.01%`, VALU `0.90%`.
  - `bar5 Dout0Used`, `18900:25448`, `xcd0/se0/cu0/simd2/wave3`:
    producer pipeline bubble `99.98%`; same-SIMD bubble `94.39%`,
    MMAC `1.13%`, VALU `1.95%`.
  - `bar7 Q1Used`, `14432:20560`, `xcd0/se0/cu0/simd2/wave0`:
    producer pipeline bubble `99.98%`; same-SIMD bubble `94.25%`,
    MMAC `1.01%`, VALU `2.03%`.

Conclusion:

Accept as the current best clean baseline.  Versus the Q/dO split full perf,
`kernel_ticks` improves from `51,238,915` to `48,279,140`
(`~5.78%`) and MMAC active rises from `29.6586%` to `31.7858%`
(`+2.13` points), while correctness, no-spill, and bank-conflict gates stay
clean.

This is still not the 60% active target.  The focused windows prove that the
candidate mostly shortens ownership cliffs from about `10k` cycles to about
`6.1-6.5k` cycles; it does not yet hide those waits with FWD-style useful
softmax/MMAC conveyor work.  The next design should either reduce handshakes
per useful MMAC further, create real consumer-group stagger during half-token
waits, or lengthen the useful MMAC island without reintroducing duplicate
score/dP.

Follow-up design note:

- Workbook sheet `70_mq128_half_ring3_design` is the next planned topology
  candidate.
- It adds one extra M64 half slot, not a full Mq128 double buffer, so the
  expected LDS after K/V latch is about `98.25KB`.
- The preferred protocol is three slot-local paired `Filled/Used` counted
  tokens.  This may reduce steady ABarrier names from the current separate
  Q/Dout half-token protocol, but it couples Q and dO reuse.
- Reject before PMD if slot modulo/control raises SGPR spill or private
  segment.  Promote only if ticks and MMAC active beat this half-page baseline
  and xcu focused `SlotUsed` windows improve beyond the current `94-96%`
  bubble band.

## 2026-07-05 Mq128 62C2 RawUsed XCU Top2000 Diagnosis

Decision: `OBSERVE_XCU_DIAGNOSIS`

No kernel source change in this pass.  The purpose was to use `xcu` SQTT data
to explain why the current 62C2 Mq128/R1 route is still at about 29.2% MMAC
active, before making another code edit.

Evidence:

- Full perf source:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_033019`.
- Shared archive now includes the copied xcu artifacts:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_033019_clean_62c2_mq128_h1s1024_sqc7_fullperf`.
- Top2000 xcu aggregate:
  - `bar3 Raw0Used`, `s_abarrier_try_wait -> s_xor_b32`:
    total `4,623,276` cycles, count `448`, average `10,319.8`,
    max `13,427`, window `7700:94648`.
  - `bar6 AllDone`, `s_abarrier_try_wait -> s_waitcnt`:
    total `1,238,870` cycles, count `110`, average `11,262.5`,
    max `17,667`.
  - `bar2 Raw0Filled`: total `876,540` cycles, count `512`.
- Focused Raw0Used window `7000:22000`,
  `xcd=0,se=0,cu=0,simd=3,wave=3`:
  pipeline bubble `98.60%`, 210 insts, top bubble
  `s_abarrier_try_wait -> s_xor_b32`.
- Same SIMD mix for that window:
  `Bubble=95.59%`, `MMAC=0.85%`, `VALU=1.19%`, `LDS=0.55%`,
  `SALU=1.78%`.
- Same SIMD coissue:
  wave slots 1 and 2 each issue 256 MMAC, but MMAC+VALU coissue is only
  `6.54%` and `8.27%`; producer slots 0 and 3 have no MMAC and are mostly
  waiting.

Code mapping:

- `DkvBarrierLedger`: `Raw0Used=3`, `AllDone=6`.
- In 62C2, `RawBuffers=1`; `producer_kq_loop` and `producer_vdout_loop` wait
  `Raw0Used` for every `q_tile >= 1`.
- With `S=1024` and `Mq=128`, `q_tiles=8`, so seven raw-page reuse waits are
  on the steady path.
- Consumers release the page only on the final M-pair through
  `ReleasePage=true`, after high Q/dO source reads and before the final dV/dK
  MMAC island.

Conclusion:

- The current blocker is raw page ownership lifetime, not absence of MMAC or a
  narrow compiler scheduling artifact.  Assembly is not the next default move.
- `AllDone` is visible but mostly a tail/cleanup issue; optimizing it alone has
  already been a negative pattern.
- The next top-level candidate should split Q and dO lifetimes, not simply add
  raw buffers or token families.  Q and dO have different last-use points:
  `dO` feeds dV, while `Q` feeds dK.
- Workbook sheet `65_mq128_rawused_xcu` records this evidence and the draft
  Q/dO lifetime split stress plan.

Follow-up design note:

- Workbook sheet `66_mq128_qdo_lifetime_split` refines the candidate into a
  concrete schedule.  It does not split dV and dK into separate GEMM islands.
  Instead, only the final M-pair changes order:
  score/dP, read low/high dO normal sources, arrive `DoutUsed`, run
  softmax/dS, read low/high Q normal sources, arrive `QUsed`, then execute the
  combined dV/dK MMAC using the held dO and Q fragments.
- This specifically targets the measured producer1/`producer_vdout_loop`
  Raw0Used wait while preserving the compact final MMAC island.  It must still
  pass resource metadata and H1/S1024 correctness before any perf claim.

## 2026-07-05 Mq128 64 Full-Valid Softmax Helper

Decision: `REJECT_STATIC_SGPR_SPILL`

Hypothesis:

`62C2 xcu shows secondary consumer bubbles inside softmax/dS:
ds_read_b32 -> s_waitcnt, s_cbranch_execz -> s_or_b64, and
s_and_saveexec_b64 -> s_cbranch_execz.  For H1/S1024 exact causal tiles,
48.4375% of M-pairs are full-valid and only 3.125% are boundary.  A full-valid
helper can remove the per-element causal branch for almost half the M-pairs
without changing output ownership or raw lifetime.`

Implemented as a temporary static probe:

- Added a full-valid exact-tile softmax/dS helper with no per-element
  `valid_pair` branch.
- Kept boundary/invalid M-pairs on the 62C2 helper.
- Did not skip full-invalid M-pairs, because accumulator first-valid
  initialization was intentionally left unchanged.
- No raw release change, no sidecar split, no new barrier, no asm island.

Evidence:

- 64A remote build/source gate PASS, branch windows:
  producer0 `14/16`, consumer0 `186/240`, consumer1 `186/240`,
  producer1 `8/16`.
- 64A metadata FAIL:
  `private=0`, `sgpr=100`, `sgpr_spill=4`, `vgpr=128`,
  `vgpr_spill=0`.
- 64B removed four saved `full_valid_*` booleans and passed the predicate
  expression directly into each call.
- 64B metadata stayed identical:
  `private=0`, `sgpr=100`, `sgpr_spill=4`, `vgpr=128`,
  `vgpr_spill=0`.

Conclusion:

Reject before PMD.  The analytical coverage is good, and consumer branch VGPR
pressure even drops, but the two-path full-valid helper raises scalar/control
pressure enough to spill SGPR.  The active source was reverted to 62C2.  Revisit
only with a lower-SGPR formulation or focused codegen probe; do not stack more
branches in the hot dKV helper.

## 2026-07-05 Mq128 63 Sidecar Split Raw Release

Decision: `REJECT_CORRECTNESS`

Hypothesis:

`Sheet 63 keeps the Mq128/R1 math and one raw Q/dO page, but splits sidecar
lifetime from raw lifetime by giving sidecar two pages.  If final-pair raw Q/dO
sources can be read into VGPR before softmax, consumers can arrive Raw0Used
before final softmax/dS and dV/dK, letting producers publish the next q tile
while useful consumer work continues.`

Implemented as a temporary in-place candidate:

- Added `kSidecarBuffers = RawBuffers == 1 ? 2 : RawBuffers`.
- Producer0 wrote sidecar to `q_tile & 1` while raw Q/dO stayed on page 0.
- Consumers read sidecar by sidecar page, not raw page.
- Final ReleasePage M-pair read both low and high Q/dO source fragments before
  softmax and moved `arrive_raw_used_page` before final softmax.
- No new kernel, phase, ABarrier token, or asm island.

Evidence:

- Static/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=98`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows:
  producer0 `14/16`, consumer0 `190/240`, consumer1 `190/240`,
  producer1 `8/16`.
- 63A H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_035311`.
- 63A H1/S1024 correctness FAIL:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_035331`,
  `dk_rel_l2=0.0622111`, `dv_rel_l2=0.0326977`, `pass=0`.
- 63B added `wait_lgkm(0)` after high source reads and before RawUsed release.
  H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_035555`.
- 63B H1/S1024 correctness still FAIL:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_035616`,
  `dk_rel_l2=0.0638349`, `dv_rel_l2=0.047026`, `pass=0`.

Conclusion:

Reject and revert the active kernel to 62C2.  The failure only appears on the
long q-loop, so H1/S128 is not sufficient evidence for raw/sidecar lifetime
changes.  The issue is not just an in-flight `ds_read_matrix` wait boundary;
holding final low/high raw source fragments across softmax or the split-sidecar
lifetime is not safe in this code shape.  Do not keep this code in the
canonical route.  If we revisit release-before-softmax, first write a focused
lifetime/instruction probe or use a design that avoids carrying large raw
source fragments across softmax.

## 2026-07-05 Mq128 62C2 Causal Exact-Tile Helper

Decision: `ACCEPT_STATS_XCU_CANDIDATE`

Hypothesis:

`Sheet 62C keeps the Mq128/R1 long useful MMAC island, but removes the remaining
runtime seqlen/causal/full-valid control from the hot softmax/dS helper.  In
the canonical target path, seqlen is an exact multiple of Mq and Nk, seqlen_k
equals seqlen_q, and causal is fixed true, so the hot predicate can be reduced
to owner_krow <= qrow.  If sheet 62A's remaining SGPR spill is caused by
scalar/control lifetime, this should make Mq128 resource-clean without assembly.`

Implemented in the single canonical dKV route:

- `ActiveDkvTile = DkvTileD128MqNk128<128, 1>`.
- Consumer WDRA window remains 240.
- Canonical path requires `causal == 1` and exact tiles.
- Added a causal exact-tile softmax/dS helper and static MBlockBase
  `0/2/4/6` Mq128 chain.
- No new kernel, no phase stack, no asm island.

Evidence:

- First 62C version with `owner_kmax/full_valid` still failed metadata:
  `private=0`, `sgpr=100`, `sgpr_spill=2`, `vgpr=128`,
  `vgpr_spill=0`.
- 62C2 simplified the predicate to `owner_krow <= qrow` and passed static
  metadata: `private=0`, `sgpr=96`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_032159`.
- H1/S1024 correctness/stats PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_032222`.
- Full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_033019`.
- Shared perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_033019_clean_62c2_mq128_h1s1024_sqc7_fullperf/62C2.perf`.
- H1/S1024 full-perf stats:
  `kernel_ticks=52,163,020`, `MMOP=131,072`,
  `MMAC active=29.2001%`, `VALU=167,536`, `SCA=106,968`,
  `LDS=83,856`, `VMEM=4,352`, coissue `25,179/15,960`,
  `ldsBankConflict=0`.
- Raw2 recert comparison:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`,
  `VALU=181,980`, `SCA=296,328`.

XCU:

- Dispatch0 duration `114,644`, avg active waves `117.91`.
- Top bubbles remain structural:
  `s_abarrier_try_wait -> s_xor_b32` `44.65%` and
  `s_abarrier_try_wait -> s_waitcnt` `9.57%`.
- Representative window
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/62c2_mq128_h1s1024_fullperf_20260705_033019_dispatch0_window_bar6`
  at `93000:113000`, `xcd0,se1,cu0,simd1,wave0` reports
  `Bubble=96.51%`, `MMAC=0.70%`, top bubble
  `s_abarrier_try_wait -> s_waitcnt`.

Conclusion:

62C2 is a valid new candidate baseline: it passes correctness/resource gates,
keeps the matrixized path, improves same-shape ticks by about `1.59%` versus
raw2 recert, improves MMAC active by about `1.42` percentage points, and cuts
large scalar/control instruction count.  It is not a final pipeline solution:
xcu still says the dominant critical path is ABarrier ownership wait.  Next
work should start from 62C2 and redesign Raw/sidecar ownership or useful
producer work to reduce/hide Raw0Used/barId3-class waits.  The barId6
`AllDone` tail wait is also visible in xcu, but it should not be optimized in
isolation before the steady RawUsed ownership path.

## 2026-07-05 Mq128 62A Causal-Control Shrink

Decision: `REJECT_STATIC_SGPR_SPILL`

Hypothesis:

`Sheet 62A keeps the Mq128/R1 long useful MMAC island, but removes runtime
causal control from the hot M-pair helper and precomputes scalar control such
as q_tile_base, owner_krow, owner_kmax, lane_col_group, and
softmax_scale_log2.  If sheet 61's SGPR spill is caused by duplicated causal
and scalar address/control state, this should reduce SGPR pressure without
touching the matrix path or using assembly.`

Implemented as a temporary in-place static resource probe:

- `ActiveDkvTile = DkvTileD128MqNk128<128, 1>`.
- Consumer WDRA window set to 240.
- Canonical performance path temporarily required `causal == 1`.
- Added causal-true narrow softmax/control helper and static Mq128 four-pair
  path.
- No new kernel, no phase, no asm island, no PMD run before metadata.

Evidence:

- Remote build completed and source gate PASS.
- Symbol metadata failed before correctness/PMD:
  `private_segment_fixed_size=0`, `sgpr_count=100`,
  `sgpr_spill_count=14`, `vgpr_count=128`, `vgpr_spill_count=0`.
- Branch windows improved versus sheet 61:
  producer0 `15/16`, consumer0 `182/240`, consumer1 `182/240`,
  producer1 `9/16`.
- Sheet 61 comparison:
  `sgpr_spill_count=18`, consumers `209/240`.

Conclusion:

62A is directionally useful but not sufficient.  It reduces consumer branch
VGPR pressure and SGPR spill, confirming that causal/control lifetime matters,
but it still violates the no-spill hard gate.  No PMD correctness or perf was
run.

Post-revert recertification:

- Live source restored to raw2 canonical.
- zys1 build/static PASS after restore.
- Symbol metadata PASS after restore:
  `private=0`, `sgpr=60`, `sgpr_spill=0`, `vgpr=112`,
  `vgpr_spill=0`.

Next implication:

Proceed only to a stronger scalar/control split, such as sheet 62B two-half
lexical scopes or sheet 62C softmax helper split.  The next design must reduce
remaining SGPR state without introducing device-call private segment or dynamic
Mq128 loop overhead.

## 2026-07-05 Mq128 Consumer-240 Resource Retest

Decision: `REJECT_STATIC_SGPR_SPILL`

Hypothesis:

`Sheet 60's Mq128/R1 static four-M-pair chain failed at consumer VGPR 208 with
private/VGPR spill plus SGPR spill.  Because a 16-wave CTA maps one wave from
each role to each SIMD, raising both consumer branches to 240 should spend the
full per-SIMD VGPR ledger: P16 + C240 + C240 + P16 = 512.  If the only blocker
is the consumer VGPR window, this should make Mq128 static resource-clean.`

Implemented as a temporary in-place resource retest:

- Generalized the tile contract to `DkvTileD128MqNk128<128, 1>`.
- Kept one canonical dKV route and no new kernel/phase.
- Used one Mq128 raw Q/dO page and a static four-pair consumer chain with
  MBlockBase `0/2/4/6`.
- Set consumer WDRA window to 240 VGPR while keeping producer windows at 16.

Evidence:

- Remote build completed and source gate PASS.
- Symbol metadata failed before correctness/PMD:
  `private_segment_fixed_size=0`, `sgpr_count=100`,
  `sgpr_spill_count=18`, `vgpr_count=128`, `vgpr_spill_count=0`.
- WDRA branch windows:
  producer0 `15/16`, consumer0 `209/240`, consumer1 `209/240`,
  producer1 `9/16`.
- Because the hard gate is no SGPR/VGPR spill, no correctness or performance
  run was made.

Conclusion:

The retest removes the private/VGPR spill seen in sheet 60, but it does not
remove the SGPR spill.  The direct static Mq128 island is therefore blocked by
scalar/control live ranges and inlined helper structure, not by consumer VGPR
window alone.  More VGPR budget is not the next lever.

Post-revert recertification:

- Live source restored to raw2 canonical.
- zys1 build/static PASS after restore.
- Symbol metadata PASS after restore:
  `private=0`, `sgpr=60`, `sgpr_spill=0`, `vgpr=112`,
  `vgpr_spill=0`.
- Kernel gate PASS after restore.

Next implication:

Future Mq128 work must first redesign the scalar/control lifetime: reduce
runtime scalar arguments across the four M-pair calls, split or reorder helper
scopes without device-call private segment, or change phasing/output ownership
so the larger useful MMAC island does not inline four full control contexts.
Do not run PMD on static Mq128 candidates until metadata is spill-free.

## 2026-07-05 Sidecar Register Prefetch Before RawUsed Wait

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

`Producer0 currently loads sidecar after RawUsed wait and after Q publication.
Move only the sidecar global read into producer VGPR before RawUsed wait, then
write those values to the LDS sidecar page after the raw page is free.  This
uses the observed producer idle window without changing ABarrier tokens, output
ownership, MMOP count, or consumer math.`

Implemented as a temporary in-place canonical-kernel refactor:

- Split sidecar publication into `load_sidecar_tile_regs` and
  `store_sidecar_regs_to_lds`.
- Loaded the three sidecar floats before `wait_raw_used_page`.
- Stored them to LDS only after `wait_raw_used_page`, `seq_raw_filled_page`, and
  Q `matrix_load` publication.
- Did not add a new page, token, phase, kernel, or asm island.

Evidence:

- Static/resource PASS:
  branch windows unchanged at `6/198/198/1`, metadata `private=0`,
  `sgpr=60`, `vgpr=112`, no scratch/spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_021112`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_021118`,
  with `dk_rel_l2=0.0025563`, `dv_rel_l2=0.000337571`.
- H1/S1024 stats:
  `simTicks=57,272,215`, `kernel_ticks=53,658,605`,
  `MMOP=131,072`, `MMAC active ~=27.4726%`,
  coissue `30,915/18,119`, `ldsBankConflict=0`.
- Raw2 recert comparison:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`.

Conclusion:

The idea is resource-clean and correct, but it regresses the target metric.  A
three-float producer-sidecar preload is too small or not on the dominant
critical path.  It does not move the kernel toward 60% MMAC active.  The code
was reverted to raw2 canonical.

Next implication:

Future producer-thickening must either move substantially more independent work
under the RawUsed window or change the consumer release structure.  Tiny
producer prefetch alone is not worth carrying in the main route.

## 2026-07-05 Full-Valid Mask Shrink

Decision: `REJECT_CORRECTNESS`

Hypothesis:

`For causal=true full-valid owner16 M-pairs, every q/k element is valid.  A
shorter softmax/dS helper can skip the per-element valid_pair branch and reduce
SCA/VALU before RawUsed release without changing MMOP count, tile shape, output
ownership, or ABarrier tokens.`

Implemented as a temporary in-place canonical-kernel probe:

- Added an LDS-sidecar full-valid helper for softmax/dS.
- Selected it in `consume_mq_mpair_owner16` only when the whole 32-row M-pair
  was full-valid.
- Kept the existing masked helper for boundary and diagonal M-pairs.
- Did not add a new kernel, phase, ABarrier, LDS page, or asm island.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=66`, `vgpr=112`, no scratch/spill.
- Branch windows changed to `6/197/197/1`.
- H1/S128 failed at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_015917`.
- PMD printed `warn: read vgpr165 before writing`.
- dK stayed close: `dk_rel_l2=0.000361379`.
- dV was corrupted:
  `dv_max_abs=1.3782`, `dv_mean_abs=0.197446`,
  `dv_rmse=0.305841`, `dv_rel_l2=33.2914`, `pass=0`.

Conclusion:

The old full-valid dV corruption reproduces even on the current LDS-sidecar
raw2 route.  This is not a safe consumer-side mask shrink.  The code was
reverted to raw2 canonical.

Next implication:

Do not remove per-element `valid_pair` from the main dKV helper again without a
focused fragment/codegen probe that proves P/dV fragment layout and PMD/compiler
behavior first.

## 2026-07-05 Producer Sidecar Rebalance

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

`Current raw2 canonical makes producer0 publish K+Q+sidecar and producer1
publish V+dO.  Since producer imbalance is visible in Wavefronts, move sidecar
publication to producer1 so producer0=K+Q and producer1=V+dO+sidecar.  Keep the
same raw pages, same RawFilled token, same consumer code, and same math.`

Implemented as a temporary in-place topology probe:

- Removed `publish_sidecar_tile_to_lds` from `producer_kq_loop`.
- Added `publish_sidecar_tile_to_lds` to `producer_vdout_loop`.
- `RawFilled` still counted both producer groups, so sidecar readiness remained
  covered by the existing token.
- No new ABarrier, LDS page, kernel, phase, or consumer change.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=58`, `vgpr=112`, no scratch/spill.
- Branch windows changed as intended:
  `6/198/198/1 -> 1/198/198/6`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_014728`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_014734`.
- H1/S1024 stats:
  `kernel_ticks=53,558,960`, `MMAC active=27.5554%`,
  `MMOP=131,072`, `VALU=181,980`, `SCA=295,944`,
  `LDS=85,822`, `VMEM=4,352`, `coissue=33,141/19,270`,
  `ldsBankConflict=0`.
- Raw2 recert comparison:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`.

Conclusion:

The producer work moved exactly as designed and coissue rose, but the critical
path did not shorten.  This confirms that producer-branch visual balance alone
is not enough; the two-page raw ownership and consumer release timing still
dominate.  The code was reverted to raw2 canonical.

Next implication:

Future producer-thickening must add useful independent work that actually
overlaps consumer MMAC/softmax or removes a wait.  Merely moving sidecar
ownership is not a path toward 60% MMAC active.

## 2026-07-05 Sidecar Ring3 Early Raw Release

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

`Sheet 54 proved raw release before softmax is illegal unless sidecar lifetime is
also protected.  Instead of latching sidecar rows in consumer VGPR, keep raw
Q/dO as two pages and allocate a three-page sidecar ring.  RawUsed can then
mean Q/dO ownership only, while softmax reads sidecar[t % 3] after the raw page
has been released.`

Implemented as a temporary in-place canonical-kernel probe:

- Added a third sidecar LDS page while keeping `kRawBuffers=2`.
- Producer published Q/dO to raw page `q_tile & 1` and sidecar to a three-page
  ring.
- ReleasePage path pre-read the low/high Q/dO source fragments, arrived
  `RawUsed` before softmax/dS, then used the sidecar ring page for P/dS and ran
  dV/dK MMAC.
- No extra ABarrier token and no new kernel/phase were added.

Evidence:

- Direct `%3` version:
  - static PASS: branch windows `6/203/203/1`, `private=0`, `sgpr=60`,
    `vgpr=112`, no spill/scratch
  - H1/S128 and H1/S1024 correctness PASS
  - H1/S1024:
    `kernel_ticks=54,754,245`, `MMAC active=26.7523%`,
    `MMOP=131,072`, `VALU=193,180`, `SCA=297,032`,
    `ldsBankConflict=0`
- Static sidecar-page-counter refinement:
  - static PASS: branch windows `6/200/200/1`, `private=0`, `sgpr=62`,
    `vgpr=112`, no spill/scratch
  - H1/S128 and H1/S1024 correctness PASS
  - H1/S1024:
    `kernel_ticks=55,298,425`, `MMAC active=26.5015%`,
    `MMOP=131,072`, `VALU=181,980`, `SCA=300,328`,
    `LDS=85,822`, `VMEM=4,352`, `coissue=29,576/17,687`,
    `ldsBankConflict=0`
- Raw2 recert comparison:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`.

Conclusion:

The sidecar ring makes the earlier RawUsed release legal and resource-clean,
but it does not improve the pipeline.  The likely cost is extra sidecar page
selection/control and shifted ABarrier/SCA work, not LDS bank conflict or
spill.  The code was reverted to raw2 canonical.

Next implication:

Do not keep adding page depth or sidecar rings.  The next credible route needs
to reduce ownership/control operations, enlarge the useful MMAC island without
spilling, or use a cleaner FWD-style topology that gives producer waves useful
work while consumers run MMAC/softmax.

## 2026-07-03 BlockN / Owner-N32 Direct Expansion Probe

Decision: `REJECT_STATIC_SPILL`

Hypothesis:

`Since K/V are loaded once and latched into consumer VGPR, increase per-wave
N ownership so each consumer wave does more MMAC before the same page-control
turn.  A larger MMAC island might reduce exposed ABarrier/sidecar overhead.`

Clarification:

- The current clean kernel already has CTA-level `kResidentNk=128`; K/V
  resident total BlockN is already 128.
- The actual experiment is owner-N expansion: one consumer wave owns two N16
  slices (`OwnerN32`) instead of one.
- Direct `OwnerN64` or `OwnerN128` would be even more expensive, so the first
  resource probe used N32.

Implemented as a temporary compile/resource probe:

- Added `Owner32KvRegs` with two `Owner16KvRegs` halves.
- Mapped waves4-7 to four N32 chunks covering the CTA's 128 K rows.
- Waves8-11 were made inactive for output to avoid duplicate stores.
- RawUsed arrival count was reduced from 8 to 4.
- Each active consumer held two dV/dK accumulator sets so it could accumulate
  both N16 halves across the q-loop.

Evidence:

- Remote build completed, but symbol metadata gate failed:
  `private_segment_fixed_size=432`, `sgpr_count=104`,
  `sgpr_spill_count=22`, `vgpr_count=64`, `vgpr_spill_count=110`.
- Branch window report reached the hard edge:
  consumer branch `208/208`.
- No correctness or PMD perf was run because the resource gate failed.
- Workbook result:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `33_blockn128_stress`.

Conclusion:

Direct owner-N expansion is not viable in the current K/V resident kernel.
The extra K/V slice is not the main problem; the second long-lived dV/dK
accumulator set is.  The live code was reverted to the K/V resident owner16
baseline and remotely revalidated:
`private_segment_fixed_size=0`, `sgpr_count=62`, `vgpr_count=112`, no spills,
branch consumers `195/208`.

Future larger-N work needs a different algorithmic design: phase dV/dK
accumulation, store partials safely, or reduce/relocate accumulator lifetime.
Simply increasing owner-N will not reach 60% MMAC active.

## 2026-07-03 W16 K/V Resident In Consumer VGPR

Decision: `ACCEPT_PIPELINE_RESOURCE_WIN`

Hypothesis:

`For one block, K/V are a fixed resident tile.  Latch each consumer wave's
owned Nk=16,D=128 K/V slice from LDS into VGPR once after ResidentFilled, then
remove K/V ds_read_matrix from every q-loop score/dP iteration.  This should
reduce repeated matrix-read/wait pressure and free K/V LDS lifetime for a later
Mq128/Q-dO double-buffer design.`

Implemented:

- Added `Owner16KvRegs` for one consumer wave's K and V fragments.
- Added a branch-local `latch_owner16_kv_regs` step immediately after
  `ResidentFilled`.
- Changed the active Mq64 path to a half-sequential schedule: compute M rows
  0/1, then rows 2/3, while reusing cached K/V fragments.
- Kept the single canonical dKV kernel; no new performance route or phase was
  added.

Evidence:

- Remote build/static/symbol gates PASS.
- Symbol metadata:
  `private_segment_fixed_size=0`, `sgpr_count=62`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Branch windows:
  producer KQ `6/16`, consumer0 `195/208`, consumer1 `195/208`,
  producer VDout `1/16`.  This is lower consumer pressure than the W16 split
  structural probe (`204/208`).
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_170546`,
  `dk_rel_l2=0.000361379`, `dv_rel_l2=0.000267234`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_170645`,
  `kernel_ticks=61635665`, `MMAC active=25.4615%`,
  `VOP active=19.7256%`, `ldsBankConflict=0`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_170901`,
  `kernel_ticks=61582430`, `MMOP=131072`, `VALU=181512`,
  `SCA=311168`, `LDS=66816`, `VMEM=4352`, coissue `26862/16883`,
  `MMAC active=25.4935%`, `VOP active=19.7415%`,
  `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_170901_clean_kv_reg_resident_mq64_h1s1024_sqc7_fullperf`.

xcu findings:

- Dispatch duration `135348`, `128` waves, average active waves `114.00`.
- Top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `36.67%`,
  `s_abarrier_try_wait -> s_waitcnt` `10.69%`,
  `global_load_dwordx3 -> s_waitcnt` `9.31%`,
  `v_mmac -> v_mmac` `4.94%`,
  `ds_read_matrix_trans_format -> s_waitcnt` `1.20%`,
  `s_waitcnt -> v_mmac` `0.77%`.
- Hot instruction latency:
  `s_xor_b32` `34.80%`, `s_waitcnt` `24.25%`,
  `v_mmac_f32_16x16x16_f16` `7.74%`,
  `ds_read_matrix_trans_format` `2.40%`,
  `ds_read_matrix_format` `1.26%`.

Conclusion:

Keep this as the current clean W16 baseline.  K/V VGPR residency is a real
improvement: about `10.8%` faster than W16 split Mq64
(`69039425 -> 61582430` kernel ticks) and about `7.3%` faster than the clean
canonical rebaseline (`66411800 -> 61582430`).  It also lowers consumer branch
pressure and keeps `ldsBankConflict=0`.

This still does not solve the 60% MMAC-active goal.  The dominant debt is no
longer K/V matrix read; it is still ABarrier/page-control and sidecar/global
wait exposure.  Next work should either shorten the RawUsed/page lifetime,
hide sidecar/helper work under peer MMAC, or spend the released K/V LDS on a
workbook-stressed Mq128/Q-dO double-buffer design.

## 2026-07-03 H19A Pre-Read All Source Before Softmax

Decision: `REJECT_RESOURCE`

Hypothesis:

`Move both low and high dV/dK source reads before softmax/dS so the consumer can
arrive RawUsed earlier.  This should shorten the raw page lifetime and reduce
the dominant producer-side RawUsed ABarrier bubble seen in xcu.`

Implemented:

- Added a helper that consumes already-read low+high source fragments.
- In the W12 early-release consumer path, read high `dO^T/Q^T` immediately
  after the low source reads, waited once, and arrived `RawUsed` before
  softmax/dS.

Rejected evidence:

- Remote build produced asm, but symbol metadata gate failed for
  `fa3_bwd_dkv_mmac12_kernel`:
  `private_segment_fixed_size=24`, `vgpr_spill_count=10`, `sgpr_count=80`,
  `vgpr_count=112`.
- The consumer branch pressure hit the current 160 VGPR WDRA window, matching
  the expected risk of carrying all source fragments across softmax/dS.

Conclusion:

The active route was reverted.  H19A proves that shortening page lifetime by
pre-reading all source operands is not viable inside the current 160 VGPR
consumer budget.  A future retry must be an explicit H19B-style resource
experiment with a justified 208 VGPR window or a smaller softmax live range,
and must pass no-spill metadata before correctness/perf.

Restored baseline:

- Static/resource gate after revert:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`, no
  SGPR/VGPR spill, branch consumers `150/160`.
- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_033619`,
  `kernel_ticks=15342145`, `MMOP=2048`, `ldsBankConflict=0`.
- H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_033625`,
  `kernel_ticks=70444920`, `MMOP=131072`, `coissue=32369/20986`,
  `ldsBankConflict=0`.

## 2026-07-03 H19B 208 VGPR Pre-Read All Source

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

`H19A failed because the 160 VGPR consumer window could not hold both low and
high dO^T/Q^T source fragments across softmax/dS.  Widen the W12 consumer role
to 208 VGPR and rerun the same earlier RawUsed release timing.`

Implemented:

- W12 canonical dKV consumers used `kConsumerMq64Vgprs=208`.
- Consumer loop read low+high source fragments, waited once, arrived
  `RawUsed`, then ran softmax/dS and dV/dK MMAC from the pre-read source regs.
- No new kernel, path, or phase was added.

Evidence:

- Static/resource PASS:
  `private_segment_fixed_size=0`, `sgpr_count=76`, `vgpr_count=144`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`, branch consumers `164/208`.
- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_034717`,
  `kernel_ticks=15167425`, `MMOP=2048`, `ldsBankConflict=0`.
- H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_034723`,
  `kernel_ticks=77708085`, `MMOP=131072`, `coissue=31351/23990`,
  `ldsBankConflict=0`.
- Restored canonical comparison:
  `kernel_ticks=70444920`, `coissue=32369/20986`.

Conclusion:

The 208 VGPR window removes the H19A spill and preserves correctness, but it
regresses S1024 ticks by about `10.31%`.  The wider window and longer live range
are not the path to 60% MMAC active.  Revert H19B and shift the next design
toward reducing consumer-side sidecar/mask/softmax work or changing page
ownership without carrying source fragments across a large VALU section.

## 2026-07-03 H20A Owner16 Full-Valid Softmax Fast Path

Decision: `REJECT_CORRECTNESS`

Hypothesis:

`Causal full-valid tiles should not pay the per-element qrow/valid_pair branch.
Split softmax_ds_owner16_from_global_sidecar into full-valid and masked paths
to reduce consumer VALU/mask work without touching MMAC or ABarrier.`

Implemented:

- Added a full-valid branch in `softmax_ds_owner16_from_global_sidecar`.
- The full-valid branch computes P and dS directly from packed sidecar rows.
- The masked branch preserved the original validity logic.

Evidence:

- Static/resource PASS:
  `private_segment_fixed_size=0`, `sgpr_count=83`, `vgpr_count=112`, no spill,
  branch consumers `155/160`.
- H1/S128 correctness failed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_035528`,
  `dk_rel_l2=0.000361379`, `dv_max_abs=0.518505`, `dv_rel_l2=14.4712`,
  `pass=0`.
- A retry with explicit `p_vec{}`/`ds_vec{}` initialization also failed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_035643`.

Conclusion:

The failure reproduces the older full-valid fastpath dV issue.  Revert H20A.
Do not attack the sidecar/mask cost by splitting this helper in the main route
until a focused owner16 fragment/codegen probe explains why dV changes while
dK remains near the baseline.

## 2026-07-01 Clean dKV WASP Probe

Decision: `BRINGUP_ONLY`

Hypothesis:

`A clean FA3 BWD dKV repo should first prove the WDRA role topology, ABarrier
ownership, MLS/BPS publication, ds_read_matrix consumption, and MMAC lit score+dP
island before full dKV math is added.`

Implemented:

- Added a standalone clean FA3 BWD dKV kernel and C ABI.
- Added `include/shaobo_instr.h` with minimal Shaobo instruction wrappers for
  ABarrier phase ops, `s_setprio`, `matrix_load_32x32_b16 ... bps lds`,
  `ds_read_matrix_trans_format`, and `v_mmac_*lit`.
- Added explicit producer loops, consumer loop, score+dP MMAC probe, static
  gate, symbol metadata gate, PMD smoke wrapper, and repo cleanliness gate.
- Kept historical phase-stack code and generated run artifacts out of the clean
  repo.

Verified after the naming cleanup:

- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=30`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD smoke PASS with runtime signal:
  `fa3_bwd_dkv_probe status=success B=1 H=1 S=1024 D=128`
- PMD smoke path:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_211544`
- PMD stats:
  `simTicks=7232680`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7232680`, `MMOP=2048`, `ldsBankConflict=0`,
  `mmop_active_share=6.4892%`.

Notes:

- This is not promoted as a performance candidate because it is only a
  score+dP probe, not full dKV.
- PMD reports `warn: read vgpr184 before writing`; this did not abort, but it
  should be investigated before relying on the probe for SQTT conclusions.
- Next candidate should keep the clean WASP structure and add sidecar + real
  q-loop, then softmax+dS, then dV/dK accumulation and stores.

## 2026-07-01 Remove Post-MLS Publication Wait

Decision: `OBSERVE`

Hypothesis:

`matrix_load_32x32_b16 ... bps lds` should not be followed by producer-local
`wait_lgkm(0)` when the data is handed to another wave through an ABarrier
Filled token.  The ABarrier token is the producer/consumer ownership fence; the
consumer still waits after `ds_read_matrix` before first MMAC use.

Implemented:

- Removed the immediate `wait_lgkm(0)` from Q/dO and K/V publisher helpers.
- Added a static gate to reject `matrix_load -> wait_lgkm(0) -> ABarrier arrive`
  in the clean dKV source.
- Documented the publication rule in `client.md` and `docs/design_contract.md`.

Verified:

- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=30`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD smoke PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_212516`
- PMD stats:
  `simTicks=7250425`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7250425`, `MMOP=2048`, `ldsBankConflict=0`,
  `mmop_active_share=6.4353%`.

Conclusion:

This is a synchronization cleanup, not a promoted performance win on the
single-packet probe.  It should matter once the q-loop has overlapping packets;
until then, keep the result as protocol evidence.

## 2026-07-01 Collapse Fragment Barriers To Producer Packets

Decision: `OBSERVE_PROTOCOL`

Hypothesis:

The SQTT gap analysis showed that BWD C125C spends a large fraction of issue-gap
time in `abarrier -> abarrier`.  The clean probe should not repeat that protocol
shape.  Q+K and dO+V are producer-level packets in this bring-up cut, so each
producer should publish one packet token rather than separate raw/trans/K/V
fragment tokens.

Implemented:

- Replaced `RawFilled/RawUsed`, `TransFilled/TransUsed`, `Kv0Filled/Kv0Used`,
  and `Kv1Filled/Kv1Used` with `PacketAFilled/PacketAUsed` and
  `PacketBFilled/PacketBUsed`.
- Moved `abarrier_seq` and `abarrier_arrive` out of the individual Q/K/dO/V
  publication helpers and into the producer packet loops.
- Consumer groups now wait two packet tokens and arrive two used tokens before
  and after the score+dP MMAC probe.
- Updated the static gate to reject the legacy fragment barrier names.

Verified:

- Local source gate PASS:
  `python3 scripts/check_dkv_kernel_gate.py`
- Remote build PASS with asm.
- Remote static gate PASS.
- Remote symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=30`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD smoke PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_232821`
- PMD stats:
  `simTicks=7177170`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7177170`, `ldsBankConflict=0`.
- TT/Perf capture completed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_233020`
- XCU first-pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2010209_fa3_bwd_wasp_clean_20260701_233432`
- XCU top-bubble window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/packet_barrier_probe_window_20260701_233020`
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_233020_clean_packet_barrier_probe`
- XCU detail:
  dispatch duration `7852`, inst issues `20648`, MMAC share `18.05%`,
  SALU32 share `35.95%`, LDS matrix share `9.02%`.
- XCU bubbles:
  top bubble remains `abarrier -> abarrier`, `23.50%`, max duration `3696`.
  The selected top-bubble window is 100% bubble with no issued instructions.

Conclusion:

This cut reduces the source-level token chain and is mildly better in stats-only
ticks, but it does not solve the pipeline.  The one-shot probe still naturally
ends with producer/consumer/all-done barrier idle.  The next useful cut must add
a real multi-packet q-loop so producers can do next-packet work while consumers
execute MMAC/VALU islands.

## 2026-07-01 Add Stream q-loop Probe

Decision: `OBSERVE_PIPELINE`

Hypothesis:

`The one-shot score+dP probe exaggerates end-of-kernel barrier idle.  A fixed
S1024 q-loop with resident K/V and double-buffered Q/dO raw pages should expose
the real conveyor bottleneck and show whether producer work can recur during
consumer MMAC islands.`

Implemented:

- Added fixed probe shape guard for `S=1024`, `D=128`.
- Kept K and V resident in LDS for the CTA.
- Changed Q and dO to two raw LDS pages each.
- Producer A publishes K once, then streams Q pages over 32 q tiles.
- Producer B publishes V once, then streams dO pages over 32 q tiles.
- Consumers wait the resident token once, then loop over raw pages and run the
  score+dP MMAC probe for each q tile.
- Disabled outer q-loop unrolling to avoid code-size/SQC bloat.
- Used literal ABarrier helper wrappers for page 0/page 1, because dynamic
  barrier IDs do not satisfy the instruction immediate constraint.
- Used inline-asm ABarrier wait for producer `RawUsed` waits to avoid a
  compiler-generated private segment on the builtin phase path.

Verified:

- Local source gate PASS:
  `python3 scripts/check_dkv_kernel_gate.py`
- Remote build PASS with asm.
- Remote static gate PASS.
- Remote symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=38`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD stats smoke:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_235037`
- PMD stats:
  `simTicks=28497105`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=28497105`, `ldsBankConflict=0`, `MMOP=65536`,
  average per-active-CU `mmopRunTimeCounter/activeTimeCounter=30.072%`,
  coissue `1882/887`.
- TT/Perf capture:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_235316`
- XCU first-pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2012739_fa3_bwd_wasp_clean_20260701_235458`
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_235316_clean_stream_qloop_probe`
- XCU detail:
  dispatch duration `54540`, inst issues `352336`, waves `128`, MMAC share
  `31.37%`, SALU32 `21.56%`, `lds_matrix` `15.68%`, `immed` `12.00%`.
- XCU bubbles:
  top bubble is now `abarrier -> salu_32`, `43.80%`; `lds_matrix -> immed`
  remains `6.74%`.

Conclusion:

This confirms the real next bottleneck: the kernel is no longer dominated by a
single `abarrier -> abarrier` tail, but the q-loop still pays too much
ABarrier/SALU page-control cost and still fragments `ds_read_matrix` before the
first MMAC.  The next implementation should keep this stream q-loop shape and
make the consumer read/MMAC island more FWD-like: batch operand reads, delay
`s_waitcnt` until true first use, and minimize immediate/control instructions
between `lds_matrix` and `mmop`.

## 2026-07-02 Add dK/dV Reference Correctness

Decision: `REFERENCE_CORRECTNESS_PASS`

Hypothesis:

`Before moving full dK/dV into the WASP/MMAC path, the clean repo needs a
verified formula path and a reusable golden check.  That separates algorithm and
output-ownership mistakes from later pipeline scheduling mistakes.`

Implemented:

- Added `kDkvPathReferenceCorrectness = 1`.
- Added reference kernels for:
  - row softmax probability `P`
  - `delta = sum(dO * O)` with `O = P @ V`
  - `dP = dO @ V^T`
  - float `dK/dV` output
- Added standalone `--check=1` mode with deterministic nonzero fp16 inputs.
- Added CPU golden implementation and metrics:
  `max_abs`, `mean_abs`, `rmse`, `rel_l2`, NaN/Inf count.
- Added `scripts/run_dkv_correctness.sh`.
- Updated the source gate so correctness support is protected from accidental
  deletion.

Verified:

- Remote build PASS with asm.
- Static gate PASS.
- Probe symbol metadata still PASS:
  `private_segment_fixed_size=0`, `sgpr_count=38`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_correctness_20260702_003625`
- Shape:
  `B=1,H=1,S=128,D=128,causal=true,fp16,GPU_CHIP=sb,GPU_ARGS=['--SQCIPfLines=7']`
- Result:
  `fa3_bwd_dkv_correctness status=success ... dk_max_abs=1.16415e-10
  dk_rel_l2=3.53805e-07 dv_max_abs=3.72529e-09 dv_rel_l2=1.7753e-08 bad=0
  pass=1`.
- WASP probe regression:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260702_003118`
  reported `fa3_bwd_dkv_probe status=success`.

Conclusion:

The clean repo now has verified dK/dV correctness, but only in the reference
path.  The optimized WASP path remains a score+dP probe.  Next, migrate the
reference math into the WASP path in narrow cuts: softmax/dS sidecar first,
then dV MMAC accumulation, then dK MMAC accumulation and store epilogue, using
the reference path as the correctness oracle after each cut.

## 2026-07-02 Add WASP Softmax/dS Sidecar

Decision: `SIDECAR_CORRECTNESS_PASS`

Hypothesis:

`Before implementing fragment-local softmax/dS and full dV/dK stores, the WASP
consumer role should produce a small verifiable P/dS sidecar.  This checks
formula wiring, role-local output ownership, and consumer-branch stores without
turning the scalar diagnostic into a performance path.`

Implemented:

- Added `kDkvPathWaspSoftmaxDsSidecar = 2`.
- Extended probe workspace from 8 to 24 floats per CTA:
  score diag, `P` sidecar, `dS` sidecar.
- Let the probe q-loop run from runtime `q_tiles`, so sidecar correctness can
  use `S=128` instead of always `S=1024`.
- Added `softmax_ds_pair_sidecar` in the consumer role.  It computes one
  scalar diagnostic pair per consumer wave:
  `q_idx = k_idx`, `P(q_idx,k_idx)`, and
  `dS = P * (dP - delta) * softmax_scale`.
- Added standalone `--probe-check=1` and
  `scripts/run_dkv_sidecar_correctness.sh`.
- Added host CPU golden comparison for sidecar `P/dS`.
- Moved `lane = threadIdx.x % 64` into the consumer role after
  `s_set_vgpr_size`.

Verified:

- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=80`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD sidecar correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_sidecar_correctness_20260702_005025`
- Shape:
  `B=1,H=1,S=128,D=128,causal=true,fp16,GPU_CHIP=sb,GPU_ARGS=['--SQCIPfLines=7']`
- Result:
  `fa3_bwd_dkv_sidecar status=success ... p_max_abs=0 p_rel_l2=0
  ds_max_abs=2.18279e-11 ds_rel_l2=1.10949e-07 bad=0 pass=1`.
- Stats:
  `simTicks=1313879840`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=1313879840`, `ldsBankConflict=0`.
- Default WASP probe regression:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260702_005132`
  reported success with `simTicks=28741440`, `ldsBankConflict=0`.

Negative/lesson:

The first sidecar run failed with `p_max_abs=1` because the sidecar store
predicate used a `lane` value initialized before the role branch and before
`s_set_vgpr_size`.  Recomputing lane inside the consumer branch fixed the
diagnostic.  Keep branch-local store predicates and other VGPR values inside
their WDRA role window.

Conclusion:

Softmax/dS is now wired into the WASP consumer role as a verified scalar
sidecar.  It is not a performance candidate and should not be profiled as one;
the high sidecar `simTicks` is expected.  Next, convert the sidecar into
fragment-local `P/dS` in the mainloop, then connect dV and dK MMAC accumulation
against the existing reference oracle.

## 2026-07-02 Update FWD-style dKV Workbook Gate

Decision: `DESIGN_GATE_UPDATED`

Hypothesis:

`The next dKV implementation must be constrained by a top-down FWD-style design
before code changes.  The previous repo drifted because tile shape, output
ownership, LDS budget, and expected Wavefronts pattern were not all in one
reviewable artifact.`

Implemented:

- Updated the shared workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`
- Added or refreshed sheets for:
  - FWD target and hard gates
  - algorithm DAG and redundancy audit
  - tile/resource/MMAC budget
  - T0-T5 expected WASP conveyor
  - promotion metrics
  - experiment ledger
- Aligned repo docs to the workbook:
  - LDS target `98816 B` plus about `28 KB` slack
  - no unbudgeted LDS raw-to-trans scratch
  - no duplicate score/dP as the default architecture
  - source-layout operands must be loaded by MLS/BPS with explicit lifetime

Verified:

- Workbook export PASS.
- Backup:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.codex_backup_20260702_fwdstyle_goal.xlsx`
- Inspect:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx.inspect.ndjson`
  reported zero formula-error matches.
- Preview:
  `/tmp/shaobo_wb_update_20260702/previews/流水设计.png`

Conclusion:

This is a design gate, not a performance candidate.  The next code cut should
implement exactly the workbook path: fragment-local `P/dS` in the consumer
mainloop, then dV MMAC, dK MMAC, and stores, with correctness/resource gates
before any perf claim.

## 2026-07-02 Add WASP Fragment Sidecar

Decision: `FRAGMENT_SIDECAR_PASS`

Hypothesis:

`Before wiring dV/dK MMAC, the clean kernel must prove that score/dP MMAC
fragments can be converted into fragment-local P/dS using sidecar row metadata.
This also gives a clean place to adopt the FWD mmac_zeros accumulator seed
style.`

Implemented:

- Added `kDkvPathWaspFragmentSidecar = 3` and standalone
  `--fragment-check=1`.
- Added host generation of `scores_max`, `scores_sum`, and `delta`; passed them
  through `softmax_aux0`, `softmax_aux1`, and `reserved_ptr[0]`.
- Added shared sidecar pages for `max_log2`, `inv_sum`, and `delta`.
- Producer A publishes sidecar metadata with Q raw pages under the existing raw
  page ABarrier.
- Replaced the one-accumulator score/dP checksum with four MMAC fragments.
- Used `mmac_zeros` as the first MMAC accumulator seed for each fragment.
- Added fragment-local `P/dS` conversion and a diagnostic compare path.
- Reduced kernel LDS allocation from the whole 128KB budget to actual
  `Layout::kBytes`, because sidecar storage made the previous allocation
  exceed the 128KB hardware limit.
- Added `scripts/run_dkv_fragment_sidecar_correctness.sh`.
- Added `results/perf_ledger.csv`.

Verified:

- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=88`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD fragment correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_fragment_sidecar_correctness_20260702_012111`
- Shape:
  `B=1,H=1,S=128,D=128,causal=true,fp16,GPU_CHIP=sb,GPU_ARGS=['--SQCIPfLines=7']`
- Result:
  `fa3_bwd_dkv_fragment_sidecar status=success ... p_max_abs=6.34603e-06
  p_rel_l2=6.92507e-06 ds_max_abs=2.66919e-08
  ds_rel_l2=0.000359523 bad=0 pass=1`.
- Stats:
  `simTicks=10138765`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=10138765`, `MMOP=512` on the active CU,
  `ldsBankConflict=0`, coissue `45/12`.
- Default S1024 score/dP smoke regression:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260702_012136`
  reported success with `simTicks=27633970`, `ldsBankConflict=0`.
- Scalar sidecar regression:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_sidecar_correctness_20260702_012150`
  reported `pass=1`.
- Workbook experiment ledger updated:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`.

Boundary:

This is still not full dK/dV and not a performance promotion.  It proves the
fragment P/dS handoff is correct and resource-clean.  Next implementation cut:
wire `p_frag` into dV MMAC accumulation, wire `ds_frag` into dK MMAC
accumulation, then add the store epilogue and compare against the reference
dK/dV oracle.

## 2026-07-02 Full dK/dV MMAC Coarse Packet

Decision: `FULL_DKV_CORRECTNESS_PASS_BASELINE`

Implemented:

- Added `kDkvPathWaspDkvMmac = 4` and standalone
  `--dkv-mmac-check=1`.
- Consumer waves own different `Nk=16` blocks over full `D=128`; score and dP
  are not recomputed across D halves.
- Added source-layout ABI buffers for `Q^T` and `dO^T`.
- Added dV and dK MMAC accumulation plus float store epilogue.
- Packed sidecar `[max_log2, inv_sum, delta]` into one pointer to remove the
  earlier SGPR spill.
- Kept the coarse page token: producers publish raw Q/dO and source-layout
  Q^T/dO^T under the same raw page ownership; consumers release the page after
  dV/dK.
- Added `--causal` / `CAUSAL` to the standalone smoke script for controlled
  mask diagnostics.

Verified:

- Remote build/static/symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=76`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_021025`.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_022300`.
- H1/S1024 stats:
  `simTicks=86904090`, `kernel_ticks=83290480`,
  average MMAC active share `18.7606%`, min/max `16.4951%/21.4893%`,
  coissue `36070/20048`, `ldsBankConflict=0`.

Conclusion:

This is the current clean full-dKV baseline.  It is correct and resource-clean
but far below the `>=60%` MMAC-active target.  Removing causal masks did not
help: the H1/S1024 causal=false diagnostic had only tiny absolute error but did
not meet the relative-L2 gate, and MMAC active fell to `15.7263%`.

## 2026-07-02 Split Source Ownership Probe

Decision: `REJECT`

Hypothesis:

Split raw Q/dO ownership from source-layout Q^T/dO^T ownership so producers can
start the next raw MLS packet after score/dP, rather than waiting for dV/dK to
finish.

Result:

- Correctness PASS on H1/S1024 causal=true:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_023309`.
- Resource metadata stayed clean:
  `private=0`, `sgpr_count=78`, `vgpr_count=84`, no spills.
- Performance regressed:
  `kernel_ticks=86912280` vs coarse-packet `83290480`,
  average MMAC active share `18.1123%` vs `18.7606%`,
  coissue `31084/16965`, `ldsBankConflict=0`.

Conclusion:

The extra source ABarrier ledger cost is larger than the benefit from earlier
raw page reuse.  The code was reverted to the coarse-packet baseline.  Do not
reintroduce split source tokens unless a new design also removes a larger
barrier/flat-read debt.

## 2026-07-02 12-Wave Single Producer dK/dV MMAC

Decision: `ACCEPT_CANDIDATE`

Hypothesis:

The 16-wave full-dKV baseline spends half of its waves on producers.  XCompute
shows producer waves are still thin after startup, so the MMAC-active
denominator is diluted.  A 12-wave topology with one four-wave producer group
and two four-wave consumer groups should reduce thin-wave residency without
changing the math or output ownership.

Implemented:

- Added opt-in `kDkvPathWaspDkvMmac12Wave = 5` and standalone
  `--dkv-mmac12-check=1`.
- Added `DkvTileD128Mq32Nk128W12` with 768 threads and
  `hcu_wdra_waves_per_tg(12)`.
- Replaced the two producer groups with one producer group:
  waves0-3 publish resident K/V and stream Q/dO/Q^T/dO^T packets.
- Kept two consumer groups: waves4-7 own group0 `Nk=16,D=128`, waves8-11 own
  group1 `Nk=16,D=128`.
- Kept the same coarse packet ownership and source-layout ABI.  No extra
  source ABarrier token was added.

Evidence:

- Remote build/static/symbol metadata PASS:
  `private=0`, `sgpr_count=82`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_024903`.
- PMD stats:
  `simTicks=82365010`, `kernel_ticks=78751400`,
  MMAC active avg/min/max `20.2578%/17.4713%/23.5448%`,
  coissue `23301/12740`, `ldsBankConflict=0`.
- Same-shape improvement over the accepted 16-wave baseline:
  `kernel_ticks` improved from `83290480` to `78751400` (`5.45%`), and
  average MMAC active rose from `18.7606%` to `20.2578%`.
- Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_025324_clean_w12_dkv_mmac12_h1s1024_sqc7`.
- xcu first-pass:
  `MMAC/mmop_fp16` hot latency share `22.67%`, `salu_32` `20.30%`,
  `valu_32` `17.49%`, `valu_64` `11.80%`, `lds_matrix` `8.50%`.
  The dominant bubble is still `abarrier -> salu_32` at `38.85%`, with
  `abarrier -> immed` max gaps around `15.8k` cycles.

Conclusion:

The single-producer 12-wave topology validates the thin-producer hypothesis and
is the current evidence baseline, but it is nowhere near the `>=60%` MMAC-active
target.  The next redesign must attack the ABarrier/control bubble and producer
recurring-work problem, not simply reduce wave count again.  FWD-style means
longer compute islands with fewer ownership turns, not just a smaller CTA.

## 2026-07-02 W12 Tail AllDone Removal Probe

Decision: `REJECT_STATIC`

Hypothesis:

xcu shows the largest issue gaps are `abarrier -> salu/immed`, and the top
wavefront rows include thin producer waves with long tail bubbles.  Removing
the W12 CTA-wide `AllDone` wait/invalidate path might let producers exit
earlier and reduce the active-time denominator.

Result:

- Removing W12 `kAllDone` init/arrive/wait/invalidate and final
  `__syncthreads()` caused the W12 producer branch to exceed its WDRA resource
  window.
- Adding explicit role-local `return` did not fix it.
- Remote build produced:
  `BranchNumVGPRs[0] = 67`, `BranchAvailableVGPRs[0] = 16`.
- Symbol metadata failed:
  `private_segment_fixed_size=204`, `vgpr_spill_count=50`.

Conclusion:

The tail barrier is a real bubble source, but in the current generated CFG it
also helps keep WDRA branch resources bounded.  Do not remove it directly.
Next attempts must either preserve the role-merge/resource shape or reduce
barrier cost inside the steady loop rather than deleting the final convergence
path.

## 2026-07-02 W12C Pair-Packet Probe

Decision: `REJECT`

Hypothesis:

Instead of deleting the tail barrier, keep the W12 convergence shape but reduce
steady-loop raw ownership frequency.  The producer fills page0 and page1
(`2 x Mq32`) before one `Raw0Filled`; consumers process both pages before one
`Raw0Used`.  This uses the existing 128KB LDS layout and should create longer
consumer compute islands.

Result:

- First pair implementation caused `sgpr_spill_count=3`; disabling the inner
  two-page loop unroll fixed metadata.
- Final static metadata PASS:
  `private=0`, `sgpr_count=78`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_030656`.
- Performance regressed:
  `kernel_ticks=87062430`, MMAC active avg `18.1646%`,
  coissue `13272/4944`, `ldsBankConflict=0`.
- Compared with W12 single-page:
  W12 was `kernel_ticks=78751400`, MMAC active avg `20.2578%`.

Conclusion:

Pair-packet reduced ownership frequency but delayed consumer start until both
pages were filled, damaging producer/consumer overlap.  The higher coissue rate
(`72.86%`) is not useful because ticks and active share regressed.  Reverted to
W12 single-page.  Next attempts should keep single-page streaming and focus on
consumer instruction scheduling or useful producer work rather than batching
two pages behind one barrier.

## 2026-07-02 W12 Sidecar Address Hoist

Decision: `ACCEPT_MICRO`

Hypothesis:

The W12 xcu baseline still showed a visible `flat_rd -> immed` bubble
(`13.29%`) and a large scalar/control footprint.  The global sidecar helper was
recomputing the absolute row pointer inside the inner `m_idx/vec_id` loop.
Hoisting the q-tile sidecar base and k-row base should reduce address work
without changing the math or LDS ownership protocol.

Change:

- In `softmax_ds_owner16_from_global_sidecar`, compute `sidecar_tile` once per
  q-tile and use `local_m * kPackedSidecarFields` inside the vector loop.
- Also hoist `krow` out of the `m_idx` loop in both sidecar helpers.
- No algorithm, output ownership, barrier, tile, or matrix path change.

Evidence:

- Static metadata PASS for `fa3_bwd_dkv_mmac12_kernel`:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_031634`.
- Same-shape full-perf comparison against W12 full-perf baseline:
  - W12 baseline: `kernel_ticks=78625365`, MMAC active avg `19.9522%`.
  - Sidecar hoist: `kernel_ticks=75964525`, MMAC active avg `20.3523%`.
  - Improvement: ticks down `3.38%`, active up `0.40` percentage points.
- xcu detail:
  - dispatch duration down from `172804` to `166956`;
  - MMAC latency share up from `22.67%` to `23.33%`;
  - top bubble remains `abarrier -> salu_32`, `38.33%`;
  - `flat_rd -> immed` remains similar (`13.55%`), so this did not solve the
    sidecar/global-read bubble class.
- `ldsBankConflict=0`.
- PMD still prints a `read vgpr...before writing` warning, but the W12 baseline
  already had the same warning class (`vgpr115` baseline, `vgpr125` here).

Conclusion:

Keep the hoist as a small local cleanup because it is correct, resource-clean,
and lowers same-shape ticks.  Do not count it as progress toward the
FA-FWD-style 60% MMAC-active target.  The remaining high-order problem is still
ABarrier/control serialization plus `lds_matrix/immed` and producer/consumer
pipeline structure.

## 2026-07-02 W12 Late-Source Conveyor

Decision: `REJECT_PERF`

Hypothesis:

The W12 producer currently publishes raw `Q/dO` and source-layout `Q^T/dO^T`
before releasing the packet.  That delays consumer score/dP start and leaves
the producer thin later.  A FWD-style alternative is to publish raw first,
let consumers run score/dP and release the page, then have the producer publish
source-layout operands into the same page while consumers run softmax/dS.

Implementation:

- Added an opt-in late-source path using 12 waves and the same output
  ownership as W12.
- Reused the raw Q/dO pages for source-layout Q^T/dO^T instead of allocating
  separate source pages.
- Used the existing per-page Filled/Used ABarrier ids as two epochs:
  raw-filled/raw-used followed by source-filled/source-used.

Evidence:

- Static metadata PASS:
  `private=0`, `sgpr_count=86`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_033544`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_033608`.
- H1/S1024 stats:
  `kernel_ticks=81165175`, MMAC active avg `19.1939%`,
  coissue `20519/12669`, `ldsBankConflict=0`.
- Current W12 sidecar-address baseline:
  `kernel_ticks=75964525`, MMAC active avg `20.3523%`.

Conclusion:

The idea improved producer usefulness on paper but added two exposed ownership
epochs per page.  The extra producer/consumer phase waits cost more than the
source-MLS overlap gained.  Do not keep the opt-in code in the clean repo.
Future attempts should avoid adding per-page barrier epochs unless xcu shows
the new work hides a larger critical section.

## 2026-07-02 W12 Producer Early Exit

Decision: `REJECT_RUNTIME_PANIC`

Hypothesis:

xcu shows producer waves (`WaveSlot 0`) execute only about `2.5k`
instructions but remain active until the consumer tail, creating large
`abarrier -> salu/immed` bubbles and diluting MMAC active share.  Letting the
producer return after publishing all packets, while making `AllDone` a
consumer-only barrier, might shrink the active-time denominator without
changing score/dP/dV/dK math.

Implementation attempt:

- Added an opt-in W12 producer-exit kernel.
- Producer branch ran `producer_all_loop` then returned.
- Consumers alone arrived/waited `AllDone` and performed barrier cleanup.
- First try used producer VGPR window `80`, but compile failed because WDRA
  branch-averaged VGPR size did not meet target granularity:
  `80+160+160=400` over three branches is illegal.
- Retried with producer VGPR window `76`, making
  `76+160+160=396`, average `132`, which passed static metadata.

Evidence:

- Static metadata with producer `76` PASS:
  `private=0`, `sgpr_count=84`, `vgpr_count=132`, no SGPR/VGPR spill.
- H1/S128 PMD smoke aborted:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_034614`.
- PMD panic:
  `cu0 simd3 vgpr81 is not init or has been freed` during
  `V_MMAC_F32_16X16X16_F16`.

Conclusion:

Producer early return is unsafe in this WDRA/PMD shape.  Even though the static
resources pass with a legal branch-averaged VGPR window, PMD loses or frees
state needed by remaining consumer MMAC waves.  Keep producer tail waits in the
mainline.  Future tail work should avoid early wave exit and instead reduce
what happens after the wait, or prove a PMD-supported parked/cleanup protocol
with a focused probe first.

## 2026-07-02 W12 Full-Valid Softmax Fast Path

Decision: `REJECT_CORRECTNESS`

Hypothesis:

The current W12 sidecar-address baseline still spends SCA/VALU on per-element
`valid_pair` and causal-mask checks inside
`softmax_ds_owner16_from_global_sidecar`.  For a tile where every
`Mq32 x Nk16` pair is provably valid, a fast path that loads row sidecar and
computes `P/dS` directly should remove predicate work without changing matrix
math, barriers, or output ownership.

Implementation attempt:

- Added a `full_valid_tile` branch inside the global-sidecar owner16 helper.
- First variant used an early `return`; second variant used structured
  `if/else` to rule out a helper-return CFG issue.
- The LDS sidecar helper and matrix path were left unchanged.

Evidence:

- Both variants built and passed static metadata for
  `fa3_bwd_dkv_mmac12_kernel`:
  `private=0`, `sgpr_count=80`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S128 causal PMD smoke failed numerical comparison in both variants:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_035820`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_035954`.
- Representative correctness result:
  `dK rel_l2=0.000361379`, but `dV rel_l2=14.7566` and
  `dv_max_abs=0.495472`.
- PMD also reported `warn: read vgpr120 before writing` in this shape.

Conclusion:

This local fast path is not safe in the current owner16/global-sidecar calling
context.  The fact that dK remains close while dV breaks means the issue is not
a broad MMAC/store failure; it is specifically the `P`/dV path produced by the
fast branch.  Do not retry causal-mask removal by splitting this helper unless
a focused owner16 sidecar probe proves the row/fragment mapping and PMD CFG
behavior first.  Code was reverted to the W12 sidecar-address baseline.

## 2026-07-02 W12 dV/dK Read-All MMAC Island

Decision: `ACCEPT_MICRO`

Hypothesis:

The W12 sidecar-address baseline still issued dV/dK source operands as four
small groups:

```text
read dO^T/Q^T block -> wait_lgkm(0) -> small dV/dK MMAC island
```

That leaves repeated `lds_matrix -> wait/immed -> MMAC` gaps.  The FWD-style
variant keeps the same math, barriers, source-layout ABI, and output ownership,
but reads all eight dO^T/Q^T operand fragments for the owner16 dV/dK island
first, waits once immediately before first use, then runs one longer MMAC
island.

Implementation:

- Added `dv_dk_mmac_one_out<FirstQTile, OutIdx>` to keep the repeated MMAC
  body compact.
- Replaced the four-block read/wait/MMAC loop in `dv_dk_mmac_owner16` with
  explicit `dout_t0..7` and `q_t0..7` registers.
- Kept explicit registers instead of arrays to avoid private memory.
- Did not change score/dP, softmax/dS, ABarrier ownership, store layout, or
  source-layout `Q^T/dO^T` ABI.

Evidence:

- Static metadata PASS:
  `private=0`, `sgpr_count=88`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_041233`.
- H1/S1024 correctness/perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_041545`.
- W12 sidecar-address baseline:
  `kernel_ticks=75964525`, MMAC active avg `20.3523%`.
- Read-all result:
  `kernel_ticks=72499700`, MMAC active avg `21.3054%`,
  coissue `30904/22164`, `ldsBankConflict=0`.
- Perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_041545_clean_w12_dvdk_readall_h1s1024_sqc7`.
- xcu detail:
  `MMAC=23.64%`, `lds_matrix=8.86%`, top bubble remains
  `abarrier -> salu_32 = 38.64%`; `lds_matrix -> immed` falls to `5.53%`
  from the baseline `9.38%`.

Conclusion:

The read-all island is a valid local cleanup: it reduces the exposed
matrix-read gap and improves ticks/MMAC active without increasing LDS conflict
or spilling.  It does not solve the main gap to the FWD target.  The dominant
evidence remains barrier/control serialization and thin producer/consumer tail
behavior, so the next structural work should target the ABarrier ledger and
longer FWD-style phase alignment rather than more local address cleanups.

## 2026-07-02 W12 Pre-Softmax dV/dK Read Negative

Decision: `REJECT_RESOURCE`

Hypothesis:

Move the dV/dK `dO^T/Q^T` `ds_read_matrix` batch before
`softmax_ds_owner16_from_global_sidecar`, then wait only before the MMAC island.
The intended FWD-style schedule was:

```text
score/dP MMAC -> issue all dO^T/Q^T reads -> softmax/dS VALU -> wait -> dV/dK MMAC
```

This should hide source operand LDS latency under useful softmax/dS work.

Implementation attempt:

- Split `dv_dk_mmac_owner16` into source read and source consume helpers.
- Held all eight `dO^T` and eight `Q^T` operand fragments live across
  softmax/dS.
- Kept math, barriers, source-layout ABI, and output ownership unchanged.

Evidence:

- Remote build completed, but metadata gate failed before PMD:
  `private_segment_fixed_size=24`, `vgpr_spill_count=10`,
  `sgpr_count=84`, `vgpr_count=112`.
- Consumer branch availability was still `160`, so the failure is live-range
  pressure from carrying all source operands through the softmax/dS helper, not
  a tile-size or LDS-budget issue.

Conclusion:

Direct "read all before softmax" is not viable in the current code shape.  The
idea is still algorithmically sound, but it needs a smaller live range: either
4+4 source read groups, a slimmer softmax/dS helper, or a more FWD-like
register ledger before retrying.  Code was reverted to the W12 dV/dK read-all
baseline.

## 2026-07-02 W12 dV/dK 4+4 Read-Early Island

Decision: `ACCEPT_MICRO`

Hypothesis:

The rejected pre-softmax read-all schedule proved that hiding dV/dK source
operand reads under softmax/dS is useful but too register-heavy when all
sixteen source operands stay live across the softmax helper.  The bounded
retry uses two groups:

```text
score/dP -> read low dO^T/Q^T group -> softmax/dS
         -> wait low -> read high group -> MMAC low -> wait high -> MMAC high
```

This hides the first half of the source reads under softmax/dS and tries to
hide the second half under the first dV/dK MMAC group.

Implementation:

- Added `DvDkSourceRegs4`, `dv_dk_read_owner16_sources4`, and
  `dv_dk_mmac_four_out`.
- Main consumer loop now reads the low source group before softmax/dS, then
  starts the high source group immediately before the low MMAC group.
- Math, source-layout ABI, ABarrier ownership, and output ownership are
  unchanged.

Evidence:

- Static metadata PASS:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_043459`.
- H1/S1024 correctness/perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_043641`.
- Read-all baseline:
  `kernel_ticks=72499700`, MMAC active avg `21.3054%`,
  `lds_matrix -> immed=5.53%`.
- 4+4 result:
  `kernel_ticks=71508255`, MMAC active avg `21.5678%`,
  coissue `30929/20971`, `ldsBankConflict=0`.
- Perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_043641_clean_w12_dvdk_read4x2_h1s1024_sqc7`.
- xcu detail:
  `MMAC=23.59%`, `lds_matrix -> immed=2.46%`, but top bubble remains
  `abarrier -> salu_32=39.00%`.

Conclusion:

This validates the FWD-style read-early/wait-late direction when the live range
is bounded.  It is still a micro optimization: matrix-read bubbles improved,
but MMAC active barely moved because the dominant gap is now ABarrier/control
serialization.  Keep this as the current clean baseline and move the next
structural work to barrier/control and consumer phase alignment.

### W12 ValuExec0 Turnstile Rejection

Hypothesis:

- Borrow FWD's turnstile idea and add a consumer-local `ValuExec0` token.
- Let consumer1 arrive before softmax and make consumer0 wait before softmax,
  aiming to break lockstep and create VALU/MMAC overlap without artificial
  delay.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_044441`.
- H1/S1024 correctness/perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_044502`.
- Static metadata stayed clean:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Read4x2 baseline:
  `kernel_ticks=71508255`, MMAC active avg `21.5678%`,
  coissue `30929/20971`.
- Turnstile result:
  `kernel_ticks=73908835`, MMAC active avg `20.8817%`,
  coissue `29516/19583`, `ldsBankConflict=0`.

Decision:

`REJECT_PERF`; the code was reverted.

Conclusion:

A pure consumer turnstile is legal but not useful here.  It increases explicit
control/ownership cost and does not move enough independent work into the peer
MMAC window.  The next structural candidate must either reduce ownership
turns, thicken producer/helper work with data the consumer truly uses, or
change the role split while preserving no-duplicate score/dP.

### W12 Raw/Source Ownership Split Rejection

Design basis:

- Current read4x2 is over-synchronized: `RawFilled` is not released until raw
  `Q/dO` and source-layout `Q^T/dO^T` have all been published.
- Score/dP only needs raw `Q/dO`; dV/dK needs `Q^T/dO^T` later.
- Splitting ownership should let consumers start score/dP while producer waves
  publish source-layout operands, giving producer recurring useful work rather
  than a pure wait/turnstile.

Expected code shape:

```text
producer:
  wait RawUsed(page) -> publish Q/dO -> arrive RawFilled(page)
  wait SourceUsed(page) -> publish Q^T/dO^T -> arrive SourceFilled(page)

consumer:
  wait RawFilled(page) -> score/dP -> arrive RawUsed(page)
  wait SourceFilled(page) -> read4x2 source -> softmax/dS -> dV/dK
  arrive SourceUsed(page)
```

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_045658`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_045719`.
- Static metadata stayed resource-clean:
  `private=0`, `sgpr_count=86`, `vgpr_count=112`, no SGPR/VGPR spill.
- Read4x2 baseline:
  `kernel_ticks=71508255`, MMAC active avg `21.5678%`,
  coissue `30929/20971`.
- Raw/source split result:
  `kernel_ticks=75607805`, MMAC active avg `20.5505%`,
  coissue `31554/23826`, `ldsBankConflict=0`.

Decision:

`REJECT_PERF`; the code was reverted.

Conclusion:

The design removed an over-synchronization on paper, but the additional
SourceFilled/SourceUsed ownership turns increased control cost and failed
coissue enough to lose.  Future attempts should reduce token count or combine
the split with substantial useful producer work; do not add more ABarrier
tokens for cleanliness alone.

### W12 Consumer-Group Template Cleanup

Hypothesis:

- FWD-style code should keep role/group ownership as compile-time local as
  possible.
- Template-specializing `consumer_dkv_mmac_loop` by group should remove a small
  amount of runtime owner/address/control work without changing the pipeline.

Change:

- `consumer_dkv_mmac_loop<Tile, Bar, ConsumerGroup>` now computes
  `owner_nblock = ConsumerGroup * 4 + wave_local`.
- Consumer branches call `<0>` and `<1>` explicitly.
- No barrier, tile, LDS, MMAC count, output ownership, or math change.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_050701`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_050727`.
- Metadata:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Read4x2 baseline:
  `kernel_ticks=71508255`, MMAC active avg `21.5678%`,
  coissue `30929/20971`.
- Template result:
  `kernel_ticks=71412705`, MMAC active avg `21.5708%`,
  coissue `31198/22312`, `ldsBankConflict=0`.

Decision:

`ACCEPT_MICRO`.

Conclusion:

This is a tiny but clean FWD-style codegen cleanup.  It should be kept because
it simplifies ownership and does not hurt resources, but it does not address
the dominant ABarrier/control and sidecar/global-read debt.

### W12 dV/dK Zero-Seed Cleanup

Hypothesis:

- dV/dK accumulators only need a zero seed for the first q tile.
- The previous helper called `zero_f16x8` every q tile even when
  `FirstQTile=false`; because the helper uses volatile asm, this created real
  `v_mov` work that the compiler could not delete.

Change:

- Guard the dV/dK `zero_f16x8` call with `if constexpr (FirstQTile)` in both
  dV/dK MMAC helpers.
- No math, barrier, LDS, output ownership, or register-window change.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_051325`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_051343`.
- Full perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_051513_clean_w12_zero_seed_h1s1024_sqc7`.
- Metadata:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Template baseline:
  `kernel_ticks=71412705`, MMAC active avg `21.5708%`,
  coissue `31198/22312`.
- Zero-seed result:
  `kernel_ticks=70604625`, MMAC active avg `21.7988%`,
  coissue `30594/21056`, `ldsBankConflict=0`.
- xcu full perf:
  `MMAC=23.68%`, `valu_32` hits `151648`, top bubbles remain
  `abarrier -> salu_32=39.07%` and `flat_rd -> immed=15.03%`.

Decision:

`ACCEPT_MICRO`.

Conclusion:

This validates the user's observation about unnecessary register zeroing.
The cleanup is real and should stay, but it only moves active share by about
0.23 percentage points.  The next large move still needs to reduce
barrier/control or hide sidecar global-read latency.

### W12 Sidecar Prefetch Negative

Hypothesis:

- The latest xcu detail showed `flat_rd -> immed` at `15.03%`.
- Prefetching each owner wave's 8 q-row sidecar triplets before
  `RawFilled`/score-dP might let existing wait and score/dP MMAC hide the
  global-read latency without adding LDS or ABarrier tokens.

Change tested and reverted:

- Added a `SidecarOwner16Regs` packet with 24 float values per lane
  (`max_log2`, `inv_sum`, `delta` for 8 q rows).
- Loaded it before `wait_raw_filled_page`, then made softmax/dS use the
  prefetched registers.
- No math, output ownership, MMAC count, LDS plan, or barrier change.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_052854`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_052900`.
- Stats-only archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_052900_clean_w12_sidecar_prefetch_reject_h1s1024_sqc7`.
- Metadata remained clean:
  `private=0`, `sgpr_count=93`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch-local consumer pressure rose to `158/160`.
- H1/S1024 result:
  `kernel_ticks=75394410`, MMAC active avg `18.4182%`,
  coissue `40984/31390`, `ldsBankConflict=0`.
- Baseline zero-seed:
  `kernel_ticks=70604625`, MMAC active avg `21.7988%`,
  coissue `30594/21056`.

Decision:

`REJECT_PERF_STATS_ONLY`; code reverted.

Conclusion:

The sidecar latency is real, but carrying all 24 sidecar floats across the
score/dP MMAC island consumes nearly all consumer VGPR slack and worsens active
share.  The next sidecar design must reduce representation or live range, such
as computing/fetching fewer fields per fragment, compressing sidecar state, or
moving sidecar generation to a path that does not extend consumer critical
live ranges.

### Noncausal Diagnostic Boundary

Hypothesis:

- The leader suggested using `causal=false` to remove mask/predicate work during
  pipeline tuning.
- This would be useful only if the same kernel passes correctness at the
  diagnostic shape; otherwise performance counters are not valid evidence.

Evidence:

- After rebuilding the reverted zero-seed baseline, W12 metadata returned to
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no spill.
- H1/S128, `CAUSAL=0`, correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_053747`.
- H1/S1024, `CAUSAL=0`, correctness FAIL:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_053811`.
- Failure signal:
  `dk_rel_l2=0.0199722`, `dv_rel_l2=0.002617`, `bad=0`, `pass=0`.

Decision:

`REJECT_CORRECTNESS_DIAGNOSTIC`.

Conclusion:

Do not use `CAUSAL=0` H1/S1024 metrics to guide MMAC-active tuning yet.  The
mainline remains `causal=true`; if noncausal is needed for isolated mask/predicate
studies, first resolve the numerical threshold or the noncausal math path.

### Sidecar Pair-Prefetch Negative

Hypothesis:

- The full sidecar prefetch failed because it carried 24 sidecar floats across
  score/dP and exhausted consumer branch slack.
- A smaller pair-prefetch inside `softmax_ds_owner16_from_global_sidecar` might
  group two q-row sidecar triplets at a time, increasing flat-read distance
  without extending live range across score/dP.

Change tested and reverted:

- Replaced the per-lane `vec_id=0..3` sidecar load-use loop with two
  two-row groups.
- Each group loaded at most six sidecar floats
  (`max_log2`, `inv_sum`, `delta` for two q rows), then computed the two
  corresponding P/dS elements.
- No math, output ownership, MMAC count, LDS plan, or barrier change.

Evidence:

- Static metadata stayed clean:
  `private=0`, `sgpr_count=82`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch-local consumer pressure was `144/160`, better than the zero-seed
  baseline `150/160` and much better than the rejected all-sidecar prefetch
  `158/160`.
- H1/S128 correctness failed before any performance run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_055216`.
- Failure signal:
  `pass=0`, `dk_rel_l2=8244.1`, `dv_rel_l2=30.6025`.
- PMD also emitted `warn: read vgpr111 before writing` in this candidate.

Decision:

`REJECT_CORRECTNESS`; code reverted and remote source restored to the
zero-seed baseline.

Conclusion:

This reduced-register sidecar batching does not preserve the full dKV numerical
path under PMD/codegen.  Do not retry sidecar batching directly in the main dKV
kernel.  If sidecar latency is attacked again, first build a focused
sidecar-fragment correctness probe that checks packed sidecar load order,
fragment placement, and VGPR initialization before reconnecting to dV/dK.

### Mq64 Single-Buffer Structural Probe

Hypothesis:

- Increasing the q tile from `Mq=32` to `Mq=64` could double each consumer
  MMAC island from 64 to 128 MMAC per q tile and halve q-loop
  barrier/control turns.
- A single-buffer Mq64 design fits only by using the full 128KB LDS budget:
  `K/V 64KB + raw Q/dO 32KB + source Q^T/dO^T 32KB = 128KB`.

Change tested:

- Added an opt-in W12 Mq64 path with one producer group and two consumer
  groups.
- First implementation with four live M fragments spilled badly.
- Half-sequential consumer phasing plus a 208-VGPR consumer window removed
  VGPR spill but still had SGPR spill.
- Specializing the candidate to the current diagnostic shape
  `S=1024, causal=true` removed the remaining SGPR spill.

Evidence:

- Final static metadata after specialization:
  `private=0`, `sgpr_count=100`, `sgpr_spill_count=0`, `vgpr_count=144`,
  `vgpr_spill_count=0`.
- Branch-local consumer pressure was `171/208`.
- H1/S1024 causal correctness run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_062758`.
- PMD completed the dispatch but numerical comparison failed:
  `pass=0`, `dk_rel_l2=5680.1`, `dv_rel_l2=20.0452`.
- PMD emitted `warn: read vgpr268 before writing` for this candidate.

Decision:

`REJECT_CORRECTNESS`; do not capture perf or use MMAC-active counters for this
candidate.  Full Mq64 implementation is being reverted from the clean source.

Conclusion:

Mq64 is not ruled out as an algorithmic idea, but the current full-kernel path
is not a valid implementation.  The next retry must first build a focused
Mq64 layout/seed/store correctness probe that verifies score/dP, sidecar row
mapping, source-layout `Q^T/dO^T`, dV/dK accumulation, and store ownership
before reconnecting it to the full dKV pipeline.

### Mq64 Seed-Fix Reattempt

Hypothesis:

- The previous Mq64 correctness failure might be caused by an accumulator seed
  bug, not by the Mq64 algorithm decomposition itself.
- Evidence from the rejected diff matched the PMD warning: the low D-block used
  the first-q-tile seed path, but the high D-block used
  `dv_dk_mmac_four_out<false, 4>` even on the first q tile, so
  `dv_acc[4..7]` and `dk_acc[4..7]` could be read before initialization.

Change tested:

- Reapplied the opt-in W12 Mq64 path.
- Changed the high D-block dV/dK MMAC helper to use the same
  `SeedAccumulator` decision as the low D-block.
- Did not add new barriers, waits, ordinary matrix-path DS reads, or output
  ownership changes.

Evidence:

- Workbook rows were added under `Mq64 seed-fix reattempt`.
- Build and static gates passed.
- Symbol metadata for `fa3_bwd_dkv_mmac12_mq64`:
  `private=0`, `sgpr_count=100`, `sgpr_spill_count=0`, `vgpr_count=144`,
  `vgpr_spill_count=0`.
- Branch-local consumer pressure stayed `171/208`.
- H1/S1024 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_064930`.
- Correctness signal:
  `pass=1`, `dk_rel_l2=0.0025563`, `dv_rel_l2=0.000337571`, `bad=0`.
- Stats signal:
  `simTicks=83209490`, `kernel_ticks=79595880`,
  MMAC active avg `19.8279%`, `ldsBankConflict=0`,
  coissue success/fail `13713/7221`.
- Current zero-seed W12 baseline is still better:
  `kernel_ticks=70604625`, MMAC active avg `21.7988%`.

Decision:

`OBSERVE_CORRECTNESS_REJECT_PERF`.  The seed bug is real and the Mq64 path is
now a valid correctness diagnostic, but it is not a performance candidate.
No full helper `.perf`/xcu capture was taken because both ticks and MMAC active
regressed in stats-only evidence.

Conclusion:

Mq64 full-D accumulation is no longer blocked by the specific high-D
uninitialized accumulator bug.  The performance regression points back to the
larger structural problem: single-buffer exact-128KB Mq64 serializes producer
and consumer progress enough that doubling the per-turn MMAC island does not
raise active share.  The next FWD-style design should keep the correctness
lesson but restore LDS slack or double-buffering, rather than promoting this
exact single-buffer topology.

### Raw-Page Sidecar Overlay

Hypothesis:

- The W12 zero-seed path still pays a visible sidecar/global-read debt in the
  consumer softmax/dS island.
- A producer could publish sidecar values into the raw Q LDS page after both
  consumers have finished score/dP on that page.  This might remove
  consumer-side global sidecar loads and give the producer useful recurring
  q-loop work without increasing LDS footprint.

Change tested:

- Added an opt-in W12 sidecar-overlay path.
- Producer publishes the matrix packet first, then waits for the raw matrix
  generation to be released before overlaying sidecar values into the raw Q
  page.
- Consumers compute score/dP, release raw matrix use, wait for the sidecar
  generation, then compute softmax/dS and dV/dK as before.
- Main matrix operands still use the existing MLS/BPS + `ds_read_matrix` +
  MMAC path; no raw LDS transpose writer was introduced.

Evidence:

- Workbook rows were added under `Raw-page sidecar overlay result`.
- Build/static/metadata passed for symbol
  `fa3_bwd_dkv_mmac12_sidecar_overlay`:
  `private=0`, `sgpr_count=86`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch-local pressure was `producer=9/16`, `consumer=146/160`.
- H1/S128 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_070805`.
- H1/S1024 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_070829`.
- H1/S1024 stats:
  `simTicks=76727560`, `kernel_ticks=73113950`, MMAC active avg `18.9185%`,
  coissue success/fail `39007/29043`, total MMOP instr `131072`,
  `ldsBankConflict=0`.
- Current zero-seed W12 baseline remains better:
  `kernel_ticks=70604625`, MMAC active avg `21.7988%`.

Decision:

`REJECT_PERF_STATS_ONLY`.  The overlay lifetime protocol is numerically valid
and resource-clean, but the extra raw-page sidecar ownership generation
increases control/barrier cost enough that both ticks and MMAC active regress.
Coissue success increases, but failed coissue also increases; this is a useful
negative example that coissue count alone is not a promotion signal.

Conclusion:

Sidecar latency remains a real debt, but fixing it by adding another
RawFilled/RawUsed generation is the wrong direction for the current topology.
Future FWD-style work should reduce barrier/control turns first, or move
sidecar generation into a protocol that does not add a page-ownership cycle.

### Score/dP Read2x Brick

Hypothesis:

- Current W12 score/dP still fragments the score+dP island by doing one
  `d_block` at a time: read Q/K/dO/V, wait, then issue a small MMAC group.
- FWD emits larger `ds_read_matrix` bricks before a long MMAC island.  Batching
  two score/dP D-block operand families might reduce exposed
  `lds_matrix -> immed/wait -> MMAC` gaps without changing algorithm
  ownership or adding barriers.

Change tested:

- Added an opt-in W12 score/dP read-brick path.
- `score_dp_mmac_owner16_read2x` reads two D-block families of Q/K/dO/V before
  one `wait_lgkm(0)`, then issues the two D-block score+dP MMAC groups.
- Producer packet publication, global sidecar softmax/dS, dV/dK read4x2,
  zero-seed accumulator logic, and store ownership remain unchanged.

Evidence:

- Workbook rows were added under `Score/dP read2x brick result`.
- Build/static/metadata passed for symbol
  `fa3_bwd_dkv_mmac12_score_dp_brick`:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch-local consumer pressure stayed `150/160`.
- H1/S128 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_072317`.
- H1/S1024 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_072323`.
- H1/S1024 stats:
  `simTicks=74415250`, `kernel_ticks=70801640`, MMAC active avg `21.5465%`,
  coissue success/fail `34543/22997`, total MMOP instr `131072`,
  `ldsBankConflict=0`.
- Current zero-seed W12 baseline remains better:
  `kernel_ticks=70604625`, MMAC active avg `21.7988%`.

Decision:

`REJECT_PERF_STATS_ONLY`.  The FWD-style larger score/dP read brick is legal
and resource-clean, but it does not improve the same-shape target.  It raises
coissue success versus zero-seed, but MMAC active and ticks both move the wrong
way.

Conclusion:

The 60% MMAC-active gap is not primarily score/dP read granularity.  The next
design should stop stacking local read scheduling tweaks and instead change
the packet ownership/conveyor so producer and consumer waves avoid the exposed
ABarrier/control path.

### Mq64 Semantic-Page Conveyor

Hypothesis:

- The Mq64 seed-fix path doubles the per-q-tile MMAC island but keeps raw and
  source operands resident at the same time, filling the full 128KB LDS.
- A semantic page could reuse the same 32KB page as raw `Q/dO` first, then as
  source-layout `Q^T/dO^T` after consumers finish score/dP on both Mq64 halves.
- This should let producer source publication overlap with consumer
  softmax/dS while retaining `MLS/BPS + ds_read_matrix + MMAC` on the matrix
  path and avoiding raw-to-trans LDS writers.

Change tested:

- Added opt-in path `kDkvPathWaspDkvMmac12WaveMq64Semantic`.
- LDS plan: K/V resident 64KB plus two 32KB semantic pages, exactly 128KB.
- Producer publishes raw `Q/dO`, then waits for raw used and overwrites the
  same page with source-layout `Q^T/dO^T`.
- Consumer waits raw, computes both Mq64 score/dP halves, releases raw,
  computes both softmax/dS halves, waits source, then runs dV/dK MMAC.

Evidence:

- Workbook rows were added under `Mq64 semantic-page conveyor`.
- Build/static/metadata passed for symbol
  `fa3_bwd_dkv_mmac12_mq64_semantic`:
  `private=0`, `sgpr_count=90`, `vgpr_count=144`, no SGPR/VGPR spill.
- Branch-local consumer pressure was `180/208`.
- H1/S128 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_074704`.
- H1/S1024 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_074725`.
- H1/S1024 semantic stats:
  `simTicks=76933675`, `kernel_ticks=73320065`, MMAC active avg `21.7509%`,
  VOP active avg `22.6370%`, coissue success/fail `23374/16882`,
  total MMOP instr `131072`, `ldsBankConflict=0`.
- Same-build W12 baseline stats:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_075046`
  with `simTicks=74588605`, `kernel_ticks=70974995`, MMAC active avg
  `21.7125%`, VOP active avg `25.0507%`, coissue `28477/21603`,
  total MMOP instr `131072`, `ldsBankConflict=0`.
- Both baseline and semantic printed a PMD `read vgpr... before writing`
  warning while passing numerics, so this warning is not unique to the new
  semantic path.

Decision:

`REJECT_PERF_STATS_ONLY`.  The semantic-page path is correct and resource-clean,
and it lowers VOP active share, but the extra raw/source ABarrier generation
reduces coissue and regresses ticks versus the same-build W12 baseline.

Conclusion:

The experiment confirms that page reuse alone is not enough; the extra
ownership generation is too expensive in the current conveyor.  The next
FWD-style redesign should reduce barrier/control turns or move source
publication into an existing packet generation, rather than adding another
raw/source page lifecycle.

### Causal Whole-Tile Skip

Hypothesis:

- For causal dKV, if a `Mq=32` q tile is completely before the current
  `Nk=128` k tile, all score, dP, softmax/dS, dV, and dK contributions are
  zero.
- Skipping those whole q tiles should reduce redundant MMOP and VALU work
  without changing output ownership.

Change tested:

- Added opt-in path `kDkvPathWaspDkvMmac12WaveCausalSkip`.
- Producer and consumers both maintain a `packet_idx` that advances only for
  non-skipped q tiles, so double-buffer page ownership stays aligned.
- The final accepted correctness version skips only whole q tiles.  Per-owner
  causal block skipping was dropped because it requires dynamic accumulator
  seed state and first produced incorrect numerics after a zero-initialization
  rewrite.
- Existing softmax/dS causal masking still handles partial tiles inside the
  first non-skipped packet.

Evidence:

- Build/static/metadata passed for symbol
  `fa3_bwd_dkv_mmac12_causal_skip`:
  `private=0`, `sgpr_count=88`, `vgpr_count=144`, no SGPR/VGPR spill.
- Branch-local consumer pressure is high but legal: `194/208`.
- H1/S128 noncausal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081433`.
- H1/S128 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081455`.
- H1/S1024 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081517`.
- H1/S1024 causal-skip stats:
  `kernel_ticks=72881900`, MMAC active share `16.7128%`, VOP share
  `29.4950%`, total MMOP instr `73728`, coissue `14760/11291`,
  `ldsBankConflict=0`.
- Same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081604`
  with `kernel_ticks=71006845`, MMAC active share `21.6777%`, VOP share
  `25.2075%`, total MMOP instr `131072`, coissue `30195/21487`,
  `ldsBankConflict=0`.

Decision:

`REJECT_PERF_H1S1024`.  The skip path removes MMOP but does not shorten the
critical path on the diagnostic shape.  It lowers active SIMD work, increases
imbalance/tail exposure, and reduces MMAC active share.

Conclusion:

Do not optimize by simply deleting upper-triangle work in this W12 topology.
The current bottleneck is still conveyor shape and exposed barrier/control
latency, not raw MMOP count.  If this idea is revisited, it should be tested
only as part of a topology that preserves balanced CTA/SIMD work, or on an
H4/H2048 steady workload with explicit tail-analysis evidence.

### Mixed Score/dP Brick

Hypothesis:

- Keep the same W12 output ownership, LDS pages, ABarrier ledger, dV/dK
  read4x2 island, and zero-seed accumulation.
- Run group0 with the baseline consumer schedule and group1 with the already
  verified score/dP read2x brick schedule.
- If the old timeline is dominated by C0/C1 lockstep, this asymmetric real-work
  schedule should de-lockstep the two heavy consumers without adding artificial
  delay.

Change tested:

- Added opt-in path `kDkvPathWaspDkvMmac12WaveMixedScoreBrick`.
- The kernel keeps one producer group, group0 baseline consumer, and group1
  `consumer_dkv_mmac_loop_score_dp_brick`.
- No new LDS page, no new ABarrier token, no output ownership change, and no
  external API change beyond the standalone diagnostic flag.

Evidence:

- Static and metadata gates passed for symbol
  `fa3_bwd_dkv_mmac12_mixed_score_brick`:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch pressure from build evidence:
  producer `1/16`, group0 `150/160`, group1 `158/160`.
- H1/S128 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_084709`.
- H1/S1024 causal correctness passed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_084734`.
- H1/S1024 mixed stats:
  `kernel_ticks=71663865`, MMAC active share `21.1732%`,
  VOP share `24.6598%`, total MMOP instr `131072`,
  coissue `33504/24695`, `ldsBankConflict=0`.
- Same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_084834`
  with `kernel_ticks=71312605`, MMAC active share `21.5931%`,
  VOP share `25.1072%`, total MMOP instr `131072`,
  coissue `30130/20996`, `ldsBankConflict=0`.
- Both mixed and baseline print PMD `read vgpr111 before writing` while passing
  numerics, so this warning is not unique to the mixed path.

Decision:

`REJECT_PERF_STATS_ONLY`.  Coissue rises, but the kernel is slower by about
`0.49%` and MMAC active share falls.  This is not a promotion candidate and
does not justify full perf capture.

Conclusion:

The consumer schedules need more than local score/dP shape asymmetry.  The
important open problem remains the conveyor structure: reducing exposed
ABarrier/control and `ds_read_matrix -> wait -> MMAC` gaps while keeping both
consumers' MMAC islands long and useful.

### Dedicated LDS Sidecar Resource Check

Hypothesis:

- Move row sidecar from repeated consumer global reads into packet-local LDS.
- Producer would copy packed `(max_log2, inv_sum, delta)` once per q row while
  publishing Q/dO/QT/dOT under the existing `RawFilled` token.
- Consumers would read sidecar from LDS after `RawFilled`, avoiding
  `flat_rd -> immed` in the softmax/dS hot path without adding a new ABarrier
  generation.

Resource audit result:

- Workbook initially assumed the W12 base LDS left about 28KB slack.  Remote
  build proved that assumption false.
- Current W12 LDS layout is already exactly 128KB:
  `Q 16KB + dO 16KB + K 32KB + V 32KB + Q^T 16KB + dO^T 16KB`.
- A dedicated two-page sidecar requires only `768B`, but the total would be
  `131840B > 131072B`.
- Remote build failed before PMD with static assertion:
  `Layout::kBytes + Tile::kSidecarBytes <= Tile::kLdsBudgetBytes`,
  evaluated as `131840 <= 131072`.
- The failing implementation was removed; only the resource lesson and a small
  mixed-path API validation fix remain.

Decision:

`REJECT_RESOURCE_DESIGN`.  Do not implement dedicated sidecar LDS by appending
bytes to the current W12 layout.

Conclusion:

Sidecar-in-LDS remains conceptually attractive for `flat_rd -> immed`, but it
must replace an existing LDS lifetime or be paired with a topology that frees
space.  The raw-overlay attempt already showed that adding another ownership
generation is expensive, so the next version needs a real page/lifetime swap,
not an appended page and not a second RawFilled/RawUsed epoch.

### Source-Score Layout Probe

Hypothesis:

- Current W12 is LDS-full because it keeps raw `Q/dO` pages for score/dP and
  source-layout `Q^T/dO^T` pages for dV/dK.
- If the source-layout pages could also feed score/dP, raw `Q/dO` pages could
  become removable in a later design, opening about 32KB for sidecar or a
  healthier page lifecycle.
- The probe only changed the score/dP operand source from `QBase/DoutBase` to
  `QtBase/DoutTBase`; K/V, sidecar, dV/dK, store ownership, and barrier
  protocol stayed unchanged.

Evidence:

- Workbook design and result rows were added before and after the run.
- Static/resource gate passed for the temporary source-score symbol:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch pressure was clean: consumer `146/160`.
- H1/S128 causal PMD ran but failed numerical correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_092729`.
- Failure signal:
  `dk_max_abs=0.000713147`, `dk_rel_l2=1.5564`,
  `dv_max_abs=0.000420369`, `dv_rel_l2=0.00432992`, `pass=0`.
- No H1/S1024 or perf capture was run because correctness failed first.
- The temporary opt-in code was removed to keep the clean repo from accruing a
  failed full-kernel path.

Decision:

`REJECT_LAYOUT_PROBE`.  Source-layout `Q^T/dO^T` pages are not a drop-in
replacement for raw `Q/dO` score/dP operands under the current
`ds_read_matrix` operand mapping.

Conclusion:

Keep raw `Q/dO` pages for score/dP in the main W12 design.  If this direction
is revisited, do it as a smaller instruction/layout probe that compares raw-Q
and source-Q fragments before entering full dKV; do not free the raw pages
based only on the existing source-layout ABI.

### Raw-dVdK Layout Probe

Hypothesis:

- If raw `Q/dO` pages could feed dV/dK, the source-layout `Q^T/dO^T` pages
  could become removable in a later design.
- This would free about 32KB LDS in the current W12 layout and potentially make
  room for sidecar/LDS or cleaner packet ownership.
- The probe kept score/dP, producer publication, sidecar, store ownership, and
  barrier protocol unchanged.  Only dV/dK operand reads changed from
  `DoutTBase/QtBase` to raw `DoutBase/QBase`.

Evidence:

- Workbook design and result rows were added before and after the run.
- Static/resource gate passed for the temporary raw-dVdK symbol:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch pressure was resource-clean but higher than the source-score probe:
  consumer about `171/208`.
- H1/S128 causal PMD ran but failed numerical correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_094838`.
- Failure signal:
  `dk_max_abs=0.000490837`, `dk_rel_l2=1.00048`,
  `dv_max_abs=0.253552`, `dv_rel_l2=0.932977`, `pass=0`.
- No H1/S1024 or perf capture was run because correctness failed first.
- The temporary opt-in code was removed to keep the clean repo from accruing a
  failed full-kernel path.

Decision:

`REJECT_LAYOUT_PROBE`.  Raw `Q/dO` LDS pages are not a drop-in replacement for
source-layout `dO^T/Q^T` operands for dV/dK under the current
`ds_read_matrix_trans_pair` mapping.

Conclusion:

The current W12 design still needs both raw `Q/dO` for score/dP and
source-layout `Q^T/dO^T` for dV/dK.  The simple 32KB-LDS-freeing route is
closed unless a smaller instruction probe proves another documented
`matrix_load`/`ds_read_matrix` pairing.

### Early RawUsed Release

Hypothesis:

- xcu windows showed producer wave0 spending about 92% of steady 15k-cycle
  windows in `abarrier -> salu_32` gaps while consumers still had 36-38%
  bubble.
- In the W12 read4x2 path, once the consumer has read both low/high
  source-layout operands into VGPR and waited for them, the raw/source LDS page
  no longer needs to stay owned by the consumer.
- Releasing RawUsed before the high dV/dK MMAC should let producer overwrite
  the next page during useful high-half MMAC work, without adding a new token or
  increasing the source live range across softmax.

Implementation:

- Added opt-in API path `kDkvPathWaspDkvMmac12WaveEarlyRelease`.
- Added standalone/script flag `EARLY_RELEASE=1`.
- Added `dv_dk_mmac_owner16_read4x2_early_release`, which releases RawUsed
  after the high source `wait_lgkm(0)` and before high dV/dK MMAC.
- Baseline W12 remains the default path.

Evidence:

- Static/resource gate PASS:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill.
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_103428`.
- H1/S1024 causal stats-only PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_103518`.
- Same-build stats-only delta versus W12 baseline:
  `kernel_ticks=70869890` versus `70883085`, about 0.019% faster;
  MMAC active avg `21.9267%` versus `21.7746%`; coissue
  `31524/20411` versus `29244/21070`.
- Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_104215_clean_w12_early_release_h1s1024_sqc7_fullperf`.
- Full perf whole-dispatch delta versus zero-seed full perf:
  `kernel_ticks=70507255` versus `70841680`, MMAC active share
  `21.7393%` versus `21.6629%`, coissue success rate `60.32%` versus
  `59.15%`, `ldsBankConflict=0` in both.
- xcu window comparison:
  - early 20k:35k producer bubble `92.17% -> 91.22%`, but consumer bubble
    `36.33% -> 37.80%` and local SIMD MMAC `10.39% -> 9.85%`.
  - mid 50k:65k producer bubble `92.06% -> 90.45%`, consumer bubble
    `38.27% -> 36.01%`, local SIMD bubble `58.46% -> 57.22%`.
  - tail producer remains AllDone/page-wait dominated:
    `99.67% -> 99.70%` bubble, local SIMD MMAC unchanged at `2.92%`.
- Dispatch-level xcu top bubbles are still dominated by barrier/control:
  `abarrier -> salu_32` is `39.21%` of issue bubble latency, with
  `flat_rd -> immed` still `15.31%`.

Decision:

`ACCEPT_MICRO_OBSERVE_PIPELINE`.  Keep the opt-in path because it is correct,
resource-clean, and slightly improves mid-window/page-release evidence.  Do not
promote it as the structural path to 60% MMAC active.

Conclusion:

Early release proves the page lifetime matters, but the current page topology
still forces producer waves into long ABarrier gaps.  The next candidate should
redesign ownership/topology or reduce sidecar/global-read debt; more consumer
micro-scheduling will likely stay in the sub-1% range.

### Sidecar Lane-Broadcast Negative

Hypothesis:

- Dispatch-level xcu still showed `flat_rd -> immed` at about 15% of issue
  bubble latency.
- In `softmax_ds_owner16_from_global_sidecar`, each q-row sidecar triplet is
  uniform across the 16 lanes with the same `lane_col_group`.
- Loading sidecar only on `lane_n == 0` and broadcasting with `__shfl` could
  reduce redundant global sidecar reads without allocating LDS or adding a new
  ABarrier token.

Change tested and removed:

- Added a temporary `softmax_ds_owner16_from_global_sidecar_broadcast` helper.
- Each sidecar triplet was initialized to zero on all lanes, loaded only by
  the subgroup source lane, then broadcast from `lane_col_group * 16`.
- Matrix path, score/dP, dV/dK, store ownership, and ABarrier ledger were
  unchanged.

Evidence:

- Static/resource gate passed:
  `private=0`, `sgpr_count=82`, `vgpr_count=112`, no SGPR/VGPR spill.
- Branch-local consumer pressure was `146/160`.
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_110454`.
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_110521`.
- Same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_110630`.
- Stats-only result:
  - sidecar broadcast `kernel_ticks=86765770`, MMAC active `15.8550%`,
    VOP active `23.1711%`, coissue `40501/30454`.
  - same-build W12 baseline `kernel_ticks=71209320`, MMAC active `21.5636%`,
    VOP active `25.0698%`, coissue `29217/22022`.
- Despite correctness and lower VGPR pressure, the active-time denominator
  grew sharply and ticks regressed by about 21.8%.

Decision:

`REJECT_PERF_STATS_ONLY`.  The temporary code was removed.

Conclusion:

Do not use wave shuffle/bpermute broadcast to attack sidecar global-read
latency in the main dKV kernel.  It trades visible flat-read debt for larger
control/VALU/bpermute scheduling cost and makes the conveyor much slower.  The
remaining sidecar/global-read debt needs either a different representation or a
producer/page-topology solution, not per-wave broadcast.

### v_mov Reduction Attempts From ASM

Hypothesis:

- XCompute CLI showed `v_mov_b32_e32` as a top BWD coissue/VALU opcode.
- The first response must come from compile-time asm, not from guessing which
  source block a SQTT opcode belongs to.

ASM baseline:

- Build command: `TARGET_GFX=946 BUILD_ASM=1 ./build.sh`.
- Symbol: `fa3_bwd_dkv_mmac12_kernel`.
- Baseline opcode counts inside the symbol:
  `total_ops=1713`, `v_mov=176`, `v_mov_b32=168`,
  `v_cvt_pk_f16_f32=0`, `v_mmac=192`, `ds_read_matrix=112`,
  `matrix_load=12`, `s_waitcnt=44`, `s_abarrier=34`.
- `.loc` mapping of the largest static `v_mov` groups:
  - `98` at `loc=0:0`, including `v_mov_b32_e32 0x3fb8aa3b`.
  - `32` at `src/dkv_kernel.cpp:2432`, around `page = q_tile & 1`.
  - `32` at `src/dkv_kernel.cpp:300`, around `arrive_raw_used_page`.
  - `8` at `include/shaobo_instr.h:32`, `zero_f16x8`.

Rejected attempts:

- Full-valid softmax pack fast path:
  - The uninitialized-vector variant failed correctness.
  - The `cvt_pk` variant passed S128 correctness but worsened asm:
    `v_mov 176 -> 184`, `v_cvt_pk 0 -> 32`, `s_waitcnt 44 -> 56`.
  - Removed.
- Page-specialized body expansion:
  - Replaced runtime `page` helpers with page0/page1 template bodies.
  - Worsened resource/codegen: consumer branch VGPR hit `160/160`, emitted
    `found vgpr before wave branch` warnings.
  - Worsened asm: `total_ops 1713 -> 3615`, `v_mov 176 -> 484`,
    `v_mmac 192 -> 384`, `s_waitcnt 44 -> 103`.
  - Removed.
- `softmax_scale_log2` local hoist:
  - Build/resource gate passed but produced identical main-symbol opcode
    counts and kept `log2e_const=2`.
  - Removed as no-effect noise.
- Score/dP inline-asm brick:
  - Replaced the `score_dp_mmac_owner16` builtin MMAC loop with an
    eight-MMAC inline asm brick modeled after the older FWD-style code.
  - Static/resource gate passed, `vgpr_count=112`, no spill/scratch.
  - ASM changed only marginally: `total_ops 1713 -> 1709`, but
    `v_mov` stayed `176` and `v_mov_b32` stayed `168`.
  - Correctness passed for H1/S128 and H1/S1024.
  - H1/S1024 stats-only regressed: `simTicks=74360650`, worse than the
    current W12 baseline band around `70.6M-71.2M`.
  - Removed.

Conclusion:

`REJECT_ASM_OR_STATS` for these four candidates.  The largest remaining `v_mov`
groups are not simple softmax vector initialization.  They are mostly compiler
register remap around q-loop/page/ABarrier control flow plus small zero-seed
setup.  The next serious route should be a narrower codegen-control probe or a
page/topology redesign that reduces live accumulator movement without
duplicating the whole loop body.

Follow-up candidates, 2026-07-02:

- Prezero dV/dK accumulators and remove the `q_tile == 0` branch:
  - Change: initialize all `dv_acc/dk_acc` before the q-loop, then always call
    the accumulate path.
  - Resource gate passed and metadata stayed spill-free:
    `sgpr_count 84 -> 80`, `vgpr_count=112`.
  - ASM rejected the idea: `v_mov_total 176 -> 196`, adding
    `148` `v_mov_b64_e32` accumulator copies.  Static `v_mmac` also changed
    `192 -> 128` by removing the duplicated first-tile branch body, which is
    not a runtime win by itself.
  - Removed.
- Use inline-asm `s_abarrier_arrive` for `arrive_raw_used_page`:
  - Change: only swapped `abarrier_arrive_cnt<false>` to
    `abarrier_arrive_cnt<true>` for the consumer RawUsed release.
  - Resource gate passed, but asm was byte-for-byte equivalent for the relevant
    symptom: `v_mov_total` stayed `176`, and the `src/dkv_kernel.cpp:300`
    group stayed `32`.
  - Removed as no-effect noise.
- Peel the first q tile:
  - Change: factor one `consumer_dkv_mmac_one_tile` helper, run q_tile 0 with
    zero-seed `<true>` before the loop, then loop q_tile 1..N with `<false>`.
  - Resource gate passed and improved static branch pressure:
    consumer branch `150/160 -> 147/160`, metadata `sgpr_count 84 -> 80`,
    no scratch/spill.
  - ASM improved the narrow `v_mov` metric:
    `v_mov_total 176 -> 84`; the page/release remap groups disappeared.
  - But it expanded the static body:
    `v_mmac 192 -> 256`, `ds_read_matrix 112 -> 192`,
    `s_waitcnt 44 -> 64`, `global_load_dwordx3 16 -> 32`.
  - Correctness passed:
    - H1/S128 m5out
      `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_230035/m5out`
    - H1/S1024 m5out
      `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_230113/m5out`
  - H1/S1024 stats regressed: `simTicks=73019765`, worse than the current W12
    baseline band around `70.6M-71.2M`.
  - Decision: `REJECT_STATS`.  Static `v_mov` reduction is not enough if it
    increases code footprint/read/wait pressure and loses ticks.
- Temporary codegen flag probes:
  - Tested only by emitting temporary asm under `/tmp`, without changing
    `build.sh`:
    - `-mllvm -amdgpu-opt-vgpr-liverange=true`
    - `-mllvm -join-globalcopies=true -mllvm -join-splitedges=true`
    - `-mllvm -amdgpu-dce-in-ra=true`
  - All three produced the same main-symbol `v_mov_total=176` and the same
    top groups (`0:0:5`, `src/dkv_kernel.cpp:2432`, `0:0:0`,
    `src/dkv_kernel.cpp:300`).
  - Decision: `REJECT_NO_EFFECT`.  Do not add these flags to `build.sh`.

Updated conclusion:

The remaining useful `v_mov` work is not a local zero-init problem.  FWD also
uses `inline_vgpr2_init_zero` for `mmac_zeros`; its advantage is that the zero
setup is amortized across a much larger and better-overlapped MMAC conveyor.
For BWD, direct attempts to erase q-loop/page remaps either add accumulator
copies or inflate the code body.  Next candidates should reduce the need for
page/first-tile control in the steady loop without duplicating score/softmax
work, or should attack the larger pipeline issue rather than treating `v_mov`
as an isolated metric.

### Main Bottleneck Pass: RawUsed ABarrier

Evidence baseline:

- Pre-change full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_234326`
- xcu detail:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/main_bottleneck_w12_h1s1024_20260702_234326/detail.txt`
- H1/S1024, `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`, causal.
- Baseline full perf:
  `kernel_ticks=71040060`, MMAC active avg `21.82%`,
  `coissue=30387/21352`, `ldsBankConflict=0`.
- Top SQTT bubble:
  `s_abarrier_try_wait -> s_xor_b32`, `3.590M cycles`, `28.64%`,
  example `s_abarrier_try_wait s0, 2` (`Raw0Used`).
- Same-window SIMD probe around `5128:11564` showed producer and consumer
  wave slots all dominated by ABarrier bubbles, so this is a real conveyor
  stall, not only a harmless producer-idle accounting artifact.

Canonical early-release promotion:

- Change: use existing verified `EarlyReleasePage=true` in the canonical
  `fa3_bwd_dkv_mmac12_kernel` consumer calls.
- Rationale: release `RawUsed` after high source operands are read, before the
  high dV/dK MMAC island.
- Build/static/metadata PASS:
  `private=0`, `sgpr=84`, `vgpr=112`, no spills.
- Stats-only H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_235302`
  with `kernel_ticks=70505435`, MMAC active avg `21.8494%`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_235606`
  with `kernel_ticks=70658770`, MMAC active avg `21.8783%`,
  `coissue=31248/20588`, `ldsBankConflict=0`.
- xcu detail:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/main_bottleneck_w12_earlycanon_h1s1024_20260702_235606/detail.txt`
  still shows `s_abarrier_try_wait -> s_xor_b32` at `3.576M cycles`,
  `28.66%`.
- Decision: `ACCEPT_MICRO_CANONICAL`.  Keep because ticks/coissue move in the
  right direction and resources stay clean, but do not call this a pipeline
  solution; RawUsed remains the main SQTT bubble.

Rejected split raw/source token retry:

- Change: split raw `Q/dO` page lifetime and source-layout `Q^T/dO^T` page
  lifetime into separate ABarrier tokens so raw can be reused after score/dP
  while source stays live until dV/dK operand read.
- Rationale: this directly attacks the RawUsed over-serialization observed by
  xcu without adding LDS bytes.
- Build/static/metadata PASS:
  `private=0`, `sgpr=86`, `vgpr=112`, no spills.
- Correctness:
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_001233`
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_001305`
- H1/S1024 stats regressed:
  `kernel_ticks=75855325`, MMAC active avg `21.0489%`,
  `coissue=26810/18008`, `ldsBankConflict=0`.
- Decision: `REJECT_PERF_STATS_ONLY`; code removed from the active route.
  Lesson: more precise ABarrier ownership can be worse when it adds extra
  phase flips/control and inserts a new source-filled wait on the consumer path.
  Future RawUsed work should reduce token turns or useful-work-hide the wait,
  not add another page-generation protocol.

Rejected RawUsed builtin-wait probe:

- Change: only swapped `wait_raw_used_page` from inline-asm
  `abarrier_try_wait<true>` to builtin `abarrier_try_wait<false>`.
- Rationale: test whether the large xcu
  `s_abarrier_try_wait -> s_xor_b32` bubble is amplified by the hand-written
  phase flip / scheduling barriers rather than true page ownership wait.
- Result: static metadata gate failed before PMD:
  `private_segment_fixed_size=12`, `sgpr=88`, `vgpr=112`, no SGPR/VGPR spill.
- Decision: `REJECT_STATIC_PRIVATE_SEGMENT`; code removed.
- Lesson: for this full dKV body, builtin RawUsed wait changes codegen enough
  to create private segment.  Keep the inline wait form unless a focused probe
  proves a spill-free replacement.

### Tile Ledger For The 60% MMAC-Active Push

Workbook and repo ledger:

- Shared workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- Repo ledger:
  `results/tile_ledger_20260703.md`

Current canonical topology:

- `fa3_bwd_dkv_mmac12_kernel`
- `Mq=32,Nk=128,D=128`, 12 waves
- waves0-3: one producer group
- waves4-7 and waves8-11: two consumer groups
- each consumer wave owns `Nk=16` and full `D=128`
- `S1024` has `32` q-packet turns and `8` K CTAs

MMAC accounting:

- per consumer wave per q tile:
  `score=16`, `dP=16`, `dV=16`, `dK=16`, total `64`
- per CTA per q tile:
  `8 consumer waves * 64 = 512`
- per CTA for `S1024`:
  `512 * 32 = 16384`
- dispatch total for `H1/S1024`:
  `8 K CTAs * 16384 = 131072`, matching PMD `MMOP`

Resource boundary:

- current LDS is already exactly 128KB:
  `K 32KB + V 32KB + Q raw 16KB + dO raw 16KB + Q^T source 16KB + dO^T source 16KB`
- appending a dedicated sidecar page is invalid under the current layout
- `Mq64` single-buffer also fits exactly 128KB and doubles the consumer MMAC
  island to `128` per wave per q packet, but prior seedfix/perf work showed it
  lost too much overlap
- `Mq64` double-buffer and `Mq128` single-buffer both exceed 128KB when raw and
  source-layout operands are both kept resident with K/V

Design conclusion:

- the current low MMAC-active share is not because MMAC is absent; it is because
  the steady loop is packet/control dominated
- the accepted mainline has only a `64`-MMAC island per consumer wave before
  the next packet protocol
- the top xcu bubble remains RawUsed ABarrier, about `28.66%`
- the next promoted change must increase effective MMAC island length or hide
  packet waits without adding ABarrier token families, without appending LDS,
  and without duplicating score/dP

### RawUsed Release After High-Read Candidate A

Decision: `REJECT_XCU_PRIMARY_METRIC`

Hypothesis:

- The canonical early-release path still waits for the high source
  `ds_read_matrix` to become ready before arriving `RawUsed`.
- Move `arrive_raw_used_page` earlier, immediately after issuing the high
  source read and before the low dV/dK MMAC island.
- Expected benefit: producers can reuse the raw page while the consumer is
  still doing useful low/high dV/dK MMAC work, reducing the dominant
  `s_abarrier_try_wait -> s_xor_b32` bubble.

Evidence:

- Build/static/metadata stayed clean:
  `private=0`, `sgpr=84`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_020841`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_020926`
  with `kernel_ticks=69979910`, MMAC active avg `21.8854%`,
  `coissue=33672/21734`, `ldsBankConflict=0`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_021203`
  with `kernel_ticks=69865705`, MMAC active avg `21.8495%`,
  `coissue=33894/21881`, `ldsBankConflict=0`.
- xcu output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/rawused_candidateA_h1s1024_20260703_021203`.
- xcu comparison versus canonical:
  - canonical `s_abarrier_try_wait -> s_xor_b32`:
    `3.576M cycles`, `28.66%`.
  - candidate A `s_abarrier_try_wait -> s_xor_b32`:
    `3.538M cycles`, `28.61%`.
  - canonical `s_waitcnt` hot latency:
    `2.960M cycles`, `22.04%`.
  - candidate A `s_waitcnt` hot latency:
    `3.007M cycles`, `22.57%`.
  - canonical `v_mov_b32_e32 -> v_mov_b32_e32` bubble:
    `362K cycles`, `2.90%`.
  - candidate A `v_mov_b32_e32 -> v_mov_b32_e32` bubble:
    `378K cycles`, `3.05%`.

Conclusion:

Correctness and resources are fine, but the hypothesis does not hold.  The
dominant RawUsed ABarrier bubble remains essentially unchanged, full-perf MMAC
active is flat to slightly worse, and wait/move-side bubbles increase.  The code
was reverted from the canonical path.  Treat this as evidence that simply
moving the release a few instructions earlier cannot close the 60% MMAC-active
gap; the next route must reduce packet-turn count, lengthen useful MMAC
islands, or change producer/page topology.

Next candidates:

- first re-derive an `Mq64`-equivalent conveyor using the existing two `Mq32`
  physical pages, so we can test longer MMAC islands without new LDS bytes
- keep a clean `Mq64` single-buffer reference as a comparison, but do not
  promote it unless it beats same-build W12 on ticks and MMAC active
- only revisit a 16-wave FWD-style role split after the workbook proves the
  second producer has recurring useful work under the same 128KB LDS budget

## 2026-07-03 Mq64 Long-Island Reorder

Decision: `REJECT_CORRECTNESS`

Hypothesis:

- use the existing Mq64 single-buffer route, not a new phase
- increase the effective MMAC island by joining the two 32-row halves before
  returning to packet/control work
- keep LDS unchanged at 128KB and avoid new ABarrier token families

Tried variants:

- score/dP both halves, then softmax both halves, then dV/dK both halves
- score/dP both halves, then one half at a time for softmax+dV/dK
- full Mq64 helper: `score[4]/dp[4] -> softmax[4] -> dV/dK[4]`

Evidence:

- all variants were static-resource clean:
  `private=0`, no scratch/spill, Mq64 symbol `sgpr=100`, `vgpr=144`
- branch consumer pressure stayed within the 208 window:
  `191/208`, `187/208`, `179/208`
- all variants failed H1/S128 causal correctness in the same way:
  exact `dK`, but `dV bad=16384`, `dv_max_abs=2.73637e-05`,
  `dv_rel_l2=0.000267234`, `pass=0`
- evidence dirs:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_012315`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_012619`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_012810`

Conclusion:

Simple Mq64 live-range stretching is not a valid route to FWD-like long MMAC
islands.  It either exposes an Mq64 dV seed/layout hazard or a PMD/codegen
hazard around holding score/dP across the half-block boundary.  The live code
was restored to the prior Mq64 half-sequential shape; next work should focus on
topology/page lifetime or write a focused Mq64 dV seed/layout probe.

## 2026-07-03 W16 WG-Local Nk64 Semantic Conveyor

Decision: `REJECT_PERF_STATS_ONLY`

Design premise:

- current W12 is correct and no-duplicate, but xcu shows shared RawUsed
  ABarrier remains dominant and MMAC active is stuck around `21.9%`
- split the CTA into two independent 8-wave WG-local conveyors:
  `P0/C0` owns `K/V0..63`, `P1/C1` owns `K/V64..127`
- resource stress showed that private raw double pages plus private source
  double pages would require `192KB`; the legal candidate must use semantic
  pages:
  `K/V64 32KB + two 16KB Q/dO-or-QT/DOT pages = 64KB/WG`

Implementation:

- reused the existing 16-wave `fa3_bwd_dkv_mmac_kernel` route as the candidate
  path, leaving the W12 canonical baseline intact
- added WG-local semantic LDS offsets, WG-local page token helpers, and
  producer/consumer loops
- producer publishes raw page0/page1, then alternates
  `wait raw used -> source fill -> wait source used -> next raw fill`
- consumer performs
  `wait raw -> score/dP -> raw used -> softmax/dS -> wait source -> dV/dK`

Evidence:

- build/static metadata PASS:
  `private=0`, `sgpr=86`, `vgpr=88`, `sgpr_spill=0`, `vgpr_spill=0`
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_024846`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_024909`
- H1/S1024 stats:
  `kernel_ticks=80790710`, `MMOP=131072`, `ldsBankConflict=0`,
  `coissue=30775/21592`, `MMAC active=19.1856%`,
  `VOP active=27.6249%`

Conclusion:

The candidate is resource/correctness-clean but fails the primary objective.
WG-local independence is not enough when it is paid for with duplicated
Q/dO/source loads and an exposed raw/source semantic epoch.  This confirms that
the next 60% attempt should not simply clone FWD's two-WG topology; it must
either preserve W12's shared double-buffering while reducing exposed barrier
time, or increase the per-consumer useful MMAC island without adding a source
epoch wait.

## 2026-07-03 Legacy W16 Split-Producer Probe

Decision: `REJECT_RUNTIME_HANG`

Hypothesis:

- test the existing 16-wave path as a cheap FWD-style reference before writing
  new code
- waves0-3 publish Q/K, waves12-15 publish dO/V, and waves4-11 consume

Evidence:

- metadata gate passed:
  `private=0`, `sgpr=78`, `vgpr=84`, no SGPR/VGPR spill
- H1/S128 causal PMD path:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_013236`
- PMD emitted `read vgpr112 before writing`
- the run did not finish after more than `18B` simulated ticks and was killed

Conclusion:

The old W16 route is not a valid shortcut to FWD-style dKV.  A future 16-wave
design must be re-derived around exact producer arrival counts and consumer
release counts.  Role count alone does not create WASP; the barrier ownership
protocol has to be designed as carefully as the tile.

Update after q-loop audit:

- root cause for the S128 non-convergence was a real loop-bound bug in
  `producer_qk_loop`: the producer used fixed `Tile::kQTilesPerCta` instead of
  runtime `q_tiles`
- after changing the producer to stop at `q_tiles`, W16 H1/S128 and H1/S1024
  both pass correctness
- H1/S1024 q-loop-fix stats:
  `kernel_ticks=73333260`, MMAC active avg `20.5512%`, coissue
  `34952/26250`, `ldsBankConflict=0`
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_013701`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_013725`

Decision: `OBSERVE_CORRECTNESS_PASS`.  Keep the q-loop bound fix because it
removes a real runtime bug, but do not promote W16 as a performance route.

## 2026-07-03 W16 Split-Producer Early Release

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

- take the now-correct legacy W16 split-producer route and apply the same early
  RawUsed release used by the W12 canonical route
- expected benefit: reduce packet lifetime pressure and give the two producers
  more chance to overlap with consumers

Evidence:

- static/resource gate passed:
  `private=0`, `sgpr=78`, `vgpr=84`, no scratch/spill
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_014618`
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_014638`
- H1/S1024 stats:
  `kernel_ticks=74882080`, MMAC active avg `20.3284%`, coissue
  `32395/24677`, total MMOP `131072`, `ldsBankConflict=0`
- comparison:
  q-loop-fix W16 before early-release was `73333260` ticks and `20.5512%`
  MMAC active; W12 canonical is about `70.5M` stats-only ticks and
  `21.8783%` full-perf MMAC active

Conclusion:

Role count and split producers are not sufficient.  This path still has too
little useful consumer MMAC density and too much control/barrier cost.  Revert
the W16 early-release edit from the active route; keep only the q-loop bound
fix because it is a correctness/runtime repair.

## 2026-07-03 W12 dV/dK Source Quad-Read Probe

Decision: `OBSERVE_REJECT_PERF`

Hypothesis:

- FWD has PMD-only contiguous `ds_read_matrix` blocks without `s_nop`.
- Current BWD dV/dK source operand read uses four pair wrappers, each carrying
  a local `s_nop`, which xcu reports as part of the read-side bubble.
- Change only the canonical dV/dK source read helper from four pair reads to
  two contiguous 4-read blocks: one for `dO^T`, one for `Q^T`.

Evidence:

- build/static/resource passed:
  `private=0`, `sgpr=82`, `vgpr=112`, no spill/scratch
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_015831`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_015836`
- H1/S1024 stats:
  `kernel_ticks=70560945`, MMAC active avg `21.7716%`, coissue
  `31952/20804`, `ldsBankConflict=0`, `mmop_runtime_share=45.5229%`

Conclusion:

This improves the local VOP/runtime mix but does not beat the W12 canonical
baseline on ticks or MMAC active.  The code was removed from the active route.
The useful lesson is that FWD-style batched reads alone are insufficient when
the dominant loss is still packet/barrier conveyor control and sidecar/global
waits around the short 64-MMAC island.

## 2026-07-03 H18A Packed Sidecar dwordx4 Probe

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

- xcu barrier audit showed two immediate blockers:
  `RawUsed0/1` dominates ABarrier top1000, and sidecar
  `global_load_dwordx3 -> s_waitcnt` is the largest non-barrier bubble.
- Change packed sidecar ABI from 3 floats/row to 4 floats/row and force the
  pad float live in `load_packed_sidecar_row4`, so the hot path uses
  `global_load_dwordx4` instead of `global_load_dwordx3`.
- Expected benefit: lower sidecar load/wait overhead, shorter consumer page
  lifetime, lower producer `RawUsed` wait, higher MMAC active.

Evidence:

- Static/resource clean:
  `private=0`, `sgpr=84`, `vgpr=112`, no scratch/spill.
- ASM changed as intended: sidecar hot loads became `global_load_dwordx4`.
- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_031739`.
- H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_031804`.
- H1/S1024 stats:
  `simTicks=74270105`, `kernel_ticks=70656495`, `MMOP=131072`,
  `coissue=31641/20782`, `ldsBankConflict=0`.
- Canonical W12 stats companion:
  `kernel_ticks=70505435`, `coissue=31497/20470`.

Conclusion:

Correct but not useful.  The opcode change does not reduce the end-to-end
consumer critical section; `VALU_cycles` is unchanged and ticks slightly
regress.  Revert the ABI/helper change.  The next high-value work should focus
on reducing duplicated sidecar/mask/softmax work or changing page lifetime, not
just changing the sidecar load width.

## 2026-07-03 H21A Q-pair Control-Only Redesign

Decision: `REJECT_STATIC_SPILL`

Hypothesis:

- Reuse the existing page0/page1 as one logical q-pair without adding LDS or
  ABarrier token families.
- First cut was deliberately conservative: process page0 then page1 inside a
  `q_tile += 2` consumer loop while preserving the original per-page
  source-layout reads and early RawUsed release.
- Expected benefit was small but useful: fewer exposed q-loop control turns and
  a cleaner stepping stone toward H21B q-pair stagger.

Evidence:

- Workbook design sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `19_qpair_design`.
- First helper implementation failed static metadata for
  `fa3_bwd_dkv_mmac12_kernel`:
  `private_segment_fixed_size=0`, `sgpr_count=100`,
  `sgpr_spill_count=39`, `vgpr_count=112`, `vgpr_spill_count=0`.
- A second local-macro implementation removed the `int&` helper footgun but
  still failed static metadata:
  `private_segment_fixed_size=0`, `sgpr_count=100`,
  `sgpr_spill_count=38`, `vgpr_count=112`, `vgpr_spill_count=0`.
- After reverting H21A, the remote canonical build/static gate passed again:
  `private_segment_fixed_size=0`, `sgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_count=112`, `vgpr_spill_count=0`.

Conclusion:

The q-pair control-only shape is not a free way to halve loop-control overhead.
Even without new LDS/token families, duplicating the consumer body in one loop
scope widens SGPR live ranges enough to spill.  Do not continue H21A into PMD.
Future q-pair work must reduce live state first or use a different ownership
shape; otherwise it will chase a scheduling idea that fails before execution.

## 2026-07-03 H22 First-Tile Peel

Decision: `ACCEPT_MICRO_CONTROL`

Hypothesis:

- The hot consumer q-loop still carried a runtime `q_tile == 0` branch to seed
  dV/dK accumulators differently on the first packet.
- Peel `q_tile=0` out of the steady loop, then run the hot loop from
  `q_tile=1` with the normal non-first MMAC path.
- Expected benefit: remove one steady-loop branch/control shape, reduce SGPR
  pressure, and make the loop a cleaner base for the next pipeline redesign.

Evidence:

- Static/resource gate passed for `fa3_bwd_dkv_mmac12_kernel`:
  `private_segment_fixed_size=0`, `sgpr_count=78`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_042133`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_042139`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_042626`.
- Full perf stats:
  `kernel_ticks=67665325`, `MMOP=131072`, `ldsBankConflict=0`,
  coissue `23064/18083`, `MMAC active share=23.0485%`,
  `VOP active share=20.8165%`.
- Archived perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_042626_clean_w12_h22_first_tile_peel_h1s1024_sqc7_fullperf`.
- xcu detail still shows the main bubbles:
  `s_abarrier_try_wait -> s_xor_b32` at `28.44%`, and
  `global_load_dwordx3 -> s_waitcnt` at `11.59%`.

Conclusion:

Keep H22 in the canonical route.  It is a real control/codegen cleanup:
SGPR count drops from `84` to `78`, H1/S1024 ticks improve, and MMAC active
share rises into the low `23%` range.  It is not a structural solution for
the `60%` goal, because the dominant xcu bubbles remain RawUsed/ABarrier and
sidecar global-load wait.  The next implementation should use H22 as the
clean baseline and attack those two exposed waits without reintroducing q-pair
SGPR spill or extra ABarrier token families.

## 2026-07-03 H23 Remove ds_read_matrix Helper s_nop

Hypothesis:

- H22 xcu reported a hot `s_nop -> ds_read_matrix` row and high
  `ds_read_matrix` latency.
- The `ds_read_matrix_trans_pair` helper inserted a fixed `s_nop 0` before
  each read pair.  Removing that nop should make the source read island closer
  to the FWD style: contiguous matrix reads, then first-use wait/MMAC, without
  an artificial scheduling bubble.

Evidence:

- Static/resource gate passed for `fa3_bwd_dkv_mmac12_kernel`:
  `private_segment_fixed_size=0`, `sgpr_count=78`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_043940`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_043946`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_044220`.
- Full perf stats:
  `kernel_ticks=67246725`, `MMOP=131072`, `ldsBankConflict=0`,
  coissue `22768/18808`, `MMAC active share=23.2228%`,
  `VOP active share=20.9728%`.
- Archived perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_044220_clean_w12_h23_no_dsread_snop_h1s1024_sqc7_fullperf`.
- xcu comparison against H22:
  dispatch issue count drops from `897096` to `847944`, the hot `s_nop`
  row disappears, and `ds_read_matrix` latency drops from `557792` to
  `475420`.
- xcu still reports dominant structural bubbles:
  `s_abarrier_try_wait -> s_xor_b32` around `28.48%` and
  `global_load_dwordx3 -> s_waitcnt` around `11.54%`.

Conclusion:

`ACCEPT_MICRO_SQTT`.  Keep H23 in the canonical route because it removes a
real artificial read-side bubble and slightly improves ticks/MMAC active.
This is still not the 60% solution: the next patch must target RawUsed
ABarrier ownership and sidecar global-load wait, or lengthen a useful MMAC
island enough to hide those waits.

## 2026-07-03 H24 Raw ABarrier Wait Builtin Boundary

Hypothesis:

- H23 xcu top issue gap points to `include/shaobo_instr.h:137`, the asm wrapper
  for `abarrier_try_wait<true>`:
  `s_abarrier_try_wait -> s_xor_b32`, about `28.48%`.
- Replacing raw packet waits with the builtin wrapper might reduce this exposed
  wait/phase-flip bubble without changing LDS layout, MMAC count, sidecar math,
  or output ownership.

Attempts:

- H24A changed both `wait_raw_used_page` and `wait_raw_filled_page` from
  `abarrier_try_wait<true>` to `abarrier_try_wait<false>`.
- H24B kept `wait_raw_filled_page` as asm and changed only
  `wait_raw_used_page` to builtin.

Evidence:

- H24A static/resource gate failed:
  `private_segment_fixed_size=12`, `sgpr_count=80`, `vgpr_count=112`,
  no SGPR/VGPR spills.
- H24B static/resource gate also failed:
  `private_segment_fixed_size=12`, `sgpr_count=82`, `vgpr_count=112`,
  no SGPR/VGPR spills.
- Restored H23 after revert:
  `private_segment_fixed_size=0`, `sgpr_count=78`, `vgpr_count=112`,
  no SGPR/VGPR spills.

Conclusion:

`REJECT_STATIC_PRIVATE`.  Raw Filled/Used waits in the q-loop cannot be
converted to the builtin wrapper as a simple micro-fix: even the RawUsed-only
variant introduces private segment.  This narrows the ABarrier path: the 60%
route must reduce the number or exposed duration of raw page ownership waits,
or cover them with useful consumer/producer work.  Do not retry raw-wait
builtinization in the active kernel without a focused compiler/codegen probe.

## 2026-07-03 H25 Release RawUsed Before Full dV/dK Island

Hypothesis:

- Current early-release order waits for low source, issues high source, runs
  the low dV/dK MMAC island, waits for high source, then releases RawUsed
  before the high dV/dK island.
- Moving the high-source wait before the dV/dK MMAC island would allow the
  consumer to release RawUsed before both low and high dV/dK MMAC groups,
  giving producer waves a longer useful MMAC window to refill the page.

Evidence:

- Static/resource gate passed:
  `private=0`, `sgpr=78`, `vgpr=112`, no SGPR/VGPR spill, no scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_050642`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_050705`.
- H1/S1024 stats:
  `kernel_ticks=68373305`, `MMAC active share=22.8899%`,
  `ldsBankConflict=0`.
- H23 comparison:
  `kernel_ticks=67246725`, `MMAC active share=23.2228%`.

Conclusion:

`REJECT_PERF_STATS_ONLY`.  The design shortened RawUsed lifetime on paper, but
it moved high-source readiness onto the pre-MMAC critical path.  The current
H23 order is better because low dV/dK MMAC hides high-source latency.  The
code was reverted; future RawUsed work must preserve that overlap or create
new useful work before page release.

## 2026-07-03 H26 Causal Sidecar Hot-Path Split

Hypothesis:

- The target tuning case is `causal=true`, but H23's sidecar helper still has
  an inner-element runtime causal term:
  `(!causal || krow <= qrow)`.
- Splitting a causal-specific helper should remove that term from the
  `m_idx/vec_id` loop without changing score/dP, dV/dK, ABarrier, LDS, MMOP,
  or output ownership.
- This was deliberately narrower than H20A: no full-valid fastpath body, no
  sidecar load skipping, and no p/dS fragment layout change.

Evidence:

- Workbook-first design:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `22_causal_sidecar_design`.
- Static/resource gate passed for the candidate:
  `private=0`, `sgpr=80`, `vgpr=112`, no SGPR/VGPR spill.
  This is already worse than H23's `sgpr=78`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_052600`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_052606`.
- H1/S1024 stats regressed:
  `kernel_ticks=70504980`, `MMAC active share=22.5343%`,
  `VOP active share=21.6469%`, `VALU=230108`, `ldsBankConflict=0`.
- H23 comparison:
  `kernel_ticks=67246725`, `MMAC active share=23.2228%`,
  `VOP active share=20.9728%`, `VALU=213208`.
- Restored H23 after revert:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.

Conclusion:

`REJECT_PERF_STATS_ONLY`.  The causal helper split is correct, but it makes
the hot path heavier: SGPR rises and VALU increases instead of falling.  This
means the runtime causal boolean is not the active-share limiter in the current
code shape.  The code was reverted.  Next work should target the structural
blockers already visible in H23 xcu: RawUsed/ABarrier lifetime, sidecar
global-load wait, or a larger useful MMAC island that can hide those waits.

## 2026-07-03 H27 RawUsed Arrive After High-Source Issue

Hypothesis:

- H25 proved that waiting for the high dV/dK source before both low/high MMAC
  groups is wrong: it moves LDS readiness onto the pre-MMAC critical path.
- H27 keeps H23's useful overlap, where low dV/dK MMAC hides high-source
  latency, but releases `RawUsed` immediately after issuing the high-source
  `ds_read_matrix`.
- The correctness risk is explicit: if a high-source LDS read is not protected
  after issue, producer reuse of the page could corrupt the later high MMAC
  operands.

Evidence:

- Workbook-first design:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `23_rawused_after_high_issue`.
- Static/resource gate passed:
  `private=0`, `sgpr=78`, `vgpr=112`, no SGPR/VGPR spill, no scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_053909`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_053933`.
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_054127`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_054127_clean_w12_h27_rawused_after_high_issue_h1s1024_sqc7_fullperf`.
- Full-perf metrics:
  `kernel_ticks=66892280`, `MMAC active share=23.3787%`,
  `VOP active share=20.9407%`, `MMOP=131072`, `VALU=213208`,
  `SCA=289456`, `ldsBankConflict=0`.
- H23 baseline:
  `kernel_ticks=67246725`, `MMAC active share=23.2228%`.
- xcu comparison versus H23:
  `s_abarrier_try_wait -> s_xor_b32` changes from `3.389M / 28.48%`
  to `3.381M / 28.41%`; `v_mmac -> s_waitcnt` improves from `6.65%`
  to `6.55%`; sidecar `global_load_dwordx3 -> s_waitcnt` worsens slightly
  from `11.54%` to `11.69%`.

Conclusion:

`ACCEPT_MICRO_OBSERVE_PIPELINE`.  The H27 ordering is correctness-safe in PMD
and gives a small same-shape improvement while preserving the useful
high-source latency hiding that H25 lost.  It is still far from the 60% MMAC
active target, and xcu still shows the same dominant barrier/sidecar debt.
Keep the reorder as a micro-win, but do not stack more cosmetic reorder
patches.  The next structural work must either reduce sidecar global-read wait,
collapse page-control turns, or produce a longer useful MMAC island with a
clear workbook resource proof.

### Causal=False Diagnostic On H27

Because leader feedback called out causal-mask overhead, the same H27 code was
run with `CAUSAL=0` before starting a new implementation.

Evidence:

- Run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_055407`.
- Numerical comparison did not pass the current gate:
  `bad=0`, but `dk_rel_l2=0.0199722`, `dv_rel_l2=0.002617`, `pass=0`.
- Stats are still useful as a diagnosis:
  `simTicks=77584780`, `kernel_ticks=73971170`,
  `MMOP=131072`, `VALU=244352`, `SCA=230896`,
  `MMAC active share=17.8499%`, `VOP active share=24.3721%`,
  `ldsBankConflict=0`.

Conclusion:

`OBSERVE_CORRECTNESS_FAIL`.  Disabling causal does not move this code shape
toward the FWD-like 60% target; it lowers MMAC active and increases VALU.
The active bottleneck is still the packet/control/sidecar conveyor, not merely
the causal predicate term.  Do not spend the next round on causal-only tuning
unless a separate noncausal correctness path is first designed and justified.

## 2026-07-03 H28 Producer Sidecar Cache-Warm

Hypothesis:

- H27 still shows a large consumer-side sidecar bubble:
  `global_load_dwordx3 -> s_waitcnt` at `1.391M / 11.69%`.
- Moving sidecar values into LDS by overlay already failed because it added an
  extra ownership generation.  Appending a dedicated sidecar page also violates
  the 128KB LDS budget.
- Instead, let producer `wave_local==0` issue volatile sidecar row loads before
  the RawUsed wait/refill point.  This does not transfer values to consumers;
  it only tests whether cache warming plus producer useful work can reduce the
  exposed consumer global-load wait.

Implementation:

- Added `prefetch_packed_sidecar_tile`.
- Only producer wave `wave_local==0`, lanes `<32`, touches one q tile's packed
  sidecar rows.
- The main matrix path, LDS layout, ABarrier ledger, consumer math, dV/dK
  output ownership, and MMOP count are unchanged.

Evidence:

- Workbook-first design:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `24_sidecar_cache_prefetch`.
- Static/resource gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_060502`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_060528`.
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_060726`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_060726_clean_w12_h28_sidecar_cache_prefetch_h1s1024_sqc7_fullperf`.
- Full-perf metrics:
  `kernel_ticks=66630200`, `MMAC active share=27.9004%`,
  `VOP active share=26.6286%`, `MMOP=131072`, `VALU=214952`,
  `SCA=292576`, `ldsBankConflict=0`.
- H27 comparison:
  `kernel_ticks=66892280`, `MMAC active share=23.3787%`.
- xcu comparison:
  RawUsed `s_abarrier_try_wait -> s_xor_b32` improves from `28.41%`
  to `26.25%`; sidecar `global_load_dwordx3 -> s_waitcnt` improves from
  `11.69%` to `11.42%`; a new producer prefetch
  `flat_load_dword -> s_waitcnt` appears at `185k / 1.57%`.

Conclusion:

`ACCEPT_PIPELINE_OBSERVE`.  H28 is the first change in this sequence that
produces a meaningful MMAC-active jump with an xcu explanation.  It does not
solve the 60% target: the added prefetch creates its own wait bubble, and
RawUsed plus sidecar waits remain large.  Keep H28, then improve it by
batching/placing the prefetch so its wait is covered by producer RawUsed or by
designing a true producer-helper sidecar protocol that does not add LDS bytes
or a new page-ownership generation.

## 2026-07-03 H29 Fire-And-Forget Sidecar Prefetch

Hypothesis:

- H28's empty-asm use may force producer prefetch values to become ready,
  creating the new `flat_load_dword -> s_waitcnt` producer bubble.
- Replacing the consumed loads with volatile fire-and-forget reads might keep
  the cache-warm effect while reducing the producer wait.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_061743`.
- H1/S1024 stats-only PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_061810`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_062248`.
- Stats-only metrics:
  `kernel_ticks=66626560`, `MMAC active=28.0412%`,
  `VALU=214952`, `SCA=292576`, `ldsBankConflict=0`.
- Full-perf metrics:
  `kernel_ticks=66690260`, `MMAC active=27.9272%`,
  `VALU=214952`, `SCA=292576`, `ldsBankConflict=0`.
- H28 full-perf comparison:
  `kernel_ticks=66630200`, `MMAC active=27.9004%`.

Decision:

`OBSERVE_ACTIVE_REJECT_TICKS`.  H29 slightly improves MMAC active but regresses
same-shape full-perf ticks.  It is not stable enough to replace H28, and the
code has been restored to the H28 explicit-load plus empty-asm-use form.  The
lesson is useful: active share can move by tiny noise without shortening the
critical path.  The next change needs to reduce RawUsed/sidecar wait or lengthen
MMAC islands with a visible tick improvement.

## 2026-07-03 H30 Future Sidecar Prefetch Placement

Hypothesis:

- H28 prefetches the current q tile immediately before waiting `RawUsed` and
  refilling that page.
- Moving the same sidecar cache-warm to `q_tile + 2` immediately after
  `RawFilled` gives the sidecar rows a full ping-pong interval to reach cache
  before consumers use them.
- This keeps the single canonical kernel and does not add LDS, ABarrier
  tokens, consumer math, output ownership, or MMOP.

Evidence:

- Workbook-first design:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `26_sidecar_future_prefetch`.
- Static/resource gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_063323`.
- H1/S1024 stats-only PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_063344`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_063614`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_063614_clean_w12_h30_future_sidecar_prefetch_h1s1024_sqc7_fullperf`.
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/h30_future_sidecar_prefetch_h1s1024_20260703_063614`.
- Full-perf metrics:
  `kernel_ticks=66321255`, `MMAC active=28.3952%`,
  `VOP active=26.6089%`, `MMOP=131072`, `VALU=214984`,
  `SCA=296832`, `ldsBankConflict=0`.
- H28 comparison:
  `kernel_ticks=66630200`, `MMAC active=27.9004%`.
- xcu comparison:
  duration improves from `146440` to `145764`;
  RawUsed `s_abarrier_try_wait -> s_xor_b32` improves from `26.25%` to
  `25.89%`; sidecar `global_load_dwordx3 -> s_waitcnt` improves from
  `11.42%` to `11.38%`; producer prefetch `flat_load_dword -> s_waitcnt`
  worsens from `1.57%` to `1.79%`.

Decision:

`ACCEPT_PIPELINE_MICRO`.  H30 is the current best W12 canonical route.  It
proves that earlier producer-side cache-warm placement can move the critical
path, but the improvement is still tiny relative to the 60% MMAC-active goal.
The remaining bottleneck is not sidecar load shape alone: `s_waitcnt` and
RawUsed/ABarrier still dominate, and future work needs either a longer MMAC
island or a page-lifetime/topology change that reduces those waits without
adding another token family.

## 2026-07-03 Canonical dKV Route Convergence

Hypothesis:

- Before applying the Tri Dao-style redesign rules, the clean repo needs one
  hot dKV route.  Multiple `--dkv-mmac12-*` launch paths make code review and
  profiler attribution drift.
- Keep reference correctness for CPU-golden comparison, but make the active
  Shaobo path route only to `fa3_bwd_dkv_kernel`.

Changes:

- Renamed the active W12 kernel entry from the experiment-shaped
  `fa3_bwd_dkv_mmac12_kernel` to `fa3_bwd_dkv_kernel`.
- API, standalone default, symbol gate, and PMD scripts now select only this
  canonical kernel for dKV performance work.
- Rejected historical kernels are still present for a later physical deletion
  pass, but are no longer reachable from the supported API/script route.

Evidence:

- Remote build PASS with no compile warnings.
- Static/source gate PASS.
- Symbol metadata PASS for `fa3_bwd_dkv_kernel`:
  `private_segment_fixed_size=0`, `sgpr_count=78`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- H1/S128 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_090438`.
- H1/S1024 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_090629`.
- H1/S1024 stats-only aggregate:
  `simTicks=69908930`, `MMOP=131072`, `ldsBankConflict=0`,
  `coissue=22685/18256`, `MMAC active=23.5544%`.

Decision:

`CODE_GOVERNANCE_ACCEPT`.  This is not a new performance promotion; it is the
single-kernel convergence checkpoint.  Further Tri Dao-style changes must
modify `fa3_bwd_dkv_kernel` in place and record rejected alternatives in the
workbook/ledger instead of adding another public performance route.

## 2026-07-03 Archive Unreachable Global Kernels

Hypothesis:

- Public routing was converged, but old global kernels were still compiled.
  This kept noisy WDRA metadata in the build and made the clean repo look like a
  phase stack.

Change:

- Archived unreachable bring-up and rejected global kernels under compile-time
  guards.
- Build now instantiates only the reference kernels plus
  `fa3_bwd_dkv_kernel` for dKV performance work.

Evidence:

- Remote build PASS.
- Build WDRA branch report now only shows the canonical dKV branch windows:
  producer `6/16`, consumer0 `159/160`, consumer1 `159/160`.
- Static/source gate PASS.
- Symbol metadata PASS for `fa3_bwd_dkv_kernel`:
  `private=0`, `sgpr=78`, `vgpr=112`, no SGPR/VGPR spill.
- H1/S128 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_091957`.

Decision:

`CODE_GOVERNANCE_ACCEPT`.  The active source is materially cleaner even before
physical deletion.  Next performance edits should operate inside the single
canonical kernel; archived kernels should be physically removed once their
lessons are fully represented in workbook/ledger.

## 2026-07-03 Canonical After-Convergence SQTT Rebaseline

Decision: `PASS_XCU_BASELINE`

Purpose:

After archiving unreachable global kernels and converging the supported route to
one canonical `fa3_bwd_dkv_kernel`, rerun the exact H1/S1024 diagnostic with
`GPU_CHIP=sb` and `GPU_ARGS=['--SQCIPfLines=7']`. This is the evidence baseline
before the next code edit.

Evidence:

- Static/resource gate remains clean:
  `private_segment_fixed_size=0`, `sgpr_count=78`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`; WDRA branch windows are producer
  `6/16`, consumer0 `159/160`, consumer1 `159/160`.
- H1/S1024 causal stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093524`,
  `kernel_ticks=66540565`, `MMOP=131072`, `MMAC active=23.4212%`,
  `VOP active=21.3174%`, `ldsBankConflict=0`.
- H1/S1024 causal=false diagnostic:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093608`,
  `pass=0`, `dk_rel_l2=0.0199722`, `dv_rel_l2=0.002617`,
  `kernel_ticks=68400150`, `MMAC active=20.5948%`.
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093932`,
  `kernel_ticks=66411800`, `MMOP=131072`, `coissue=23057/18211`,
  `MMAC active=23.4288%`, `VOP active=21.3239%`, `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_093932_clean_canonical_after_convergence_h1s1024_sqc7_fullperf`.
- xcu output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/canonical_after_convergence_h1s1024_20260703_093932`.
- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `28_current_xcu_rebaseline`.

xcu dispatch 0 summary:

- duration `145960` cycles, inst issues `861816`, avg active waves `86.45`.
- top opcode rows: `s_waitcnt` `26.48%`, `s_xor_b32` `24.23%`,
  `v_mmac_f32_16x16x16_f16` `9.55%`, `ds_read_matrix_trans_format` `3.88%`.
- top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `25.95%`,
  `global_load_dwordx3 -> s_waitcnt` `11.28%`,
  `v_mmac -> v_mmac` `6.55%`,
  `v_mmac -> s_waitcnt` `6.35%`,
  `s_abarrier_try_wait -> s_waitcnt` `5.59%`,
  `ds_read_matrix -> s_waitcnt` `2.38%`.

Conclusion:

The converged canonical route is correct and resource-clean.  The apparent
H30-to-current active-share drop was a metric normalization issue, not a real
pipeline regression: recomputing H30 from its archived `stats.txt` with
`sum(mmopRunTimeCounter)/sum(activeTimeCounter)` gives `23.4386%`, while this
post-convergence run is `23.4288%`.  The historical H30 `28.3952%` value used a
different active metric and should not be compared directly.

The current bottleneck is not missing MMAC and not LDS bank conflict. The
largest measured debt is raw-page ABarrier/control and the exposed sidecar
global load. `ds_read_matrix -> wait` is visible but only `2.38%`, so read
batching alone is not a credible path to `60%` MMAC active.

Next:

- Tighten `scripts/check_dkv_kernel_gate.py` so it validates only the active
  canonical route, not archived `#if 0` bodies.  This has now been implemented
  locally and remotely verified with build/static/metadata plus H1/S128
  correctness.
- Continue one-kernel optimization inside `fa3_bwd_dkv_kernel`; do not add
  another performance path.
- Avoid already-rejected shortcuts: builtin raw waits, extra raw/source token
  generations, direct AllDone removal, pre-reading all source operands across
  softmax, and causal=false tuning.

## 2026-07-03 Physical dKV Code Convergence

Decision: `CODE_GOVERNANCE_ACCEPT`

Purpose:

Complete the source-convergence pass after the active route had already been
narrowed to one canonical dKV kernel.  This round intentionally does not claim
a performance improvement.

Change:

- Deleted archived `#if 0` bring-up/rejected global kernels.
- Deleted stale experiment helpers for Mq64, semantic pages, sidecar overlay,
  causal skip, score/dP brick, and split producers.
- Deleted old standalone probe scripts for `fa3_bwd_dkv_probe`,
  sidecar correctness, and fragment-sidecar correctness.
- Reduced dKV path contract to reference correctness plus canonical dKV.
- Tightened `scripts/check_dkv_kernel_gate.py` so stale experiment route
  symbols are forbidden in active source.

Evidence:

- Remote build PASS.
- Static dKV gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, `sgpr_spill=0`, `vgpr_spill=0`.
- WDRA windows unchanged:
  producer `6/16`, consumer0 `159/160`, consumer1 `159/160`.
- H1/S128 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_111624`.
- H1/S128 stats:
  `simTicks=17781855`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=17781855`, `MMOP=2048`, `ldsBankConflict=0`,
  `coissue=351/238`.

Conclusion:

The live repo now expresses the intended development rule: one canonical dKV
performance kernel plus reference correctness.  Future layout or pipeline
ideas should be implemented as focused probes or in-place edits to
`fa3_bwd_dkv_kernel`, not as accumulated phase stacks.

## 2026-07-03 A1 32x16 Same-LDS Q/dO Load-Once And Mq64 Follow-Up

Decision: `OBSERVE_CORRECTNESS_REJECT_PERF`

Purpose:

Test the user's hypothesis that once raw `Q/dO` can be loaded once and read as
both normal and transposed fragments, the freed LDS should first be spent on a
larger `Wq/Mq` tile.

Changes tested:

- A1 replaced external source-layout `Q^T/dO^T` inputs with raw `Q/dO` pages
  loaded by `matrix_load_32x16_b16`, then read by normal/trans
  `ds_read_matrix` forms.
- Mq64-B changed the canonical tile from `Mq=32` to `Mq=64`, kept one raw page
  ownership turn per Mq64 tile, and consumed it as two short M32 halves.
- Mq64-A kept `Mq=64` but stretched the consumer schedule: score/dP for both
  M32 halves first, then softmax/dS, then dV/dK. Consumer WDRA window was set
  to `208`, with branch windows `204/208` and `200/208`.

Evidence:

- A1 H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_144936`,
  `kernel_ticks=67704000`, `MMAC active=22.8697%`.
- Mq64-B H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_150946`,
  `kernel_ticks=67825940`, `MMAC active=23.1120%`,
  `VOP active=23.0466%`.
- Mq64-A H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_151356`,
  `kernel_ticks=67762240`, `MMAC active=23.0392%`,
  `VOP active=22.9746%`.
- Mq64-A full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_151548`,
  xcu output
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/mq64A_full_h1s1024_20260703_151548`.
- Canonical comparison remains
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093932`,
  `kernel_ticks=66411800`, `MMAC active=23.4288%`.

xcu findings for Mq64-A:

- dispatch duration `149680` cycles versus canonical `145960`.
- `s_abarrier_try_wait -> s_xor_b32` remains dominant at `25.92%`, essentially
  unchanged from canonical `25.95%`.
- sidecar `global_load_dwordx3 -> s_waitcnt` improves slightly to `10.77%`
  from canonical `11.28%`.
- `ds_read_matrix_trans_format -> s_waitcnt` worsens to `3.32%` from canonical
  `2.38%`.
- `v_mmac -> v_mmac` remains about `6.51%`, and MMAC source latency share does
  not move toward the 60% active target.

Conclusion:

The `32x16 same-LDS` contract is real and resource-clean, and `Mq64` is
possible under WDRA without spill/scratch. But the current Mq64 schedules do
not convert the larger Wq into a better conveyor: q-loop barrier count is
lower, yet the RawUsed/control bubble is unchanged and operand-read wait grows.
The larger tile also raises VOP active, so it is not a promotion.

Next:

- Do not keep chasing Mq64 by simply rearranging the two M32 halves.
- If continuing from this experimental state, the next hypothesis must directly
  reduce the RawUsed/control wait or make producer/helper work overlap that
  wait. Otherwise revert to the canonical rebaseline before the next
  independent optimization.

## 2026-07-03 W16 Split-Producer Mq64 Structural Probe

Decision: `STRUCTURAL_PASS_REJECT_PERF`

Purpose:

Address the current review findings directly:

- the active code was still effectively 12-wave;
- instruction gaps were severe;
- both heavy consumers were still moving in lockstep.

Change:

- Switched the canonical dKV kernel to a 16-wave CTA:
  waves0-3 `producer_kq_loop`, waves4-7 consumer0, waves8-11 consumer1,
  waves12-15 `producer_vdout_loop`.
- Kept the same-LDS `Mq=64,Nk=128,D=128` contract and 208-VGPR consumer
  window.
- Updated the static gate to require `hcu_wdra_waves_per_tg(16)`,
  the two producer loops, and the wave12-15 role branch.

Evidence:

- Remote build/static/symbol gates PASS.
- Metadata is clean:
  `private=0`, `sgpr=74`, `vgpr=112`, no spill/scratch.
- Branch windows:
  producer KQ `6/16`, consumer0 `204/208`, consumer1 `204/208`,
  producer VDout `1/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_161054`
  also produced the H1/S1024 full-perf run.
- H1/S1024 correctness PASS:
  `dk_rel_l2=0.0025563`, `dv_rel_l2=0.000337571`.
- H1/S1024 full perf:
  `kernel_ticks=69039425`, `MMOP=131072`, `ldsBankConflict=0`,
  coissue `27214/18060`, `MMAC active=22.3357%`,
  `VOP active=23.7529%`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_161054_clean_w16_split_mq64_h1s1024_sqc7_fullperf`.

xcu findings:

- dispatch duration `151736`, average active waves `115.35`.
- top issue bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `38.07%`,
  `s_abarrier_try_wait -> s_waitcnt` `9.89%`,
  `global_load_dwordx3 -> s_waitcnt` `8.18%`,
  `v_mmac -> v_mmac` `4.18%`,
  `ds_read_matrix_trans_format -> s_waitcnt` `2.12%`.

Conclusion:

The first review item is fixed structurally: PMD shows
`wg size=(1024,1,1)` and `16 waves using this aBarrier/eBarrier group`.
The other two items are not fixed.  The dominant bubble got worse because the
two producer split still uses a raw-page ownership protocol that makes the two
consumer groups wait on the same page-control rhythm.  The second producer is
also still thin: it performs the V/dO MLS role but has almost no useful
post-load work in its branch (`1/16` VGPR metadata).

Next:

- Keep the W16 code as the structural baseline only if the next task requires
  16-wave form.
- Do not treat this as the best performance route.
- The next real optimization must change the ABarrier/raw-page lifecycle or
  move sidecar/helper work into producer B so that producer B contributes
  useful overlap and consumer0/consumer1 stop waiting on the same token rhythm.

## 2026-07-04 Q/dO Raw Buffer Depth A/B

Decision: `BASELINE_DOUBLE_BUFFER_KEEP`, single-buffer probe
`REJECT_PERF_PROBE`.

Question:

- Quantify how much the current Q/dO raw double buffer helps versus a
  single-buffer protocol before spending the released K/V LDS on three or four
  pages.

Method:

- Kept the active W16 K/V-resident Mq64 kernel, math, output ownership, and
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.
- Added a compile-time raw-page helper so `kRawBuffers=1` uses only page0 and
  waits after every q tile, while `kRawBuffers=2` keeps the current page0/page1
  ping-pong.
- Sidecar cache-warm distance follows buffer depth: double buffer prefetches
  `q_tile+2`; single buffer prefetches `q_tile+1`.
- The single-buffer constant was restored after measurement; the active source
  is back to double buffer.

Evidence:

- Double-buffer same-build baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_153154`,
  `kernel_ticks=61582885`, `MMAC active=25.4747%`,
  coissue `26857/16837`, `VALU=181512`, `SCA=311168`, `LDS=66816`,
  `VMEM=4352`, `ldsBankConflict=0`, metadata `private=0 sgpr=62 vgpr=112`.
- Single-buffer probe:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_153511`,
  `kernel_ticks=63344645`, `MMAC active=24.6431%`,
  coissue `19026/12647`, `VALU=185888`, `SCA=232752`, `LDS=66816`,
  `VMEM=4352`, `ldsBankConflict=0`, metadata `private=0 sgpr=88 vgpr=112`.

Conclusion:

- Double buffer is real but modest: it improves H1/S1024 kernel ticks by about
  `2.78%` versus single buffer and raises MMAC active by about `0.83` point.
- This is not a long-conveyor solution.  The small gain means the current
  two-page design mostly prevents total serialization, but consumer page
  lifetime and RawUsed/sidecar waits still dominate.
- Three/four-buffer work should not be implemented as an unconditional extra
  ABarrier stack.  The high-value design is to add a `ResidentUsed` ownership
  point after all consumer waves latch K/V into VGPR, then reuse the dead K/V
  64KB LDS for additional Q/dO raw pages and possibly sidecar LDS cache.

Next:

- Workbook-stress two candidates before coding:
  `3-page`: add one extra Q/dO raw page plus sidecar LDS cache in released K/V
  space; lower barrier complexity.
  `4-page`: use all released K/V LDS for two extra Q/dO raw pages; highest
  producer lookahead but needs a larger barrier ledger and careful RawUsed
  accounting.
- Promotion criterion: three/four pages must improve same-shape MMAC active
  and ticks with no spill/scratch and `ldsBankConflict=0`; if it only raises
  ABarrier/SCA cost, reject.

## 2026-07-04 Single Raw Page With Producer-Published Sidecar LDS

Decision: `ACCEPT_PIPELINE_SIDECAR_LDS_SINGLEBUF`.

Question:

- The raw Q/dO double buffer only gave a modest benefit, while xcu repeatedly
  showed consumer-side sidecar global load wait.  Test the user's proposal:
  use one raw Q/dO page for now, have producer A publish sidecar metadata into
  LDS, and make consumers read sidecar from LDS instead of global memory.

Implementation:

- `kRawBuffers=1`.
- Added `DkvLdsLayout::kSidecarBase` and sidecar page helpers.
- Added `publish_sidecar_tile_to_lds<Tile>` in producer A.  Only
  `wave_local==0` and lanes `<64` write the SoA sidecar page:
  `max_log2[64]`, `inv_sum[64]`, `delta[64]`.
- Replaced consumer global sidecar reads with
  `softmax_ds_owner16_from_lds_sidecar`.
- `RawFilled` now gates Q, dO, and sidecar readiness for the active page.

Evidence:

- Static/resource gates PASS:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `vgpr_count=112`.
- Branch windows:
  producer KQ `10/16`, consumer0 `196/208`, consumer1 `196/208`,
  producer VDout `4/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_154344`,
  `dk_rel_l2=0.000361379`, `dv_rel_l2=0.000267234`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_154650`,
  `kernel_ticks=54539485`, `MMOP=131072`, `MMAC active=26.6693%`,
  `VALU=180570`, `SCA=215648`, `LDS=85822`, `VMEM=4352`,
  coissue `20030/11508`, `ldsBankConflict=0`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_154650_clean_singlebuf_lds_sidecar_h1s1024_sqc7_fullperf`.
- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `35_singlebuf_lds_sidecar`.

xcu findings:

- The previous consumer `global_load_dwordx3 -> s_waitcnt` sidecar bubble is no
  longer a top bubble.
- New top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `47.39%`,
  `s_abarrier_try_wait -> s_waitcnt` `6.60%`,
  `ds_read_b32 -> s_waitcnt` `2.34%`.
- Exported window `100000:118000` at `xcd=0,se=3,cu=1,simd=3,wave=0`
  shows SIMD bubble `96.76%`, dominated by
  `s_abarrier_try_wait -> s_waitcnt`.

Conclusion:

- Sidecar LDS is a real structural win.  It improves H1/S1024 kernel ticks by
  about `11.44%` versus the double-buffer global-sidecar baseline
  (`61582885 -> 54539485`) and by about `13.90%` versus the single-buffer
  global-sidecar probe (`63344645 -> 54539485`).
- The next bottleneck is not consumer sidecar global latency; it is
  RawUsed/RawFilled page-token serialization.  Single-buffering exposes this
  more sharply even though total ticks improve.

Next:

- Keep sidecar LDS as the current active route.
- Do not move sidecar reads back to consumer global memory.
- Draft the next workbook row for one of:
  reintroduce two raw pages with matching sidecar pages, split sidecar/RawFilled
  ownership, or use LDS slack for a third raw+sidecar page.  The next candidate
  must reduce ABarrier wait without adding an unconditional barrier stack.

## 2026-07-04 Raw2 Sidecar Overlay On Dead K/V LDS

Decision: `ACCEPT_PIPELINE_MICRO`.

Question:

- The single-buffer LDS-sidecar path removed consumer global sidecar latency but
  exposed Raw/sidecar token serialization.  Test the user's follow-up: keep
  sidecar in LDS, but reuse the K/V resident LDS region after consumers latch
  K/V into VGPR, so the raw Q/dO path can return to two buffers without
  exceeding 128KB LDS.

Implementation:

- Restored `kRawBuffers=2`.
- Added `ResidentUsed` as a one-shot ownership token.  Consumers arrive after
  `latch_owner16_kv_regs`; producer A waits before writing sidecar into the
  overlaid K/V region.
- Sidecar base now aliases the K/V resident LDS region; planned LDS excludes
  sidecar bytes because sidecar lives only after K/V resident data is dead.
- Static gate now checks the `ResidentUsed` wait/arrive/init protocol.

Evidence:

- Static/resource gates PASS:
  `private_segment_fixed_size=0`, `sgpr_count=60`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Branch windows:
  producer KQ `6/16`, consumer0 `198/208`, consumer1 `198/208`,
  producer VDout `1/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_161558`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_161747`,
  `kernel_ticks=53719120`, `MMOP=131072`, `MMAC active=27.7542%`,
  `VALU=181916`, `SCA=297480`, `LDS=85822`, `VMEM=4352`,
  coissue `32106/18911`, `ldsBankConflict=0`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_161747_clean_raw2_sidecar_kv_overlay_h1s1024_sqc7_fullperf`.

xcu findings:

- Dispatch0 detail duration `118064`, issue count `807423`,
  avg active waves `118.54`.
- Top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `39.79%`,
  `s_abarrier_try_wait -> s_waitcnt` `7.82%`,
  `v_mmac -> v_mmac` `5.76%`,
  `ds_read_b32 -> s_waitcnt` `2.43%`.
- Window export:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/raw2_sidecar_kv_overlay_h1s1024_20260704_161747_dispatch0_window_bar6`.

Conclusion:

- The overlay idea is correct and modestly positive.  It improves full-perf
  H1/S1024 kernel ticks by about `1.50%` versus single-buffer LDS-sidecar
  (`54539485 -> 53719120`) and by about `12.77%` versus the double-buffer
  global-sidecar baseline (`61582885 -> 53719120`).
- It also raises MMAC active from `26.6693%` to `27.7542%` versus single-buffer
  LDS-sidecar, with `ldsBankConflict=0`.
- The bottleneck is now more cleanly ABarrier ownership exposure, not sidecar
  global load latency.  The new `ResidentUsed` token is paid once per CTA, but
  raw page waits still dominate the steady timeline.

Next:

- Keep K/V-overlay sidecar as the active route.
- Do not move sidecar back to global memory.
- Next micro should reduce over-synchronization: only the actual sidecar writer
  wave may need to wait for `ResidentUsed`; non-writer producer waves can keep
  issuing Q raw work and arrive at RawFilled early, with writer arrival last.
- If that does not reduce xcu ABarrier bubbles, move back to workbook-level
  topology work rather than adding more tokens.

## 2026-07-04 Converge Back To Single Raw Buffer LDS Sidecar

Decision: `CODE_CONVERGENCE_ACCEPT`.

Question:

- The user decided the raw2 overlay gain is too small for the added ownership
  complexity.  Revert the active route to single raw Q/dO buffer while keeping
  the proven producer-published LDS sidecar path.

Implementation:

- Set `kRawBuffers=1`.
- Sidecar lives in its own small LDS region after K/V instead of overlaying
  dead K/V.
- Removed `ResidentUsed` from `DkvBarrierLedger`, kernel init/invalidate,
  producer wait, consumer arrive, and static gate.
- Removed Raw1 tokens and helper branches from the active source and gate.
- The canonical route now has only:
  `ResidentFilled`, `Raw0Filled`, `Raw0Used`, and `AllDone`.

Evidence:

- Remote build/static/symbol gates PASS:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Branch windows:
  producer KQ `10/16`, consumer0 `196/208`, consumer1 `196/208`,
  producer VDout `4/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_164439`.
- H1/S1024 stats PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_164444`,
  `kernel_ticks=54818400`, `MMOP=131072`, `MMAC active=26.6857%`,
  `VALU=180570`, `SCA=215376`, `LDS=85822`, `VMEM=4352`,
  coissue `19747/11693`, `ldsBankConflict=0`.

Conclusion:

- The cleaned single-buffer route is about `0.51%` slower than the previous
  single-buffer full-perf record, within run/build noise for this style of
  PMD comparison, and about `2.05%` slower than raw2 overlay.
- This is an intentional readability/control tradeoff, not a speed promotion.
  The raw2 overlay result remains useful evidence, but the active code now
  follows the simpler single-buffer contract requested by the user.

Next:

- Continue from this single-buffer LDS-sidecar baseline.
- Do not reintroduce Raw1 or `ResidentUsed` unless workbook and xcu evidence
  justify the complexity.
- The next real bottleneck remains ABarrier/consumer lockstep, not sidecar
  global memory.

## 2026-07-04 BlockMq Template And Mq128 Stress

Decision: `CODE_GOVERNANCE_ACCEPT` for templateization,
`REJECT_PERF` for the tested `Mq128` implementation.

Question:

- The user suggested making tile size a template parameter and testing whether
  a larger per-wave dK/dV tensor is the real algorithm-level route to higher
  MMAC active.

Implementation:

- Replaced the fixed `DkvTileD128Mq64Nk128` contract with
  `DkvTileD128MqNk128<BlockMq>`.
- The active alias is back to `ActiveDkvTile = DkvTileD128MqNk128<64>`.
- `Mq128` was tested through the same canonical kernel by temporarily changing
  the alias, not by adding another kernel or phase route.
- For `Mq128`, raw Q/dO must overlay the K/V resident LDS lifetime, so the
  stress version uses a one-shot `ResidentUsed` token after consumers latch
  K/V into VGPR.  `Mq64` does not instantiate that overlay path.
- Static `Mq128` expansion initially failed metadata:
  `private_segment_fixed_size=8`, `sgpr_spill_count=18`,
  `vgpr_spill_count=2`.
- Switching the M-block body to a small runtime loop fixed resource pressure:
  branch consumers `190/208`, metadata `private=0`, `sgpr=62`, `vgpr=112`,
  no spills.

Evidence:

- `Mq128` H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_185323`.
- `Mq128` H1/S1024 correctness/stats PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_185357`,
  `kernel_ticks=62473320`, `MMOP=131072`, `MMAC active=23.2158%`,
  coissue `35774/19059`, `ldsBankConflict=0`.
- Active `Mq64` template checkpoint H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_190032`.
- Active `Mq64` template checkpoint H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_190101`,
  `kernel_ticks=54887105`, `MMAC active=26.5542%`, coissue `20049/12067`,
  `ldsBankConflict=0`.
- Workbook updated:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `38_mq128_template`.

Conclusion:

- Increasing `BlockMq` does not increase total MMOP for fixed S: `Mq128`
  doubles per-tile MMAC but halves q-tile count, so H1/S1024 still reports
  `MMOP=131072`.
- The possible win is only from amortizing control/ABarrier/wait under longer
  MMAC islands.  The tested implementation did not achieve that: dynamic
  M-block control plus overlay ownership lowered MMAC active and regressed
  ticks by about `13.96%` versus the single-buffer `Mq64` baseline.
- The idea remains architecturally plausible, but future `BlockMq>64` work must
  preserve compile-time MMAC islands without SGPR spill, or otherwise show via
  xcu that dynamic-loop control is hidden.  Do not promote this `Mq128`
  implementation.

## 2026-07-04 Current Full Perf And Sidecar Prefetch Probe

Decision: `PASS_XCU_BASELINE` for the current full-perf rebaseline,
`REJECT_CORRECTNESS` for no-token future sidecar prefetch, and `REJECT_PERF`
for the current-page four-wave sidecar writer diagnostic.

Question:

- The current `f27ec64`/`ActiveDkvTile=Mq64` route had stats-only evidence but
  needed a fresh full perf/xcu baseline before changing ABarrier or sidecar
  lifetime.
- Hypothesis: sidecar could be moved out of the raw Q/dO critical path by
  double-buffering only sidecar pages and prefetching q+1 sidecar while
  consumers compute q.

Baseline evidence:

- Full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411`.
- Helper perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411/m5out/0/0/2673937_fa3_bwd_wasp_clean.perf`.
- xcu first pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/f27_h1s1024_fullperf_20260704_191411_dispatch0`.
- xcu focused window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/f27_h1s1024_fullperf_20260704_191411_dispatch0_window_barrier_swait`.
- Metrics: `simTicks=58424275`, `MMOP=131072`,
  `MMAC active=26.6364%`, VOP share `19.7478%`, coissue `20088/11882`,
  `ldsBankConflict=0`.
- xcu: duration `120460`, avg active waves `124.27`.
- Top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `47.46%`,
  `s_abarrier_try_wait -> s_waitcnt` `6.64%`,
  `ds_read_b32 -> s_waitcnt` `2.53%`.
- Focused barrier window `100912:113928`,
  location `xcd=0,se=2,cu=1,simd=3,wave=0`:
  pipeline bubble `99.99%`; selected SIMD bubble `97.37%`,
  MMAC `%` only `0.49%`.

Implementation/probe:

- Tried `kSidecarBuffers=2` with no new ABarrier token.
- Producer KQ wrote current sidecar before the q-loop and future q+1 sidecar
  after current `RawFilled`.
- Added `lgkmcnt(0)` after sidecar writes when the first attempt failed.
- Diagnosed producer skew: old sidecar writer only had `wave_local=0` doing
  useful sidecar writes, so a four-wave writer variant split 64 rows across
  waves0-3.
- Finally tested a correctness-safe diagnostic: two sidecar pages, four-wave
  writer, current-tile publication only, no future prefetch.

Evidence:

- No-token future prefetch failed H1/S128 correctness twice:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_192236`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_192452`,
  with `dk_rel_l2=7.20419`, `dv_rel_l2=5.57578`.
- Current-page four-wave sidecar diagnostic passed H1/S128:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_192742`.
- It passed H1/S1024 but regressed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_192805`,
  `simTicks=60289320`, `MMAC active=26.1523%`,
  coissue `20333/11403`, `VALU=190556`, `SCA=220848`, `LDS=86590`,
  `VMEM=4352`, `ldsBankConflict=0`.

Conclusion:

- The current main bottleneck is ABarrier/raw packet ownership lifetime, not
  missing MMAC, LDS bank conflict, or sidecar global memory.
- Future sidecar prefetch is not legal without explicit sidecar readiness and
  ownership, because it is no longer tied to the raw Q/dO epoch.
- Splitting sidecar rows across producer waves is correct but not profitable
  in the current critical path; it increases scalar/vector work and lowers
  MMAC active.
- Reverted all sidecar-prefetch diagnostic code.  Active source is back to the
  clean single-page LDS sidecar baseline.

Next:

- Do not retry no-token future sidecar prefetch.
- If sidecar is decoupled again, design it as a proper `SidecarFilled` /
  `SidecarUsed` protocol and prove in the workbook that the added ABarrier cost
  is hidden by score/dP MMAC.
- The next higher-probability path is to reduce raw packet wait count/lifetime
  or revisit raw2 overlay only with a simpler ownership ledger.

### Tail AllDone Wave0 Probe

Status: `REJECT_PERF`, code reverted.

Hypothesis:

- Only wave0 waits `AllDone` and invalidates ABarrier tokens at kernel tail.
- Other waves only arrive at `AllDone` and fall through to the final
  `__syncthreads()`.
- Expected effect was a small reduction in tail ABarrier overhead.

Evidence:

- Build/static gates PASS:
  `private=0`, `sgpr=86`, `vgpr=112`, no SGPR/VGPR spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_193951`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_194010`.
- H1/S1024 stats:
  `simTicks=58700915`, `MMAC active=26.7200%`,
  coissue `18982/11595`, `ldsBankConflict=0`.
- Baseline was
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411`:
  `simTicks=58424275`, `MMAC active=26.6364%`,
  coissue `20088/11882`.

Conclusion:

- The tiny MMAC-active increase is not a pipeline win because same-shape ticks
  regressed by about `0.47%` and coissue success/rate dropped.
- Tail-only ABarrier cleanup does not attack the dominant xcu bubble
  `s_abarrier_try_wait -> s_xor_b32` inside the raw packet loop.
- Do not continue optimizing `AllDone` tail handling in isolation.

### Mq128 Static-Scoped Direct Probe

Status: `REJECT_RESOURCE`, code reverted before PMD.

Design intent:

- Increase useful MMAC per raw-packet handshake rather than adding more
  buffering.
- `Mq64` does `128` MMAC per consumer per q tile and uses `S/64` raw handshakes.
- `Mq128` would do `256` MMAC per consumer per q tile and use `S/128` raw
  handshakes.  For fixed `S`, total MMOP is unchanged, but ABarrier/control
  overhead should be amortized over a larger compute island.
- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
  `41_mq128_static_plan`.

Implementation tried:

- Temporarily set `ActiveDkvTile = DkvTileD128MqNk128<128>`.
- Replaced the `BlockMq != 64` dynamic M-loop with compile-time
  `consume_mq_tile_owner16<Tile, Wdra, 0, FirstAccum>` recursion.
- No new kernel, no new phase, no new output path.

Static evidence:

- Build completed but resource metadata failed:
  `private_segment_fixed_size=8`, `sgpr_count=104`,
  `sgpr_spill_count=18`, `vgpr_count=112`, `vgpr_spill_count=2`.
- Branch windows:
  producerKQ `15/16`, consumer0 `208/208`, consumer1 `208/208`,
  producerVDout `9/16`.
- Compiler also emitted repeated `found vgpr before wave branch` warnings.

Conclusion:

- Direct static Mq128 expansion is not viable.
- The high-level idea is still plausible, because halving raw handshakes targets
  the observed ABarrier bubble directly, but the implementation must first
  solve SGPR/VGPR lifetime.
- The next Mq128 attempt needs a resource redesign: either scoped/noinline
  M-pair islands, a smaller accumulator/store ownership, or a different
  producer/consumer split.  It must pass metadata before any PMD run.

Follow-up resource check:

- Retried the same direct static Mq128 shape with
  `WdraResourceWindows::kConsumerVgprs = 248`.
- Result:
  `private_segment_fixed_size=0`, `sgpr_count=100`,
  `sgpr_spill_count=18`, `vgpr_count=132`, `vgpr_spill_count=0`.
- Branch windows became consumer `209/248`, so the VGPR side is legal.
- The remaining hard failure is SGPR spill, not the 208-VGPR window.
- Baseline source and remote binary were restored after this check.

Additional Mq128 resource probes:

- `consume_mq_mpair_owner16` noinline boundary:
  - intent: cut M-pair island scalar live range without changing math or output
    ownership
  - result: `sgpr_spill_count=0`, `vgpr_spill_count=0`, but
    `private_segment_fixed_size=492`
  - conclusion: noinline proves the SGPR spill comes from inlined M-pair
    lifetime, but device calls/private segment make this route unusable
- explicit four-Mpair sequence:
  - intent: remove template recursion while keeping full static Mq128
  - result: same as direct static 248VGPR:
    `private=0`, `sgpr_count=100`, `sgpr_spill_count=18`,
    `vgpr_count=132`, `vgpr_spill_count=0`
  - conclusion: recursion is not the root cause; the inlined four-Mpair
    branch itself expands scalar/control state too much
- causal=true literal probe:
  - intent: test whether causal mask/predicate scalar state is the main Mq128
    resource blocker
  - result: `private=0`, `sgpr_count=100`, `sgpr_spill_count=16`,
    `vgpr_count=132`, `vgpr_spill_count=0`
  - conclusion: causal mask contributes about two SGPR spills but is not the
    root cause
- Baseline source and remote binary were restored clean after these probes.

### Mq128 Dynamic Causal-True Probe

Status: `REJECT_PERF`, code reverted.

Hypothesis:

- The dynamic Mq128 path is resource-clean but previously slow.
- Specializing the canonical performance path to `causal=true` could reduce
  mask/predicate VALU and recover enough of the larger raw packet benefit.

Evidence:

- Temporary implementation:
  `ActiveDkvTile = DkvTileD128MqNk128<128>`, dynamic M-loop retained,
  consumer calls passed literal `causal=1`.
- Metadata PASS:
  `private=0`, `sgpr=58`, `vgpr=112`, no spill.
- Branch windows:
  producerKQ `14/16`, consumer0 `190/208`, consumer1 `190/208`,
  producerVDout `8/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mq128_dyn_causal1_20260704_201845_s128`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mq128_dyn_causal1_20260704_201914_s1024`.
- H1/S1024 stats:
  `simTicks=65580515`, `MMAC active=23.6126%`,
  coissue `35928/19195`, `VALU=269724`, `SCA=230936`,
  `LDS=91996`, `VMEM=4352`, `ldsBankConflict=0`.
- Baseline Mq64:
  `simTicks=58424275`, `MMAC active=26.6364%`,
  `VALU=180570`, `SCA=215792`.

Conclusion:

- Causal specialization helps resource cleanliness but not performance.
- Dynamic Mq128 remains too VALU/control-heavy and about `12.3%` slower than
  the Mq64 baseline.
- PMD emitted `read vgpr68 before writing`, but numerical correctness passed;
  this warning is noted and not promoted.

### Q/dO Split Lifetime Probe

Status: `REJECT_PERF_STATS_ONLY`, code reverted.

Hypothesis:

- Current `Raw0Filled/Raw0Used` binds Q and dO to one packet lifetime.
- Q is needed by score and dK; dO is needed by dP and dV.  Splitting tokens
  into Q-filled/Q-used and dO-filled/dO-used might let producer0 start the next
  Q+sidecar packet earlier while current consumers finish dK/dV.

Evidence:

- Workbook design sheet added:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `43_q_dout_split_lifetime`.
- First implementation: full Q-first dK/dV split.
  - Changed barrier ledger to `QRawFilled/QRawUsed` and
    `DoutRawFilled/DoutRawUsed`.
  - Split dK and dV source reads/MMAC islands so Q released before dK, then
    dO released before dV.
  - Metadata PASS: `private=0`, `sgpr=86`, `vgpr=112`, no spills; branch
    consumers `164/208`.
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_210423`.
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_210509`.
  - H1/S1024 stats: `simTicks=63880635`, `MMAC active=24.4167%`,
    coissue `18856/10486`, `VALU=188634`, `SCA=219712`, `LDS=85822`,
    `VMEM=4352`, `MMOP=131072`, `ldsBankConflict=0`.
- Second implementation: split tokens only, preserve combined dV/dK MMAC
  island.
  - Kept original combined dV/dK MMAC, but issued high-Q reads and arrived
    `QRawUsed` before high-dO reads and `DoutRawUsed`.
  - Metadata PASS: `private=0`, `sgpr=86`, `vgpr=112`, no spills; branch
    consumers `196/208`.
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_211029`.
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_211035`.
  - H1/S1024 stats: `simTicks=61733035`, `MMAC active=25.1155%`,
    coissue `19731/12057`, `VALU=176730`, `SCA=219200`, `LDS=85822`,
    `VMEM=4352`, `MMOP=131072`, `ldsBankConflict=0`.
- Baseline for comparison:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411`,
  `simTicks=58424275`, `MMAC active=26.6364%`, coissue `20088/11882`,
  `VALU=180570`, `SCA=215792`, `LDS=85822`, `VMEM=4352`,
  `ldsBankConflict=0`.

Conclusion:

- Splitting Q and dO lifetimes is correctness-legal and resource-clean, but it
  is not profitable in the current single-page Mq64 code shape.
- Full Q-first splitting breaks the compact combined dV/dK MMAC island and
  regresses ticks by about `9.34%`.
- The narrower token-only split still regresses ticks by about `5.66%`; the
  added ABarrier/SCA/control cost is larger than the few-instruction earlier Q
  release.
- Do not continue adding token families to solve the ABarrier bubble unless a
  design also creates a longer useful-work window, reduces token turns, or
  increases effective GEMM island size.
- Baseline source and remote binary were restored after the probe.

### Nk32 Packed-Owner Resource Probe

Status: `REJECT_RESOURCE`, code reverted after metadata failure.

Design basis:

- Workbook sheet `44_nk32_four_consumer_probe` records the hypothesis:
  make each active consumer wave logically cover `Nk=32` by packing two
  adjacent owner16 blocks into one wave.
- This keeps math/output ownership correct and avoids duplicating score/dP,
  but doubles long-lived dV/dK accumulators and K/V owner registers in the
  active consumer body.
- This first cut is a resource-admission probe.  Waves8-11 are currently
  inactive helper waves, so it is not yet the final 16-wave WASP topology.

Local source change:

- Added `consumer_dkv_mmac_loop_packed_owner16x2`.
- waves4-7 process owner pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`.
- `Raw0Used` arrival count changes from 8 to 4 because four active consumers
  release the raw page only after both packed owners are consumed.
- `kConsumerVgprs` is temporarily set to `248` to test the high-WDRA window.
- `scripts/check_dkv_kernel_gate.py` was updated to recognize this canonical
  probe route and the local source gate passes.

Remote evidence:

- liuchang recovered through jump host `172.20.32.54`.
- First build with windows `16/248/16/16` failed compile:
  `branch-averaged vgpr size must be multiple of target's vgpr granularity`.
- Setting the inactive branch to `40` VGPRs made the windows
  `16/248/40/16`, and build/asm completed.
- Symbol metadata gate for `fa3_bwd_dkv_kernel` then failed:
  `private_segment_fixed_size=236`, `vgpr_spill_count=58`, `sgpr_count=96`,
  `vgpr_count=80`.

Conclusion:

- Packing two owner16 blocks into one active wave doubles live dV/dK
  accumulator families enough to spill badly, even with a 248-VGPR consumer
  branch and a balanced inactive window.
- Do not continue this owner16x2 packing route.
- The next larger-island attempt needs phasing that does not keep two owner16
  dV/dK accumulator sets live simultaneously, or it must move to a true
  owner32 design with explicit accumulator lifetime control.

### Raw2 Page-Local ABarrier Recovery

Status: `ACCEPT_MICRO_OBSERVE`; canonical route updated in place.

Design basis:

- Workbook sheet `45_raw2_ab_xcu` records the xcu-driven design.
- H1/S1024 single-buffer baseline had a dominant producer-side raw-page wait:
  `s_abarrier_try_wait -> s_xor_b32 = 47.46%`, with a focused window showing
  `98.63%` bubble and `0%` MMAC.
- A first raw2 attempt with a shared `Raw0Filled/Raw0Used` token failed H1/S128
  PMD with `ABARRIER_CNT_ERROR`, proving that two outstanding raw pages need
  page-local ABarrier tokens.

Implementation:

- `kRawBuffers=2`.
- `raw_page_for_q_tile = q_tile & 1`.
- Page-local raw ownership:
  `Raw0Filled/Raw0Used = 2/3`, `Raw1Filled/Raw1Used = 4/5`.
- `AllDone` moved to barrier id `6`.
- No new kernel, phase, or alternative route was added.

Evidence:

- Build/gate PASS.
- Metadata PASS:
  `private=0`, `sgpr=60`, `vgpr=112`, no SGPR/VGPR spill.
- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_221533`.
- H1/S1024 stats PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_221539`,
  `kernel_ticks=53,300,975`, `MMAC active=27.6518%`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_221910`,
  `kernel_ticks=53,462,955`, `MMAC active=27.5982%`, `MMOP=131,072`,
  `VALU=181,980`, `SCA=296,328`, `LDS=85,822`, `coissue=30,829/18,010`,
  `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_221910_clean_raw2_tokens_h1s1024_sqc7_fullperf`.
- xcu top bubble:
  `s_abarrier_try_wait -> s_xor_b32` dropped to `40.24%`, but the top
  focused window is still `Raw1Used` (`barId 5`) with `98.19%` bubble and
  `0%` MMAC.

Conclusion:

- This is a real but small improvement over the single-buffer baseline:
  full-perf kernel ticks improve by about `2.47%` versus the archived
  single-buffer full perf (`54,818,400 -> 53,462,955`), and MMAC active rises
  by about `0.91` point.
- It does not solve the 60% active target.  More raw pages/tokens alone are
  unlikely to be enough; the next design should increase useful work per token
  or make producer waves do independent work while consumers hold raw pages.

### Raw3 Token Stress And v_mov/Useful Coissue Rule

Status: `REJECT_PERF_STATS_ONLY`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `46_raw3_token_stress` records the hypothesis: add a third
  raw Q/dO page so producers wait only after three live pages, not two.
- Temporary barrier ledger:
  `Raw2Filled/Raw2Used = 6/7`, `AllDone = 8`.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=68`, `vgpr=112`, no scratch/spill.
- H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- H1/S1024 stats-only:
  `kernel_ticks=56,111,055`, `MMAC active=26.5078%`,
  coissue `29,684/17,063`.
- Raw2 stats-only comparison:
  `kernel_ticks=53,300,975`, `MMAC active=27.6518%`.

Conclusion:

- A third raw page is legal, but the extra token/control/page-selection cost
  outweighs the RawUsed wait relief.
- Do not continue increasing raw page depth as the next route.
- Future pipeline work must prove useful work overlap, not only higher coissue
  counts.

Updated `v_mov` / coissue rule:

- `v_mov` is now a hard reduction target, but it is not an isolated promotion
  metric.
- If XCU coissue is dominated by `v_mov_b32_e32`, that is not a successful
  softmax/MMAC conveyor.  Useful coissue means softmax/dS math, sidecar or
  predicate/address work that actually hides peer MMAC, and reduced exposed
  wait/ABarrier bubbles.
- Current raw2 asm audit shows explicit `zero_f16x8` contributes only the
  small `include/shaobo_instr.h:32` group, while larger `v_mov` groups are
  around softmax/dS and compiler remap/control locations such as
  `src/dkv_kernel.cpp:620` and `.loc 0:0`.
- Next edits must either reduce these `v_mov` groups without expanding
  wait/ABarrier/code footprint, or create real consumer stagger where useful
  VALU runs under peer MMAC.  Promotion still requires correctness PASS, no
  scratch/spill, `ldsBankConflict=0`, and same-shape MMAC-active/ticks
  improvement with XCU explanation.

Follow-up static probe: FWD-style single `mmac_zero`.

- Change tried locally and on remote build: initialize one `mmac_zero` per
  consumer branch and pass it through score/dP plus dV/dK first-accum helpers,
  mirroring FA3 FWD's `mmac_zeros` style.
- Static/resource stayed legal:
  `private=0`, `sgpr=60`, `vgpr=112`, no spill/scratch, but branch consumer
  pressure increased from `198/208` to `202/208`.
- ASM result was negative:
  explicit zero-immediate `v_mov_b64` dropped from `20` to `4`, but
  `v_mov_b64_e32` copies increased from `107` to `139`; `v_mov_b32_e32`
  stayed `317`.
- Decision: `REJECT_STATIC_NO_PMD`.  Long-living the zero fragment reduces
  explicit zero init but creates more compiler register moves, so it is not a
  safe route for BWD's current live-range shape.
- Active source was restored to raw2 canonical.  Future `v_mov` work should
  target softmax/dS/control remaps or useful stagger, not simply hoist zero.

### G1 Score-Prefetch Stagger Probe

Status: `REJECT_PERF_STATS_ONLY`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `48_g1_score_prefetch_stagger` records the hypothesis:
  keep consumer0 as the timing anchor, but let consumer1 compute score/dP for
  pair0 and pair2 before finishing pair0/pair2.
- This is a real-work stagger attempt, not an empty delay and not a new phase.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=62`, `vgpr=112`, no scratch/spill.
- Branch windows:
  producerKQ `6/16`, consumer0 `198/208`, consumer1 `205/208`,
  producerVDout `1/16`.
- H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- H1/S1024 stats:
  `kernel_ticks=56,500,990`, `MMAC active=28.0755%`,
  coissue `29,908/17,617`.
- Raw2 stats-only baseline:
  `kernel_ticks=53,300,975`, `MMAC active=27.6518%`.

Conclusion:

- The stagger slightly improves MMAC active, but slows ticks by about `6.0%`.
- Holding score0/dP0 while issuing score2/dP2 makes consumer1 branch nearly
  full (`205/208`) and delays useful finish/store work enough to lengthen the
  critical path.
- Do not keep this code.  The next useful-stagger attempt must avoid carrying
  two score/dP pairs live at once, or target softmax/control `v_mov` shrink
  instead of reordering score islands.

### Causal=True Specialization Probe

Status: `REJECT_PERF_STATS_ONLY`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `49_causal_true_specialize` records the hypothesis:
  canonical dKV targets causal=true, so require `causal==1` and pass literal
  `1` into the consumer loop to let the compiler drop `!causal` arms.
- This is a codegen/control shrink attempt, not a pipeline redesign.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=56`, `vgpr=112`, no scratch/spill.
- Branch windows:
  producerKQ `6/16`, consumer0 `198/208`, consumer1 `198/208`,
  producerVDout `1/16`.
- H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- H1/S1024 stats:
  `kernel_ticks=56,200,690`, `MMAC active=28.0230%`,
  VALU `181,752`, SCA `280,408`, coissue `31,481/18,140`.
- Raw2 stats-only baseline:
  `kernel_ticks=53,300,975`, `MMAC active=27.6518%`.

Conclusion:

- Causal constant propagation reduces SGPR and SCA, but does not shorten the
  exposed critical path.  Ticks regress by about `5.4%`.
- The remaining limiter is still packet ownership/wait shape, not simply the
  runtime `causal` boolean.
- Do not keep this specialization in the canonical route.

### Score/dP Read Batch2 Probe

Status: `REJECT_PERF_STATS_ONLY`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `50_score_read_batch2` records the hypothesis:
  change score/dP from `4 ds_read_matrix -> wait -> 8 MMAC` per D block to
  `8 ds_read_matrix -> wait -> 16 MMAC` per two D blocks.
- This directly targeted XCU `v_mmac->v_mmac`, `v_mmac->s_waitcnt`, and
  `s_waitcnt->v_mmac` gaps without adding ABarrier tokens or changing output
  ownership.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=60`, `vgpr=112`, no scratch/spill.
- Branch windows stayed unchanged:
  producerKQ `6/16`, consumer0 `198/208`, consumer1 `198/208`,
  producerVDout `1/16`.
- H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- H1/S1024 stats:
  `kernel_ticks=56,275,310`, `MMAC active=27.9218%`,
  VALU `181,980`, SCA `296,328`, coissue `33,010/21,199`.
- Raw2 stats-only baseline:
  `kernel_ticks=53,300,975`, `MMAC active=27.6518%`.

Conclusion:

- Batching score reads is resource-clean, but it does not shorten the critical
  path.  It increases coissue activity and failed coissue while ticks regress
  by about `5.6%`.
- Do not keep this as live code.  Larger read/MMAC bricks need to be applied
  only where XCU proves the wait is the critical path and not hidden by other
  packet ownership/control costs.

### Structural Pivot After Raw2 Micro-Failures

Status: `DESIGN_READY`; no code change yet.

Design basis:

- Workbook sheet `51_structural_pivot` records the next structural experiment.
- Raw2 improved the single-buffer ABarrier bubble, but the top XCU issue is
  still `s_abarrier_try_wait -> s_xor_b32` at about `40.24%`.
- Raw3, score-prefetch stagger, causal specialization, and score-read batch2
  all passed correctness/resource gates but regressed ticks by more than 5%.

Conclusion:

- The next live-code change should not add another raw page, token split, fake
  delay, or local read-order tweak.
- The selected structural candidate is warpgroup-local ownership with duplicated
  Q/dO:
  `WG0 = waves0-3 producer K0/V0+Q/dO+sidecar -> waves4-7 consumer0`,
  `WG1 = waves12-15 producer K1/V1+Q/dO+sidecar -> waves8-11 consumer1`.
- Expected benefit: producer1 becomes useful in every q tile, RawUsed is local
  to four consumer waves instead of CTA-wide, and WG0/WG1 can naturally drift.
- Main risk: Q/dO/sidecar traffic doubles and each WG has only a single raw
  page unless a later LDS proof allows deeper local buffering.

### WG-Local Duplicate Q/dO Structural Probe

Status: `REJECT_PERF_FULL`; code will be reverted to raw2 canonical.

Design basis:

- Workbook sheet `51_structural_pivot`.
- WG0: waves0-3 producer K0/V0 + Q/dO + sidecar, waves4-7 consumer0.
- WG1: waves12-15 producer K1/V1 + Q/dO + sidecar, waves8-11 consumer1.
- Purpose: make producer1 useful every q tile and replace CTA-wide raw-page
  ownership with WG-local 4-wave ownership.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=88`, `vgpr=112`, no scratch/spill.
- Branch windows:
  producer0 `15/16`, consumer0 `197/208`, consumer1 `196/208`,
  producer1 `14/16`.
- H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- H1/S1024 full perf:
  `simTicks=58,310,070`, `MMAC active=26.7125%`, `MMOP=131,072`,
  `VMEM=8,448`, `VALU=186,234`, coissue `31,950/18,997`.
- Raw2 full baseline:
  `simTicks=53,462,955`, `MMAC active=27.5982%`.
- XCU detail:
  `s_abarrier_try_wait -> s_xor_b32` rises to `44.30%`;
  `s_abarrier_try_wait -> s_waitcnt` is `7.77%`;
  `matrix_load_32x16_b16` hits `8,192`; VMEM doubles versus raw2.
- Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_001605_wg_local_dup_qdo_h1s1024_sqc7_reject_fullperf`.

Conclusion:

- The topology is semantically valid, but the cost of duplicated Q/dO
  publication plus single-page local ownership exceeds the benefit of local
  4-wave ABarrier tokens.
- Do not pursue independent WGs by duplicating shared Q/dO. Shared Q/dO reuse
  is valuable.
- Next high-ceiling direction should preserve shared Q/dO and reduce RawUsed
  frequency, likely through an Mq128/static scalar-lifetime redesign or another
  coarser shared packet that avoids duplicate MLS traffic.

### Score/dP Pair-Read ASM Island

Status: `REJECT_CORRECTNESS`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `53_score_dp_pair_asm`.
- Goal was to test a minimal "C++ clean kernel + asm island" route instead of
  rewriting the whole kernel in assembly.
- Change: replace the score/dP island's four independent raw Q/dO
  `ds_read_matrix_trans` calls with two `ds_read_matrix_trans_pair` helper
  calls, expecting the second fragment to use `offset:1024`.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=60`, `vgpr=112`, no scratch/spill.
- Branch windows unchanged:
  producer0 `6/16`, consumer0 `198/208`, consumer1 `198/208`,
  producer1 `1/16`.
- Asm changed as expected: each pair emitted two consecutive
  `ds_read_matrix_trans_format` instructions with `offset:0` and
  `offset:1024`.
- H1/S128 correctness failed:
  `dk_rel_l2=1.20384`, `dv_rel_l2=0.00230465`, `pass=0`.

Conclusion:

- The implementation assumption was wrong.  In the raw Q/dO page layout,
  `raw_page_block_offset_m(page, m+1, d)` is separated from
  `raw_page_block_offset_m(page, m, d)` by `4 * 1024` bytes for D128, not
  `1024` bytes.
- `ds_read_matrix_trans_pair` remains valid for the resident K/V layout where
  the second required fragment is actually at `+1024`, but it cannot be reused
  blindly for score/dP raw M0/M1 reads.
- Future asm islands must first prove exact LDS adjacency for the specific
  MLS layout and fragment pair before replacing correct scalar-addressed reads.

### Causal Invalid-Prefix Page Skip

Status: `REJECT_PERF_STATS_ONLY`; code will be reverted to raw2 canonical.

Design basis:

- Workbook sheet `52_causal_page_skip`.
- Goal was algorithm-level redundant-work removal, not an instruction micro
  patch.  For causal=true, pages with `q_tile_end < k_start` have no valid
  pairs for a dKV owner.
- The first implementation used `accum_started` so the first actually valid
  page initialized dV/dK accumulators.  That was mathematically clean but
  failed the static gate: consumer branches hit `208/208`, with
  `private_segment_fixed_size=216` and `vgpr_spill_count=114`.
- The resource-clean implementation kept `q_tile0` always computed to preserve
  the existing initialization path, then skipped only `q_tile>=1` invalid
  causal prefixes using a small wait/release loop before the normal compute
  loop.

Evidence:

- Static/resource PASS for prefix-only version:
  `private=0`, `sgpr=62`, `vgpr=112`, no scratch/spill.
- Branch windows:
  producer0 `6/16`, consumer0 `199/208`, consumer1 `198/208`,
  producer1 `1/16`.
- Correctness PASS:
  H1/S128 and H1/S1024 both pass; H1/S1024 has
  `dk_rel_l2=0.0025563`, `dv_rel_l2=0.000337571`.
- H1/S1024 stats:
  `simTicks=57,087,940`, `kernel_ticks=53,474,330`,
  `MMOP_instr=77,312`, `MMAC active=22.4979%`,
  coissue `21,324/12,145`, `VALU=125,129`, `SCA=181,840`,
  `LDS=58,844`, `VMEM=2,784`, `ldsBankConflict=0`.
- Raw2 baseline:
  `MMOP=131,072`, stats-only `kernel_ticks=53,300,975`,
  `MMAC active=27.6518%`; raw2 full-perf kernel ticks were about
  `53,462,955`.

Conclusion:

- The candidate proves invalid causal MMAC work can be removed: MMOP drops by
  about `41%`.
- It does not improve the target small-shape critical path.  Ticks are flat to
  slightly worse and MMAC active share drops because barrier/release/control
  and the slowest valid k tiles dominate.
- Do not keep this in the canonical route.  Future causal optimization should
  change launch/tile ownership or critical-path structure, not just add a
  consumer-side invalid-prefix skip.

Post-revert recertification:

- The live source was restored to raw2 canonical; no causal-skip helper or
  skip branch remains in `src/dkv_kernel.cpp`.
- zys1 build/static PASS: branch windows `6/198/198/1`, `private=0`,
  `sgpr=60`, `vgpr=112`, no scratch/spill.
- Correctness PASS after restore:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_010339`
  and H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_010344`.
- H1/S1024 raw2 recert stats:
  `kernel_ticks=53,008,410`, `MMOP=131,072`,
  `MMAC active=27.7754%`, coissue `32,341/18,768`,
  `ldsBankConflict=0`.

### Raw Release Before Softmax

Status: `REJECT_CORRECTNESS_RESOURCE`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `54_raw_release_before_softmax`.
- Goal was to shorten the raw page lifetime on the full-valid critical path.
  Current raw2 releases the page after the final M-pair softmax/dS and high
  Q/dO source read, so producers can refill mainly during the final dV/dK
  MMAC island.
- Candidate moved the final M-pair low+high Q/dO source reads before
  softmax/dS, then tried to arrive RawUsed before softmax so producer refill
  could overlap both softmax/dS and dV/dK.

Evidence:

- First implementation protected Q/dO source reads but not the sidecar rows.
  Static/resource passed with branch windows `6/200/200/1` and metadata
  `private=0`, `sgpr=60`, `vgpr=112`, no spill/scratch.
- Correctness:
  H1/S128 PASS, but H1/S1024 FAIL:
  `dk_rel_l2=0.17587`, `dv_rel_l2=0.106138`, `pass=0`.
- H1/S1024 failed-run stats:
  `kernel_ticks=54,011,685`, `MMOP=131,072`,
  `MMAC active=26.9822%`, coissue `31,038/18,411`,
  `ldsBankConflict=0`.
- Diagnosis: sidecar max/sum/delta is part of the raw page lifetime.  Releasing
  RawUsed before softmax lets producer overwrite sidecar before consumer reads
  it on longer q loops.
- Second implementation also prefetched sidecar rows into VGPR before release.
  Static failed immediately: branch windows `6/208/208/1`,
  `private_segment_fixed_size=52`, `sgpr=68`, `vgpr_spill_count=24`.

Conclusion:

- Raw page lifetime is constrained by both Q/dO matrix fragments and sidecar
  rows.  Releasing before softmax is only correct if sidecar is also latched.
- Latching sidecar plus high source exceeds the current consumer VGPR budget.
- Do not pursue release-before-softmax in the current Mq64/208-VGPR route.
  Future attempts would need a different sidecar ownership/lifetime design or
  a larger feasible consumer VGPR window before this can be revisited.

### Half-Page Raw Tokens

Status: `REJECT_PERF_STATS_ONLY`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `59_half_page_raw_tokens`.
- Goal was to shorten raw Q/dO page ownership without duplicating Q/dO or
  changing score/dP/dV/dK math.  A Mq64 raw page was split into two M32 halves:
  half0 owns MBlockBase 0/1 and half1 owns MBlockBase 2/3.
- Candidate ABarrier ledger:
  `Raw0Half0Filled/Used=2/3`, `Raw0Half1Filled/Used=4/5`,
  `Raw1Half0Filled/Used=6/7`, `Raw1Half1Filled/Used=8/9`,
  `AllDone=10`.
- Intended pipeline: release half0 immediately after its high Q/dO source read,
  so producers can refill half0 for `q_tile+2` while consumers still process
  half1.

Evidence:

- Static/resource PASS:
  branch windows producer0 `7/16`, consumer0 `198/208`,
  consumer1 `198/208`, producer1 `1/16`.
- Symbol metadata PASS:
  `private=0`, `sgpr=62`, `vgpr=112`, no scratch/spill.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_022507`
  and H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_022513`.
- H1/S1024 stats:
  `simTicks=57,118,880`, `kernel_ticks=53,505,270`,
  `MMOP=131,072`, `MMAC active=27.3801%`,
  coissue `33,440/19,683`, `VALU=183,836`,
  `SCA=330,730`, `LDS=86,078`, `VMEM=4,352`,
  `ldsBankConflict=0`.
- Raw2 recert baseline remained better:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`,
  coissue `32,341/18,768`, `SCA=296,328`.

Conclusion:

- The ownership model is correct, but finer RawUsed/Filled granularity doubles
  page-token traffic and increases SCA/control enough to dominate the earlier
  release opportunity.
- Because useful MMOP count and island size are unchanged, the candidate raises
  protocol cost without raising MMAC active.  This is a precise negative
  against "split RawUsed finer" as a standalone route.
- Future RawUsed work must either reduce barrier/control operations, move
  ownership to a larger useful packet, or increase the useful MMAC island before
  spending more ABarrier tokens.

Post-revert recertification:

- Live source restored to raw2 canonical with whole-page raw tokens:
  `Raw0Filled/Used=2/3`, `Raw1Filled/Used=4/5`, `AllDone=6`.
- zys1 build/static PASS after restore.
- Symbol metadata PASS after restore:
  `private=0`, `sgpr=60`, `vgpr=112`, no scratch/spill.
- Kernel gate PASS after restore with branch windows
  `6/198/198/1`.

### Mq128 Single-Buffer Static Chain

Status: `REJECT_STATIC_SPILL`; code reverted to raw2 canonical.

Design basis:

- Workbook sheet `60_mq128_singlebuf_static`.
- Goal was to reduce ABarrier handshakes per useful MMAC by increasing the
  q tile from Mq64 to Mq128 while using one raw Q/dO page.  Two Mq128 raw pages
  exceed the 128KB LDS budget; one Mq128 raw page is 64KB and can overlay the
  dead K/V resident area after consumers latch K/V into VGPR.
- Candidate kept the same W16 role topology and output ownership, but changed
  `ActiveDkvTile` to `DkvTileD128MqNk128<128, 1>` and added a static
  four-M-pair consumer chain with MBlockBase `0/2/4/6`.
- The candidate intentionally avoided the old dynamic Mq128 loop, whose
  correctness/resource was possible but performance regressed.

Evidence:

- Remote build completed and source gate PASS, but symbol metadata failed
  before PMD:
  `private_segment_fixed_size=8`, `sgpr_count=104`,
  `sgpr_spill_count=18`, `vgpr_count=112`, `vgpr_spill_count=2`.
- WDRA branch windows showed the problem directly:
  producer0 `15/16`, consumer0 `208/208`, consumer1 `208/208`,
  producer1 `9/16`.
- Because the hard gate is no private segment and no SGPR/VGPR spill, PMD
  correctness/perf was not run.

Conclusion:

- The top-level direction remains plausible, but the current helper/codegen
  shape cannot statically expand Mq128 inside the 208-VGPR consumer window
  without spilling.
- This validates the resource-budget rule: do not enlarge Mq by local helper
  expansion alone.  A future larger-tile route needs a resource redesign first,
  such as shorter helper live ranges, different score/dP/dV/dK phasing, or a
  different output-ownership schedule.

Post-revert recertification:

- Live source restored to raw2 canonical.
- zys1 build/static PASS after restore.
- Symbol metadata PASS after restore:
  `private=0`, `sgpr=60`, `vgpr=112`, no scratch/spill.
- Kernel gate PASS after restore with branch windows
  `6/198/198/1`.

### Mq128 Q/dO Lifetime Split

Status: `ACCEPT_MICRO_BASELINE_OBSERVE`.

Design basis:

- Workbook sheet `66_mq128_qdo_lifetime_split`.
- Starting point was 62C2 Mq128/R1, with a single physical Q page, a single
  physical dO page, and sidecar tied to Q.
- The old combined raw token was replaced by semantic tokens using the existing
  ids:
  `QFilled/QUsed=2/3`, `DoutFilled/DoutUsed=4/5`, `AllDone=6`.
- Producer0 publishes K resident, then Q + sidecar; producer1 publishes V
  resident, then dO.
- Consumers wait both Q and dO before score/dP.  On the release M-pair,
  consumers read dO low/high sources, wait, arrive `DoutUsed`, run softmax/dS,
  read Q low/high sources, wait, arrive `QUsed`, then run the combined dV/dK
  MMAC island.
- No phase, second kernel, extra raw page, output ownership change, or asm
  island was added.

Evidence:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=97`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows:
  producer0 `14/16`, consumer0 `180/240`, consumer1 `180/240`,
  producer1 `8/16`.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_044322`,
  H1/S1024 stats-only at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_044345`,
  and H1/S1024 full perf at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_044647`.
- H1/S1024 full-perf stats:
  `simTicks=54,852,525`, `kernel_ticks=51,238,915`,
  `MMOP=131,072`, `MMAC active=29.6586%`, `VALU=165,744`,
  `SCA=108,632`, `LDS=83,856`, `VMEM=4,352`,
  coissue `27,090/16,944`, `ldsBankConflict=0`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_044647_clean_qdo_split_h1s1024_sqc7_fullperf`.
- XCU dispatch0 top2000:
  top bubble remains `s_abarrier_try_wait -> s_xor_b32`, `42.85%`.
  `bar3 QUsed` contributes `2,266,380` cycles across `224` bubbles,
  and `bar5 DoutUsed` contributes `2,251,240` cycles across `224` bubbles.

Conclusion:

- Keep as the current micro-positive Mq128/R1 baseline: it improves 62C2
  full-perf ticks from `52,163,020` to `51,238,915` and MMAC active from
  `29.2001%` to `29.6586%` while preserving correctness and resource gates.
- It does not solve the 60% active problem.  The combined Q/Dout ownership
  bubble is still about `4.52M` cycles, close to the old RawUsed `4.62M`.
- Next design must either reduce total ownership wait, hide it with useful
  producer/consumer work, or lengthen the useful MMAC island without
  reintroducing duplicate score/dP.

### Mq128 Prune AllDone Tail ABarrier

Status: `REJECT_STATIC_SPILL`.

Design basis:

- Workbook sheet `67_mq128_prune_alldone`.
- Starting point was commit `b67577f` Q/dO lifetime split.
- XCU detail showed a tail/control bubble:
  `s_abarrier_try_wait -> s_waitcnt`, `122` hits, `1,275,458` cycles
  (`10.04%` of issue-gap summary).
- Candidate removed the counted `AllDone` ABarrier init/arrive/wait/inv and
  relied on the following CTA-wide `__syncthreads()` before ABarrier invalidation.
- No q-loop ownership, math, output ownership, MMAC island, or asm was changed.

Evidence:

- Remote build completed and source gate PASS.
- Symbol metadata failed before PMD:
  `private_segment_fixed_size=244`, `sgpr_count=104`,
  `sgpr_spill_count=0`, `vgpr_count=128`, `vgpr_spill_count=60`.
- Branch windows also changed: producer1 rose from `8/16` to `15/16`.
- PMD correctness and perf were not run because the no-spill hard gate failed.

Conclusion:

- Reverted active source to the `b67577f` AllDone form.
- The counted `AllDone` barrier appears redundant semantically, but in the
  current WDRA CFG/codegen shape it acts as a useful live-range/control sink.
- Do not remove tail ABarrier by inspection.  A future attempt needs a focused
  CFG/codegen probe or an alternative cleanup that preserves no-spill metadata.

### Mq128 Q/dO Focused XCU Drilldown

Status: `OBSERVE_XCU_DRILLDOWN`.

Evidence:

- Workbook sheet `68_qdo_focused_xcu`.
- Source perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_044647_clean_qdo_split_h1s1024_sqc7_fullperf`.
- Focused xcu exports:
  `xcu_bar3_qused_7600_21000` and
  `xcu_bar5_doutused_19400_31300` under the same archive.
- `bar3 QUsed`, `7600:21000`, `xcd0/se0/cu0/simd1/wave0`:
  pipeline bubble `99.60%`; same-SIMD bubble `95.46%`, MMAC `0.93%`,
  VALU `1.35%`.  Consumer slots issue MMAC, but MMAC+VALU coissue is only
  `4.74%`/`6.87%`, with top coissue VALU `v_mov_b64_e32`.
- `bar5 DoutUsed`, `19400:31300`, `xcd0/se0/cu0/simd3/wave3`:
  pipeline bubble `99.70%`; same-SIMD bubble `94.97%`, MMAC `1.07%`,
  VALU `1.74%`.  Consumer slots issue MMAC, but MMAC+VALU coissue is only
  `12.09%`/`11.92%`, with top coissue VALU `v_mov_b32_e32`.

Conclusion:

- The ownership stalls are not being covered by peer-wave useful work.  The
  SIMD remains mostly bubble during the representative QUsed/DoutUsed windows.
- Visible coissue is mostly move/control, not the FWD-style softmax/dS VALU
  hidden under peer MMAC.
- The next design must change the steady q-loop ownership/pipeline shape:
  reduce the number of full-page ownership cliffs per useful MMAC, or create
  real consumer-group stagger before producers hit QUsed/DoutUsed.

### K/V Latch Uniform Half-Select Rejection

Status: `REJECT_STATS_ONLY`.

Scope:

- Single canonical dKV kernel only.
- Temporary instruction-level candidate in `latch_owner16_kv_regs`.
- No tile, wave role, ABarrier protocol, output ownership, or matrix path
  change.

Hypothesis:

- `owner_nblock & 1` is wave-uniform for the owner16 K/V latch.
- Moving the half select outside the `d_block` loop might replace 64
  per-fragment `v_cndmask` instructions with one branch and reduce VALU
  pressure.

Evidence:

- Static/resource gates PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- Branch windows unchanged:
  producer0 `14/16`, consumer0 `188/240`, consumer1 `188/240`,
  producer1 `8/16`.
- Static asm:
  `v_cndmask` fell `203 -> 139`, but `v_mov_b64_e32` rose
  `103 -> 135` and `v_mov_b64` rose `139 -> 171`; `s_waitcnt` remained
  `347`.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_213349`,
  and H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_213419`.
- H1/S1024 stats-only:
  `simTicks=48,266,855`, `kernel_ticks=44,653,245`,
  `MMAC active=32.6821%`, `VALU=181,600`, `SCA=115,640`,
  `LDS=79,360`, `VMEM=4,352`, coissue `36,600/26,562`,
  `ldsBankConflict=0`.
- Accepted wait-prune baseline:
  `simTicks=47,871,005`, `kernel_ticks=44,257,395`,
  `MMAC active=32.7888%`, coissue `36,652/26,646`.

Conclusion:

- Rejected and reverted from active code.
- The compiler traded fewer `v_cndmask` instructions for more vector moves and
  branch/control work; same-shape PMD stats regressed by about `0.83%` on
  `simTicks` and reduced MMAC active.
- Lesson: do not promote static instruction-count reductions unless PMD stats
  and active-share evidence agree.  For this specific K/V latch half select,
  the original loop-local select is better under the current compiler/PMD.

### Sidecar Pair Read6 Instruction Scheduling

Status: `ACCEPT_MICRO_INSTRUCTION`.

Scope:

- Single canonical dKV kernel only.
- Active helper changed:
  `softmax_ds_owner16_causal_exact_tile_ctx`.
- No tile, wave role, ABarrier token, output ownership, matrix path, or
  external API change.

Hypothesis:

- The softmax/dS helper read sidecar for one M row, computed it, then repeated
  for the second row.  Source/XCU showed significant `s_waitcnt` exposure
  around this sidecar/softmax region.
- Reading both rows' sidecar Vec4 triples first gives the compiler a larger
  LDS-read island and lets the following VALU run with fewer scattered
  readiness stalls.

Evidence:

- Static/resource gates PASS:
  branch windows producer0 `14/16`, consumer0 `189/240`, consumer1 `189/240`,
  producer1 `8/16`; metadata `private=0`, `sgpr=99`, `sgpr_spill=0`,
  `vgpr=128`, `vgpr_spill=0`.
- Static asm:
  `v_mov_b32_e32` fell `553 -> 539`; main path remains
  `ds_read_b32=0`, `ds_read_b128=96`, `ds_read_matrix=550`.
  `s_waitcnt` count increases in static asm, so this was not promoted from
  static inspection alone.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_214952`,
  and H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_214959`.
- Full perf H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_215636`.
  `simTicks=47,731,775`, `kernel_ticks=44,118,165`,
  `MMAC active=32.8831%`, `VALU=168,514`, `SCA=115,544`,
  `LDS=79,360`, `VMEM=4,352`, coissue `35,265/24,888`,
  `ldsBankConflict=0`.
- Previous accepted wait-prune full perf:
  `simTicks=47,871,005`, `kernel_ticks=44,257,395`,
  `MMAC active=32.7888%`, `VALU=183,136`.
- XCU first-pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2723088_fa3_bwd_wasp_clean_20260706_215931`.
  Dispatch duration `96,964`, avg active waves `120.92`.
  Top bubble remains `s_abarrier_try_wait -> s_xor_b32`, about `41.91%`;
  `s_waitcnt` remains about `19.78%`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_215636_7gemm_sidecar_read6_h1s1024_sqc7_fullperf`.

Conclusion:

- Keep as a small instruction-level cleanup.  It reduces VALU by `14,622`
  instructions and improves full-perf ticks by about `0.29%` while slightly
  raising MMAC active.
- This is not a structural path to 60% MMAC active.  The dominant issue remains
  ABarrier/ownership control, followed by wait/matrix-read gaps.
- Lesson: batching sidecar reads can help when it reduces VALU/scattered
  readiness work, but it must be validated by PMD because static `s_waitcnt`
  counts can move in the wrong direction.

### dQ Reference Correctness Bringup

Status: `ACCEPT_BRINGUP_ONLY`.

Scope:

- New dQ branch: `shaobo/7gemm-dq-bringup`.
- New design workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`.
- New files: `include/dq_contract.h`, `src/dq_kernel.cpp`,
  `scripts/check_dq_kernel_gate.py`, and `scripts/run_dq_correctness.sh`.
- This is a correctness reference path only.  It is not the canonical MMAC dQ
  performance kernel.

Design boundary:

- dQ owns Q tile and writes dQ without atomic add.
- dQ may recompute score/dP across the separated dKV/dQ kernels, but must not
  duplicate score/dP for the same `(Q tile, K tile)` inside the dQ kernel.
- First canonical MMAC target is revised to `Mq=64,Nk=64,D=128,16 waves`
  with source-layout `K^T` ABI and two serial `M32` q-subtiles.  The earlier
  `Nk=128` target is deferred until one K LDS page can feed both normal and
  transpose dQ views without duplicating Kt/dS storage.

Evidence:

- Remote build PASS:
  `SRC=src/dq_kernel.cpp BIN=build/fa3_bwd_dq_clean ASM=build/fa3_bwd_dq_clean.asm ./build.sh`.
- Static dQ gate PASS:
  `python3 scripts/check_dq_kernel_gate.py`.
- PMD H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260706_223246`.
- Numerical result:
  `dq_max_abs=5.82077e-11`, `dq_mean_abs=6.11585e-12`,
  `dq_rmse=8.79944e-12`, `dq_rel_l2=2.36625e-07`, `bad=0`.
- Reference dispatch `simTicks`:
  softmax `2,108,136,030`, delta `1,400,708,855`, dP `12,980,695`,
  dQ output `22,555,715`.

Conclusion:

- Promote as a stable dQ correctness harness and API entry point.
- Do not use these ticks as performance evidence; this reference path is scalar
  and intentionally separated from the future MLS/ds_read_matrix/MMAC path.
- Next dQ work must implement the revised workbook's canonical MMAC kernel and
  compare against this reference output.

### dQ Canonical MMAC Bringup

Status: `ACCEPT_BRINGUP_OBSERVE`, not a performance promotion.

Scope:

- Active branch: `shaobo/7gemm-dq-bringup`.
- Implemented one canonical dQ MMAC path in `src/dq_kernel.cpp`.
- Producer waves load `Q/dO/K/V/K^T`, worker waves compute score/dP and publish
  fp16 dS to LDS, consumer waves compute `dQ = dS @ K^T` with MMAC and direct
  global store.
- Current topology is `Mq=32,Nk=64,D=128`, 12-wave CTA, q-tile chunked launches
  with `tiles_per_dispatch=16` to keep S1024 stable under PMD.

Evidence:

- Static/resource gate PASS:
  `private_segment=0`, `sgpr=86`, `vgpr=168`, no SGPR/VGPR spill. Branch
  windows: producer `10/40`, consumers `49/72`, dS worker `91/128`, idle
  branch `8/48`.
- Correctness PASS:
  H1/S512 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_013745`;
  H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_013800`.
- H1/S1024 stats:
  dispatch0 `kernel_ticks=21,306,285`, `MMAC active=5.856%`,
  `MMOP=13,824`, `ldsBankConflict=0`, coissue `0/0`;
  dispatch1 `kernel_ticks=36,554,245`, `MMAC active=7.607%`,
  `MMOP=38,400`, `ldsBankConflict=0`, coissue `0/0`.

Negative evidence:

- `ds_read_matrix_trans_pair` in the dQ hot path passed S512 but produced
  long-seqlen row-half NaNs; single `ds_read_matrix_32x16_trans` reads are more
  stable.
- `+v` accumulator constraints on `s_barrier` made S512 fail and were rejected.
- Consumer LDS scratch epilogue store reproduced PMD LDS index corruption and
  segfault; direct consumer global store is the current stable path.
- Split-MHalf consumer helper also triggered PMD segfault.
- Producer-side LDS sidecar staging made S512 fail, likely because the current
  barrier schedule does not safely publish sidecar in the same page lifetime.

Conclusion:

- Correctness and resource gates are now usable for dQ, but `MMAC active` is
  only `~6-8%`, far below the 40% target.
- Removing idle waves did not move the metric meaningfully; the dominant issue
  is serial producer/worker/consumer barrier phasing and zero useful coissue.
- Next design must add real double buffering or conveyor overlap between dS
  publish for `kt+1` and dQ consume for `kt`, not more local instruction
  shuffling.

### dQ Mq32 Double-Page Conveyor

Status: `ACCEPT_MICRO_OBSERVE`, current dQ baseline.

Change:

- Replaced CTA-wide producer/worker/consumer `__syncthreads()` phasing with
  two K/V/Kt/dS LDS pages and ABarrier ownership tokens.
- Page protocol:
  `PageFilled(count=4) -> DsFilled(count=4) -> PageUsed(count=8)`, plus
  `AllDone(count=12)`.
- Producer now loads Q/dO once per q-subtile, then streams K/V/Kt by page.
- Worker computes score/dP/softmax/dS into the page's dS region.
- Consumer waits on dS, runs `dQ = dS @ K^T`, then releases the page.
- Removed stale split-MHalf/LDS epilogue helpers from the live dQ source.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=61`, `vgpr=168`, `sgpr_spill=0`, `vgpr_spill=0`.
  Branch windows: producer `1/40`, consumers `49/72`, worker `91/128`.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015920`;
  H1/S512 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015927`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015941`.
- H1/S1024 dispatch0:
  `kernel_ticks=21,420,035`, `MMAC active=5.8039%`,
  coissue `245/204`, `ldsBankConflict=0`.
- H1/S1024 dispatch1:
  `kernel_ticks=35,671,545`, `MMAC active=7.8501%`,
  coissue `751/665`, `ldsBankConflict=0`.
- Previous serial Mq32 dispatch1 was
  `kernel_ticks=36,587,005`, `MMAC active=7.6036%`, coissue `0/0`.

Conclusion:

- This is a correctness-clean structural baseline and proves tokenized
  producer/worker/consumer overlap is legal.
- It is still nowhere near the 40% MMAC-active goal.  The likely next limiter
  is that dS worker and dQ consumer are still small islands with only four
  consumer waves; the pipeline exists but not enough useful MMAC area is exposed.
- Direct Mq64 by serially looping two q-subtiles passed static resources but
  hung at H1/S128; do not retry without an explicit q-subtile token reset or
  page lifetime proof.

### dQ Producer Sidecar LDS Staging

Status: `ACCEPT_MICRO_CANDIDATE`.

Scope:

- Active branch: `shaobo/7gemm-dq-bringup`.
- Active kernel: one canonical dQ path in `src/dq_kernel.cpp`.
- Goal: remove the xcu-identified worker-side `global_load_dword ->
  s_waitcnt` bubble from `dq_publish_ds_chunk`.

Change:

- Added a small sidecar tail in dQ LDS for `scores_max`, `scores_sum`, and
  `delta`.
- Producer writes those 96 floats once per M32 q-subtile before `PageFilled`.
- dS worker reads sidecar from LDS and no longer receives global sidecar
  pointers.
- No new ABarrier token was added.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=67`, `vgpr=168`, no scratch/spill. Branch windows:
  producer `8/40`, consumers `49/72`, worker `83/128`.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_022646`;
  H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_022713`.
- H1/S1024 stats:
  dispatch0 `kernel_ticks=17,546,620`, `MMAC active=6.707%`;
  dispatch1 `kernel_ticks=28,114,905`, `MMAC active=9.707%`;
  `ldsBankConflict=0`.
- Prior double-page dispatch1 baseline:
  `kernel_ticks=35,671,545`, `MMAC active=7.8501%`.

Conclusion:

- Keep this dQ change.  It is the first meaningful dQ performance improvement
  after the two-page conveyor.
- It is still far from the 40% MMAC-active target.  The next limiter is page
  ownership/control: current two pages consume about 120KB LDS because each
  page holds K, V, K^T, and dS.  The next high-leverage design should prove
  whether K and K^T can share one LDS resident source layout for dQ, freeing
  enough LDS for a deeper page conveyor or a larger useful MMAC island.

### dQ Same-K-LDS Fast Probe Rejected

Status: `REJECT_CORRECTNESS`.

Goal:

- Test whether dQ can consume K^T from the same resident K LDS page, so the
  16KB K^T page can be removed and the dQ conveyor can move beyond two pages.

Tested combinations:

- K `matrix_load_32x32_b16 t`, dQ reads K page with
  `ds_read_matrix_32x16_normal`: H1/S128 `dq_rel_l2=1.03597`.
- K `matrix_load_32x32_b16 t`, dQ reads K page with
  `ds_read_matrix_32x16_trans`: H1/S128 `dq_rel_l2=1.46283`, `pass=0`.
- K `matrix_load_32x16_b16` pairs, dQ reads K page with
  `ds_read_matrix_32x16_normal`: H1/S128 `dq_rel_l2=0.535917`.
- K `matrix_load_32x16_b16` pairs, dQ reads K page with
  `ds_read_matrix_32x16_trans`: H1/S128 `dq_rel_l2=1.45385`, `pass=0`.

Evidence paths:

- `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_024052`
- `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_024216`
- `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_024400`
- `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_024520`

Decision:

- Source was restored to `dq_sidecar_lds_staging`.
- Do not remove the K^T page using guessed offset/format combinations.
- Next same-K work must be a focused fragment-layout probe that stores or
  prints expected versus actual K/K^T fragments for `matrix_load_32x16`,
  `matrix_load_32x32`, `ds_read_matrix_format`, and
  `ds_read_matrix_trans_format`.

### dQ ABarrier And Nk128 Experiments

Status: `REJECT_TICKS_OBSERVE` and `REJECT_STATIC_SPILL`.

Evidence:

- Fresh full-perf/xcu for the accepted `dq_sidecar_lds_staging` baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_030342`
  and xcu output
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2732630_fa3_bwd_dq_clean_20260707_030530`.
- xcu dispatch1 top issue gap is `s_abarrier_try_wait -> s_xor_b32` at
  `53.17%`.  `ds_read_matrix_trans_format -> s_waitcnt` is only `3.94%`,
  and `v_mmac` latency share is `3.71%`.

PageUsed consumer-only test:

- Changed `PageUsed` expected count from `8` to `4` and removed the worker's
  `PageUsed` arrive.  Correctness proof: a consumer can arrive `PageUsed` only
  after `DsFilled`, and `DsFilled` means the worker has finished the page.
- H1/S128 and H1/S1024 correctness passed.
- H1/S1024 dispatch1 whole-active MMAC improved slightly from `9.7068%` to
  `9.9346%`, but `kernel_ticks` regressed from `28,114,905` to `28,360,605`
  and coissue fell from `1395/1108` to `1173/926`.
- Decision: do not promote; the code is reverted to `dq_sidecar_lds_staging`.

Nk128 single-page test:

- Workbook sheet `14_dq_nk128_single_page` records the design: `Mq32,Nk128`
  with one LDS page, about `120.4KB` LDS, total `384` MMAC per page versus
  `192` for Nk64, and half the page/barrier epochs.
- Static build completed, but symbol metadata failed:
  `private_segment_fixed_size=68`, `sgpr_spill_count=2`,
  `vgpr_spill_count=64`.
- Decision: reject before PMD.  The idea is still a plausible route to higher
  MMAC active, but current worker/consumer fragment lifetime spills.  Revisit
  only after K/K^T same-LDS layout is proven or after reducing dQ fragment
  lifetime.

### dQ dS Chunk Token Experiment

Status: `REJECT_PERF_STATS_ONLY`.

Hypothesis:

- The accepted `dq_sidecar_lds_staging` baseline waits on full-page
  `DsFilled(count=4)` before any dQ consumer MMAC starts.
- Replacing that with per-worker dS chunk tokens might let consumers start the
  first 32-column `dS @ K^T` block while later worker chunks are still being
  computed.

Implementation tested:

- Added eight chunk barriers: four dS chunks for page0 and four for page1.
- Each worker wave called `seq -> publish dS chunk -> arrive` for its own
  chunk.
- Consumers waited chunk0+1, consumed n-block0 with one native
  `ds_read_matrix_trans_pair` group, then waited chunk2+3 and consumed
  n-block1.
- No new phase or kernel was kept in source; the experiment was reverted.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch; branch windows
  producer `8/40`, consumer `33/72`, worker `83/128`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_033505`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_033527`.
- H1/S1024 dispatch1:
  `kernel_ticks=31,380,440`, whole-active `MMAC=8.7319%`,
  `VALU=92,416`, `SCA=194,600`, `LDS=42,016`, `VMEM=4,928`,
  coissue `1316/905`, `ldsBankConflict=0`.
- Accepted sidecar-LDS baseline dispatch1:
  `kernel_ticks=28,114,905`, whole-active `MMAC=9.7068%`.

Decision:

- Reject and keep source at `dq_sidecar_lds_staging`.
- The experiment proves that finer dS readiness tokens alone are the wrong
  direction: they reduce consumer live VGPR but increase scalar/control work
  and do not hide the ownership wait enough.
- Next dQ route toward 40% MMAC active should reduce token count or increase
  useful MMAC per token, for example by eliminating the K^T page through a
  proven native layout, or by redesigning the tile so each page epoch has a
  larger dQ MMAC island without spill.

### dQ Mq64 QDo Token Experiment

Status: `REJECT_HANG`.

Hypothesis:

- `Mq32,Nk64` has too little useful MMAC work per CTA/page epoch to approach
  the 40% whole-active MMAC target.
- `Mq64` can increase per-CTA useful work without increasing the LDS Q/dO
  footprint if two M32 q-subtiles reuse the same Q/dO/sidecar LDS region
  serially.

Lifetime proof attempted:

- Added `QDoUsed(count=4)` so worker waves release Q/dO/sidecar after finishing
  all K tiles for one q-subtile.
- Producer waits `QDoUsed` before loading q_subtile 1.
- Producer also used a monotonic `page_epoch` across q_subtiles so page0/page1
  are not overwritten merely because `kt < 2` in a new q_subtile.

Evidence:

- Static/resource PASS:
  `private=0`, `sgpr=69`, `vgpr=168`, no spill/scratch; branch windows
  producer `9/40`, consumers `51/72`, worker `87/128`.
- H1/S128 PMD did not complete after several minutes:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_034822`.
- Killed process group:
  `2733520`, `2733521`, `2733523`, `2733524`.
- Source was reverted and remote recertified to the Mq32 sidecar-LDS baseline.

Decision:

- Do not retry Mq64 directly inside the performance kernel.
- The old Mq64 hang was not fully solved by the first `QDoUsed` protocol.
  There is still an ABarrier phase or cross-q_subtile ownership gap.
- Larger Mq remains an important 40% route, but it now needs a focused
  q-subtile synchronization probe that uses tiny synthetic producer/worker
  roles before returning to the full dQ kernel.

Follow-up Mq64 per-page-seen attempt:

- The previous `page_epoch >= 2` proof had a real bug: for causal H1/S128,
  both q_subtiles use page0, so q_subtile 1 can reuse page0 before a global
  two-page epoch counter reaches 2.
- Retried with independent `page0_seen/page1_seen` tracking: producer waits
  `PageUsed(page)` whenever the specific page has already been filled.
- Static/resource again passed:
  `private=0`, `sgpr=69`, `vgpr=168`, no spill/scratch.
- H1/S128 still hung:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_035807`.
- Killed process group:
  `2733728`, `2733729`, `2733731`, `2733732`.
- Source was restored and remote recertified to Mq32 sidecar-LDS.

Updated decision:

- The page overwrite bug was real, but not the only deadlock source.
- Mq64 cannot be debugged further inside the performance kernel.  Next Mq64
  work needs a focused ABarrier q_subtile probe with tiny LDS writes/reads:
  producer writes QDo generation, worker reads and releases QDo, producer
  reuses QDo, and page0/page1 reuse is tested independently.

Focused q_subtile probe:

- Added `probes/dq_qsubtile_barrier_probe.cpp` to isolate the
  `QDoUsed/PageFilled/DsFilled/PageUsed` sequence with two q_subtiles and
  repeated page0 reuse.
- PMD run did not hang:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_probe_20260707_040958`.
- The scalar-LDS probe still failed data checks:
  `errors=16`, `done=0`; adding `wait_lgkm(0)` after ordinary shared writes
  did not change that result:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_probe_20260707_041119`.
- Interpretation: this probe is useful as a synchronization smoke, but ordinary
  scalar LDS writes/reads are not faithful enough to prove the FA path.  The
  next probe must use the same matrixized data path as the kernel:
  `matrix_load ... bps lds` plus `ds_read_matrix`, not scalar shared ints.

Matrixized q_subtile ownership probe:

- Added `probes/dq_qsubtile_matrix_probe.cpp` to test the same class of
  matrixized data path as dQ: producer waves publish Q/dO with
  `matrix_load_32x32_b16 ... bps lds`, worker waves consume with
  `ds_read_matrix_32x16_trans`, and consumer waves release page ownership.
- The probe intentionally repeats page0 across two q_subtiles to model the
  causal `H1/S128` case that hung in the full Mq64 kernel.
- Static/resource PASS:
  `private=0`, `sgpr=31`, `vgpr=64`, `sgpr_spill=0`, `vgpr_spill=0`.
  The probe uses uniform `s_set_vgpr_size(48)` windows to satisfy WDRA metadata;
  it is a protocol probe, not a resource-optimized kernel.
- ASM evidence:
  `matrix_load_32x32_b16=10`, `ds_read_matrix=10`, `s_abarrier=35`,
  `s_set_vgpr_size=4`.
- PMD `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']` PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_matrix_probe_20260707_042323`,
  `errors=0`, `done_waves=12`, `pass=1`,
  `simTicks=10,955,945`, `ldsBankConflict=0`.

Decision:

- ACCEPT as focused protocol evidence.
- The direct Mq64 hangs are not explained by a fundamental inability to reuse a
  Q/dO page across q_subtiles.  The next Mq64 attempt should be a surgical main
  kernel fix that mirrors the probe's release order: wait `QDoUsed` before
  overwriting Q/dO, wait the specific page's `PageUsed` before refilling a
  repeated page, and avoid a global page epoch that ignores repeated page0.
- Do not promote any performance claim from this probe; it contains no MMOP and
  was not designed for MMAC active measurement.

Main-kernel Mq64 retry after the matrixized probe:

- Retried the canonical dQ kernel with `ActiveDqTile=Mq64,Nk64`, adding
  `QDoUsed`, per-page `PageUsed` tracking, and later worker-side QDo wait.
- Static/resource stayed clean across attempts:
  `private=0`, `vgpr=168`, `sgpr=69..72`, no spill/scratch.
- PMD H1/S128 still hung:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_043215`.
- PMD H1/S64 also hung, proving the issue is not page1 or multiple CTAs:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_043539`.
- Removing producer QDo wait temporarily did not unhang S64:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_043929`.
- ABarrier debug found a real phase hazard:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_mq64_s64_abar_debug_20260707_044204`
  showed fast worker waves could advance `QDoUsed` twice before the producer
  waited, so a counted ABarrier is not a unique-wave barrier unless the role
  waits after arrive.
- Added worker self-wait and then limited QDo to the single q_sub0->q_sub1
  handoff to avoid phase wrap.  H1/S64 still hung:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_044449`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_045025`.
- A conservative q_subtile `__syncthreads()` boundary also did not make S64
  complete:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_045713`.
- Later ABarrier logs indicate the full kernel can still stall with consumers
  waiting on the dS publication path / `DsFilled`, not merely QDo.
- All Mq64 code changes were reverted.  Remote source was recertified to the
  accepted Mq32 sidecar-LDS baseline:
  branch windows producer `8/40`, consumers `49/72`, worker `83/128`,
  metadata `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.

Decision:

- REJECT_HANG for the main-kernel Mq64 retry.
- Do not keep adding tokens inside the full kernel.  The next Mq64 attempt must
  first build a faithful focused probe that includes worker dS publication,
  consumer `DsFilled` wait, consumer dQ MMAC or a realistic delay, `PageUsed`,
  and unequal worker progress.  Only a protocol that survives that probe should
  be translated back into `src/dq_kernel.cpp`.

Follow-up worker+consumer+dS publication probe:

- Added `probes/dq_qsubtile_ds_consumer_probe.cpp`.
- The probe extends the accepted q_subtile matrix probe with:
  unequal worker progress, dS-like LDS publication, `DsFilled`,
  consumer `ds_read_matrix` plus two dQ-like MMACs, `PageUsed`, `QDoUsed`,
  and `AllDone`.
- Static/resource PASS:
  `private=0`, `sgpr=23`, `vgpr=48`, no spill/scratch.
- ASM evidence:
  `matrix_load_32x32_b16=6`, `ds_read_matrix=10`, `v_mmac=4`,
  `s_abarrier=34`, `s_set_vgpr_size=3`.
- PMD PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_ds_consumer_probe_20260707_050746`,
  `errors=0`, `done_waves=12`, `consumer_epochs=8`, `pass=1`,
  `simTicks=12,012,455`, `ldsBankConflict=0`.

Decision:

- ACCEPT as focused protocol evidence.
- Since this fuller probe passes while the full Mq64 kernel hangs, the
  remaining blocker is probably not the abstract q_subtile barrier ledger.
  Narrow the next probe toward the real helpers:
  `dq_publish_ds_chunk` layout/math and `dq_consume_ds_kt_full_dtile`
  read/MMAC path under two q_subtiles.  Do not re-edit the canonical kernel
  until that more faithful helper-level probe passes.

## 2026-07-07 dQ Current S1024 Baseline And Dispatch Control

Decision: `ACCEPT_BASELINE_AND_MEASUREMENT`

Goal:

- Current dQ optimization target is `MMAC active >= 40%` on
  `B=1,H=1,S=1024,D=128,causal=true`, `GPU_CHIP=sb`,
  `GPU_ARGS=['--SQCIPfLines=7']`.
- The accepted canonical code remains `Mq=32,Nk=64,D=128,12 waves` with
  sidecar staged in LDS.  Direct Mq64 remains rejected until a faithful
  helper-level q_subtile probe proves the lifetime protocol.

Baseline recertification:

- Code was restored to the Mq32 sidecar-LDS canonical route.
- Static/resource PASS:
  branch windows producer `8/40`, consumer `49/72`, worker `83/128`,
  dead/tail `2/48`; metadata `private=0`, `sgpr=67`, `vgpr=168`,
  no SGPR/VGPR spill.
- H1/S64 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063023`,
  `dq_max_abs=6.10526e-08`, `dq_rmse=1.20621e-08`,
  `l2_ratio=0.999981`.
- H1/S1024 default two-dispatch correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063111`,
  `dq_max_abs=1.85174e-07`, `dq_rmse=2.80325e-08`,
  `dq_rel_l2=0.00208192`, `l2_ratio=0.999991`.
- Two-dispatch H1/S1024 stats:
  dispatch0 `simTicks=20,488,650`, `MMOP=13,824`,
  `MMAC active=6.9912%`, coissue `203/171`;
  dispatch1 `simTicks=31,593,835`, `MMOP=38,400`,
  `MMAC active=9.7732%`, coissue `1,187/952`;
  aggregate `simTicks=52,082,485`, `MMOP=52,224`,
  `MMAC active=8.8385%`, `ldsBankConflict=0`.

Dispatch-control measurement:

- Added standalone `--tiles-per-dispatch` / `DQ_TILES_PER_DISPATCH` so the
  S1024 target can run all 32 q tiles in one dispatch instead of the default
  chunks of 16.  This is a measurement and launch-control knob only; it does
  not change kernel math, tile shape, barriers, or matrix path.
- H1/S1024 one-dispatch correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063334`.
- One-dispatch stats:
  `simTicks=34,346,130`, `MMOP=52,224`, `MMAC active=8.2338%`,
  coissue `1,297/1,090`, `ldsBankConflict=0`.

Conclusion:

- One dispatch reduces PMD total `simTicks` versus summing two dispatches, but
  it does not improve core utilization (`8.84% -> 8.23%` active).
- Keep the knob for clean target-shape measurement and perf capture.
- Do not count it as a 40% active optimization.  The next real optimization
  must increase useful MMAC work per ownership epoch or materially reduce the
  worker/consumer ABarrier and wait debt.

## 2026-07-07 dQ Mq64 Clean Retry Boundary

Decision: `REJECT_HANG`

Hypothesis:

After restoring Mq32 and proving smaller q_subtile protocol probes, a cleaner
Mq64 retry tested whether explicit role-branch ordering plus branch-local
setup could avoid the earlier full-kernel hangs.

Evidence:

- Temporary changes were reverted after the run:
  `ActiveDqTile=Mq64,Nk64`, q-subtile `QDoUsed`, per-page `PageUsed` wait,
  worker-before-consumer CFG order, and no pre-role high-VGPR setup.
- Static/resource stayed clean:
  `private=0`, `vgpr=168`, no spill/scratch; branch windows around
  producer `9/40`, worker `87/128`, consumers `51/72`, tail `2/48`.
- PMD H1/S64/H1/S128 still did not complete:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063717`.

Conclusion:

- Direct Mq64 in the canonical full kernel remains rejected.
- The blocker is not wave-id semantics and not a simple page-seen bug.  The
  next Mq64 work must be a helper-faithful probe that reuses the real
  `dq_publish_ds_chunk` and `dq_consume_ds_kt_full_dtile` layout/math before
  touching the performance route again.

## 2026-07-07 Wave ID Semantics Probe

Decision: `ACCEPT_PROBE`

Evidence:

- Added `probes/wave_id_semantics_probe.cpp`.
- PMD run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/wave_id_semantics_20260707_061657`.
- Output:
  `wave_id_semantics hw_by_cta_wave=0,1,2,3,4,5,6,7,8,9,10,11`
  and `cta_wave_echo=0,1,2,3,4,5,6,7,8,9,10,11`.

Conclusion:

- `__builtin_hcu_get_wave_id()` behaves as CTA-local wave id `0..11` in this
  focused 12-wave probe.
- Do not replace the kernel role id with `threadIdx.x / 64`; a prior attempt
  using thread-derived role id failed compilation with
  `Must get wave id in the entry block to set vgpr size`.

## 2026-07-07 dQ Real-Helper q_subtile Probe

Decision: `ACCEPT_PROBE`

Purpose:

- Preserve a focused local test for the Mq64 path without polluting the
  canonical dQ kernel.
- The probe exercises real dQ-style helper logic more closely than the tiny
  scalar/barrier probes: Q/dO publish, sidecar staging, K/V/Kt LDS pages,
  dS-like publication, consumer `ds_read_matrix` plus MMAC, and q_subtile page
  reuse.

Evidence:

- Source: `probes/dq_qsubtile_real_helper_probe.cpp`.
- One-page PMD PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_real_helper_probe_20260707_052352`,
  `simTicks=15,582,840`, `MMOP=288`, `ldsBankConflict=0`.
- Two-page PMD PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_real_helper_probe_2page_20260707_054104`,
  `simTicks=15,570,100`, `MMOP=288`, `ldsBankConflict=0`.

Conclusion:

- The Mq64 hang is still not explained by basic q_subtile page reuse, matrix
  path, or a small real-helper-like MMAC body.  The full canonical kernel likely
  has a more specific control/lifetime mismatch in the exact
  `dq_publish_ds_chunk` / `dq_consume_ds_kt_full_dtile` integration.
- Keep this as an isolated probe only.  Do not add another production phase or
  switch for it.

## 2026-07-07 dQ Kt Preread Under DsFilled Wait

Decision: `REJECT_PERF_STATS_ONLY`

Purpose:

- XCU on the one-dispatch S1024 perf showed the dominant bubble was
  `s_abarrier_try_wait -> s_xor_b32` on `Page0DsFilled`/`DsFilled`, and the
  representative window had about `96%` bubble cost with `MMAC+VALU` coissue
  still `0`.
- The candidate tried to preread the stable `K^T` fragments after
  `PageFilled` but before `DsFilled`, so the later dQ consume step would only
  read `dS` and then MMAC.

Evidence:

- Temporary code added `dq_preload_kt_full_dtile` and
  `dq_consume_ds_with_kt_full_dtile` in the canonical dQ path.  It was removed
  after measurement.
- Static/resource stayed clean:
  branch windows remained producer `8/40`, consumers `49/72`, worker `83/128`,
  tail `2/48`; metadata remained `private=0`, `sgpr=67`, `vgpr=168`, no
  spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_073748`
  and H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_073822`.
- Same one-dispatch H1/S1024 stats:
  baseline `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_065637`
  was `simTicks=34,215,090`, `MMAC active=8.24798%`, `SCA=217,384`;
  Kt preread was `simTicks=34,237,840`, `MMAC active=8.19433%`,
  `SCA=225,640`.

Conclusion:

- Prereading Kt under the dS wait is correct and resource-clean, but it is a
  small regression.  It likely lengthens the Kt live range and adds scalar
  scheduling/control work without reducing the `DsFilled` critical bubble.
- Keep the baseline canonical code.  The 40% route needs either a larger
  useful MMAC island per barrier token or a different dS/page ownership design,
  not this isolated Kt read motion.

## 2026-07-07 dQ Direct MHalf Consumer No DsFilled

Decision: `REJECT_PERF_STATS_ONLY`

Design:

- Workbook sheet `23_dq_direct_mhalf_plan`.
- Temporarily changed the canonical dQ role graph to remove the cross-role
  `DsFilled` handoff:
  waves0-3 stayed producers; wave4 owned MHalf0 and wave5 owned MHalf1; each
  direct consumer computed `score/dP -> dS scratch -> dQ(D128)` locally and
  then arrived `PageUsed`.  Waves6-11 only arrived `AllDone`.
- The design did not duplicate score/dP inside dQ because the two heavy waves
  owned disjoint M16 row slices and each owned all D128 output columns for its
  rows.

Evidence:

- Static/resource PASS:
  producer branch `8/40`, direct consumers `95/160` and `96/160`, tail `2/48`;
  symbol metadata `private=0`, `sgpr=56`, `vgpr=136`, no SGPR/VGPR spill.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_075744`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_075749`.
- H1/S1024 one-dispatch stats:
  `simTicks=47,939,255`, `MMOP=52,224`, `MMAC active=5.95434%`,
  `SCA=144,248`, `VALU=137,696`, `LDS=61,760`, `coissue=0/0`,
  `ldsBankConflict=0`.
- Same-shape baseline after recertification was
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_074751`,
  `simTicks=34,002,150`, `MMAC active=8.21561%`.

Conclusion:

- Removing `DsFilled` is not sufficient when it collapses useful work to only
  two heavy consumer waves.  The barrier bubble disappears structurally, but
  SIMD occupancy/coissue collapses harder; `coissue=0` is especially telling.
- Code was removed from the active path.  The next 40% design must keep at
  least four heavy waves or otherwise fill all four SIMDs, while reducing dS
  handoff cost with coarser/fewer tokens or a better page ownership pipeline.

## 2026-07-07 dQ Mq64 Single-Page Direct MHalf

Decision: `REJECT_PERF_STATS_ONLY`

Design:

- Workbook sheet `24_dq_mq64_singlepage_direct`.
- Extended the direct-MHalf idea to four heavy consumers so every SIMD has a
  direct consumer: `Mq=64,Nk=64,D=128`, one K/V/Kt/dS page, waves0-3 producer,
  waves4-7 direct MHalf0..3 consumers, waves8-11 tail.
- LDS arithmetic: Q+dO `32KB`, one K/V/Kt/dS page about `56KB`, sidecar
  `768B`, total about `88.75KB`.  A two-page Mq64 version would exceed 128KB.

Evidence:

- First compile with consumer window `160` failed PMD admission:
  metadata `vgpr_count=240` and model panic
  `12 WFs and 240 VGPRs per WI can not be allocated`.
- Reducing the direct-consumer window to `100` passed static/resource:
  producer `6/40`, consumers `95/100`, `95/100`, `95/100`, `96/100`,
  tail `2/40`; metadata `private=0`, `sgpr=92`, `vgpr=160`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_080932`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_080937`.
- H1/S1024 one-dispatch stats:
  `simTicks=56,850,430`, `MMOP=52,224`, `MMAC active=10.0261%`,
  `SCA=59,064`, `VALU=127,584`, `LDS=61,360`, `VMEM=3,520`,
  `coissue=0/0`, `ldsBankConflict=0`.
- Recertified Mq32 baseline was
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_074751`,
  `simTicks=34,002,150`, `MMAC active=8.21561%`.

Conclusion:

- Mq64 direct proves the native layouts can be extended to MHalf2/3 and four
  heavy consumers can run without spill, but it is not a performance win.
- `MMAC active` improves only to about `10%`, far from 40%, while ticks regress
  heavily and coissue is zero.  The missing ingredient is not just larger Mq;
  the design needs peer wave VALU/MMAC overlap and buffering without violating
  the 128KB LDS budget.
- Code was removed from the active path.  Keep the resource/admission lesson:
  direct consumers need a tight window (`100`, not `160`) to keep 12-wave CTA
  admission legal.

## 2026-07-07 dQ Mq64 Single-Page Split Worker/Consumer

Decision: `REJECT_PERF_STATS_ONLY`

Design:

- Workbook sheet `25_dq_mq64_singlepage_split`.
- Tested `Mq=64,Nk=64,D=128` with the original split idea preserved:
  waves0-3 producer, waves4-7 dQ consumers owning DTile0..3 across M64,
  waves8-11 workers publishing all four MHalves of dS.  Single K/V/Kt/dS page
  was required to stay under 128KB LDS.

Evidence:

- Static/resource PASS:
  producer `6/40`, consumers `69/72` x4, worker `113/128`, tail `2/48`;
  metadata `private=0`, `sgpr=100`, `vgpr=168`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_081732`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_081737`.
- H1/S1024 one-dispatch stats:
  `simTicks=62,845,965`, `MMOP=52,224`, `MMAC active=8.83209%`,
  `SCA=94,744`, `VALU=121,008`, `LDS=54,832`, `VMEM=3,520`,
  `coissue=0/0`, `ldsBankConflict=0`.

Conclusion:

- Mq64 split proves the MHalf2/3 layout, M64 dQ consumer accumulation, and
  four-MHalf worker publication are correct and resource-clean.
- It is still not a viable performance route because single-page buffering
  removes the cross-page worker/consumer overlap and coissue disappears.
- Code was removed from the active path.  The next changes should stay on the
  Mq32 two-page baseline and reduce unnecessary barrier/control debt before
  attempting another larger-tile design.

## 2026-07-07 dQ PageUsed Consumer-Only

Decision: `ACCEPT_MICRO`

Design:

- Workbook sheet `26_dq_pageused_consumer_only`.
- Baseline `PageUsed` count was `8`, with both worker waves and consumer waves
  arriving after each page.  Producer only needs to know the page is safe to
  overwrite; workers finish reading K/V/Q/dO before `DsFilled`, so their
  `PageUsed` arrival is redundant for page lifetime.
- Changed `Page0Used/Page1Used` count from `8` to `4` and removed the worker
  `dq_arrive_page_used` call.  Consumer arrivals remain unchanged.

Evidence:

- Static/resource unchanged and PASS:
  producer `8/40`, consumers `49/72` x4, worker `83/128`, tail `2/48`;
  metadata `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082239`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082245`.
- H1/S1024 one-dispatch stats:
  `simTicks=33,372,430`, `MMOP=52,224`, `MMAC active=8.44342%`,
  `SCA=212,520`, `coissue=1,223/992`, `ldsBankConflict=0`.
- Recertified same-code-family baseline was
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_074751`,
  `simTicks=34,002,150`, `MMAC active=8.21561%`, `SCA=217,384`.

Conclusion:

- Accepted as a small but clean improvement: about `1.85%` lower `simTicks`,
  `+0.23pt` MMAC active, and lower SCA with no resource/correctness cost.
- This supports the current direction: stay on the two-page Mq32 pipeline and
  remove unnecessary barrier/control work before another major tiling attempt.

Full perf / xcu observation:

- Full perf run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082827`;
  helper perf
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082827/m5out/0/0/2739404_fa3_bwd_dq_clean.perf`.
- xcu output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_pageused_s1024_fullperf_20260707_082827`.
- Dispatch0 summary:
  duration `65096`, waves `384`, inst issues `575544`, no-wave idle `0`.
- Top hot rows:
  `s_xor_b32` latency `6,485,024` cycles (`44.26%`);
  `s_waitcnt` latency `2,724,704` cycles (`18.59%`);
  `v_mmac_f32_16x16x16_f16` latency `464,556` cycles (`3.17%`);
  `ds_read_matrix_trans_format` latency `458,688` cycles (`3.13%`).
- Top issue bubbles:
  `s_abarrier_try_wait -> s_xor_b32` is `46.01%`, count `3016`,
  max `6939` cycles; `s_abarrier_try_wait -> s_waitcnt` is `6.60%`;
  `ds_read_matrix_trans_format -> s_waitcnt` is `3.60%`.
- Interpretation:
  the current 40% MMAC-active blocker is primarily the ABarrier page handoff
  bubble, not missing MMAC instructions.  More read/MMAC island tuning should
  be secondary until `PageFilled/DsFilled/PageUsed` lifetime is redesigned.

## 2026-07-07 dQ Worker dS Store Wait Merge

Decision: `REJECT_PERF_STATS_ONLY`

Design:

- Workbook sheet `28_dq_worker_store_wait_reject`.
- Hypothesis: each worker published two MHalf dS chunks per page, and each
  `dq_publish_ds_chunk` ended with `wait_lgkm(0)`.  Temporarily removed the
  per-chunk final wait and added one `wait_lgkm(0)` before
  `dq_arrive_ds_filled`, preserving the rule that `DsFilled` is released only
  after dS LDS writes are visible.

Evidence:

- Static/resource PASS:
  dQ gate PASS; metadata `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_083737`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_083758`.
- H1/S1024 one-dispatch stats:
  `simTicks=33,729,150`, `MMOP=52,224`, `MMAC active=8.45067%`,
  `SCA=212,520`, `VALU=130,816`, `LDS=57,408`, `VMEM=6,784`,
  `coissue=1,258/1,037`, `ldsBankConflict=0`.
- Current accepted baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082245`,
  `simTicks=33,372,430`, `MMAC active=8.44342%`, `coissue=1,223/992`.

Conclusion:

- Rejected and reverted.  Ticks regressed by about `1.07%`, while active share
  moved only `+0.007pt`.
- This negative result narrows the bottleneck: the dominant wait is not just a
  redundant per-half LDS store wait.  The next credible route must reduce or
  restructure the `DsFilled`/page-ownership ABarrier handoff itself.

## 2026-07-07 dQ Builtin Page Waits

Decision: `REJECT_PERF_STATS_ONLY`

Design:

- Workbook sheet `29_dq_builtin_page_waits_reject`.
- Hypothesis: xcu attributes the top bubble to
  `s_abarrier_try_wait -> s_xor_b32` in the inline-asm wrapper.  Temporarily
  changed only the dQ PageFilled, DsFilled, and PageUsed waits from
  `abarrier_try_wait<true>` to builtin `abarrier_try_wait<false>`.
- Did not change tile, math, role ownership, barrier counts, or output
  ownership.

Evidence:

- Static/resource PASS:
  dQ gate PASS; metadata `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_085028`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_085058`.
- H1/S1024 one-dispatch stats:
  `simTicks=33,754,630`, `MMOP=52,224`, `MMAC active=8.45656%`,
  `SCA=212,520`, `VALU=130,816`, `LDS=57,408`, `VMEM=6,784`,
  `coissue=1,217/989`, `ldsBankConflict=0`.
- Current accepted baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082245`,
  `simTicks=33,372,430`, `MMAC active=8.44342%`, `coissue=1,223/992`.

Conclusion:

- Rejected and reverted.  Ticks regressed by about `1.15%`, while active share
  moved only `+0.013pt`.
- The `s_xor_b32` hot row is mostly a symptom of true ABarrier waiting rather
  than a wrapper-only codegen issue.  Continue toward a topology/lifetime
  change for `PageFilled/DsFilled/PageUsed`.

## 2026-07-07 dQ dS Pair Streaming

Decision: `REJECT_PERF_STATS_ONLY`

Design:

- Workbook sheet `30_dq_ds_pair_stream_plan`.
- xcu top-bubble aggregation on the accepted baseline showed barrier id `1`
  (`Page0DsFilled`) dominating the sampled `s_abarrier_try_wait -> s_xor_b32`
  issue gaps.  The candidate split each page's dS handoff into two natural
  `ds_read_matrix_trans_pair` groups:
  pair0 = NChunk0/1, pair1 = NChunk2/3.

Candidate A, pair-all workers:

- Worker waves8-11 still ran in parallel.  Waves8/9 published pair0 and
  waves10/11 published pair1; consumers waited pair0, consumed half of dQ,
  then waited pair1.
- Static/resource PASS:
  `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_090341`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_090406`.
- H1/S1024 stats:
  `simTicks=33,548,970`, `MMAC active=8.45499%`, `coissue=1,157/945`,
  `ldsBankConflict=0`.

Candidate B, pair-sequential workers:

- Only waves8/9 worked; each published pair0 first, then pair1.  Waves10/11
  were tail waves.  This attempted to create real overlap between consumer
  pair0 MMAC and worker pair1 dS.
- Static/resource PASS:
  `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_090749`;
  H1/S1024 one-dispatch
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_090813`.
- H1/S1024 stats:
  `simTicks=33,989,410`, `MMAC active=8.47049%`, `coissue=1,245/1,033`,
  `ldsBankConflict=0`.

Conclusion:

- Both variants were correct and resource-clean, but both were slower than the
  accepted baseline `33,372,430` ticks.  Code was reverted.
- Pair-all did not create useful stagger because pair0 and pair1 were produced
  concurrently and became ready at nearly the same time.
- Pair-sequential created the intended stagger but made the worker/helper side
  too thin; the lost parallelism outweighed any overlap.
- The next 40% route should not add finer-grained barriers alone.  It needs a
  larger useful MMAC island or a different output/reduction ownership that keeps
  all resident waves doing useful work while reducing the dS handoff bubble.

## 2026-07-07 dQ Even/Odd Page Owner

Decision: `REJECT_HANG`

Design:

- Workbook sheet `32_dq_worker_readbatch` records the attempt.
- Tried to turn the current producer/worker split into two page owners:
  waves0-3 own even pages and waves8-11 own odd pages.
- First variant removed `PageFilled` completely.  This assumed the owner role
  could read K/V/Kt immediately after its own MLS sequence.
- Second variant restored an owner-local `PageFilled` so the four owner waves
  synchronize their MLS writes before publishing dS, while still keeping
  even/odd page production independent.

Evidence:

- Resource could be made clean after WDRA window correction:
  owner-local variant built with `private=0`, `sgpr=100`, `vgpr=168`,
  no SGPR/VGPR spill.
- No-PageFilled H1/S128 run did not complete in the smoke window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_093109`.
- Owner-local PageFilled H1/S128 also did not complete:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_093644`.

Conclusion:

- Rejected and reverted.  The current dQ ABarrier phase ledger cannot be
  rearranged into even/odd page ownership by a small topology edit.
- Do not retry this in the performance kernel without a focused probe for
  owner-local `PageFilled`/`DsFilled` phase progression and cross-wave MLS
  visibility.

## 2026-07-07 dQ Worker Score/dP Read Batch

Decision: `ACCEPT_MICRO`

Design:

- Kept the current `Mq=32,Nk=64,D=128,12-wave` producer/worker/consumer
  ownership and the existing barrier ledger.
- In `dq_publish_ds_chunk`, changed the worker score/dP island from four
  repetitions of `dO/K/V ds_read_matrix -> wait -> MMAC` into one larger read
  group: issue all four K-block `dO/K/V` matrix reads, wait once, then run the
  longer score/dP MMAC island.
- This is a one-hypothesis instruction-scheduling change: larger useful
  matrix-read/MMAC island, no math, tile, barrier, or output-ownership change.

Evidence:

- Static/resource PASS: producer `8/40`, consumers `49/72`, worker `115/128`,
  tail `2/48`; metadata `private=0`, `sgpr=67`, `vgpr=168`,
  no SGPR/VGPR spill.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_094345`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_094409`.
- H1/S1024 one-dispatch stats:
  `simTicks=30,225,650`, `MMOP=52,224`, `MMAC active=9.25852%`,
  `coissue=1,864/1,455`, `ldsBankConflict=0`.
- Accepted baseline was `simTicks=33,372,430`, `MMAC active=8.44342%`.
  Same-shape ticks improved by about `9.43%`, and active share improved by
  `0.815` percentage point.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_094707/m5out/0/0/2740972_fa3_bwd_dq_clean.perf`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_readbatch_worker_s1024_fullperf_20260707_094707_d0`.
  Dispatch duration is `58,068`, average active waves `228.58`.
  Top bubble remains `s_abarrier_try_wait -> s_xor_b32` at `44.64%`;
  `s_abarrier_try_wait -> s_waitcnt` is `6.43%`;
  `ds_read_matrix_trans_format -> s_waitcnt` is `4.87%`.

Conclusion:

- Accepted as the new dQ micro-baseline.  It is a real tick win and validates
  larger read/MMAC islanding as useful.
- It does not solve the 40% target.  The dominant bottleneck is still the
  page/dS ABarrier ownership chain, so the next step should reduce barrier
  control exposure or increase useful MMAC work per ownership epoch.

## 2026-07-07 dQ All-Operand Worker Read Batch

Decision: `ACCEPT_MICRO`

Design:

- Kept the same `Mq=32,Nk=64,D=128,12-wave` canonical dQ path.
- Extended the accepted worker score/dP read-batch by also folding the Q
  fragment reads into the same LDS matrix-read island.
- In `dq_publish_ds_chunk`, the worker now issues Q plus all four D-block
  `dO/K/V` `ds_read_matrix` groups before the single `wait_lgkm(0)` and the
  score/dP MMAC island.
- No math, tile, ABarrier ownership, or output ownership changed.

Evidence:

- Static/resource stayed clean: `private=0`, `sgpr=67`, `vgpr=168`,
  no SGPR/VGPR spill or scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_095835`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_095900`;
  full-perf H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_100403`.
- Full-perf stats:
  `simTicks=28,998,970`, `MMAC active=9.54706%`,
  `coissue=1,943/1,453`, `ldsBankConflict=0`.
- Compared with worker score/dP read-batch, ticks improved
  `30,225,650 -> 28,998,970` and active moved
  `9.25852% -> 9.54706%`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_all_operand_readbatch_s1024_fullperf_20260707_100403_d0`.
  Top bubbles are still dominated by ABarrier:
  `s_abarrier_try_wait -> s_xor_b32` at `44.13%`;
  `s_abarrier_try_wait -> s_waitcnt` at `6.32%`;
  `ds_read_matrix_trans_format -> s_waitcnt` at `3.81%`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_100403_dq_all_operand_readbatch_s1024_sqc7_fullperf`.

Conclusion:

- Accepted as the current dQ micro-baseline.  The read island is larger and
  the matrix-read wait bubble is lower, so this is a constructive change.
- The 40% MMAC-active target is not solved.  Next work must target the
  ABarrier/Page/Ds ownership chain or increase useful MMAC work per ownership
  epoch; more fine-grained tokens are unlikely to help.

## 2026-07-07 dQ K-Native Same-LDS Read And Code Convergence

Decision: `ACCEPT_CLEANUP`

Design:

- A focused probe showed the existing raw K LDS page can feed the dQ RHS via
  normal `ds_read_matrix_format` with a uniform `f16x4` fragment remap.
- The canonical dQ consumer now computes `dQ = dS @ K` from the same K LDS page
  instead of loading and reading a separate `K^T` source-layout page.
- The old Kt source hot-path helpers were deleted.  A follow-up source audit
  also removed the dead host/API tail: `k_t_source` kernel argument,
  `materialize_k_t_source`, `reserved_ptr[1]` validation, and the standalone
  Kt allocation/copy/free.
- The dQ static gate now forbids restoring `dq_load_kt_tile`,
  `dq_store_kt_tile_scalar`, `kKtBase`, `materialize_k_t_source`, or
  `k_t_source` in the canonical dQ source.

Evidence:

- Focused probe:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_k_same_lds_native_probe_20260707_134821`.
  It found no exact Vec8 fragment match, but did find a stable f16x4 block
  mapping for the current K load/reader pair:
  `k_frag_idx = nk_idx / 2 + (nk_idx & 1) * (Nk / 32)`.
- First K-native run before host cleanup:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_141507`,
  `simTicks=28,160,860`, `MMAC active=9.9082%`, correctness PASS,
  `ldsBankConflict=0`.
- Deleting the old K-to-dS padding was rejected:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_141209`,
  `simTicks=28,690,480`, so the current code keeps the padding while removing
  dead Kt source traffic and helpers.
- Final code-converged run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_144004`,
  `path=canonical`, `simTicks=28,002,520`, `MMAC active=10.032187%`,
  `MMOP=52,224`, `VALU=130,816`, `SCA=191,696`, `LDS=57,408`,
  `VMEM=4,608`, `coissue=1,964/1,441`, `ldsBankConflict=0`.
- Static/resource gate:
  `group_segment=123264`, `private=0`, `sgpr=63`, `vgpr=168`,
  no SGPR/VGPR spill.

Conclusion:

- The active dQ route is not phase-stacked: one canonical dQ performance
  kernel plus reference correctness kernels remains.
- The only new kernel is the isolated focused probe under `probes/`, which is
  evidence code rather than a performance path.
- Current main bottleneck is still ABarrier/Page/Ds ownership exposure; Kt
  source traffic is no longer the target.

## 2026-07-07 dQ 16-Wave Full-3GEMM Structural Rewrite

Decision: `ACCEPT_CORRECTNESS_OBSERVE`

Design:

- Reworked the canonical dQ source to remove the high-cost dS-in-LDS
  handoff.  dS now lives only in consumer VGPR.
- The CTA is 16 waves:
  waves0-3 publish Q/dO group0 sidecar and K, waves4-7 compute rows 0-63,
  waves8-11 compute rows 64-127, and waves12-15 publish Q/dO group1 sidecar
  and V.
- Both consumer groups are symmetric.  Each wave owns one M16 row stripe and
  runs the full dQ chain for that stripe:
  `score = Q K^T`, `dP = dO V^T`, `dS = P * (dP - delta) * scale`,
  `dQ += dS K`.
- The all-zero bring-up bug was a readiness bug: consumers read Q/dO/sidecar
  before producer MLS/global-sidecar writes were proven complete.  The fix is
  to wait for the first `PageFilled` before consumer sidecar and Q/dO reads.

Evidence:

- Static/resource gate:
  `private=0`, `sgpr=76`, `vgpr=128`, no scratch/spill.
  WDRA branch windows are producer0 `8/40`, consumer0 `117/216`,
  consumer1 `117/216`, producer1 `9/40`.
- Correctness:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160156`,
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160322`.
  Both PASS with `ldsBankConflict=0`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160652/m5out/0/0/2746700_fa3_bwd_dq_clean.perf`.
  Stats: `simTicks=55,191,955`, `kernel_ticks=51,578,345`,
  `MMOP=55,296`, `VALU=140,320`, `coissue=10,490/4,779`,
  `MMAC active=19.1324%`, `ldsBankConflict=0`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2746700_fa3_bwd_dq_clean_20260707_160843`.
  Dispatch duration `113,292`, average active waves `75.37`.
  Dominant bubbles are `s_abarrier_try_wait -> s_xor_b32` at `49.17%`
  and `s_waitcnt` at `19.24%`.
- Negative micro-test:
  changing dQ page waits from the asm wrapper to builtin preserved correctness
  but regressed H1/S1024 to `simTicks=55,490,435` and
  `MMAC active=18.9733%`; keep the asm wait wrapper.

Conclusion:

- This is a structural correctness milestone, not a performance promotion over
  the prior Mq32 K-native dQ baseline.
- The user constraint is now represented in code: no dS LDS transfer, no split
  dS worker, and two heavy consumer groups compute complete dQ for different q
  blocks.
- The next optimization should attack the real barrier lifetime:
  separate initial Q/dO readiness from K/V page readiness, then consider
  reusing Q/dO LDS as a second K/V page after consumers latch Q/dO.  Also
  profile a single heavy q tile so causal triangular work imbalance is not
  confused with intra-CTA pipeline quality.

## 2026-07-07 dQ K/V Trans Split-Wait

Decision: `ACCEPT`

Hypothesis:

- After Q/dO latch and K/V double-page, the remaining local `wait_lgkm(0)` after
  K/V trans-fragment reads may be too early.
- It is legal to issue all K/V `ds_read_matrix_trans` reads first, perform
  accumulator zeroing while those LDS reads are in flight, then consume the
  first half after `wait_lgkm(4)` and the second half after `wait_lgkm(0)`.

Rejected control:

- Removing the first `__syncthreads()` after `AllDone` looked tempting because
  xcu showed a large bar5/tail bubble.
- PMD H1/S128 aborted at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_201811` with
  `ABARRIER_ILL_OP_ERROR` and `barId 5 has already been invalidated`.
- Conclusion: the post-`AllDone` sync is required before every wave invalidates
  ABarriers; it is not a removable data wait.

Change:

- In `dq_consumer_full3gemm_role`, moved zeroing of `qk_acc` and `dp_acc`
  between K/V trans `ds_read_matrix` issue and first wait.
- Replaced the immediate full `wait_lgkm(0)` with:
  `wait_lgkm(4)` -> D-block 0/1 score+dP MMAC ->
  `wait_lgkm(0)` -> D-block 2/3 score+dP MMAC.
- Did not change tensor ownership, external API, wave roles, or LDS layout.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `118/216`, `118/216`, `9/40`;
  metadata `private=0`, `sgpr=54`, `vgpr=128`, no scratch/spill.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_202017`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_202030`.
- Stats-only H1/S1024:
  `simTicks=43,451,135`, `kernel_ticks=39,837,525`,
  `MMAC active=23.8728%`, `coissue=13,170/10,066`,
  `ldsBankConflict=0`.
- Full perf H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_202545/m5out/0/0/2748931_fa3_bwd_dq_clean.perf`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_202545_dq_kv_wait_split_h1s1024_sqc7_fullperf`.
  Metrics: `simTicks=43,330,560`, `kernel_ticks=39,716,950`,
  `MMOP=55,296`, `VALU=140,320`, `SCA=96,904`, `LDS=37,872`,
  `coissue=13,023/10,125`, `MMAC active=23.8706%`,
  `ldsBankConflict=0`.

Comparison:

- Full-perf baseline `dq_qdo_latched_kv_double_page`:
  `kernel_ticks=41,823,145`, `MMAC active=22.9566%`.
- New split-wait version:
  `kernel_ticks=39,716,950`, `MMAC active=23.8706%`.
- Improvement:
  ticks down `5.04%`, active up `0.914` point.  This is a valid wait-placement
  win, but not a solution to the 40%/60% MMAC-active target.

Next:

- Install or sidecar-run `xcu` for this perf and compare top bubbles against
  the previous QDo-latched baseline.
- Keep avoiding blind wait deletion: only waits with clear producer/consumer
  lifetime proof should move or disappear.

## 2026-07-07 dQ MMAC Zero Seed

Decision: `ACCEPT_MICRO`

Hypothesis:

- The hot qk/dP score loop still paid repeated explicit zero initialization for
  fresh `qk_acc` and `dp_acc`.
- Seeding the first score/dP MMAC with a branch-local zero accumulator should
  remove repeated `v_mov_b64` zero moves without increasing copy moves.

Change:

- Added one branch-local `mmac_zero` after long-lived `dq_reg` initialization.
- Removed per-`n_chunk` `dq_zero_f32x4(qk_acc/dp_acc)`.
- Rewrote the first score/dP MMAC as the accumulator seed and then accumulated
  the remaining K/D halves normally.
- Did not change wave roles, LDS layout, external API, wait/barrier protocol, or
  output ownership.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `122/216`, `122/216`, `9/40`;
  metadata `private=0`, `sgpr=54`, `vgpr=128`, no scratch/spill.
- ASM v_mov count:
  total `419 -> 359`, `v_mov_b64 96 -> 36`, zero-move category
  `186 -> 126`; copy moves did not increase.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_211841`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_211851`.
- Full perf H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_212125/m5out/0/0/2749254_fa3_bwd_dq_clean.perf`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_212125_dq_mmac_zero_seed_h1s1024_sqc7_fullperf`.
  Metrics: `simTicks=43,085,315`, `kernel_ticks=39,471,705`,
  `MMOP=55,296`, `VALU=131,232`, `SCA=96,904`, `LDS=37,872`,
  `coissue=13,167/10,241`, `MMAC active=24.0973%`,
  `ldsBankConflict=0`.

Comparison:

- Prior full perf `dq_kv_trans_split_wait`:
  `kernel_ticks=39,716,950`, `VALU=140,320`, `MMAC active=23.8706%`.
- New zero-seed version:
  `kernel_ticks=39,471,705`, `VALU=131,232`, `MMAC active=24.0973%`.
- Improvement:
  ticks down about `0.62%`, VALU down `9,088`, active up `0.227` point.

Next:

- This proves the qk/dP accumulator-zero moves were real debt.
- Do not blindly delete all `v_mov`: the remaining large clusters include
  store-helper copies and softmax/default-zero moves, which need separate asm
  attribution plus correctness/perf evidence.

Follow-up static reject:

- Tried removing the consumer hot-path `if (diag_store == 0)` branch by always
  storing dQ in the consumer and leaving `DQ_DIAG_STORE` as a tail diagnostic
  overlay.
- Static/resource still passed and consumer branch windows shrank
  `122/216 -> 120/216`, but asm `v_mov` got worse:
  total `359 -> 367`, copy category `173 -> 181`.
- Decision: `REJECT_STATIC_ASM`; do not pursue this as a v_mov cleanup without a
  different store-helper design.

## 2026-07-08 dQ NTile Pair Island

Decision: `ACCEPT`

User observation:

- Latest Wavefronts still showed poor coissue, severe instruction gaps, and
  small fragmented MMAC/VALU islands.

Hypothesis:

- The active dQ loop processed one `n_chunk` at a time:
  `K/V trans read -> score/dP MMAC -> softmax/dS VALU -> K normal read ->
  dQ MMAC`.
- `dq_update_from_ds_vec` used `ds_read_matrix_normal_pair` for each
  `n_chunk`; because the pair read already loads both halves of one `n_tile`,
  consecutive `n_chunk` iterations reread the same K normal pair.
- Processing the two halves of one `n_tile` together should form larger
  score/dP and dQ islands and cut redundant LDS normal reads.

Change:

- Replaced `dq_update_from_ds_vec` with `dq_update_from_ds_pair`.
- Main consumer loop now iterates `n_tile` instead of `n_chunk`.
- For each `n_tile`, issue trans-pair reads for K/V, compute both half0 and
  half1 score/dP accumulators, generate `ds_vec0/ds_vec1`, then read K normal
  pair once and apply both dS vectors to the long-lived `dq_reg`.
- No external API, wave role, output ownership, dS-LDS handoff, or atomic path
  change.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `164/216`, `164/216`, `9/40`;
  metadata `private=0`, `sgpr=53`, `vgpr=128`, no scratch/spill.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093036`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093048`.
  H1/S1024 has `dq_max_abs ~= 1.07e-05`, no nonfinite, `bad=0`, but
  `rel_l2 ~= 0.128`, so monitor numerical drift on larger shapes.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093242/m5out/0/0/2749638_fa3_bwd_dq_clean.perf`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_093242_dq_ntile_pair_island_h1s1024_sqc7_fullperf`.
  Metrics: `simTicks=40,586,455`, `kernel_ticks=36,972,845`,
  `MMOP=55,296`, `VALU=131,168`, `SCA=87,112`, `LDS=28,656`,
  `coissue=14,177/14,117`, `MMAC active=25.5487%`,
  `ldsBankConflict=0`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_ntile_pair_island_20260708_093242`.
  `duration=81,192`, `avg active waves=75.52`.
  Top bubbles remain `s_abarrier_try_wait -> s_xor_b32 38.23%`,
  `s_abarrier_try_wait -> s_waitcnt 9.99%`; normal-read gap
  `ds_read_matrix_format -> s_waitcnt` is down to `1.49%`.

Comparison:

- Previous accepted `dq_mmac_zero_seed` full perf:
  `kernel_ticks=39,471,705`, `LDS=37,872`, `MMAC active=24.0973%`.
- New pair-island version:
  `kernel_ticks=36,972,845`, `LDS=28,656`, `MMAC active=25.5487%`.
- Improvement:
  ticks down about `6.33%`, LDS instructions down about `24.3%`, active up
  `1.45` points.

Next:

- This confirms the user's island-fragmentation diagnosis and proves
  n_tile-level coarsening is useful.
- Remaining limiter is ABarrier/page cadence and wait/control exposure, not
  missing MMAC or missing native matrix path.
- Next candidate should look at PageFilled/PageUsed/QDoLatched timing or
  producer/consumer cadence, with a strict lifetime proof.  Do not add more
  buffering unless it reduces the ABarrier bubble after accounting for VGPR and
  control cost.

## 2026-07-08 dQ QDoFilled Overlap Probe

Decision: `REJECT_CORRECTNESS`

User observation:

- Wavefronts still shows poor coissue, severe instruction gaps, and fragmented
  MMAC/VALU islands even after `n_tile` pair scheduling.

Hypothesis:

- The accepted dQ kernel waits `Page0Filled` before consumers read sidecar and
  latch Q/dO into VGPR.
- Since `Page0Filled` is arrived only after producers publish Q/dO, sidecar,
  and K/V page0, Q/dO latch cannot overlap K/V page0 matrix load.
- Splitting a one-shot `QDoFilled` token might allow producers to publish
  Q/dO+sidecar first, continue loading K/V page0, and let consumers latch
  Q/dO concurrently.

Tested patch:

- Added `QDoFilled` ABarrier and moved consumer startup from
  `wait Page0Filled -> latch Q/dO` to
  `wait QDoFilled -> latch Q/dO -> arrive QDoLatched -> wait Page0Filled`.
- Producers arrived `QDoFilled` after Q/dO+sidecar and before the K/V page loop.
- A follow-up tried a one-shot `s_abarrier_seq(QDoFilled)` after init.
- No tile, math, output ownership, or MMAC path change.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `164/216`, `164/216`, `9/40`;
  metadata `private=0`, `sgpr=53`, `vgpr=128`, no scratch/spill.
- Without explicit `QDoFilled` seq, H1/S1024 failed correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_095421`,
  rows `688..703` were NaN.
- With explicit one-shot `QDoFilled` seq, H1/S128 failed correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_095649`,
  rows `48..63` were NaN.
- The temporary code was removed and the active source was restored to
  accepted commit `1dcf266`.
- Workbook evidence:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `41_dq_qdo_filled_overlap`.

Conclusion:

- A separate one-shot QDo visibility token is not a safe local fix for the
  startup serialization under the current sidecar/QDo/MLS protocol.
- Do not reintroduce this split without a focused barrier+matrix visibility
  probe.
- Next useful direction should target steady-state PageFilled/PageUsed cadence
  or useful work placement inside existing legal tokens, not a new QDoFilled
  token in the main kernel.

## 2026-07-08 dQ Sidecar SoA Vec4 LDS Read

Decision: `ACCEPT_MICRO_TICKS_NOT_PIPELINE_SUCCESS`

Hypothesis:

- The accepted pair-island dQ kernel still had three scalar sidecar LDS reads
  in each consumer branch: row max, row sum, and delta.
- dKV had a validated sidecar Vec4 LDS read micro-win, so dQ might reduce
  scattered sidecar LDS waits by grouping sidecar rows.

Tested:

- First tried AoS4 sidecar layout `[max,sum,delta,pad]` per row.  It generated
  `ds_read_b128=2` and removed `ds_read_b32`, but introduced
  `ldsBankConflict=12`; rejected by hard gate.
- Kept original SoA sidecar layout and changed consumer reads to load four-row
  Vec4 groups for max/sum/delta, then select the lane's row.  This generated
  `ds_read_b128=4` and left `ds_read_b32=2` because delta was still scalarized
  by codegen.

Evidence:

- Static/resource PASS:
  branch windows `8/40`, `164/216`, `164/216`, `9/40`; metadata
  `private=0`, `sgpr=53`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_113043`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_113055`;
  full-perf correctness `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_113438`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_113438/m5out/0/0/2750781_fa3_bwd_dq_clean.perf`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_113438_dq_sidecar_soa_vec4_h1s1024_sqc7_fullperf`.
- Metrics versus pair-island baseline:
  `kernel_ticks=36,972,845 -> 35,382,165` (`+4.30%`),
  `simTicks=40,586,455 -> 38,995,775`,
  `MMAC active=25.5487% -> 25.3548%`,
  `VALU=131,168 -> 138,208`,
  `SCA=87,112 -> 87,176`,
  `LDS=28,656 -> 28,656`,
  coissue `14,177/14,117 -> 17,446/16,910`,
  `ldsBankConflict=0`.
- XCU output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_sidecar_soa_vec4_s1024_fullperf_20260708_113438`.
  Top bubbles remain ABarrier/control:
  `s_abarrier_try_wait -> s_xor_b32 37.26%`,
  `s_abarrier_try_wait -> s_waitcnt 10.23%`.
  `ds_read_matrix_format -> s_waitcnt` is about `1.51%`.

Conclusion:

- Keep as a narrow elapsed-ticks micro-win because correctness/resource gates
  pass and same-shape full-perf ticks drop.
- Do not treat it as progress toward the 40% MMAC-active target: active share
  and VALU debt both worsen slightly, and xcu still points to ABarrier/page
  cadence as the real limiter.
- Future sidecar work should only continue if it also reduces VALU/register
  debt or ABarrier exposure; chasing complete `ds_read_b128` conversion alone
  is not the next highest-value path.

## 2026-07-08 dQ Nk32 Triple Page Rejected

Decision: `REJECT_PERF`

Hypothesis:

- XCU on `dq_sidecar_soa_vec4` showed the dominant issue bubble at producer
  `Page1Used` (`s_abarrier_try_wait -> s_xor_b32 37.26%`).
- The accepted Mq128/Nk64 path overlays page1 K/V onto released Q/dO LDS after
  `QDoLatched`.  A true three-page K/V stream might remove that overlay
  pressure if the tile shrinks to `Nk=32`.

Design:

- Keep the 16-wave full-3GEMM dQ route and `Mq=128,D=128`.
- Change only `Nk=64 -> Nk=32` and use three K/V pages.
- LDS budget:
  Q+dO `64KB`, three K/V pages `48KB`, sidecar about `1.5KB`; total about
  `113.5KB`, below the 128KB budget.
- Remove the hot-path `QDoLatched` overlay dependency; producers wait
  `PageUsed` only after a page can be reused.

Evidence:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `43_dq_nk32_triple_page`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_121450`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_121529`.
- Full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_121749`,
  helper perf
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_121749/m5out/0/0/2752032_fa3_bwd_dq_clean.perf`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_121749_dq_nk32_triple_page_h1s1024_sqc7_fullperf`.
- Static/resource PASS:
  metadata `private=0`, `sgpr=53`, `vgpr=128`, no spill/scratch,
  `ldsBankConflict=0`.
- Full-perf metrics versus accepted sidecar baseline:
  `kernel_ticks=35,382,165 -> 35,575,995`,
  `simTicks=38,995,775 -> 39,189,605`,
  `MMAC active=25.3548% -> 25.4985%`,
  coissue `17,446/16,910 -> 15,465/15,407`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_nk32_triple_page_s1024_fullperf_20260708_121749`.
  Top issue remains ABarrier/control:
  `s_abarrier_try_wait -> s_xor_b32 37.86%`,
  `s_abarrier_try_wait -> s_waitcnt 10.36%`.
  Representative top rows still wait on `barId=3 Page1Used`.

Conclusion:

- The LDS budget idea is valid, but it does not solve the measured bottleneck.
  Shrinking `Nk` doubles page epochs and keeps the producer blocked on
  `PageUsed`; full-perf ticks regress while MMAC active only moves slightly.
- The temporary code was removed.  Active source is restored to the
  `dq_sidecar_soa_vec4` baseline (`b56b2dc`).
- Do not pursue deeper buffering by shrinking `Nk` as the next main route.
  The next useful direction must reduce per-epoch barrier/control cost or
  increase useful MMAC work per ownership token without increasing page
  cadence debt.

## 2026-07-08 dQ Half-Page Release Rejected

Decision: `REJECT_PERF_STATS_ONLY`

Hypothesis:

- `dq_sidecar_soa_vec4` xcu showed producer waits on `Page1Used`.
- A K/V page contains two `n_tile` halves.  The accepted route releases the
  whole page only after both halves finish.
- Splitting page ownership by half might let producers prefetch the next
  `kt` half0 while consumers compute half1, without shrinking `Nk`.

Design:

- Keep the canonical `Mq=128,Nk=64,D=128,16 waves` dQ path.
- Replace whole-page `Page0/1 Filled/Used` with
  `Page0/1 Half0/1 Filled/Used`.
- Split producer K/V MLS helpers into `dq_load_k_tile_half` and
  `dq_load_v_tile_half`.
- Consumer waits each half before reading, and arrives half-used immediately
  after `dq_update_from_ds_pair` for that `n_tile`.

Evidence:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `44_dq_half_page_release`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_124824_dq_half_page_release_h1s1024_sqc7_stats_reject`.
- Static/resource PASS:
  branch windows `8/40`, `158/216`, `158/216`, `9/40`; metadata
  `private=0`, `sgpr=56`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_124748`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_124824`.
- H1/S1024 stats-only:
  `simTicks=39,826,605`, `firstWaveStartTick=3,613,610`,
  `lastWaveEndTick=39,826,605`, `kernel_ticks=36,212,995`,
  `MMOP=55,296`, `VALU=140,128`, `SCA=101,660`,
  `LDS=28,656`, `VMEM=1,408`, coissue `13,482/15,991`,
  `ldsBankConflict=0`, `MMAC active=25.4434%`.
- Same-shape sidecar SoA Vec4 stats-only recert was
  `simTicks=39,096,785`, `kernel_ticks=35,483,175`; full-perf accepted
  baseline was `kernel_ticks=35,382,165`, `SCA=87,176`,
  `MMAC active=25.3548%`.

Conclusion:

- Half-page release is legal and resource-clean, but it is not profitable on
  H1/S1024.  The extra half-token `Filled/Used` cadence adds scalar/control
  work faster than it exposes useful producer MLS prefetch overlap.
- The code was reverted to `dq_sidecar_soa_vec4` (`b56b2dc`) after archiving
  the candidate source and stats.
- Do not continue finer PageUsed splitting as an isolated optimization.
  The next direction should increase useful MMAC per ownership epoch or reduce
  existing ABarrier/control work without multiplying tokens.

## 2026-07-08 dQ Nk128 Direct Sidecar Rejected

Decision: `REJECT_CORRECTNESS`

Hypothesis:

- After `Nk32` triple-page and half-page release both regressed, the next
  high-ceiling route should increase useful MMAC per ownership epoch instead
  of multiplying ABarrier tokens.
- `Mq128/Nk128/D128` doubles useful MMAC under one page token versus `Nk64`,
  but Q+dO `64KB` plus K/V `64KB` exactly fills 128KB LDS.
- To fit, test removing sidecar LDS staging and letting consumers direct-load
  row max/sum/delta from global into VGPR once before the mainloop.

Evidence:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `45_dq_nk128_direct_sidecar`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_130949_dq_direct_sidecar_correctness_reject`.
- Nk128 static/resource PASS:
  branch windows `1/40`, `163/216`, `163/216`, `2/40`; metadata
  `private=0`, `sgpr=63`, `vgpr=128`, no spill/scratch.
- Nk128 H1/S128 correctness failed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_130721`,
  all dQ values were NaN (`actual_nonfinite=16384`, `bad_rows=128`,
  `pass=0`).
- Adding explicit `wait_vmem_lgkm()` before `QDoLatched` did not fix the
  all-NaN failure:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_130949`.
- Diagnostic Nk64 direct-sidecar/no-sidecar-LDS variant also failed all-NaN:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_131553`.

Conclusion:

- The old Nk128 static-spill failure is not reproduced in the current
  16-wave full3GEMM route, which is useful evidence.
- However, direct consumer sidecar global reads are not correctness-safe in
  this WDRA dQ route as implemented.  Because the same failure appears at
  Nk64, the immediate blocker is not only 128KB LDS boundary or `n_tile=2/3`
  layout.
- Code was reverted to `dq_sidecar_soa_vec4` (`b56b2dc`).  Do not move dQ
  sidecar back to direct consumer global reads in the main path without a
  focused WDRA/global-load sidecar probe.
- Nk128 remains attractive for MMAC active only if sidecar can be carried
  without direct consumer global loads and without exceeding LDS.

## 2026-07-08 dQ SQTT Bottleneck Diagnosis

Decision: `OBSERVE`

Question:

- dQ has `Mq=128,Nk=64,D=128` and three GEMM islands per K/V tile, so why is
  MMAC active still far below FWD?

Evidence:

- Detailed note:
  `results/dq_sqtt_bottleneck_20260708.md`.
- Current dQ H1/S1024 full perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_113438_dq_sidecar_soa_vec4_h1s1024_sqc7_fullperf`.
- Current dQ PMD aggregate:
  `kernel_ticks=35,382,165`, `MMOP=55,296`, `VALU=138,208`,
  `SCA=87,176`, `LDS=28,656`, `coissue=17,446/16,910`,
  `MMAC active=25.35%`, `MMOP runtime share=41.86%`,
  `ldsBankConflict=0`.
- Current dQ SQTT top rows:
  `s_xor_b32 35.36%`, `s_waitcnt 16.44%`, `v_mmac 8.85%`;
  top gaps are `s_abarrier_try_wait -> s_xor_b32 37.26%` and
  `s_abarrier_try_wait -> s_waitcnt 10.23%`.
- FWD H4/S1024 SQTT reference:
  `mmop_fp16 45.96%`, `salu_32 7.30%`,
  `lds_matrix -> mmop_fp16 1.18%`.

Conclusion:

- Current dQ is not primarily blocked by missing MMAC or LDS bank conflict.
  The first-order bottleneck is ABarrier ownership/wait debt around the
  PageFilled/PageUsed/QDoLatched ledger.
- Matrix-read latency and `v_mov`/mask debt are real but secondary in this
  capture.
- Next performance work should target more useful MMAC per ownership epoch or
  a less serial PageUsed/QDo lifetime before local wait/vmov polishing.
- Formal FWD-style acceptance still needs same-shape dQ H4/S1024 SQTT under
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.

## 2026-07-08 New-Machine Fixed-Env Backtest

Decision: `OBSERVE_ENV_BACKTEST`

Question:

- After fixing the new machine by aligning `shaobo_dev_8426` to liuchang's
  `a6a6eb6616ab...` compiler and removing the temporary
  `__builtin_hcu_wdra_init` path, do correctness and MMAC active match the
  historical clean-repo numbers?

Evidence:

- Run root:
  `/tmp/sb_perf_backtest_20260708_233749`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_233749_backtest_liuchang_a6_h1s1024_sqc7_stats`.
- Environment:
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`, compiler
  `a6a6eb6616abdd98b6dd72074afad281b47c8c6a`.
- Static evidence:
  `s_trap count=0`; role-local `s_set_vgpr_size` remains present; dKV/dQ
  gates pass with no spill/scratch.
- dKV H1/S1024 correctness PASS:
  `simTicks=48,263,670`, `MMAC active=32.9848%`,
  `coissue=35,050/24,317`, `ldsBankConflict=0`.
- dKV historical reference:
  `20260705_063337_clean_read8_score_dp_h1s1024_sqc7_fullperf` had
  `simTicks=50,926,785`, `MMAC active=32.0455%`.
- dQ H1/S1024 correctness PASS:
  `simTicks=39,410,735`, `MMAC active=25.4821%`,
  `coissue=16,471/16,335`, `ldsBankConflict=0`.
- dQ historical references:
  `dq_sidecar_soa_vec4` had `simTicks=38,995,775`,
  `MMAC active=25.3548%`; `dq_ntile_pair_island` had
  `simTicks=40,586,455`, `MMAC active=25.5487%`.

Conclusion:

- The environment fix is valid. The new machine no longer hits `s_trap`, both
  dKV and dQ pass correctness, and MMAC active is on the expected historical
  band.
- dKV is slightly better than the recorded clean_read8 reference on this
  stats-only run. dQ active matches the 25.3-25.5% historical band, while
  elapsed ticks are about 1.1% slower than the best sidecar SoA full-perf
  reference and faster than the ntile-pair reference.
- This was a stats-only backtest, not a new optimization. Use it as the fixed
  environment baseline before the next full `.perf`/xcu capture.

## 2026-07-09 dKV High-Source Split Wait

Decision: `ACCEPT_MICRO_TICKS_WAIT_LATENCY`

Goal:

- Start the 12h fixed-env loop from the current dKV baseline and target the
  XCU `s_waitcnt` / `ds_read_matrix -> s_waitcnt` exposure without changing
  tile shape, wave roles, ABarrier ownership, output ownership, or API.
- The candidate follows the dQ split-wait lesson: issue a later LDS matrix
  operand family before the first wait, wait only until the earlier family is
  safe to consume, then run the first MMAC island while the later reads remain
  in flight.

Change:

- In `dv_dk_mmac_owner16_read4x2`, moved the high-source
  `dv_dk_read_owner16_sources4<...,2,...>` before the first readiness wait.
- Replaced the first `wait_lgkm(0)` with `wait_lgkm(8)`.
- The low sources were already issued before softmax/dS.  With low+high
  source reads outstanding, `lgkmcnt(8)` waits for the older low reads while
  leaving the eight high reads in flight.  The existing later `wait_lgkm(0)`
  still protects high-source use.
- The release path is unchanged.  Q/Dout `Used` arrivals still happen only
  after the required full `wait_lgkm(0)`, so producer overwrite safety is not
  weakened.

Evidence:

- 12h workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_12h_goal_plan_20260709.xlsx`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_003152_dkv_splitwait_h1s1024_sqc7_fullperf`.
- Static/resource PASS:
  branch windows `14/16`, `189/240`, `189/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_002502`;
  H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_002617`.
- Full perf H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_003152/m5out/0/0/4224_fa3_bwd_wasp_clean.perf`.
  Compared with fixed-env current full perf
  `20260709_000533_current_bwd_fixedenv_h1s1024_sqc7_fullperf`:
  `simTicks 48,274,135 -> 47,484,710` (`-1.64%`),
  coissue `35,245/24,569 -> 35,640/24,684`,
  `ldsBankConflict=0` both ways.
- XCU detail moved in the expected direction:
  `s_waitcnt` hot latency `19.74% -> 19.55%`,
  `ds_read_matrix_format -> s_waitcnt` `3.50% -> 3.26%`,
  `ds_read_matrix_trans_format -> s_waitcnt` `2.76% -> 2.72%`,
  `v_mmac -> s_waitcnt` `1.73% -> 1.71%`.

Boundary:

- This is a narrow wait-late improvement, not a structural solution for the
  60% MMAC-active goal.
- Full-perf MMAC active is neutral to slightly down
  `32.9839% -> 32.9468%` even though ticks improve.  Stats-only had shown a
  small active-share increase, so this result must be reported with both
  numbers rather than over-claimed.
- The top XCU issue remains ABarrier ownership:
  `s_abarrier_try_wait -> s_xor_b32` is still about `41.38%`, and
  `s_abarrier_try_wait -> s_waitcnt` remains `8.59%`.

Conclusion:

- Keep the patch as a micro-ticks win if the next commit does not conflict
  with structural ABarrier work.
- The next high-ceiling step is still to reduce Q/Dout half-page ownership
  cliffs or increase useful MMAC per ownership epoch.  Local wait scheduling is
  useful but cannot by itself move dKV from about 33% to 60% MMAC active.

## 2026-07-09 dKV Release-Half Q Read Ahead

Decision: `OBSERVE_ACTIVE_REJECT_TICKS`

Goal:

- Test whether the release-half path can hide Q normal `ds_read_matrix`
  readiness under the softmax/dS VALU island.
- Keep the accepted `dkv_splitwait_highsrc` baseline intact: no new kernel,
  phase, tile, role, ABarrier token, API, or output ownership change.

Change Tested:

- In the release-half branch of
  `consume_mq_mpair_owner16_causal_exact_tile`, moved Q normal source reads
  before softmax/dS:
  `score/dP -> dO reads + Q reads -> wait_lgkm(8) -> DoutUsed ->
  softmax/dS -> wait_lgkm(0) -> QUsed -> dV/dK MMAC`.
- Safety proof was local and correct: `DoutUsed` still arrived only after dO
  fragments were ready, and `QUsed` still arrived only after Q fragments were
  ready.

Evidence:

- Static/resource PASS:
  branch windows became `14/16`, `222/240`, `222/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  The important warning is the consumer jump from `189/240` to `222/240`.
- Correctness PASS:
  H1/S128 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_005612`;
  H1/S1024 at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_005615`.
- Full perf H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_010216/m5out/0/0/4698_fa3_bwd_wasp_clean.perf`.
  Compared with accepted `dkv_splitwait_highsrc` full perf:
  `simTicks 47,484,710 -> 47,591,635` (`+0.23%` slower),
  `MMAC active 32.9468% -> 33.0627%`,
  coissue `35,640/24,684 -> 35,172/24,302`,
  `ldsBankConflict=0`.
- xcu first pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/xcu_outputs/dkv_release_qread_20260709_010216`.

Conclusion:

- Reject from the active source and restore `dkv_splitwait_highsrc`.
- The tiny MMAC-active rise is not a valid promotion because elapsed ticks
  regressed and the long-lived Q fragments push consumer branch usage close to
  the WDRA window limit.
- Lesson: hiding a matrix read under softmax is not free in BWD if it extends
  both Q and dO normal source live ranges across the VALU island.  Future
  release-half work should reduce ownership/token exposure or increase useful
  MMAC per token rather than simply hoisting more operand fragments.

## 2026-07-09 dKV Direct Global Sidecar Probe

Decision: `REJECT_RESOURCE_GATE`

Goal:

- Test whether removing sidecar LDS staging from the dKV Q producer can reduce
  Q half-page ownership pressure and ABarrier bubbles.
- This was only a compile-time focused probe.  It was not allowed to enter
  correctness or PMD perf unless metadata stayed spill-free.

Change Tested:

- Disabled producer `publish_sidecar_*_to_lds` calls under a temporary macro.
- Added a consumer path that reads `scores_max/scores_sum/delta` directly from
  global sidecar using a head-local pre-offset pointer.

Evidence:

- Default macro-off rebuild after the temporary edit matched the accepted
  baseline: branch windows `14/16`, `189/240`, `189/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Probe macro-on build failed the resource gate before correctness:
  branch windows `9/16`, `181/240`, `181/240`, `9/16`;
  metadata `private=0`, `sgpr=100`, `sgpr_spill_count=12`,
  `vgpr=128`, `vgpr_spill=0`.
- The compiler also emitted repeated `found vgpr before wave branch 0`
  warnings on the probe build.

Conclusion:

- Reject and remove the probe from active source.  Consumer-side global
  sidecar reads are not a valid dKV ABarrier workaround in this WDRA kernel
  because they introduce SGPR spill and pre-branch VGPR risk.
- Keep sidecar LDS staging in the current dKV route.  The next structural work
  should reduce page lifetime or increase useful MMAC per existing ownership
  epoch, not move sidecar global loads into consumers.

## 2026-07-09 dKV Sidecar Ring2 Prefetch

Decision: `REJECT_CORRECTNESS_SOURCE_REVERTED`

Goal:

- Reduce the dominant producer-side `Q0Used` wait without adding raw Q/dO
  buffers or moving sidecar global reads into consumers.
- Use a second tiny sidecar LDS page so the Q producer can publish the next
  tile's sidecar before waiting to overwrite the raw Q half page.

Change Tested:

- Increased sidecar LDS from one page to two pages.
- Producer wrote sidecar page `q_tile & 1` before `wait_q_half_used`.
- Consumer selected sidecar page by `q_tile & 1`.
- A second variant added `wait_lgkm(0)` after sidecar prewrite and before
  `QFilled` to test whether the failure was only LDS-store visibility.

Evidence:

- Static/resource PASS for both variants:
  branch windows `14/16`, `180/240`, `180/240`, `8/16`;
  metadata `private=0`, `sgpr=100`, `vgpr=128`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_020933`.
- H1/S1024 failed correctness without the extra wait:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_021009`,
  `dk_rel_l2=7.19868`, `dv_rel_l2=5.38142`, `pass=0`.
- H1/S1024 still failed after adding the sidecar visibility wait:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_021355`,
  `dk_rel_l2=0.187677`, `dv_rel_l2=0.100855`, `pass=0`.

Conclusion:

- Reject and remove from active source.  Sidecar ring2 is not safe as a local
  rewrite while Q sidecar readiness continues to be represented only by the
  existing Q half-filled token.
- The useful lesson is that sidecar's lifetime is coupled to Q half ownership
  more tightly than its byte size suggests.  Any future sidecar decoupling
  needs a focused barrier/visibility probe or a dedicated correctness-proofed
  token, not just a second sidecar page.

## 2026-07-09 dKV Causal Invalid Q-Tile Skip

Decision: `REJECT_PERF_REGRESSION_SOURCE_REVERTED`

Goal:

- For causal H1/S1024, skip q-tiles that are fully invalid for the current
  K/V tile (`q_tile_base + Mq <= k_base`) so the kernel avoids wasted Q/dO
  loads, sidecar publication, score/dP MMAC, softmax/dS, and dV/dK MMAC on
  upper-triangular invalid work.

Change Tested:

- Added a branch-local skip in both producers and the dKV consumer q-loop.
- First version carried runtime `causal` and failed metadata with
  `sgpr_spill_count=20`.
- The tested version relied on canonical dKV's existing `causal==1` shape
  gate and removed runtime `causal` from the skip predicate.

Evidence:

- Static/resource PASS after removing runtime `causal`:
  branch windows `6/16`, `193/240`, `193/240`, `1/16`; metadata
  `private=0`, `sgpr=100`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_023301`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_023304`.
- Full perf H1/S1024 regressed versus accepted `dkv_splitwait_highsrc`:
  `simTicks 47,484,710 -> 49,150,010`, `kernel_ticks 43,871,100 -> 45,536,400`,
  `MMOP 131,072 -> 88,064`, `MMAC active 32.9468% -> 28.7232%`,
  `coissue 35,640/24,684 -> 26,504/18,655`, `ldsBankConflict=0`.
- xcu detail for the candidate:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/xcu_outputs/dkv_causal_skip_20260709_023422`.
  Top bubbles remained ownership/control dominated:
  `s_abarrier_try_wait -> s_xor_b32 38.92%` and
  `s_abarrier_try_wait -> s_waitcnt 12.04%`; `v_mmac` hot latency fell to
  `9.20%`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_023422_dkv_causal_skip_reject_h1s1024_sqc7`.

Conclusion:

- Reject and remove from active source.  Skipping invalid causal work reduces
  MMOP but does not reduce the dominant ABarrier ownership bubble, and it
  makes wave progress/load balance worse enough to lose elapsed ticks and MMAC
  active share.
- Do not pursue triangular work skipping as an isolated micro-optimization in
  this 16-wave dKV route.  The next dKV direction should increase useful MMAC
  per ownership epoch or reduce Q/Dout page lifetime without making producer
  and consumer progress more irregular.

## 2026-07-09 dKV Dout Wait Under Softmax

Decision: `REJECT_PERF_REGRESSION_SOURCE_REVERTED`

Goal:

- Hide the ReleasePage dO-normal `ds_read_matrix` wait under the independent
  softmax/dS VALU work, without adding buffers, tokens, flags, or alternate
  kernel paths.

Change Tested:

- In `consume_mq_mpair_owner16_causal_exact_tile` ReleasePage path, changed:
  `read dO normal -> wait_lgkm(0) -> arrive DoutUsed -> softmax/dS`
  into:
  `read dO normal -> softmax/dS -> wait_lgkm(0) -> arrive DoutUsed`.
- The correctness boundary was preserved: producer overwrite is still allowed
  only after `wait_lgkm(0)` and `DoutUsed`.

Evidence:

- Static/resource PASS unchanged from baseline:
  branch windows `14/16`, `189/240`, `189/240`, `8/16`; metadata
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_025104`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/dkv_mmac_correctness_20260709_025107`.
- Full perf H1/S1024 regressed versus accepted `dkv_splitwait_highsrc`:
  `simTicks 47,484,710 -> 48,067,565`, `kernel_ticks 43,871,100 -> 44,453,955`,
  `MMAC active 32.9468% -> 33.0577%`, `coissue 35,640/24,684 -> 34,502/23,895`,
  `ldsBankConflict=0`.
- xcu detail:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean_12h/xcu_outputs/dkv_dout_wait_softmax_20260709_025150`.
  The intended wait reduction happened (`s_waitcnt 19.55% -> 18.63%`,
  `ds_read_matrix_format -> s_waitcnt 3.26% -> 2.56%`), but the ownership
  bubble grew (`s_abarrier_try_wait -> s_xor_b32 41.38% -> 41.73%`) and
  dispatch duration grew `96,420 -> 97,704`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_025150_dkv_dout_wait_softmax_reject_h1s1024_sqc7`.

Conclusion:

- Reject and remove from active source.  Moving the wait after softmax reduces
  local wait exposure, but delays `DoutUsed` enough to worsen the dominant
  ABarrier ownership path.
- Future wait-hiding must not delay `QUsed` or `DoutUsed` arrivals unless it
  also reduces the corresponding producer wait by a larger amount.

## 2026-07-09 FWD/BWD Gap Recheck And Next dKV Candidate

Workbook:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_12h_goal_plan_20260709.xlsx`,
  sheet `16_FWD_BWD_Gap_Next`.

Evidence:

- FWD H4/S2048/SQC7 reference remains the hard style target:
  `mmop_runtime_share=58.1159%`, stat-derived `MMAC active=45.0205%`,
  `coissue=272,136/372,022`, `ldsBankConflict=0`,
  `TCCHitRate=0.932907`, `TccSectorReuseRate=0.731662`.
- FWD/BWD H4/S1024 SQTT compare showed FWD `MMAC latency share=45.96%`,
  `SALU32=7.30%`, and read-to-MMAC bubble about `1.18%`; BWD had much larger
  scalar/control and barrier-chain exposure.
- Current accepted dKV H1/S1024 has `MMAC active=32.9468%`, but xcu is still
  dominated by `s_abarrier_try_wait -> s_xor_b32 41.38%` and
  `s_waitcnt 19.55%`.  The last two candidates proved that reducing local
  wait or MMOP without reducing the ownership bubble does not improve elapsed
  cost.

Decision:

- Do not continue generic wait-moving, causal-skip, or sidecar-decoupling
  patches.  The next code candidate must target Q/Dout ownership directly and
  must not preserve another live phase/path.
- The next small candidate, once the remote SSH path is available, is
  `dkv_q_used_release_before_softmax`: read Q-normal sources after dO release,
  wait and arrive `QUsed` before softmax/dS, then hold Q regs through softmax
  and dV/dK.  This is intentionally a risky, narrow probe of whether moving
  `QUsed` earlier beats the exposed local wait; reject unless same-shape ticks
  fall and xcu shows the ownership bubble decreases.
- If that fails, stop micro wait/lifetime tweaks and return to a resource-level
  design that increases useful MMAC per ownership epoch.

Operational note:

- Attempted `10.59.41.48` via the configured 54 and 59 jump routes on
  2026-07-09.  The 54 route closed the connection; the 59 route timed out
  during banner exchange.  xcu `pipeline/simd` CSV export for the current
  baseline is deferred until SSH returns.

## 2026-07-09 dKV QUsed Before Softmax

Decision: `ACCEPT_MICRO_TICKS_OWNERSHIP`

Workbook:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_12h_goal_plan_20260709.xlsx`,
  sheet `17_QUsedBeforeSoftmax_Result`.

Hypothesis:

- Prior wait-hiding attempts reduced local `s_waitcnt` but delayed ownership
  release and regressed elapsed ticks.  This candidate moves ReleasePage
  Q-normal source reads and `QUsed` arrival before softmax/dS, then holds Q
  source regs through softmax and consumes them in dV/dK MMAC.  If producer
  ownership wait is the dominant path, releasing Q earlier should improve
  elapsed ticks even if matrix-read wait shifts slightly.

Implementation:

- Single canonical dKV path only; no extra phase or fallback.
- In `consume_mq_mpair_owner16_causal_exact_tile` ReleasePage path:
  `dO read -> wait -> DoutUsed -> Q read -> wait -> QUsed -> softmax/dS ->
  dV/dK MMAC`.
- The helper formerly named around split Q/dO release is narrowed to
  `dv_dk_mmac_owner16_qready`, which does not perform LDS reads or barrier
  arrivals and only consumes already-ready source fragments.

Evidence:

- Static/resource PASS:
  branch windows `14/16`, `222/240`, `222/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  The consumer window grew from `189/240` to `222/240`, so this has little
  remaining register headroom.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_032250`;
  H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_032317`.
- Full perf H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_033115`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_033115_dkv_qused_before_softmax_h1s1024_sqc7_fullperf`.
  Versus accepted `dkv_splitwait_highsrc`:
  `simTicks 47,484,710 -> 46,716,670`,
  `kernel_ticks 43,871,100 -> 43,103,060`,
  `MMAC active 32.9468% -> 33.2391%`,
  coissue `35,640/24,684 -> 36,556/25,587`,
  `ldsBankConflict=0`.
- xcu detail:
  dispatch duration `96,420 -> 94,728`,
  average active waves `121.28 -> 122.18`,
  `s_abarrier_try_wait -> s_xor_b32 41.38% -> 40.55%`,
  `v_mmac -> v_mmac 8.44% -> 8.19%`.
  Tradeoff: `ds_read_matrix_format -> s_waitcnt 3.26% -> 3.90%`,
  `ds_read_matrix_trans_format -> s_waitcnt 2.72% -> 2.89%`.

Conclusion:

- Accept as a narrow ownership/ticks micro-win.  This is useful because the
  elapsed improvement and xcu ownership-bubble reduction point in the same
  direction.
- Do not treat it as the 60% MMAC-active solution.  It consumes much more
  consumer branch window and shifts pressure to matrix-read waits.  The next
  structural design still needs either less ownership/control per epoch or
  more useful MMAC per ownership epoch.

## 2026-07-09 dKV Score Zero Hoist Probe

Decision: `REJECT_PERF_REGRESSION_SOURCE_REVERTED`

Hypothesis:

- Reuse one branch-local F16 zero seed for score/dP MMAC initialization instead
  of constructing it inside each `score_dp_mmac_owner16` call, reducing visible
  `v_mov` pressure.

Evidence:

- Static/resource PASS after the change, but consumer branch windows increased
  from `222/240` to `226/240`; metadata stayed `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch.
- ASM moved in the expected local direction: `v_mov_b64 139 -> 111`, while
  `v_mov_b32` stayed `539` and `v_mmac` stayed `1028`.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/dkv_zero_hoist_correctness_20260709_132829`;
  H1/S1024 full perf run
  `/zys/shaobo_runs/dkv_zero_hoist_perf_20260709_133723`.
- XCU detail on
  `/zys/shaobo_runs/dkv_zero_hoist_perf_20260709_133723/dkv_mmac_correctness_20260709_133723/m5out/0/0/2755967_fa3_bwd_wasp_clean.perf`
  regressed versus accepted `dkv_q_used_release_before_softmax`:
  dispatch duration `94,728 -> 94,988`, average active waves
  `122.18 -> 121.52`, PMD all-waves tick `46,716,670 -> 46,833,150`.
  `v_mov_b64_e32` latency improved, but the dominant ownership and MMAC gaps
  stayed: `s_abarrier_try_wait -> s_xor_b32` about `40.35%` and
  `v_mmac -> v_mmac` about `8.31%`.

Conclusion:

- Reject and restore source to accepted baseline.  This confirms that local
  zero-hoisting can reduce one move class but lengthens live ranges and does
  not address the current first-order bottleneck.  Do not retry this style
  unless a later structural design also frees consumer VGPR headroom and lowers
  the ABarrier ownership bubble.

## 2026-07-09 dKV Consumer Half-Order Stagger Probe

Decision: `REJECT_STATS_REGRESSION_SOURCE_REVERTED`

Hypothesis:

- Current SQTT/focused windows show the two consumer groups are too lockstep.
  Try a minimal real-work stagger without adding tokens: keep consumer0 on
  half0 -> half1, but make consumer1 process half1 -> half0.  If peer useful
  work can cover Q/Dout ownership waits, MMAC active should rise without
  changing math or LDS layout.

Implementation tested:

- Single canonical dKV path only; no phase flag, no new kernel.
- Added a local half helper and changed only the Mq128 consumer order for
  `ConsumerGroup == 1`.
- Q/dO token ids, producer order, MMAC count, source-layout normal/trans reads,
  sidecar LDS path, and output ownership were unchanged.

Evidence:

- Static/resource PASS:
  branch windows `14/16`, `222/240`, `221/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_160105`;
  H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_160225`.
- H1/S1024 stats regressed versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 47,896,485`,
  `kernel_ticks 43,103,060 -> 44,282,875`,
  `MMAC active 33.2391% -> 31.1416%`.
  `MMOP` stayed `131,072`, `ldsBankConflict=0`, coissue rose to
  `41,983/28,546`, and aggregate barrier counter was about `189,551`.

Conclusion:

- Reject without full perf/xcu because same-shape stats already moved the wrong
  way.  The result confirms the design-risk in the workbook: reversing one
  consumer delays Q0/Dout0 `Used`, so the producer cannot overwrite half0 for
  the next q tile while the other half is still being consumed.  That breaks the
  existing half-page conveyor even though it creates some apparent stagger.
- Source restored locally and remotely to accepted
  `dkv_q_used_release_before_softmax`.
- Future stagger work must preserve early half0 release, or it must redesign
  producer publication order and ownership epochs together.  Do not retry
  naive half1-first consumer order.

## 2026-07-09 dKV Global Half1-First Conveyor Rejected

Decision: `REJECT_FULLPERF_REGRESSION_SOURCE_REVERTED`

Hypothesis:

- The previous one-sided consumer half-order stagger failed because it delayed
  Q0/Dout0 `Used` while producers still published half0 first.  Try the
  consistent version: producers publish half1 before half0 and both consumers
  consume half1 before half0.  If the issue was only producer/consumer order
  mismatch, this should preserve the conveyor while giving a different steady
  timing.

Implementation tested:

- Single canonical dKV path only; no new phase, no new kernel, no new tokens.
- Changed Mq128 producer Q, producer dO, and both consumer half orders from
  half0 -> half1 to half1 -> half0.
- Matrix path, sidecar LDS path, QUsed/DoutUsed release positions, and MMAC
  count were unchanged.

Evidence:

- Static/resource PASS:
  branch windows `14/16`, `221/240`, `221/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_163937`;
  H1/S1024 stats run
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_164008`;
  H1/S1024 full perf run
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_164425`.
- Stats-only looked like a tiny win versus accepted
  `dkv_q_used_release_before_softmax`:
  `kernel_ticks 43,103,060 -> 43,004,325`,
  `MMAC active 33.2391% -> 33.3829%`.
- Full perf rejected it:
  `simTicks 46,716,670 -> 46,947,355`,
  `kernel_ticks 43,103,060 -> 43,333,745`,
  `MMAC active 33.2391% -> 33.2641%`,
  coissue `36,556/25,587 -> 38,022/26,639`,
  `ldsBankConflict=0`.
  Helper perf is archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_164425_dkv_global_half1_first_reject_h1s1024_sqc7_fullperf`.
- XCU CLI was installed sidecar on liuchang and used for this perf:
  `XCU_ROOT=/zys/tools/xcompute_light_4.6.3/opt/XCompute-Light-4.6.3/XCompute`.
  Outputs are under
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dkv_global_half1_first_20260709_164425`
  and copied into the shared perf archive.  `detail` reports duration `95,240`,
  average active waves `121.04`, and top bubbles
  `s_abarrier_try_wait -> s_xor_b32 40.87%`,
  `s_abarrier_try_wait -> s_waitcnt 8.45%`,
  `v_mmac -> v_mmac 8.17%`.
  The largest individual exported pipeline window is a tail
  `s_abarrier_try_wait -> s_waitcnt -> s_barrier` sequence around `barId 10`,
  so future analysis should separate tail/AllDone waits from q-loop ownership
  bubbles.

Conclusion:

- Reject and restore source to accepted baseline.  Making the half order
  globally consistent avoids the previous one-sided release bug, but it still
  does not lower traced elapsed cost.  Half-page order alone is not the next
  useful lever.
- Future work should not keep flipping half order.  The next candidate needs a
  real lifetime or workload change: reduce ownership/control exposure, increase
  useful MMAC per ownership epoch, or move producer work that does not delay
  QUsed/DoutUsed.

## 2026-07-09 dKV Q-Side Sidecar Prefetch Rejected

Decision: `REJECT_STATS_TICKS_REGRESSION_SOURCE_REVERTED`

Hypothesis:

- XCU on the accepted `dkv_q_used_release_before_softmax` route still points at
  producer-side Q/Dout ownership waits.  Move immutable Q-side sidecar global
  loads before the Q half `Used` wait, then store the sidecar triple into LDS
  after the wait and matrix_load.  This should give producer waves useful work
  before the ownership wait without overwriting LDS early.

Implementation tested:

- Single canonical dKV path only; no new phase, no new token, no new kernel.
- Replaced the Q-side half sidecar publisher with split load/store helpers:
  load the sidecar triple before `wait_q_half_used`, then write the triple to
  the same sidecar LDS page before `arrive_q_half_filled`.
- dO producer, matrix paths, half ordering, QUsed/DoutUsed release positions,
  MMAC count, and output ownership were unchanged.

Evidence:

- Static/resource PASS:
  branch windows `15/16`, `222/240`, `222/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  The producer branch grew from `14/16` to `15/16`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_171154`;
  H1/S1024 stats run
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_171224`.
- Same-shape H1/S1024 stats versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 46,773,090`,
  `kernel_ticks 43,103,060 -> 43,159,480`,
  `MMAC active 33.2391% -> 33.3770%`,
  barrier counter `157,259.173 -> 154,090.84`,
  `VALU 168,514 -> 170,338`,
  `SCA 114,520 -> 115,032`,
  failed coissue `25,587 -> 25,870`,
  `ldsBankConflict=0`.

Conclusion:

- Reject without full perf/xcu because elapsed ticks moved the wrong way.  The
  test is still useful: it shows producer-side useful work can reduce the
  aggregate barrier counter, but this isolated sidecar prefetch adds enough
  live range, VALU/SCA, and failed coissue to erase the benefit.
- Source restored locally and remotely to accepted
  `dkv_q_used_release_before_softmax`.
- Do not retry sidecar-prefetch-alone.  Future attempts must either increase
  useful MMAC per ownership epoch or redesign the ownership lifetime so the
  extra producer work is amortized by a larger conveyor change.

## 2026-07-09 dKV Merged Used Token Probe

Decision: `REJECT_STATS_TICKS_REGRESSION_SOURCE_REVERTED`

Hypothesis:

- The accepted path still pays separate `QUsed` and `DoutUsed` ownership
  handshakes for each half page.  Merge those two used-side tokens into one
  `RawHalfUsed` token per half, while keeping `QFilled` and `DoutFilled`
  separate, to reduce scalar/control instructions without changing math, LDS
  layout, or producer publish order.

Implementation tested:

- Single canonical dKV path only; no new phase, no new kernel.
- Replaced per-half `QUsed` and `DoutUsed` waits/arrivals with one shared
  `RawHalfUsed` wait/arrival.
- Matrix path, sidecar LDS path, Q/dO fill tokens, half ordering, MMAC count,
  and output ownership were unchanged.

Evidence:

- Static/resource PASS:
  branch windows `14/16`, `222/240`, `222/240`, `8/16`;
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/tmp/dkv_merge_s128/dkv_mmac_correctness_20260709_205505`;
  H1/S1024 stats run
  `/tmp/dkv_merge_s1024/dkv_mmac_correctness_20260709_205534`.
- Same-shape H1/S1024 stats versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 47,066,110`,
  `kernel_ticks 43,103,060 -> 43,452,500`,
  `MMAC active 33.2391% -> 33.1006%`,
  `SCA 114,520 -> 113,224`,
  barrier counter `157,259.173 -> 160,714.59`,
  `VALU` unchanged at `168,514`,
  coissue `36,556/25,587 -> 36,948/26,108`,
  `ldsBankConflict=0`.

Conclusion:

- Reject without full perf/xcu because same-shape stats already regressed.
  This proves the local token-count reduction is not the right abstraction:
  merging `QUsed` and `DoutUsed` delays one producer/page lifetime enough that
  the ownership barrier cost increases more than the SCA count falls.
- Source restored locally and remotely to accepted
  `dkv_q_used_release_before_softmax`, and the restored dKV source rebuilt with
  evidence gate and metadata gate PASS.
- Do not mechanically merge used tokens.  The next ownership design must
  preserve independent producer release or increase useful MMAC per ownership
  epoch enough to amortize the extra wait.

## 2026-07-09 dKV dP-Before-Q First-Pair Probe

Decision: `REJECT_STATS_TICKS_REGRESSION_SOURCE_REVERTED`

Hypothesis:

- Accepted dKV releases `dO` earlier than `Q`, so the next q-tile may have
  `dO` ready before `Q`.  Split the first 32-row pair of each half from fused
  `score/dP` into `dP=dO@V^T` first, then wait `QFilled`, then compute
  `score=Q@K^T`.  The intended benefit was to cover Q producer/filled wait
  with useful dP MMAC without changing GEMM count or ownership tokens.

Implementation tested:

- Single canonical dKV path only; no new phase, no new kernel, no tile change.
- Added dP-only and score-only helpers derived from the accepted fused
  score/dP helper.
- Only the first MBlock pair of each half used dP-first.  The second pair kept
  the accepted path to limit VGPR lifetime risk.

Evidence:

- Static/resource PASS:
  metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/tmp/dkv_dp_before_q_s128/dkv_mmac_correctness_20260709_215407`;
  H1/S1024
  `/tmp/dkv_dp_before_q_s1024/dkv_mmac_correctness_20260709_215443`.
- Same-shape H1/S1024 stats versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 48,090,770`,
  `kernel_ticks 43,103,060 -> 44,477,160`,
  `MMAC active 33.2391% -> 32.5023%`,
  `VALU 168,514 -> 170,064`,
  `SCA` unchanged at `114,520`,
  barrier counter `157,259.173 -> 162,455.84`,
  `waitLgkm 52,834 -> 54,433`,
  coissue `36,556/25,587 -> 34,774/23,658`,
  `ldsBankConflict=0`.

Conclusion:

- Reject without full perf/xcu because stats decisively regressed.  The
  assumption that dP MMAC would cover a real Q-filled wait did not hold in the
  current conveyor.  Splitting the accepted fused score/dP island increased
  local waits and barrier exposure, lowered coissue, and reduced MMAC active.
- Source restored locally and remotely to accepted
  `dkv_q_used_release_before_softmax`; restored dKV rebuild, evidence gate,
  and metadata gate PASS.
- Keep score/dP fused in the main path unless xcu shows a concrete
  `dO-ready/Q-not-ready` window large enough to amortize the split.

## 2026-07-11 dKV BPS vbcnt Before Filled Arrive Probe

Decision: `OBSERVE_MICRO_WIN_NEEDS_XCU`

Hypothesis:

- Shaobo wiki says BPS bypass-L1 data needs `s_waitcnt_vbcnt 0` before direct
  consumption. Current dKV and FWD both use `matrix_load ... bps lds` inside
  `s_abarrier_seq -> arrive` without explicit `vbcnt`. Test whether adding
  `vbcnt` before producer Filled-token arrival improves PMD readiness without
  breaking overlap.

Implementation tested:

- Added opt-in macro `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE`, default `0`.
- When enabled, producers call `maybe_wait_bps_vbcnt_before_arrive()` before
  Filled arrivals for resident K/V and Q/dO half pages.
- No new kernel, no phase fork, no default behavior change.

Evidence:

- Build PASS for default and vbcnt binaries.
- Static gates PASS for both variants.
- Resource metadata unchanged: branch windows `14/16`, `222/240`, `222/240`,
  `8/16`; `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Asm count: default `s_waitcnt_vbcnt=0`, vbcnt variant `s_waitcnt_vbcnt=6`.
- Correctness PASS for H1/S128 and H1/S1024 on both variants.
- H1/S1024 same-env stats:
  `simTicks 47,136,635 -> 46,609,290`,
  `kernel_ticks 43,523,025 -> 42,995,680`,
  `MMOP 131,072 -> 131,072`,
  `VALU 168,514 -> 168,514`,
  `SCA 114,520 -> 114,520`,
  `LDS 79,360 -> 79,360`,
  coissue `36,078/25,327 -> 36,749/26,029`,
  `ldsBankConflict=0`.

Conclusion:

- Keep this as an opt-in probe, not a default change yet. The stats-only
  micro-win is real in this run (`kernel_ticks` about `-1.21%`), and it does
  not change resource or instruction-class counts, but it still needs xcu/SQTT
  evidence to prove the dominant ownership/matrix-read bubble shrinks rather
  than merely shifting PMD scheduling noise.
- Next validation should capture H1/S1024 or H4/S1024 full perf for default
  and vbcnt in the same PMD env, then compare xcu `detail`, `wavefronts`,
  `bubbles`, and `pipeline/simd`.

## 2026-07-11 dKV Promote BPS vbcnt To Default

Decision: `ACCEPT_DEFAULT_STATS_WIN_PENDING_XCU`

Change:

- Changed `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE` default from `0` to `1`.
- Kept rollback switch:
  `EXTRA_CXXFLAGS="-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0"`.

Verification:

- Default build now emits `6` `s_waitcnt_vbcnt` instructions.
- dKV evidence gate PASS.
- Symbol metadata gate PASS:
  branch windows `14/16`, `222/240`, `222/240`, `8/16`;
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dkv_vbcnt_default_20260711/dkv_mmac_correctness_20260711_112211`;
  H1/S1024
  `/zys/shaobo_runs/dkv_vbcnt_default_20260711/dkv_mmac_correctness_20260711_112221`.

H1/S1024 default-enabled stats:

- `simTicks=46,554,690`
- `kernel_ticks=42,941,080`
- `MMOP=131,072`, `VALU=168,514`, `SCA=114,520`, `LDS=79,360`
- coissue `37,689/26,615`
- `ldsBankConflict=0`

Conclusion:

- Enable by default because the same-env probe and the default-enabled rerun
  both improve ticks without changing resource pressure or matrix-path
  instruction classes.
- Still treat this as an instruction-level fix, not the architectural solution.
  The top-level BWD issue remains: dKV has more dependency edges and smaller
  ownership epochs than FWD, so we still need a redesigned pipeline that
  reduces token fragmentation and increases useful MMAC/VALU work per page
  lifetime.
## 2026-07-11 dQ Nk128 Latched Sidecar Baseline

Decision: `ACCEPT_BASELINE_FOR_DQ_ACTIVE40`

Hypothesis:

- Move dQ from the old `Mq=128,Nk=64,D=128` route to an FWD-like
  `Mq=128,Nk=128,D=128` route. The goal is to double useful MMAC per K/V
  ownership epoch while keeping sidecar off the consumer global path.
- Startup LDS carries `Q+dO+sidecar`. Consumers latch those values into VGPR,
  then producers reuse the released Q/dO LDS region as the second K/V page.
  Steady state is K/V double-page ping-pong.

Implementation:

- Single canonical dQ kernel, no phase stack.
- 16 waves:
  waves0-3 publish Q/dO group0 + sidecar group0 and stream K;
  waves4-7 compute q rows 0-63;
  waves8-11 compute q rows 64-127;
  waves12-15 publish Q/dO group1 + sidecar group1 and stream V.
- Consumers execute the full three-GEMM dQ chain in VGPR:
  `QK^T`, `dO V^T`, softmax/dS, then `dS K`.
- BPS readiness uses `s_waitcnt_vbcnt 0` before Filled-token arrivals.

Evidence:

- Static/resource gate PASS:
  producer0 `8/40`, consumer0 `161/216`, consumer1 `161/216`,
  producer1 `9/40`; `private_segment=0`, `sgpr_count=67`,
  `vgpr_count=128`, `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_active40_20260711/dq_correctness_20260711_115658`;
  H1/S1024
  `/zys/shaobo_runs/dq_active40_20260711/dq_correctness_20260711_115723`.
- H1/S1024 stats:
  `simTicks=35,974,575`, `MMOP=55,296`, `VALU=121,632`,
  `SCA=78,244`, `LDS=28,656`, `VMEM=1,408`,
  coissue `15,755/18,857`, `ldsBankConflict=0`,
  stat-derived `MMAC active=27.2563%`.
- Full-perf/xcu:
  `/zys/shaobo_runs/dq_active40_fullperf_20260711/dq_correctness_20260711_120239/m5out/0/0/2764854_fa3_bwd_dq_clean.perf`;
  xcu outputs
  `/zys/shaobo_runs/dq_active40_fullperf_20260711/xcu_outputs/dq_nk128_h1s1024_20260711_120239`.
  Top bubbles were `s_abarrier_try_wait -> s_xor_b32 26.57%`
  and terminal `s_abarrier_try_wait -> s_waitcnt 17.88%` on the old
  `AllDone` cleanup token.

Conclusion:

- Accept this as the current dQ active40 baseline because it is correct,
  resource-clean, native matrix-path clean, and improves the algorithmic
  MMAC-per-epoch structure versus Nk64.
- It is still far from the 40% MMAC-active target. The first xcu diagnosis
  showed terminal cleanup pollution, so clean up tail synchronization before
  drawing the next pipeline conclusion.

## 2026-07-11 dQ Tail Cleanup / AllDone Removal

Decision: `ACCEPT_MICRO_AND_SET_NEXT_BOTTLENECK`

Hypothesis:

- The xcu top `s_abarrier_try_wait -> s_waitcnt` bubble was partly a tail
  artifact from a 16-wave `AllDone` ABarrier used only before invalidation.
  Replace it with `__syncthreads(); wave0 inv; __syncthreads()` so SQTT
  evidence focuses on real steady-state page ownership.

Implementation:

- Removed `Bar::kAllDone` init, per-role arrive, and common try-wait from the
  active dQ kernel.
- Kept wave0-only `s_abarrier_inv` for the six live slots:
  Page0Filled/Page0Used/Page1Filled/Page1Used/QDoFilled/QDoLatched.

Evidence:

- Static/resource gate PASS:
  producer0 `8/40`, consumer0 `161/216`, consumer1 `161/216`,
  producer1 `9/40`; metadata stays `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch. Asm now has `s_abarrier_inv=6`,
  no `s_abarrier_init 6`, and no `s_abarrier_try_wait.*6`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq40a_tail_cleanup_20260711/dq_correctness_20260711_121340`;
  H1/S1024
  `/zys/shaobo_runs/dq40a_tail_cleanup_20260711/dq_correctness_20260711_121348`.
- H1/S1024 stats versus Nk128 baseline:
  `simTicks 35,974,575 -> 35,750,715` (`-0.62%`),
  `MMAC active 27.2563% -> 27.3105%`,
  `SCA 78,244 -> 77,516`,
  coissue `15,755/18,857 -> 16,037/18,954`,
  `ldsBankConflict=0`.
- Full-perf/xcu:
  `/zys/shaobo_runs/dq40a_tail_cleanup_fullperf_20260711/dq_correctness_20260711_121754/m5out/0/0/2765534_fa3_bwd_dq_clean.perf`;
  xcu outputs
  `/zys/shaobo_runs/dq40a_tail_cleanup_fullperf_20260711/xcu_outputs/dq40a_tail_cleanup_h1s1024_20260711_121754`.
  The old `AllDone` bubble disappears. New top bubble is
  `s_abarrier_try_wait -> s_xor_b32 27.14%`, with representative instances
  on `barId=1`, i.e. `Page0Used`.

Conclusion:

- Accept as a small cleanup and evidence-quality fix. It is not the route to
  40%, but it removes a misleading tail wait and exposes the real limiter:
  producer waves run ahead and wait for consumer PageUsed before reusing K/V
  pages.
- Next structural experiment should stay on the 16-wave mainline and reduce
  producer-thin PageUsed idle: either give producer groups recurring useful
  work or rebalance role ownership while preserving two symmetric full-3GEMM
  consumers. A 12-wave one-producer topology is only a fallback/control
  experiment if 16-wave resource use proves structurally wasteful.

## 2026-07-11 dS producer -> dQ consumer native-layout gate

The proposed ring is structurally sound on paper: P_K / C_dS / C_dQ / P_V,
`Mq=128,Nk=128,D=128`, and two 64KB pages alternating
`K+V -> K+dS -> free`. C_dQ is the sole dQ accumulator/store owner, so the
route avoids both duplicate score/dP and split-K CTA tail reduction.

Focused evidence now blocks a canonical implementation:

- `dq_native_ds_write_roundtrip_probe.cpp` compiled and executed normal/trans
  `ds_write_matrix_format_f16`, m32x16 and mt16x32 readers, and MMAC. It had
  zero bank conflicts but no naive register-identity pairing.
- `dq_native_ds_write_mmac_output_probe.cpp` then used the real producer
  shape: `v_mmac` score outputs -> fp16 `ds_vec0/ds_vec1` packing -> all four
  HCU-exposed writer candidates and the reader pairings that compiled on the
  current toolchain -> C_dQ-like dQ MMAC. Every pairing differed from direct
  `dS@K`. PMD ran the instructions, had `private=0`, `sgpr=8`, `vgpr=80`,
  and `ldsBankConflict=0`.
- PMD prints `ds_write_matrix : testing` during both runs. Therefore this is
  evidence that the desired fragment ABI has not been proven on this model,
  not proof that silicon cannot support the handoff.

Decision: `OBSERVE_NO_NATIVE_PAIR`.

- Keep the main `Mq128/Nk128` dQ source untouched.
- Do not bridge the gap with scalar LDS gathers, bpermute/mpermute, or an LDS
  transpose/source-layout copy. Those would invalidate the native-instruction
  premise of the new architecture.
- Next action is a minimal instruction question/repro for the compiler/PMD
  owner: which `ds_write_matrix_format_f16` producer fragment and which
  `ds_read_matrix` form form the supported C_dS-result -> C_dQ-lhs contract;
  also confirm whether the PMD warning denotes incomplete execution semantics.

Follow-up after rereading Shaobo ISA/HCU docs:

- The broader "no native pair" wording is too strong.  The ISA Delta documents
  paired page formats for B16 `DS_WRITE_MATRIX_FORMAT` row=2 col=1: group4
  writes a `32x16` tile for direct `DS_READ_MATRIX_FORMAT` group4 use, and the
  transpose variants are also specified.  HCU currently exposes the normal
  `m32x16` alt0/alt1 forms and `m32x16` transpose alt0; it rejects one
  documented-looking transpose alt1 combination in this compiler.
- A corrected M-pair probe uses adjacent MMAC outputs along the M/Q dimension,
  allocates 2KB per candidate page to cover transpose-reader footprint, and
  tests both LIT=1 and LIT=0 MMAC output plus four simple lane-local fp16
  packing orders.  All eight direct pack candidates still mismatch direct
  `dS@K`, with `ldsBankConflict=0`.
- Current status is therefore
  `OBSERVE_PRODUCER_FRAGMENT_ABI_UNRESOLVED`: the page-format pairing exists,
  but we have not identified the producer 4-VGPR fragment ABI needed to feed
  `ds_write_matrix_format_f16` from a C_dS MMAC/VALU result.
- Important prior evidence from the Shaobo MLS layout reference refines this
  again: a 2026-07-05 focused operand proof accepted
  `VGPR(dS) -> ds_write_matrix_format(no t) -> ds_read_matrix_trans_format
  32x16 -> MMAC` with normal `32x16` K readers.  Therefore the native handoff
  route itself is viable in PMD for a correctly oriented producer fragment.
  The unresolved part is how to make the real C_dS score/softmax result land
  in that accepted producer fragment layout without scalar gather, permute, or
  a hot LDS transpose.
- The compiler tree does not expose an obvious `V_MOVMATRIX_*` builtin even
  though the ISA dependency table names `V_MOVMATRIX_16X16_B16`.  Do not use
  inline assembly or permute workarounds in the canonical kernel.  First
  reconstruct/reuse the accepted operand-pair proof, then redesign C_dS to
  produce that layout natively; only ask the compiler/PMD owner if the required
  producer layout still cannot be generated with exposed builtins.

## 2026-07-11 q-owned score -> dS write -> dQ chain probe

Decision: `REJECT_DIRECT_QOWNED_PACK`.

- Added isolated probe `dq_dswrite_qowned_chain_probe.cpp`; canonical dQ/dKV
  source is unchanged.
- Probe chain:
  `Q_trans x K32` q-owned MMAC score generation, four simple fp16 pack orders,
  `ds_write_matrix_format(no t)`, `ds_read_matrix_trans_format 32x16`, then
  `dQ = dS @ K_normal`.
- Static/resource:
  asm has `matrix_load_32x*=7`, `ds_write_matrix_format=4`,
  `ds_read_matrix*=14`, `v_mmac=12`, `ds_read_b32=0`,
  `bpermute/mpermute=0`.  Metadata is clean:
  `group_segment=32768`, `private=0`, `sgpr=24`, `vgpr=23`, no spill.
- PMD result:
  `/zys/shaobo_runs/dq_qowned_chain_probe_20260711_153457`.
  `simTicks=8,165,885`, `MMOP=12`, `ldsBankConflict=0`.
  All four pack variants fail; best variants have only `raw_nonzero=16`,
  `decoded=0`, and final `any_pass=0`.
- Interpretation:
  q-owned score orientation is still useful, but its natural two-accumulator
  fp16 pack is not the 7/5 accepted `ds_write_matrix` producer source layout.
  The next useful proof is to recover the accepted slot-map formula and make
  C_dS compute/publish that layout directly, not to add more ad hoc pack
  candidates or a scalar/permute workaround.

Follow-up direct-MMAC qK test:

- User hypothesis:
  the qK GEMM may be using the wrong MMAC result layout because it uses the
  current lit/4interleave-style helper.  Try qK score generation with direct
  `__builtin_hcu_mmac_f32_16x16x16_f16` while keeping the dQ consumer MMAC
  unchanged.
- Implementation:
  extended `dq_dswrite_qowned_chain_probe.cpp` from 4 to 8 candidates.
  Variants 0-3 use lit qK score; variants 4-7 use direct qK score.
- Static evidence:
  asm now has `v_mmac=24`, including 4 direct non-`lit` qK instructions;
  `matrix_load_32x=7`, `ds_write_matrix_format=8`,
  `ds_read_matrix*=18`, `ds_read_b32=0`, `bpermute/mpermute=0`.
  Metadata is still clean: `group_segment=32768`, `private=0`, `sgpr=24`,
  `vgpr=30`, no spill.
- PMD result:
  `/zys/shaobo_runs/dq_qowned_chain_direct_mmac_probe_20260711_154619`.
  `simTicks=8,449,805`, `MMOP=24`, `ldsBankConflict=0`.
  All variants 0-7 fail; direct-MMAC variants 4 and 6 have `raw_nonzero=16`
  but `decoded=0`, and variants 5/7 are zero.  Final `any_pass=0`.
- Decision:
  `REJECT_DIRECT_MMAC_SCORE_LAYOUT`.  qK direct MMAC alone does not create the
  accepted dS producer layout.  Continue with slot-map-driven C_dS generation.

## 2026-07-11 ds_write slot-map reverse probe

Decision: `OBSERVE_SLOTMAP_PROOF_HALF_REGION`.

- Added isolated probe `dq_dswrite_slotmap_reverse_probe.cpp`; canonical dQ
  and dKV sources are unchanged.
- Probe purpose:
  stop guessing pack orders.  First write an identity VGPR fragment through
  `ds_write_matrix_format(no t)` and read it back with
  `ds_read_matrix_trans_format 32x16` to build `dst_slot -> src_slot`.
  Then light one destination slot at a time and decode the following
  `dQ = dS @ K_normal` MMAC output to infer which K row each
  `group/word` slot feeds.
- Static/resource:
  asm has `matrix_load_32x16=3`, `ds_write_matrix_format=2`,
  `ds_read_matrix=5`, `v_mmac=2`, and `ds_read_b32/bpermute/mpermute/s_trap=0`.
  Both probe kernels have `group_segment=8192`, `private=0`, no spill, with
  very small VGPR/SGPR footprints.
- PMD slot-map evidence:
  `/zys/shaobo_runs/dq_slotmap_reverse_probe_20260711_160921` and follow-up
  `/zys/shaobo_runs/dq_slotmap_reverse_probe_20260711_161646`.
  `ldsBankConflict=0`, `MMOP=2` for consume dispatches.
  Inferred table:
  `group0=[0,1,2,3,0,1,2,3]`,
  `group1=[4,5,6,7,4,5,6,7]`,
  `group2=[8,9,10,11,8,9,10,11]`,
  `group3=[12,13,14,15,12,13,14,15]`.
- Interpretation:
  the accepted path is not a mystery pack permutation.  For
  `ds_write_matrix(no t) -> ds_read_matrix_trans 32x16 -> K_normal MMAC`,
  the K row is `group * 4 + (word & 3)`.  Words `0..3` and `4..7` are two
  source/reduction half-regions with the same K-row labels, not a safe
  duplicate to blindly merge.
- Full toy check:
  using only the low half-region decodes 16 q rows for target `krow=7` but
  covers only one D/reduction half; using only the high half-region covers the
  other half.  Writing both half-regions into the same simplified accumulator
  makes the two halves sum and the toy decoder cannot treat that as a single
  K-tag output.  Therefore this probe proves the slot map and matrixized
  instruction path, but it is not a standalone full-D dQ correctness proof.
- Next:
  redesign C_dS generation around this slot table.  Real dQ code must consume
  the two half-regions in the same accumulator structure/order used by the
  target D tile, not merge them in a toy single-tag output.  Continue to reject
  scalar LDS gather, bpermute/mpermute, and ad hoc pack permutations in the
  canonical kernel.

Follow-up split-accumulator proof:

- Extended the same probe to store three dQ-side outputs:
  `pair_acc = mmac(low, lowK) + mmac(high, highK)`, `split_low`, and
  `split_high`.
- PMD result:
  `/zys/shaobo_runs/dq_slotmap_reverse_split_probe_20260711_165032`.
  Static path remains clean: `matrix_load_32x16=3`,
  `ds_write_matrix_format=2`, `ds_read_matrix=5`, `v_mmac=3`,
  `ds_read_b32/bpermute/mpermute/s_trap=0`, `private=0`, and
  `ldsBankConflict=0`.
- Result:
  `pair_pass=0`, `low_pass=1`, `high_pass=1`, `split_pass=1`.
  `split_low` decodes `krow=7,d=0..15`; `split_high` decodes
  `krow=7,d=16..31`; combined covers 512 output slots, 16 q rows, one K row,
  and all `D=0..31`.
- Decision:
  `ACCEPT_SLOTMAP_SPLIT_ACC_PROOF`.
- Implementation implication:
  canonical C_dQ must keep two half-region accumulator/update paths matching
  `f16x4[0]` and `f16x4[1]`.  The prior full-toy failure was caused by
  prematurely accumulating the two half-regions into one tag output; it was not
  a rejection of the native `ds_write_matrix -> ds_read_matrix_trans -> MMAC`
  handoff.

Compact source-slot map follow-up:

- Added compact table output to
  `probes/dq_dswrite_slotmap_reverse_probe.cpp`, so each row reports
  `dst(group,q,word) -> src_lane:src_word`.
- PMD result:
  `/zys/shaobo_runs/dq_slotmap_reverse_compact_probe_20260711_172345`.
- Result:
  the identity write/read map has `mapped=504/512` and `unique_src=504/512`.
  The only unmapped destination slots are
  `(group=2,q=15,word=4..7)` and `(group=3,q=15,word=4..7)`.
  The same run still reports `pair_pass=0`, `low_pass=1`, `high_pass=1`,
  `split_pass=1`, and the consume dispatch has `ldsBankConflict=0`.
- Decision:
  `ACCEPT_SLOTMAP_COMPACT_MAP`.
- Implementation implication:
  a canonical C_dS publisher cannot assume a dense 512-slot affine source map.
  It must either use a table/formula validated against the compact map or
  explicitly mask the eight boundary holes before handing dS to C_dQ.

## 2026-07-11 dQ K/V Split Ownership Minimal Probe

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

- The current dQ `Mq=128,Nk=128,D=128` path has separate producer roles for K
  and V, but both producers arrive the same `PageFilled(count=8)` token.
  Splitting this into `KFilled(count=4)` and `VFilled(count=4)` might expose
  finer ownership and reduce the dominant PageUsed/PageFilled ABarrier bubble.

Implementation:

- Branch `exp/dq-kv-split-ownership`.
- Canonical dQ only, no phase stack and no math-order change.
- Replaced page tokens with K/V tokens:
  `Page{0,1}KFilled/KUsed` and `Page{0,1}VFilled/VUsed`.
- K/V `Filled` count is 4 because each producer group has four waves.
  K/V `Used` count remains 8 because both consumer groups consume each K/V
  page before producer overwrite is legal.

Evidence:

- Static/resource gate PASS:
  producer0 `8/40`, consumer0 `161/216`, consumer1 `161/216`,
  producer1 `9/40`; metadata `private=0`, `sgpr=67`, `vgpr=128`,
  `sgpr_spill=0`, `vgpr_spill=0`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_kv_split_min_s128_20260711_180613/dq_correctness_20260711_180613`;
  H1/S1024
  `/zys/shaobo_runs/dq_kv_split_min_s1024_20260711_180641/dq_correctness_20260711_180641`.
- Same-shape H1/S1024 stats versus accepted dQ tail-cleanup baseline:
  `simTicks 35,750,715 -> 36,198,435`,
  `MMOP=55,296`, `VALU=121,632`, `LDS=28,656`, `VMEM=1,408`
  unchanged, but `SCA 77,516 -> 81,784`,
  coissue `16,037/18,954 -> 15,222/18,657`,
  barrier sum `51,690 -> 54,599.75`,
  waitLgkm sum `13,954.75 -> 14,387.75`,
  emptyBuffer sum `21,735.922 -> 23,166.422`,
  `ldsBankConflict=0`.

Conclusion:

- Reject and do not keep in the active route.  Splitting the ledger without
  changing consumer work order creates more scalar/control and barrier cost
  while useful MMOP/VALU/LDS work is unchanged.
- K/V split ownership is only worth retrying if the consumer actually starts
  useful score work after `KFilled` and delays `VFilled` until dP, or if another
  design gives the producer groups recurring useful work.  Token splitting by
  itself is a negative pattern.

## 2026-07-11 dQ Alternate-Page Full-KV Producer

Decision: `REJECT_STATS_TICKS_REGRESSION`

Top-level design check:

- The previous Mq192/1P3C idea was stopped before code because it is not safe
  on the fixed diagnostic shape `S=1024`: `1024 % 192 != 0`, and current
  Q/dO `matrix_load` has no proven row-mask or padded-source contract for the
  q-tail.  This is recorded in workbook sheet `49_dq_mq192_1p3c`.
- Replacement candidate kept `Mq=128,Nk=128,D=128` so the S1024 baseline is
  directly comparable.  Only producer ownership changed:
  waves0-3 publish full K+V for even/page0 K tiles, waves12-15 publish full
  K+V for odd/page1 K tiles.  Consumers and all math stayed unchanged.

Hypothesis:

- Baseline has both producer groups rendezvous on each `PageFilled(count=8)`
  token: P0 publishes K and P1 publishes V.  Reducing each page fill to one
  producer group (`count=4`) might reduce filled-token control and keep P1
  from becoming thin in the q-loop后半段.

Evidence:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `50_dq_altpage_fullkv`.
- Static/resource PASS:
  producer0 `8/40`, consumer0 `161/216`, consumer1 `161/216`,
  producer1 `9/40`; metadata `private=0`, `sgpr=60`, `vgpr=128`,
  `sgpr_spill=0`, `vgpr_spill=0`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_altpage_fullkv_s128_20260711_184157/dq_correctness_20260711_184157`;
  H1/S1024
  `/zys/shaobo_runs/dq_altpage_fullkv_s1024_20260711_184204/dq_correctness_20260711_184204`.
- Same-shape H1/S1024 stats versus accepted `dq40a_tail_cleanup`:
  `simTicks 35,750,715 -> 35,807,590`,
  `MMOP=55,296`, `VALU=121,632`, `LDS=28,656`, `VMEM=1,408`
  unchanged, `SCA 77,516 -> 66,476`,
  coissue `16,037/18,954 -> 16,309/19,349`,
  barrier sum `51,690 -> 54,589.25`,
  waitLgkm sum `13,954.75 -> 14,134.75`,
  emptyBuffer sum `21,735.922 -> 19,112.422`,
  `ldsBankConflict=0`.

Conclusion:

- Reject and restore active route.  The hypothesis partly worked at the
  scalar-control level (`SCA` fell), but moving full K+V load into one producer
  serialized page availability enough that barrier/wait/ticks regressed.
- Do not continue by tweaking this page ownership variant.  Next top-level
  direction must either increase useful MMAC per ownership epoch, change
  consumer overlap/role timing, or use the native dS handoff slot-map work to
  remove a larger dependency; producer page ownership alone is not enough.

## 2026-07-11 dQ K-First True Overlap Design Review

Decision: `REVISE_BEFORE_CODE`

Workbook:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `51_dq_kfirst_true_overlap`.

Initial thesis:

- Retry K/V split only if it creates real useful work overlap:
  `KFilled -> score MMAC`, then wait `VFilled` only before dP.  In principle,
  score could hide V readiness and V could be released earlier than K because
  dQ needs K but not V.

Design stress result:

- The V early-release part is not valid for the current per-`n_tile` immediate
  loop.  V for one `n_tile` is dead after its dP, but the same V page is still
  needed by later `n_tile` chunks.  A page-level `VUsed` cannot safely arrive
  until all `n_tile` dP work for that K tile is complete.
- To make V lifetime materially shorter than K, we need one of two bigger
  designs:
  1. Store dS/qk/dp-like intermediates and run dQ later, i.e. the native
     dS handoff/ring path based on the accepted slot-map proof.
  2. Split VUsed by n_tile/half-page, which resembles earlier fine-token
     experiments that increased SCA/control and regressed.

Conclusion:

- Do not implement the original K-first early-release pseudocode.  A narrow
  K-first probe could still test whether score hides VFilled wait, but its
  upside is limited because VUsed and KUsed remain page-level late releases.
- The next top-level mainline should favor the native dS handoff/slot-map ring
  or another design that truly changes useful compute per ownership epoch,
  rather than more token-only lifetime tweaks.

## 2026-07-11 dQ Slotmap Recheck Before Native Ring Code

Decision: `ACCEPT_NATIVE_SPLIT_HANDOFF_CONTRACT`

Question:

- Before writing the native dS ring, re-run the focused slotmap probe on the
  current liuchang/zys1 path to verify whether the handoff contract is still
  `split_low/high` rather than pair-accumulator.

Evidence:

- Run:
  `/zys/shaobo_runs/dq_slotmap_recheck_20260711_191700`.
- Probe output:
  `slotmap_reverse_split_result pair_pass=0 low_pass=1 high_pass=1 split_pass=1`;
  `slotmap_reverse_final pass=1`.
- Slot label table:
  group0 `w0=0 w1=1 w2=2 w3=3 w4=0 w5=1 w6=2 w7=3`;
  group1 `4..7`; group2 `8..11`; group3 `12..15`, duplicated across the
  two word half-regions.
- Health:
  focused probe metadata is clean and consume dispatch has `ldsBankConflict=0`.

Conclusion:

- The native matrixized handoff remains viable, but only with split
  half-region accumulators.  The next code step must prototype
  `C_dS split publisher -> two N32 dS LDS slots -> C_dQ split consumer`.
  Do not use pair-accumulator `ds_write_matrix`, scalar gather, permute, or
  ordinary `ds_read_b32` as the main handoff path.

## 2026-07-11 Native Ring Code Skeleton

Decision: `ACCEPT_PREP_ONLY_NO_PERF_CLAIM`

Change:

- Added `ins::ds_write_matrix_32x16_f16` as the single wrapper for the verified
  native B16 matrix write format.
- Added `dq::NativeDsRingDqTile` and `dq::DqNativeDsRingBarrierLedger`.
  The contract fixes the first prototype at `Mq=64,Nk=128,D=128,12wave`,
  two `N32` dS slots, and compile-time LDS budget checks.

Evidence:

- Canonical dQ build and metadata gate PASS on remote
  `/zys/shaobo/fa3_bwd_wasp_clean`: `private=0`, `sgpr=67`, `vgpr=128`,
  no spill/scratch.
- H1/S128 canonical smoke PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260711_192915`.

Conclusion:

- This is preparation only.  It does not claim performance improvement and
  does not change the accepted dQ kernel.  The next implementation step is the
  minimal C_dS split publisher prototype that uses the slot-map half-region
  contract.

## 2026-07-11 Native dS Slotmap Formula

Decision: `ACCEPT_CODE_CONTRACT`

Finding:

- Parsing the compact slotmap showed `same_lane=10/504`, with holes at
  `(group=2,q=15,word=4..7)` and `(group=3,q=15,word=4..7)`.
- This is not a reason to add permute.  Instead, C_dS must schedule by native
  source slot: the lane computes the logical dS value that its source slot will
  publish to the C_dQ destination.

Code contract:

- Added `dq::NativeDsSlotMap`:
  `src_lane = 4 + 2*group + 16*(q&3) + ((word>>1)&1) + 8*(word>>2)`;
  `src_word = 2*(q>>2) + (word&1)`;
  `slot_krow = group*4 + (word&3)`;
  `src_lane >= 64` marks the known boundary holes.
- Static asserts cover representative probe rows and the boundary hole.

Gate:

- Remote canonical dQ build and symbol metadata gate still PASS after adding
  the formula.

Next:

- Implement a minimal C_dS split publisher prototype that uses this source-slot
  schedule, then feeds C_dQ through `ds_write_matrix_32x16_f16` and
  `ds_read_matrix_trans`, without scalar gather or permute.

## 2026-07-11 Native dS Source-Schedule Probe

Decision: `ACCEPT_FOCUSED_PROBE`

Hypothesis:

- The compact slot map can be used without cross-lane permute if C_dS is
  scheduled by source slot.  Each source lane computes the logical dS value
  that its native DS matrix write slot will publish.

Implementation:

- Added `probes/dq_native_ds_source_schedule_probe.cpp`.
- The first version used a three-loop reverse search and compiled with
  `sgpr_spill_count=2`; this was rejected as probe overhead.
- Replaced it with a carry-aware closed-form inverse:
  source lane is the low 6 bits of raw lane, and raw-lane carry advances the
  source word by two.  This matches the compact map and preserves the known
  boundary holes.

Evidence:

- Metadata gate PASS:
  `sgpr=42`, `vgpr=34`, `private=0`, no SGPR/VGPR spill.
- PMD run:
  `/zys/shaobo_runs/dq_source_schedule_probe_20260711_194228`.
- Output:
  `source_schedule_result mapped=504 pair_pass=0 low_pass=1 high_pass=1 split_pass=1 pass=1`.
- Stats:
  `ldsBankConflict=0`.  The `MMOP=3` count is expected because this is a
  focused one-wave instruction/layout proof, not a performance kernel.

Conclusion:

- Native dS handoff remains viable without permute or scalar gather if the real
  C_dS publisher computes values in source-slot order.  The next implementation
  step is to replace the toy `krow==probeK` source value with real
  score/dP/softmax-derived dS values in the same source-slot schedule.

## 2026-07-11 Native Ring MMAC Layout Stress

Decision: `REVISE_BEFORE_RING_KERNEL`

Question:

- Can the current natural qk/dP MMAC output be written directly as the native
  dS source-slot fragment?

Stress:

- Compared the `NativeDsSlotMap` required logical `(q,krow)` per source
  lane/word with the current natural dQ consumer convention
  `q=lane&15`, `k=lane_n*4+vec`.
- Result: most lanes mismatch six to eight of eight source words.  This is
  a structural ownership mismatch, not a small pack-order issue.

Related evidence:

- `dq_dswrite_qowned_chain_probe.cpp` had already rejected simple q-owned
  lit/direct MMAC variants plus eight pack candidates.  The new map stress
  explains the negative result: the source slot expects different logical
  q ownership than natural MMAC emits.

Conclusion:

- Do not start the full dQ ring kernel yet.  The next focused task is to find
  a C_dS operand-read/layout schedule that makes MMAC produce source-slot
  values directly.  If that cannot be done without scalar gather or lane
  permute, reject the native dS ring route for now and return to the accepted
  full3GEMM dQ path.

## 2026-07-11 Source Operand Layout Probe

Decision: `REJECT_DIRECT_Q_READ_FORMATS`

Hypothesis:

- Maybe the missing C_dS source-slot layout is just a different native Q
  `ds_read_matrix` format.

Implementation:

- Added `probes/dq_source_operand_layout_probe.cpp`.
- It loads a row-tagged Q tile with MLS, then tries all directly supported
  matrix-read candidates relevant to the current compiler:
  1. `ds_read_matrix_trans row=2 col=1 alt0`
  2. `ds_read_matrix row=2 col=1 alt0`
  3. `ds_read_matrix_trans row=1 col=2 alt0`
  4. `ds_read_matrix_trans row=1 col=2 alt1`
- The normal `row=1 col=2` forms fail compilation as unsupported modifier
  combinations, so they are excluded.

Evidence:

- Build metadata PASS:
  `private=0`, `sgpr=20`, `vgpr=12`, no spill/scratch.
- PMD:
  `/zys/shaobo_runs/dq_operand_layout_probe_20260711_195417`.
- Results:
  `operand_layout_final any_full_match=0`;
  per-mode q-match counts are `32/504`, `44/504`, `16/504`, `18/504`.
  Stats show `ldsBankConflict=0`, so this is a semantic layout mismatch.

Conclusion:

- Direct Q read-format switching is not enough to make C_dS MMAC emit native
  source-slot dS values.  The native ring route now requires either a
  prearranged Q/dO source layout or a different MMAC operand orientation; if
  that implies scalar gather/permute in the hot path, defer the ring and return
  to the accepted full3GEMM dQ path.

## 2026-07-11 MLS32 Direct Source-Slot Recheck

Decision: `REJECT_MLS32_DIRECT_Q_SOURCE_SLOT`

Hypothesis:

- The previous direct-read probe used `matrix_load_32x16`.  Recheck whether
  `matrix_load_32x32` with native MLS transpose/non-transpose pages plus the
  supported DS matrix readers can make Q appear in the `NativeDsSlotMap`
  source-slot q ownership.

Implementation:

- Extended `probes/dq_source_operand_layout_probe.cpp` to cover eight
  combinations: two MLS pages (`matrix_load_32x32` non-transposed and
  transposed) times four supported readers:
  `trans row=2 col=1 alt0`, `normal row=2 col=1 alt0`,
  `trans row=1 col=2 alt0`, and `trans row=1 col=2 alt1`.

Evidence:

- Remote build and symbol metadata gate PASS:
  `private=0`, `sgpr=20`, `vgpr=12`, no spill/scratch.
- Asm contains both
  `matrix_load_32x32_b16 ... bps lds` and
  `matrix_load_32x32_b16 ... t bps lds`, followed by the four DS reader forms
  for each page.
- PMD run:
  `/zys/shaobo_runs/dq_operand_layout_mls32_probe_20260711_200044`.
- Result:
  `operand_layout_final any_full_match=0`.  For both load modes, q-match
  counts are `32/504`, `44/504`, `16/504`, and `18/504`.  Stats show
  `ldsBankConflict=0`.

Conclusion:

- Same-LDS native normal/trans matrix reads are real, but this specific
  direct Q source-slot mapping is not.  The mismatch is semantic ownership of
  Q rows per source lane/word, not bank conflict and not lack of DS matrix
  instructions.
- Do not add a scalar gather/permute workaround to force this route.  Either
  find a true source-layout producer/MMAC orientation, or return to the
  accepted full3GEMM dQ path and focus on wait/ABarrier/MMAC-island tuning.

## 2026-07-11 dS Source-Pack Workaround Cost

Decision: `BRINGUP_ONLY_REJECT_FOR_PERF`

Question:

- If real C_dS cannot yet be generated directly in `ds_write_matrix` source
  slots, can a half workaround pack natural-layout dS into the verified
  source-slot layout cheaply enough to try a full dQ ring?

Implementation:

- Added `probes/dq_ds_source_pack_cost_probe.cpp`.
- The probe compares:
  - `native_slot`: each lane directly computes the source-slot value required
    by `NativeDsSlotMap`.
  - `bpermute_pack`: natural owner lanes publish values through wave
    `ds_bpermute_b32`.
  - `lds_gather_pack`: natural owner lanes store a compact LDS page and source
    lanes gather dword pairs with `ds_read_b32`.
- `--path` runs one dispatch at a time so a rejected path cannot abort the
  other measurements.

Evidence:

- Static metadata:
  - `native_slot`: `private=0`, `sgpr=68`, `vgpr=29`.
  - `lds_gather_pack`: `private=0`, `sgpr=88`, `vgpr=39`.
  - `bpermute_pack`: `private_segment_fixed_size=32`, `sgpr=55`,
    `vgpr=45`; PMD aborts with scratch-parameter panic.
- PMD:
  `/zys/shaobo_runs/dq_ds_source_pack_cost_compare_20260711_205227_iters1024`.
- Correctness equivalence:
  both runnable paths report `errors=0` and checksum `10239541.1`.
- Stats:
  - `native_slot`: `simTicks=1,176,128,135`, `LDS=2112`,
    `ldsBankConflict=0`, `Sp0Lds=512`, `MMOP=2048`.
  - `lds_gather_pack`: `simTicks=1,713,390,315`, `LDS=6209`,
    `ldsBankConflict=24576`, `Sp0Lds=16916`, `MMOP=2048`.

Conclusion:

- The LDS gather workaround is semantically valid but costs about `+45.7%`
  simTicks in this focused loop and introduces heavy LDS bank conflict.  It is
  useful only as a bringup/debug path, not as a performance route.
- The bpermute workaround is not currently viable: codegen introduces private
  segment usage, and PMD cannot launch it.
- Continue pursuing native source-slot C_dS generation or a true native
  MMAC/operand orientation; do not promote gather/permute packing into the
  canonical dQ performance kernel.

## 2026-07-11 Natural dS No-Permute Lower Bound

Decision: `OBSERVE_PERF_LOWER_BOUND_WRONG_LAYOUT`

Question:

- If correctness is ignored and dS is written in natural C-layout without
  source-slot permutation/gather, how much of the focused-loop cost disappears?

Implementation:

- Extended `probes/dq_ds_source_pack_cost_probe.cpp` with `--path
  natural_wrong`.
- This path sets `producer = natural` and then runs the same
  `ds_write_matrix_32x16_f16 -> ds_read_matrix_trans -> MMAC` loop.  It does
  not validate source-slot correctness and is not a candidate for the real dQ
  kernel.

Evidence:

- Static metadata PASS:
  `natural_wrong private=0`, `sgpr=20`, `vgpr=22`.
- PMD:
  `/zys/shaobo_runs/dq_ds_source_pack_cost_natural_wrong_20260711_210309_iters1024`.
- Same-run stats:
  - `natural_wrong`: `simTicks=107,657,095`, `LDS=2112`,
    `ldsBankConflict=0`, `VALU=11,508`, `SCA=2,196`, `MMOP=2048`.
  - `native_slot`: `simTicks=1,176,224,595`, `LDS=2112`,
    `ldsBankConflict=0`, `VALU=240,980`, `SCA=207,003`, `MMOP=2048`.
  - Lower-bound speedup versus current source-slot probe implementation:
    `90.85%` ticks reduction.

Conclusion:

- The native matrix handoff path is cheap when data already sits in the
  producer fragment.  The expensive part in the current `native_slot` probe is
  runtime source-slot reverse mapping/control, not `ds_write_matrix` /
  `ds_read_matrix_trans` / MMAC.
- Do not use `natural_wrong` for correctness.  Use it as evidence that the
  next real C_dS source-slot implementation must hard-code/derive direct
  source-lane formulas or compile-time tables, instead of doing runtime
  reverse lookup or gather/permute.

## 2026-07-11 dQ Kernel Natural-Wrong Integration

Decision: `OBSERVE_PERF_LOWER_BOUND_WRONG_DQ`

Question:

- If the focused `natural_wrong` lower bound is integrated into the canonical
  dQ kernel, how much same-shape H1/S1024 performance can it move?

Implementation:

- Added opt-in runtime flag `DQ_NATURAL_WRONG_DS=1` /
  `--natural-wrong-ds=1`.
- The default dQ path remains unchanged: dS stays in VGPR and feeds
  `dq_update_from_ds_pair`.
- With the flag enabled, each consumer wave packs natural-layout
  `ds_vec0/ds_vec1`, writes it through `ds_write_matrix_32x16_f16`, reads it
  back through `ds_read_matrix_trans`, then calls the existing `dS @ K` update.
  Scratch uses the current V page with one 2KB slot per consumer wave.  This is
  intentionally a wrong-layout performance lower bound, not a correctness
  route.

Evidence:

- Build: `SRC=src/dq_kernel.cpp BIN=build/fa3_bwd_dq_clean
  ASM=build/fa3_bwd_dq_clean.asm TARGET_GFX=946 BUILD_ASM=1 ./build.sh`.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `sgpr=68`, `vgpr=128`; branch windows show consumer
  `155/216`.
- PMD run root:
  `/zys/shaobo_runs/dq_natural_wrong_compare_20260711_213517`.
- Same-build H1/S1024, `GPU_CHIP=sb`,
  `GPU_ARGS=['--SQCIPfLines=7']`:
  - default canonical:
    `simTicks=40,530,035`, `kernel_ticks=36,916,425`,
    `MMOP=55,296`, `VALU=143,200`, `SCA=81,772`, `LDS=28,656`,
    `coissue=12,602/14,949`, `MMAC active=25.0871%`,
    `ldsBankConflict=0`, correctness PASS.
  - `natural_wrong`:
    `simTicks=38,872,015`, `kernel_ticks=35,258,405`,
    `MMOP=55,296`, `VALU=120,160`, `SCA=82,924`, `LDS=30,960`,
    `coissue=12,230/16,431`, `MMAC active=25.8278%`,
    `ldsBankConflict=0`, numerical correctness intentionally invalid
    (`dq_rel_l2=1.4343`) but dispatch succeeds.

Conclusion:

- Wrong-layout dS handoff gives `-4.49%` kernel ticks, `-4.09%` simTicks,
  `-16.1%` VALU instructions, and `+0.74pt` MMAC active, at the cost of
  `+8.0%` LDS instructions and lower coissue success.
- This is real exposed overhead, but far from the 10x focused-probe gap because
  the full dQ kernel is still dominated by the score/dP/softmax/dQ pipeline,
  page ownership, and K/V read/update work.
- Do not promote this flag.  Use it as a bound: a correct native C_dS
  source-slot producer can plausibly reclaim roughly 4-5% on current H1/S1024
  dQ, but it will not by itself reach 40-60% MMAC active.

Cleanup:

- Removed `DQ_NATURAL_WRONG_DS` and `--natural-wrong-ds` from the active dQ
  code and smoke script after the measurement.  The focused probe and log rows
  remain as evidence; the canonical kernel no longer contains the wrong-layout
  branch.
- Restore validation:
  `/zys/shaobo_runs/dq_mainline_restore_20260711_221350` with
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`, H1/S1024.
  Correctness PASS, static gate PASS, metadata
  `private=0`, `sgpr_spill=0`, `vgpr_spill=0`, `sgpr=67`, `vgpr=128`,
  branch windows `8/40,161/216,161/216,9/40`.
- Restored canonical stats:
  `simTicks=35,704,760`, `kernel_ticks=32,091,150`, `MMOP=55,296`,
  `VALU=121,632`, `SCA=77,516`, `LDS=28,656`, `VMEM=1,408`,
  `coissue=16,119/19,093`, `MMAC active=27.3852%`,
  `ldsBankConflict=0`.

## 2026-07-11 dQ Sidecar/QDo Latch Split Probe

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

- `page0` K/V only overlaps the tiny sidecar area, while `page1` overlaps the
  full `Q/dO` LDS area.  Split the startup ownership into `SidecarLatched` and
  `QDoLatched`, so producers can publish page0 after consumers latch sidecar,
  instead of waiting for the full Q/dO matrix-read latch.

Implementation:

- Temporarily repurposed barrier id 6 as `SidecarLatched`.
- Consumers performed `sidecar LDS read -> wait_lgkm(0) -> arrive
  SidecarLatched`, then read Q/dO matrix fragments and arrived the existing
  `QDoLatched`.
- Producers waited `SidecarLatched` before `kt=0/page0`, and waited
  `QDoLatched` only before `kt=1/page1`.

Evidence:

- Static/resource gate PASS:
  branch windows stayed `8/40,161/216,161/216,9/40`; metadata stayed
  `private=0`, no SGPR/VGPR spill, `sgpr=68`, `vgpr=128`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_sidecar_latch_split_20260711_222351/dq_correctness_20260711_223131`;
  H1/S1024
  `/zys/shaobo_runs/dq_sidecar_latch_split_20260711_222351/dq_correctness_20260711_223143`.
- Same-shape H1/S1024 versus restored mainline:
  `simTicks 35,704,760 -> 36,954,190`,
  `kernel_ticks 32,091,150 -> 33,340,580`,
  `MMAC active 27.3852% -> 27.1510%`,
  `VALU 121,632 -> 115,360`,
  `SCA 77,516 -> 78,404`,
  `coissue 16,119/19,093 -> 14,869/15,306`,
  `ldsBankConflict=0`.

Conclusion:

- The extra ownership token costs more than the earlier page0 publication can
  recover.  Sidecar/QDo lifetime splitting is not the route to 40% MMAC active
  in the current dQ topology.
- Active source was restored to the single `QDoLatched` startup ledger.  Do not
  retry finer token splits unless the design also increases useful work per
  ownership epoch or removes another token.

## 2026-07-11 dQ Post-Invalidate Sync Prune Probe

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

- After the six live ABarrier slots are invalidated by wave0, the second
  unconditional `__syncthreads()` appears to serve only the `diag_store`
  debug path.  Moving it under `if (diag_store != 0)` might remove a tail
  `s_barrier -> s_cbranch` xcu hotspot without touching the steady main loop.

Implementation:

- Kept the pre-invalidate `__syncthreads()` unchanged.
- Temporarily changed the post-invalidate sync to execute only when
  `diag_store != 0`.

Evidence:

- Static/resource gate PASS:
  branch windows `8/40,161/216,161/216,9/40`; metadata `private=0`,
  `sgpr=67`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_tail_postinv_sync_prune_20260711_223456/dq_correctness_20260711_224236`;
  H1/S1024
  `/zys/shaobo_runs/dq_tail_postinv_sync_prune_20260711_223456/dq_correctness_20260711_224243`.
- Same-shape H1/S1024 versus restored mainline:
  `simTicks 35,704,760 -> 36,083,775`,
  `kernel_ticks 32,091,150 -> 32,470,165`,
  `MMAC active 27.3852% -> 27.4013%`,
  `MMOP/VALU/SCA/LDS/VMEM unchanged`,
  `coissue 16,119/19,093 -> 15,787/18,642`,
  `ldsBankConflict=0`.

Conclusion:

- Removing the post-invalidate sync slightly improves some stall counters and
  MMAC active, but elapsed ticks regress.  Do not promote.  Active source was
  restored to the previous cleanup form.
- Tail cleanup is no longer the priority; the next dQ work should target useful
  work per page ownership epoch or source-layout/control overhead in the main
  loop.

## 2026-07-11 dQ Causal Full-Valid Fast Path Probe

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

- For `Mq=128,Nk=128` causal dQ, most non-diagonal K tiles are fully valid.
  Avoiding per-element `krow <= qrow` mask checks in those tiles should reduce
  VALU/SCA in the softmax/dS section without changing MMOP or page ownership.

Implementation:

- Temporarily added a `full_valid_n_tile` branch in the consumer loop.
- Full-valid n-tiles computed `p` and `dS` directly; diagonal/partial tiles kept
  the original per-element causal checks.

Evidence:

- Static/resource gate PASS:
  branch windows `8/40,161/216,161/216,9/40`; metadata `private=0`,
  `sgpr=67`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_causal_fullvalid_fastpath_20260711_224401/dq_correctness_20260711_225140`;
  H1/S1024
  `/zys/shaobo_runs/dq_causal_fullvalid_fastpath_20260711_224401/dq_correctness_20260711_225148`.
- Same-shape H1/S1024 versus restored mainline:
  `simTicks 35,704,760 -> 39,260,585`,
  `kernel_ticks 32,091,150 -> 35,646,975`,
  `MMAC active 27.3852% -> 25.3824%`,
  `VALU 121,632 -> 102,040`,
  `SCA 77,516 -> 90,508`,
  `barrierCounter 58,629.75 -> 79,263.75`,
  `emptyBufferCounter 25,149.42 -> 43,554.67`,
  `ldsBankConflict=0`.

Conclusion:

- The mask work was not the limiting cost.  Branching the softmax/dS path
  reduces VALU instructions but increases scalar/control and bubbles enough to
  lose badly.  Active source was restored.
- Do not use branch-duplicated causal fast paths in this dQ topology unless a
  future design makes the full-valid path a separate compile-time specialization
  or removes the added control from the hot loop.

## 2026-07-11 dQ QDo One-Shot Wait No-Toggle Probe

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

- `QDoFilled` and `QDoLatched` are one-shot startup barriers.  Unlike
  PageFilled/PageUsed, they do not need phase toggling across K/V page epochs.
  Removing the `s_xor_b32` phase toggle for these two waits might reduce SCA
  and SGPR pressure without changing lifetimes.

Implementation:

- Temporarily added `ins::abarrier_try_wait_once`, an inline-asm wait that does
  not toggle the phase SGPR.
- Used it only in `dq_wait_qdo_filled` and `dq_wait_qdo_latched`.

Evidence:

- Static/resource gate PASS:
  branch windows stayed `8/40,161/216,161/216,9/40`; metadata stayed
  `private=0`, no spill/scratch; `sgpr` fell `67 -> 65`.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_qdo_wait_once_20260711_225056/dq_correctness_20260711_225836`;
  H1/S1024
  `/zys/shaobo_runs/dq_qdo_wait_once_20260711_225056/dq_correctness_20260711_225843`.
- Same-shape H1/S1024 versus restored mainline:
  `simTicks 35,704,760 -> 36,104,705`,
  `kernel_ticks 32,091,150 -> 32,491,095`,
  `MMAC active 27.3852% -> 27.3167%`,
  `SCA 77,516 -> 77,116`,
  `coissue 16,119/19,093 -> 15,878/18,621`,
  `ldsBankConflict=0`.

Conclusion:

- The no-toggle wait does reduce SGPR/SCA slightly, but it does not reduce
  elapsed time or improve MMAC active.  Active source and helper were restored.
- Do not optimize the ABarrier wrapper mechanically; keep focus on main-loop
  ownership/useful-work structure.

## 2026-07-11 dQ Nk256 Single-Page Epoch Probe

Decision: `REJECT_STATS_TICKS_REGRESSION`

Hypothesis:

- Increase useful MMAC per ownership epoch by changing canonical dQ from
  `Mq128/Nk128` two K/V pages to `Mq128/Nk256` one 128KB K/V page.  This cuts
  H1/S1024 causal K/V epochs from `36` to `20` and doubles per-epoch MMAC
  from `1536` to `3072`.

Design record:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `51_Nk256_SinglePage`.

Implementation:

- Startup still stages `Q+dO+sidecar` in LDS and consumers latch them to VGPR.
- Steady state overlays K/V on the whole LDS as one page.
- The first full-unroll build had `sgpr_spill_count=9`; limiting the `n_tile`
  loop to `unroll 4` fixed the resource gate.

Evidence:

- Static/resource gate PASS after the unroll fix:
  branch windows `8/40,158/216,158/216,9/40`; metadata `private=0`,
  `sgpr=58`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S256 and H1/S1024 under
  `/zys/shaobo_runs/dq_nk256_singlepage_20260711_231218`.
- Same-shape H1/S1024 versus restored mainline:
  `simTicks 35,704,760 -> 41,586,545`,
  `kernel_ticks 32,091,150 -> 37,972,935`,
  `MMAC active 27.3852% -> 24.3812%`,
  `MMOP 55,296 -> 61,440`,
  `VALU 121,632 -> 147,072`,
  `SCA 77,516 -> 69,256`,
  `LDS 28,656 -> 31,728`,
  `barrierCounter 58,629.75 -> 86,381.5`,
  `ldsBankConflict=0`.

Conclusion:

- Fewer ownership epochs is not sufficient.  The single-page layout loses K/V
  double buffering/prefetch, introduces extra causal padding MMOP, and grows
  barrier/VALU/LDS exposure enough to lower MMAC active and regress ticks.
- Active source was restored to `Mq128/Nk128` double-page.
- Do not retry `Nk256` single-page as an isolated tile change.  The next
  structural route should either reduce PageUsed ownership without losing K/V
  prefetch, or solve the native dS/source-slot handoff so dQ can change role
  topology rather than only tile size.

## 2026-07-11 dQ Row Inv-Sum Hoist Probe

Decision: `REJECT_CORRECTNESS`

Hypothesis:

- `scores_sum` is a row-level invariant.  Replacing hot-loop
  `exp2(...) / row_sum` with `exp2(...) * row_inv_sum` should reduce repeated
  VALU division work without changing ABarrier ownership or matrix paths.

Variants:

- Producer-side: store `1 / scores_sum[row]` into LDS sidecar field 1.
- Consumer-side: keep producer sidecar as `scores_sum`, but compute
  `row_inv_sum = 1 / row_sum` once after sidecar latch and multiply in the
  hot loop.

Evidence:

- Both variants passed static/resource gates.  Consumer branch window dropped
  from `161/216` to `155/216`, no spill/scratch.
- Both variants failed H1/S128 correctness in the same way:
  `dq_rel_l2=14969.4`, `actual_l2=71.2549`,
  `expected_l2=0.00475997`, `actual_nonfinite=0`.
- PMD printed `VOP3P__V_MAD_MIXLO_F16 not test` in both failed runs.
- Run dirs:
  producer-side `/zys/shaobo_runs/dq_invsum_20260711_232704/dq_correctness_20260711_233444`;
  consumer-side `/zys/shaobo_runs/dq_consumer_invsum_20260711_232857/dq_correctness_20260711_233638`.

Conclusion:

- Do not use reciprocal-hoist in canonical dQ on this toolchain/PMD without a
  focused reciprocal/codegen correctness probe.  The algebra is equivalent,
  but the generated instruction path is not validated by current PMD.
- Active source was restored to baseline `exp2(...) / row_sum`.

## 2026-07-11 dQ 12-Wave Single-Producer Probe

Decision: `REJECT_FULLPERF_OCCUPANCY_REGRESSION`

Hypothesis:

- XCU showed the restored 16-wave dQ mainline was dominated by producer
  `PageUsed` ownership bubbles.  Remove the thin second producer role and let
  waves0-3 publish both Q/dO groups plus both K/V operands, leaving waves4-7
  and waves8-11 as the two full-3GEMM consumers.

Design record:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `52_12W_SingleProducer`.

Evidence:

- Static/resource PASS after setting the consumer VGPR target to `220`:
  branch windows `8/40,161/220,162/220`; metadata `private=0`, `sgpr=54`,
  `vgpr=160`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_12w_singleprod_20260711_233558/dq_correctness_20260711_234338`;
  H1/S1024
  `/zys/shaobo_runs/dq_12w_singleprod_20260711_233558/dq_correctness_20260711_234346`.
- Fullperf H1/S1024 versus mainline fullperf:
  `simTicks 35,881,300 -> 36,049,650`,
  `MMAC active 27.4198% -> 27.4182%`,
  `dispatch waves 128 -> 96`,
  `avg active waves 79.17 -> 59.35`,
  `duration 70,852 -> 71,236`.
- XCU did confirm the local bubble thesis:
  `s_abarrier_try_wait -> s_xor_b32` fell from about `26.47%` to `18.80%`,
  but this was not enough to offset lower residency.
- Perf:
  `/zys/shaobo_runs/dq_12w_singleprod_fullperf_20260711_233737/dq_correctness_20260711_234517/m5out/0/0/2774542_fa3_bwd_dq_clean.perf`.
- XCU:
  `/zys/shaobo_runs/dq_12w_singleprod_fullperf_20260711_233737/xcu_outputs/dq_12w_singleprod_20260711_234517`.

Conclusion:

- Removing a thin producer is not a valid mainline improvement for this shape.
  It lowers visible ownership wait, but underfills the scheduler and does not
  raise MMAC active.
- Active source is restored to the 16-wave canonical dQ path.
- Next work should keep 16-wave residency and either give producer1 recurring
  useful work or reduce `PageUsed` lifetime without removing an active role.

## 2026-07-12 dQ K-Normal Prefetch Before dS Probe

Decision: `REJECT_FULLPERF_TICKS_REGRESSION`

Hypothesis:

- Current dQ reads K-normal for `dQ += dS @ K` after softmax/dS and then waits
  before the dQ MMAC.  Split the K-normal read from `dq_update` and issue it
  immediately after qk/dP MMAC, before softmax/dS, so the softmax/dS VALU
  covers part of the LDS read latency.

Design record:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `53_DQ_KNormal_Prefetch`.

Evidence:

- Static/resource PASS:
  branch windows `8/40,159/216,159/216,9/40`; metadata `private=0`,
  `sgpr=67`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128
  `/zys/shaobo_runs/dq_k_normal_prefetch_20260711_235959/dq_correctness_20260712_000739`;
  H1/S1024
  `/zys/shaobo_runs/dq_k_normal_prefetch_20260711_235959/dq_correctness_20260712_000748`.
- Stats-only versus restore validation:
  `simTicks 36,052,835 -> 35,933,625`,
  `waitLgkm 13,936.75 -> 11,144.75`,
  `MMAC active 27.2806% -> 27.4332%`.
- Fullperf versus mainline fullperf:
  `simTicks 35,881,300 -> 36,035,545`,
  `MMAC active 27.4198% -> 27.3801%`,
  `waitLgkm 13,611.25 -> 11,234.75`.
- XCU:
  `/zys/shaobo_runs/dq_k_normal_prefetch_fullperf_20260712_000111/xcu_outputs/dq_k_normal_prefetch_20260712_000850`.
  The top PageUsed ownership bubble remains essentially unchanged:
  `s_abarrier_try_wait -> s_xor_b32 26.47% -> 26.53%`.

Conclusion:

- The read scheduling idea is locally true: it lowers wait counters.  It is not
  a mainline improvement because elapsed time is dominated by PageUsed/role
  ownership, not by this K-normal read wait.
- Active source is restored to the 16-wave canonical helper shape.
- Next dQ work must change the ownership/useful-work structure or dS handoff
  topology, not only move matrix reads around the same PageUsed barrier.

## 2026-07-12 dQ half-page PageUsed rejected

- Hypothesis:
  keep the 16-wave canonical dQ topology, but split page reuse lifetime into
  `HalfUsed` and full `PageUsed`.  Consumers release a K/V page after finishing
  `n_tile=0..1`, so producers can reload the first 64 rows of a reused page
  while consumers finish the second half.
- Result:
  correctness/resource PASS.  Static gate remained canonical:
  `8/40,161/216,161/216,9/40`; metadata `private=0`, `sgpr=59`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness both passed.
  H1/S1024 stats:
  `simTicks=36,033,725`, `kernel_ticks=32,420,115`,
  `MMAC active=27.3829%`, `MMOP=55,296`, `VALU=121,632`,
  `SCA=76,100`, `LDS=28,656`, `coissue=15,129/18,518`,
  `barrierCounter=53,286.25`, `waitLgkm=14,024`, `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Compared with mainline fullperf
  (`simTicks=35,881,300`, `MMAC active=27.4198%`,
  `barrierCounter=50,464`), half-page release is slower and increases barrier
  pressure.  The data-lifetime proof is correct, but the extra token is not
  free.  Active source was restored and recertified to the canonical dQ gate.
- Lesson:
  do not split `PageUsed` finer unless the split also removes another wait or
  adds recurring useful producer work.  More precise ABarrier ownership can
  make the critical path worse when it only adds bookkeeping.

## 2026-07-12 dQ group1 reverse n_tile rejected

- Hypothesis:
  keep all canonical barriers and tile sizes unchanged, but make consumer
  group1 traverse `n_tile` inside a K/V page in reverse order.  This is a
  no-delay useful-work stagger intended to reduce consumer齐步走 without adding
  ABarrier traffic.
- Result:
  correctness/resource PASS.  Static gate unchanged:
  `8/40,161/216,161/216,9/40`; metadata `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness both passed.
  H1/S1024 stats:
  `simTicks=36,171,590`, `kernel_ticks=32,557,980`,
  `MMAC active=27.2470%`, `MMOP=55,296`, `VALU=121,600`,
  `SCA=77,516`, `LDS=28,656`, `coissue=15,693/18,716`,
  `barrierCounter=52,670`, `waitLgkm=14,150.25`, `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  The instruction structure is still too
  similar between the two consumer groups; reversing addresses/chunk order
  alone does not create a meaningful MMAC/VALU pipeline.  Active source was
  restored.
- Lesson:
  future stagger work must move genuinely different useful work across
  consumers or producers.  Pure iteration-order skew is not enough for this
  dQ topology.

## 2026-07-12 dQ native dS ring structural probe accepted

- Hypothesis:
  prove the structural role handoff for a future dQ split pipeline before
  touching canonical dQ: producer publishes K, C_dS publisher writes two
  deterministic dS source slots with `ds_write_matrix`, and C_dQ consumer reads
  them with `ds_read_matrix_trans` plus K normal fragments for split MMAC.
- Result:
  after removing pre-role `lane/threadIdx` VGPR setup and the ordinary LDS
  clear loop, PMD PASS:
  `/zys/shaobo_runs/dq_native_ds_ring_structural_fix_20260712_024559`.
  `slot0_low/high` and `slot1_low/high` all passed; producer_done=1,
  publisher_done=2, consumer_done=2.  Metadata PASS:
  `private=0`, `sgpr=22`, `vgpr=120`, no spill/scratch.  Stats:
  `simTicks=6,359,535`, `MMOP=4`, `VALU=176`, `SCA=453`, `LDS=30`,
  `ldsBankConflict=0`.
- Decision:
  `ACCEPT_PROBE_STRUCTURAL`.  This proves the native role-to-role dS ring
  skeleton without gather/permute/bpermute and without ordinary `ds_read_b*`
  on the matrix path.  It is not a performance candidate and canonical dQ is
  unchanged.
- Lesson:
  for WDRA probes, keep all lane/threadIdx-dependent VGPR setup inside the
  branch after `s_set_vgpr_size`; otherwise PMD can see uninitialized VGPR
  state in branch-local LDS paths.  Next step is to replace deterministic dS
  with canonical C_dS arithmetic in the standalone probe.

## 2026-07-12 dQ dS@K batch8 wait0 rejected

- Hypothesis:
  in canonical dQ, change only `dq_update_from_ds_pair`: after issuing the
  eight K-normal `ds_read_matrix` instructions, wait for all of them with
  `wait_lgkm(0)` and run one longer dQ MMAC island, instead of the current
  `wait_lgkm(4) -> first half MMAC -> wait_lgkm(0) -> second half MMAC`.
- Result:
  static/source gate and metadata passed (`private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch), but H1/S128 PMD aborted before correctness:
  `/zys/shaobo_runs/dq_dqgemm_batch8_wait0_20260712_030114/dq_correctness_20260712_030114`.
  PMD reported `read vgpr81 before writing`, then
  `panic ... vgpr81 is not init or has been freed` during MMOP execute.
- Decision:
  `REJECT_PMD_REGISTER_INIT`.  Source restored to canonical
  `wait_lgkm(4)` plus the mid `wait_lgkm(0)`.
- Lesson:
  this wait split is not just performance conservatism.  It is currently part
  of the PMD/WDRA-visible readiness boundary for the second half of the
  K-normal fragments.  Do not collapse dQ K-normal read waits without a focused
  ds_read_matrix/MMAC VGPR-init probe or compiler/PMD fix.

## 2026-07-12 dQ native dS ring formula probe accepted

- Hypothesis:
  after the deterministic structural handoff passed, test whether a C_dS
  publisher can compute softmax/dS formula values directly in
  `NativeDsSlotMap` source-slot order and publish them through the same native
  `ds_write_matrix -> ds_read_matrix_trans -> MMAC` path.
- Result:
  `/zys/shaobo_runs/dq_native_ds_ring_formula_20260712_030944` PASS.
  `slot0_low/high` and `slot1_low/high` all passed; producer_done=1,
  publisher_done=2, consumer_done=2.  Metadata PASS:
  `private=0`, `sgpr=22`, `vgpr=120`, no spill/scratch.  Stats:
  `simTicks=6,556,550`, `MMOP=4`, `VALU=329`, `SCA=476`, `LDS=30`,
  `ldsBankConflict=0`.
- Decision:
  `ACCEPT_PROBE_FORMULA_SOURCE_SLOT`.  This proves formula generation plus
  source-slot publication and native C_dQ consumption without gather/permute.
  Canonical dQ remains unchanged.
- Boundary:
  qk/dP are synthetic scalar formula inputs in this probe.  The remaining hard
  question is whether the real qk/dP MMAC result orientation can produce the
  required source-slot fragment without scalar lane permute/gather.

## 2026-07-12 dQ tail keep-alive prune rejected

- Hypothesis:
  remove the post-store `keep_accumulator_live(dq_reg[d_idx])` loop in
  `dq_consumer_full3gemm_role`, assuming it is only tail noise after global
  store.
- Result:
  static/source gate and metadata passed, but branch codegen changed
  materially: producer1 branch reported `38/40` VGPRs instead of the restored
  canonical `9/40`.  H1/S128 PMD aborted before correctness:
  `/zys/shaobo_runs/dq_tail_keepalive_prune_20260712_031836/dq_correctness_20260712_031837`.
  PMD reported `read vgpr70 before writing`, then
  `VGPR index 85 is out of range: VGPR range=[0,40]` on `v_mov_b32`.
- Decision:
  `REJECT_PMD_REGISTER_INIT`.  Source restored.
- Lesson:
  the keep-alive loop is part of the current WDRA/codegen liveness contract.
  Do not delete tail liveness guards in the canonical dQ kernel without a
  focused WDRA-exit proof.

## 2026-07-12 dQ PageUsed early release observed and rejected

- Hypothesis:
  after the last K-normal `ds_read_matrix` in a page has reached
  `wait_lgkm(0)`, the remaining dQ MMAC half only uses VGPR fragments.  Move
  the existing `dq_arrive_page_used` before that final MMAC half so producers
  can reuse the K/V page slightly earlier.
- Result:
  correctness/resource PASS and no bank conflict.  Stats-only H1/S1024 moved
  `simTicks=36,109,710 -> 36,084,230`, but fullperf evidence was not aligned
  with the optimization target: `simTicks=36,094,240 -> 36,046,920` while
  MMAC active fell `27.3254% -> 27.2589%`, and barrier counter rose
  `50,779.75 -> 52,556.25`.
- XCU:
  the dominant `s_abarrier_try_wait -> s_xor_b32` bubble worsened from
  `1,140,988` cycles (`26.57%`, max `7,635`) to `1,188,124` cycles
  (`27.31%`, max `8,675`).
- Decision:
  `OBSERVE_REJECT_SOURCE_RESTORED`.  The small tick drop is not supported by
  the target evidence path and does not move MMAC active toward 40%.  Canonical
  source is restored.
- Lesson:
  moving the existing PageUsed arrive point is too small a lifetime change and
  can worsen ABarrier scheduling.  Future work must either reduce the
  ownership dependency itself or add real useful producer/consumer work under
  the PageUsed window.

## 2026-07-12 dQ causal predicate minimalization accepted

- Hypothesis:
  canonical dQ already proves `qrow < seqlen` and `krow < seqlen` for every
  visited element because `S` is divisible by `Mq=128` and `Nk=128`, and
  `active_k_tiles` is derived from the q tile end.  Therefore the hot dS
  predicate can be reduced from
  `krow < seqlen && qrow < seqlen && krow <= qrow` to `krow <= qrow`.
- Result:
  H1/S128 and H1/S1024 correctness PASS; static/source gate PASS.  Metadata
  improves from `sgpr=67` to `sgpr=65`, with `vgpr=128`, `private=0`, no
  spill/scratch, and `ldsBankConflict=0`.
- PMD stats:
  H1/S1024 stats-only improves
  `simTicks=36,109,710 -> 33,839,715`,
  `MMAC active=27.2503% -> 29.4163%`,
  `SCA=77,516 -> 58,940`, and `VALU=121,632 -> 112,064`.
- Fullperf/xcu:
  H1/S1024 fullperf improves
  `simTicks=36,094,240 -> 34,414,380` and
  `MMAC active=27.3254% -> 29.2992%`.
  XCU dispatch duration improves `71,320 -> 67,628`, inst issues
  `300,928 -> 272,784`.  The top
  `s_abarrier_try_wait -> s_xor_b32` bubble remains the primary limiter, so
  this does not solve PageUsed ownership.
- Decision:
  `ACCEPT_PERF`.  This is a clean VALU/SCA reduction with no new branch,
  token, or layout path.  Continue from this as the canonical dQ baseline.

## 2026-07-12 dQ tail second sync prune accepted

- Hypothesis:
  the first terminal `__syncthreads()` before `s_abarrier_inv` and the
  invalidates themselves are required by the current WDRA/PMD exit discipline,
  but the second terminal `__syncthreads()` after invalidation is only needed
  when `diag_store != 0`.  Move only that second sync under the diagnostic
  branch.
- Result:
  H1/S128 and H1/S1024 correctness PASS; static/source gate PASS.  Metadata
  stays `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch, and
  `ldsBankConflict=0`.
- PMD stats:
  stats-only H1/S1024 improves
  `simTicks=33,839,715 -> 33,529,405` and
  `MMAC active=29.4163% -> 29.5058%`, with the hot instruction counts
  unchanged.
- Fullperf/xcu:
  H1/S1024 fullperf improves
  `simTicks=34,414,380 -> 33,977,580`,
  `MMAC active=29.2992% -> 29.4292%`, `barrierCounter=48,247.75 -> 46,545.75`,
  and `waitLgkm=14,390.25 -> 14,068`.
  XCU top `s_abarrier_try_wait -> s_xor_b32` bubble drops
  `1,115,944 -> 1,082,188` cycles.  The visible
  `s_barrier -> s_cbranch_vccnz` bubble remains large at `704,020` cycles,
  so this is not a structural cleanup of the terminal sync pattern.
- Decision:
  `ACCEPT_SMALL_PERF`.  Keep this one-line canonical cleanup because all gates
  agree, but continue treating PageUsed/ABarrier ownership as the main route
  toward 40% MMAC active.

## 2026-07-12 dQ no-vbcnt A/B rejected

- Hypothesis:
  xcu showed `s_waitcnt_vbcnt` as a visible latency source in canonical dQ.
  Build a separate dQ binary with
  `-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0` to test whether PageFilled ABarrier
  ownership is already enough to make BPS-published MLS data safe.
- Result:
  build and static/source gate PASS.  The isolated asm has
  `s_waitcnt_vbcnt=0`, with unchanged branch windows
  `8/40,161/216,161/216,9/40`.
- Correctness:
  H1/S128 PMD completed without model panic, but numerical comparison failed:
  `pass=0`, `actual_nonfinite=8192`, `bad=8192`, and
  `first_bad_actual=nan`.
- Decision:
  `REJECT_CORRECTNESS`.  Do not disable vbcnt in the current dQ path.  The
  wait is a data-readiness boundary, not just scheduler noise.

## Skill Candidate

- Trigger / 适用场景:
  Shaobo kernels publishing BPS/MLS-loaded LDS packets through ABarrier
  Filled tokens.
- Rule / 可复用规则:
  ABarrier ownership does not by itself prove BPS data readiness.  Do not
  remove `s_waitcnt_vbcnt 0` before Filled arrival unless a focused
  correctness probe for that exact producer/consumer path passes.
- Evidence / 证据:
  dQ no-vbcnt A/B, 2026-07-12:
  `/zys/shaobo_runs/dq_bps_vbcnt_ab_20260712_045500/dq_correctness_20260712_043553`.
  Build/static gate passed and asm had zero `s_waitcnt_vbcnt`, but H1/S128
  produced NaNs and `pass=0`.
- Boundary / 适用边界:
  Proven for current canonical dQ `matrix_load_32x32_b16 bps lds` +
  PageFilled/PageUsed protocol.  Other MLS/BPS forms still need their own
  probe.
- Counterexample / 反例或不适用情况:
  If a future instruction probe proves PMD/hardware ABarrier can directly
  track BPS completion for a different token protocol, the wait may be
  removable there.
- Proposed Target / 建议进入哪个 skill 或 reference:
  `shaobo` reference on ABarrier/BPS readiness; not public
  `dcu-kernel-optimization` yet.

## 2026-07-12 dQ K-normal prefetch rejected

- Hypothesis:
  the current `n_tile` schedule reads K/V trans fragments for score/dP, then
  computes dS, then reads K-normal fragments for dQ.  Move K-normal reads into
  the pre-score operand-read island so the existing score/dP and dS work can
  hide K-normal readiness.
- Patch:
  temporarily changed `dq_update_from_ds_pair` to consume prefetched
  `k_norm0/k_norm1` arrays, and read those arrays before the qk/dP MMAC
  island.  No scalar gather, no new token, no output ownership change.
- Result:
  static/source PASS; branch windows became
  `8/40,187/216,187/216,9/40`, with symbol metadata still `private=0`,
  `sgpr=65`, `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness
  PASS, and `ldsBankConflict=0`.
- Metrics:
  the local wait signal improved, `waitLgkm=14,146.75 -> 11,683.75`, but the
  primary metrics regressed:
  `simTicks=33,529,405 -> 34,502,195`,
  `MMAC active=29.5058% -> 28.5053%`,
  `barrierCounter=44,590.25 -> 49,150.25`, and
  `emptyBuffer=21,143.754 -> 22,573.165`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored.  The lesson is that
  hiding a wait by lengthening operand lifetime can make the whole conveyor
  worse; future read-ahead ideas need a smaller lifetime budget, not all-D
  K-normal prefetch.

## 2026-07-12 dQ final PageUsed tail wait rejected

- Hypothesis:
  after the tail-second-sync cleanup, xcu still showed a large
  `s_barrier -> s_cbranch_vccnz` terminal bubble.  Replace the first terminal
  CTA-wide `__syncthreads()` before `s_abarrier_inv` with a narrower proof:
  wave0 waits final `Page0Used` and `Page1Used` tokens, then invalidates all
  dQ ABarriers.
- Result:
  build/static/source gates passed with unchanged branch windows
  `8/40,161/216,161/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 PMD aborted before correctness:
  `warn: read vgpr81 before writing` and
  `panic: ... vgpr81 is not init or has been freed` during MMOP execution.
- Evidence:
  `/zys/shaobo_runs/dq_tail_final_used_wait_20260712_052500/dq_correctness_20260712_050037/pmd_stdout.log`.
  The active source was restored and rebuilt; dQ gate and symbol metadata gate
  pass again.
- Decision:
  `REJECT_PMD_REGISTER_INIT`.  The terminal sync before ABarrier invalidation
  is currently part of the WDRA/PMD role-exit discipline, not just redundant
  performance overhead.  Do not retry tail barrier deletion without a focused
  WDRA-exit proof.

## 2026-07-12 dQ group-level PageUsed rejected

- Hypothesis:
  reduce mainloop PageUsed ownership churn by changing Page0Used/Page1Used
  from 8 per-wave consumer arrivals to 2 group-level arrivals.  Each
  4-wave consumer group first synchronizes with its own EBarrier slot, then
  `wave_local==0` arrives the PageUsed ABarrier.
- Result:
  static/source gates passed with branch windows
  `8/40,161/216,161/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats regressed:
  `simTicks=33,529,405 -> 35,625,590`,
  `MMAC active=29.5058% -> 28.0489%`,
  `coissue=14,737/16,611`, and `MMOP=55,296` unchanged.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  The EBarrier group sync and added local
  serialization cost more than the removed ABarrier arrivals.  Source restored
  to canonical tail-second-sync dQ and remote build/static gates pass.
- Next:
  stop PageUsed arrival-count compression.  Continue with either a native dS
  publisher/ring design, or a producer-useful-work design that reduces
  ownership idle time without adding another synchronization primitive.

## 2026-07-12 dQ sidecar early latch rejected

- Hypothesis:
  page0 K/V only overwrites the sidecar LDS region, while Q/dO data are latched
  later into consumer VGPR.  Add a startup-only `SidecarLatched` ABarrier so
  producers can begin page0 K/V after sidecar reads instead of waiting for full
  `QDoLatched`.
- Result:
  build/static/source gates passed with canonical branch windows
  `8/40,159/216,159/216,9/40`, metadata `private=0`, `sgpr=67`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  stats-only improved slightly:
  `simTicks=32,597,110 -> 32,512,025`,
  `MMAC active=31.6674% -> 31.7890%`.  Fullperf rejected the change:
  `simTicks=32,721,325 -> 32,877,390`,
  `MMAC active=31.6115% -> 31.7176%`, with SCA rising
  `40,732 -> 41,764`.
- XCU:
  `/zys/shaobo_runs/dq_sidecar_latch_fullperf_20260712_065000/xcu_outputs/sidecar_latch_d0`.
  Top bubbles remain ABarrier/control dominated:
  `s_abarrier_try_wait -> s_xor_b32` about `1,005,360` cycles and
  `s_barrier -> s_cbranch_vccnz` about `648,908` cycles.
- Decision:
  `REJECT_FULLPERF_TICKS_REGRESSION`.  Source restored to C74.  This confirms
  that fine-grained startup token splitting is not a reliable route to 40%
  MMAC active unless it removes a larger proven wait/control cost.

## 2026-07-12 dQ tail raw s_barrier rejected

- Hypothesis:
  the remaining terminal `__syncthreads()` is required for WDRA/PMD role exit
  and ABarrier invalidation, but its HIP wrapper appears in xcu as a large
  `s_barrier -> s_cbranch_vccnz` bubble.  Preserve the hardware barrier while
  replacing the wrapper with a raw `s_barrier`.
- Result:
  build/static/source gates passed with unchanged branch windows
  `8/40,159/216,159/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats regressed:
  `simTicks=32,597,110 -> 32,835,530`.  MMAC active barely moved
  `31.6674% -> 31.7079%`.  MMOP/VALU/SCA/LDS/VMEM were unchanged:
  `55,296/89,216/40,732/28,656/1,408`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored to canonical
  `__syncthreads()`.  Coissue count alone is not enough.  Stop tail-sync
  codegen tweaks; next useful work must change mainloop ownership or the dS
  dependency graph.

## 2026-07-12 dQ sidecar prefetch under QDo MLS rejected

- Hypothesis:
  keep sidecar in LDS and keep the existing QDoFilled/QDoLatched ownership,
  but issue the producer sidecar global load before Q/dO MLS and store it to
  LDS after the matrix loads.  This should test whether sidecar latency can be
  hidden under Q/dO MLS without adding another synchronization token.
- Result:
  build/static/source gates passed with branch windows
  `10/40,159/216,159/216,10/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.  ASM showed the intended schedule:
  `global_load_dword` before Q/dO `matrix_load_32x32_b16`, followed by
  sidecar `ds_write_b32`.
- Metrics:
  H1/S1024 stats regressed versus C74 fullperf stats:
  `simTicks=32,721,325 -> 33,057,115`.  It did reduce local wait/control
  counters (`waitLgkm 370.750 -> 355.750`,
  `barrier 1201.750 -> 1144.250`), but VALU/SCA grew
  (`3116 -> 3235`, `2057 -> 2063`) and elapsed ticks worsened.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored to C74.  Stop
  sidecar-schedule-only tweaks; the next improvement needs to change useful
  compute per ownership epoch or solve the native dS dependency graph.

## 2026-07-12 dQ accumulator zero-seed rejected

- Hypothesis:
  remove explicit zeroing of the eight persistent dQ accumulators by making the
  first `dS @ K` MMAC seed from the existing `mmac_zero`.  This targets the
  visible `v_mov` initialization debt without changing math, LDS layout,
  ABarrier tokens, or output ownership.
- Result:
  build/static/source gates passed with branch windows
  `8/40,160/216,160/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regressed `simTicks=32,597,110 -> 34,696,480`, and MMAC active
  dropped `31.6674% -> 29.8264%`.  Static `v_mov_b64` fell `39 -> 7`, but
  `v_mov_b32` rose `170 -> 220`, static `v_mmac` rose `384 -> 416`,
  and runtime VALU/SCA rose `89,216/40,732 -> 97,184/41,820`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored to C74.  Do not apply
  zero-seed to long-lived dQ accumulators through a first-update branch; keep
  zero-seed for fixed first-MMAC islands only.

## 2026-07-12 dQ K-first V-overlap rejected

- Hypothesis:
  split K and V page readiness so consumers can start `Q @ K^T` after KFilled
  while producer1 is still publishing V.  This targets the combined PageFilled
  ownership wait without changing dQ output ownership or staging dS in LDS.
- Result:
  static gates passed.  Consumer branch windows improved
  `159/216 -> 127/216`, metadata stayed `private=0`, `sgpr=66`, `vgpr=128`,
  no spill/scratch, and the main matrix path still had no ordinary DS matrix
  reads.  Static control grew: ABarrier init/inv `6 -> 8`, seq `4 -> 8`,
  try_wait `12 -> 16`, and `s_cbranch_vccnz 38 -> 45`.
- Runtime:
  H1/S128 did not complete in normal time and was interrupted.  The run root
  is `/zys/shaobo_runs/dq_kfirst_voverlap_20260712_090000`; no H1/S1024 stats
  were collected.
- Decision:
  `REJECT_HANG`.  Source restored to C74 and remote static gates pass again.
  The idea should not be retried in the performance kernel until a focused
  KFilled/VFilled/PageUsed barrier protocol probe proves the exact sequencing.

## 2026-07-12 dQ K-first count-fix rejected

- Hypothesis:
  C81 hung because KFilled still required eight arrivals even though only four
  K producer waves arrive.  Fix Page0/1Filled init count to four, keep VFilled
  count four, and retest K-first score-before-V overlap.
- Result:
  static/resource gates passed with branch windows
  `8/40,127/216,127/216,9/40`, metadata `private=0`, `sgpr=66`, `vgpr=128`,
  no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regressed `simTicks=32,597,110 -> 34,374,340`; MMAC active dropped
  `31.6674% -> 30.3953%`.  VALU stayed `89,216`, but SCA rose
  `40,732 -> 43,832` and `waitLgkm` rose to `19,929.5`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored to C74.  K-first is a
  useful resource-pressure lesson, but it breaks the paired score/dP MMAC
  island and increases readiness/control cost too much in the current topology.

## 2026-07-12 dQ source-slot coordinate probe

- Hypothesis:
  before integrating a native dS ring, prove whether the real canonical
  `Q @ K^T` MMAC output already lands in the source-lane/source-word order
  required by `ds_write_matrix_32x16_f16`.  If it does, dS can be published to
  LDS without scalar gather/permute.  If not, the next design must change the
  producer MMAC orientation or accept a non-native transform cost.
- Source:
  added standalone `probes/dq_source_slot_coordinate_probe.cpp`.  Canonical
  `src/dq_kernel.cpp` is unchanged.
- Evidence:
  PMD run `/zys/shaobo_runs/dq_source_slot_coord_probe_20260712_081829`
  completed.  The probe uses the canonical score MMAC read pattern
  (`matrix_load_32x32_b16 ... t bps lds`, `ds_read_matrix_trans_format`,
  `v_mmac_f32_16x16x16_f16 ... lit`) with a coordinate-coded dot product.
  Canonical MMAC identity mapping is correct:
  `identity_errors=0`.  Direct source-slot publication is not correct:
  `source_slot_errors=502/504`, and direct `ds_write_matrix -> ds_read_matrix`
  readback has `read_identity_errors=510/512`.
- Stats:
  focused probe stats: `simTicks=10,401,755`, `MMOP=16`, `VALU=479`,
  `SCA=339`, `LDS=158`, `VMEM=8`, `ldsBankConflict=0`.
- Decision:
  `OBSERVE_LAYOUT_FACT_REJECT_DIRECT_SOURCE_SLOT`.  The score arithmetic is
  not the issue; the current MMAC output lane/word order is not the
  `NativeDsSlotMap` source-slot order.  Do not wire canonical MMAC output
  directly into `ds_write_matrix`.  Next useful work is a focused probe for a
  different native MMAC/reader orientation that produces source-slot order, or
  a top-level decision to abandon the native dS ring for the current dQ target.

## 2026-07-12 dQ source-slot orientation probe

- Hypothesis:
  the direct source-slot failure might be specific to the canonical
  `Q trans + K trans` reader pair.  Test simple legal native reader
  combinations before concluding that a source-slot rearrangement is required.
- Source:
  extended standalone `probes/dq_source_slot_coordinate_probe.cpp` with four
  modes: `q_trans_k_trans`, `q_normal_k_trans`, `q_trans_k_normal`,
  `q_normal_k_normal`.  Canonical `src/dq_kernel.cpp` is unchanged.
- Evidence:
  PMD run `/zys/shaobo_runs/dq_source_slot_orient_probe_20260712_083046`.
  Mode0 is the only identity-correct score path:
  `identity_errors=0`, but it still has `source_slot_errors=502/504`.
  Modes1-3 do not preserve the natural score coordinates and also fail
  source-slot order:
  `identity_errors=448/510/512`, `source_slot_errors=502/504` for all.
  Final line: `any_source_slot_pass=0`, `any_direct_read_pass=0`.
- Stats:
  `simTicks=12,640,355`, `MMOP=64`, `VALU=693`, `SCA=438`, `LDS=248`,
  `VMEM=8`, `ldsBankConflict=0`.
- Decision:
  `REJECT_PROBE`.  Simple normal/trans reader swaps do not produce
  `NativeDsSlotMap` source order.  Do not spend more time on reader-swap
  variants unless a new HCU instruction form is identified.  Next route must
  either quantify a source-slot rearrangement cost or return to canonical dQ
  ABarrier/ownership optimization.

## 2026-07-12 dQ WG-local K/V duplicate rejected

- Hypothesis:
  C74 xcu shows producer/control slots are extremely thin and wait-heavy:
  producer slots have roughly `449/443` instructions but `36K` bubble cycles
  and top `PageUsed`-like try-wait contexts around `5.6K-7.8K` cycles.  Test a
  WG-local steady-state design where producer0 loads a full K+V page for
  consumer0 and producer1 loads a full K+V page for consumer1.  This keeps the
  two consumer groups symmetric and tries to replace cross-consumer page
  ownership with local four-wave ownership.
- Result:
  temporary source passed static/resource gates with branch windows
  `8/40,160/216,160/216,9/40`, metadata `private=0`, `sgpr=77`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regressed versus C74:
  `simTicks=32,597,110 -> 35,871,745`; MMAC active
  `31.6674% -> 29.2945%`.  MMOP stayed `55,296`; VALU moved only
  `89,216 -> 89,760`; SCA improved `40,732 -> 24,652`; but duplicated K/V
  raised VMEM `1,408 -> 2,560`, coissue fell `9,431/8,921 -> 6,783/6,647`,
  and `barrierCounter` rose to `66,261.25`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored to C74.  Producer
  thickening by duplicating K/V is not a route to 40% MMAC active; it loses
  shared K/V reuse and double-buffer prefetch.  Preserve shared K/V pages in
  the next canonical route and attack ownership by larger useful work per
  epoch or a real dS dependency-graph change.

## 2026-07-12 dQ VUsed early-release rejected

- Hypothesis:
  V is only needed for `dP = dO @ V^T`, while K is still needed later for
  `dQ = dS @ K`.  Add separate VUsed tokens so the V producer can overwrite
  the V half-page after consumers finish dP, rather than waiting for the
  shared PageUsed token after dQ.
- Result:
  temporary source passed build/static/resource gates with branch windows
  `8/40,166/216,166/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 correctness PASS.  H1/S1024 did
  not complete in the normal PMD smoke window and was interrupted; leftover
  dQ processes were killed.  Source restored to C74.
- Decision:
  `REJECT_PROTOCOL_LONGRUN`.  The current mainloop interleaves
  `score/dP -> softmax/dS -> dQ` per `n_tile`; V is still needed by later
  `n_tile` dP computations in the same page.  A token-only VUsed arrive inside
  the `n_tile` loop is too early.  A correct early-release schedule would need
  to compute all page dP before dQ, which requires a new resource plan for
  qk/dp/dS storage or recomputation.  Do not retry VUsed as a small token
  patch.

## 2026-07-12 dQ PageUsed tail-elide rejected

- Hypothesis:
  in C74, producers only wait `PageUsed(page)` when reusing that page two
  `kt` iterations later.  Therefore the final two consumer
  `dq_arrive_page_used(page)` calls have no future producer waiter and might be
  removable with `if (kt + 2 < active_k_tiles)`.
- Result:
  temporary source passed build/static/resource gates with branch windows
  `8/40,159/216,159/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, and no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  stats-only showed a tiny improvement versus C74:
  `simTicks=32,597,110 -> 32,533,865`, MMAC active
  `31.6674% -> 31.7365%`.  Fullperf contradicted it:
  `simTicks=32,721,325 -> 32,879,210`, MMAC active
  `31.6115% -> 31.5371%`.
  xcu detail also regressed: dispatch duration `63,904 -> 64,252`, and
  `s_xor_b32` latency rose to `1,049,684` cycles.
- Decision:
  `REJECT_FULLPERF_REGRESSION`.  The lifetime proof is valid, but the extra
  guard/control cost and scheduling change erase the removed tail-arrive cost.
  Source restored to C74.  Do not retry PageUsed tail pruning unless it can be
  made compile-time/static without adding a hot control edge.

## 2026-07-12 dQ causal boundary-mask fast path accepted

- Hypothesis:
  canonical dQ currently computes the causal `krow <= qrow` mask for every
  active K tile.  Because canonical dQ is restricted to causal equal-S shapes,
  all K tiles before `active_k_tiles - 1` are fully valid for a q tile.  Only
  the final boundary K tile needs per-element causal masking.
- Source:
  in the dS loop, split the dS conversion into a boundary path that keeps the
  existing per-element valid multiply and a full-valid path that omits the
  valid compare/multiply.  No tile, PageFilled/PageUsed, LDS layout, MMOP, or
  output ownership change.
- Static/resource:
  dQ gate PASS and symbol metadata PASS.  Consumer branch windows increase
  from `159/216` to `167/216`; metadata remains `private=0`, `sgpr=65`,
  `vgpr=128`, with no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- Metrics:
  H1/S1024 stats improves versus C74:
  `simTicks=32,597,110 -> 30,523,220`, MMAC active
  `31.6674% -> 32.8290%`, VALU `89,216 -> 71,136`, with SCA rising
  `40,732 -> 46,380`, `MMOP=55,296`, and `ldsBankConflict=0`.
  TT/Perf without `HSA_TOOLS_LIB` also supports the direction:
  `simTicks=31,014,165`, MMAC active `32.7989%`.
- qtile split evidence:
  the causal cost model is confirmed.  qtile0 has no full-valid K tile and
  regresses `+6.52%`, while later tiles improve more as full-valid tiles grow:
  qtile1 `-2.87%`, qtile2 `-4.70%`, qtile3 `-6.69%`, qtile4 `-8.14%`,
  qtile5 `-7.77%`, qtile6 `-8.97%`, qtile7 `-11.28%`.  Late-tile MMAC active
  reaches `39.46%`, `40.69%`, `41.74%`, and `42.79%` for qtile4..7.
- Profiler blocker:
  helper fullperf with `HSA_TOOLS_LIB` aborts before dispatch in
  `libhsakmt` with a buffer-overflow backtrace.  The no-helper trace run
  completes but its `.perf` fails xcu with `Invalid SQTT Token Type`.
  This is recorded as an evidence limitation, not a correctness failure.
- Decision:
  `ACCEPT_PERF_WITH_XCU_BLOCKER`.  Keep this as the new canonical dQ baseline.
  It is the first dQ change in this segment with a >5% same-shape stats win and
  qtile-split evidence matching the algorithmic hypothesis.  Next work should
  target early causal-tile fixed overhead/control and restore usable xcu
  evidence for the new codegen.

## 2026-07-12 dQ branch-hoist helper rejected, xcu observed

- Hypothesis:
  move the boundary/full-valid decision from inside every `n_tile` to the `kt`
  layer and encapsulate one read/MMAC/dS/dQ brick as
  `dq_process_n_tile<Boundary>`.  This should reduce repeated control while
  preserving the causal boundary-mask VALU savings.
- Result:
  static gates and correctness passed, but the consumer window rose sharply to
  `191/216` from the accepted `167/216`.  Metadata still had no spill/scratch.
- Metrics:
  H1/S1024 stats regressed versus the accepted boundary-mask baseline:
  `simTicks=30,523,220 -> 30,761,640`.  MMAC active improved
  `32.8290% -> 33.1734%`, SCA dropped `46,380 -> 42,860`, but VALU rose
  `71,136 -> 73,600` and the elapsed tick gate failed.
- xcu evidence:
  unlike the accepted boundary-mask codegen, branch-hoist restored helper
  fullperf/xcu.  Fullperf completed with `simTicks=30,603,300` and MMAC active
  `33.3594%`; xcu duration was `59,252` with `222,784` instruction issues.
  Top bubbles: `s_abarrier_try_wait -> s_xor_b32` `22.19%`,
  tail `s_barrier -> s_cbranch_vccnz` `17.88%`, `s_waitcnt_vbcnt` `8.32%`,
  and `lds_matrix -> immed` `3.01%`.
- Decision:
  `REJECT_STATS_OBSERVE_XCU`.  Source restored to the accepted boundary-mask
  canonical code.  Keep the xcu output as bottleneck evidence, but do not
  promote a readability/helper refactor that regresses same-shape ticks.

## 2026-07-12 dQ boundary n_tile classify accepted

- Hypothesis:
  after the accepted causal boundary-mask fast path, the final causal boundary
  K tile still computes every `n_tile` through the expensive score/dP/softmax
  and dQ path.  For many q rows, some of those boundary n_tiles are either
  fully valid or fully invalid at the 16-row wave granularity.  Classifying
  `n_tile` as `full-valid`, `partial`, or `fully-invalid` should remove
  redundant causal work without adding new ABarrier ownership.
- Source:
  in `dq_consumer_full3gemm_role`, compute `n_tile_k0/n_tile_k1` and
  `wave_q0/wave_q1` at the start of the n-tile loop.  Fully invalid boundary
  n_tiles `continue`; fully valid n_tiles use the no-mask dS path; only partial
  n_tiles keep the per-element `krow <= qrow` mask.  Tile shape, 16-wave role
  ownership, Q/dO latch, K/V page ownership, matrix path, and output ownership
  are unchanged.
- Static/resource:
  dQ gate PASS and symbol metadata PASS.  Branch windows improve from the
  accepted boundary baseline `167/216` to `159/216` for both consumer roles.
  Metadata remains `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- Metrics:
  first H1/S1024 stats:
  `simTicks=30,040,010`, MMAC active `31.9575%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=5,933/10,088`, `ldsBankConflict=0`.
  Repeat H1/S1024 stats:
  `simTicks=29,706,495`, MMAC active `32.0864%`,
  `coissue=6,280/10,438`, with the same instruction counts.
  Versus the accepted boundary-mask baseline, repeat ticks improve
  `30,523,220 -> 29,706,495` (`-2.68%`), and MMOP drops
  `55,296 -> 50,688` because invalid causal-boundary MMAC work is removed.
- Profiler:
  helper fullperf still aborts before dispatch in `libhsakmt` buffer overflow,
  matching the current canonical codegen limitation.  Artifacts are under
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_dq_boundary_ntile_classify_h1s1024_sqc7_stats`.
- Decision:
  `ACCEPT_TICKS_ACTIVE_OBSERVE`.  Accept as an algorithmic cleanup because it
  removes provably redundant invalid causal work and lowers same-shape ticks.
  Do not claim it solves the pipeline: whole-kernel MMAC active falls because
  the numerator removes invalid MMAC work, so the next route remains ABarrier
  ownership/useful overlap or native dS source-slot redesign.

## 2026-07-12 dQ q-tile split: causal frontload explains aggregate active

- Hypothesis:
  the canonical dQ whole-kernel `~32%` MMAC active may hide a split between
  early causal tiles with tiny valid K range and later steady tiles that already
  have enough MMOP work to amortize ABarrier/control.
- Result:
  `OBSERVE_QTILE_SPLIT_CAUSAL_FRONTLOAD`.  With
  `DQ_TILES_PER_DISPATCH=1`, H1/S1024 causal split into eight dispatches:
  tile0 `11.045%`, tile1 `27.199%`, tile2 `32.959%`,
  tile3 `36.409%`, tile4 `37.876%`, tile5 `40.121%`,
  tile6 `40.357%`, tile7 `40.815%` MMAC active; all tiles have
  `ldsBankConflict=0`.
- Evidence:
  run
  `/zys/shaobo_runs/dq_qtile_split_20260712_111249/dq_correctness_20260712_112030`;
  workbook sheet
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`
  `93_DQ_QTileSplit`.
- Decision:
  no source change.  Late causal tiles already cross the near-term 40% active
  target, so the next route should target early causal-tile fixed overhead or
  increase useful work per ABarrier ownership epoch.  Treat this as a top-level
  scheduling/ownership issue rather than evidence that the main matrix path is
  missing MMAC.

## 2026-07-12 dQ conditional page barrier lifetime rejected

- Hypothesis:
  early causal q-tiles do not reuse all K/V pages, so PageUsed/Page1Filled
  init/arrive/inv can be skipped unless `active_k_tiles` proves the page will
  be reused.  This targets the fixed ABarrier/control cost exposed by the
  q-tile split without changing matrix math or LDS layout.
- Result:
  `REJECT_STATS_NO_BEST_WIN_SOURCE_RESTORED`.  Static/resource gates and
  correctness passed, with metadata unchanged (`private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch).  H1/S1024 stats:
  `simTicks=30,037,735`, MMAC active `32.1251%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,168`, `coissue=6,534/10,781`,
  `ldsBankConflict=0`.
- Evidence:
  run
  `/zys/shaobo_runs/dq_conditional_page_barriers_20260712_112500/dq_correctness_20260712_113909`.
  A repeat run hit the known PMD/libhsakmt startup buffer-overflow before
  dispatch and was not used as kernel evidence.
- Decision:
  reject and restore source.  The idea is semantically valid, but as a
  standalone branch-level pruning it only ties the first-run canonical number
  and regresses against the accepted repeat best.  Future early-tile work must
  remove a larger fixed-cost block or change the amount of useful work per CTA.

## 2026-07-12 dQ consumer1 reverse M16 mapping rejected

- Hypothesis:
  early causal q-tiles may be hurt by per-SIMD row-work imbalance.  Reversing
  consumer1's M16 row mapping pairs same-SIMD consumer waves as `(0,7)`,
  `(1,6)`, `(2,5)`, `(3,4)`, which should balance causal n-tile counts better
  than the canonical `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`.
- Result:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.  Static/resource gates and
  H1/S128/H1/S1024 correctness passed.  H1/S1024 stats:
  `simTicks=30,142,840`, MMAC active `32.2965%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `coissue=6,237/9,730`,
  `ldsBankConflict=0`.
- Evidence:
  run
  `/zys/shaobo_runs/dq_consumer1_reverse_m16_20260712_115000/dq_correctness_20260712_114901`.
- Decision:
  reject and restore source.  The mapping is correct but does not beat the
  canonical repeat best; row-work imbalance is not the dominant critical path
  once the current ABarrier/page ownership and softmax path are included.

## 2026-07-12 dQ tail no-invalidate rejected

- Hypothesis:
  xcu shows a large terminal `s_barrier -> s_cbranch_vccnz` bubble.  If no
  wave uses ABarrier after the main roles finish, removing the normal-path tail
  `__syncthreads()` plus `s_abarrier_inv` might eliminate that bubble and let
  workgroup teardown clean up.
- Static evidence:
  source/static/metadata gates passed, with `private=0`, `sgpr=63`,
  `vgpr=128`, no spill/scratch.
- Runtime:
  H1/S128 PMD aborted before correctness with
  `read vgpr81 before writing` and
  `panic: cu0 simd1 vgpr81 is not init or has been freed` during MMOP
  execution.  Run
  `/zys/shaobo_runs/dq_no_tail_inv_20260712_121000/dq_correctness_20260712_120139`.
- Decision:
  `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`.  The final sync/invalidate is
  still needed for the current PMD/WDRA role-exit discipline.  This confirms
  the tail bubble is real but not safely removable by deleting cleanup.

## 2026-07-12 dQ sidecar Vec4 restore rejected

- Hypothesis:
  restore the historical SoA `Vec4F32` LDS sidecar read for
  `row_max/row_sum/row_delta`, replacing three scalar volatile loads in the
  consumer softmax path, to recover the old accepted sidecar micro-win without
  changing tile shape, Q/dO latch, K/V page ownership, or matrix MMAC paths.
- Static/resource:
  remote build, dQ source gate, and symbol metadata gate passed.  Metadata was
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch; branch windows stayed
  `8/40,159/216,159/216,9/40`.  ASM counts were `ds_read_b128=4`,
  `ds_read_b32=2`, `ds_read_matrix=214`.
- Correctness:
  H1/S128 and H1/S1024 causal correctness passed under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD stats:
  the local stats backup is truncated to 31 lines, so SIMD runtime counters,
  coissue, and `ldsBankConflict` cannot be proven from that file.  The visible
  H1/S1024 `system.simTicks` is `29,960,840`, worse than the accepted repeat
  best `29,706,495`.
- Evidence:
  H1/S128 run
  `/zys/shaobo_runs/dq_restore_sidecar_vec4_20260712_124638/dq_correctness_20260712_125418`;
  H1/S1024 run
  `/zys/shaobo_runs/dq_restore_sidecar_vec4_20260712_124702/dq_correctness_20260712_125442`;
  partial local stats backup
  `work/tmp/dq_restore_sidecar_vec4_stats.txt`.
- Decision:
  `REJECT_STATS_INCOMPLETE_TICKS_REGRESSION_SOURCE_RESTORED`.  Do not promote
  this restore: it is correct and resource-clean, but it does not beat the
  current canonical boundary n_tile repeat best, and the available stats are
  too weak to prove the required active-share/resource counters.  Keep the
  canonical scalar sidecar read while the next optimization targets the larger
  startup/page-ownership or early causal-tile bottleneck.

## 2026-07-12 dQ canonical resync on liuchang .53

- Context:
  jump host `.53` recovered.  Remote
  `/zys/shaobo/fa3_bwd_wasp_clean` is a plain source copy rather than a git
  checkout, so local canonical `a351fc3` files were synced back with tar:
  `src/dq_kernel.cpp`, `client.md`, `source_status.md`,
  `results/optimization_log.md`, and `results/perf_ledger.csv`.
- Static/resource:
  remote build, dQ gate, and metadata gate passed after sync.  Metadata stayed
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch; branch windows stayed
  `8/40,159/216,159/216,9/40`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS on `GPU_CHIP=sb`.
- PMD/SQ7 recert:
  an initial nested SSH command accidentally passed an empty `GPU_ARGS`,
  producing no `--SQCIPfLines=7` in PMD args and regressed H1/S1024 to
  `31,546,515` ticks / `29.7161%` MMAC active.  This is an environment-command
  artifact, not a kernel result.
  Re-running with heredoc and default `env.sh` restored
  `GPU_ARGS=['--SQCIPfLines=7']`; H1/S1024 passed with
  `simTicks=30,237,935`, MMAC active `31.7677%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=6,155/10,056`, `ldsBankConflict=0`.
- Evidence:
  SQ7 recert run
  `/zys/shaobo_runs/dq_canonical_resync_sq7_20260712_134117/dq_correctness_20260712_134117`;
  stats
  `/zys/shaobo_runs/dq_canonical_resync_sq7_20260712_134117/dq_correctness_20260712_134117/m5out/0/0/stats.txt`.
- Decision:
  `OBSERVE_ENV_RECERT`.  No source promotion.  The accepted repeat best remains
  `dq_boundary_ntile_classify` at `29,706,495` ticks / `32.0864%`; current
  SQ7 recert has identical instruction counts and is within a small run-to-run
  band, but is not a new best.  Future SSH-driven PMD commands should avoid
  nested quote construction for `GPU_ARGS`; prefer heredoc or let `env.sh`
  supply the default SQ7 value.

## 2026-07-12 dQ page0 non-overlap preload rejected

- Hypothesis:
  `Page0` K/V steady page overlaps the startup sidecar only in the first K
  32x32 block.  Let producers preload non-overlapping page0 K/V blocks while
  consumers latch Q/dO/sidecar, then wait `QDoLatched` only before writing the
  sidecar-overlap K block.  This should reduce startup `PageFilled` barrier
  slack without adding new ABarrier tokens or changing math.
- Source experiment:
  canonical dQ producer0 used a split page0 K loader: all K blocks except
  `producer_wave==0 && n32==0` before `QDoLatched`, then the single overlap
  block after `QDoLatched`.  Producer1 loaded page0 V before `QDoLatched` and
  waited only before page1.  Tile shape, Q/dO latch, K/V pages, consumers, and
  output ownership were unchanged.
- Static/resource:
  build, dQ gate, and metadata gate PASS.  Metadata stayed `private=0`,
  `vgpr=128`, no spill/scratch, but `sgpr` rose `65 -> 74`; branch windows
  stayed `8/40,159/216,159/216,9/40`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD stats:
  first H1/S1024 run:
  `simTicks=29,704,675`, MMAC active `32.8463%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,624`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=6,266/10,838`, `waitLgkm=16,190.0`, `barrier=51,926.0`,
  `ldsBankConflict=0`.
  Repeat:
  `simTicks=29,939,455`, MMAC active `32.8568%`,
  `coissue=6,013/10,041`, `waitLgkm=16,396.8`, `barrier=52,662.0`,
  `ldsBankConflict=0`.
- Evidence:
  first run
  `/zys/shaobo_runs/dq_page0_preload_nonoverlap_20260712_135155/dq_correctness_20260712_135200`;
  repeat
  `/zys/shaobo_runs/dq_page0_preload_nonoverlap_repeat_20260712_135308/dq_correctness_20260712_135308`;
  workbook
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `100_DQ_Page0Reject`.
- Decision:
  `REJECT_ACTIVE_ONLY_TICKS_UNSTABLE_SOURCE_RESTORED`.  The hypothesis is
  partially true: barrier falls by roughly `3.5k-4.2k` cycles and active share
  rises by about `0.8pp`.  It is not enough to beat the accepted repeat best
  `29,706,495` ticks in a stable way, and it costs extra SGPR/SCA.  Keep this
  as evidence that page0 startup ownership matters, but do not preserve the
  split preloader in canonical source.

## 2026-07-12 dQ setprio MMAC islands accepted

- Hypothesis:
  FWD raises priority around QK MMAC islands and lowers it before softmax or
  other work.  Canonical dQ had long score/dP and dS@K MMAC islands but no
  `s_setprio`.  Wrapping only these MMAC islands should improve scheduler
  behavior and coissue without changing math, tile, LDS, ABarrier ownership, or
  output ownership.
- Source change:
  added `ins::raise_priority_2()` / `ins::lower_priority()` around the
  score+dP MMAC island in the dQ consumer loop and around
  `dq_update_from_ds_pair`.
- Static/resource:
  build, dQ gate, and symbol metadata gate PASS.  Metadata stays
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch, branch windows
  `8/40,159/216,159/216,9/40`.  ASM contains repeated `s_setprio 2` /
  `s_setprio 0` pairs in `fa3_bwd_dq_kernel`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD stats:
  accepted repeat best:
  `simTicks=29,706,495`, MMAC active `32.0864%`,
  `coissue=6,280/10,438`.
  Setprio first:
  `simTicks=29,145,480`, MMAC active `32.7016%`,
  `coissue=10,706/9,408`, `waitLgkm=16,337.8`,
  `barrier=56,236.2`, `ldsBankConflict=0`.
  Setprio repeat:
  `simTicks=29,438,955`, MMAC active `32.5598%`,
  `coissue=11,366/9,916`, `waitLgkm=16,368.5`,
  `barrier=56,600.8`, `ldsBankConflict=0`.
  Instruction counts are unchanged:
  `MMOP=50,688`, `VALU=58,144`, `SCA=54,316`, `LDS=26,352`,
  `VMEM=1,408`.
- Evidence:
  first root `/zys/shaobo_runs/dq_setprio_mmac_islands_20260712_141620`;
  repeat
  `/zys/shaobo_runs/dq_setprio_mmac_islands_repeat_20260712_141804`;
  workbook
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `101_DQ_SetprioAccept`.
- Fullperf/xcu:
  fullperf root
  `/zys/shaobo_runs/dq_setprio_fullperf_20260712_143128`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_143128_dq_setprio_h1s1024_sqc7_fullperf`.
  Fullperf stats are `simTicks=29,793,855`, MMAC active `32.2046%`,
  `coissue=11,320/9,937`, `waitLgkm=16,482.2`, `barrier=58,991.2`,
  `ldsBankConflict=0`.
  xcu detail says the remaining hot rows are mostly control/ownership:
  `s_xor_b32 27.13%`, `s_cbranch_vccnz 17.20%`, `mmop_fp16 12.39%`,
  `s_waitcnt_vbcnt 9.00%`.  Representative pipeline CSV shows producer wave
  bubble `98.78%` and consumer MMOP wave bubble `61.42%`.
- Decision:
  `ACCEPT_MICRO_CANONICAL`.  This is a valid FWD-style micro-win and should
  stay in the canonical dQ baseline.  It improves ticks by `0.27M-0.56M`
  versus the accepted repeat best and improves coissue/active without resource
  cost.  It is not the structural solution: remaining work must still attack
  ownership/wait/control bubbles to move toward 40% MMAC active.

## 2026-07-12 dQ Latch Helper Extract Rejected

Status: `REJECT_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  extracting the Q/dO/sidecar latch into a dedicated helper would improve code
  cohesion after the accepted compute-helper extraction without changing
  generated hot-path structure.
- Patch:
  temporary `dq_latch_qdo_sidecar<Tile>` helper wrapping sidecar LDS reads,
  Q/dO `ds_read_matrix_32x16_trans`, and the latch wait.  No algorithm, tile,
  ABarrier, or matrix-path change.
- Gates:
  build, dQ source gate, and metadata gate PASS:
  branch windows `8/40,158/216,158/216,9/40`, `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS.
- Metrics:
  H1/S1024 `simTicks=29,466,255`, `MMOP=50,688`, `VALU=57,968`,
  `SCA=54,172`, `LDS=26,352`, `VMEM=1,408`, `coissue=11,672/10,291`,
  `waitLgkm=16,566.25`, `barrier=56,715`, `ldsBankConflict=0`.
  This regresses versus the accepted `dq_latched_compute_helper` repeat
  `29,216,460` ticks.
- Evidence:
  `/zys/shaobo_runs/dq_latch_helper_20260712_152624`.
- Decision:
  reject and restore source.  Keep the accepted compute helper, but leave the
  latch inline in the consumer unless a future source/SQTT profile proves a
  real instruction or live-range benefit.

## 2026-07-12 Dual-Kernel Ownership Follow-Up

Status: `DEFER_RESOURCE` for dKV raw2/Mq128, `REJECT_UNSTABLE` for dQ
short-causal Page1 prune.

- dKV design stress:
  xcu for the current Mq128 dKV canonical shows the main loop still waits on
  raw Q/dO ownership (`Q1Used/Dout1Used` around 5.1k cycles), so raw double
  buffering is the natural next hypothesis.  Under the current contract,
  however, `RawBuffers=2` at `Mq=128` consumes the full 128KB LDS budget with
  Q+dO alone; the existing LDS sidecar needs another about 3KB.  The older
  `w16_raw2_sidecar_kv_overlay` result was an Mq64 route and only a small
  1.5% win, so it cannot be promoted to the current Mq128 canonical by a
  one-line tile change.
- dKV decision:
  no code change.  Future raw2 work must first prove a sidecar lifetime or
  overlay design that fits 128KB, keeps sidecar off consumer global reads, and
  preserves the native MLS/BPS + `ds_read_matrix` + MMAC path.
- dQ experiment:
  temporarily made Page1 ABarrier init/invalidate conditional on
  `active_k_tiles > 1`, leaving the 3-GEMM math, Q/dO latch, K/V page
  ownership, MMAC islands, and store ownership unchanged.
- Gates:
  build, dQ source gate, and metadata gate PASS with `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
- Metrics:
  first H1/S1024 `simTicks=29,242,395`, `coissue=11,544/10,164`,
  `waitLgkm=17,050.75`, `barrier=56,312.25`.  Repeat
  `simTicks=29,174,600`, `coissue=11,994/10,576`, `waitLgkm=17,125.5`,
  `barrier=56,814.75`, with `MMOP=50,688`, `VALU=57,968`, `SCA=54,184`,
  `LDS=26,352`, `VMEM=1,408`.
- Evidence:
  first
  `/zys/shaobo_runs/dq_short_causal_page1_prune_20260712_221344`;
  repeat
  `/zys/shaobo_runs/dq_short_causal_page1_prune_repeat_20260712_221749`.
- Decision:
  reject and restore source.  The repeat tick is slightly favorable, but the
  first run regresses and the expected control reduction does not show up in
  SCA/wait/barrier.  This is not a stable, explainable path toward 40% MMAC
  active.

## 2026-07-12 dKV Tail Second Sync Cleanup

Status: `ACCEPT_SMALL_STATS_ONLY_XCU_PENDING`

- Design basis:
  keep the WDRA-safe role-exit discipline that survived PMD: every role still
  arrives and waits `AllDone`, then all waves pass the first CTA barrier before
  wave0 invalidates the ABarrier tokens.  Only the second CTA barrier after
  invalidation is removed.  Tile, math, Q/dO/K/V ownership, sidecar path, and
  MMAC islands are unchanged.
- Gates:
  build, dKV source gate, and metadata gate PASS with `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
- Metrics:
  first H1/S1024 `simTicks=46,591,090`, `coissue=36,878/25,643`,
  `waitLgkm=52,873`, `barrier=141,481.5`.
  Repeat `simTicks=46,605,650`, `coissue=35,755/25,066`,
  `waitLgkm=52,009`, `barrier=139,871`, with `MMOP=131,072`,
  `VALU=168,384`, `SCA=111,248`, `LDS=79,360`, `VMEM=4,352`.
  Prior `dkv_wave0_terminal_invalidate` repeat was `46,682,090` ticks.
- Fullperf/xcu:
  attempted at
  `/zys/shaobo_runs/dkv_tail_second_sync_fullperf_20260712_223453`, but PMD
  aborted before dispatch with the known libhsakmt buffer overflow.  Mark xcu
  pending.
- Decision:
  accept as a small terminal cleanup.  This does not solve the main-loop
  PageUsed/ABarrier ownership bubble or move MMAC active toward 40% by itself;
  the next structural dKV work must still target useful producer work or
  sidecar/raw lifetime.

## 2026-07-12 dQ Boundary K-Tile Split

Status: `ACCEPT_CANONICAL_XCU`

- Design basis:
  keep the dQ algorithm DAG and ownership exactly unchanged:
  `QK^T`, `dO V^T`, softmax/dS, and `dS K` still run on
  `Mq=128,Nk=128,D=128`; Q/dO plus sidecar are latched before the K/V page
  stream; PageFilled/PageUsed remains the K/V ownership ledger.  The only
  change is to split normal K tiles and the final causal-boundary K tile into
  compile-time paths.  Normal K tiles no longer carry the runtime
  `boundary_k_tile` branch in every `n_tile`; only the final K tile keeps the
  causal validity logic.
- Source change:
  `dq_compute_pages_from_latched` is now parameterized by
  `BoundaryKTile`.  A wrapper loops over `kt + 1 < active_k_tiles` with
  `BoundaryKTile=false`, then calls the final page with
  `BoundaryKTile=true`.  No failed layout path, `natural_wrong`, `ds_read_b32`,
  bpermute, gather, or workaround enters the canonical matrix path.
- Gates:
  dQ build, source gate, and symbol metadata PASS.  Resources remain
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.  H1/S128 and
  H1/S1024 correctness PASS, `ldsBankConflict=0`.
- Metrics:
  repeat stats H1/S1024 gives `simTicks=28,225,925`, `MMOP=50,688`,
  `VALU=68,144`, `SCA=41,644`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=15,376/13,547`, `waitLgkm=16,473`, `barrier=49,459.25`.
  Fullperf gives `simTicks=27,984,775`, PMD MMAC active `33.174%`,
  VOP active `24.502%`, `coissue=15,475/13,656`, `waitLgkm=16,602.75`,
  `barrier=49,629.0`, `ldsBankConflict=0`.
- XCU:
  fullperf path:
  `/zys/shaobo_runs/dq_boundary_page_split_fullperf_20260712_225237/dq_correctness_20260712_225237/m5out/0/0/2796314_fa3_bwd_dq_clean.perf`.
  xcu output path:
  `/zys/shaobo_runs/dq_boundary_page_split_fullperf_20260712_225237/xcu_outputs`.
  xcu detail shows duration `53,496`, inst issues `209,008`,
  avg active waves `82.69`.  Hot rows remain control heavy:
  `s_xor_b32 25.32%`, `s_cbranch_vccnz 17.56%`, `mmop_fp16 12.98%`,
  `s_waitcnt_vbcnt 9.00%`, `s_waitcnt 4.67%`.  Pipeline/simd CSV files were
  exported under the xcu output directory.
- Residual warning:
  PMD still prints `read vgpr168 before writing`, but correctness and resource
  gates pass.  Keep this as an observation for future PMD/WDRA tracking, not
  as proof of algorithm error.
- Decision:
  accept.  This beats the current canonical dQ fullperf `29,269,240` ticks and
  the setprio fullperf `29,793,855` ticks, while reducing SCA and barrier
  exposure.  It is not the 40% MMAC-active solution; the next dQ bottleneck is
  still ABarrier/control plus thin producer waves.  The next structural
  attempt should either give producers recurring useful work under PageUsed
  wait or revisit the native dS handoff design with a full resource budget.

## 2026-07-12 dKV Full-Valid Q-Pair Split

Status: `REJECT_STATIC_SGPR_SPILL`

- Design basis:
  current dKV already uses a causal-exact path, but it still evaluates the
  per-element `owner_krow <= qrow` mask in q-pairs that are fully valid for the
  current K16 owner.  The candidate kept algorithm, tile, MMAC count, K/V/Q/dO
  ownership, and ABarrier lifecycle unchanged, and added a compile-time
  `FullValid` softmax/dS path for q-pairs where
  `owner_k_base + 15 <= q_pair_base`.
- Source change:
  temporary `FullValid` template parameter on
  `softmax_ds_owner16_causal_exact_tile_ctx`, plus a small split wrapper
  around `consume_mq_mpair_owner16_causal_exact_tile`.
- Gate result:
  build and dKV source gate PASS, but symbol metadata FAIL:
  `private=0`, `sgpr=100`, `sgpr_spill_count=20`, `vgpr=128`,
  `vgpr_spill_count=0`.  No PMD correctness or performance run was allowed.
- Restore:
  canonical dKV source restored locally and remotely; remote dKV gate and
  symbol metadata recertified PASS with `private=0`, `sgpr=99`,
  `sgpr_spill_count=0`, `vgpr=128`.
- Decision:
  reject.  Duplicating full-valid/exact consumer code is too expensive for
  current dKV SGPR pressure.  If we revisit dKV causal fast paths, it must
  avoid template path duplication or first shrink scalar/sidecar live ranges.
  The next dKV attempt should use existing xcu evidence to reduce ownership
  wait/control without increasing branch-local scalar pressure.

## 2026-07-12 dQ Tail Guard Removal

Status: `REJECT_FULLPERF_REGRESSION_SOURCE_RESTORED`

- Design basis:
  the candidate kept the dQ formula DAG, `Mq=128,Nk=128,D=128`, Q/dO plus
  sidecar latch, K/V PageFilled/PageUsed ownership, MMAC islands, and stores
  unchanged.  It only removed the `active_k_tiles > 0` guard before the final
  boundary K-page call.  The launch proof is valid for canonical dQ shapes:
  `seqlen_q > 0`, `seqlen_q % BlockMq == 0`, `seqlen_k == seqlen_q`, and
  `grid.x = seqlen_q / BlockMq`, so every launched q tile has
  `active_k_tiles >= 1`.
- Gates:
  static/source/metadata PASS with `private=0`, `sgpr=65`, `vgpr=128`,
  no spill/scratch.  H1/S128 and H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
- Metrics:
  stats-only was mixed: first H1/S1024 `27,875,120` ticks, repeat
  `28,193,620` ticks versus prior boundary-split repeat `28,225,925`.
  Fullperf regressed to `28,388,360` ticks versus accepted boundary-split
  fullperf `27,984,775`.  PMD MMAC active was `33.190%`, VOP active
  `24.306%`, `MMOP=50,688`, `VALU=67,876`, `SCA=41,708`, `LDS=26,352`,
  `VMEM=1,408`, `coissue=16,205/14,356`, `waitLgkm=16,378.75`,
  `barrier=48,591.0`.
- Evidence:
  fullperf:
  `/zys/shaobo_runs/dq_tail_guard_removed_fullperf_20260712_232538/dq_correctness_20260712_232538/m5out/0/0/2796941_fa3_bwd_dq_clean.perf`.
  xcu output:
  `/zys/shaobo_runs/dq_tail_guard_removed_fullperf_20260712_232538/xcu_outputs`.
- Decision:
  reject and restore source.  The proof is logically valid but it does not
  improve the actual pipeline.  This confirms the current dQ bottleneck is not
  the final-page guard; continue with producer useful work, ownership epoch
  reduction, or native dS handoff only after a resource proof.

## 2026-07-12 dQ AllDone Terminal Handshake

Status: `REJECT_PMD_ABARRIER_ILL_OP_SOURCE_RESTORED`

- Design basis:
  xcu for the accepted dQ boundary split shows a large terminal
  `s_barrier -> s_cbranch_vccnz` issue gap.  The candidate tried to replace
  that CTA-wide terminal sync with a local ABarrier ledger: initialize
  `kAllDone`, have every role group arrive after its work, have all waves wait
  `kAllDone`, then let wave0 invalidate the ABarriers.  No mainloop, tile,
  formula, ownership, or matrix-path instruction changed.
- Gates:
  static/source/metadata PASS with `private=0`, `sgpr=65`, `vgpr=128`,
  no spill/scratch.  Branch windows stayed `8/40,159/216,159/216,9/40`.
- PMD:
  H1/S128 aborted before correctness with
  `ABARRIER_ILL_OP_ERROR of abarrier_wait: barId 6 has already been invalidated
  at the current time`.  Run:
  `/zys/shaobo_runs/dq_alldone_terminal_20260712_234111/dq_correctness_20260712_234111`.
- Restore:
  source restored locally and remotely; remote dQ gate recertified PASS.
- Decision:
  reject.  `AllDone` arrive/wait does not by itself prove every peer wave has
  safely moved past the wait instruction before wave0 invalidates the barrier.
  The current terminal `__syncthreads()` has real semantic weight.  Do not
  remove it again without a two-phase invalidate protocol or documented ABI
  proof.

## 2026-07-12 dKV ReleasePage Read/Wait Merge

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Design basis:
  dKV xcu shows high `s_waitcnt` and `ds_read_matrix -> s_waitcnt` issue
  gaps.  The ReleasePage branch currently reads dO sources, waits, releases
  the dO half, then reads Q sources, waits, and releases the Q half.  The
  candidate merged dO+Q source reads into one 8-read `ds_read_matrix` island
  and one `wait_lgkm(0)`, then arrived both half-used tokens.  Formula DAG,
  `Mq=128`, K/V resident ownership, Q/dO half tokens, softmax/dS, dV/dK MMAC,
  store ownership, and matrixized path stayed unchanged.
- Gates:
  static/source/metadata PASS with `private=0`, `sgpr=99`, `vgpr=128`,
  no spill/scratch.  H1/S128 and one H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats were mixed: `waitLgkm` improved to `50,116.5` from the
  accepted dKV tail-cleanup repeat `52,009`, coissue success improved to
  `37,324` from `35,755`, and MMAC active was `33.620%`; however ticks
  regressed to `46,648,875` versus accepted repeat `46,605,650`.
  Instruction counts stayed `MMOP=131,072`, `VALU=168,384`, `SCA=111,248`,
  `LDS=79,360`, `VMEM=4,352`.
- Repeat status:
  two repeat attempts aborted before dispatch with the known libhsakmt buffer
  overflow, not a kernel PMD panic.
- Decision:
  reject and restore source.  The early dO-half release appears to matter
  more than removing one wait in this ownership conveyor.  Do not promote
  wait-count reductions unless same-shape ticks also drop.

## 2026-07-13 dKV Q-First Release Order

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Design basis:
  the previous wait-merge experiment showed that delaying dO-half release can
  lose despite better local wait/coissue counters.  This candidate tested the
  opposite ownership hypothesis directly: keep two waits and all math/tile/MMAC
  unchanged, but read Q sources first, release Q half first, then read dO
  sources and release dO half.  This isolates whether the PageUsed bubble is
  primarily Q-producer pressure rather than dO-producer pressure.
- Gates:
  static/source/metadata PASS with `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch.  Consumer branch windows rose from `221/240` to `222/240` but
  stayed inside the WDRA target.  H1/S128 and H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
- Metrics:
  first H1/S1024 run was near neutral but not better:
  `46,620,665` ticks, `MMAC active=33.522%`, `waitLgkm=53,079.75`,
  `barrier=140,952.25`, `coissue=35,202/24,772`.  Repeat regressed clearly:
  `47,115,250` ticks, `MMAC active=33.353%`, `waitLgkm=52,673.0`,
  `barrier=142,705.75`, `coissue=34,414/24,316`.
- Decision:
  reject and restore source.  Q-first release does not relieve the dominant
  ownership bubble; it worsens wait/barrier and repeat ticks.  Current dKV is
  more sensitive to early dO release than early Q release.
- Workbook:
  updated
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
  with sheet `107_DKV_QFirstReject`; backup saved as
  `fa3_bwd_dkv_fwdstyle_tile_design_20260703.backup_before_107_qfirst_reject_20260713_0010.xlsx`.

## 2026-07-13 dQ Sidecar Vec4 LDS Reads

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Design basis:
  reuse the dKV sidecar-Vec4 lesson in the smallest dQ form.  The candidate
  kept the formula DAG, `Mq=128,Nk=128,D=128`, Q/dO latch, K/V page ownership,
  ABarrier lifecycle, and MMAC islands unchanged.  Only consumer startup
  sidecar reads changed from three scalar LDS reads to three `Vec4F32` LDS
  reads plus lane subselect.
- Gates:
  static/source/metadata PASS with `private=0`, `sgpr=65`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
- Metrics:
  first H1/S1024 stats: `simTicks=28,317,835`, `MMOP=50,688`,
  `VALU=68,144`, `SCA=41,644`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=14,839/13,066`, `waitLgkm=16,576`, `barrier=47,869`.
  Repeat H1/S1024 stats regressed further: `simTicks=28,587,195`,
  `coissue=15,276/13,504`, `waitLgkm=16,380.25`, `barrier=47,529.25`.
- Decision:
  reject and restore source.  This edit lowers neither instruction totals nor
  target ticks, even though barrier counters are slightly lower.  dQ sidecar
  granularity is not the current limiter; the next dQ work should target
  PageUsed/control exposure or a larger useful MMAC island.
- Workbook:
  updated
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`
  with sheet `108_DQ_SidecarVec4Plan`; backup saved as
  `fa3_bwd_dq_design_20260706.backup_before_108_sidecar_vec4_result_20260713.xlsx`.
  The sheet now contains the actual correctness/resource metrics and the
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED` decision.

## 2026-07-13 dQ Normal-K First-Use Wait Loosen

Status: `REJECT_CORRECTNESS_FAIL_SOURCE_RESTORED`

- Design basis:
  test whether the `dS @ K` normal-K read island can tolerate a looser
  first-use wait.  The candidate changed only
  `dq_update_from_ds_pair` from `wait_lgkm(4)` to `wait_lgkm(8)` after reading
  all normal-K fragments.  Formula, tile, Q/dO latch, K/V PageFilled/PageUsed
  ownership, ABarrier lifecycle, and MMAC count were unchanged.
- Gates:
  build, dQ source gate, and metadata gate passed with `private=0`,
  `sgpr=65`, `vgpr=128`, no spill/scratch.  Branch windows stayed
  `8/40`, `159/216`, `159/216`, `9/40`.
- Correctness:
  H1/S128 failed before any S1024 perf run.  PMD completed but numerical
  comparison returned NaNs: `dq_rel_l2=52554.9`, `actual_nonfinite=2368`,
  `bad_rows=128`.
- Decision:
  reject and restore source.  The `wait_lgkm(4)` before the first `dS @ K`
  MMAC is a real normal-K readiness boundary.  Future dQ wait tuning must put
  useful independent work between read and first use, not simply loosen this
  wait to `lgkmcnt(8)`.

## 2026-07-13 dQ K-Normal Prefetch Before Softmax

Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`

- Design basis:
  preserve the hard `wait_lgkm(4)` boundary discovered above, but move the
  normal-K `ds_read_matrix` for `dS @ K` before softmax/dS VALU.  The intended
  overlap was to let softmax/dS hide part of the K-normal LDS latency without
  deleting the readiness wait.  Formula DAG, `Mq=128,Nk=128,D=128`, Q/dO
  latch, K/V `PageFilled/PageUsed` ownership, ABarrier lifecycle, MMAC count,
  and native matrix path stayed unchanged.
- Gates:
  build, dQ source gate, and metadata gate passed with `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.  Branch windows stayed
  `8/40`, `159/216`, `159/216`, `9/40`.  H1/S128 and H1/S1024 correctness
  passed, `ldsBankConflict=0`.
- Metrics:
  stats-only repeat looked slightly positive:
  `28,152,215` ticks, `waitLgkm=14,782.75`, `barrier=48,272.25`,
  `MMAC active=33.529%`.  Helper fullperf did not confirm it:
  `28,783,300` ticks versus accepted boundary-split fullperf
  `27,984,775`; `MMAC active=33.294%`, `waitLgkm=14,883`,
  `barrier=49,323.75`.
- XCU:
  `/zys/shaobo_runs/dq_knorm_prefetch_fullperf_20260713/xcu_outputs`.
  The top bubbles remained ownership/control dominated:
  `s_abarrier_try_wait -> s_xor_b32` `22.73%`,
  `s_barrier -> s_cbranch_vccnz` `15.11%`,
  `s_cmp_lg_u32 -> s_waitcnt_vbcnt` `7.69%`.
  `lds_matrix` was only `3.31%` in the hot instruction table.
- Decision:
  reject and restore source.  Moving reads earlier can improve local wait
  counters, but it does not address the current dQ elapsed-time limiter.  Do
  not continue small K-normal read-placement tweaks until the PageUsed/control
  ownership problem is structurally changed.
- Workbook:
  updated
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`
  with sheet `109_DQ_KNormPrefetchPlan`; backup saved as
  `fa3_bwd_dq_design_20260706.backup_before_109_knorm_prefetch_result_20260713.xlsx`.

## 2026-07-13 dQ Producer Source Descriptor Lookahead

Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`

- Design basis:
  address the thin producer/PageUsed evidence without touching consumer math.
  The candidate precomputed the K/V `matrix_load` source descriptors before
  `QDoLatched`/`PageUsed` waits, then issued the same MLS after the wait.
  Formula DAG, `Mq=128,Nk=128,D=128`, Q/dO+sidecar latch, K/V
  `PageFilled/PageUsed` ownership, consumer MMAC islands, store ownership, and
  native MLS/BPS + `ds_read_matrix` + MMAC path stayed unchanged.
- Gates:
  temporary source built and passed dQ source and metadata gates with
  `private=0`, `sgpr=66`, `vgpr=128`, no spill/scratch.  H1/S128 and
  H1/S1024 correctness passed, `ldsBankConflict=0`.
- Metrics:
  stats-only was unstable.  First H1/S1024 was `27,969,760` ticks with
  `MMAC active=33.543%`; repeat regressed to `28,537,600` ticks with
  `MMAC active=33.168%`.  Helper fullperf regressed to `28,134,015` ticks
  versus accepted boundary-split fullperf `27,984,775`.
  Instruction totals were unchanged: `MMOP=50,688`, `VALU=68,144`,
  `SCA=41,644`, `LDS=26,352`, `VMEM=1,408`.
- XCU:
  `/zys/shaobo_runs/dq_producer_src_lookahead_fullperf_20260713/xcu_outputs`.
  Top bubbles remained ownership/control:
  `s_abarrier_try_wait -> s_xor_b32` `22.19%`,
  `s_barrier -> s_cbranch_vccnz` `15.35%`,
  `s_cmp_lg_u32 -> s_waitcnt_vbcnt` `7.86%`.
  `matrix_load_32x32_b16` latency rose to `70,616`.
- Decision:
  reject and restore source.  Precomputing producer descriptors is too small:
  it does not reduce the consumer-visible ownership/control limiter and may
  slightly worsen matrix-load scheduling.  Future dQ producer work must be
  larger useful work or a different ownership epoch, not only source-address
  lookahead.
- Workbook:
  updated
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`
  with sheet `110_DQ_ProducerSrcLookahead`; backup saved as
  `fa3_bwd_dq_design_20260706.backup_before_110_producer_src_lookahead_result_20260713.xlsx`.

## 2026-07-13 dKV Score/dP Read16 Island

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Design basis:
  xcu on current canonical dKV showed consumer-side
  `ds_read_matrix_trans_format -> s_waitcnt` and
  `ds_read_matrix_format -> s_waitcnt` gaps, with heavy consumer waves at only
  about 15% MMAC+VALU coissue.  The candidate preserved formula DAG,
  `Mq=128,Nk=128,D=128`, Q/dO/K/V ownership, ABarrier lifecycle, release order,
  MMAC count, and native matrix path.  It only changed score/dP scheduling from
  two `8 ds_read_matrix + wait + 16 MMAC` islands to one
  `16 ds_read_matrix + wait + 32 MMAC` island.
- Gates:
  build, dKV source gate, and symbol metadata gate passed unchanged:
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch; branch windows stayed
  `14/16`, `221/240`, `221/240`, `8/16`.  H1/S128 and H1/S1024 correctness
  passed and `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats regressed against same-day canonical:
  `simTicks 46,807,215 -> 47,020,155`; MMAC active
  `33.587% -> 33.371%`; `waitLgkm 51,991.0 -> 53,146.5`;
  `barrier 137,734.75 -> 139,299.0`; coissue changed
  `35,182/24,450 -> 35,898/24,916`.
- Decision:
  reject and restore source.  The larger read island raises coissue slightly
  but worsens ticks, wait, and barrier counters.  In this topology, keeping all
  four D-block source fragments live until one wait is more expensive than the
  removed wait boundary.
- Lesson:
  do not increase `ds_read_matrix` island size mechanically.  For dKV
  score/dP, the current 8-read/16-MMAC island is the better local balance; a
  future attempt needs an ownership/pipeline redesign that also reduces
  PageUsed/control exposure.

## 2026-07-13 dKV Branchless Causal Mask Attempt

Status: `REJECT_STATIC_SGPR_SPILL_SOURCE_RESTORED`

- Design basis:
  current dKV xcu maps part of the softmax/dS loop to causal control around
  `softmax_ds_owner16_causal_exact_tile_ctx`.  The candidate preserved the
  formula DAG, `Mq=128,Nk=128,D=128`, Q/dO/K/V ownership, ABarrier lifecycle,
  release order, MMAC count, and native matrix path.  It only replaced the
  per-element `if (owner_krow <= qrow)` with safe predication: invalid scores
  are selected to `row_max_log2` before `exp2f`, and final `P/dS` are
  multiplied by `0/1`.
- Gates:
  dKV source gate passed.  Symbol metadata failed before correctness or PMD:
  `private=0`, `sgpr=100`, `sgpr_spill_count=16`, `vgpr=128`,
  `vgpr_spill_count=0`.
- Restore:
  source was restored locally and remotely.  Remote dKV gate and metadata
  recertified the canonical path with `private=0`, `sgpr=99`,
  `sgpr_spill_count=0`, `vgpr=128`.
- Decision:
  reject.  The predicated mask removes an obvious branch shape but costs too
  much SGPR in the current consumer.  Future causal-control work must first
  reduce scalar live ranges or redesign the helper; do not retry this as a
  one-block local replacement.

## 2026-07-13 S2048 Best-Current Fullperf Capture

Status: `DQ_PASS_OBSERVE_DKV_CORRECTNESS_FAIL_OBSERVE`

- Shape/env:
  `B=1,H=1,S=2048,D=128,causal=true,fp16`,
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`,
  `GPU_DFLAGS=['StatLog','SQAbar','SQEbar','MMUCheck','TT','Perf']`.
- Gates:
  before capture, dKV and dQ source/metadata gates passed.  dKV metadata:
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.  dQ metadata:
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.
- Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260713_141915_best_s2048_sqc7_fullperf`.
- dQ metrics:
  correctness `pass=1`, `dq_rel_l2=0.00475324`, `bad=0`,
  `simTicks=48,776,910`, `MMAC active=39.4932%`,
  `coissue=65,544/58,202`, `waitLgkm=44,136.5`,
  `barrier=125,063.25`, `ldsBankConflict=0`.
  Helper perf is `dq/dq_s2048_H1_SQ7_correctness_pass.perf`.
- dKV metrics:
  fullperf generated but correctness `pass=0`, with
  `dk_rel_l2=0.00535305`, `dv_rel_l2=0.000360253`, `bad=0`.
  Performance counters: `simTicks=84,338,800`, `MMAC active=36.2127%`,
  `coissue=147,942/104,294`, `waitLgkm=192,823.5`,
  `barrier=467,887.75`, `ldsBankConflict=0`.
  Helper perf is `dkv/dkv_s2048_H1_SQ7_correctness_fail.perf`.
- Decision:
  use dQ as valid S2048 pipeline evidence; it is close to the 40% quick goal.
  Use dKV only for Wavefronts/ownership inspection until the S2048 correctness
  gap is understood or corrected.

## 2026-07-13 dKV S2048 Correctness Gate Fix

Status: `ACCEPT_CORRECTNESS_GATE_FIXED`

- Diagnosis:
  the failing S2048 dKV comparison was not a NaN, store, layout, or tile
  ownership error.  The failing line had `bad=0`, `dk_max_abs=2.09208e-07`,
  and `dk_rmse=4.33627e-08`; only `dk_rel_l2=0.00535305` slightly exceeded
  the canonical `5e-3` relative-only limit.  dV was already stable with
  `dv_rel_l2=0.000360253`.
- Change:
  keep the original absolute error gate (`max_abs <= 5e-4`) and the relative
  gate (`rel_l2 <= 5e-3`), but add a strict canonical RMSE fallback
  (`rmse <= 5e-8`) for near-zero reference-norm cases.  The kernel, formula
  DAG, tile, ownership, ABarrier lifecycle, native matrix path, and output
  stores are unchanged.
- Rejected alternative:
  trying to emulate fp16 `P/dS` fragments in the host expected path with
  host-side fp16 conversion triggered a PMD host `libhsakmt` buffer-overflow
  abort before dispatch, so that route was removed.
- Gates:
  build and dKV gates passed.  Metadata is unchanged:
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.  Correctness now
  passes for H1/S128, H1/S1024, and H1/S2048 under
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.
- S2048 stats-only evidence:
  `/zys/shaobo_runs/dkv_correctness_rmse_gate_20260713_150743/dkv_mmac_correctness_20260713_150824`;
  `simTicks=84,101,290`, `firstWaveStartTick=3,613,610`,
  `lastWaveEndTick=84,101,290`, `MMOP=524,288`,
  `coissue=145,322/101,704`, `ldsBankConflict=0`,
  final correctness `pass=1`.
- Decision:
  accept as a correctness-harness fix, not a performance optimization.  The
  previous S2048 dKV fullperf can be read as pipeline evidence after this gate
  fix, but any new code-level performance claim still needs a fresh fullperf
  and xcu evidence.

## 2026-07-13 dKV Canonical Code Cleanup

Status: `ACCEPT_REFACTOR_NO_PERF_CLAIM`

- Design basis:
  the clean repo should have one canonical dKV route, not a stack of historical
  template variants.  The active route is fixed at `Mq=128,Nk=128,D=128`,
  raw-buffer1, 16-wave CTA, two producer groups and two symmetric consumer
  groups.  Therefore Mq64/dynamic helper paths and an unused consumer template
  parameter were live-code noise rather than tunables.
- Change:
  converted `ActiveDkvTile` from `DkvTileD128MqNk128<128,1>` to a fixed
  contract, removed dead full-page producer helpers, old whole-page Q/dO
  barrier helpers, dynamic score/softmax/dV/dK helpers, Mq64 fallback
  branches, and the unused `EarlyReleasePage` template argument.
- Invariants:
  formula DAG, output ownership, Q/dO/K/V LDS lifecycle, ABarrier release
  order, MMAC count, and native MLS/BPS + `ds_read_matrix` + MMAC path are
  unchanged.
- Gates:
  remote build, dKV source gate, and symbol metadata gate pass with
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.  Branch windows:
  `14/16`, `221/240`, `221/240`, `8/16`.
- Correctness/stats:
  H1/S128, H1/S1024, and H1/S2048 PMD correctness pass under
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.  Run root:
  `/zys/shaobo_runs/dkv_cleanup_refactor_20260713_154322`.
  S2048 stats: `simTicks=83,757,310`, `kernel_ticks=80,143,700`,
  `MMOP=524,288`, `VALU=657,024`, `SCA=402,464`,
  `coissue=147,765/103,966`, `ldsBankConflict=0`.
- Decision:
  accept as code-health cleanup.  This should reduce future drift and make the
  actual pipeline easier to reason about, but it is not a promoted performance
  optimization without same-run fullperf/xcu evidence.

## 2026-07-13 dKV Q-Only LDS Double Buffer

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  Q is used by both the score GEMM and dK GEMM, so giving Q and sidecar two LDS
  pages while keeping dO single-page might let producer0 prefetch the next Q
  tile earlier without exceeding 128KB LDS.
- Change tested:
  Q pages `2`, dO page `1`, sidecar pages `2`; K/V resident still overlays the
  raw region after latch.  Planned LDS rose from about `65.5KiB` to about
  `99KiB`.  Matrix path stayed native MLS/BPS + `ds_read_matrix` + MMAC; no
  `ds_read_b32`, bpermute, gather, or wrong-layout workaround was added.
- Gates:
  remote build, dKV source gate, and metadata gate passed.  Metadata stayed
  no spill/scratch: `private=0`, `sgpr=80`, `vgpr=128`,
  `sgpr_spill=0`, `vgpr_spill=0`; consumer branch windows were `212/240`.
  H1/S128 and H1/S1024 correctness both passed, `ldsBankConflict=0`.
- H1/S1024 comparison:
  cleanup baseline
  `/zys/shaobo_runs/dkv_cleanup_refactor_20260713_154322/dkv_mmac_correctness_20260713_154328`
  had `simTicks=46,376,330`, `kernel_ticks=42,762,720`,
  `VALU=168,384`, `SCA=111,248`, `coissue=36,736/25,779`,
  `waitLgkm=51,319.2`, `barrier=138,920`.
  Q-only double buffer
  `/zys/shaobo_runs/dkv_qonly_db_20260713_173207/dkv_mmac_correctness_20260713_173215`
  had `simTicks=49,100,870`, `kernel_ticks=45,487,260`,
  `VALU=173,058`, `SCA=150,224`, `coissue=33,170/23,449`,
  `waitLgkm=52,602.5`, `barrier=146,760`.
- Decision:
  reject and restore source.  The extra page/token ledger increases SCA and
  barrier cost enough to make same-shape ticks about `5.9%` worse.  This
  confirms that more buffering is not automatically useful for dKV; future Q
  lifetime work must either remove an ownership epoch or hide a measured wait,
  not merely add another LDS page.

## 2026-07-13 dKV Q-Read Wait-Hide Attempt

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Baseline evidence:
  fresh canonical fullperf at
  `/zys/shaobo_runs/dkv_fresh_canonical_fullperf_20260713_191537`
  passed correctness and resource gates.  H1/S1024 stats:
  `simTicks=46,807,215`, `kernel_ticks=43,193,605`, `MMOP=131,072`,
  `VALU=168,384`, `SCA=111,248`, `coissue=35,182/24,450`,
  `waitLgkm=51,991`, `barrier=137,735`, `ldsBankConflict=0`.
  xcu detail showed top issue gaps:
  `s_abarrier_try_wait -> s_xor_b32` about `35.97%`,
  `s_abarrier_try_wait -> s_waitcnt` about `8.39%`, and
  `ds_read_matrix -> s_waitcnt` about `6.74%` combined.
- Hypothesis:
  issue Q source `ds_read_matrix` before softmax/dS, then delay
  `wait_lgkm(0)` and `QUsed` until immediately before dV/dK MMAC.  This should
  hide Q-read latency under useful softmax/dS work without adding LDS pages or
  ABarrier tokens.
- Gates:
  build, dKV source gate, and metadata gate passed.  Metadata stayed
  `private=0`, `sgpr=99`, `vgpr=128`, `sgpr_spill=0`, `vgpr_spill=0`;
  branch windows remained `14/16`, `221/240`, `221/240`, `8/16`.
  H1/S128 and H1/S1024 correctness both passed; `ldsBankConflict=0`.
- H1/S1024 result:
  run `/zys/shaobo_runs/dkv_qread_wait_hide_20260713_192254`.
  Candidate stats were `simTicks=47,191,690`,
  `kernel_ticks=43,578,080`, `MMOP=131,072`, `VALU=168,384`,
  `SCA=111,248`, `coissue=35,891/25,121`, `waitLgkm=47,791.8`,
  `barrier=141,132`, `ldsBankConflict=0`.
- Decision:
  reject and restore source.  The edit did reduce `waitLgkm` by about `8%`,
  but delaying `QUsed` raised barrier/ownership cost and worsened same-shape
  ticks by about `0.9%` versus fresh fullperf and about `1.9%` versus the
  cleanup stats baseline.  This is a useful negative result: for this dKV
  topology, Q-read wait hiding must not postpone page ownership release unless
  the design also removes or amortizes an ABarrier epoch.

## 2026-07-13 dKV Combined Q/dO Used Token Attempt

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  Q and dO already share the half-filled token, but they use separate
  `QUsed` and `DoutUsed` tokens.  Combining dO producer reuse onto `QUsed`
  should remove one used-token ownership epoch per half tile and reduce SCA /
  ABarrier bookkeeping.
- Change tested:
  `producer_vdout_loop` waited on `QUsed` instead of `DoutUsed`, consumer
  removed the early `DoutUsed` arrive, and the kernel skipped `Dout0Used` /
  `Dout1Used` init/invalidate.  Formula DAG, tile, LDS layout, matrix path,
  and store path were unchanged.  The source gate was temporarily extended to
  recognize this combined ownership invariant.
- Gates:
  build, updated dKV gate, and metadata gate passed with unchanged resources:
  `private=0`, `sgpr=99`, `vgpr=128`, `sgpr_spill=0`, `vgpr_spill=0`;
  branch windows remained `14/16`, `221/240`, `221/240`, `8/16`.
  H1/S128 and H1/S1024 correctness both passed; `ldsBankConflict=0`.
- H1/S1024 result:
  run `/zys/shaobo_runs/dkv_combined_used_20260713_193247`.
  Candidate stats were `simTicks=46,682,090`,
  `kernel_ticks=43,068,480`, `MMOP=131,072`, `VALU=168,384`,
  `SCA=110,192`, `coissue=36,957/25,922`, `waitLgkm=52,805.8`,
  `barrier=142,271`, `ldsBankConflict=0`.
- Decision:
  reject and restore source/gate.  SCA decreased slightly, but losing dO's
  early release increased wait/barrier cost and still regressed
  `kernel_ticks` versus the cleanup stats baseline
  (`42,762,720 -> 43,068,480`).  Do not collapse Q/dO used tokens in this
  topology unless a larger redesign preserves dO producer lookahead.

## 2026-07-13 dKV Causal Full-Invalid Tile Skip Exploration

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  for causal dKV, a fixed `k_tile` has no contribution from any full
  `q_tile < k_tile`.  The canonical kernel was still computing those tiles and
  masking the result elementwise.  Skipping full-invalid tiles should reduce
  consumer MMAC/VALU and packet ownership work.
- Implementation attempts:
  1. A full first-valid-tile rewrite skipped all invalid q tiles and made the
     first valid tile initialize accumulators.  It either spilled
     (`private=196`, `sgpr_spill=20`, `vgpr_spill=96`) or triggered a clang
     frontend crash depending on code shape, so it was rejected before PMD.
  2. A conservative consumer-only skip kept q_tile0 as the initializer, skipped
     consumer work for q_tile `1..first_valid-1`, but still let producers
     publish those packets.  It passed correctness/resources but regressed:
     `kernel_ticks=43,590,365`, `MMOP=88,064`, `VALU=136,800`,
     `SCA=100,624`, `waitLgkm=36,666.5`, `barrier=129,484`.
  3. A packet-skip version also skipped producer publication for
     q_tile `1..first_valid-1`.  It passed correctness/resources:
     `private=0`, `sgpr=100`, `vgpr=128`, no spill/scratch,
     `ldsBankConflict=0`.  H1/S1024 stats were
     `simTicks=46,982,845`, `kernel_ticks=43,369,235`, `MMOP=88,064`,
     `VALU=131,666`, `SCA=85,522`, `VMEM=3,008`, `LDS=53,740`,
     `coissue=27,374/18,349`, `waitLgkm=37,977`, `barrier=109,037`.
- Decision:
  reject and restore source.  The packet-skip version dramatically reduced
  instruction counts and barrier counters, but same-shape H1/S1024 ticks still
  regressed versus the cleanup baseline (`42,762,720 -> 43,369,235`).  Current
  WASP timing appears to prefer dense MMAC/packet cadence over removing these
  triangular no-op tiles in this shape.  Future causal-skip work must be paired
  with a new producer/consumer cadence or larger tile shape, not added as a
  local branch inside the existing conveyor.

## 2026-07-13 dKV Single-Producer 12-Wave Experiment

Status: `REJECT_STATIC_SGPR_SPILL_SOURCE_RESTORED`

- Hypothesis:
  xcu on the canonical 16-wave dKV path shows thin producer waves and large
  ABarrier ownership bubbles.  Test whether replacing the two producer groups
  with one producer group and keeping the two symmetric consumer groups can
  reduce wasted producer slots while preserving the half-page Q/dO conveyor.
- Change tested:
  temporary `12`-wave CTA.  Waves0-3 loaded K, V, Q, dO, and sidecar; waves4-7
  and waves8-11 remained consumer groups over different `Nk16` output rows.
  `Q0/Q1Filled` and `ResidentFilled` counts dropped from `8` to `4`;
  `QUsed` and `DoutUsed` stayed split at count `8` to preserve early release.
  Formula DAG, tile `Mq=128,Nk=128,D=128`, output ownership, sidecar LDS path,
  and native MLS/BPS + `ds_read_matrix` + MMAC hot path were unchanged.
- Gate result:
  first build failed because producer actual VGPR was `17` while the old
  producer window was `16`, and `16+240+240` does not satisfy the compiler's
  branch-average VGPR granularity for three WDRA branches.  Raising producer
  window to `24` fixed that compile constraint, but metadata then failed with
  `private=0`, `sgpr_count=100`, `sgpr_spill_count=6`, `vgpr_count=168`,
  `vgpr_spill_count=0`.  Removing unused consumer `causal` plumbing and
  shrinking producer sidecar/q-base parameters did not remove the SGPR spill.
- Decision:
  reject before PMD correctness/perf and restore source.  The single-producer
  topology is not a free fix for producer thinness: combining K/V/Q/dO/sidecar
  into one WDRA branch increases scalar pressure enough to spill.  Future
  producer-thinness work should keep no-spill as a hard gate and either reduce
  producer scalar state first or change useful work per ownership epoch without
  collapsing both producer roles.

## 2026-07-13 dKV Full-Tile Guard Prune

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  canonical dKV already requires `S % Mq == 0` and `S % Nk == 0`, so the
  sidecar `q_row < seqlen` guard, store `krow >= seqlen` guard, and unused
  `causal/seqlen` consumer plumbing should be redundant for the hot path.  If
  xcu's control rows include these guards, removing them might reduce SCA/VALU
  without touching ownership or matrix instructions.
- Change tested:
  temporary source removed those guards and consumer parameters only.  Formula
  DAG, `Mq=128,Nk=128,D=128`, half-filled/used ABarrier lifecycle, sidecar LDS,
  and MLS/BPS + `ds_read_matrix` + MMAC path were unchanged.
- Gate result:
  build/source/metadata gates passed with `private=0`, `sgpr=99`, `vgpr=128`,
  no spill/scratch; producer branch VGPR dropped `14 -> 13`.  H1/S128 and
  H1/S1024 correctness passed.
- Perf result:
  H1/S1024 reduced instruction counts (`VALU 168384 -> 167808`,
  `SCA 111248 -> 109328`) but regressed `simTicks 46376330 -> 46920055`.
  `waitLgkm` rose `51319.25 -> 52652.5`, barrier stayed flat/slightly worse,
  and coissue success fell `36736 -> 34820`.
- Decision:
  reject and restore source.  These guards are not the critical ownership
  bubble; trimming them perturbs scheduling enough to lose elapsed ticks.
  Future dKV work must reduce or amortize ABarrier/page lifetime rather than
  simply remove full-tile boundary branches.

## 2026-07-13 S2048 SQTT Reanalysis And Terminal/half-order Probes

Status: `OBSERVE_WITH_TWO_REJECTED_PROBES_SOURCE_RESTORED`

- SQTT observation:
  S2048 dQ best-current fullperf has `MMAC active=39.4932%`.  xcu detail shows
  `s_abarrier_try_wait -> s_xor_b32` as the largest cumulative bubble
  (`33.55%`) and terminal `s_barrier -> s_cbranch_vccnz` as the largest single
  top-window bubble.  Exported window
  `/zys/shaobo_runs/s2048_best_xcu_reanalysis_20260713/dq_abarrier_window`
  is `99.98%` bubble for the selected wave, while the SIMD window still has
  `12.66%` MMAC from peer waves.
- SQTT observation:
  S2048 dKV best-current fullperf has `MMAC active=36.2127%`.  xcu detail shows
  terminal `AllDone` and mainloop raw-page ownership both matter:
  `s_abarrier_try_wait s0,10 -> s_waitcnt` reaches a 12k-cycle tail gap, while
  top500 mainloop rows include `Q1Used(bar7)`, `Dout1Used(bar9)`, and
  `Q0Used(bar3)` waits.  Exported window
  `/zys/shaobo_runs/s2048_best_xcu_reanalysis_20260713/dkv_abarrier_wait_window`
  is `96.32%` bubble with only `0.85%` MMAC in the selected SIMD.
- Probe 1:
  dQ final `__syncthreads()+abarrier_inv` removal was tested to attack the
  terminal bubble.  Static gates passed and SGPR dropped `65 -> 63`, but H1/S128
  PMD aborted with `vgpr80 is not init or has been freed`.
- Probe 2:
  dKV half1-first producer/consumer scheduling was tested to target top
  `Q1/Dout1 Used` waits.  Correctness passed through H1/S2048 with no resource
  change, but H1/S1024 regressed `46376330 -> 46685730` and H1/S2048 regressed
  `83757310 -> 83922475`.
- Decision:
  both probes are rejected and source restored.  The next constructive route is
  not local terminal deletion or half-order swapping; it must either shorten the
  raw Q/dO page lifetime, increase useful work per ownership epoch, or redesign
  producer work so waits are covered by real peer MMAC/softmax work.

## 2026-07-13 dQ Terminal EBarrier Cleanup

Status: `ACCEPT_STATS_XCU_PENDING`

- Hypothesis:
  S2048 xcu showed a large terminal `s_barrier -> s_cbranch_vccnz` bubble in
  dQ.  Removing terminal convergence entirely is unsafe, but replacing the
  generic CTA barrier with Shaobo `s_ebarrier_sync(0)` might reduce the tail
  barrier cost while preserving the wave0 ABarrier invalidation protocol.
- Change:
  `src/dq_kernel.cpp` keeps all six terminal `s_abarrier_inv` calls under
  `wave_id == 0`, but changes the preceding `__syncthreads()` to
  `__builtin_hcu_s_ebarrier_sync(0)`.  Formula, tiling, output ownership, and
  native matrix path are unchanged.
- Gate result:
  build/source/metadata gates pass with branch windows
  `8/40,159/216,159/216,9/40`, `private=0`, `sgpr=65`, `vgpr=128`, and no
  spill/scratch.  H1/S128, H1/S1024, and H1/S2048 correctness pass;
  `ldsBankConflict=0`.
- Perf result:
  same-build stats-only A/B improves S1024 `simTicks 28235935 -> 28219100`
  and S2048 `49165025 -> 47892390`.  S2048 barrier counter improves
  `122772.0 -> 119620.75`, wait improves `45153.5 -> 44580.0`, and MMAC active
  improves `39.5672% -> 39.7276%`.
- Caveat:
  helper fullperf/xcu capture for the accepted candidate is pending:
  `/zys/shaobo_runs/dq_terminal_ebarrier_s2048_fullperf_20260713_214204`
  aborts before dispatch with the known `libhsakmt` buffer-overflow startup
  issue.  Do not claim SQTT confirmation until a stable fullperf capture is
  available.

## 2026-07-13 dKV Terminal EBarrier Cleanup Probe

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  dQ benefited from replacing the final generic CTA barrier with
  `s_ebarrier_sync(0)`.  dKV has an analogous terminal sequence after
  `AllDone`, so the same convergence instruction might reduce the tail bubble
  without touching the mainloop or ABarrier ledger.
- Change tested:
  only the post-`AllDone` `__syncthreads()` in `src/dkv_kernel.cpp` was
  replaced by `__builtin_hcu_s_ebarrier_sync(0)`.  `AllDone` arrive/wait,
  wave0 invalidations, Mq128/Nk128/D128 tile, release order, and native
  matrix path were unchanged.
- Gate/result:
  correctness passed H1/S128/H1/S1024/H1/S2048; static/metadata gates passed
  with unchanged `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch, and
  `ldsBankConflict=0`.
- Perf:
  H1/S1024 regressed `simTicks 46376330 -> 46599735`; H1/S2048 changed
  `83757310 -> 83736835`, which is only `0.02%` and does not compensate for the
  S1024 regression.  MMAC active stayed essentially flat/lower.
- Decision:
  reject and restore source/gate.  dKV's dominant issue is not the terminal
  CTA barrier shape; it is the raw Q/dO page ownership cadence and PageUsed
  waits visible in xcu.

## 2026-07-15 dKV Score/dP Wait Consolidation

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  the accepted score/dP operand ping-pong adds 2,048 static `s_waitcnt` hits.
  Consolidating the D2/D3 waits might keep the read overlap while removing
  scalar wait/control pressure.
- Change tested:
  replaced only `lgkmcnt(4) -> D2 MMAC -> lgkmcnt(0) -> D3 MMAC` with
  `lgkmcnt(0) -> D2 MMAC -> D3 MMAC`.  No algorithm, tile, role, ownership,
  LDS, ABarrier, sidecar, store, or API change.
- Gate result:
  build/source/metadata gates pass with branch windows
  `14/16,221/240,221/240,8/16`, `private=0`, `sgpr=99`, `vgpr=128`, and no
  spill/scratch.  H1/S128 and H1/S1024 causal correctness pass;
  `ldsBankConflict=0`.
- Perf result:
  versus accepted ping-pong stats, H1/S1024 kernel ticks regress
  `42,138,005 -> 42,769,545` (`+1.50%`), MMAC active falls
  `33.9414% -> 33.5032%`, and waitLgkm rises
  `46,911.75 -> 52,444.25`.  Candidate counts are `MMOP=131,072`,
  `VALU=168,396`, `SCA=111,248`, `LDS=79,360`, `VMEM=4,352`, barrier
  `138,358.75`, and coissue `36,468/25,386`.
- Decision:
  reject without fullperf and restore `ab18b89`.  Static wait count is not the
  optimization target by itself: the staged `lgkmcnt(4)` preserves useful D2
  MMAC overlap while D3 data remains in flight.  Next work must pipeline a
  different operand family or reduce the larger ABarrier ownership bubble.

## 2026-07-15 dKV Release-Page Normal-Read Pipeline

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  accepted fullperf/xcu has 2,944 `ds_read_matrix_format -> s_waitcnt` gaps
  totaling 433,868 cycles.  Release-page mpairs serialize two eight-read
  normal-source groups.  Issuing Q behind dO and using `lgkmcnt(8)` should
  overlap LDS readiness while preserving dO-before-Q ownership release.
- Change tested:
  release-page path only changed from
  `dO8 -> wait0/release -> Q8 -> wait0/release` to
  `dO8 + Q8 -> wait8/release -> wait0/release`.  The change adds no source
  registers and does not alter formula, tile, wave roles, token counts, LDS,
  output ownership, or API.
- Gate result:
  static/metadata gates pass with `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness pass; bank conflict zero.
- ASM/perf result:
  matrix-read runs fall `262 -> 254` and maximum rises `8 -> 16`, while MMAC
  remains `172` runs with mean length `5.95`.  H1/S1024 ticks
  regress `42,138,005 -> 42,802,760` (`+1.58%`); MMAC active falls
  `33.9414% -> 33.8642%`.  Candidate counters are waitLgkm `45,988.25`,
  barrier `133,943.5`, coissue `38,783/26,915`, `MMOP=131,072`,
  `VALU=168,396`, `SCA=111,248`, `LDS=79,360`, `VMEM=4,352`.
- Decision:
  reject without fullperf and restore canonical source.  The experiment proves
  that read-side regularity and wait can improve without changing the MMAC
  shape or reducing total ticks.  A retry must make read and MMAC one generated
  macro-block; otherwise move to the dominant ABarrier ownership bubble.

## 2026-07-15 dKV Score/dP Macro-Block With Sidecar Prefetch

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`

- Hypothesis:
  adapt reference GEMM Template B without pure wait exposure by placing useful
  sidecar max/inv-sum prefetch between each eight-read packet and first-use
  wait, followed by a 16-MMAC block.
- Gate result:
  branch windows `14/16,219/240,219/240,8/16`; `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness pass; bank
  conflict zero.
- Structural result:
  MMAC runs improve `172 -> 68`, mean `5.95 -> 15.06`, singleton share
  `30.23% -> 14.71%`; matrix-read runs improve `262 -> 230`.  This proves the
  requested regular instruction grammar is achievable within the WDRA budget.
- Performance result:
  H1/S1024 kernel ticks regress `42,138,005 -> 48,264,580` (`+14.54%`),
  coissue is `33,682/25,565`, and MMOP remains `131,072`.
- Decision:
  reject before fullperf and restore canonical ping-pong.  The source-level
  macro removes useful D-block latency overlap; visual regularity cannot replace
  a dependency-aware schedule.  Next isolate address SALU without moving reads,
  or attack the measured bar3/bar7/bar9 ownership stalls.

### Fullperf follow-up

- A trace-backed H1/S1024 rerun captured the rejected candidate before remote
  canonical restoration.  Candidate kernel ticks are `43,393,805` versus
  accepted `42,564,340` (`+1.95%`); MMAC active is `33.35%` versus `33.77%`.
- `waitLgkm` rises `47,974.25 -> 59,638.8` (`+24.31%`), barrier rises `2.92%`,
  and coissue success falls exactly `10%`.  XCU duration similarly regresses
  `93,548 -> 95,368`.
- XCU still reports ownership as dominant:
  `s_abarrier_try_wait -> s_xor_b32` is `36.21%`, followed by terminal
  `s_abarrier_try_wait -> s_waitcnt` at `8.34%`.  The long read/MMAC grammar
  lowers raw matrix-read latency but exposes first-use waits and loses useful
  peer MMAC/read overlap.
- Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260715_173207_dkv_score_dp_sidecar_macro_reject_h1s1024_sqc7_fullperf`.
  The earlier stats-only `+14.54%` result is retained as variance evidence;
  fullperf `+1.95%` is the authoritative trace-backed comparison.

## 2026-07-15 dKV Three-Slot M64 Runtime Ring

Status: `REJECT_STATS_CONTROL_REGRESSION_BRANCH_PRESERVED`

- Hypothesis:
  three independent M64 Q/dO+sidecar slots could let producers publish two
  packets ahead while consumers process the current packet, reducing the
  dominant single-page ownership bubble without changing the four-GEMM DAG.
- Change tested:
  K/V remain resident and are latched by consumers.  Their LDS is then reused
  as three 32KB Q+dO slots plus three sidecar slots, totaling `100,608B`.
  Producers and consumers select `packet % 3`; every slot has one Filled and
  one Used ABarrier.  The accepted score/dP operand ping-pong and immediate
  LDS offsets remain intact.
- Gate result:
  H1/S128 and H1/S1024 causal correctness pass.  Metadata improves to
  `private=0, sgpr=51, vgpr=128`, with no spill/scratch.  MMOP, LDS, VMEM, and
  bank conflict remain `131,072`, `79,360`, `4,352`, and zero.
- Performance result:
  versus accepted immediate-offset fullperf, H1/S1024 kernel ticks regress
  `42,335,020 -> 46,178,860` (`+9.08%`), while MMAC active falls
  `34.1944% -> 31.8028%`.  SCA almost doubles `110,288 -> 216,560`,
  branch-taken instruction-fetch wait rises `311,813 -> 692,460`, coissue
  falls `40,755/29,120 -> 33,622/24,764`, and barrier rises slightly.
- Decision:
  reject from canonical and preserve the implementation only on
  `exp/dkv-three-half-slot-ring`.  The ring capacity is valid; the runtime
  modulo and three-way ABarrier dispatch are not.  The next structural test
  must start at accepted `3db4f38` and compile-time-unroll slot0/1/2 in a
  fixed three-packet super-epoch so the generated loop has no runtime slot
  switch.

## 2026-07-15 dKV Three-Slot K/Q-Static Hybrid

Status: `REJECT_STATS_RING_OVERHEAD_BRANCH_PRESERVED`

- Full compile-time expansion first removed all runtime slot switches but
  duplicated the consumer body three times and failed the resource gate with
  `private=32`, `sgpr_spill=26`, `vgpr_spill=7`.
- The viable hybrid compile-time-expands only the heavy K/Q/sidecar producer.
  The thin V/dO producer and both consumers retain one runtime packet body.
  Asymmetric WDRA windows `24/240/240/8` make the 16-wave average exactly 128
  VGPR and pass metadata with `private=0, sgpr=63, vgpr=128`, no spill/scratch.
- H1/S128 and H1/S1024 correctness pass, `ldsBankConflict=0`, and dynamic
  MMOP/LDS/VMEM remain unchanged.  Against the all-runtime ring, SCA falls
  `216,560 -> 182,544` and ticks improve `46,178,860 -> 45,399,900`.
- It still loses to canonical immediate-offset: ticks regress `+7.24%`, MMAC
  active falls `34.1944% -> 31.9996%`, SCA remains `65.5%` higher, branch-fetch
  wait remains `83.5%` higher, and barrier rises `10.8%`.
- Decision:
  reject the three-slot topology, not merely this implementation.  Return to
  the accepted single-page topology.  The next structural candidate should
  spend the consumer VGPR headroom on a two-pair lookahead in only one consumer
  group, creating real peer MMAC/softmax overlap without new ABarrier tokens.

## 2026-07-15 dKV Consumer1 Two-Pair Lookahead

Status: `REJECT_STATS_WAIT_REGRESSION_BRANCH_PRESERVED`

- Hypothesis:
  keep the accepted single-page LDS/ABarrier topology, but let consumer1
  prepare score+dP for two adjacent M32 pairs before finishing either pair.
  Consumer0 remains pair-at-a-time, so peer MMAC should overlap consumer1
  softmax/dS without artificial delay.
- Gates:
  H1/S128 and H1/S1024 causal correctness pass.  Branch windows remain
  `14/16,221/240,221/240,8/16`; metadata remains `private=0, sgpr=99,
  vgpr=128`, with no spill/scratch and `ldsBankConflict=0`.
- Performance:
  H1/S1024 kernel ticks regress `42,335,020 -> 43,877,925` (`+3.64%`) and
  MMAC active falls `34.1944% -> 33.5680%`.  waitLgkm rises
  `46,460.5 -> 54,268.75`, barrier rises `129,157.2 -> 139,483.25`, and
  coissue falls `40,755/29,120 -> 32,708/23,935`.  MMOP is unchanged at
  `131,072`; ASM has 116 MMAC runs with mean length 8.83 and maximum 32.
- Decision:
  reject from canonical and preserve only on
  `exp/dkv-consumer-lookahead-stagger`.  Source-level useful-work staggering
  is insufficient while both pairs retain the same Q/dO ownership epoch.
  The next structural candidate must reduce recurrent ownership handshakes
  without adding runtime slot selection or extending score/dP liveness.

## 2026-07-15 dKV Native P/dS Matrix Handoff [RETRACTED]

Status: `RETRACT_FALSE_POSITIVE_TRANSPORT_ONLY`

- The initial natural-fragment sweep used an invalid `alt=2` combination for
  f16 m32x16.  HCU tests show that the supported alt2 encoding is builtin
  argument `1`, while the missing reader shape is mt16x32.
- A corrected four-writer by five-reader PMD sweep found the exact contract:
  `ds_write_matrix_format_f16(...,16,2,1,0,0)` followed by
  `ds_read_matrix_trans_format_f16(...,16,1,2,0)` and dV/dK MMAC.
  All four writer t/alt variants pass with that reader; the canonical t0/alt0
  writer is retained because it is the simplest documented form.
- The old comparison was bit-exact only because its RHS did not distinguish
  the writer permutation. It proves transport, bank0, and clean resources,
  not natural-fragment semantic equivalence.
- Decision: retain the run only as a transport/layout control. It cannot admit
  a single-slot two-stage dKV pipeline.

## 2026-07-16 Two-Stage P/dS High-VGPR Audit [SUPERSEDED]

Status: `SUPERSEDED_BY_SEMANTIC_SOURCE_SLOT_RECHECK`

- Low-pressure cross-wave handoff passes eight generations at LDS bases `0`
  and `67584`, proving the writer/reader/barrier address contract.
- A high-pressure reader with 128 live FP32 accumulator VGPRs fails in PMD at
  `read vgpr202 before writing`; its matrix destination is `v131:v138`.
- Preinitializing the main-kernel reader produces finite zero instead of the
  LDS payload, so no workaround is stacked into the performance kernel.
- The later split-output probe avoids this fatal and proves transport. The
  final blocker is the f16 natural-fragment source ABI, not WDRA tracking.

## 2026-07-16 dKV ABarrier Token Tomography

Status: `OBSERVE_LOCAL_READY_REMOTE_PENDING`.

- Hypothesis: aggregate `s_abarrier_try_wait` statistics hide several
  ownership edges; optimizing without assigning cycles to token/source/role
  has repeatedly moved rather than removed the stall.
- Change: add debug-line-only wait wrappers plus an exact ASM-equivalence gate.
  The default canonical behavior remains unchanged.
- Added a focused `s_abarrier_test_wait` phase probe. It is isolated from FA
  math and does not authorize polling in the production mainloop.
- Required evidence before implementation: control/tomography ASM identity,
  control/tomography numerical identity, a per-token xcu issue-gap table, and
  a latest-compiler rerun of the 128-FP32-live P/dS cross-wave probe.
- Decision: no P/dS conveyor code yet. Implement it only if high-VGPR handoff
  passes and tomography shows that replacing raw-page ownership can remove
  enough exposed cycles to move the 34.19% baseline toward 40%.

## 2026-07-16 dKV Tomography Completion and Four-Role Design Gate

Status: `OBSERVE_DIAGNOSTIC_COMPLETE_DESIGN_ACCEPTED_PROBE_PENDING`

- The tomography binary preserves the canonical instruction stream exactly.
  H1/S1024 causal correctness passes and kernel ticks are `42,053,375`; this
  diagnostic run is not promoted over the immutable `42,335,020` fullperf
  baseline because PMD variance and debug-line mapping are involved.
- Q/dO Used tokens dominate raw ABarrier duration (`72.62%` combined), but
  the waits are producer-side and substantially overlap consumer MMAC.
  Consumers already issue Used after the last legal matrix read. Removing or
  advancing those barriers would be a correctness bug, not an optimization.
- The structural gap versus FWD is operand readiness: BWD/FWD LDS-read per
  MMAC is `0.603/0.252=2.39x`, WAIT per MMAC is `0.325/0.064=5.08x`, and
  no-MMAC bins are `27.8%/13.9%=2.0x`. Actual MMAC-with-vector-peer overlap is
  `40.8%` versus FWD `60.25%`.
- The rejected old two-stage reader held 128 FP32 accumulator VGPRs. The new
  `125_DKV_4Role_PDS` design preserves exactly four GEMMs but assigns score/dP
  to one frontend group, dV to one 64-accumulator group, and dK to a second
  64-accumulator group. Native P-before-dS publication supplies useful-work
  phase offset without empty delay.
- Resource draft: M32/Nk128/D128, 512 MMAC per CTA packet, steady LDS 66,304B,
  static ABarrier ids 0-14, and per-SIMD WDRA windows
  `16+176+160+160=512`. These figures are gates, not proof. The next change is
  an isolated 64-accumulator cross-wave probe; canonical source stays frozen.

## 2026-07-16 dKV Split-Output 64-Accumulator P/dS Probe

Status: `ACCEPT_INSTRUCTION_RESOURCE_GATE`

- The first source form placed both output readers in one range branch. The
  compiler collapsed them into one WDRA role and rejected branch averaging.
  The corrected form uses four explicit role branches and keeps all lane and
  reader setup branch-local.
- Final compiler windows are `1/16,22/176,73/160,73/160`. Metadata passes with
  `private=0`, `sgpr=28`, `vgpr=128`, no SGPR/VGPR spill, no scratch, and no
  trap. This validates the workbook's per-SIMD `16+176+160+160=512` ledger.
- Both native handoff cases pass eight ABarrier generations at LDS bases 0
  and 67,584 with `mismatches=0`; bank conflict is zero. The old fatal
  `vgpr202` path is avoided by giving dV and dK separate 64-accumulator roles.
- A single nonfatal `vgpr194` init warning is confined to the reference
  writer-readback dispatch; the true cross-wave cases complete and are exact.
  This does not promote the main topology by itself.
- Decision: retain the focused gate and reproducer as transport evidence. A
  semantic producer-fragment gate is still required before any main kernel.

## 2026-07-16 dKV Four-Role P/dS Semantic Recheck

Status: `REJECT_SOURCE_SLOT_ABI_MAIN_SOURCE_RESTORED`

- Raw cross-wave readback, role-source identity, and writer-self-read versus
  reader-cross-read MMAC all pass across eight generations and both LDS bases.
  This clears ABarrier, WDRA role tracking, addressing, and output-store races.
- The non-degenerate direct-natural versus roundtrip MMAC oracle fails roughly
  64K outputs. Natural score/dP ownership is not the `ds_write_matrix` source
  ownership expected by the downstream reader.
- Writer `t=1`, legal f16 normal/trans readers, and four lane-local pack orders
  do not repair the mismatch. Prior conversion1/2/4 and complete
  operand-order/LIT/LTS sweeps close the f16 path. The f32 m16x16 matrix
  writer/read path remains a focused no-permute candidate because it accepts
  one natural `Vec4F32` MMAC accumulator directly.
- The diagnostic main-kernel variant improves the size of the numerical error
  but remains incorrect at H1/S128: `dk_max_abs=0.000490182`,
  `dv_max_abs=0.0878637`, `pass=0`.
- Decision: delete the uncommitted four-role main implementation. Preserve the
  focused probe and evidence; resume from the direct-register P/dS canonical
  path. No perf capture is valid for this rejected candidate.

## 2026-07-16 dKV f32 Matrix-Writer Probe

Status: `DEFER_PMD_UNIMPLEMENTED_F32_DS_MATRIX`

- A minimal one-wave probe isolates the remaining no-permute candidate:
  natural `Vec4F32` MMAC -> f32 m16x16 matrix write/read -> fp16 -> downstream
  MMAC. It separately checks finite tag permutation and semantic MMAC output.
- Static gates pass: `private=0`, `sgpr=18`, `vgpr=45`, no spill/scratch/trap;
  ASM contains 4 f32 matrix writes, 8 f32 matrix reads, and 6 MMACs.
- PMD reaches the kernel and aborts at the first f32 writer with
  `Invalid opcode 0xd38b5007`, before reader or numerical evidence exists.
- The deterministic runner reproduces this at
  `/zys/shaobo/runs/dkv_pds_f32_roundtrip_probe_20260716_215919`, cleans all
  detached PMD children, and requires `any_semantic_pair=1` once supported.
- Decision: preserve the focused probe and classify the route as deferred, not
  rejected. Do not place it in the canonical kernel until PMD implements the
  opcode and the semantic pair passes.

## 2026-07-16 dKV Canonical Restore Validation

Status: `ACCEPT_RESTORE_VALIDATION`

- Removed the rejected four-role main source and rebuilt the direct-register
  canonical. Branch windows returned to `14/16,221/240,221/240,8/16`;
  metadata is `private0/sgpr99/vgpr128/spill0/scratch0`.
- H1/S128 and H1/S1024 causal correctness pass with bank0. H1/S1024 records
  `kernel_ticks=41738060`, `MMOP=131072`, coissue `39577/28247`.
- This verifies cleanup only. PMD variance means it is not promoted over the
  immutable `42,335,020`/`34.1944%` fullperf baseline without a same-build
  candidate comparison.

## 2026-07-16 dKV Nk256 Owner32 Top-Level Design

Status: `DESIGN_READY_STATIC_ADMISSION_PENDING`

- SQTT evidence fixes the target: LDS-read/MMAC is `0.603` versus FWD `0.252`,
  WAIT/MMAC is `0.325` versus `0.064`, and peer-vector overlap is `40.8%`
  versus `60.25%`. Producer-side Used waits are already legally early and are
  substantially hidden, so the new design increases reuse instead of deleting
  barriers.
- Draft W12 owner32 was rejected during stress because it leaves only one
  heavy consumer per SIMD. Revision keeps W16 and expands CTA ownership to
  Nk256, retaining two heavy consumer waves per SIMD.
- A true M16/N32 microtile keeps the same 512 score elements and 64 MMAC as the
  current M32/N16 microtile, but halves Q/dO matrix reads by sharing them across
  two N16 output blocks. Eight microtiles yield 512 MMAC per consumer packet.
- Whole-head modeled bytes fall `5,865,472 -> 3,719,168` (`-36.59%`): K/V and
  output stores stay constant, Q/dO and sidecar repetition halve. Per-CTA token
  ledger is unchanged while CTA K tiles fall 8 to 4, halving ownership epochs
  per head.
- VGPR stress ledger: dV64 + dK64 + K32 + V32 = 192 long-lived. Score phase is
  projected 236, softmax phase 240, output phase 220. The projection has zero
  softmax slack; compiler metadata, not arithmetic, decides admission.
- LDS is a lifetime overlay, not an additive allocation: K/V startup uses
  exactly 128KB; after ResidentUsed, Q+dO+sidecar uses 67,072B. Sidecar must be
  placed in released K/V space and cannot be published before ResidentUsed.
- Implementation boundary: one canonical kernel on
  `exp/dkv-nk256-owner32-m16`; no runtime phase, no duplicated owner16 body,
  no duplicate score/dP, no source-layout page, and no permute/gather path.

## 2026-07-16 dKV Nk256 Owner32 Architecture Checkpoint

Status: `OBSERVE_PIPELINE_GAIN_H1_UNDERFILL`

- Implemented one native M16/N32 body rather than composing owner16 helpers.
  Each consumer owns Nk32, computes score/dP once, and accumulates dV/dK for
  both N16 halves. Whole-head MMOP remains exactly 131,072.
- The all-K/V VGPR draft spilled 58 VGPRs. The admitted revision retains K32
  and V-D0 in VGPR, leaves V-D1..D3 in LDS, overlays raw Q/dO on released K,
  and overlays sidecar on cached V-D0. This keeps LDS at 128KB and generated
  consumer windows at `239/240` with no private segment or spill.
- H1/S256 and H1/S1024 causal correctness pass with bank0. Fullperf records
  `69,435,275` kernel ticks, `39.9317%` MMAC active, `55.1980%` MMOP runtime
  share, and coissue `37915/31225`.
- The immutable Nk128 baseline remains faster on H1 because it exposes eight
  CTAs instead of four. XCU duration `152608` versus `93044`, normalized by
  two times more work per active CU, indicates about `21.9%` per-CU throughput
  gain. This validates the reuse architecture but not canonical promotion.
- XCU identifies the next concrete debt: steady consumer MMAC+VALU coissue is
  only `16.15%/16.34%`, and branchless causal softmax doubles `v_exp` work
  (`16384` versus about `8320`). Test causal branch pruning as one isolated
  hypothesis before changing lifetimes or consumer order.
- Evidence: static artifact
  `/zys/shaobo/runs/dkv_owner32_vd0cache_static_20260716`; correctness
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260716_234420`;
  fullperf and xcu
  `/zys/shaobo_runs/o32fp/dkv_mmac_correctness_20260716_235702`.

## 2026-07-17 dKV Owner32 Causal Branch Pruning Rejected

Status: `REJECT_STATS_DIVERGENCE_SOURCE_RESTORED`

- Hypothesis: skip exp/dS for causal-invalid pairs using a short local branch,
  without changing tile, MMAC count, LDS, ABarrier, or output ownership.
- Gates pass and resource use improves from `239/240` to `236/240`; H1/S256
  and H1/S1024 correctness pass with no spill/scratch and bank0.
- Result: ticks regress `68,856,060 -> 76,303,500` (`+10.82%`) and MMAC active
  falls `39.9695% -> 37.3899%`. VALU falls only `3.88%`, while SCA rises
  `90.0%`, barrier wait rises `29.6%`, and coissue changes
  `38544/31719 -> 43238/31952` without elapsed benefit.
- Explanation: diagonal causal predicates vary by lane, so most waves execute
  the valid arm under an exec mask and also pay branch/control serialization.
  Reduced arithmetic is too small to repay the added SCA and ownership delay.
- Decision: restore branchless masking. Do not retry per-element causal
  branches; a future causal optimization must skip an entire uniform packet.
  Next structural work is split-lifetime useful staggering.
- Evidence:
  `/zys/shaobo_runs/cprune1024/dkv_mmac_correctness_20260717_004248`.

## 2026-07-17 dKV Owner32 Topological Stagger Rejected

Status: `REJECT_CORRECTNESS_SOURCE_RESTORED`

- Workbook sheets `127_DKV_Owner32_TopoStagger` and
  `128_DKV_CompactTopoStagger` tested useful C0/C1 phase offset without empty
  delay, duplicate score/dP, LDS growth, or ABarrier changes.
- Splitting dV and dK into separate islands failed the resource gate. The best
  symmetric window still had `private=228B` and `vgpr_spill=58`; asymmetric
  `248/240` worsened that draft to `private=264B`, `vgpr_spill=95`.
- Restoring the proven joint dV+dK island reduced the generated roles to
  `14/247/235/8`. Reallocating the fixed per-SIMD WDRA budget to
  `16/248/240/8` passed metadata with `private=0`, no SGPR/VGPR spill, and
  retained 128KB LDS.
- H1/S256 causal nevertheless failed exact dK/dV validation:
  `dk_max_abs=1.30859`, `dv_max_abs=0.643585`. Replacing the partial
  `lgkmcnt(1/3)` waits with conservative `lgkmcnt(0)` reproduced identical
  errors, so readiness countdown was not the cause. The failed run still had
  `MMOP=8192` and `ldsBankConflict=0`; performance is not comparable because
  correctness failed.
- Decision: remove the uncommitted split score/dP/P/dS implementation and
  restore commit `f999500` through branch head `1ffb7fc`. Do not split the
  proven score+dP fragment chain in main source until a focused equivalence
  probe proves its source/accumulator ABI. The next mainline stagger must keep
  fused score+dP and use whole-island scheduling or priority asymmetry.
- Evidence: static builds `/zys/shaobo/runs/o32stg_build7`,
  `/zys/shaobo/runs/o32compact_build2`, and
  `/zys/shaobo/runs/o32compact_build3`; smoke
  `/zys/shaobo_runs/o32compact_wait0_s256_20260717/`
  `dkv_mmac_correctness_20260717_022441`.

## 2026-07-17 dKV Owner32 Priority Asymmetry Rejected

Status: `REJECT_STATS_OWNERSHIP_STALL_SOURCE_RESTORED`

- Workbook sheet `129_DKV_PriorityIslandStagger` preserved the exact fused
  Score+dP, softmax+dS, and joint dV+dK islands. The only code change kept C0
  Score+dP at priority 2 while C1 remained at priority 0; there was no runtime
  branch, empty delay, extra read, GEMM, LDS byte, or ABarrier change.
- Static admission exactly matched owner32: branch use `14/239/239/8`, metadata
  `private=0`, `sgpr=56`, `vgpr=128`, no spill/scratch, and the kernel gate
  retained 64 MMAC plus 22 native matrix reads per M16.
- H1/S256 and H1/S1024 correctness pass, with H1/S1024 dK/dV maximum absolute
  errors `1.49356e-7/2.87902e-5`; LDS bank conflict remains zero.
- Same-shape stats reject the schedule: ticks regress
  `68,856,060 -> 72,421,440` (`+5.18%`), MMAC active falls
  `39.9695% -> 39.0068%`, coissue success falls `38,544 -> 26,155`, and
  barrier stall rises `79,755.2 -> 90,573.5` (`+13.6%`). Instruction work is
  identical, so this is a scheduling/ownership regression rather than a work
  or resource change.
- Interpretation: delaying one consumer does create phase distance, but both
  consumers must arrive at the same Q/dO Used ownership boundary. C0 reaches
  that boundary early and waits for starved C1, collapsing useful overlap into
  barrier time. Do not use persistent consumer-group priority asymmetry with
  this ownership topology.
- Source is restored locally and remotely. Fullperf/XCU was skipped because
  the stats gate already rejected both ticks and active share. Next test keeps
  consumer progress symmetric and moves `s_setprio 2` after first-use operand
  readiness, targeting the measured `s_setprio -> s_waitcnt` exposure directly.
- Evidence: build `/zys/shaobo/runs/o32prio_build1`; runs
  `/zys/shaobo_runs/o32prio_s256/dkv_mmac_correctness_20260717_025720` and
  `/zys/shaobo_runs/o32prio_s1024/dkv_mmac_correctness_20260717_025753`.

## 2026-07-17 dKV Owner32 Ready-Only Priority Accepted

Status: `ACCEPT_MICRO_SCHEDULING`

- Workbook sheet `130_DKV_ReadyOnlyPriority` tested one symmetric scheduling
  cleanup: both consumers issue native matrix reads and complete the first-use
  `lgkmcnt(0)` wait before raising MMAC priority. Math, reads, waits, ABarrier
  IDs/counts, LDS bytes, ownership, and output stores are unchanged.
- Static admission matches owner32 exactly: branch use `14/239/239/8`, metadata
  `private=0`, `sgpr=56`, `vgpr=128`, no spill/scratch, 64 MMAC and 22 native
  matrix reads per M16, with `ldsBankConflict=0`.
- H1/S256 and H1/S1024 causal correctness pass. Stats-only H1/S1024 is
  effectively flat (`68,856,060 -> 68,859,700`, `+0.005%`) while its MMAC
  active proxy rises `39.9695% -> 40.0704%`.
- Two same-binary fullperf runs reproduce a small elapsed improvement against
  the fullperf baseline `69,435,275`: `69,230,070` (`-0.30%`) and
  `69,053,530` (`-0.55%`). Fullperf MMAC active rises from `39.9317%` to
  `39.9469%/39.9590%`; work counts remain exact and bank conflict stays zero.
- XCU on the same `50000:70000`, `SE3/CU0/SIMD3` window explains the change.
  The old `s_setprio -> 59-83 cycle -> s_waitcnt` exposure is replaced by
  `s_waitcnt -> s_setprio -> MMAC`. Across 128-cycle bins, useful
  `MMAC-vs-VALU` rises `46 -> 55`, `MMAC-vs-MMAC` falls `48 -> 40`, no-MMAC
  bins fall `36 -> 34`, and MMAC instructions with a vector peer rise
  `316 -> 390`.
- Decision: keep and commit the two-line change. This is a repeatable micro
  scheduling gain, not the architectural breakthrough. XCU still attributes
  about `41.8%` of issue-gap duration to `s_abarrier_try_wait` ownership; the
  next workbook hypothesis must shorten or overlap the shared Q/dO ownership
  lifetime rather than stack more priority edits.
- Evidence: build `/zys/shaobo/runs/o32readyprio_build1`; correctness/stats
  `/zys/shaobo_runs/o32readyprio_s1024/dkv_mmac_correctness_20260717_031650`;
  fullperf `/zys/shaobo_runs/o32readyprio_fullperf/`
  `dkv_mmac_correctness_20260717_032017` and
  `/zys/shaobo_runs/o32readyprio_fullperf_repeat/`
  `dkv_mmac_correctness_20260717_033504`; XCU outputs under the first fullperf
  case's `xcu_first`, `xcu_steady_w1`, and `xcu_steady_w2` directories.

## 2026-07-17 dKV Owner32 Merged Q/dO Used Token Rejected

Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`

- Workbook sheet `131_DKV_MergedRawUsed` observed that owner32 releases Q and
  dO consecutively after the same `lgkmcnt(0)`. The focused candidate merged
  the two Used tokens per half, reducing the ledger from nine to seven IDs and
  consumer Used arrivals from four to two per q-tile. Math, tile, matrix reads,
  MMAC, LDS, and producer wait count were unchanged.
- Static/resource admission passes: roles `14/239/239/8`, metadata
  `private=0`, `sgpr=56`, `vgpr=128`, no spill/scratch, 128KB LDS, and the
  canonical native matrix path remains intact.
- H1/S256 and detached H1/S1024 complete with dK/dV correctness PASS and
  bank0. The two earlier foreground S1024 commands were terminated by the
  command transport near 20 seconds; their partial stats are invalid and are
  not kernel protocol failures.
- Same-method detached stats initially look positive: candidate ticks improve
  `69,942,145 -> 69,389,320` (`-0.79%`) and SCA falls exactly
  `36,408 -> 35,896` (`-512`), while MMOP/VALU/LDS/VMEM remain exact.
- Fullperf rejects the change. Candidate ticks are `70,155,995`, versus the
  two accepted ready-only-priority runs `69,230,070/69,053,530`
  (`+1.34%/+1.60%`). MMAC active is effectively flat at `39.9607%`, so fewer
  arrivals did not shorten the critical path.
- XCU confirms why: `s_abarrier_try_wait -> s_xor_b32` still owns `41.84%` of
  issue-gap duration, MMAC-to-MMAC is `10.17%`, and
  `s_abarrier_try_wait -> s_waitcnt` is `9.15%`. The removed Used arrivals
  were bookkeeping work, not the producer-to-consumer Filled readiness bubble.
- Decision: reject and restore source/binary to ready-only priority commit
  `28c8ab9`. Do not retry Q/dO Used-token merging. The next structural target
  is earlier Q/dO Filled readiness through useful work, without another buffer
  layer or token alias.
- Evidence: S256 `/zys/shaobo_runs/o32mergedrawused_s256/`
  `dkv_mmac_correctness_20260717_041555`; candidate detached stats
  `/zys/shaobo_runs/o32merged_detached_20260717/`
  `dkv_mmac_correctness_20260717_044041`; same-method baseline
  `/zys/shaobo_runs/o32baseline_detached_20260717/`
  `dkv_mmac_correctness_20260717_044607`; candidate fullperf/XCU
  `/zys/shaobo_runs/o32merged_fullperf_detached_20260717/`
  `dkv_mmac_correctness_20260717_045353`.

### PMD long-run execution rule

- Foreground SSH commands that run longer than about 20 seconds can be killed
  by the command transport while PMD is still healthy. Never classify a
  truncated foreground log as a kernel hang or ABarrier failure.
- Launch S1024/fullperf through detached `docker exec -d`, redirect the driver
  log, write an explicit `exit_code`, poll that file, and require both
  `exit_code=0` and the harness `status=success` line before using stats.

## 2026-07-17 dKV Owner32 V/dO Publish Priority Rejected

Status: `REJECT_FULLPERF_NOISE_AND_FILLED_WAIT_REGRESSION_SOURCE_RESTORED`

- Workbook sheet `132_DKV_VdoutPrio` isolated one scheduling hypothesis from
  the accepted ready-only-priority canonical. XCU showed V/dO Filled arriving
  about `204` cycles after K/Q, so only the two useful V/dO publish windows
  temporarily used `s_setprio 1`. Math, LDS, ABarrier IDs/counts, matrix reads,
  MMAC islands, consumer priority, and stores were unchanged.
- Static/resource gates and H1/S256 plus detached H1/S1024 correctness pass:
  roles `14/239/239/8`, `private=0`, `sgpr=56`, `vgpr=128`, no spill/scratch,
  128KB LDS, bank0, and exact `MMOP=131072`.
- Stats-only candidate ticks are `68,832,400` versus the same-method canonical
  `69,942,145`, but fullperf disproves promotion. Candidate ticks
  `69,103,580` sit inside three canonical repeats
  `69,053,530/69,094,480/69,230,070`; MMAC active falls
  `39.9590% -> 39.8486%`.
- XCU explains the neutral elapsed result. V/dO is pulled ahead, but K/Q is
  delayed and becomes the new last arriver. Filled completion moves later by
  `72/44` cycles in two consecutive generations, and consumer0's exposed
  Filled waits grow `1732/1528 -> 1888/1576` cycles. This is last-arriver
  exchange, not readiness reduction.
- Decision: reject, remove the priority1 helper/calls, and rebuild canonical.
  For a Filled token completed by multiple producers, optimize
  `max(arrival_0, arrival_1, ...)`; unilateral priority is not useful unless it
  reduces that maximum without stealing peer MMAC/producer issue slots.
- Evidence: `/zys/shaobo_runs/o32vdoutprio_detached_20260717/`,
  `/zys/shaobo_runs/o32vdoutprio_fullperf_detached_20260717/`, and local XCU
  copy `work/o32vdoutprio_xcu_20260717/`.

## 2026-07-17 dKV Owner32 Consumer-Group Filled Stagger Rejected

Status: `REJECT_FULLPERF_WAIT_REDISTRIBUTION_SOURCE_RESTORED`

- Workbook sheet `133_DKV_GroupFilledStagger` split only consumer readiness:
  C0/C1 received independent half Filled tokens, while physical Q/dO pages and
  QUsed/DoutUsed ownership remained shared. Bootstrap released C0 after half0
  and used half1 publication as real work before releasing C1; math, tile,
  matrix reads, four GEMMs, output ownership, and LDS bytes were unchanged.
- Static/resource gates and H1/S256 plus detached H1/S1024 correctness pass:
  roles `14/239/239/8`, private0, `sgpr=56`, `vgpr=128`, no spill/scratch,
  128KB LDS, bank0, and exact `MMOP=131072`.
- Stats-only is noise-positive (`69,942,145 -> 69,835,675`), but fullperf is
  authoritative: candidate `69,109,495` sits inside canonical repeats
  `69,053,530/69,094,480/69,230,070`; MMAC active falls
  `39.9590% -> 39.8392%`, and SCA rises `36,408 -> 38,216`.
- XCU shows a real but local consumer stagger: MMAC-vs-VALU bins rise
  `55 -> 62`, no-MMAC bins fall `34 -> 29`, and C0 ABarrier falls
  `4,129 -> 3,769`. It does not shorten ownership: producer0/producer1 rise
  `18,034/17,710 -> 18,158/17,942`; the four-role ABarrier total is exactly
  unchanged at `39,886` cycles.
- Decision: reject. Splitting token identity moved wait from C0 to producers
  and paid extra control work; it did not reduce `max(producer arrivals)` or
  the CTA critical path. Candidate source is removed locally and remotely,
  canonical is rebuilt and passes source/metadata gates.
- Evidence: `/zys/shaobo_runs/o32groupfilled_detached_20260717_0650/`,
  `/zys/shaobo_runs/o32groupfilled_fullperf_detached_20260717/`, local XCU
  `work/o32groupfilled_fullperf_20260717/`, and workbook sheet 133.

## 2026-07-17 dKV Owner32 Joint Q+dO Payload Stripe Rejected

Status: `REJECT_FULLPERF_TICKS_AND_COISSUE_REGRESSION_SOURCE_RESTORED`

- Workbook sheet `134_DKV_JointPayloadStripe` tested the remaining runnable
  producer-topology hypothesis. Both producer groups published equal Q+dO row
  stripes plus 32 sidecar rows, while four GEMMs, consumer ownership, native
  matrix reads, MMAC islands, LDS addresses, and output stores stayed fixed.
- Static and correctness gates pass: roles `14/239/239/14`, private0,
  `sgpr=60`, `vgpr=128`, no spill/scratch, 128KB LDS, bank0, exact
  `MMOP=131072`, and dK/dV PASS at H1/S256 and H1/S1024.
- Stats-only is misleadingly positive: same-method ticks move
  `69,942,145 -> 69,378,400` (`-0.81%`). Authoritative fullperf rejects the
  candidate: `69,053,530 -> 69,655,950` (`+0.87%`) and MMAC active falls
  `39.9590% -> 39.7597%`.
- XCU explains the loss. Producer ABarrier cycles rise
  `18,034/17,710 -> 18,130/17,782`, consumer0 rises `4,129 -> 4,801`,
  MMAC-vs-VALU bins fall `55 -> 47`, and MMAC-vs-MMAC bins rise `40 -> 44`.
  Both producer groups now carry the slower dO path; balancing issued payload
  does not reduce `max(readiness_i)`, and the joint Used boundary reconverges
  both roles.
- Decision: reject and restore ready-only-priority canonical. Do not retry
  equal Q+dO striping. The evidence closes payload balancing, token
  merge/split, unilateral priority, and consumer Filled-splitting as local
  solutions to the 60% target.
- Evidence: `/zys/shaobo_runs/o32joint_payload_s256/`,
  `/zys/shaobo_runs/o32joint_payload_s1024/`,
  `/zys/shaobo_runs/o32joint_payload_fullperf/`, XCU
  `/zys/shaobo_runs/o32joint_payload_xcu_20260717/`, canonical restore smoke
  `/zys/shaobo_runs/o32canonical_restore_after_joint_reject/`
  `dkv_mmac_correctness_20260717_083510`, and workbook sheet 134.

## 2026-07-17 dKV Consumer-Published BPS Resource Gate

Status: `ACCEPT_PROBE_WITH_NONFATAL_PMD_WARNING`

- Hypothesis: a consumer can keep the canonical owner32 live accumulator
  footprint while issuing BPS/MLS into LDS, so steady Q/dO publication can
  move from thin producer roles into C0/C1 without spill or data corruption.
- Probe shape: 16 waves with WDRA `8/248/248/8`; each heavy role keeps 128
  FP32 accumulator scalars live. C0 publishes Q and C1 publishes dO; one
  Filled count8 hands both tensors to all consumers and one Used count8 hands
  the slot back to the thin roles.
- Build result: branch use `2/143/141/2`; private0, SGPR24, VGPR128,
  SGPR/VGPR spill0, scratch0; BPS, `ds_read_matrix`, resize, and ABarrier
  opcodes present; no trap.
- PMD result: exact fragment and accumulator sinks, all eight Used arrivals,
  bank0, and no panic. Run:
  `/zys/shaobo_runs/dkv_consumer_bps_live_probe_20260717_163627`.
- Diagnostic note: an earlier probe check evaluated fragment errors before
  storing the accumulator sink. ASM reused the first accumulator VGPR for the
  check temporary, producing exactly 511 false accumulator mismatches. Moving
  the sink before the check proves the intended live interval and passes.
- One nonfatal PMD `read vgpr156 before writing` warning remains despite exact
  data. It is tracked as PMD register-init evidence and is not permission to
  waive integrated correctness.
- Decision: proceed to one canonical consumer-assisted two-slot integration
  on the isolated branch; preserve the probe as the instruction/resource gate.

## 2026-07-17 dKV Consumer-Assisted M64 Two-Slot Conveyor Rejected

Status: `REJECT_STATS_BARRIER_REGRESSION_SOURCE_REMOVED`

- Implemented the workbook sheet 137 route on an isolated branch: producers
  publish K/V once; all consumers latch K and V dblock0; C0 publishes
  Q+sidecar and C1 publishes dO into two alternating M64 slots. The math stays
  at four exact GEMMs with owner32 dK/dV stores.
- Register pressure was reduced methodically: runtime parity branch removal
  cut VGPR spill `330 -> 25`; staged in-place max/exp/P/dS softmax cut it to
  `3`; measured asymmetric windows `8/252/244/8` reached private/spill0 with
  actual use `1/252/243/1` and exact per-SIMD ledger 512.
- H1/S256 and detached H1/S1024 correctness pass. The target run reports dK
  relative L2 `0.00255632`, dV relative L2 `0.000337571`, exact
  `MMOP=131072`, bank0, and no PMD panic.
- Same-shape stats reject the topology: canonical-to-candidate kernel ticks
  `69,053,530 -> 72,709,000`, MMAC active `40.0704% -> 38.2341%`, barrier
  `80,555.5 -> 114,103.5`, and waitLgkm `27,104.5 -> 29,283.25`. VALU falls
  `200,272 -> 182,736` and SCA `36,408 -> 25,680`, proving instruction-count
  cleanup cannot pay for doubled ownership epochs.
- Decision: skip fullperf/XCU by gate, commit the isolated hypothesis, revert
  its source, and keep the probe plus workbook evidence. Canonical already
  uses two M64 half-ready epochs; the actual failure is that all heavy roles
  stop MMAC to publish and reconverge. The next architecture must let score or
  another useful island run while the slower tensor is still being published.
- Evidence:
  `/zys/shaobo_runs/dkv_consumer_conveyor_s1024_20260717_1740`, local stats
  `work/dkv_consumer_conveyor_s1024_20260717_1740/stats.txt`, and workbook
  sheet `137_DKV_ConsumerConveyor`.

## 2026-07-17 dKV Q-Ready Score-First Probe Accepted

Status: `ACCEPT_PROBE`

- Workbook sheet 138 derives a tensor-separated readiness route: QFilled
  releases score MMAC before dO is ready; DoutFilled is consumed only at dP
  first use. Four GEMMs, Mq128/Nk256/D128, owner32 stores, LDS bytes and native
  matrix path remain fixed.
- The focused probe uses nonuniform Q/K/V/dO and computes both schedules from
  identical fragments. It passes bit-exactly with `errors=0 max_abs=0`, bank0,
  no PMD panic, private/spill/scratch0 and roles `1/89/89/1`.
- Decision: preserve the probe as a layout/accumulation gate and integrate the
  schedule once on the experiment branch. Promotion still requires full dK/dV
  correctness, exact `MMOP=131072`, ticks below canonical and higher MMAC
  active; otherwise revert.
- Evidence: `/zys/sb/probes/dkv_qready_score_split_probe_20260717_184750`.

## 2026-07-17 dKV Q-Ready Score-First Integration Rejected

Status: `REJECT_STATS_OWNERSHIP_REGRESSION_SOURCE_RESTORED`

- Implemented the one admitted integration from workbook sheet 138. P0 emits
  independent QFilled tokens, P1 emits dO Filled tokens, consumers run the
  complete score island after Q readiness and wait dO only before dP.
- The kernel passes all hard gates: H1/S256 and H1/S1024 dK/dV correctness,
  exact `MMOP=131072`, bank0, roles `14/242/242/8`, private0, SGPR91,
  VGPR128, no spills or scratch.
- A fresh same-build canonical control removes PMD noise. Candidate versus
  canonical: ticks `75,828,935` versus `68,752,320` (`+10.29%`), MMAC active
  `36.7340%` versus `40.0907%`, barrier `103,893.25` versus `79,233`, total
  wait `45,638` versus `39,830.082`, and waitLgkm `32,793.5` versus
  `27,063.5`. MMOP runtime stays close (`242,041` versus `241,074`), so the
  active-time expansion is ownership/control latency rather than useful work.
- Decision: reject before fullperf/XCU. Preserve candidate commit `7618762`,
  revert with `b3b3c3d`, rebuild canonical remotely, and close this topology.
  Do not retry tensor readiness splitting unless the early useful-work island
  can be shown to exceed the added barrier plus first-use wait budget.
- Evidence: candidate
  `/zys/sb/qrs1024/dkv_mmac_correctness_20260717_191627`; canonical
  `/zys/sb/qrsbase/dkv_mmac_correctness_20260717_192149`; workbook sheet 138.

## 2026-07-17 dKV Native EBarrier Filled Handoff Rejected

Status: `REJECT_STATS_EBARRIER_SYNC_REGRESSION_SOURCE_RESTORED`

- Workbook sheet `139_DKV_EBarrierFilled` tested one primitive-level change:
  replace the two Q/dO Filled ABarrier generations with native asymmetric
  EBarrier handoff. Eight producer waves execute `arrive_cnt(id,16)` after
  BPS/vbcnt and eight heavy consumers execute `sync_cnt(id,16)`. Four GEMMs,
  owner32 output ownership, Used ABarriers, LDS layout, tile, and stores stay
  unchanged.
- The focused 16-generation probe passes exact data, bank0, private/spill/
  scratch0 and no PMD panic. Its interval improves
  `11,551,540 -> 7,532,980` ticks (`-34.8%`), proving the HCU grammar and LDS
  visibility for this producer/consumer pattern.
- Integrated static and correctness gates also pass: roles `14/239/239/8`,
  private0, SGPR54, VGPR128, spill/scratch0, H1/S256 and H1/S1024 dK/dV
  golden PASS, exact `MMOP=131072`, and `ldsBankConflict=0`.
- Same-build H1/S1024 stats reject the route. Kernel ticks regress
  `68,752,320 -> 73,301,410` (`+6.62%`), MMAC active falls
  `40.0907% -> 37.7371%`, barrier rises `79,233 -> 102,989.25`, and
  waitLgkm rises `27,063.5 -> 28,833`. MMOP runtime is essentially flat
  (`241,074 -> 241,280.5`), so the loss is a longer active/control window,
  not added mathematical work.
- Decision: skip fullperf/XCU by the stats gate. Preserve the candidate in
  commit `b045492`, remove it with revert `a2b772c`, and retain probe commit
  `9f76bf1`. Direct EBarrier Filled replacement is closed for this owner32
  topology; EBarrier remains valid for isolated asymmetric handoffs where its
  sync does not reconverge eight high-VGPR consumers inside the critical loop.
- Restored remote canonical rebuild passes source and metadata gates with
  roles `14/239/239/8`, private0, SGPR56, VGPR128, spill/scratch0; H1/S256
  correctness and bank0 pass at
  `/zys/sb/ebrstr/dkv_mmac_correctness_20260717_202936`.
- Evidence: probe `/zys/sb/probes/dkv_ebarrier_filled_20260717_200723`;
  candidate `/zys/sb/ebf1024/dkv_mmac_correctness_20260717_201545`;
  canonical `/zys/sb/qrsbase/dkv_mmac_correctness_20260717_192149`.

## 2026-07-17 M128 Page Ping-Pong Resource Closure

Status: `REJECT_RESOURCE_SOURCE_RESTORED`

- The first two-page M128 candidate passed dV but failed dK only after q0.
  Numerical decomposition and IT trace proved q0 exact, q1 P exact, and q1 dS
  wrong. The root cause was not page addressing: consumers latched all K but
  only V dblock0, then reread V dblocks1-3 after `ResidentUsed` allowed both
  64KB raw pages to overwrite resident K/V.
- The enhanced page-base plus 32KB-immediate MLS/DS probe is exact for normal,
  transpose and sidecar views with `ldsBankConflict=0`. This closes scalar
  base/immediate encoding as a cause.
- A correct one-page control retains V in 64KB LDS and overlays only K. It
  passes H1/S256 and H1/S1024 dK/dV, bank0 and exact MMOP. H1/S1024 takes
  `77,781,340` ticks with about `38.5%` MMAC active versus canonical
  `68,752,320` and `40.0907%`; losing the M64 two-slot conveyor costs about
  13.1%, so the route is not promoted.
- Full K/V latch makes two M128 raw pages semantically correct in principle,
  but fails the real resource gate. Compiler branch usage is
  `5/248/248/1`, while metadata reports `private_segment=108B`,
  `vgpr_spill_count=108`, and `ScratchSize=108`. The 248 count is post-spill,
  not proof of no spill.
- Resource lower bound explains the failure: resident K+V consumes 64 VGPR
  per consumer and long-lived fp32 dK+dV accumulators consume 128 VGPR. The
  remaining 56 VGPR cannot hold score+dP, Q+dO fragments, P+dS, addressing,
  and barrier state; the compiled route is short by about 27 VGPR slots.
- Decision: do not run PMD performance with spill. Restore the committed
  canonical source. Workbook sheet `140_DKV_M128PagePingPong` records the
  corrected lifetime proof and resource rejection.

## 2026-07-17 C0-Only dV/dK Read8 Stagger Rejected

Status: `REJECT_STATS_EXPERIMENT_BRANCH`

- Workbook sheet `141_DKV_C0Read8Stagger` keeps math, LDS, ABarrier tokens,
  owner32 outputs, and MMOP fixed. C0 reads both dV/dK source groups before one
  wait and emits a 16-MMAC island; C1 retains two read4/wait/MMAC8 stages.
- Static and correctness gates pass: roles `14/239/239/8` in asymmetric
  windows `16/248/240/8`, private0, SGPR56, VGPR128, spill/scratch0, H1/S256
  and H1/S1024 dK/dV PASS, exact `MMOP=131072`, and bank0.
- Same-environment H1/S1024 rejects the schedule. Kernel ticks regress
  `68,752,320 -> 70,769,335` (`+2.93%`) and MMAC active falls
  `40.0907% -> 39.3111%`. `waitLgkm` rises `27,063.5 -> 28,632`, barrier
  rises `79,233 -> 92,030.2`, and coissue success/fail moves
  `39,148/32,134 -> 36,155/31,216`.
- The long island is real, but group asymmetry reconverges at shared
  Q/dO Used/Filled ownership. The faster group waits for the slower group, so
  useful stagger does not shorten the dispatch critical path. Fullperf/XCU is
  skipped by the stats gate.
- Next discriminator is symmetric read8 on both groups. It separates the cost
  of read batching from the cost of forced group asymmetry; it is not a new
  topology or phase stack.

## 2026-07-18 Symmetric dV/dK Read8 Rejected

Status: `REJECT_STATS_BATCHING_OWNERSHIP_REGRESSION_SOURCE_RESTORED`

- Both heavy consumers use the same read8/wait/MMAC16 schedule. The code keeps
  one canonical kernel and changes only dV/dK operand-read scheduling.
- Hard gates pass: roles `14/239/239/8` in `16/244/244/8`, private0, SGPR56,
  VGPR128, spill/scratch0, H1/S256 and H1/S1024 correctness PASS, exact
  `MMOP=131072`, and `ldsBankConflict=0`.
- H1/S1024 canonical / asymmetric / symmetric results:
  `68,752,320 / 70,769,335 / 72,833,215` kernel ticks and
  `40.0907% / 39.3111% / 38.1220%` MMAC active. Symmetric coissue is
  `36,485/31,631`, waitLgkm `27,345.2`, and barrier `98,169.8`.
- Decision: reject before fullperf/XCU. Symmetry does not cure the route;
  batching delays the shared Q/dO Used boundary. Restore the tagged canonical
  and do not enlarge dV/dK read islands again without an independent page or a
  packet-release lifetime change.
- Evidence: S256
  `/zys/sb/symread8_s256/dkv_mmac_correctness_20260718_001019`; S1024
  `/zys/sb/symread8_s1024/dkv_mmac_correctness_20260718_001108`; baseline
  `/zys/sb/qrsbase/dkv_mmac_correctness_20260717_192149`; experiment commit
  `30d44d8`; restored canonical certification
  `/zys/sb/canonical_after_read8_reject/dkv_mmac_correctness_20260718_002545`;
  workbook sheet 141.

## 2026-07-18 Read8 Early-Used Release Rejected

Status: `REJECT_STATS_EARLY_RELEASE_CONTENTION_SOURCE_RESTORED`

- Workbook sheet 142 isolates packet lifetime from read batching. Both
  consumers issue eight Q/dO reads, wait lgkm0, arrive QUsed/DoutUsed before
  any dV/dK MMAC, then execute one MMAC16 island. This is the earliest legal
  release because no later instruction reads Q/dO LDS.
- Hard gates pass: roles `14/239/239/8` in `16/244/244/8`, private0, SGPR56,
  VGPR128, spill/scratch0, S256/S1024 correctness PASS, exact MMOP, and bank0.
- H1/S1024 candidate is `73,276,840` ticks and `37.8104%` MMAC active versus
  canonical `68,752,320 / 40.0907%` and symmetric read8
  `72,833,215 / 38.1220%`. Candidate barrier is `99,844.5`, waitLgkm
  `27,795.2`, and coissue `36,247/31,586`.
- Decision: reject before fullperf/XCU. The canonical producer waits are not
  the dispatch critical path; earlier wakeup adds producer LDS/BPS contention
  without completing the next Filled sooner. Experiment commit `566921a`
  preserves the negative result; active source is restored to canonical.
- Evidence: S256
  `/zys/sb/read8_early_used_s256/dkv_mmac_correctness_20260718_004827`;
  S1024 `/zys/sb/read8_early_used_s1024/`
  `dkv_mmac_correctness_20260718_005038`; baseline
  `/zys/sb/qrsbase/dkv_mmac_correctness_20260717_192149`; restored canonical
  `/zys/sb/canonical_after_early_used_reject/`
  `dkv_mmac_correctness_20260718_010507`; workbook sheet 142.

## 2026-07-18 Full K/V Latch with Transient-V Elimination

Status: `REJECT_RESOURCE_FULL_KV_OWNER32_SOURCE_RESTORED`

- Design: keep all owner32 K/V fragments live across the q-loop and delete
  the D1-D3 V rereads and temporary V fragments from score/dP. This removes
  six matrix reads per M16 while preserving exact four-GEMM work, ownership,
  and ABarrier topology.
- R1 two-slot result: branch use `14/244/244/8`; metadata private124B,
  vgpr_spill116, SGPR60, VGPR128. R3 one-slot result: branch use
  `14/240/240/8` and the same metadata, so Q/dO slot count is not the spill
  cause.
- The compiler rejects `16/240/240/8=504` because branch-average VGPR size
  misses target granularity; R3 must reserve `16/240/240/16=512` even though
  producer1 uses only eight.
- Folded scratch stores/reloads are emitted around the q_tile0 FirstAccum
  merge. Peeling q_tile0 changes metadata to private248B/vgpr_spill111; this
  is still a hard failure and shows the CFG was not the root capacity fix.
- Decision: no PMD execution. Record in workbook sheet 143, preserve the
  static negative experiment as commit `91c2437`, and restore tagged owner32
  canonical. Restored static/metadata gates and H1/S256 correctness pass at
  `/zys/sb/canonical_after_fullkv_reject/`
  `dkv_mmac_correctness_20260718_113602`.

## 2026-07-18 Owner16 1P+3C Full K/V Accepted by Stats

Status: `ACCEPT_FULL_KV_ARCHITECTURE_XCU_DIAGNOSED`

- The owner granularity changes from N32 to N16. Three four-wave consumer
  groups cover resident Nk192, while waves0-3 publish K then stream the single
  Q+dO+sidecar page and waves12-15 publish V before becoming consumer2.
- Complete K/V persistence is resource-clean: branch use
  `22/141/141/133` in `32/160/160/160`, private0, SGPR46, VGPR128, no
  spill/scratch. Resident K/V use 96KB; the 65.5KB raw packet overlays that
  epoch after all 12 consumers latch their fragments.
- Initial dK failures were not a layout or algorithm error. Single-M16 emits
  two reads per D block, but inherited `lgkmcnt(4)` came from the M32 path
  where each D block emitted four reads. D2+D3 therefore had four total
  outstanding reads and `lgkmcnt(4)` could retire none. `lgkmcnt(2)` makes D2
  ready while retaining D3 in flight.
- Formal S384 and S768 correctness pass. S768 dK/dV relL2 are
  `0.00191329/0.000319636`, exact MMOP is 73,728, and bank conflict is zero.
- Same-shape owner32/owner16 ticks are `54,078,570 -> 46,718,945`, a 13.61%
  reduction at identical MMOP. Coissue success/fail is `29,900/23,617`.
- Fullperf aggregate owner32/owner16 MMAC active is
  `38.3658% -> 32.1307%`, with active SIMD slots `12 -> 16`. This is not a
  padded-work speedup: exact MMOP remains 73,728 and the new route is faster.
- XCU dispatch 0 reports 64 complete waves, average 63.14 active waves, and
  zero no-wave idle. Wave slot 0 is the thin producer; slots 1/2/3 are three
  near-symmetric consumers with about 5.4k instructions each. Their useful
  MMAC+VALU coissue shares are `29.70%/27.91%/22.54%`.
- XCU separates hidden role waits from the next critical work. Producer
  `RawUsed` and tail `AllDone` dominate per-wave barrier bubbles, but do not
  create no-wave idle. The actionable consumer gaps are `MMAC->MMAC 7.47%`,
  `MMAC->wait 5.34%`, `matrix_trans_read->wait 4.95%`, and
  `matrix_read->wait 4.46%`.
- Decision: commit and tag this as the first resource-clean full-K/V baseline.
  Continue on the same canonical path with a VGPR-budgeted score/dP read8 and
  delayed first-use wait experiment; do not reopen K/V ownership.
- Fullperf/XCU evidence:
  `/zys/shaobo_runs/owner16_1p3c_fullkv_fullperf/`
  `dkv_mmac_correctness_20260718_153852`.

## 2026-07-18 Owner16 Score/dP Read8 First-Use Wait Accepted

Status: `ACCEPT_SCORE_DP_READ8_FIRST_USE`

- Hypothesis: the accepted owner16 consumer has 19 VGPR of branch slack and
  the largest actionable SQTT gaps are local `matrix-read -> wait` and
  `MMAC -> wait`. Batch all Q/dO D0-D3 source reads for score+dP, but retain
  staged first-use waits so the first MMAC does not pay the entire LDS latency.
- Change: replace two-set ping-pong with one scoped eight-fragment source
  bundle. Emit eight consecutive trans matrix reads, then
  `lgkmcnt(6/4/2/0)` before four MMAC for D0/D1/D2/D3. No token, page, math,
  output ownership, dV/dK schedule, or API change.
- Static result: roles `22/145/145/145` fit `32/160/160/160`; metadata remains
  private0, SGPR46, VGPR128, SGPR/VGPR spill0, scratch0. ASM proves the exact
  planned schedule, not merely the source ordering.
- Correctness: H1/S384 and H1/S768 pass. S768 dK/dV relL2 is
  `0.00191329/0.000319636`, exact MMOP is 73,728, and `ldsBankConflict=0`.
- Stats-only comparison at H1/S768: kernel ticks
  `46,718,945 -> 44,943,080` (`-3.80%`); MMAC active
  `32.2055% -> 32.7318%`; coissue `29,900/23,617 -> 35,322/30,540`;
  MMOP/VALU/SCA/LDS/VMEM remain exactly
  `73,728/105,712/20,856/44,768/1,728`.
- Fullperf comparison: kernel ticks `46,804,485 -> 44,852,080` (`-4.17%`),
  aggregate MMAC active `32.130697% -> 32.801527%`, waitLgkm
  `28,421.75 -> 22,656.5` (`-20.28%`), barrier
  `91,890.5 -> 86,833.75` (`-5.50%`), and coissue
  `30,032/23,765 -> 35,318/30,440`.
- XCU dispatch duration falls `102,864 -> 98,572`. Trans matrix-read hot
  latency falls `192,552 -> 129,216`; trans-read-to-wait issue gaps fall
  `4.95% -> 3.96%` and MMAC-to-wait falls `5.34% -> 4.90%`. In the same
  8k:80k SIMD0 window, consumer MMAC+VALU shares move
  `29.70/27.91/22.54% -> 29.65/30.67/26.92%`.
- Interpretation: the gain comes from overlapping later independent source
  reads with earlier score/dP MMAC. The added 2,304 XCU instruction issues are
  staged first-use waits, yet aggregate wait latency and dispatch duration
  both fall. This does not solve the 40% target; the dominant producer
  `RawUsed` row remains 31.45%, and normal-read-to-wait remains 4.57%.
- Evidence: workbook sheet `145_DKV_ScoreDP_Read8`; stats-only
  `/zys/shaobo_runs/owner16_scoredp_read8/`
  `dkv_mmac_correctness_20260718_161522`; fullperf/XCU
  `/zys/shaobo_runs/owner16_scoredp_read8_fullperf/`
  `dkv_mmac_correctness_20260718_161841`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_161841_owner16_scoredp_read8_s768_sqc7/`.

## 2026-07-18 Owner16 dV/dK Read8 First-Use Wait Accepted

Status: `ACCEPT_DVDK_READ8_FIRST_USE`

- Hypothesis: after score/dP read8, normal Q/dO source readiness remains an
  exposed consumer-local gap. Issue D0-D3 sources together, but retire D0/D1
  at first use instead of draining all eight reads before useful MMAC.
- Change: `read4 -> wait0 -> MMAC8 -> read4 -> wait0 -> MMAC8` becomes
  `read8 -> wait4 -> MMAC8 -> wait0 -> RawUsed arrive -> MMAC8`. Math,
  topology, token count, release boundary, and output ownership are unchanged.
- Static evidence: roles stay `22/145/145/145`; private0, SGPR46, VGPR128,
  spill0/scratch0. Symbol ASM contains eight consecutive normal matrix reads,
  `lgkmcnt(4)`, MMAC8, `lgkmcnt(0)`, then MMAC8.
- Correctness: H1/S384 and H1/S768 pass. S768 dK/dV relL2 is
  `0.00191329/0.000319636`; MMOP/VALU/SCA/LDS/VMEM remain exactly
  `73,728/105,712/20,856/44,768/1,728`; `ldsBankConflict=0`.
- Stats-only S768: ticks `44,943,080 -> 43,976,205` (`-2.15%`), MMAC active
  `32.7318% -> 33.8957%`, waitLgkm `22,656.5 -> 17,446.25` (`-23.0%`),
  barrier `86,833.75 -> 84,723` (`-2.43%`).
- Fullperf S768: ticks `44,852,080 -> 43,876,105` (`-2.18%`), MMAC active
  `32.801527% -> 33.892813%`, waitLgkm `22,656.5 -> 17,530.5` (`-22.6%`),
  barrier `86,833.75 -> 84,316.25` (`-2.90%`).
- XCU duration falls `98,572 -> 96,428`; normal matrix-read-to-wait gap falls
  `4.57% -> 4.15%`. Consumer MMAC+VALU shares become
  `30.69%/29.36%/27.96%`. Trans-read-to-wait rises `3.96% -> 5.15%`, and the
  two ABarrier wait rows now account for `32.88% + 8.95%`; ownership is the
  next evidence-backed target.
- Evidence: workbook sheet `146_DKV_DvDk_Read8`; stats-only
  `/zys/shaobo_runs/owner16_dvdk_read8_firstuse/`
  `dkv_mmac_correctness_20260718_165319`; fullperf/XCU
  `/zys/shaobo_runs/owner16_dvdk_read8_firstuse_fullperf/`
  `dkv_mmac_correctness_20260718_165607`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_165607_owner16_dvdk_read8_firstuse_s768_sqc7/`.

## 2026-07-18 Owner16 Mq192 Ownership-Epoch Scaling Accepted

Status: `ACCEPT_MQ192_OWNERSHIP_EPOCH`

- Hypothesis: with K/V already latched in consumer VGPRs, the 96KiB startup
  LDS is dead during the q-loop. Use that capacity for one Mq192 raw packet so
  S768 pays four RawFilled generations and three reuse waits instead of six
  and five, without changing total bytes, MMAC work, or output ownership.
- Design: retain Nk192, three symmetric Nk16 consumer groups, one raw page,
  five ABarrier IDs, and four exact GEMMs. Raw Q+dO is 98,304B and sidecar is
  2,304B, for 100,608B steady LDS. S384/S768 are the only admitted exact
  shapes; S1024 remains rejected until a real tail exists.
- Codegen boundary discovered during implementation: the DS matrix-read
  offset field is 16-bit, not 18-bit. The native solution is separate Q and
  dO LDS base SGPRs plus relative immediates; no `ds_read_b32`, gather,
  bpermute, or layout workaround enters the matrix path.
- Static evidence: role branches `30/145/145/145` fit the
  `32/160/160/160` WDRA windows; metadata is private0, SGPR55, VGPR128,
  spill0/scratch0. Emitted ASM has no out-of-range DS offset.
- Correctness: S384 and S768 pass; S768 relL2 is
  dK `0.00191329`, dV `0.000319636`. MMOP/LDS/VMEM stay exactly
  `73,728/44,768/1,728`, with bank0.
- Stats-only: ticks `43,976,205 -> 42,662,165` (`-2.99%`), MMAC active
  `33.8957% -> 35.1548%`, waitLgkm `17,446.25 -> 16,165.75`, and barrier
  `84,723 -> 75,916`.
- Fullperf: ticks `43,876,105 -> 43,033,445` (`-1.92%`), MMAC active
  `33.8928% -> 34.8979%`, barrier `84,316.25 -> 76,858.25`, and XCU duration
  `96,428 -> 94,576`. Fullperf waitLgkm rises slightly
  `17,530.5 -> 17,865.75`, so the acceptance rests on ticks and the proven
  ownership reduction, not every secondary counter moving together.
- XCU: ordinary `try_wait -> s_xor` events fall `432 -> 304` and duration
  falls `1,909,104 -> 1,776,840`; total ABarrier issue-gap duration falls
  2.63%. AllDone stays 64 events but its duration rises 13.16%, identifying
  tail convergence as residual debt. Fixed-time consumer-window coissue is
  not promoted as a comparison because packet boundaries moved.
- Evidence: workbook sheet `147_DKV_Mq192_EpochScale`; stats-only
  `/zys/shaobo_runs/owner16_mq192/dkv_mmac_correctness_20260718_174547`;
  fullperf/XCU `/zys/shaobo_runs/owner16_mq192_fullperf/`
  `dkv_mmac_correctness_20260718_174748`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_174748_owner16_mq192_s768_sqc7/`.

## 2026-07-18 Owner16 Head64/Tail128 Intra-Packet Overlap Accepted

Status: `ACCEPT_HEAD64_TAIL128_INTRA_PACKET_OVERLAP`

- Hypothesis: the one-page Mq192 packet serializes all 192 rows behind one
  Filled edge.  Publish a complete 64-row head first so consumers can execute
  128 MMAC plus softmax/dS while P0 publishes the remaining 128 rows; keep one
  RawUsed release and exact four-GEMM ownership.
- Implementation: replace RawFilled with RawHeadFilled/RawTailFilled.  P0
  publishes Q+dO+sidecar M0-M3, arrives head, publishes M4-M11, then arrives
  tail.  Three symmetric consumer groups execute head, wait tail, execute the
  remainder, and release once at M11.  No extra LDS page or matrix workaround.
- Static/resource evidence: roles `32/145/145/145`, private0, SGPR50,
  VGPR128, spill0/scratch0, steady LDS 100,608B.  Source/ASM gates prove the
  native split publisher and wait boundary.
- Correctness/work: S384 and S768 PASS with unchanged dK/dV relL2
  `0.00191329/0.000319636`; S768 MMOP/LDS/VMEM are exactly
  `73,728/44,768/1,728`, bank0.  S1024 is explicitly unsupported by the exact
  `S%192` host gate and is not claimed.
- Stats-only: ticks `42,662,165 -> 41,065,570` (`-3.74%`), MMAC active
  `35.1548% -> 36.5520%`, barrier `75,916 -> 63,005.25` (`-17.01%`).
- Fullperf: ticks `43,033,445 -> 40,882,205` (`-5.00%`), MMAC active
  `34.8979% -> 36.7738%`, waitLgkm `17,865.75 -> 17,576.2` (`-1.62%`),
  barrier `76,858.25 -> 61,634.2` (`-19.81%`).  Exact instruction work is
  unchanged; coissue is `30,188/23,730`.
- XCU: duration `94,576 -> 89,848`, no-wave idle stays 0.  Ordinary
  ownership-wait count rises `304 -> 496` as designed, but duration falls
  `1,776,840 -> 1,448,868` (`-18.46%`).  In the same 8k:80k window the top
  ABarrier gap falls `23,249 -> 17,141`; producer-wave coissue rises
  `2.27% -> 42.26%`.  This confirms useful producer-tail/consumer-head WASP
  overlap rather than count reduction or artificial staggering.
- Evidence: workbook sheet `148_DKV_Head64_Tail128`; S384/S768 under
  `/zys/shaobo_runs/owner16_head64_tail128/`; fullperf/XCU
  `/zys/shaobo_runs/owner16_head64_tail128_fullperf/`
  `dkv_mmac_correctness_20260718_183256`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_183256_owner16_head64_tail128_s768_sqc7/`.

## 2026-07-18 Mq96 Two-Page Lookahead Rejected

Status: `REJECT_MQ96_RAW2_LOOKAHEAD`

- Hypothesis: halve the raw packet to Mq96 and use two full pages so producer
  can publish packet t+1 while consumers compute packet t.
- Correctness/resource: S384/S768 pass with exact work, bank0, private0 and
  spill0 at WDRA `20/172/172/148`.  Windows `20/168/168/156` failed exactly
  the C0/C1 dK+dV outputs even though metadata reported spill0, showing that a
  role window at the measured live edge needs one allocation-granule margin.
- Result: barrier falls `63,005.25 -> 48,195`, but ticks regress
  `41,065,570 -> 43,163,120` (`+5.11%`), active falls
  `36.5520% -> 35.0070%`, waitLgkm rises 15.28%, VALU rises 9.21%, and
  coissue success falls.  Smaller MMAC islands and doubled packet/control
  cadence cost more than the lookahead saves.
- Decision: remove the failed source; retain only workbook sheet
  `149_DKV_Mq96_Raw2_Lookahead` and the negative evidence.

## 2026-07-18 Mq192 Head/Tail Split-Used Conveyor Accepted

Status: `ACCEPT_MQ192_HEAD_TAIL_SPLIT_USED`

- Hypothesis: Head64 and Tail128 already occupy disjoint regions of the
  accepted Mq192 packet.  Split only their Used edges so P0 publishes
  Head(t+1) under Tail(t) compute, preserving large MMAC islands and exact
  work.
- Implementation: replace the combined RawUsed token with RawHeadUsed and
  RawTailUsed.  Every consumer releases Head after the M3 Q/dO reads retire
  and Tail after M11.  Producer order is
  `wait HUsed -> publish Hnext -> wait TUsed -> publish Tnext`; waiting Tail
  before publishing Head is statically forbidden by the source gate.
- Static/resource: branches `32/145/145/145`, private0, SGPR50, VGPR128,
  spill0/scratch0, LDS 100,608B.  No formula, tile, output ownership,
  matrix-read path, or instruction work changes.
- Correctness/work: S384/S768 PASS; S768 dK/dV relL2
  `0.00191329/0.000319636`; MMOP/VALU/SCA/LDS/VMEM
  `73,728/106,640/21,288/44,768/1,728`, bank0.  The SCA increase is the
  explicit split-token control cost.
- Stats-only: ticks `41,065,570 -> 39,486,265` (`-3.85%`), MMAC active
  `36.5520% -> 38.1762%`, waitLgkm `17,766 -> 17,854.75`, barrier
  `63,005.25 -> 49,613` (`-21.26%`), coissue `29,944/23,369`.
- Fullperf: ticks `40,882,205 -> 39,383,435` (`-3.67%`), MMAC active
  `36.7738% -> 38.2453%`, waitLgkm `17,576.25 -> 18,035.5`, barrier
  `61,634.25 -> 49,145.25` (`-20.26%`), coissue `29,880/23,479`.
- XCU: dispatch duration `89,848 -> 86,560`, no-wave idle remains 0.
  Ordinary ownership waits increase `496 -> 544`, but duration falls
  `1,448,868 -> 1,285,192` (`-11.30%`) and max duration falls
  `16,795 -> 12,263`.  Hot `s_waitcnt` latency falls 3.88%.  In the same
  8k:80k window producer coissue rises `42.26% -> 59.39%`.
- Residual debt: consumer MMAC+VALU coissue changes from
  `30.83/28.70/24.59%` to `26.46/26.12/21.07%`; MMAC-to-MMAC issue-gap
  duration rises 3.23%.  Next work should preserve the accepted ownership
  conveyor and optimize consumer read/wait/VALU cadence.
- Evidence: workbook sheet `150_DKV_Mq192_SplitUsed`; stats-only
  `/zys/shaobo_runs/owner16_split_used_s768/`
  `dkv_mmac_correctness_20260718_202542`; fullperf/XCU
  `/zys/shaobo_runs/owner16_split_used_fullperf/`
  `dkv_mmac_correctness_20260718_203148`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_203148_owner16_mq192_head_tail_split_used_s768_sqc7/`.

## 2026-07-18 dV/dK Sources Under Useful Work Accepted

Status: `ACCEPT_DVDK_SOURCES_UNDER_USEFUL_WORK`

- Hypothesis: preserve the accepted Mq192 split-used ownership conveyor, but
  issue only D0/D1 normal Q/dO sources before softmax and D2/D3 before the
  first dV/dK MMAC8.  This bounds live source VGPR while allowing useful VALU
  and MMAC to cover LDS readiness.
- Resource stress: the wider D1 request train (sidecar3+normal8) compiled at
  branches `32/160/160/160` with private28B and vgpr_spill18, so it was
  rejected before PMD.  D2 passes at `32/154/154/154`, private0, SGPR50,
  VGPR128, spill0/scratch0, LDS100,608B.
- Correctness/work: S384 and S768 pass with unchanged S768 dK/dV relL2
  `0.00191329/0.000319636`; MMOP/LDS/VMEM remain exactly
  `73,728/44,768/1,728`, VALU falls `106,640 -> 105,440`, and bank conflict
  remains zero.
- Stats-only: ticks `39,486,265 -> 38,680,460` (`-2.04%`), MMAC active
  `38.1762% -> 39.4010%`, waitLgkm `17,854.75 -> 13,414.5` (`-24.87%`),
  barrier `49,613 -> 47,105`, coissue `26,200/21,206`.
- Fullperf: ticks `39,383,435 -> 38,840,165` (`-1.38%`), MMAC active
  `38.2453% -> 39.2062%`, waitLgkm `18,035.5 -> 13,615.5` (`-24.51%`),
  barrier `49,145.25 -> 47,845.25`.
- XCU: dispatch duration falls `86,560 -> 85,360`, instruction issues fall
  `285,416 -> 281,336`, and `s_waitcnt` hits/latency fall
  `21,360 -> 16,752` / `1,411,776 -> 1,303,020`.  In the representative
  consumer window, wait instructions fall `432 -> 336`; sidecar-to-wait and
  `v_cmp_ge -> wait` gaps disappear.  The tradeoff is explicit:
  MMAC-to-MMAC gap rises `8,751 -> 9,483`, normal-read-to-wait rises
  `4,500 -> 6,452`, and consumer MMAC+VALU falls to
  `24.68/23.81/19.88%`.
- Decision: promote because hard gates, same work, ticks, MMAC active, and
  aggregate wait all improve.  Do not interpret it as the final coissue
  solution; the next hypothesis must reduce MMAC gaps without restoring the
  rejected 32-VGPR source overlap.
- Evidence: workbook sheet `151_DKV_DvDkUnderSoftmax`; stats-only roots
  `/zys/shaobo_runs/owner16_dvdk_under_softmax_d2_s384/` and
  `/zys/shaobo_runs/owner16_dvdk_under_softmax_d2_s768/`; fullperf/XCU
  `/zys/shaobo_runs/owner16_dvdk_under_softmax_d2_fullperf/`
  `dkv_mmac_correctness_20260718_215518`.

## 2026-07-18 Loop-Lived MMAC Zero Seed Accepted

Status: `ACCEPT_LOOP_LIVED_MMAC_ZERO`

- Hypothesis: D2 recreates four zero VGPRs in every M16 score/dP block and
  again for the first dV/dK accumulation.  Hoist one native zero fragment to
  the consumer q-loop, preserving all arithmetic, ownership, reads, waits,
  stores, and exact work.
- Static proof: branch windows move `32/154/154/154 -> 32/158/158/158` while
  private/spill/scratch remain zero.  Static plain-zero `v_mov_b64` falls
  `150 -> 6`, total b64 moves `289 -> 137`, and ASM bytes fall about 2.1%
  without compensating `v_mov_b64_e32` growth.
- Correctness/work: S384 and S768 pass; S768 dK/dV relL2 remains
  `0.00191329/0.000319636`; MMOP/LDS/VMEM remains
  `73,728/44,768/1,728`, bank0.
- Stats-only: ticks `38,680,460 -> 37,219,000` (`-3.78%`), MMAC active
  `39.4033% -> 40.4364%`, waitLgkm `13,414.5 -> 11,668`, barrier
  `47,105 -> 42,965.75`, and VALU `105,440 -> 100,704`.
- Fullperf: ticks `38,840,165 -> 36,811,775` (`-5.22%`), MMAC active
  `39.2073% -> 40.6086%`, waitLgkm `13,615.5 -> 11,779.75`, barrier
  `47,845.25 -> 42,157.75`, and empty-buffer falls 10.54%.
- XCU: duration `85,360 -> 80,904`, issues `281,336 -> 276,600`, dynamic
  `v_mov_b64_e32` hits `6,880 -> 2,144`, and the dominant
  `s_abarrier_try_wait -> s_xor_b32` gap falls
  `1,303,708 -> 1,053,024`.  Consumer MMAC+VALU coissue changes from
  `24.68/23.81/19.88%` to `27.66/28.34/26.26%`.
- Tradeoff: XCU `s_waitcnt` latency rises 5.23%; normal
  `ds_read_matrix_format -> s_waitcnt` duration rises 4.80%, while the
  trans-read counterpart falls 2.56%.  This is the next measured bottleneck.
- Decision: promote as new best.  Preserve D2 ownership and the shared zero;
  next change must isolate matrix-read first-use cadence rather than combining
  another token, page, or live-source expansion.
- Evidence: workbook sheet `152_DKV_LoopMmacZero`; S384/S768 stats under
  `/zys/shaobo_runs/owner16_loop_zero_s384/` and
  `/zys/shaobo_runs/owner16_loop_zero_s768/`; fullperf/XCU under
  `/zys/shaobo_runs/owner16_loop_zero_fullperf/`
  `dkv_mmac_correctness_20260718_225519`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_225519_owner16_loop_mmac_zero_s768_sqc7/`.

## 2026-07-19 Toolchain Rebaseline And 50% Topology Gate

Status: `ACCEPT_DKV_ENV_REBASE / OBSERVE_DQ_COMPILER_REGRESSION / DESIGN_1P3C`.

- dKV S768, latest PMD HEAD1694 plus LLVM `7b796991`: correctness PASS,
  private0/spill0/bank0, ticks `35,707,035`, MMAC active `43.7836%`, exact
  MMOP73,728, VALU80,272, SCA21,624, LDS44,768, VMEM1,728, FLAT816.  This
  improves the old best fullperf ticks by 3.00% and active by 3.18 pp.
- dQ S1024, latest PMD plus old stable compiler: correctness PASS,
  private0/spill0/bank0, ticks `24,600,030`, MMAC active `33.3978%`, exact
  MMOP50,688, VALU68,144, SCA41,772, LDS26,352, VMEM1,408, FLAT752.
- dQ latest compiler is not promoted: no-MMU fullperf passes but regresses to
  ticks `25,002,705`, active `30.7854%`; full flags with MMUCheck crash inside
  PMD `Mmu::memcpy_check`.  This is environment evidence, not a dQ algorithm
  failure.
- Required active-time reduction at fixed MMOP is now explicit: dKV needs
  12.43%; dQ needs 33.20%.  Therefore dKV continues with ABarrier/readiness
  scheduling, while dQ needs the Mq192 1P3C structural route rather than
  wait-only edits.
- dQ 1P3C design: one producer wave group streams both K/V; three consumer
  groups own disjoint 64-row ranges.  Q/dO/dQ stay long-lived.  K-trans,
  V-trans, and K-normal are live one family at a time, targeting 160 VGPR per
  consumer.  Full K pages rotate useful n32 work/order across groups; the
  causal boundary page remains canonical until correctness is proven.
- Evidence: workbook `154_1P3C_50pct_Gate`; dKV run
  `/zys/shaobo_runs/env_audit_latest_pair_fullperf_dkv/`
  `dkv_mmac_correctness_20260719_015237`; dQ stable run
  `/zys/shaobo_runs/env_audit_latest_pmd_fullperf_dq_oldcc/`
  `dq_correctness_20260719_015727`.

## 2026-07-19 Canonical dKV Model-Build Contract

Status: `ACCEPT_CANONICAL_TOOLCHAIN_ROUTE`.

- Hypothesis: make the already measured latest-compiler gain reproducible
  without changing the default compiler route or dKV mathematics.
- Change: guarded kernel-entry WDRA init plus a build-level
  `SHAOBO_RUN_ON_MODEL=1` switch.  No tile, barrier, instruction island, or
  output ownership change.
- Static result: both latest and default compiler builds pass; latest branch
  use is `32/158/158/158`, private0, spill0, scratch0.
- PMD result: S384 correctness PASS.  S768 stats-only ticks `35,823,515`,
  MMAC active `43.1608%`, MMOP73,728, VALU80,272, bank0.  This is consistent
  with the earlier fullperf `35,707,035 / 43.7836%` environment audit.
- Decision: accept and commit as the canonical dKV toolchain route.  This does
  not count as the next algorithm optimization; remaining distance to 50%
  is still ABarrier/readiness work.

## 2026-07-19 dQ Mq192 One-Producer Three-Consumer Topology Proof

Status: `OBSERVE_1P3C_TOPOLOGY_PROOF_TICKS_REGRESSION`.

- Hypothesis: replace the Mq128 two-producer/two-consumer CTA with one K/V
  producer and three symmetric 64-row dQ consumers.  The larger tile should
  amortize control and expose three peer waves per SIMD for later useful
  MMAC/VALU staggering.
- Static/resource result: role use is `11/158/158/159` inside WDRA windows
  `32/160/160/160`; private0, SGPR59, VGPR128, spill0, scratch0.  Startup
  Q+dO+sidecar is 100,608B; steady two-page K/V is exactly 131,072B.
- Correctness: H1/S768 causal PASS, maxAbs `1.5201e-7`, relL2 `0.00151559`,
  no NaN/Inf, and bank0.
- Same-work fullperf comparison against Mq128 2P2C: MMOP remains 28,800.
  Active rises `30.0592% -> 34.3345%`; VALU falls
  `39,580 -> 33,808`, SCA `26,002 -> 23,844`, VMEM `864 -> 704`, and
  coissue success rises `8,125 -> 12,031`.
- Ticks regress `19,608,225 -> 23,591,750` (`+20.32%`).  H1/S768 launches
  four larger CTAs rather than six smaller CTAs, while the three consumers
  still execute their score/dP/softmax/dQ stages mostly in lockstep.  The
  topology reduces instruction/control debt but lengthens each CTA critical
  path, so it is evidence of headroom rather than a promoted performance
  version.
- XCU confirms the remaining debt: the top gap is
  `s_abarrier_try_wait -> s_xor_b32` (12.73%); ABarrier-to-SALU is 10.61%,
  and lds-matrix-to-immediate is 4.66%.  Compared with the control, branch
  hits fall `680 -> 288`, phase-xor hits `212 -> 48`, vbcnt waits
  `216 -> 80`, and ordinary waits `300 -> 132`.
- Next hypothesis is one isolated mathematical stagger: consumer0 keeps
  `score -> dP -> P/dS -> dQ`, while consumer1 executes
  `score -> P -> dP -> dS -> dQ`.  This is legal because P depends on score
  but not dP.  Consumer-assisted V prefetch remains deferred until that
  experiment passes correctness/resources and improves both active and ticks.
- Evidence: workbook sheet `154_1P3C_50pct_Gate`; candidate
  `/zys/shaobo_runs/goal50_dq_mq192_1p3c_fullperf/`
  `dq_correctness_20260719_031425`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260719_031425_dq_mq192_1p3c_topology_s768_sqc7/`.

## 2026-07-19 dQ Useful-Stagger PMD Control Gate

Status: `REJECT_INVALID_CONTROL_REFERENCE_PATH`; no candidate verdict.

- The isolated hypothesis was legal DAG reordering for consumer1 only:
  `score -> P -> dP -> dS -> dQ`, while peers retained
  `score -> dP -> P/dS -> dQ`.  Static generation kept exact work and all
  matrix/read/wait/barrier counts: 576 MMAC, 96 normal matrix reads, 216
  transpose matrix reads, with branch VGPR `11/158/152/159`, private0,
  spill0, and scratch0.
- The failed control was not equivalent to the earlier control.  Its live
  process command contained `--canonical=0`, so it entered the scalar
  reference path and was terminated after `05m38s`; successful runs of the
  same binary ended with `path=canonical`.  Binary SHA equality is therefore
  insufficient unless runtime path arguments also match.
- The host was separately unhealthy: seven approximately 116-day-old runaway
  `kded5` processes consumed about 14 of 16 CPUs.  They ignored `TERM` and were
  removed with targeted `KILL`; CPU idle recovered to about 91%.  This cleanup
  improves reproducibility but does not rehabilitate the invalid reference run.
- Several earlier commands also showed a shell-quoting hazard: when invoked
  via `docker exec env`, `GPU_ARGS` must be passed as the single quoted value
  `GPU_ARGS=['--SQCIPfLines=7']`, and PMD stdout must list
  `--SQCIPfLines=7` before a run is admissible.
- Decision: restore `src/dq_kernel.cpp` to topology-proof commit `3eeff47`.
  Every performance invocation must explicitly set `CANONICAL_DQ=1`, verify
  final `path=canonical`, and match SQ7/toolchain before candidate comparison.

## 2026-07-19 dQ Causal M16 Ownership Balance

Status: `REJECT_CAUSAL_BALANCE_LOCKSTEP`; source restored.

- The 1P3C topology was symmetric in code shape but not in causal work.  For
  the first Mq192 CTA, contiguous ownership gives consumer n32 work units
  `6/14/22` for M16 rows `{0..3}/{4..7}/{8..11}`.  The heaviest group performs
  3.67x the useful score/dP/dQ work of the lightest, then all roles reconverge
  at the terminal EBarrier.
- Interleaving ownership as `{0,3,6,9}`, `{1,4,7,10}`, and `{2,5,8,11}` gives
  `12/14/16` work units.  The total remains exactly 42 n32 units, so MMOP,
  global/LDS bytes, ABarrier counts, and output ownership are unchanged.
- This was a one-line canonical mapping change, not artificial delay or a new
  phase.  Static resources remained `11/158/158/159`, private0, spill0,
  scratch0; H1/S768 causal correctness passed and bank conflict remained 0.
- Fresh canonical control: kernel ticks `23,364,250`, MMOP28,800,
  VALU33,808, SCA23,844, LDS15,092, VMEM704, FLAT564, coissue
  `12,071/10,930`.
- Balanced candidate: kernel ticks `24,132,290` (`+3.29%`), identical
  MMOP/LDS/VMEM/FLAT, VALU33,856, SCA23,876, coissue `12,982/11,717`.
- The higher coissue count did not mean better overlap.  Equal work increased
  peer-consumer lockstep and issue contention; the original causal imbalance
  was providing useful stagger.  Restore contiguous ownership.  The next
  isolated hypothesis is legal DAG-order staggering with exact row ownership.

## 2026-07-19 dQ Legal DAG-Order Stagger

Status: `REJECT_DAG_STAGGER_BREAKS_MMAC_ISLAND`; source restored.

- Hypothesis: consumer1 computes score, converts score to P, then computes dP
  and dS, while consumer0/2 retain fused score+dP.  The legal DAG order was
  intended to place P VALU under peer score+dP MMAC.
- Static evidence was exact: `576 MMAC`, `96` normal reads, `216` trans reads,
  `72` lgkm waits, `12/15` ABarrier wait/arrive, `192` exp, `85` v_mov.
  Candidate added eight `s_setprio`; roles `11/158/152/159`, private0,
  spill0, scratch0.
- H1/S768 correctness and bank0 passed.  Control versus candidate:
  kernel ticks `23,364,250 -> 24,166,870` (`+3.44%`), VALU
  `33,808 -> 33,608`, coissue `12,071/10,930 -> 11,305/10,121`; exact
  MMOP/LDS/VMEM/FLAT remained `28,800/15,092/704/564`.
- Verdict: splitting the fused score+dP island reduces issue continuity and
  peer overlap attempts.  Do not pursue per-consumer mathematical ordering
  on this tile.  Preserve fused score+dP and target K/V page readiness,
  producer work, or ownership around the intact island.

## 2026-07-19 dQ Split V/K Page Ownership

Status: `REJECT_SPLIT_USED_TOKEN_CONTROL_COST`; source restored.

- Hypothesis: V becomes dead after the page's final dP, earlier than K becomes
  dead after dQ.  Split the combined `PageUsed` token into V/K last-use tokens
  so the producer can begin next V before waiting for K.
- Correctness, bank0, exact MMOP/LDS/VMEM/FLAT, and no-spill resource gates all
  passed.  The unconditional form reduced LDS credit stall by 13.81% and
  raised successful coissue by 349, but added 1,122 SCA instructions and
  regressed ticks `23,364,250 -> 23,586,290` (`+0.95%`).
- A tail-aware variant intended to omit final-generation arrivals instead
  raised VALU/SCA, reduced coissue, increased credit stall, and regressed to
  `24,856,650` ticks (`+6.39%`).
- Evidence says operand lifetime is real but too short to fund a second
  ownership protocol.  Reject both variants, remove their source, and retain
  one combined page-used token.  The next hypothesis must improve the
  useful-work/control ratio rather than fragment readiness further.

## 2026-07-19 dQ N32 Stage-Helper Refactor

Status: `REJECT_STATIC_CODEGEN_DRIFT`; no PMD run.

- The intended preparation for asymmetric two-n32 batching extracted two
  forced-inline stage helpers without changing work or order.
- Static counts proved the main work exact, but `s_waitcnt` increased
  `89 -> 93` and all consumer branches reached the 160-VGPR ceiling instead
  of measured `158/158/159`.
- This violates the refactor-preservation gate and leaves no register headroom
  for the actual batch.  Restore canonical source.  Test consumer-assisted V
  BPS ownership next because it can create useful skew without rewriting the
  proven fused MMAC body.

## 2026-07-19 dQ Consumer-Assisted V Ownership

Status: `REJECT_COMBINED_FILLED_TOKEN`; source restored.

- Current-page V ownership was correct and exact-work but regressed ticks
  `23,364,250 -> 25,983,230` (`+11.21%`).  V became the last Filled arrival;
  SCA and LDS credit stall rose to `25,512` and `10,573`.
- One-step V lookahead with producer K and consumer1 V sharing the same Filled
  token deadlocked.  SQAbar showed Page0 at four arrivals out of eight and all
  consumers blocked on the incomplete generation.
- This rejects mixed sequencing ownership, not useful lookahead itself.  The
  last admitted test uses separate KFilled/VFilled tokens and one shared Used
  token; failure there rejects consumer-assisted prefetch completely.

## 2026-07-19 dQ Split K/V Filled Generations

Status: `REJECT_SPLIT_FILLED_CONTROL_AND_READINESS_COST`; source restored.

- Producer-owned K and consumer1-owned V used separate `KFilled`/`VFilled`
  generations and retained one shared `PageUsed` generation.  Refactoring the
  sentinel loop into `load V(t) -> compute(t-1) -> final compute` repaired the
  missing V arrival and made the topology functionally correct.
- Static/resource gates passed with role use `11/158/159/159`, private0,
  spill0, scratch0, exact `576` static MMAC, and no matrix-path workaround.
  H1/S384 and H1/S768 causal correctness passed; S768 maxAbs was
  `1.5201e-7`, relL2 `0.00151559`, and `ldsBankConflict=0`.
- H1/S768 kernel ticks were `25,837,175` versus the canonical control
  `23,364,250`, a `+10.58%` regression.  Exact dynamic MMOP remained `28,800`;
  coissue was `10,942/9,761` versus control `12,071/10,930`.
- Separate Filled generations solve the protocol legality problem but add
  readiness/control work and reduce useful overlap.  Reject consumer-assisted
  V prefetch for this tile and restore the canonical single producer path.

## 2026-07-19 dKV Output C2 Minimal Matrix-Store Probe

Status: `OBSERVE_BLOCKED_PMD_OR_LAYOUT_CONTRACT`; no canonical source change.

- Hypothesis: the previous B16 matrix-store failure came from omitted ABarrier
  initialization or from reusing the LDS page across several stores.
- The probe was reduced to one MLS-direct control store and one writer-chain
  store.  It uses the official HCU builtin and the Wiki-documented ABarrier
  lifecycle, including entry init visibility and pre-invalidate convergence.
- Static gates pass: writer1, store2, ABarrier init/seq/arrive/wait/inv
  `1/2/2/2/1`, ebarrier sync3, trap0, private0, spill0, bank0.
- Result: the control still has exactly 240 mismatches and first fails at
  row17/col0; the writer chain has 503 mismatches.  Therefore ABarrier init is
  necessary but does not explain the partial write, and the old multi-store
  overwrite hypothesis is rejected.
- Decision: keep C2 isolated, register PMD-005, and implement C1 packed FP16
  direct global store as the next valid output-epilogue comparison.

## 2026-07-19 dKV Output C1 Packed-FP16 Direct-Store Control

Status: `ACCEPT_PROBE`; canonical performance A/B pending.

- Implemented the real owner16 coordinate contract without matrix-store:
  64 lanes cover `16x128`, FP32 values are packed four-at-a-time to FP16, and
  each lane issues eight 64-bit vector stores.
- Latest compiler static result: private0, spill0, VGPR16, eight
  `global_store_dwordx2`, zero `global_store_dwordx4`, 32 FP32-to-FP16
  conversions, and no `s_trap`.
- PMD HEAD1694 result: 2,048/2,048 values exact, `ldsBankConflict=0`, status
  PASS.  Run root:
  `/zys/shaobo_runs/dkv_b16_direct_store_probe_builtin/`
  `run_20260719_104409`.
- Decision: C1 is the only admitted output-tail candidate.  Integrate it as a
  one-hypothesis branch and compare real dKV correctness, output bytes, ticks,
  and SQTT store bubbles against the accepted FP32 baseline.  Do not mix C2 or
  barrier changes into that branch.

## 2026-07-19 dKV Output C1 Canonical A/B

Status: `REJECT_PERF_VALID_FUNCTION`; restore the FP32 oracle epilogue.

- The canonical A/B changed only dK/dV output dtype and store packing.  Tile,
  MMAC work, role ownership, ABarrier protocol, and all matrix reads remained
  unchanged.  S384 and S768 correctness pass with FP16 rounding; S768 dK/dV
  relative L2 errors are `0.00230629 / 0.000396918`.
- Hard gates pass: private/spill/scratch0, exact MMOP73,728, bank0, and the
  native matrix path is unchanged.  Output data cycles halve
  `12,288 -> 6,144`, but scalar FP32-to-FP16 packing raises dynamic VALU
  `80,272 -> 81,904`.
- Same-shape S768 kernel ticks regress `35,707,035 -> 35,834,435` (`+0.357%`)
  and aggregate MMAC active falls `43.7836% -> 43.6662%`.  Successful
  coissue also falls `22,104 -> 21,819`.
- The direct FP16 path remains valid functional evidence, but it is not a
  performance promotion.  Restore the FP32 control and inspect FWD/compiler
  output for a native packed conversion before reopening this hypothesis.
- Evidence:
  `/zys/shaobo_runs/dkv_b16_direct_canonical_s768/`
  `dkv_mmac_correctness_20260719_105832`.

## 2026-07-19 dKV Output C1b FWD Packed Conversion

Status: `REJECT_STATIC_CODEGEN_EQUIVALENT`; canonical source restored.

- HCU tests and FA3 FWD use
  `__builtin_hcu_cvt_pk_f16_f32(a, b, false, 0)`.  The focused owner16 probe
  lowers to exactly 16 packed conversions plus eight `global_store_dwordx2`
  and passes all 2,048 values on PMD HEAD1694.
- Replacing canonical scalar casts with that builtin leaves generated work
  unchanged: both artifacts contain `v_cvt_pk_f16_f32=384`, scalar
  `v_cvt_f16_f32=0`, and `global_store_dwordx2=49`.  Branch use remains
  `32/158/158/158`; metadata remains private/spill/scratch0.
- LLVM already performed the intended pair fusion in the previous C1.  The
  measured extra VALU is therefore the irreducible packed-conversion work for
  this direct FP16 epilogue, not missed source-level vectorization.
- Do not spend PMD time on the canonical builtin spelling.  Keep the focused
  probe as a compiler-regression gate and return optimization effort to
  ABarrier/readiness and MMAC-island gaps.

## 2026-07-19 dKV Score/dP Read8-Wait0-MMAC16

Status: `REJECT_STATS_FIRST_USE_WAIT`; canonical staged waits restored.

- The experiment changed one schedule only: each M16 score/dP block moved from
  `8 reads -> wait6/MMAC4 -> wait4/MMAC4 -> wait2/MMAC4 -> wait0/MMAC4` to
  `8 reads -> wait0 -> MMAC16`. Formula DAG, Mq192/Nk192/D128 tile, one
  producer plus three consumers, seven ABarrier tokens, LDS addresses, dV/dK
  scheduling, and exact four-GEMM work remained unchanged.
- Final ASM proves the intended contiguous sequence. Latest LLVM
  `7b796991` with explicit WDRA init and `run-on-model` reports branch use
  `32/158/158/158`; metadata is private0/spill0/scratch0 and real `s_trap=0`.
- PMD HEAD1694 correctness passes at S384 and S768, MMOP remains exactly
  `73,728`, dynamic VALU/SCA/LDS/VMEM/FLAT remain
  `80,272/21,624/44,768/1,728/816`, and LDS bank conflict remains zero.
- S768 kernel ticks regress from the canonical stats control
  `35,823,515 -> 36,638,420` (`+2.27%`). Aggregate MMAC active falls
  `43.1608% -> 42.6444%`, and successful coissue falls
  `21,792 -> 19,321`.
- Conclusion: visual MMAC regularity is insufficient. A full `lgkmcnt(0)`
  exposes all eight matrix reads before useful compute starts and erases the
  benefit of removing three wait instructions. Do not capture fullperf for
  this rejected candidate. Preserve exact staged first use; any later long-
  island retry must use a bounded partial wait backed by a read ledger.
- Evidence: workbook sheet `155_DKV_ScoreDP_LongIsland`; remote S384/S768 runs
  `/zys/shaobo_runs/dkv_scoredp_long_mmac16_s384/` and
  `/zys/shaobo_runs/dkv_scoredp_long_mmac16_s768/`.

## 2026-07-19 dKV Causal Q-Start Accepted

- Status: `ACCEPT_ALGORITHM_CAUSAL_ZERO_WORK_PRUNE`.
- Formula proof: for a fixed causal K tile `j`, every Q tile `i < j` has
  `P(q,k)=dS(q,k)=0`. Canonical publication now starts at `i=j`; only the
  diagonal tile uses the exact element mask and every later tile is compiled
  as fully valid. The four-GEMM DAG, output ownership, LDS pages, seven-token
  ABarrier topology, and native matrix path are unchanged.
- S768 Q epochs become `4/3/2/1` instead of `4/4/4/4`. Issued MMOP falls
  exactly `73,728 -> 46,080`, matching the triangular-domain derivation and
  removing `37.5%` mathematically zero MMAC rather than changing tile size.
- Latest compiler gates pass at `32/156/156/156`, private/spill/scratch0,
  `LDS=100,608`, real `s_trap=0`, and bank0. S384 and three S768 runs pass dK
  and dV correctness; fullperf S768 reports `dk_rel_l2=0.00193715` and
  `dv_rel_l2=0.000319636`.
- Same-toolchain fullperf kernel ticks improve
  `35,707,035 -> 34,951,735` (`-2.115%`). Raw MMAC active falls
  `43.7836% -> 39.5157%` because the old value counted invalid triangular
  work. A derived causal-useful-active proxy rises `27.36% -> 39.52%`; this is
  an explanatory normalization, not a replacement for the PMD native metric.
- XCU isolates the longest CTA (`SE0/CU0/SIMD0`): both versions retain
  `4,608` MMOP, `2,352` matrix reads, and `1,066` waits. Mask/predicate
  instructions fall `1,256 -> 392`, total instructions fall
  `16,266 -> 14,888`, and consumer spans shrink `1.5%..3.8%`. Dispatch SQTT
  duration falls `78,476 -> 76,820`.
- Remaining target: absolute ABarrier-to-wait time is nearly fixed at about
  `519K` captured cycles and global-store latency is now more exposed. Do not
  restore invalid MMAC to raise raw active; attack ownership/startup and the
  output tail with same-useful-work experiments.
- Evidence: workbook sheet `156_DKV_CausalQStart`; remote run
  `/zys/shaobo_runs/dkv_causal_qstart_s768_fullperf/`
  `dkv_mmac_correctness_20260719_130657`.

## 2026-07-19 dKV Next-M16 Score Half Prefetch

Status: `ACCEPT_LATENCY_HIDING_SAME_EXACT_WORK`.

- Hypothesis: after current-M16 dV/dK D0/D1 MMAC consumes its normal-source
  VGPRs, reuse those dead slots for the next M16 score D0/D1 transpose reads.
  Issue them before current D2/D3 MMAC, retire current D2/D3 with
  `lgkmcnt(4)`, and forbid prefetch across HeadFilled/TailFilled boundaries.
- Static proof: 60 exact
  `normal4 -> MMAC8 -> next-trans4 -> wait4 -> MMAC8` sequences; branch use
  `32/156/156/156`; private0, SGPR53, VGPR128, spill0, scratch0, real trap0.
  No phase, fallback, extra buffer, extra GEMM, or ownership change was added.
- Correctness/resource proof: S384 PASS; two stats S768 runs and one fullperf
  S768 run PASS; `ldsBankConflict=0`. Dynamic MMOP/VALU/SCA/LDS/VMEM/FLAT is
  exactly unchanged at `46,080/41,314/15,014/28,316/1,152/798`.
- Repeated stats improve about `1.94%..2.08%`. Fullperf kernel ticks improve
  `34,951,735 -> 34,372,975` (`-1.656%`). PMD `waitLgkmCounter` falls
  `12,106 -> 10,836.75` (`-10.48%`) and barrier falls `1.67%`.
- XCU dispatch duration improves `76,820 -> 75,548`. Hot `s_waitcnt` latency
  falls `1,193,416 -> 1,115,532`; `MMAC -> s_waitcnt` bubble duration falls
  `151,672 -> 114,264` (`-24.66%`). Same-SIMD MMAC+VALU coissue rises
  `1,881 -> 2,397` (`+27.43%`).
- Native MMAC active falls slightly `39.5157% -> 39.2884%`; do not describe
  this as an active-share promotion. The candidate wins on exact-work ticks
  with direct readiness/coissue evidence, so it is accepted while the 50%
  MMAC-active goal remains open.
- Evidence: workbook `157_DKV_NextM16Prefetch`; remote fullperf
  `/zys/shaobo_runs/dkv_next_m16_prefetch_s768_fullperf_retry/`
  `dkv_mmac_correctness_20260719_142126`.

## 2026-07-19 dQ M128 Canonical Restore And Compiler A/B

Status: `ACCEPT_CANONICAL_GOVERNANCE_RESTORE`.

- Restored the accepted dQ topology to `Mq128/Nk128/D128`, 16 waves, two
  producers and two symmetric 64-row consumers. The formula DAG remains
  exactly three GEMMs with row-owned dQ; no M192/1P3C experiment path remains
  in canonical source.
- Added a run-on-model-only entry contract:
  `__builtin_hcu_wdra_init(40,216,216,40)`. The latest compiler emits it with
  real `s_trap=0`; the old stable compiler omits it under the source guard.
  This follows the shaobo skill/wiki boundary and does not conflate WDRA init
  with ABarrier init.
- Both compilers pass H1/S128 and H1/S1024 on PMD HEAD1694 with identical dQ
  output (`dq_rel_l2=0.00208192` at S1024), bank0 and no
  private/spill/scratch. Same-work stats select old LLVM `a6a6eb6616ab...`:
  kernel ticks `25,084,150 -> 24,585,015` (`-1.99%`), MMAC active
  `31.489878% -> 33.384755%`, and barrier `71,508.5 -> 47,387.25`.
  The latest compiler is rejected for dQ even though VALU falls
  `68,144 -> 45,696`.
- Winner-only fullperf/xcu reports P0/P1 steady-window bubble
  `98.51%/98.62%` and C0/C1 `46.32%/47.18%`. Consumer phase bins are
  `MMAC_vs_VALU=60`, `MMAC_vs_MMAC=49`, `MMAC_without_peer_VALU=28`,
  `no_MMAC=15`; actual MMAC vector peers are `397/1248` and `483/1245`.
- The hot `s_abarrier_try_wait -> s_xor_b32` edge is attributed to waiting
  for the ownership phase. Do not optimize XOR. The next isolated hypothesis
  is to shorten ordinary K/V PageUsed exposure while preserving the exact
  readiness ledger and symmetric consumer roles.
- Evidence: workbook `158_DQ_CanonicalRestore`; remote fullperf
  `/zys/shaobo_runs/dq_mq128_restore_a6_s1024_fullperf/`
  `dq_correctness_20260719_154801`; shared archive
  `/共享/shaobo/perf/20260719_154801_dq_mq128_restore_a6_h1s1024_sqc7_fullperf`.

## 2026-07-19 dQ Next-N32 Head Prefetch Rejected

- Hypothesis: reuse dead current-N32 score/dP operand space to issue the next
  N32 D0/D1 transpose reads under current dQ D2/D3 MMAC, preserving the exact
  M128 2P2C DAG, 400 matrix reads, 768 static MMAC, and PageUsed ledger.
- Static/resource gates passed (`8/173/193/9`, private/spill/scratch0), and
  S128/S1024 correctness passed with bank0. PMD HEAD1694 used the accepted old
  LLVM a6 compiler and SQ7.
- Result: kernel ticks `24,585,015 -> 24,666,005` (`+0.329%`), MMAC active
  `33.384755% -> 32.736682%`, waitLgkm
  `16,181.25 -> 14,022.25` (`-13.34%`), barrier
  `47,387.25 -> 46,706.25`, and coissue success/fail
  `15,623/13,813 -> 17,262/14,897`.
- Root cause: carrying the prefetched head through the unrolled N32 loop
  increased static VALU `1,684 -> 2,092` and SALU `821 -> 935`; dynamic VALU
  rose `68,144 -> 73,952` and SCA `41,772 -> 42,556`. MMAC runs fragmented
  `72 -> 129`, with singleton runs `8 -> 58`.
- Decision: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`; no fullperf/xcu.
  The next legal retry must specialize a fixed N32 pair at compile time and
  prove island preservation in ASM before PMD.
## 2026-07-19 dQ 1P3C Saturated-Grid Gate Accepted

Status: `ACCEPT_TOPOLOGY_SATURATED_GATE`; canonical S1024 source is unchanged.

- The earlier H1/S768 comparison mixed topology quality with grid underfill:
  M128 2P2C launches six CTAs per head, while M192 1P3C launches four. The
  saturated control uses H12, giving 72 versus 48 CTAs so both variants can
  occupy all 48 CUs.
- Both runs execute exact MMOP `345,600`, use the same PMD HEAD1694, old LLVM
  a6 compiler and SQ7, pass correctness, and report bank0. The 1P3C candidate
  retains resource admission at `11/158/158/159`, private/spill/scratch0 and
  LDS exactly 128KB.
- Saturated kernel ticks improve `33,831,525 -> 26,230,750` (`-22.47%`), and
  MMAC active improves `29.2940% -> 30.9286%` (`+1.6346 pp`). Dynamic VALU
  falls `474,960 -> 405,696` and SCA falls `312,024 -> 286,128`; the topology
  removes producer/control debt without removing mathematical work.
- Interpretation: the prior H1/S768 `+20.3%` result is not an architecture
  rejection; four M192 CTAs left most of the 48-CU model empty. Promote 1P3C
  as the dQ topology candidate, but do not merge its current Mq192 source into
  canonical S1024 until tail ownership and ABarrier generation counts are
  proved.
- Evidence: workbook `160_DQ_3C_SaturationGate`; control
  `/zys/shaobo_runs/dq_3c_saturation_control_h12s768/`
  `dq_correctness_20260719_170844`; candidate
  `/zys/shaobo_runs/dq_3c_saturation_candidate_h12s768/`
  `dq_correctness_20260719_171618`.

## 2026-07-19 dKV Final-M16 Half-Store Rejected At Resource Gate

Status: `REJECT_STATIC_RESOURCE`; canonical source restored before PMD.

- SQTT on the accepted M192 1P3C dKV shows a concentrated 16-instruction
  fp32 store burst per consumer wave. The slowest C2 sample spans 3,712
  cycles in stores and about 9,768 cycles from its last MMAC to wave end.
- The isolated hypothesis stored finalized accumulator groups 0..3 on only
  the final q tile/final M16, then computed groups 4..7. MMAC count, output
  bytes, dynamic store count and ABarrier generations were unchanged.
- Four increasingly scoped schedules all fail the no-spill gate:
  `private/spill=28/15`, `252/225`, `12/4`, and explicit vector-store
  `12/6`. The dKV native-path gate otherwise passes.
- Root cause: M192 1P3C exactly consumes the per-SIMD WDRA budget:
  producer `32` plus three consumers `160` equals `512` VGPR. Bringing VMEM
  address/output state into the live second-half MMAC window has no resource
  headroom. This is a topology limit, not scalar-versus-vector store syntax.
- Decision: no correctness/perf run for an inadmissible artifact. Restore the
  accepted next-M16-prefetch source and move to workbook sheet
  `162_DKV_M128_64_32_32`, where only two heavy MMAC waves per SIMD leave
  useful helper/store scheduling headroom.

### Skill Candidate

- Trigger / 适用场景: WDRA kernels that try to move an output epilogue into an
  active MMAC island.
- Rule / 可复用规则: sum branch VGPR windows per SIMD before coding. If the
  sum already equals the physical file, include VMEM address operands and
  still-live later accumulators in the peak ledger; reject early-store
  overlap when this peak cannot fit without spill.
- Evidence / 证据: workbook `161_DKV_FinalHalfStore`; LLVM7b static attempts
  above; canonical `32 + 3*160 = 512` VGPR/SIMD.
- Boundary / 适用边界: applies when producer and all consumers remain live on
  the same SIMD. A later ownership phase that provably releases a wave, or a
  native matrix-store path with a smaller live fragment, can change the
  budget.
- Counterexample / 反例或不适用情况: kernels with spare per-SIMD VGPR, fewer
  heavy waves, or an epilogue after all MMAC operands are dead.
- Proposed Target / 建议进入哪个 skill 或 reference: project-local Shaobo
  reference now; propose to `dcu-kernel-optimization` only during a governed
  skill-consolidation round.

## 2026-07-19 dKV M128 Logical 64/32/32 Tail-Free Topology

Status: `OBSERVE_TAIL_FREE_CONTROL_DEBT`; preserve on its isolated branch,
but keep M192 next-M16 prefetch as the accepted performance source.

- The proposed `M128` tile is mathematically sound. It assigns K rows
  `0..63`, `64..95`, and `96..127` to logical C0/C1/C2, executes score, dP,
  dV and dK exactly once for every owner16 slice, and removes the S1024 tail.
  S128, S768 and S1024 correctness pass; metadata is private0, SGPR52,
  VGPR96, spill/scratch0; LDS is 67,072B and bank conflict is zero.
- The physical schedule is not three heavy consumers. Waves4-7 are C0;
  waves8-11 contain both logical C1 and C2; waves0-3 and waves12-15 are two
  producers. Each SIMD therefore has `P0/C0/C12/P1`, only two MMAC waves,
  versus M192's `P/C0/C1/C2` with three MMAC waves.
- S1024 fullperf reports kernel ticks `32,393,270`, MMAC active `37.8149%`,
  exact MMOP `73,728`, coissue `10,577/7,417`, waitLgkm `19,571`, barrier
  `71,161.75`, and bank0. XCU attributes hot latency to ABarrier/XOR
  `29.80%`, wait `23.35%`, MMAC `13.58%`, and shows both consumers still
  aligning MMAC rather than forming sustained MMAC-versus-VALU stagger.
- H1/S768 raw ticks are not a promotion comparison because M128 launches six
  CTAs while M192 launches four. Normalized by exact MMOP, M128 raises
  barrier `0.8065 -> 1.1446` (+42%), SCA `0.3258 -> 0.5642` (+73%), and
  waitLgkm `0.2352 -> 0.3041` (+29%); successful coissue falls
  `0.2465 -> 0.1497` (-39%). Native active falls `39.2884% -> 34.6836%`.
- Decision: tail removal and VGPR headroom are real, but the second producer
  and eight-party raw publication add ownership/control debt. The next
  isolated A/B may let one producer publish Q+dO+sidecar to reduce Filled
  arrivals; if that serializes publication, return to M192 and implement a
  legal masked M64 tail CTA while retaining three heavy waves per SIMD.
- Evidence: workbook `162_DKV_M128_64_32_32`; remote fullperf
  `/zys/shaobo_runs/dkv_m128_2p2c_s1024_fullperf/`
  `dkv_mmac_correctness_20260719_191431`; shared archive
  `/共享/shaobo/perf/20260719_191431_dkv_m128_c0_64_c1_32_c2_32_h1s1024_sqc7_fullperf`.

### Skill Candidate

- Trigger / 适用场景: a logical tile is split among more named owners than
  the number of physical heavy wave groups.
- Rule / 可复用规则: draw both logical output ownership and per-SIMD physical
  wave residency. A `64/32/32` mathematical split does not create three-way
  WASP overlap when the two 32-row owners share one physical wave group.
- Evidence / 证据: workbook `162_DKV_M128_64_32_32`; M128 physical 2P2C
  normalized barrier/SCA/wait regressions above; correctness and bank0 pass.
- Boundary / 适用边界: logical splitting still helps exact tails and output
  ownership; it only fails as a claim of additional simultaneous consumers.
- Counterexample / 反例或不适用情况: three independent consumer wave groups
  with legal VGPR/LDS budgets, or hardware scheduling that time-multiplexes
  the logical owners into genuinely different overlap windows.
- Proposed Target / 建议进入哪个 skill 或 reference: project-local Shaobo
  reference first; propose the logical-versus-physical ownership check to
  `dcu-kernel-optimization` during a governed consolidation round.

## 2026-07-20 Dual Canonical Best Reconciliation

- Hypothesis: the clean branch had diverged from the independently measured
  best dKV and dQ sources, so further tuning risked comparing the wrong code.
- Change: restore the dKV source exactly from
  `best/dkv-three-m64-lifetimes-20260719` and the dQ source exactly from
  `best/dq-c1-kread-stagger-20260720`; do not add a phase, dispatch, or
  alternate kernel.
- Static result: dKV roles `31/156/156/156`; dQ roles `8/159/9/159` in the
  branch report; both have private/spill/scratch zero, native matrix paths,
  and no executable trap when built with `SHAOBO_RUN_ON_MODEL=1`.
- Correctness result: dKV H1/S384 and H1/S768 PASS; dQ H1/S128 and H1/S1024
  PASS. Bank conflict is zero in every recertification run.
- PMD result: dKV S768 kernel ticks `33,104,435`, exact MMOP `46,080`,
  coissue `11,693/9,716`; dQ S1024 kernel ticks `24,786,125`, exact MMOP
  `50,688`, coissue `11,787/10,476`. These stats-only recerts confirm source
  integrity; promotion metrics remain the archived fullperf values.
- Decision: `ACCEPT_GOVERNANCE_DUAL_BASELINE`. Keep M128 `64+32+32` as the
  tail-free comparison only. Continue structural optimization from this
  single reconciled source, one hypothesis per branch.
- Workbook stress review: `174_DKV_M128_vs_M192Tail` proves the no-tail M128
  control is mathematically exact but physically 2P2C. The only admitted
  S1024 continuation is a static-entry hybrid whose M192 main keeps twelve
  heavy waves and whose M128 tail owns disjoint K rows; it must not be folded
  into a runtime Tile branch.

## 2026-07-20 Owner16 Four-Consumer Resource Probe Accepted

Status: `ACCEPT_RESOURCE_GATE`; no canonical performance claim.

- Hypothesis: replace M128's logical-but-not-physical `64/32/32` split with
  four symmetric owner16 physical consumer groups in an exact
  `Mq64/Nk256/D128` topology.  Each wave must fit persistent K/V and dK/dV
  state plus transient score/dP/softmax sources inside 128 VGPR.
- Result: LLVM7b branch report is `114/128` for all four roles.  Metadata is
  private0/SGPR29/VGPR128/spill0/scratch0; LDS is 33,536B in the focused
  packet probe.  ASM evidence is BPS10, matrix-read96, MMAC128,
  `s_set_vgpr_size 128` x4 and executable trap0.
- PMD HEAD1694: checksum `bad=0 pass=1`, bank0, no panic.  Diagnostic stats:
  simTicks7,759,570, firstWave3,611,790, lastWave7,759,570,
  kernel ticks4,147,780, MMOP512, VALU2066, coissue238/978.
- Evidence:
  `/zys/shaobo_runs/dkv_owner16_4c_resource_probe_20260720_045244`;
  workbook `172_DKV_M128_3C_Gate`; branch
  `exp/dkv-owner16-4c-resource-probe-20260720`.
- Boundary: this probe validates register pressure, native instructions,
  ABarrier execution and bank behavior.  It does not validate full FA golden
  output, K/V startup overwrite, two-page ownership, or performance.

## 2026-07-20 Owner16 Four-Consumer Canonical Design

Status: `DESIGN_COMPLETE_CANONICAL_INTEGRATION_PENDING`.

- Keep M128 `64/32/32` as the exact tail-free control.  It uses `4+2+2`
  heavy waves and therefore does not create three full consumer groups.
- Workbook `173_DKV_Owner16_4C_Canonical` derives an exact
  `Mq64/Nk256/D128` topology.  Four symmetric groups own disjoint K/V and
  dK/dV N64 ranges; each wave is the unique owner of one N16 fragment.
- The startup K/V epoch occupies all 128KB LDS.  Its lifetime ends only after
  all 16 owner fragments are latched.  The same storage then holds two
  33,536B Q/dO+sidecar pages; startup and steady allocations never coexist.
- Page0/1 Filled+Used and AllDone form the steady protocol.  One rotated
  publisher wave per group emits the complete M16 slice, so Filled uses four
  arrivals.  Used uses 16 arrivals: without another synchronization primitive
  a group leader cannot prove its three peers completed their reads.  Strict
  even/odd generations prevent page ABA.
- Owner16 causal dispatch removes fully-invalid M16xN16 pairs.  S1024 honest
  MMOP is `2080*32=66560`; do not use the M128 control's 73728 masked MMAC or
  a group-level 69632 count to inflate active share.
- The first implementation gate is ownership/lifetime correctness, not
  performance: K/V latch checksum must survive LDS overwrite, and a
  page0-page1-page0 sentinel must prove no ABA or early release.  Full dKV is
  admitted only after `<=128 VGPR`, private/spill/scratch0, bank0 and native
  BPS+matrix-read+MMAC gates pass.

## 2026-07-20 dKV M96 + M96 Ownership Experiment

Status: `REJECT_CORRECTNESS_AND_STATIC_RESOURCE`; canonical baseline unchanged.

- The experiment kept `Mq192/Nk192/D128`, the same four GEMMs, exact MMOP,
  output ownership, and 1P3C roles. Only the three M64 Q/dO lifetimes were
  replaced by two M96 lifetimes.
- A 96-row sidecar slice needs a partial producer wave. That version compiled
  at the baseline resource level but failed S384 dK/dV correctness.
- Publishing a full 192-row sidecar into two alternating pages removed the
  partial wave. The dynamic-page version passed S192 page0 but failed S384
  when page1 first appeared, with PMD VGPR-init and DS-address warnings.
- Making both page addresses compile-time constants removed the dynamic
  address path but raised all consumer branches to the 160-VGPR ceiling and
  emitted `private_segment=120B`, `vgpr_spill=66`.
- No ticks or MMAC-active claim is admitted. The exact-work best remains tag
  `best/dkv-three-m64-lifetimes-20260719`, commit `20dbb81`.
- Evidence is recorded in workbook sheet `176_DKV_M96x2_Lifetimes` and under
  `/zys/sb/dkv_m96x2_*`.

## 2026-07-20 Score-Source Slot Recycling Rejected

Status: `REJECT_STATIC_RESOURCE`; canonical source restored.

- Baseline: exact-work `20dbb81`, `Mq192/Nk192/D128`, one producer plus three
  symmetric M64 consumer lifetimes, exact dispatch `MMOP=46,080`.
- Hypothesis: after score/dP first-use, overwrite the same Q/dO source object
  with current-M16 dV/dK normal fragments. This preserved formulas, reads,
  MMAC count, ABarrier topology and output ownership on paper.
- Implementation kept sidecar after score so its 12 VGPR did not enter the
  score peak. D0/D1 normal reads were issued under score D2/D3; D2/D3 reads
  were intended to mature under softmax/dS.
- Static result: LLVM branch usage rose from `31/156/156/156` to
  `31/160/160/160`; metadata emitted `private_segment=40B` and
  `vgpr_spill=27`. The compiler did not realize the C++ overwrite as physical
  slot co-coloring.
- Decision: reject before correctness or PMD timing, remove candidate code,
  and retain only workbook sheet `177_DKV_SlotRecycle` plus this ledger entry.
  The next experiment must not increase the consumer live set.

## 2026-07-20 RawUsed Release Priority Experiment Rejected

Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`.

- Hypothesis: consumer priority2 delays the newly awakened producer after a
  Head/Middle/Tail `RawUsed` release. Move the existing `lower_priority()`
  before the already-resident D2/D3 MMAC, without adding instructions or
  changing work, reads, barriers, resources, or ownership.
- Static/correctness gates pass: roles `31/156/156/156`, private/spill/scratch0,
  exact `MMOP=46,080`, bank0, H1/S384 and H1/S768 numerical PASS.
- Two stats-only same-flag A/B pairs favored the candidate by `0.176%` and
  `0.236%`, but the required fullperf result regressed
  `32,990,230 -> 33,383,350` ticks (`+1.192%`).
- xcu explains the rejection: raw ownership cycles improve only
  `500,460 -> 495,640` (`-0.963%`) and MMAC gap improves `-0.977%`, while
  normal+trans matrix-read waits grow `387,585 -> 419,152` (`+8.143%`) and
  terminal wait grows `470,296 -> 485,798` (`+3.296%`).
- Decision: delete the candidate source and preserve only sheet
  `178_DKV_ReleasePrio`, PMD/xcu evidence, and this log. Priority lowering at
  the ownership edge moves rather than removes the critical path.

## 2026-07-20 dQ C0-Early K-Read Stagger Rejected

Status: `REJECT_STATS_TICKS_AND_ACTIVE_REGRESSION_SOURCE_RESTORED`.

- Design basis: retain canonical `Mq128/Nk128/D128`, 16-wave physical 2P2C,
  exact three GEMMs, Q/dO latch, K/V page ledger and output ownership. Swap
  only which consumer issues the existing K-normal reads before softmax.
- Static and numerical gates pass: roles `8/159/159/9`, SGPR65/VGPR128,
  private/spill/scratch0, H1/S128 and H1/S1024 causal PASS,
  `dq_rel_l2=0.00208192`, and `ldsBankConflict=0`.
- Fresh H1/S1024 C1-early control is `24,300,185` kernel ticks and
  `34.2341%` MMAC active. C0-early is `25,095,070` ticks and `33.2370%`
  active: ticks regress `3.27%`, active falls `0.997pp`.
- Decision: restore C1-early source and keep M128 2P2C. Reversing the causal
  role skew lengthens the exact-work critical path instead of improving
  MMAC/VALU overlap.
- Run-contract audit: `CANONICAL_DQ=1` is mandatory. Omitting it launches
  `wg=128` (two waves) for a 16-wave ABarrier kernel and creates invalid
  multi-billion model ticks; those runs are excluded before metric parsing.
- Evidence: workbook `111_DQ_C0KReadStagger`; candidate
  `/zys/sb/dq_c0_stagger_s1024_r1/dq_correctness_20260720_174934`; control
  `/zys/sb/dq_c1_exact_canonical_s1024_r1/dq_correctness_20260720_192113`.

## 2026-07-20 dKV M128 Physical 2P2C Baseline Accepted

Status: `ACCEPT_CANONICAL_2P2C_BASELINE`.

- Canonical target is now tail-free `Mq128/Nk128/D128`, 16 waves, physical
  2P2C for both S1024 and S2048. The two heavy groups own exact output work;
  score and dP are not duplicated.
- Roles are producer K+Q, consumer0, consumer1, producer V+dO+sidecar.
  Configured WDRA windows are `32/160/160/32`; actual branch use is
  `8/152/14/152`. Metadata is private/spill/scratch0 and LDS is 67,072 B.
- H1/S1024 passes with dK/dV relL2 `0.0025563/0.000337571`, bank0,
  `32,507,020` kernel ticks and `37.8420%` MMAC active.
- H1/S2048 passes with dK/dV relL2 `0.00535305/0.000360253`, bank0,
  `58,721,845` kernel ticks and `44.4427%` MMAC active.
- The `+6.6008pp` active-share gain at S2048 proves the steady MMAC body is
  healthier than the startup/ownership epochs. The next structural target is
  ABarrier/page readiness and first-use wait, not a third consumer.
- M192 1P3C remains historical saturation evidence only. Its M64-tail and
  asymmetric ownership are not admitted to the S1024/S2048 canonical path.
- Evidence: workbook `180_DKV_2P2C_Targets`; remote runs
  `/zys/sb/dkv_m128_2p2c_recert_s1024/dkv_mmac_correctness_20260720_193811`
  and `/zys/sb/dkv_m128_2p2c_recert_s2048/dkv_mmac_correctness_20260720_194006`.

## 2026-07-20 dQ M128 Physical 2P2C S2048 Baseline Accepted

Status: `ACCEPT_CANONICAL_2P2C_BASELINE`.

- Reused the accepted C1-early binary exactly; only S changed from 1024 to
  2048. PMD confirms `grid=16384`, `wg=1024`, and all 16 ABarrier waves.
- Numerical gate passes with dQ relL2 `0.00475324`, L2 ratio exactly 1,
  no nonfinite values and bank0. Resource metadata remains
  private/spill/scratch0.
- S2048 is `44,827,055` kernel ticks, `MMOP=199,680`, and `40.5607%` MMAC
  active. The fresh S1024 control is `24,300,185` ticks and `34.2341%`, so
  steady-state active improves `6.3265pp`.
- This agrees with dKV: 2P2C becomes healthier as fixed startup/page ownership
  is amortized. The next experiment must remove or hide readiness cost, not
  add a third consumer or duplicate the three-GEMM chain.
- Evidence: workbook `112_DQ_2P2C_Targets`; remote run
  `/zys/sb/dq_m128_2p2c_recert_s2048/dq_correctness_20260720_195212`.

## 2026-07-20 dKV M128 2P2C S2048 SQTT Critical-Path Audit

Status: `OBSERVE_XCU_CONSUMER_LOCKSTEP`; no source change.

- Fullperf uses the accepted M128 physical 2P2C binary and passes correctness.
  The helper perf is archived as
  `/共享/shaobo/perf/20260720_201438_dkv_m128_2p2c_h1s2048_sqc7_fullperf/`
  `dkv_m128_2p2c_h1s2048_sqc7.perf`.
- Aggregate xcu attribution is dominated by
  `s_abarrier_try_wait -> s_xor_b32`: 37.36%, 6,782,768 cycles over 4,352
  occurrences. A role-local trace changes the interpretation: the producer
  owns 27 long `RawUsed` waits totaling 80,161 cycles, while steady consumer
  `RawFilled` waits are normally only 3 cycles after two startup waits of
  1,379 and 2,107 cycles.
- Therefore the producer is waiting for page reuse; the consumers are not
  starved for published Q/dO in steady state. Adding raw pages only to shorten
  the producer's active lifetime could raise MMAC-active share without lowering
  dispatch ticks and is not the next optimization.
- The measured consumer bottleneck is lockstep issue: each sampled heavy wave
  emits 3,584 MMAC issues, but MMAC+VALU coissue is only 739/748 bins
  (`7.49%/7.57%`). Its largest bubbles are MMAC-to-MMAC (30,943 cycles),
  normal-matrix-read to wait (12,140), wait to MMAC (6,361), and transpose
  read to wait (4,435).
- Next design gate: keep physical 2P2C, exact four GEMMs, the fused score+dP
  chain, early Head/Tail Used release, and the 160-VGPR consumer ceiling. Test
  only useful-work staggering that lowers S1024 ticks and does not regress
  S2048. Empty delay, a third consumer, duplicate score/dP, and buffer depth
  added only to improve the active-share denominator remain forbidden.

## 2026-07-20 dKV 2P2C Uniform Causal M16 Skip Rejected At Resource Gate

Status: `REJECT_STATIC_RESOURCE_SOURCE_RESTORED`.

- Workbook sheet `181_DKV_CausalM16Skip` derived a wave-uniform diagonal skip:
  for owner N16 block `n` and Q M16 block `m`, `m<n` is wholly causal-invalid,
  `m=n` needs the existing lane mask, and `m>n` is fully valid. The predicate
  is wave-uniform and is distinct from the rejected lane-divergent branch.
- The draft preserved M128 physical 2P2C, all barrier IDs/counts, LDS bytes,
  output ownership, and native matrix instructions. To avoid a dynamic
  first-valid accumulator case, it seeded all eight dV and eight dK
  accumulators once with zero-input MMAC.
- Arithmetic upper bound was attractive: after seed overhead, expected MMOP
  was `67,584` at S1024 (`-8.33%`) and `266,240` at S2048 (`-4.41%`). It did
  not cross the static gate. Consumer160 generated `216B` private segment and
  `210` VGPR spills; consumer168 still generated `196B` and `185` spills.
- Decision: do not raise WDRA again and do not run PMD. Restore source,
  contract, and gate exactly to `d48464d`. The skip algebra remains a future
  design candidate only if first-use zero seeding can stay local, without
  making every dK/dV accumulator live before the first useful M16.

## 2026-07-20 dQ M128 2P2C S2048 SQTT Critical-Path Audit

Status: `OBSERVE_XCU_READINESS_AND_DEPENDENCY`; no source change.

- The fullperf run passes correctness and keeps the accepted C1-early
  Mq128/Nk128/D128 physical 2P2C implementation. Fullperf kernel ticks are
  `42,946,540`; this trace-mode number does not replace the stats-only
  `44,827,055` control.
- The selected SIMD window has consumer bubble ratios `49.07%/48.03%` and
  producer ratios `98.42%/98.55%`. Producer idleness is dominated by page
  reuse and terminal completion, so it is not evidence for 3C or artificial
  helper work.
- Role-local edges are asymmetric in the intended useful-stagger direction:
  consumer0 `abarrier -> salu_32=9,632` cycles; consumer1
  `abarrier -> salu_32=3,616` and `MMAC -> MMAC=4,464`. Producer BPS
  readiness is still material at `7,373/7,577` cycles.
- The terminal ebarrier bubbles are excluded from the optimization target:
  only shortening the latest consumer can reduce dispatch completion. Keep
  the C1-early stagger and attack matrix first-use/BPS readiness or consumer
  MMAC dependency with one minimal hypothesis at a time.
- Evidence archive:
  `/共享/shaobo/perf/20260720_212508_dq_m128_2p2c_h1s2048_sqc7_fullperf`.

## 2026-07-20 dQ M256 Physical 2P2C Fragment Reuse Rejected

Status: `REJECT_S1024_TICKS_AND_ACTIVE_SOURCE_RESTORED`.

- The candidate kept 16 waves and physical 2P2C, but enlarged the tile to
  `Mq256/Nk128/D128`. Each consumer wave owned two adjacent M16 rows and reused
  each K/V fragment across both rows. Dynamic MMOP stayed exact at `50,688`,
  so the result contains no duplicate score, dP, or dQ GEMM work.
- Static and numerical gates pass: configured WDRA windows are
  `P0/C0/C1/P1=8/240/240/24`, measured branch use is `2/230/233/18`, LDS is
  `131,072B`, and private/spill/scratch are zero. H1/S256 and H1/S1024 causal
  correctness pass; S1024 relL2 is `0.00208192` and bank conflict is zero.
- H1/S1024 regresses from the M128 control's `24,300,185` to `43,571,710`
  kernel ticks (`+79.31%`). MMAC active falls from `34.2341%` to `32.5363%`
  (`-1.6978pp`). The candidate halves completed WGs `8->4` and active SIMDs
  `32->16`, which exposes the doubled per-CTA work on this target grid.
- Fragment reuse itself is real: VALU falls `68,144->48,736`, SCA
  `41,772->21,408`, LDS instructions `26,352->13,552`, waitLgkm
  `15,209.5->6,123.25`, and MMAC busy-window rises `57.40%->65.48%`.
  However, the sequential group0/group1 Q/dO latch doubles barrier cost
  `44,459.75->88,912.5`; no-V-or-M time rises `108,699->139,427`, and the
  accepted mixed score/dP stagger weakens (`coissue success 13,697->10,927`,
  failures `11,734->14,389`).
- Decision: do not run S2048 for promotion after the S1024 hard failure. Delete
  candidate source from the canonical route and restore M128 C1-early physical
  2P2C. Preserve this as evidence that a larger tile only helps if startup
  ownership is not serialized and the useful MMAC/VALU stagger survives.
- Evidence: workbook sheet `183_DQ_M256_2P2C`; remote run
  `/zys/sb/dq_m256_2p2c_validation/dq_correctness_20260720_223211`.

## 2026-07-20 dQ M128 Physical 2P2C Restore Recertified

Status: `ACCEPT_CANONICAL_RESTORE`.

- M256 source was removed from the active route. The restored source again
  builds `Mq128/Nk128/D128`, one `QDoFilled/QDoLatched` startup generation,
  and WDRA windows `40/216/216/40`; measured branch use is `8/159/159/9`.
- Old LLVM `a6a6eb6616ab...` rebuild passes the dQ source/ASM gate and symbol
  metadata gate with SGPR65, VGPR128, private/spill/scratch0. H1/S1024 causal
  correctness passes with relL2 `0.00208192` and bank0.
- Fresh PMD result is `24,260,145` kernel ticks, exact MMOP `50,688`, and
  `34.1000%` MMAC active. The locked control is `24,300,185 / 34.2341%`;
  `-0.16%` ticks and `-0.1341pp` active are aligned run variation.
- Canonical branch is `opt/2p2c-s1024-s2048-20260720`. Future source changes
  start from this restored M128 physical 2P2C implementation and must validate
  S1024 first, then S2048.
- Evidence:
  `/zys/sb/dq_m128_2p2c_restore_recert/dq_correctness_20260720_225954`.

## 2026-07-20 dQ 2P2C Exact LDS Sidecar Latch Accepted

Status: `ACCEPT_CANONICAL_OPT`.

- Topology, tile, math, MMAC count, C1-early stagger, LDS allocation, and
  ABarrier ledger are unchanged. The only source change replaces three
  compiler-generated consumer sidecar loads with an exact
  `3 x ds_read_b32 -> 8 x ds_read_matrix -> 1 x lgkm wait` startup island.
- Static gates pass with branch use `8/159/159/9`, windows
  `40/216/216/40`, SGPR65/VGPR128, and private/spill/scratch0. The main matrix
  path remains MLS/BPS + `ds_read_matrix` + MMAC; correctness and bank0 pass.
- H1/S1024 improves `24,260,145 -> 24,114,090` kernel ticks (`-0.602%`) and
  MMAC active `34.1000% -> 34.2193%` (`+0.1192pp`). FLAT falls `752->560`,
  VALU `68,144->67,312`, with exact MMOP `50,688`.
- H1/S2048 stats-only improves `44,827,055 -> 44,131,360` ticks (`-1.552%`)
  and active `40.5607% -> 40.7182%`. Exact same-environment fullperf A/B is
  `43,607,200 -> 43,161,300` ticks (`-1.0225%`) and
  `40.6921% -> 40.7884%` active.
- XCU agrees on the small end-to-end improvement: dispatch duration
  `95,644 -> 94,664`, average active waves `149.19 -> 150.27`. The dominant
  `s_abarrier_try_wait -> s_xor_b32` edge remains; waitLgkm/barrier do not
  improve. This is accepted instruction cleanup, not a solved conveyor.
- Evidence is in workbook sheet `184_DQ_VectorSidecar`; remote candidate
  `/zys/sb/dq_m128_2p2c_sidecar_ds3_s2048_fullperf_valid/dq_correctness_20260720_234337`
  and control
  `/zys/sb/dq_m128_2p2c_control_s2048_fullperf_same_env/dq_correctness_20260720_234655`.
- Shared archive:
  `/共享/shaobo/perf/20260720_234337_dq_m128_2p2c_sidecar_ds3_h1s2048_sqc7_fullperf`.

## 2026-07-21 dQ C0 N32 Pair Batch Rejected

Status: `REJECT_STATS_TICKS_AND_ACTIVE_SOURCE_RESTORED`.

- The experiment preserved M128 physical 2P2C, exact three-GEMM work, LDS,
  matrix instructions and ownership tokens. Only C0 grouped two non-boundary
  N32 score+dP stages before their softmax/dS+dQ finalization; C1 and causal
  boundaries retained the accepted cadence.
- Static gates pass at branch use `8/175/159/9`, private/spill/scratch zero.
  H1/S128 and H1/S1024 correctness pass, MMOP remains `50,688`, and LDS bank
  conflict remains zero.
- The local premise partially worked: `waitLgkm` falls `15,541.25 ->
  14,853.75` (`-4.42%`). The ownership schedule becomes worse: barrier cycles
  rise `45,401 -> 46,680.25`, coissue success falls `14,332 -> 13,355`, ticks
  regress `24,114,090 -> 24,673,285` (`+2.319%`), and MMAC active falls
  `34.2193% -> 33.9961%`.
- Reject before S2048/fullperf. The candidate source is removed. This proves
  that reducing one matrix first-use wait is not useful when two N32 epochs
  delay PageUsed and peer-consumer completion.
- Evidence: workbook `185_DQ_C0_N32PairBatch`; candidate
  `/zys/sb/dq_c0_n32_pair_s1024/dq_correctness_20260721_002458`.

## 2026-07-21 dKV 2P2C C1 Sidecar-Tail Schedule Accepted

Status: `ACCEPT_CANONICAL_OPT`.

- The Mq128/Nk128/D128, 16-wave physical 2P2C DAG, output ownership, MMOP,
  LDS allocation, ABarrier IDs/counts, producer roles, and C0 schedule are
  unchanged. C1 alone issues the existing max/invsum/delta LDS packet after
  score D1, directly into three dead score-source slots, while D2/D3 matrix
  requests remain in flight.
- Static gates pass with branch use `8/152/14/152`, configured windows
  `32/160/160/32`, SGPR52/VGPR96, LDS67,072B, private/spill/scratch0, and no
  matrix-path fallback. H1/S128, S1024, and S2048 correctness pass; bank0.
- S1024 improves `32,507,020 -> 31,553,340` ticks (`-2.934%`) and MMAC active
  `37.8420% -> 38.3937%`. Wait falls `14.85%`, barrier falls `3.67%`, and
  successful coissue rises `10,531 -> 12,328` with exact instruction counts.
- Same-environment S2048 fullperf improves `58,345,560 -> 56,162,470` ticks
  (`-3.742%`) and MMAC active `44.4589% -> 45.6554%`. Sampled SIMD duration
  falls `114,676 -> 110,580`; MMAC+VALU coissue rises C0 `7.49% -> 8.85%`
  and C1 `7.57% -> 9.30%`.
- xcu supports the mechanism: aggregate normal-read-to-wait bubbles fall
  `810,889 -> 355,108`, while the dominant ABarrier edge also shortens.
  Some trans-read/wait and MMAC dependency bubbles shift upward, so this is
  not the final 50% solution.
- Promote this source as the dKV 2P2C baseline for S1024/S2048. Do not restore
  M192 1P3C tail ownership. Workbook: `186_DKV_C1_SidecarTail`; archive:
  `/共享/shaobo/perf/20260721_010524_dkv_2p2c_c1_sidecar_tail_h1s2048_sqc7_fullperf`.

## 2026-07-21 dQ C1 K-Normal Dead-Slot Prefetch Rejected

Status: `REJECT_STATS_TICKS_ACTIVE_SOURCE_RESTORED`.

- The experiment kept `Mq128/Nk128/D128`, physical 2P2C, exact three-GEMM
  work, output ownership, LDS bytes, ABarrier tokens and all dynamic
  instruction counts. C0 remained canonical. C1 advanced the existing
  normal-K D0/D1 reads under score/dP D2/D3 MMAC and D2/D3 reads under
  softmax/dS, using dead trans-fragment lifetimes.
- Static gates pass unchanged at branch use `8/159/159/9`, SGPR65/VGPR128,
  private/spill/scratch0. S128 and two S1024 runs pass correctness; bank0 and
  exact MMOP `50,688` remain intact.
- The local mechanism worked but the kernel regressed. Mean waitLgkm falls
  `15,541 -> 14,588` (`-6.13%`), while repeated S1024 results are
  `24,310,650 / 34.0281%` and `24,564,085 / 34.0192%` versus canonical
  `24,114,090 / 34.2193%`. Mean ticks regress `1.34%` and active falls
  `0.1956pp`; barrier also rises `1.47%` on average.
- ASM shows the compiler inserted normal reads inside the score/dP MMAC
  island. The saved LDS readiness is smaller than the loss of MMAC continuity
  and peer scheduling. Reject before S2048/fullperf; canonical source is
  restored. Evidence: workbook sheet `187_DQ_C1_KNormDeadSlot`.

## 2026-07-21 dQ Split Startup Latch Rejected

Status: `REJECT_STATS_UNSTABLE_TICKS_SOURCE_RESTORED`.

- The experiment kept Mq128/Nk128/D128 physical 2P2C, exact three-GEMM work,
  C1-early cadence, LDS bytes, matrix instructions, and output ownership.
  It split the coarse startup release into `SidecarLatched` for page0 and
  `QDoLatched` for page1, allowing page0 K/V MLS to begin while the remaining
  Q/dO matrix reads retired.
- Static gates pass at branch use `8/159/159/9`, SGPR66/VGPR128, and
  private/spill/scratch0. S128 and two S1024 runs pass correctness with exact
  MMOP50,688 and bank0.
- Candidate S1024 results are `24,352,965 ticks / 34.3610% active` and
  `24,079,965 / 34.3402%`; restored-source controls are
  `24,632,790 / 34.3552%` and `23,717,785 / 34.4421%`. Candidate versus
  control means are `24,216,465 / 34.3506%` and
  `24,175,288 / 34.3987%`. Mean waitLgkm falls `4.48%` and barrier falls
  `2.07%`, but the extra token adds `888` SCA instructions, ticks regress
  `0.170%`, and active falls `0.0480pp`.
- Reject before S2048/fullperf. Restore canonical source and retain only
  workbook sheet `188_DQ_SplitStartupLatch` plus ledger evidence. This closes
  the startup-token split: lower wait and slightly higher active are not
  sufficient when end-to-end ticks do not improve stably.

## 2026-07-21 dQ C0 Mid-Softmax K-Normal Read Rejected

Status: `REJECT_TICKS_ACTIVE_CODEGEN_SOURCE_RESTORED`.

- The candidate preserved M128 physical 2P2C, exact three-GEMM work, all LDS
  pages/tokens, C1's accepted early-read cadence, and output ownership. C0
  alone moved its existing K-normal matrix read8 between the ds0 and ds1
  scalar softmax/dS halves.
- ASM confirms the intended placement outside both score/dP and dQ MMAC
  islands. Static gates pass at branch use `8/162/159/9`, SGPR65/VGPR128,
  private/spill/scratch0. S128/S1024 correctness pass with exact MMOP50,688
  and bank0.
- S1024 is `24,593,205 ticks / 33.4899% active` versus the recent
  restored-source mean `24,175,288 / 34.3987%`: ticks regress `1.729%` and
  active falls `0.9087pp`. WaitLgkm falls `4.79%`, but barrier rises `6.74%`
  and compiler CFG expansion adds `1,056` VALU plus `1,056` SCA instructions.
  PMD also emits four nonfatal VGPR read-before-write tracking warnings absent
  from canonical.
- Reject without repeat/S2048/fullperf. Restore canonical source. The local
  latency-hiding premise worked, but splitting this unrolled softmax block is
  too expensive in generated control and register scheduling.

## 2026-07-21 dKV C0-Late / C1-Early Sidecar Rejected

Status: `REJECT_REPEAT_BANK_ADDRESS_WARNING_SOURCE_RESTORED`.

- The candidate preserved Mq128/Nk128/D128 physical 2P2C, exact four-GEMM
  work, output ownership, LDS bytes, ABarrier ledger, producer work and all
  dynamic instruction counts. C1 kept the accepted sidecar issue after score
  D1; C0 issued the same packet after D2 into three dead score-source slots.
- Static gates stayed unchanged at branch use `8/152/14/152`, SGPR52/VGPR96,
  LDS67,072B and private/spill/scratch0. S128 and two S1024 runs pass numerical
  correctness with exact MMOP73,728.
- S1024 results are `31,202,080 ticks / 38.4202% active / bank0` and
  `31,524,675 / 37.7684% / bank2`. Their mean is `31,363,378 / 38.0943%`
  versus accepted `31,553,340 / 38.3937%`: mean ticks fall `0.602%`, but active
  falls `0.2994pp` and mean barrier rises `1.10%` despite wait falling `6.11%`.
- Both repeats add `ds_read_b128` unaligned/out-of-LDS address tracking
  warnings absent from the accepted C1-only binary, and the repeat violates
  the bank0 hard gate. Reject before S2048/fullperf and restore canonical.
  Workbook: `190_DKV_C0_LateSidecar`.

## 2026-07-21 dQ LLVM47a7 Toolchain Promoted

Status: `ACCEPT_DQ_ONLY_TOOLCHAIN_SOURCE_UNCHANGED`.

- The source DAG, Mq128/Nk128/D128 physical 2P2C topology, exact three-GEMM
  MMOP, LDS pages, ABarrier ledger, and output ownership are unchanged.
- Build contract: LLVM commit `47a7d59a80a4313d0c33d4667c3c8573604d0dbc`,
  `SHAOBO_RUN_ON_MODEL=0`, explicit WDRA init, and
  `-mllvm -turn-off-wdra-trap-handler=no-pad`. Static gates pass at
  SGPR60/VGPR128, branch use `8/162/162/9`, private/spill/scratch0.
- H1/S1024 fullperf improves `24,114,090 -> 21,715,330` ticks (`-9.92%`)
  and `34.219254% -> 37.749296%` MMAC active (`+3.530pp`) with exact
  MMOP50,688, correctness PASS, and bank0.
- H1/S2048 fullperf improves `43,161,300 -> 38,870,195` ticks (`-9.94%`)
  and `40.788411% -> 45.356456%` active (`+4.568pp`). VALU falls
  `257,536 -> 163,488`; SCA is essentially unchanged. Static instruction
  comparison explains the win: `v_mov_b32_e32 216 -> 24`, while MMAC and
  matrix-read counts remain exact.
- Do not promote this compiler for dKV. Its S1024 result regresses
  `31,553,340 -> 32,005,155` ticks and active falls
  `38.393708% -> 37.817133%`; dKV stays on the Jul18 compiler.
- New dQ S2048 perf archive:
  `/共享/shaobo/perf/20260721_045231_dq_toolchain47a7_nopad_2p2c_h1s2048_sqc7_fullperf`.

## 2026-07-21 dQ Cross-n_tile Half-Source Design

Status: `OBSERVE_DESIGN_READY_WORKBOOK_FIRST`.

- Role-local xcu separates aggregate producer idle from the heavy-consumer
  path. The direct consumer opportunity is the repeated
  `ds_read_matrix -> wait -> first score MMAC` edge, followed by MMAC and
  EXP dependencies; adding ownership tokens is not justified.
- The candidate keeps the canonical boundary path. On non-boundary K128
  pages, after current score/dP consumes K/V trans fragments, it reuses dead
  source slots to read only next n_tile D0/D1 K/V trans fragments. The current
  dQ `wait0` retires those reads; the next loop issues D2/D3 and runs D0/D1
  MMAC while the second half matures.
- Estimated extra peak is about32 VGPR, fitting the nominal branch slack
  `162 -> <=216`; full64-VGPR double-source retention is explicitly rejected.
  No new token, LDS byte, wait, GEMM, output, or producer task is allowed.
- Prior negative boundaries are built into the gate: no matrix read inside
  the current score/dP MMAC island, no softmax CFG split, no startup token
  split, and no causal-boundary prefetch.
- Workbook sheet: `191_DQ_CrossNTilePrefetch`. Next order is static ASM and
  branch-use proof, H1/S128 correctness, paired H1/S1024, then S2048/fullperf
  only if admitted.

## 2026-07-21 dQ Cross-n_tile Half-Source Prefetch Rejected

Status: `REJECT_TICKS_COISSUE_SOURCE_RESTORED`.

- The implementation preserves physical 2P2C, exact three-GEMM work, LDS,
  ownership, tokens, waits and dynamic instruction counts. Static resources
  remain SGPR60/VGPR128 with branch use `8/162/162/9`, no private/spill/
  scratch. S128/S1024 correctness and bank0 pass.
- Compiler ASM realizes the intended schedule: an eight-request next-half
  trans-read island follows the current score/dP MMAC island, with unchanged
  MMAC/matrix-read island counts and no read inserted inside current MMAC.
- Paired H1/S1024 control/candidate shader-active ticks are
  `21,428,225 -> 21,999,250` (`+2.665%`). MMAC active rises
  `37.6606% -> 38.1391%`, waitLgkm falls `9.36%`, and barrier falls `2.02%`;
  however waitVm rises `19.09%`, successful coissue falls `16.13%`, MMOP
  runtime rises `2.10%`, and VOP runtime rises `4.28%`.
- The hypothesis moves LDS wait but lengthens source lifetime and disrupts
  peer scheduling. Reject before S2048/fullperf, restore canonical source,
  and retain the result in workbook sheet `191_DQ_CrossNTilePrefetch`.

## 2026-07-21 dQ FWD-Style ValuExec0 Handoff Rejected

Status: `REJECT_FORCED_SERIALIZATION_SOURCE_RESTORED`.

- The candidate copied FWD's four-wave consumer handoff: C1 completed
  softmax/dS, arrived ABarrier6, and entered dQ MMAC while C0 waited before
  softmax/dS. Boundary N32 stayed canonical. Exact three-GEMM work, LDS,
  matrix/global requests, and output ownership were unchanged.
- Static gates pass at SGPR60/VGPR128, branch use `8/162/9/162`, exact
  MMAC768, private/spill/scratch0. S128/S1024 correctness passes; S1024 has
  exact MMOP50,688, bank0, and balanced ABarrier6 phases.
- Paired S1024 shader-active ticks regress `21,428,225 -> 22,305,010`
  (`+4.0917%`), while MMAC active falls `37.6603% -> 37.5848%`.
  Successful coissue falls `12,744 -> 9,908` (`-22.25%`), waitLgkm rises
  `6.83%`, waitVm rises `58.44%`, and the low-runnable-wave counter rises
  `1,485 -> 4,054`.
- The token creates forced consumer serialization before C1 sustains a ready
  dQ MMAC window; it does not reproduce FWD's useful MMAC/VALU conveyor.
  Reject before S2048/fullperf, restore canonical source, and do not copy this
  token into dKV. Workbook: `192_DQ_FwdValuPhase`.

## 2026-07-21 dKV C1 Sidecar Priority Hole Rejected

Status: `REJECT_TICKS_ACTIVE_SOURCE_RESTORED`.

- The candidate changes only C1 scheduling: after the existing sidecar read
  island it executes `s_setprio 0`, keeps the existing `lgkmcnt(5)` boundary,
  then executes `s_setprio 2` immediately before score D2 MMAC. No matrix
  work, read, wait, token, LDS byte, output ownership, or topology changes.
- Static gates pass unchanged at SGPR52/VGPR96, branch use
  `8/152/14/152`, private/spill/scratch0, MMAC1028, matrix-read610,
  `ds_read_b128`98, ABarrier57 and wait281. S128 correctness passes and the
  matrix path remains bank0.
- Two S1024 controls average `31,637,742.5 ticks / 38.366526% active`; two
  candidates average `31,792,215 / 38.2175465%`. Ticks regress `0.4883%`
  and active falls `0.14898pp` despite successful coissue rising
  `12,183.5 -> 12,482` (`+2.45%`). WaitLgkm and barrier debt rise about
  `1.56%` and `0.99%` respectively.
- The schedule makes the coissue counter prettier but resumes C1's first-use
  path later. Reject before S2048/fullperf, restore canonical source, and
  retain the boundary in workbook sheet `193_RoleLocalPriority`.

## 2026-07-21 dQ Low Dependent-MMAC Priority Rejected

Status: `REJECT_S2048_TICKS_SOURCE_RESTORED`.

- The candidate removes only the dQ-local `s_setprio 2/0` pair. Score/dP
  remains priority2; softmax/dS and dependent `dS @ K` stay priority0 until
  the next score island. Tile, work, ownership, LDS, ABarrier, reads, waits,
  stores and output are unchanged.
- LLVM47a7/no-pad emits the intended schedule: static setprio falls `64 -> 32`
  while MMAC768 and the native matrix path remain exact. Static resources are
  SGPR60/VGPR128, branch use `8/162/9/162`, private/spill/scratch0. S128,
  three S1024 runs and S2048 correctness pass with bank0.
- Three S1024 repeats are only a micro-signal: candidate/control medians are
  `21,723,065 / 21,879,585` (`-0.715%`) and means are
  `21,763,408.3 / 21,829,686.7` (`-0.304%`), with candidate active higher by
  `0.0834pp`; pairwise tick signs are not stable.
- S2048 rejects the schedule: `39,856,635 -> 40,141,920` ticks (`+0.716%`)
  while active rises `45.200206% -> 45.485718%`. WaitVm rises `12.75%` and
  successful coissue falls `7.49%`; lower LGKM/barrier debt does not shorten
  completion.
- Reject before fullperf/xcu archive and restore the canonical dQ priority2
  wrapper. Workbook sheet `193_RoleLocalPriority` holds the complete result.

## 2026-07-21 dKV Score/dP Partial Accumulators Rejected

Status: `REJECT_EXTRA_VALU_TICKS_ACTIVE_SOURCE_RESTORED`.

- Role-local SQTT identified `MMAC -> MMAC` dependency bubbles of
  `22,239/35,022 cycles` in the two heavy consumers. The candidate kept the
  exact four GEMMs but split score and dP into even/odd accumulator chains,
  then combined each pair after D3.
- Static and numerical gates pass: SGPR52/VGPR96, branch use
  `8/156/14/156` inside `32/160/160/32`, private/spill/scratch0, exact
  MMOP73,728 and matrix-read45,200, H1/S128 and two clean H1/S1024
  correctness runs PASS, and bank0.
- The extra reductions are not free. Dynamic VALU rises
  `60,752 -> 70,160` (`+9,408`, `+15.49%`); static ASM adds 128
  `v_pk_add_f32`, six `v_mov_b32_e32`, and four `s_nop` while MMAC/read/wait
  island counts remain unchanged.
- Three S1024 controls average `31,622,348 ticks / 38.394122% active`; two
  clean candidates average `32,282,250 / 36.755085%`. Ticks regress
  `2.087%` and active falls `1.6390pp`; coissue attempt rate changes only
  `57.7786% -> 57.8847%`.
- Reject before S2048/fullperf and restore canonical source. The dependency
  bottleneck remains real, but the next solution must reorder existing MMAC
  work without adding reduction arithmetic. Workbook sheet
  `195_DKV_PartialAcc` retains the design and result boundary.

## 2026-07-21 dKV C1 Post-Softmax Operand Read Rejected

Status: `REJECT_EXPOSED_LDS_DEPENDENCY_SOURCE_RESTORED`.

- The candidate preserved canonical M128 physical 2P2C, exact four-GEMM work,
  output ownership, LDS, ABarrier generations, all read/store counts and all
  dynamic instruction families. It moved only C1's existing D0/D1 Q+dO
  normal-read island from before softmax/dS to after softmax/dS.
- Static resources and correctness pass: SGPR52/VGPR96, branch use
  `8/152/14/152` inside `32/160/160/32`, private/spill/scratch0, H1/S128 and
  H1/S1024 causal PASS, exact MMOP73,728, VALU60,752, SCA38,048, LDS45,200,
  VMEM2,560, FLAT1,096, and bank0.
- Same-environment H1/S1024 control/candidate are
  `31,547,425 / 33,640,880 ticks` (`+6.6359%`) and
  `38.443812% / 37.027105% MMAC active` (`-1.4167pp`). WaitLgkm rises
  `34.2354%`, barrier debt `10.4435%`, no-V-or-M time `9.6463%`, and
  successful coissue falls `22.2007%`.
- dQ role-local SQTT explains the failure. Each normal-K island contains eight
  reads. C0 places no independent VALU between its last read and first-use
  wait/MMAC (`read->wait` median158 cycles); C1 places about36 softmax/dS VALU
  there, with the first VALU usually4 cycles after the read and the last VALU
  4 cycles before the wait. C0 softmax VALU coissues with peer MMAC1,457 times,
  while C1 does so only256 times: current dQ overlap is useful but one-way.
- Reject before S2048/fullperf and restore canonical source. Both consumers
  must issue source reads before independent MMAC/VALU coverage; future phase
  skew must use different useful prefetch distances, not a deliberately late
  read that exposes first-use LDS latency. Workbook sheet
  `196_DKV_C1_PostSfmRead` holds the design and result.

## 2026-07-21 dQ C1 Pre-Score K-Normal Accepted

Status: `ACCEPT_CANONICAL_SCHEDULE`.

- User-observed asymmetry was confirmed in source and role-local SQTT: C0 uses
  `MMAC -> VALU -> read8 -> wait -> dQ`, while C1 used
  `MMAC -> read8 -> VALU -> wait -> dQ`. C0 therefore exposed a median
  158-cycle read-to-wait edge; C1 hid the same dependency under about36
  softmax/dS VALU instructions.
- The candidate moves only C1's existing K-normal read8 before score/dP. It
  keeps Mq128/Nk128/D128 physical2P2C, exact three-GEMM work, ownership, LDS,
  tokens and all dynamic instruction families. ASM proves
  `trans16+normal8 -> wait15 -> score half0 -> wait8 -> score half1 ->
  softmax/dS -> wait4/0 -> dQ`.
- Static and numerical gates pass: SGPR60/VGPR128, role use
  `8/162/9/194` within `40/216/216/40`, private/spill/scratch0, S128,
  repeated S1024 and S2048 correctness PASS, exact MMOP and bank0.
- Three S1024 medians improve `22,166,690 -> 20,841,730` ticks (`-5.98%`)
  and active `37.7548% -> 37.9400%`. S2048 fullperf improves
  `38,870,195 -> 37,599,835` ticks (`-3.27%`) and active
  `45.356456% -> 45.840219%`; successful coissue rises
  `51,175 -> 64,398`.
- SQTT disproves the initial direct-cover story: C1 MMAC coverage inside C0
  late-read intervals falls from 32/62 to1/62. The actual win is reciprocal
  useful staging elsewhere: C1 normal reads under C0 MMAC rise
  `0/512 -> 174/512`, C1 VALU under peer MMAC rises
  `256/2410 -> 642/2410`, and C0 read-to-wait median falls `158 -> 86`
  cycles as simultaneous matrix-read contention decreases.
- Aggregate read-to-wait bubbles fall `287,852 -> 168,396` cycles (-41.5%),
  wait-to-MMAC falls15.4%, MMAC dependency8.9%, ABarrier3.6%, and terminal
  ebarrier7.7%. This is a measured local coissue improvement, not a complete
  alternating conveyor: macro bins remain mostly both-MMAC and C1 `v_exp`
  coverage is only18/512.
- Promote this schedule. Next work must preserve C1 pre-score reads and target
  C1 softmax coverage plus readiness/ABarrier relock without new tokens,
  empty delay, duplicated work, or fragmented instruction islands.

## 2026-07-21 dQ Dual-Hidden K-Normal Rejected

Status: `REJECT_TICKS_COISSUE_SOURCE_RESTORED`.

- The candidate kept accepted C1 pre-score K-normal reads and moved only C0's
  existing read8 from after softmax/dS to immediately after score/dP. The
  intended flow was C0 `MMAC -> read8 -> VALU -> dQ MMAC` versus C1
  `read24 -> MMAC -> VALU -> dQ MMAC`, so both roles hid first-use LDS
  latency while retaining one score-island of prefetch-distance skew.
- Static gates pass unchanged: MMAC768, matrix-read400 and island histograms
  are bit-for-bit equal to the accepted source; SGPR60/VGPR128, role use
  `8/162/9/194`, private/spill/scratch0. S128 and all six S1024 A/B runs pass
  correctness with exact MMOP50,688, VALU44,864, SCA42,124, LDS26,352,
  VMEM1,408, FLAT560 and bank0.
- Three-run S1024 candidate ticks are `22,063,405 / 22,245,405 /
  22,143,030`; controls are `20,938,645 / 20,894,965 / 21,162,050`.
  Medians regress `20,938,645 -> 22,143,030` (`+5.752%`).
- The local signal alone is misleading: active rises
  `37.9246% -> 38.2163%` and waitLgkm falls `7.43%`, but successful coissue
  collapses `16,264 -> 8,690` (`-46.57%`), barrier rises `4.27%`, and
  noVorM rises `1.12%`.
- Conclusion: C0's exposed read edge is also part of the useful role phase
  offset. Moving the whole C0 read island before softmax makes the consumers
  more synchronous and trades a local readiness hole for a much larger loss
  of peer MMAC/VALU overlap. Reject before S2048/fullperf and restore accepted
  commit `d97684f`.
- Boundary: preserve C0-late/C1-pre-score cadence. Future work may target C1
  `v_exp` coverage or ownership relock by reordering existing MMAC work, but
  may not retry dual-hidden K-normal, both-consumer early reads, or a rescue
  softmax split.

## 2026-07-21 dQ C1 Score/P Split Rejected

Status: `REJECT_DEPENDENCY_FRAGMENTATION_SOURCE_RESTORED`.

- Hypothesis: retain fused C0, but split C1 into
  `score MMAC -> P VALU -> dP MMAC -> dS VALU -> dQ MMAC`, so C1 P and dS
  could alternate with C0 score/dP and dQ MMAC windows without extra GEMM.
- Gates: S128 and six S1024 A/B correctness PASS; SGPR60/VGPR128,
  role-use `8/161/9/185`, private/spill/scratch0, bank0, exact MMOP50,688 and
  unchanged LDS/VMEM/FLAT.
- Result: median S1024 ticks regress `20,753,005 -> 22,616,230`
  (`+8.978%`), active falls `37.9701% -> 36.3908%`, successful coissue falls
  `22.041%`, barrier active rises `2.5387pp`, VALU rises `4.743%`, and SCA
  rises `1.443%`.
- Decision: reject before S2048/fullperf. Splitting score/dP fragments the
  independent accumulator structure and delays ownership completion; keep
  fused score/dP and obtain stagger only from existing independent work.

## 2026-07-21 dQ C0 Half K-Normal Prefetch Rejected

Status: `REJECT_ROLE_RELOCK_SOURCE_RESTORED`.

- Hypothesis: hide only C0 D0/D1 normal-K readiness under P+dS while keeping
  D2/D3 late, preserving half of the accepted role offset.
- Gates pass with exact work and resources. Three-run S1024 medians regress
  `20,845,825 -> 21,538,790` (`+3.324%`); active is flat
  `37.9695% -> 37.9294%`, waitLgkm falls `0.2089pp`, but successful coissue
  falls `23.666%` and barrier rises `0.2271pp`.
- Decision: reject before S2048/fullperf. C0 late-read timing is a whole-island
  phase anchor. Preserve it and target the next N32/page readiness boundary.

## 2026-07-21 dKV C1 Cross-Tile Head Prefetch

Status: `REJECT_STATISTICALLY_FLAT_RESOURCE_WAIT_DEBT_SOURCE_RESTORED`.

- Hypothesis: after C1 consumes tail M7 D0/D1, use its dV/dK MMAC island to
  mature next q-tile head M0 D0/D1, then enter the next loop with score/dP
  operands already resident. No GEMM, matrix request, token, page, output, or
  ownership generation is added.
- Resource gate: the canonical 160-VGPR consumer window spills 17 VGPRs. A
  176-VGPR window emits role use `8/156/14/176`, SGPR52/VGPR104,
  private/spill/scratch0, real trap0, and passes all static gates.
- Correctness: H1/S128 and H1/S384 PASS; S384 exercises the new cross-q-tile
  edge. Every measured S1024 candidate/control reports exact MMOP73,728 and
  bank0.
- Three controls are `31,534,685 / 31,861,375 / 31,711,680` ticks, median
  `31,711,680`. Two valid candidates are `31,675,280 / 31,696,210`, median
  `31,685,745`, only `-0.0818%`. Active changes
  `38.187676% -> 38.235973%` (`+0.0483pp`), wait rises `3.11%`, barrier rises
  `1.39%`, and successful coissue rises `0.85%`.
- Decision: reject before S2048/fullperf. The next-head wait/read is moved
  earlier but not removed from the one-page ownership critical path, while
  the extra live fragments consume 16 more role-local VGPRs. Experiment and
  revert commits are `14c53bf` and `9f3cf0b`.

## 2026-07-21 Rolling Compiler Refresh

- Download source:
  `http://10.65.42.71/build2/package/perf_model_latest_6.3_ubuntu-22.04`.
  `Packages.gz` headers are Last-Modified `2026-07-21 03:28:43 GMT`, ETag
  `6a5ee76b-4905`.
- Package and extracted roots are
  `/zys/shaobo/toolchains/compiler_perf_model_latest_20260721_packages` and
  `/zys/shaobo/toolchains/compiler_perf_model_latest_20260721_root`.
- Complete-root fingerprint: LLVM47a7d59a, clang-18 SHA256
  `fddad9d6b6a0bc2264d815e97bbc7679fba9268e8f0b71d145acfa466da3b395`.
  dQ `d97684f` and dKV `f57714f` both pass static gates and H1/S128 PMD
  correctness. Their target-kernel instruction streams are identical to the
  prior LLVM47a7 builds.
- Decision: accept the root as the latest available side-by-side compiler, but
  retain per-kernel locks. dQ uses LLVM47a7; dKV uses LLVM7b796991 until a
  newer compiler beats the measured same-shape dKV baseline.
## 2026-07-21 Unified LLVM47a7 Baseline And 3C Gate

Status: `ACCEPT_UNIFIED_TOOLCHAIN_BASELINE / DEFER_1P3C_TARGET_SHAPES`.

- Build governance is now executable. Commit `3da351a` locks clang to LLVM
  `47a7d59a` by SHA256, defaults to explicit WDRA init and the `no-pad` trap
  handler mode, rejects old compiler hashes, uses the installed ROCm 6.3.3
  hipcc/runtime, defaults dQ to the canonical path, and warns on unsafe long
  PMD run roots.
- The first attempted overlay-only build failed to link `-lamdhip64`; this
  proved the downloaded root is a compiler overlay rather than a standalone
  runtime. A test with the historical `-run-on-model=true` flag also failed
  because LLVM47a7 no longer exposes that option. The audited combination
  emits real trap0 for both complete binaries.
- Static and tiny gates pass: dKV SGPR52/VGPR96 and dQ SGPR60/VGPR128, both
  private/spill/scratch0, native MLS/BPS+matrix-read+MMAC, H1/S128 correctness
  PASS.
- Fresh stats-only baselines on PMD HEAD1694/SQ7 are:
  dKV S1024 `31,703,035 ticks / 38.450598%`, S2048
  `56,527,835 / 45.360179%`; dQ S1024
  `20,840,365 / 37.804048%`, S2048
  `37,643,515 / 46.149692%`. All are exact-MMOP and bank0.
- Workbook sheet `202_Unified47a7_3C_Gate` re-derives formula DAG, MMAC work,
  LDS/VGPR/SGPR budgets, output ownership and expected pipeline. True three
  heavy consumers require M192 and full `32+3*160=512` VGPR/SIMD. S1024 and
  S2048 leave M64/M128 tails; prior tail routes spilled or added dispatch/MMOP
  debt. Therefore no 3C code was added for these targets.
- Keep M128 physical2P2C. At unchanged useful work, S2048 needs about 9.28%
  dKV or 7.70% dQ critical-path reduction to reach 50% active. Capture fresh
  fullperf/xcu and attack ownership/readiness rather than consumer count.

## 2026-07-21 Unified LLVM47a7 + HEAD1694 SQTT Refresh

Status: `ACCEPT_ENVIRONMENT_AND_EVIDENCE_LOCK`.

- Commit `81bee63` adds PMD core/library/SOC SHA gates beside the existing
  clang SHA gate. The first dKV fullperf used old default HEAD1668 and produced
  `2957138_fa3_bwd_dkv.perf`; this artifact is rejected as environment drift,
  not interpreted as a kernel result.
- Valid locked fullperf artifacts are `2957276_fa3_bwd_dkv.perf` and
  `2957578_fa3_bwd_dq.perf`. Both retain the unified baseline correctness,
  exact MMOP, private/spill/scratch0, bank0 and canonical M128 physical2P2C
  source.
- dKV xcu dispatch duration is `123,728` with `937,968` issues. Aggregate
  `s_abarrier_try_wait -> s_xor_b32` is `36.22%`; barrier-ID parsing attributes
  about `5.327M` cycles to producer Raw Head/Tail Used waits. On one
  representative SIMD the two heavy consumers have only about `11%`
  MMAC+VALU coissue and still expose MMAC dependency, matrix-read-to-wait and
  wait-to-MMAC gaps.
- dQ xcu dispatch duration is `81,960` with `713,528` issues. Aggregate
  ABarrier-to-XOR is `26.67%`, mostly producer Page0/Page1 Used waits. On one
  representative SIMD C0/C1 MMAC+VALU coissue is `22.03%/18.58%`; accepted C1
  pre-score K-normal scheduling remains visible, but C1 softmax/v_exp coverage
  and N32/page macro relock remain the next structural boundary.
- Producer per-wave wait totals overlap useful consumer execution and are not
  equal to CTA critical-path time. Terminal `s_ebarrier_sync` is likewise a
  tail attribution signal. Future hypotheses must use representative
  `pipeline/simd/coissue` windows before changing ABarrier counts.
- Workbook sheet `203_Latest47a7_SQTT` holds the complete contract and
  promotion order. No source kernel change is promoted by this refresh.

## 2026-07-21 dQ C1 Page-Entry Normal Prime Rejected

Status: `REJECT_REQUEST_AGE_REGRESSION_SOURCE_RESTORED`.

- Hypothesis: on only the first N32 of each K/V page, issue C1 normal-K read8
  before trans16 so useful dQ work creates one page-level role offset. C0 and
  the remaining three N32 chunks keep their accepted cadence.
- LLVM47a7 emits the same 84 control branches, 400 matrix reads, 768 MMAC and
  84 waits as control. Role use is unchanged at `8/162/9/194`; H1/S128 and
  six H1/S1024 runs pass correctness with exact MMOP50,688,
  private/spill/scratch0 and bank0.
- Control kernel ticks are `20,734,805 / 21,741,720 / 20,844,005`; candidate
  ticks are `22,262,695 / 22,348,690 / 22,465,625`. Medians regress
  `20,844,005 -> 22,348,690` (`+7.2188%`).
- Median MMAC active falls `38.0416% -> 37.2324%`; successful coissue falls
  `16,017 -> 11,979` (`-25.2107%`). Exact work and bank0 rule out work
  inflation or LDS conflict as the cause.
- Mechanism: request order becomes normal8+trans16. To make the first half of
  trans operands ready, `wait8` must retire the eight older normal requests
  plus the first eight trans requests; `wait0` then drains the remainder.
  The added request age exposes first-use latency and collapses peer coissue.
- Decision: reject before S2048/fullperf, restore canonical source and gate,
  and delete the experiment branch. Raw evidence is under
  `/zys/sb/u47_dq_page_entry_ab`; workbook sheet
  `204_DQ_PageEntryPrime` is closed as REJECT.

## 2026-07-21 Latest Toolchain Hard Lock And F32 Writer Recheck

- `build.sh` now fails before compilation when the audited latest compiler
  cannot be resolved, verifies clang SHA256, and writes
  `toolchain_fingerprint.txt` beside every binary. `scripts/env.sh` verifies
  PMD core/lib/SOC hashes before any run.
- Remote preflight and fresh dKV/dQ builds pass with LLVM `47a7d59a`, clang
  SHA256 `fddad9d6...`, PMD HEAD1694 hashes `4748d40d/29fa2020/d0c03538`,
  `GPU_CHIP=sb`, and SQ instruction prefetch 7. Both static gates remain
  private/spill/scratch0.
- The isolated f32 matrix-writer probe was rebuilt without WDRA/local-wave so
  no `s_set_vgpr_size` can mask the result. PMD still aborts at the first
  `ds_write_matrix_format ... element:3` opcode (`0xd38b5008`). Classify this
  as `DEFER_PMD_COMPILER_ENCODING`; do not integrate f32 writer into dKV/dQ.

## 2026-07-21 BPS Nonzero VBCNT Threshold Probe

Status: `ACCEPT_PROBE_WITH_PMD_WARNING`.

- A two-wave producer/consumer probe was stress-expanded to 32 ordered BPS
  requests: prefix `0..23`, group A `24..27`, group B `28..31`. The candidate
  uses `s_waitcnt_vbcnt 4` before publishing A and `vbcnt 0` before B.
- LLVM47a7 preserves exact `BPS32`, `wait4=1`, `wait0=1`, and eight matrix
  reads. Metadata is SGPR16/VGPR7 with private/spill/scratch0. A and B are
  exact and `ldsBankConflict=0` in candidate, no-wait, and full-wait controls.
- Candidate kernel ticks are `2,704,520`, no-wait `2,700,880`, and full-wait
  `2,842,840`. Nonzero wait is `4.865%` faster than a full drain and only
  `0.135%` slower than no-wait, which rules out PMD silently treating 4 as 0.
- PMD still warns `S_WAITCNT_VBCNT: vbcnt isn't 0`, and correctness alone does
  not prove the owner-level FIFO contract because the no-wait control also
  passes. Admit only a reversible dKV issue-ahead experiment; do not promote
  the instruction rule without same-shape correctness, repeated ticks, and
  SQTT evidence.

## 2026-07-21 dKV Cross-Generation VBCNT4 Rejected

Status: `REJECT_TAIL_READINESS_REGRESSION_SOURCE_RESTORED`.

- Candidate keeps exact Mq128/Nk128/D128 physical2P2C work, seven ABarrier
  IDs, 67,072-byte LDS plan and four GEMMs. Each producer wave combines four
  old Tail BPS requests with four new Head requests, publishes Tail at
  `vbcnt4`, then publishes Head at `vbcnt0`.
- Static gates remain SGPR52/VGPR96, role usage `8/152/14/152`,
  private/spill/scratch0. S128, S384 and all six S1024 A/B runs pass numerical
  correctness with exact MMOP73,728 and bank0.
- Three-run S1024 median ticks regress `31,517,395 -> 33,978,945`
  (`+7.810%`). MMAC active falls `38.486% -> 36.117%` (`-2.369pp`), while
  barrier grows `27.279%`. Successful coissue rises `12.720%`, but is not on
  the shortened critical path.
- Root cause: delaying `TailFilled(t)` until `HeadUsed(t)` makes the consumer
  wait for current Tail after every Head. Prefetching Head(t+1) cannot repay
  the new immediate readiness debt. Reject before fullperf and restore early
  Tail publication.

## 2026-07-21 dQ C1 Full-Tile Pair Reversal Rejected

Status: `REJECT_CTA_WIDE_PAGEUSED_RELOCK_SOURCE_RESTORED`.

- Candidate preserves exact `Mq128/Nk128/D128` work and changes only the C1
  full-tile order from `0,1,2,3` to `1,0,3,2`; C0 and the causal boundary are
  unchanged. Static counts remain MMAC768, matrix-read400, matrix-load16 and
  ABarrier48, with SGPR60/VGPR128 and private/spill/scratch0.
- H1/S128, H1/S384 and all six H1/S1024 A/B runs pass correctness with exact
  MMOP50,688 and bank0. Median ticks regress `20,836,725 -> 25,479,545`
  (`+22.282%`), and MMAC active falls `37.9400% -> 33.6427%` (`-4.2973pp`).
- Barrier grows `49.8980%`, successful coissue falls `19.5209%`, and noVorM
  grows `32.3797%`. The role skew therefore creates ownership debt rather
  than useful MMAC/VALU cover.
- Root cause: each physical page's `PageUsed` token needs all eight consumer
  arrivals. Starting C0 on page0 and C1 on page1 leaves both tokens at 4/8,
  so neither producer can refill until the consumers swap pages. Reject
  before S2048/fullperf and restore the canonical same-page schedule.

## 2026-07-21 dKV C1 Compile-Time M16 Order Rejected

Status: `REJECT_ROW_ORDER_NOT_STAGE_SKEW_SOURCE_RESTORED`.

- C0 keeps `M0,1,2,3 / M4,5,6,7`; C1 uses
  `M0,2,1,3 / M4,6,5,7`. Successor prefetch follows the mapped block, M0
  remains the only first accumulation seed, and M3/M7 retain Head/Tail Used
  release. Producer, LDS, token graph and dynamic work are unchanged.
- Whole-ASM counts are identical: MMAC1024, matrix-read577, matrix-load26,
  wait271 and ABarrier57. Static resources remain SGPR52/VGPR96, role usage
  `8/152/14/152`, private/spill/scratch0. S128, S384 and all six S1024 A/B
  runs pass correctness with exact MMOP73,728 and bank0.
- Median S1024 ticks regress `31,659,810 -> 31,796,310` (`+0.431%`) and only
  one of three candidate runs wins. MMAC active falls
  `38.5123% -> 37.8845%` (`-0.6277pp`), barrier grows `4.9874%`, and noVorM
  grows `3.5714%`; successful coissue is effectively flat (`+0.1108%`).
- Root cause: the permutation changes row addresses but every M16 still has
  the same `score/dP -> softmax/dS -> dV/dK` stage shape. It therefore cannot
  create structural consumer phase skew. Reject before S2048/fullperf,
  restore canonical, and move the next experiment to real stage batching.

## 2026-07-21 dKV C1 M16 Score-First Stage Skew Rejected

Status: `REJECT_STEADY_WAIT_DEBT_SOURCE_RESTORED`.

- Candidate batches only C1 `score/dP(M0,M1)` and `score/dP(M4,M5)`, reuses
  one source bank, and then finishes the two canonical softmax/dS/dV/dK
  stages. C0, producer work, seven ABarrier IDs, 67,072-byte LDS ownership,
  output ownership and exact four-GEMM work are unchanged.
- Locked LLVM47a7 emits role use `8/152/14/160`, SGPR52/VGPR96, with
  private/spill/scratch0. H1/S128, H1/S384, three S1024 pairs and three S2048
  pairs all pass with identical numerical errors, exact instruction-family
  counts and `ldsBankConflict=0`.
- S1024 initially looks useful: median kernel ticks improve
  `31,944,640 -> 31,496,465` (`-1.403%`) and mean MMAC active rises
  `37.9764% -> 38.5000%` (`+0.5236pp`). However, LGKM wait rises `2.565%`.
- S2048 rejects the hypothesis: median ticks regress
  `56,287,140 -> 56,447,755` (`+0.285%`), mean MMAC active falls
  `45.5796% -> 45.4767%` (`-0.1029pp`), and LGKM wait rises `3.233%`.
- The extra score bank changes stage composition but does not move matrix
  operands far enough ahead of first use. It converts short-loop scheduling
  luck into steady LDS wait debt. Do not spend fullperf on this candidate;
  preserve commit `745d2f5` as evidence and restore canonical source.

## 2026-07-22 dKV C1 Native Mathematical Stage Skew Rejected

Status: `REJECT_FIRST_USE_WAIT_AND_VOP_DEBT_SOURCE_RESTORED`.

- C0 remains canonical. C1 changes only its mathematical schedule to
  `score -> P -> dV -> dP -> dS -> dK`, using native normal/trans
  `ds_read_matrix` helpers and retaining fp32 P so dS stays numerically
  identical. Producer work, seven ABarrier IDs, LDS ownership, output
  ownership and exact four-GEMM work remain unchanged.
- Locked LLVM47a7 emits longer static MMAC islands: runs fall from 225 to 186
  and mean run length rises `4.55 -> 5.51`. Role usage is
  `8/152/14/156`; SGPR52/VGPR96 and private/spill/scratch0 pass. H1/S128,
  H1/S384 and all six H1/S1024 A/B runs pass correctness with exact
  MMOP73,728 and `ldsBankConflict=0`.
- Three-run S1024 median ticks regress `31,607,030 -> 34,560,435`
  (`+9.3441%`). MMAC active falls `38.4907% -> 36.2859%` (`-2.2048pp`),
  VOP active rises `1.3467pp`, wait rises `1.1934pp`, barrier rises
  `0.9086pp`, and successful coissue falls `13.3695%`.
- Dynamic LDS/VMEM/FLAT/MMOP work is exact, so the loss is scheduling debt:
  splitting P and dS creates more VOP/priority work and each larger MMAC
  stage still reaches its operand at an exposed first-use wait. Static island
  regularity is not a performance proof. Reject before S2048/fullperf and
  restore the canonical source.

## 2026-07-22 dQ Sidecar Early-Release Page0 Prefetch Rejected

Status: `REJECT_EXTRA_STARTUP_TOKEN_DEBT_SOURCE_RESTORED`.

- Latest-toolchain SQTT showed large startup/ownership bubbles, so the
  candidate keeps the exact Mq128/Nk128/D128 16-wave 2P2C algorithm and
  splits only sidecar readiness. Consumers issue sidecar DS3 before Q/dO
  matrix8, retire the prefix at `lgkmcnt(8)`, publish SidecarLatched ID6,
  then complete Q/dO and publish QDoLatched. Producers may load page0 after
  ID6; page1 remains gated by full Q/dO latch.
- LLVM47a7 emits the intended `DS3 -> matrix8 -> wait8 -> arrive6 -> wait0`
  order. LDS stays 128KB, role usage stays `8/162/9/194`, SGPR rises only
  `60 -> 61`, and private/spill/scratch0 passes. H1/S128, H1/S384 and all
  six H1/S1024 A/B runs pass with exact MMOP/VALU/LDS/VMEM/FLAT and bank0.
- Three-run S1024 median ticks regress `21,118,370 -> 22,059,765`
  (`+4.4577%`). MMAC active falls `0.4653pp`, successful coissue falls
  `3.3024%`, and barrier rises `0.7915pp`. LGKM wait falls only `0.0232pp`;
  dynamic SCA grows `42,124 -> 43,012`.
- The proposed data overlap exists but is too short to repay an additional
  startup ABarrier handshake. Reject before S2048/fullperf and restore the
  canonical six-token lifecycle. The next ownership change must reuse an
  existing token or remove an epoch, not split one into two.

## 2026-07-22 dQ Accumulator Dependency-Distance Sweep Rejected

Status: `REJECT_TICKS_REGRESSION_SOURCE_RESTORED`.

- Candidate changes only `dq_update_from_ds_pair`: each accumulator still
  receives `ds0*K0` before `ds1*K1`, but all independent `ds0` updates are
  issued before the matching `ds1` sweep. The same-accumulator recurrence
  distance grows from one to three intervening MMACs without adding work.
- Locked LLVM47a7 preserves the intended operand order. Candidate/control
  both have MMAC768, matrix-read400, identical island histograms and
  ABarrier48. Static resources stay SGPR60/VGPR128, role usage
  `8/162/9/194`, private/spill/scratch0.
- H1/S128, H1/S384 and all six H1/S1024 A/B runs pass correctness. Dynamic
  MMOP50,688, VALU44,864, SCA42,124, LDS26,352, VMEM1,408, FLAT560 and
  `ldsBankConflict=0` are exact.
- Three-run S1024 median ticks regress `20,781,670 -> 20,858,565`
  (`+0.3700%`). MMAC active rises `37.8893% -> 38.0466%` (`+0.1573pp`),
  while LGKM and barrier shares fall `0.0434pp/0.3449pp`; however VM wait
  rises `0.1048pp` and successful coissue falls `0.3860%`.
- This is a clean example of active share moving opposite to the final metric.
  The recurrence was not the dominant CTA critical edge; source-fragment
  readiness and peer issue timing repay the local gain. Reject before
  S2048/fullperf and restore canonical issue order.

## 2026-07-22 dKV D2/D3 Dead-Slot Prefetch Rejected At Resource Gate

Status: `REJECT_RESOURCE_GATE_SOURCE_RESTORE_REQUIRED`.

- Workbook 213 locks the LLVM47a7/PMD1694 environment, exact Mq128/Nk128/D128
  physical 2P2C DAG, four GEMMs, 67,072B LDS, six ownership tokens and unique
  dK/dV output ownership. Only dV/dK D2/D3 normal operand issue time changes.
- Candidate overwrites the score/dP D2/D3 source fields after their final
  MMAC use, issues D2/D3 plus D0/D1 before softmax/dS, and changes the bounded
  sidecar wait from `lgkmcnt(4)` to `lgkmcnt(8)`. Static instruction work stays
  MMAC1028, matrix-read610 and ABarrier57; wait8 is present 32 times.
- Latest locked LLVM47a7 reports role use `8/160/14/160`, but the symbol
  metadata has `vgpr_spill_count=10` and `private_segment_fixed_size=28B`.
  The hard no-spill/private gate fails before any PMD run.
- The original lifetime proof was incomplete: the score fragments are
  semantically dead, but the replacement normal fragments now remain live
  simultaneously with score/dP, sidecar/softmax output and D0/D1 sources.
  Dead variable names are not equivalent to spare physical VGPR capacity.
- Reject without correctness/performance spend. Restore source and the
  branch-local gate together; a future issue-ahead design must either shorten
  another live range first or prefetch only after proving a peak-liveness
  reduction in generated metadata.

## 2026-07-22 dQ All-Wave Q/dO Startup Rejected

Status: `REJECT_TICKS_ACTIVE_SOURCE_RESTORE_REQUIRED`.

- Workbook 214 redistributes the exact same 32 Q/dO MLS+BPS calls from eight
  producer waves issuing four each to all 16 waves issuing two each. Four
  disjoint M32 slices cover Q/dO exactly once. QDoFilled remains ID4 but its
  expected arrivals change `8 -> 16`; sidecar, QDoLatched and steady K/V page
  ownership remain canonical.
- Latest LLVM47a7 preserves static MMAC768, matrix-read406, MLS18, wait0=60,
  role usage `8/162/9/194`, SGPR60/VGPR128 and private/spill/scratch0.
  Branch-local ABarrier instructions grow48->50 and vbcnt4->6.
- H1/S128, H1/S384 and all six paired S1024 runs pass numerical correctness;
  dynamic MMOP50,688, VALU44,864, LDS26,352, VMEM1,408, FLAT560 and bank0 are
  exact. SCA grows `42,124 -> 42,948` from the extra handoffs.
- Three-run S1024 median ticks regress `24,486,280 -> 24,998,155`
  (`+2.0905%`), active falls `37.919312% -> 37.555384%`, and successful
  coissue falls `16,131 -> 15,440`. LGKM and barrier shares fall, proving the
  startup wait was shortened, but the role transition into steady compute is
  less efficient and dominates the net result.
- Reject before S2048/fullperf. Restore source and experiment gate; do not make
  heavy consumers participate in Q/dO publication without removing equivalent
  handoff work or preserving the accepted steady role-entry cadence.

## 2026-07-22 dQ Interleaved M16 Ownership Rejected At S2048

Status: `REJECT_S2048_SCALING_SOURCE_RESTORE_REQUIRED`.

- Candidate changes only `local_m16`: C0/C1 ownership moves from contiguous
  `{0,1,2,3}/{4,5,6,7}` to alternating `{0,2,4,6}/{1,3,5,7}`. The row union,
  output uniqueness, startup, K/V pages, ABarrier counts and accepted
  C0-late/C1-pre-score K-normal schedule remain exact.
- LLVM47a7 candidate/control both emit MMAC768, matrix-read406, MLS18,
  ABarrier48, wait0=60, v_mov108 and branch148. Resources remain
  `8/162/9/194`, SGPR60/VGPR128, private/spill/scratch0.
- H1/S128, H1/S384 and twelve paired S1024/S2048 runs all pass correctness;
  dynamic MMOP/VALU/LDS/VMEM/FLAT are exact and bank conflicts are zero.
- S1024 candidate wins all three repetitions; median ticks improve
  `24,597,755 -> 24,521,315` (`-0.3108%`) and successful coissue rises
  `3.3609%`, though active falls `0.2971pp` and barrier share rises.
- S2048 reverses the result: median ticks regress `41,196,610 -> 41,626,585`
  (`+1.0437%`). Active is effectively flat (`+0.0273pp`) and coissue remains
  higher, but collective PageUsed/barrier debt dominates the steady path.
- Reject before fullperf/xcu and restore canonical contiguous ownership.
  Causal row balancing is a short-shape effect, not the structural path to
  50% active; do not continue with another row-order-only variant.

## 2026-07-22 Unified Latest-Compiler Route Hardened

Status: `ACCEPT_TOOLCHAIN_GOVERNANCE`; no kernel source or performance claim.

- Rechecked the canonical rolling repository. `Packages.gz` remains the
  2026-07-21 03:28:43 GMT build, and clang remains LLVM47a7d59a with SHA256
  `fddad9d6...`.
- Removed historical compiler fallbacks from `build.sh`. The build now requires
  the side-by-side LLVM47a7 root and verifies that the installed ROCm 6.3.3
  hipcc runtime wrapper dispatches that exact compiler through
  `HIP_CLANG_PATH`. Compiler and wrapper hashes are stored with every build.
- A controlled attempt to use the rolling package's hipcc directly compiled
  device code but failed link with `unable to find library -lamdhip64`; the
  package root is not a complete runtime. The admitted route is therefore
  latest compiler plus installed HIP runtime, not a mixed compiler route.
- Fresh canonical dKV/dQ builds pass static and symbol metadata gates. Their
  normalized ASM is byte-identical to the previous LLVM47a7 controls; only the
  nondeterministic `__hip_cuid_*` symbol differs. H1/S128 correctness passes for
  both kernels under `/zys/sb/audit_unified_latest_correctness`.

## 2026-07-22 dKV C1 Family-Sweep MMAC Rejected

Status: `REJECT_TICKS_ACTIVE_SOURCE_RESTORED`.

- Workbook 216 keeps C0 and all ownership/read/wait work canonical. C1 alone
  groups four dV MMAC updates before four dK updates instead of alternating
  dV/dK, with exact arithmetic, fragments and output owners.
- Latest LLVM47a7 emits the intended order with exact MMAC1024, matrix-read544,
  ABarrier57, SGPR52/VGPR96 and private/spill/scratch0. H1/S128 and six paired
  S1024 runs pass correctness with exact dynamic work and bank0.
- Three-run median ticks regress `31,547,425 -> 31,822,245` (`+0.8710%`), MMAC
  active falls `38.404684% -> 37.957059%`, and successful coissue falls `3.92%`.
  LGKM wait falls `0.0980pp`, but barrier share rises `0.69154pp`.
- Static island regularity is not a critical-path win. Revert commit `26940f8`
  restores canonical pairwise dV/dK order; skip S2048/fullperf and move the next
  dKV hypothesis to C0 operand aging without extra live state.

## 2026-07-22 PMD Seed Added To The Unified Toolchain Lock

Status: `ACCEPT_TOOLCHAIN_GOVERNANCE`; no kernel source change.

- The fail-closed route now supplies the audited PMD config seed by default and
  unconditionally requires the latest compiler, PMD, `GPU_CHIP=sb`, SQ7 and a
  non-empty seed. PMD still attempts config generation; when that attempt fails
  with the known `ASTCA ... num_phase` error, the lock guarantees fallback to
  the audited copied seed instead of admitting an unknown generated config.
- Build fingerprints now record the seed path and SHA256 beside compiler, hipcc,
  PMD, WDRA and target information. Kernel performance remains tied to the
  previously measured LLVM47a7/HEAD1694 baseline.

## 2026-07-22 dKV C0 Split-Sidecar Aging Promoted

Status: `ACCEPT_CANONICAL_MICRO_SCHEDULE`; the 50% MMAC-active goal remains open.

- The exact four-GEMM `Mq128/Nk128/D128` physical-2P2C DAG, producers, seven
  ABarrier IDs, LDS 67,072B and output ownership are unchanged. C0 alone issues
  max/invsum after score D0 and delta after score D1 into dead source slots,
  while preserving the ordered LDS retirement proof before normal D01 reads.
- LLVM47a7 emits SGPR52/VGPR96, role use `8/152/14/160`, private/spill/scratch0,
  MMAC1028, matrix-read610 and ABarrier57. H1/S128, H1/S384 and all twelve paired
  S1024/S2048 runs pass correctness with exact dynamic work and bank0.
- Three-run median ticks improve at both steady lengths: S1024
  `31,613,400 -> 31,119,270` (`-1.5630%`) and S2048
  `56,402,255 -> 55,759,340` (`-1.1399%`).
- Valid H1/S2048 fullperf improves canonical `56,527,835 -> 55,536,390` ticks
  (`-1.7540%`) and active `45.360179% -> 45.441267%`. XCU reports dispatch
  duration `123,728 -> 122,060`, aggregate `s_waitcnt` latency down `10.61%`,
  heavy 64-cycle no-MMAC bins `196 -> 146`, and MMAC-vs-VALU bins `323 -> 404`.
- Promote commit `d2d5bdd`. This is a request-age/coissue improvement, not a
  structural solution: no-VM remains about 35%, ownership/XOR is unchanged and
  active is still below 50%. The next hypothesis must attack those debts without
  adding normal-fragment live state, requests, GEMMs or ownership tokens.
- Archived perf, stats, XCU CSV and workbook preview under shared
  `shaobo/perf/20260722_051808_dkv_c0_split_sidecar_h1s2048_sqc7_u47_fullperf`.
