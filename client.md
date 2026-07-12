# Client

## Mission

Build clean Shaobo FA3 BWD kernels in the FA3 FWD style.  The current preserved
dKV baseline remains the 7-gemm focused dKV line; dQ is now reopened on a
separate branch and must follow the same workbook-first discipline.  The main
optimization target is MMAC active share, with FA3 FWD as the hard benchmark.
Correctness, no scratch/spill, `ldsBankConflict=0`, and explainable SQTT
evidence are required before any performance claim.

## Current Environment

- New-machine container: `shaobo_dev_8426` on `10.59.41.48`, reached through
  `ssh -F work/ssh/shaobo_new_perf_config shaobo-new-perf-via-hedr`.
- For a new Codex conversation, use the Mac-side SSH config at
  `/Users/zhangyushun/Documents/Codex/2026-06-08/shaobo-hip-shaobo-demo/work/ssh/shaobo_new_perf_config`,
  then run `docker exec -it shaobo_dev_8426 bash` and work from
  `/zys/shaobo/fa3_bwd_wasp_clean`.
- Keep experiment output outside the clean repo under
  `/zys/shaobo_runs/<short_case_name>`; do not drop `m5out` folders or `.perf`
  files into the source tree.
- `build.sh` now defaults to the zwj/liuchang overlay compiler when present:
  `/home/zhangyushun/toolchains/zwj_liuchang_llvm_7940/bin/clang++`,
  llvm commit `7940bbec4a9c...`.  It still accepts `CLANGXX`, `HIPCC`, and
  `HIP_CLANG_PATH` overrides for controlled experiments.
- Formal switch evidence:
  standard dQ and dKV builds print `toolchain zwj_liuchang_llvm_7940 overlay`,
  both produce `s_trap=0` with role-local `s_set_vgpr_size`, and PMD H1/S128
  correctness passes for dQ
  `/zys/shaobo_runs/formal_zwj7940_overlay/dq_correctness_20260709_122743`
  and dKV
  `/zys/shaobo_runs/formal_zwj7940_overlay/dkv_mmac_correctness_20260709_122745`.

## Current dKV State

- Current accepted dKV source is `dkv_q_used_release_before_softmax`:
  `Mq=128,Nk=128,D=128,16 waves`, with Q/dO half-page ownership and sidecar
  staged in LDS by the producer.
- Latest observed dKV candidate:
  `dkv_half_filled_merge` keeps `QUsed`/`DoutUsed` independent but merges only
  the Q/dO half-filled readiness tokens.  H1/S128 and H1/S1024 correctness
  pass with no spill/scratch and `ldsBankConflict=0`; repeat H1/S1024 is
  `simTicks=46,698,470`, MMAC active `33.3278%`, `SCA=111,944`, versus prior
  best `46,716,670` ticks and `SCA=114,520`.  Treat it as
  `OBSERVE_STATS_REPEAT_WIN_FULLPERF_PMD_STARTUP_BLOCKED` until fullperf/xcu
  captures prove whether the ABarrier ownership bubble actually moved.
- Current accepted dKV micro cleanup:
  `dkv_wave0_terminal_invalidate` keeps the `AllDone` role-exit token but
  makes terminal ABarrier invalidation wave0-only.  Removing `AllDone`
  entirely caused metadata spill (`private_segment=244`, `sgpr_spill=2`,
  `vgpr_spill=60`), so `AllDone` remains required.  The accepted cleanup
  passes H1/S128/H1/S1024 correctness/resources and repeats H1/S1024 at
  `46,682,090` ticks, `SCA=111,248`, `coissue=37,013/25,997`,
  `ldsBankConflict=0`.  It is only a tail-control win; mainloop ABarrier/page
  ownership remains the dKV bottleneck.
- Latest dKV fullperf/xcu:
  `/zys/shaobo_runs/dkv_wave0_inv_fullperf_20260712_211315` passes H1/S1024
  correctness with `simTicks=46,829,510`, `MMOP=131,072`,
  `VALU=168,384`, `SCA=111,248`, `LDS=79,360`, `VMEM=4,352`,
  `ldsBankConflict=0`.  xcu shows the dominant rows are ownership/control:
  `s_xor_b32 34.64%`, `s_waitcnt 19.54%`, MMAC `10.73%`; the selected
  Q1/Dout1-used window has `s_abarrier_try_wait -> s_xor_b32` around
  `5.1k` cycles, and tail AllDone has `s_abarrier_try_wait -> s_waitcnt`
  around `12.3k` cycles.
- Latest rejected terminal experiment:
  wave0-only final wait / non-wave0 early exit passed static gates but PMD
  aborted on H1/S128 with `vgpr47 is not init or has been freed` during MMAC.
  Source is restored.  Do not reduce dKV terminal convergence again without a
  focused WDRA-exit ABI proof.
- Fixed-env H1/S1024 full perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_033115_dkv_qused_before_softmax_h1s1024_sqc7_fullperf`,
  `simTicks=46,716,670`, `kernel_ticks=43,103,060`,
  `MMAC active=33.2391%`, `coissue=36,556/25,587`,
  `ldsBankConflict=0`, static metadata `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch.
- Previous fixed-env dKV baseline `dkv_splitwait_highsrc`:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_003152_dkv_splitwait_h1s1024_sqc7_fullperf`,
  `simTicks=47,484,710`, `MMAC active=32.9468%`, `ldsBankConflict=0`,
  static metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
- The accepted QUsed-before-softmax micro-win moves ReleasePage Q-normal reads
  and `QUsed` arrival before softmax/dS, then runs dV/dK MMAC with already
  ready Q source registers.  xcu shows dispatch duration `96,420 -> 94,728`
  and the dominant `s_abarrier_try_wait -> s_xor_b32` bubble
  `41.38% -> 40.55%`.  It also grows consumer branch windows
  `189/240 -> 222/240` and worsens some `ds_read_matrix -> s_waitcnt` share,
  so treat it as a small ownership/ticks win, not the structural route to
  `60%` MMAC active.
- Rejected dKV probes after this baseline:
  release-half Q read-ahead improved active slightly but regressed ticks and
  pushed consumer windows to `222/240`; direct global sidecar failed metadata
  with `sgpr_spill_count=12` before correctness; sidecar ring2 prefetch was
  static-clean but failed H1/S1024 correctness even after adding an LDS
  visibility wait; score-zero-hoist reduced one `v_mov` class but regressed
  ticks; naive consumer half-order stagger passed correctness/resources but
  regressed H1/S1024 `simTicks 46,716,670 -> 47,896,485` and MMAC active
  `33.2391% -> 31.1416%`; Q-side sidecar prefetch reduced barrier counter but
  regressed ticks; merging QUsed/DoutUsed into one RawHalfUsed lowered SCA but
  regressed ticks and MMAC active because it delayed independent page release;
  dP-before-Q first-pair split passed correctness/resources but regressed
  `simTicks 46,716,670 -> 48,090,770` and MMAC active
  `33.2391% -> 32.5023%`, so keep score/dP fused unless xcu proves a real
  dO-ready/Q-not-ready gap; after half-filled merge, removing producer1
  `seq_q_half_filled` reduced SCA but regressed repeat ticks
  `46,698,470 -> 46,755,345` and MMAC active `33.3278% -> 33.1816%`;
  flattening half0/half1 readiness into one full-Mq128 `Q0Filled` token
  regressed to `47,544,770` ticks and `31.6659%` MMAC active, proving the
  half-page conveyor is useful.
  Do not keep any of these in active source.
- Next dKV work should target ABarrier/page lifetime or useful MMAC per
  ownership epoch while keeping sidecar LDS-local and the hot matrix path on
  `matrix_load` + `ds_read_matrix` + MMAC.

## Current dQ Override

- 2026-07-12 current canonical:
  `dq_boundary_ntile_classify` is still the accepted best.  Repeat H1/S1024
  under `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']` is
  `simTicks=29,706,495`, MMAC active `32.0864%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=6,280/10,438`, `ldsBankConflict=0`.
- 2026-07-12 latest code-governance result:
  `dq_compute_pages_from_latched` is accepted as canonical because it passed
  correctness/resources and repeated at `29,216,460` ticks.  The follow-up
  `dq_latch_qdo_sidecar` extraction was rejected and removed: it still passed
  H1/S128/H1/S1024 and resource gates but regressed H1/S1024 to
  `29,466,255` ticks with unchanged instruction counts.
- 2026-07-12 latest rejected dQ probe:
  `dq_setprio_narrow_dqmmac` moved `s_setprio 2` in `dq_update_from_ds_pair`
  after K-normal matrix reads.  Correctness/resources passed, but H1/S1024
  regressed to `29,979,040` ticks.  Source has been restored locally and
  remotely.  Keep the current dQ priority island covering read/wait/MMAC until
  xcu evidence says otherwise.
- 2026-07-12 latest dQ fullperf/xcu:
  `/zys/shaobo_runs/dq_canonical_fullperf_20260712_212222` passes H1/S1024
  correctness with `simTicks=29,269,240`, `MMOP=50,688`,
  `VALU=57,968`, `SCA=54,172`, `LDS=26,352`, `VMEM=1,408`,
  `ldsBankConflict=0`.  xcu shows `s_xor_b32 26.70%`,
  `s_cbranch_vccnz 17.35%`, MMAC `12.52%`, `s_waitcnt_vbcnt 8.96%`.
  Top Page0Used wait reaches `6.3k` cycles; terminal CTA sync is still
  visible.  This confirms the next dQ direction is ownership/control exposure
  or useful work per ownership epoch, not matrix-path replacement.
- 2026-07-12 latest rejected terminal experiment:
  adding final AllDone then letting only wave0 wait/invalidate caused H1/S128
  PMD abort with `vgpr81 is not init or has been freed` during MMAC despite
  clean static gates.  Source is restored and dQ gate recertified.  Keep all
  role waves converged through terminal cleanup in the canonical path.
- 2026-07-12 `.53` recert:
  jump host `.53` recovered and remote `/zys/shaobo/fa3_bwd_wasp_clean` was
  resynced from local canonical commit `a351fc3`.  Build/gates pass with
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch and branch windows
  `8/40,159/216,159/216,9/40`; H1/S128 and H1/S1024 causal correctness pass.
  The SQ7 recert run gives `simTicks=30,237,935`, MMAC active `31.7677%` with
  identical instruction counts to the best run, so it is an
  `OBSERVE_ENV_RECERT`, not a new best.
- Command caution:
  a nested-quote SSH command accidentally dropped `--SQCIPfLines=7` and
  produced a false regression (`31,546,515` ticks / `29.7161%`).  Prefer
  heredoc or the `scripts/env.sh` default for PMD runs.
- Latest structural reject:
  page0 non-overlap preload reduced barrier and raised active share to about
  `32.85%`, but repeat H1/S1024 was `29,939,455` ticks and did not beat the
  accepted `29,706,495`; source restored.  Page0 startup ownership matters, but
  splitting the single sidecar-overlap K block is not enough.
- 2026-07-12 latest rejected micro-restore:
  restoring the old SoA `Vec4F32` consumer sidecar LDS reads passed remote
  build/static/metadata and H1/S128/H1/S1024 correctness, but the available
  H1/S1024 stats backup is truncated and only proves
  `system.simTicks=29,960,840`, slower than the accepted repeat best
  `29,706,495`.  Decision:
  `REJECT_STATS_INCOMPLETE_TICKS_REGRESSION_SOURCE_RESTORED`.  Canonical dQ
  source is restored to scalar volatile sidecar LDS reads; do not restore old
  sidecar micro-wins without same-shape repeat and full active/resource
  evidence.
- 2026-07-12 cleanup:
  active `include/dq_contract.h` now only carries the canonical dQ contract
  (`ActiveDqTile`, `DqBarrierLedger`, and optimization targets).  Native dS
  ring/source-slot structs were moved to `probes/dq_probe_contract.h`, so the
  performance route has no leftover wrong-layout/prototype path constants.
  Remote build/asm/static gates pass with `private=0`, `sgpr=65`, `vgpr=128`,
  no spill/scratch.  H1/S128 and H1/S1024 correctness pass.  H1/S1024 fullperf
  archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_dq_contract_cleanup_h1s1024_sqc7_fullperf`,
  `simTicks=30,262,960`, `MMAC active=32.0547%`, `ldsBankConflict=0`.
  This is an `OBSERVE_CLEANUP_RECERT`, not a performance optimization.
- Current canonical source has moved from the earlier 12-wave dS-worker route
  to a 16-wave full-3GEMM dQ route.  Do not describe the current dQ kernel as
  12-wave: the active source uses `__launch_bounds__(1024, 1)` and
  `hcu_wdra_waves_per_tg(16)`.
- Wave roles:
  waves0-3 publish Q/dO group0 sidecar and K, waves4-7 compute q rows 0-63,
  waves8-11 compute q rows 64-127, and waves12-15 publish Q/dO group1 sidecar
  and V.
- dS is no longer staged in LDS.  Each consumer computes its complete dQ
  chain in VGPR: `QK^T`, `dO V^T`, softmax/dS, then `dS K`.
- Current canonical dQ baseline:
  `Mq=128,Nk=128,D=128`, startup `Q+dO+sidecar` LDS latch, steady K/V
  double-page ping-pong, BPS `s_waitcnt_vbcnt` before Filled arrivals, and
  no terminal `AllDone` ABarrier.  The latest accepted C74 branchless causal
  mask keeps this topology and only removes two per-element dS causal branches.
  H1/S1024 stats after C74:
  `simTicks=32,597,110`, `MMAC active=31.6674%`,
  `MMOP=55,296`, `VALU=89,216`, `coissue=9,431/8,921`,
  `ldsBankConflict=0`.  H1/S1024 fullperf:
  `simTicks=32,721,325`, `MMAC active=31.6115%`.  Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `74_DQ_BranchlessCausal`.  xcu still shows remaining
  ABarrier/control/BPS readiness debt (`s_xor_b32`, `s_cbranch_vccnz`,
  `s_waitcnt_vbcnt`); this is the next optimization class, not missing MMAC.
- Latest structural rejection:
  sheet `75_DQ_SingleProducer12` tested a 12-wave single-producer topology:
  waves0-3 publish both Q/dO sidecar groups and both K+V pages, with two
  consumer groups unchanged.  Correctness/resources were clean after producer
  VGPR window `40 -> 48`, but H1/S1024 regressed:
  `simTicks=32,597,110 -> 32,779,565` while MMAC active was flat
  `31.6674% -> 31.6917%`.  Source is restored to C74.  Lesson: reducing
  producer/control count is not enough if it serializes K+V page publication.
- Latest token-split rejection:
  sheet `76_DQ_SidecarLatch` tested a startup-only `SidecarLatched` token so
  page0 K/V could begin after consumers read LDS sidecar rows, before full
  Q/dO latch.  Correctness/resources were clean, but H1/S1024 fullperf
  regressed `32,721,325 -> 32,877,390` while MMAC active only moved
  `31.6115% -> 31.7176%`.  Source is restored to C74.  Lesson: fine-grained
  startup token splitting can improve stats-only noise but still lose under
  SQTT/fullperf because extra ABarrier/control does not reduce the critical
  path.
- Latest tail-control rejection:
  sheet `77_DQ_Tail_RawSBarrier` preserved the mandatory pre-invalidate tail
  sync but emitted raw `s_barrier` instead of HIP `__syncthreads()`.  It
  passed correctness/resources, but H1/S1024 stats regressed
  `32,597,110 -> 32,835,530` with MMAC active only
  `31.6674% -> 31.7079%`.  Source is restored to C74.  Lesson: the remaining
  terminal sync should stay canonical for now; codegen-only tail tweaks are
  not enough to reach 40%.
- Latest scalar-algebra rejection:
  sheet `78_DQ_ExactKTile` replaced the canonical exact causal
  `active_k_tiles` calculation with the algebraic form `q_tile + 1`.
  Static/resource gates passed and metadata showed fewer SGPRs
  (`65 -> 58`), but H1/S1024 stats regressed
  `32,597,110 -> 32,615,310`, MMAC active fell
  `31.6674% -> 31.6334%`, and SCA rose `40,732 -> 42,344`.
  Source is restored to C74.  Lesson: do not promote scalar simplifications
  from static metadata alone; same-shape ticks and instruction mix are the
  decision evidence.
- Latest producer-schedule rejection:
  sheet `79_DQ_SidecarPrefetchMLS` moved the producer sidecar global load
  before Q/dO MLS, then stored sidecar to LDS after the matrix loads.  Static
  gates and H1/S128/H1/S1024 correctness passed, and ASM matched the intended
  `global_load -> matrix_load -> ds_write` schedule.  H1/S1024 stats still
  regressed versus C74 fullperf stats `32,721,325 -> 33,057,115`; lower
  local `waitLgkm`/barrier counters were offset by higher VALU/SCA and worse
  elapsed ticks.  Source is restored to C74.  Lesson: stop sidecar-schedule-only
  tweaks; the next route must change useful compute per ownership epoch or the
  native dS dependency graph.
- Latest zero-init rejection:
  sheet `80_DQ_DqRegZeroSeed` tried to zero-seed the long-lived dQ
  accumulators on the first `dS @ K` update.  Correctness/resources were clean,
  but H1/S1024 regressed `32,597,110 -> 34,696,480` ticks and MMAC active fell
  `31.6674% -> 29.8264%`.  Static `v_mov_b64` improved `39 -> 7`, but the
  first-update path added code size/control and raised VALU/SCA.  Source is
  restored to C74.  Lesson: zero-seed is safe for fixed first-MMAC islands,
  not for persistent accumulators that need runtime first-update state.
- Latest structural rejection:
  sheet `81_DQ_KFirstVOverlap` split K/V readiness and changed consumer order
  to `KFilled -> score -> VFilled -> dP -> dS -> dQ`.  Static resources looked
  promising (`consumer 159/216 -> 127/216`), but H1/S128 hung.  Source is
  restored to C74.  Lesson: K-first overlap needs a focused
  KFilled/VFilled/PageUsed protocol probe before it can re-enter the
  performance kernel.
- Latest K-first follow-up:
  sheet `82_DQ_KFirstCountFix` fixed the KFilled arrival count and passed
  H1/S128/H1/S1024 correctness, proving the C81 hang cause.  It still
  regressed H1/S1024 `32,597,110 -> 34,374,340` ticks and MMAC active
  `31.6674% -> 30.3953%`.  Lesson: reducing live VGPR by separating K and V
  fragments is not enough if it breaks the paired score/dP MMAC island and
  raises wait/control.
- Latest rejected producer-ownership variants:
  K/V split tokens regressed `35,750,715 -> 36,198,435` by adding
  scalar/control and barrier debt.  Alternate-page full-KV producers lowered
  SCA (`77,516 -> 66,476`) but still regressed ticks
  `35,750,715 -> 35,807,590` because one producer publishing full K+V
  serialized page availability and raised barrier/wait.  Do not continue
  page-ownership-only tweaks as the main route.
- K-first true-overlap design review:
  workbook sheet `51_dq_kfirst_true_overlap` found a pre-code counterexample.
  In the current per-`n_tile` immediate loop, V cannot be page-level released
  after one dP because later `n_tile` chunks still need the same V page.
  K-first can only hide `VFilled` wait, not materially shorten V lifetime,
  unless we either store dS/qk/dp-like intermediates or add fine n_tile tokens.
  The latter resembles known token-control regressions; the former points back
  to the native dS handoff/slot-map ring.
- Next top-level design candidate:
  workbook sheet `52_dq_native_ds_ring` proposes a correctness-first
  `Mq64,Nk128,D128,12wave` native dS handoff/ring prototype.  Roles are
  producer, C_dS publisher, and C_dQ consumer/writer.  dS is streamed through
  two `N32` LDS slots using the previously accepted slot-map direction rather
  than scalar gather/permute.  This is the first candidate that truly changes
  the dependency graph: C_dS softmax/VALU can overlap with C_dQ dQ MMAC.
- Latest source-slot probe:
  workbook sheets `56_DQ_SourceSlot_NativeRing` and
  `57_DQ_DSRead_ALT_SourceSlot` now show that MLS32 direct normal/trans ALT
  readers do not produce the required `NativeDsSlotMap` q ownership.
  `normal_32x16_alt1` was the only new legal reader and gives `40/504` q-match;
  no tested legal reader reaches `504/504`.  Sheet
  `58_DQ_MLS32x16_SourceSlot` also rejects the official
  `matrix_load_32x16_b16` pair for this stricter source-slot contract: best
  q-match is still `44/504`.  Do not implement the native dS ring by adding
  gather/permute around direct-load/direct-reader routes; find a native
  producer/MMAC source-slot orientation first, or return to canonical full-3GEMM
  dQ barrier/page cadence work.
- Latest canonical dQ micro-probe:
  sheet `59_DQ_ScoreDP_Wait12` tested splitting the score/dP read wait into
  `wait12 -> dblock0 -> wait8 -> dblock1 -> wait0 -> dblock2/3`.  It passed
  correctness/resources but regressed H1/S1024 to `simTicks=36,199,800` and
  `MMAC active=27.1810%`.  The source is restored to canonical `wait_lgkm(8)`;
  do not retry finer score/dP wait splitting while PageUsed/ABarrier ownership
  remains the dominant limiter.
- Latest tail-cleanup probe:
  sheet `60_DQ_TailNoInvFastExit` tried skipping terminal
  `__syncthreads()+abarrier_inv` on the normal path because xcu showed
  `s_barrier -> s_cbranch_vccnz` at `18.28%`.  Static/resource passed, but
  H1/S128 PMD aborted with `vgpr81 is not init or has been freed` in MMOP.
  Source is restored.  Treat the terminal sync/invalidate as required by the
  current WDRA/PMD role-exit path unless a focused WDRA-exit probe proves
  otherwise.
- Latest accepted source-slot probe:
  sheet `61_DQ_SourceSlot_FastFormula` replaces the focused probe's runtime
  reverse-search source-slot mapping with a closed-form formula.  It preserves
  the verified mapping (`mismatches=0`, `mapped=504/512`) and PMD passes:
  `/zys/shaobo_runs/dq_source_slot_fast_formula_20260712_020039`,
  `ds_source_pack_cost_pass=1`, `simTicks=102,442,795`, `MMOP=2048`,
  `VALU=10,593`, `LDS=2,112`, `ldsBankConflict=0`, resource gate
  `private=0 sgpr=20 vgpr=29 no spill`.  This is probe-only; canonical
  `src/dq_kernel.cpp` remains unchanged.  It reopens the native C_dS route:
  next probe should compute real dS directly into source-slot order and feed
  `dS @ K` through `ds_write_matrix -> ds_read_matrix_trans -> MMAC` without
  gather/permute.
- Latest accepted native C_dS handoff probe:
  sheet `62_DQ_RealCDS_SourceSlot_Probe` modifies only
  `probes/dq_native_ds_source_schedule_probe.cpp`.  Accepted run
  `/zys/shaobo_runs/dq_real_ds_source_slot_bits_20260712_022239` reports
  `read_errors=0`, `mapped=504`, `frag_low/high PASS`,
  `split_low/high PASS`, `simTicks=10,236,135`, `MMOP=3`, `VALU=419`,
  `SCA=495`, `LDS=67`, `ldsBankConflict=0`, and resource gate
  `private=0 sgpr=22 vgpr=39 no spill`.  ASM includes
  `ds_write_matrix_format`, `ds_read_matrix_trans_format`,
  `ds_read_matrix_format`, and `v_mmac ... lit`; no gather/permute route is
  needed for the handoff itself.  Boundary: the accepted probe uses
  deterministic half bit-pattern dS values; the previous float-formula variant
  failed due PMD/codegen half-arithmetic noise.  Canonical integration still
  needs a C_dS publisher that computes dS with the existing arithmetic path and
  packs those values into the verified source-slot layout.
- Active accepted dQ source remains the 16-wave tail-cleanup route until a new
  prototype passes correctness/resource/perf gates.  The native dS ring is
  allowed to start as 12-wave because its role decomposition is fundamentally
  different (`producer -> C_dS -> C_dQ`), not because 12-wave is generally
  preferred over 16-wave.
- Current evidence:
  after Q/dO latch, K/V double-page reuse, K/V trans split-wait, qk/dP
  MMAC-zero seeding, and `n_tile` pair-island scheduling, H1/S128
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093036`
  and H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093048`
  both PASS; full perf archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_093242_dq_ntile_pair_island_h1s1024_sqc7_fullperf`
  reports `simTicks=40,586,455`, `kernel_ticks=36,972,845`,
  `VALU=131,168`, `SCA=87,112`, `LDS=28,656`,
  `MMAC active=25.5487%`, `ldsBankConflict=0`.
- This promotes a real pipeline change over the K-normal split-wait baseline:
  Q/dO are latched into consumer VGPR, producers wait `QDoLatched`, then reuse
  the released Q LDS region as page1 for K/V; then K/V trans fragment reads use
  `wait_lgkm(4)` for the first half and `wait_lgkm(0)` for the second half.
  The latter improves full-perf ticks about `5.04%` over the QDo-latched
  baseline.
- Latest micro-win:
  the qk/dP hot loop now seeds the first score/dP MMAC with a branch-local
  `mmac_zero` instead of zeroing `qk_acc` and `dp_acc` every `n_chunk`.
  ASM `v_mov` total drops `419 -> 359`, `v_mov_b64` drops `96 -> 36`, and
  same-shape full-perf ticks improve about `0.62%` versus the K/V split-wait
  baseline.
- Latest sidecar micro-win:
  consumer sidecar reads now use the original SoA layout with four-row Vec4
  LDS reads.  Full perf
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_113438_dq_sidecar_soa_vec4_h1s1024_sqc7_fullperf`
  improves `kernel_ticks=36,972,845 -> 35,382,165`, correctness PASS,
  no spill/scratch, and `ldsBankConflict=0`.  This is not a pipeline-quality
  win: global MMAC active slips `25.5487% -> 25.3548%`, VALU rises, and xcu
  still shows ABarrier/control as the top bubble.
- Latest structural win:
  the consumer now processes the two `n_chunk` halves of one `n_tile` together.
  This forms larger score/dP and dQ MMAC islands and removes duplicate K normal
  pair reads in the dQ update.  Versus `dq_mmac_zero_seed`, full-perf ticks
  improve about `6.33%`, LDS instructions drop `37,872 -> 28,656`, and MMAC
  active improves `24.0973% -> 25.5487%`.
- Rejected guardrail:
  removing the first `__syncthreads()` after `AllDone` caused
  `ABARRIER_ILL_OP_ERROR` (`barId 5` already invalidated), so it is required
  for barrier invalidation lifetime.
- Current xcu diagnosis:
  top bubbles remain ABarrier/control:
  `s_abarrier_try_wait -> s_xor_b32 37.26%` and
  `s_abarrier_try_wait -> s_waitcnt 10.23%` on the current sidecar SoA Vec4
  baseline.  Normal matrix-read wait is no longer the top limiter, so the next
  direction is PageFilled/PageUsed/QDoLatched cadence with a written
  data-lifetime proof, not more per-`n_chunk` local scheduling.
- Rejected next-step candidate:
  splitting startup with a new one-shot `QDoFilled` token was static-clean but
  correctness-bad: H1/S1024 failed rows `688..703` without seq, and H1/S128
  failed rows `48..63` with one-shot seq.  Do not reintroduce this token
  without a focused barrier+matrix visibility probe.
- Rejected next-step candidate:
  `Nk=32` true three-page K/V streaming was correctness/resource clean, but
  full perf regressed versus sidecar SoA Vec4:
  `kernel_ticks=35,382,165 -> 35,575,995`.  XCU still showed
  `Page1Used`-class ABarrier wait (`s_abarrier_try_wait -> s_xor_b32 37.86%`).
  The experiment is archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_121749_dq_nk32_triple_page_h1s1024_sqc7_fullperf`
  and removed from active code.  Current active source is restored to
  `b56b2dc` / `dq_sidecar_soa_vec4`.
- Rejected next-step candidate:
  half-page/n_tile `PageUsed` release kept `Mq128/Nk64` and passed
  H1/S128/H1/S1024 correctness plus static resources, but stats-only H1/S1024
  regressed `kernel_ticks=35,483,175 -> 36,212,995`; SCA grew
  `87,176 -> 101,660`.  The result is archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_124824_dq_half_page_release_h1s1024_sqc7_stats_reject`.
  Do not pursue finer page tokens as an isolated optimization.
- Rejected next-step candidate:
  `Mq128/Nk128` with direct consumer sidecar global reads passed static
  resources (`private=0`, no spill/scratch, consumer `163/216`) but failed
  H1/S128 correctness with all dQ values NaN.  Adding an explicit VMEM/LDS
  wait did not fix it, and an `Nk64` direct-sidecar diagnostic failed the same
  way.  Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_130949_dq_direct_sidecar_correctness_reject`.
  Sidecar LDS staging remains required in the current main path.

## dQ Reopen Contract

- Active branch: `shaobo/dq-xcu-guided-dq-kernel`.
- Design workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`.
- Scope: implement a standalone dQ kernel after the dKV focused line.
- Output ownership: Q tile owns dQ and stores once after reducing across all
  K/V tiles; no atomic add in the first dQ path.
- Algorithm boundary: because dKV and dQ are separate kernels, dQ may recompute
  score/dP across kernels, but must not duplicate score/dP for the same
  `(Q tile, K tile)` inside dQ.
- Current correctness-clean MMAC tile: `Mq=128,Nk=64,D=128,16 waves`.
- Current dQ pipeline baseline is a single K/V page with two producer groups
  and two symmetric consumer groups.  Q/dO are loaded once per CTA q tile,
  sidecar is staged in LDS, and K is read by normal `ds_read_matrix` for
  `dS K`.
- `PageUsed` is consumer-only in the current code; worker-side PageUsed arrival
  was removed after proving it was redundant for page overwrite lifetime.
- The previous Mq32 K-native path remains the performance reference, but the
  active source follows the new 16-wave ownership because it matches the
  required dQ topology.
- `Nk=128` or double-buffered K/V is a later upgrade only after Q/dO lifetime
  release is proven.
- Producer rule: producer publishes Q/dO plus packed sidecar to LDS, and streams
  K/V through LDS; consumer should not direct-load sidecar global in the hot
  path.
- Evidence flow: design workbook -> code -> static gates -> H1/S128
  correctness -> H1/S1024 PMD/xcu diagnosis -> update workbook/ledger/log.

## Active dQ Target

- Current target: `MMAC active >= 40%` on
  `B=1,H=1,S=1024,D=128,causal=true`, `GPU_CHIP=sb`,
  `GPU_ARGS="['--SQCIPfLines=7']"`.
- Current canonical dQ source path: `Mq=128,Nk=64,D=128,16 waves`,
  two producer groups and two symmetric full-3GEMM consumer groups.  dS stays
  in VGPR; there is no dS LDS handoff and no separate dS worker.
- Previous Mq32 K-native path remains the small-shape performance reference:
  `Mq=32,Nk=64,D=128,12 waves`, sidecar-LDS, two
  K/V/K-to-dS-pad/dS pages, same-K LDS normal read for dQ RHS, no hot Kt
  source path.
- Previous Mq32 recertified baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063111`,
  aggregate `MMAC active=8.8385%`, `simTicks=52,082,485`, correctness PASS,
  `ldsBankConflict=0`.
- S1024 one-dispatch measurement knob:
  `DQ_TILES_PER_DISPATCH=32`, run
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063334`,
  `simTicks=34,346,130`, `MMAC active=8.2338%`.  Use this for cleaner perf
  capture, but do not count it as an optimization.
- Latest xcu bottleneck:
  one-dispatch full perf
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_072949`
  is dominated by `DsFilled` ABarrier wait bubbles.  Kt preread under this wait
  was correct/resource-clean but regressed to `simTicks=34,237,840`,
  `MMAC active=8.1943%`, so it was removed.
- Previous Mq32 accepted one-dispatch improvement:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082245`,
  `simTicks=33,372,430`, `MMAC active=8.44342%`, `SCA=212,520`,
  correctness PASS, `ldsBankConflict=0`.
- Previous Mq32 accepted dQ micro-baseline:
  worker score/dP read-batch in `dq_publish_ds_chunk`, recorded in workbook
  sheet `32_dq_worker_readbatch`.  It batches four K-block `dO/K/V`
  `ds_read_matrix` groups before one `wait_lgkm(0)` and a longer MMAC island.
  H1/S1024 one-dispatch correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_094409`,
  `simTicks=30,225,650`, `MMAC active=9.25852%`, `coissue=1,864/1,455`,
  `ldsBankConflict=0`.  Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_094707_dq_worker_readbatch_s1024_sqc7_fullperf`.
  XCU still shows the main blocker is ABarrier:
  `s_abarrier_try_wait -> s_xor_b32` about `44.64%`.
- Previous Mq32 best dQ micro-baseline:
  K-native same-LDS RHS plus all-operand worker read-batch.  It removes the
  separate `K^T` source-layout page and dead host/API `k_t_source` tail while
  keeping the K-to-dS padding that preserved the best LDS offsets.  H1/S1024
  canonical stats:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_144004`,
  `simTicks=28,002,520`, `MMAC active=10.032187%`,
  `coissue=1,964/1,441`, `SCA=191,696`, `VMEM=4,608`,
  `ldsBankConflict=0`.  Latest full perf before host cleanup remains:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_141848_dq_current_best_h1s1024_sqc7`.
  XCU still shows the main blocker is ABarrier/Page/Ds ownership:
  `s_abarrier_try_wait -> s_xor_b32` about `45.61%`.
- Mq64 single-page direct/split variants are rejected: they are correct and
  resource-clean, but lose overlap/coissue and regress ticks badly.
- Next evidence step: stay on Mq32 two-page baseline, preserve worker/consumer
  overlap, and attack page/dS ABarrier ownership or increase useful MMAC per
  token before another larger-tile redesign.

## Current Canonical State

- Repo focus: `/zys/shaobo/fa3_bwd_wasp_clean`.
- Active candidate tile: `Mq=128, Nk=128, D=128`, 16 waves, `GPU_CHIP=sb`,
  `GPU_ARGS="['--SQCIPfLines=7']"`.
- Wave roles:
  - waves0-3: producer K + Q + sidecar
  - waves4-7: consumer group 0, owns `Nk=0..63`
  - waves8-11: consumer group 1, owns `Nk=64..127`
  - waves12-15: producer V + dO
- Main path: `matrix_load_32x16/32x32 ... bps lds` +
  normal/trans `ds_read_matrix` + `v_mmac_*lit`.
- Q and dO are split into two M64 semantic half-page ownership regions on the
  same Mq128 physical LDS pages:
  `Q0=bar2/3`, `Dout0=bar4/5`, `Q1=bar6/7`, `Dout1=bar8/9`;
  `AllDone=bar10`. K/V is latched into consumer VGPR, then the raw pages
  overlay the K/V LDS region.
- Canonical target path is exact causal tiles only:
  `causal == 1`, `seqlen_q % Mq == 0`, `seqlen_k == seqlen_q`,
  `seqlen_k % Nk == 0`.
- The hot softmax/dS helper uses exact-tile causal predicate
  `owner_krow <= qrow` to remove runtime seqlen/full-valid control.
- Output ownership is unique: every consumer owns one `Nk16 x D128` dV/dK
  slice. Do not duplicate score/dP for the same owner.

## Latest Evidence

- dQ chunk-token experiment was rejected and removed from source.  It replaced
  full-page `DsFilled` with per-worker dS chunk barriers, passed static and
  H1/S128/H1/S1024 correctness, but H1/S1024 dispatch1 regressed from
  `28,114,905` ticks / `9.7068%` whole-active MMAC to `31,380,440` ticks /
  `8.7319%` whole-active MMAC.  The conclusion is that finer-grain ABarrier
  readiness alone raises scalar/control debt faster than it exposes useful dQ
  MMAC.
- dQ Mq64 with a new `QDoUsed` q-subtile token also rejected.  Static/resource
  was clean (`private=0`, `sgpr=69`, no spill), but H1/S128 PMD hung at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_034822`.
  A follow-up with independent `page0_seen/page1_seen` fixed one real page
  overwrite bug but still hung at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_035807`.
  Larger Mq remains a likely 40% route, but it needs a focused q_subtile
  ABarrier ownership probe before re-entering the performance kernel.
- Added `probes/dq_qsubtile_barrier_probe.cpp`.  It does not hang, but scalar
  LDS checks fail (`errors=16`, `done=0`), so it is not a faithful proof for
  the FA matrix path.  Next synchronization probe must use
  `matrix_load ... bps lds` plus `ds_read_matrix`.
- Current dQ source is restored to `dq_sidecar_lds_staging`; remote recertified
  static PASS with `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch,
  consumer branch `49/72`, worker `83/128`.
- The next dQ 40% route should reduce token count or increase useful MMAC per
  ownership epoch, not split existing tokens more finely.

- dQ current baseline:
  `Mq=32,Nk=64,D=128,12 waves`, two K/V/Kt/dS LDS pages.
- dQ H1/S1024 correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015941`,
  `dq_max_abs=1.85174e-07`, `dq_rel_l2=0.00208192`, `bad=0`, `pass=1`.
- dQ static/resource:
  `private=0`, `sgpr=61`, `vgpr=168`, no spill; branch windows producer
  `1/40`, consumers `49/72`, worker `91/128`.
- dQ H1/S1024 stats:
  dispatch0 `kernel_ticks=21,420,035`, `MMAC active=5.8039%`,
  coissue `245/204`; dispatch1 `kernel_ticks=35,671,545`,
  `MMAC active=7.8501%`, coissue `751/665`; `ldsBankConflict=0`.
- dQ is still far from the 40% MMAC-active target.  The next dQ change must
  increase useful MMAC island/role balance, not just replace barriers.

- Current best clean micro-baseline: sidecar Vec4 LDS read aggregation on top
  of workbook sheet `71_mq128_score_dp_read8_design`.
  It changes only the softmax/dS sidecar reads from scalar per-row loads into
  `Vec4F32` loads for row max-log2, inverse-sum, and delta.
- Sidecar Vec4 H1/S1024 full-perf stats:
  `kernel_ticks=44,260,125`, `simTicks=47,873,735`,
  `MMAC active=32.6559%`, `MMOP=131,072`, `VALU=183,136`,
  `SCA=115,608`, `LDS=79,360`, `VMEM=4,352`,
  `coissue=36,479/26,644`, `ldsBankConflict=0`.
- Sidecar Vec4 static/resource:
  branch windows producer0 `14/16`, consumer0 `188/240`,
  consumer1 `188/240`, producer1 `8/16`; metadata `private=0`,
  `sgpr=99`, `sgpr_spill=0`, `vgpr=128`, `vgpr_spill=0`;
  asm `ds_read_b32=0`, `ds_read_b128=96`, `ds_read_matrix=550`.
- Sidecar Vec4 full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_194400_7gemm_sidecar_vec4_h1s1024_sqc7_fullperf`.
- Sidecar Vec4 xcu result:
  dispatch duration `103,988 -> 97,276` vs read8, avg active waves
  `115.47 -> 120.93`; the old `ds_read_b32 -> s_waitcnt` sidecar bubble
  disappears.  The top bottleneck remains ABarrier:
  `s_abarrier_try_wait -> s_xor_b32` about `41.86%`.
- Current best clean micro-baseline: workbook sheet
  `71_mq128_score_dp_read8_design`.
  It changes only `score_dp_mmac_owner16` from four small
  `4 ds_read_matrix_trans -> wait -> 8 MMAC` islands into two larger
  `8 ds_read_matrix_trans -> wait -> 16 MMAC` islands.
- Read8 H1/S1024 full-perf stats:
  `kernel_ticks=47,313,175`, `MMOP=131,072`, `VALU=165,872`,
  `SCA=115,608`, `LDS=83,856`, `VMEM=4,352`,
  `coissue=36,333/25,091`, `ldsBankConflict=0`.
- Read8 static/resource:
  branch windows producer0 `14/16`, consumer0 `180/240`,
  consumer1 `180/240`, producer1 `8/16`; metadata `private=0`,
  `sgpr=99`, `sgpr_spill=0`, `vgpr=128`, `vgpr_spill=0`.
- Read8 full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_063337_clean_read8_score_dp_h1s1024_sqc7_fullperf`.
- Read8 xcu result:
  dispatch duration `106,108 -> 103,988` vs half-page, avg active waves
  `114.79 -> 115.47`, `v_mmac -> s_waitcnt` gap `3.92% -> 1.62%`,
  `s_waitcnt -> v_mmac` `0.96% -> 0.60%`.  The top bottleneck remains
  `s_abarrier_try_wait -> s_xor_b32` at about `40.93%`.
- Half-page conveyor H1/S1024 full-perf stats:
  `kernel_ticks=48,279,140`, `MMAC active=31.7858%`, `MMOP=131,072`,
  `VALU=165,872`, `SCA=115,608`, `LDS=83,856`, `VMEM=4,352`,
  `coissue=33,962/22,131`, `ldsBankConflict=0`.
- Static/resource for the half-page conveyor:
  branch windows producer0 `14/16`, consumer0 `180/240`,
  consumer1 `180/240`, producer1 `8/16`; metadata `private=0`,
  `sgpr=99`, `sgpr_spill=0`, `vgpr=128`, `vgpr_spill=0`.
- Half-page full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_053321_clean_half_page_conveyor_h1s1024_sqc7_fullperf`.
- Half-page xcu top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` is still dominant at `40.42%`;
  `s_abarrier_try_wait -> s_waitcnt` is `9.07%`; `v_mmac -> v_mmac`
  is `7.50%`.
- Half-page focused windows:
  `bar3 Q0Used` same-SIMD bubble `96.04%`, `bar5 Dout0Used` `94.39%`,
  and `bar7 Q1Used` `94.25%`.  Individual cliffs are shorter than the Q/dO
  split, but they are still not covered by useful peer-wave work.
- Rejected follow-up: sheet `70_mq128_half_ring3_design` implemented a
  three-slot M64 ring (`Slot0/1/2 Filled/Used`, count=8) and passed
  static/resource/correctness, but regressed H1/S1024 stats-only:
  `kernel_ticks=50,617,385`, `MMAC active=30.2521%`, `SCA=213,896`
  versus same-debug half-page `48,268,220`, `31.6990%`, `SCA=115,608`.
  Do not keep or retry ring depth alone; pairing Q and dO at slot granularity
  lost the early lifetime benefit and added control/SCA cost.
- Q/dO lifetime split comparison:
  `kernel_ticks=51,238,915`, `MMAC active=29.6586%`.
- 62C2 H1/S1024 full-perf stats:
  `kernel_ticks=52,163,020`, `MMAC active=29.2001%`, `MMOP=131,072`,
  `VALU=167,536`, `SCA=106,968`, `ldsBankConflict=0`.
- Raw2 recert comparison:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`, `VALU=181,980`,
  `SCA=296,328`.
- Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_033019_clean_62c2_mq128_h1s1024_sqc7_fullperf`.
- XCU evidence:
  dispatch0 duration `114,644`, avg active waves `117.91`;
  top bubbles are `s_abarrier_try_wait -> s_xor_b32` `44.65%` and
  `s_abarrier_try_wait -> s_waitcnt` `9.57%`.
- XCU top2000 diagnosis:
  `bar3 Raw0Used` totals `4,623,276` cycles across `448` bubbles
  (`avg=10,319.8`, `max=13,427`, window `7700:94648`);
  `bar6 AllDone` totals `1,238,870` cycles across `110` bubbles.
- Representative xcu window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/62c2_mq128_h1s1024_fullperf_20260705_033019_dispatch0_window_bar6`;
  `93000:113000`, `xcd0,se1,cu0,simd1,wave0`,
  `Bubble=96.51%`, `MMAC=0.70%`.
- Focused Raw0Used window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/62c2_mq128_h1s1024_raw0used_window_7000_22000`;
  `7000:22000`, `xcd0,se0,cu0,simd3,wave3`,
  pipeline `Bubble=98.60%`, same-SIMD mix `Bubble=95.59%`,
  `MMAC=0.85%`, `VALU=1.19%`.  Consumer slots issue MMAC but
  MMAC+VALU coissue is only `6.54%` and `8.27%`.
- Rejected after raw2: raw3 page depth, consumer1 score-prefetch stagger,
  causal=true specialization, score read batch2, WG-local duplicate Q/dO, and
  causal invalid-prefix page skip, raw release before softmax, and sidecar
  ring3 early release, producer sidecar rebalance, full-valid mask shrink, and
  sidecar register prefetch.  All
  resource/correctness-clean candidates regressed or failed to improve
  same-shape performance.
- Rejected before 62C2: direct Mq128/C208, Mq128/C240, and 62A still spilled
  SGPR.  62C2 is the first Mq128/R1 version that is no-spill and faster.
- Rejected after 62C2: sheet 63 sidecar/raw lifetime split with early RawUsed
  release.  It was static-clean (`private=0`, `sgpr=98`, no spill, consumer
  `190/240`) and H1/S128 passed, but H1/S1024 failed for both no-wait and
  wait-before-release variants.  Do not keep or retry release-before-softmax in
  the canonical kernel without a focused lifetime/instruction probe.
- Rejected after 62C2: sheet 64 full-valid softmax helper.  It targets xcu
  softmax branch bubbles and covers 48.4375% of M-pairs analytically, but both
  saved-bool and inline-predicate variants failed metadata with
  `sgpr_spill=4`.  Do not stack another full-valid branch into the hot helper
  unless the design first removes SGPR/control pressure.

## Current Diagnosis

The kernel does have MMAC and the matrix path is not the primary missing piece.
Read8 is the current best clean micro-baseline, but the active limiter is still
packet ownership and barrier lifetime:

- ABarrier waits dominate xcu despite better ticks and active share.  The main
  steady culprits are now split across half-page `Q0/Dout0/Q1` waits, not
  eliminated.
- `ds_read_matrix -> wait -> MMAC` is visible but secondary versus
  `s_abarrier_try_wait` bubbles.
- The Mq128 exact-tile route plus half-page split plus read8 score/dP batching
  is a better baseline, not a 60% active solution.
- Assembly is not the next default move; only a proven hot island should become
  asm, and only after topology/resource work fails.

## Next Experiment

Use the read8 score/dP batching source as the canonical baseline before the next
experiment.  The next design should be workbook-first and must attack the
remaining ABarrier/consumer lockstep, not repeat ring-depth alone and not just
batch more reads.  A viable candidate needs one of:

- preserve early Q/dO lifetime separation while adding useful lookahead;
- create real consumer-group softmax/dS or MMAC work under half-token waits;
- lengthen useful MMAC islands without duplicating score/dP or increasing
  ABarrier/SCA more than the work it hides.

This remains a topology/resource problem first.  Assembly is still only a
last-resort short island after xcu/asm proves a specific compiler-generated
loop is the blocker.

Do not treat this split as the 60% active solution; it is a measured micro
baseline and a better attribution scaffold.
Do not directly retry sheet 63's release-before-softmax path: long q-loop
correctness failed even though H1/S128 passed.
Do not directly retry sheet 64's full-valid two-path helper: it fails the
no-spill static gate.
Do not directly delete the tail `AllDone` ABarrier: sheet
`67_mq128_prune_alldone` showed static regression to
`private_segment_fixed_size=244` and `vgpr_spill_count=60`, likely because the
current WDRA CFG/codegen uses it to limit post-branch live range.
Workbook sheet `68_qdo_focused_xcu` adds the current focused-window rule:
`bar3 QUsed` and `bar5 DoutUsed` representative windows still show about
`95%` same-SIMD bubble and only about `1%` MMAC; visible coissue is mostly
`v_mov`, not useful softmax/dS.  The next successful topology must improve
these focused windows, not only aggregate coissue.

Workbook sheet `51_structural_pivot` records the rejected WG-local duplicate
Q/dO structural probe:

```text
WG0: waves0-3 producer K0/V0 + Q/dO + sidecar
     waves4-7 consumer0, owns Nk0..63

WG1: waves12-15 producer K1/V1 + Q/dO + sidecar
     waves8-11 consumer1, owns Nk64..127
```

It passed correctness/resource gates but regressed H1/S1024 full perf to
`simTicks=58,310,070`, `MMAC active=26.7125%`, and doubled VMEM. Do not pursue
independent warpgroup ownership by duplicating shared Q/dO.

Workbook sheet `52_causal_page_skip` records a rejected algorithm candidate:

```text
If causal && q_tile_end < k_base:
  producers advance the raw token without Q/dO/sidecar MLS publication;
  consumers wait the raw token, immediately arrive RawUsed, and skip
  score/dP/softmax/dV/dK for that page.
```

The resource-clean prefix-only implementation passed H1/S128 and H1/S1024
correctness and reduced H1/S1024 MMOP from `131,072` to `77,312`, but
`kernel_ticks` stayed flat/slightly worse at `53,474,330` and MMAC active
dropped to `22.4979%`. Do not keep this consumer-side skip in the canonical
route; future causal work needs launch/tile ownership or critical-path
restructuring.

Workbook sheet `53_score_dp_pair_asm` records the first asm-island negative:

```text
Keep the C++ clean kernel and replace only the hot score/dP read island:
  four independent ds_read_matrix_trans reads
    -> two ds_read_matrix_trans_pair asm helper calls
```

It built cleanly and produced the intended pair-read asm, but H1/S128
correctness failed. Root cause: the existing `ds_read_matrix_trans_pair`
assumes the second fragment is `+1024` bytes from the first, while raw Q/dO
M0/M1 score/dP reads are separated by `4 * 1024` bytes in the D128 raw page.
The source is reverted to the raw2 baseline. Future asm islands must first
prove the exact LDS adjacency relation for the target layout.

Workbook sheets `54_raw_release_before_softmax` and
`55_sidecar_ring3_early_raw_release` record two raw-lifetime negatives:

```text
54: pre-read Q/dO source and arrive RawUsed before softmax
    -> H1/S1024 correctness fails because sidecar is still live.

54b: pre-read sidecar rows to VGPR too
    -> static fails with private=52, vgpr_spill=24.

55: keep raw2 but add sidecar ring3, then release raw before softmax
    -> correctness/resource pass, but H1/S1024 kernel_ticks=55,298,425,
       MMAC active=26.5015%, worse than raw2 recert 53,008,410 / 27.7754%.
```

Do not retry page/ring-depth work in isolation.  A future raw-lifetime design
must also remove ownership/control cost or enlarge a useful MMAC island.

Workbook sheet `56_producer_sidecar_rebalance` records a producer topology
negative:

```text
producer0: K + Q + sidecar   ->   K + Q
producer1: V + dO            ->   V + dO + sidecar
```

It passed correctness/resource gates and moved branch windows from
`6/198/198/1` to `1/198/198/6`, but H1/S1024 regressed to
`kernel_ticks=53,558,960`, `MMAC active=27.5554%`.  Coissue rose, but the
RawUsed/consumer critical path did not shorten.  Do not equate producer visual
balance with useful pipeline overlap.

Workbook sheet `57_full_valid_mask_shrink` records a correctness negative:

```text
If an owner16 M-pair is fully causal-valid:
  skip per-element valid_pair checks in softmax/dS
```

It passed static/resource gates with branch windows `6/197/197/1` and
metadata `private=0`, `sgpr=66`, `vgpr=112`, no spill/scratch, but H1/S128
failed with `dv_rel_l2=33.2914` while dK stayed close. PMD also warned
`read vgpr165 before writing`. Do not remove per-element `valid_pair` from the
main dKV helper without a focused fragment/codegen probe first.

Workbook sheet `58_sidecar_reg_prefetch_wait` records a producer-thickening
negative:

```text
load sidecar row_max/sum/delta into producer VGPR before RawUsed wait
write sidecar to LDS only after page ownership is granted
```

It was resource-clean and correct, with branch windows still `6/198/198/1`,
but H1/S1024 regressed to `kernel_ticks=53,658,605` and
`MMAC active ~=27.4726%` versus raw2 `53,008,410` / `27.7754%`. Tiny producer
prefetch is not enough to move the RawUsed critical path.

Workbook sheet `59_half_page_raw_tokens` records an ownership-granularity
negative:

```text
Raw0/Raw1 whole-page tokens
  -> Raw0Half0, Raw0Half1, Raw1Half0, Raw1Half1 Filled/Used tokens
```

It passed static/resource gates (`private=0`, `sgpr=62`, `vgpr=112`,
no spill/scratch) and H1/S128/H1/S1024 correctness, but H1/S1024 regressed to
`kernel_ticks=53,505,270`, `MMAC active=27.3801%`, `SCA=330,730` versus raw2
`53,008,410`, `27.7754%`, `SCA=296,328`.  The code is reverted to raw2
whole-page tokens.  Do not split RawUsed finer as a standalone fix; it adds
protocol/control cost unless paired with a larger useful MMAC island or fewer
ownership handshakes.

Workbook sheet `60_mq128_singlebuf_static` records a larger-tile resource
negative:

```text
Mq64/R2 static two-pair chain
  -> Mq128/R1 static four-pair chain
```

The LDS budget works only with one raw page, but the direct static four-pair
consumer expansion failed metadata before PMD:
`private=8`, `sgpr=104`, `sgpr_spill=18`, `vgpr_spill=2`, with both consumer
branches at `208/208`.  The code is reverted to raw2.  Larger Mq still looks
like the right architectural lever, but it needs a resource redesign first;
do not keep a dynamic Mq128 loop as the performance route.

Workbook sheet `61_mq128_vgpr240_retest` records the follow-up resource
boundary:

```text
Mq128/R1 static four-pair chain
  consumer window 208 -> 240
  per-SIMD ledger P16 + C240 + C240 + P16 = 512
```

This removed the sheet-60 private/VGPR spill, but metadata still failed before
PMD: `private=0`, `sgpr=100`, `sgpr_spill=18`, `vgpr=128`,
`vgpr_spill=0`, with consumer branches at `209/240`.  The code is reverted to
raw2 and remote recert PASS (`private=0`, `sgpr=60`, `vgpr=112`,
no spill).  Conclusion: static Mq128 is blocked by SGPR/control live range and
helper shape, not by consumer VGPR window alone.  Next larger-Mq work must
redesign scalar lifetime/phasing before another PMD run.

Workbook sheet `62_mq128_sgpr_control_shrink` is the next design basis:

```text
62A: specialize the hot canonical route to causal=true and shrink control args
62B: if needed, split Mq128 into two lexical Mq64 halves without new tokens
62C: if needed, split softmax/dS helper around precomputed q/k scalars
```

62A has now been tested as a static-only probe.  It helped but did not pass:
`sgpr_spill` dropped from `18` to `14`, and consumer branch windows improved
from `209/240` to `182/240`, but metadata still failed with `sgpr=100`.
No PMD was run, and the code is reverted to raw2.  The lesson is useful:
causal/control state is part of the blocker, but 62A alone is too weak.
Continue with 62B/62C only if they further reduce scalar/control live range
without device-call private segment or dynamic Mq128 loop overhead.

Current accepted instruction-level baseline is `w16_mq128_sidecar_pair_read6`
on the canonical dKV kernel.  It keeps Mq128/Nk128/16-wave and only changes the
softmax sidecar schedule: read both M rows' sidecar Vec4 triples, then compute
both rows.  H1/S1024 full perf improved from wait-prune
`simTicks=47,871,005`, `MMAC active=32.7888%`, `VALU=183,136` to
`simTicks=47,731,775`, `MMAC active=32.8831%`, `VALU=168,514`, with no
spill/scratch and `ldsBankConflict=0`.  Perf archive:
`/Volumes/172.20.68.76/共享/shaobo/perf/20260706_215636_7gemm_sidecar_read6_h1s1024_sqc7_fullperf`.
This is a micro-baseline; XCU still shows ABarrier/control as the main bubble.

## Workflow Rules

1. Update the shared workbook before changing tile shape, output ownership,
   ABarrier ledger, or operand lifetime.
2. Keep one canonical dKV kernel. Rejected experiments are reverted and recorded
   in `results/perf_ledger.csv` and the workbook.
3. Run static gates, H1/S128 correctness, H1/S1024 correctness, PMD stats, then
   XCU CLI. GUI Wavefronts is optional human review only.
4. Judge by MMAC active share first, then same-shape ticks and SQTT explanation.
   Coissue dominated by `v_mov` is not a success.
5. Commit every accepted or rejected evidence checkpoint before starting the
   next hypothesis.

## dQ Current State

Active branch: `shaobo/dq-xcu-guided-dq-kernel`.

Active dQ kernel: `src/dq_kernel.cpp`, canonical path
`kDqPathCanonicalDq`, tile `Mq=128,Nk=64,D=128`, 16-wave CTA.

Current accepted structural dQ candidate is `dq_mq128_16wave_full3gemm`:

- Producers stage Q/dO group sidecar into LDS and stream K/V through one LDS
  page.
- Two symmetric consumer groups own different q-row ranges and each compute
  the full dQ chain locally: `QK^T`, `dO V^T`, dS, `dS K`.
- dS stays in VGPR.  There is no dS LDS transfer and no separate dS worker.
- Static/resource PASS: `private=0`, `sgpr=76`, `vgpr=128`,
  no spill/scratch; branch windows are `8/40`, `117/216`, `117/216`,
  `9/40`.
- Correctness PASS H1/S128 and H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160156`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160322`.
- H1/S1024 full perf:
  `simTicks=55,191,955`, `MMAC active=19.1324%`,
  `ldsBankConflict=0`.

This is structurally correct but not the small-shape performance winner.  The
previous Mq32 K-native route remains the performance reference at
`simTicks=28,002,520`, but it used the old split dS-worker topology that the
current source intentionally replaced.  The next dQ design question is how to
reduce ABarrier PageFilled/PageUsed exposure or increase useful MMAC work per
ownership epoch in the 16-wave full-3GEMM topology.

Fast same-K-LDS probe result:

- Earlier guessed combinations of K `matrix_load_32x32`/`matrix_load_32x16` plus
  dQ `ds_read_matrix_32x16_normal/trans` were static-clean but numerically
  wrong on H1/S128 (`dq_rel_l2` between `0.535917` and `1.46283`).
- A later focused fragment-layout probe found the accepted f16x4 K fragment
  remap used by the current canonical path.  Do not restore the Kt page by
  guessing offsets; use focused probes and gates.

Latest dQ evidence:

- Fresh xcu for `dq_sidecar_lds_staging`:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2732630_fa3_bwd_dq_clean_20260707_030530`.
- Dispatch1 top issue gap is still ABarrier:
  `s_abarrier_try_wait -> s_xor_b32 = 53.17%`.
- Use whole-active MMAC as the target metric:
  `sum(mmopRunTimeCounter) / sum(activeTimeCounter)`.  The local busy-window
  ratio `sum(mmopRunTimeCounter) / sum(runTimeCounter)` is already about
  `40.8%`, but it hides ABarrier idle/bubble time and is not the 40% target.
- PageUsed `8 -> 4` was logically correct and raised whole-active MMAC
  `9.7068% -> 9.9346%`, but regressed dispatch1 ticks
  `28,114,905 -> 28,360,605`; code reverted.
- Nk128 single-page was rejected at static metadata:
  `private=68`, `sgpr_spill=2`, `vgpr_spill=64`; code reverted and remote
  recertified to the sidecar-LDS baseline.

Latest q_subtile ownership probe:

- `probes/dq_qsubtile_matrix_probe.cpp` is accepted as focused protocol
  evidence for the next Mq64 attempt.
- It uses the matrixized path (`matrix_load_32x32_b16 ... bps lds` plus
  `ds_read_matrix_32x16_trans`) and repeats page0 across two q_subtiles.
- PMD `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_matrix_probe_20260707_042323`,
  `errors=0`, `done_waves=12`, `pass=1`, `ldsBankConflict=0`.
- Implication:
  the prior Mq64 hangs are likely in the full kernel's actual wait/release
  wiring, not a fundamental ABarrier or matrix-path limitation.  Next canonical
  dQ edit should retry Mq64 surgically with this release order, then measure
  H1/S1024 whole-active MMAC toward the 40% target.

Latest Mq64 main-kernel retry:

- Retried the canonical dQ kernel with `Mq=64,Nk=64`; resource gates remained
  clean (`private=0`, `vgpr=168`, `sgpr=69..72`, no spill/scratch).
- H1/S128 and H1/S64 both hung under PMD, so the code was reverted and the
  remote source was recertified to the Mq32 sidecar-LDS baseline.
- ABarrier logs exposed one concrete bug: a counted ABarrier is not a
  unique-wave barrier.  Fast worker waves can arrive a later q_subtile before
  slow waves finish the earlier q_subtile unless they also wait after arrive.
- Even after fixing that QDo phase hazard, the full kernel still stalled around
  the `DsFilled` / consumer path.  The accepted q_subtile probe was therefore
  too thin: it did not include realistic dS publication plus consumer dQ work.
- Next step for 40% MMAC active:
  build a focused worker+consumer+dS publication probe before touching
  `src/dq_kernel.cpp` again.  The active performance code remains the accepted
  Mq32 sidecar-LDS baseline.

Follow-up probe result:

- `probes/dq_qsubtile_ds_consumer_probe.cpp` passed PMD:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_ds_consumer_probe_20260707_050746`,
  `errors=0`, `done_waves=12`, `consumer_epochs=8`, `pass=1`.
- This probe includes unequal worker progress, dS-like LDS publication,
  `DsFilled`, consumer matrix reads/MMAC, `PageUsed`, and `QDoUsed`.
- Interpretation:
  the q_subtile barrier protocol is viable even with a consumer path.  The full
  Mq64 hang should be narrowed next to the real dQ helper bodies
  (`dq_publish_ds_chunk` and `dq_consume_ds_kt_full_dtile`) before any further
  canonical-kernel edit.

Current dQ baseline for the 40% MMAC-active goal:

- Active code is the Mq32/Nk64/D128 two-page `dq_pageused_consumer_only`
  variant: PageUsed is consumer-owned (`4` arrivals), workers no longer arrive
  PageUsed.  This supersedes the older dispatch1-only note above where
  PageUsed was reverted.
- Accepted H1/S1024 one-dispatch stats:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082245`,
  `simTicks=33,372,430`, `MMAC active=8.44342%`, `MMOP=52,224`,
  `coissue=1,223/992`, `ldsBankConflict=0`.
- Full perf/xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082827/m5out/0/0/2739404_fa3_bwd_dq_clean.perf`;
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_pageused_s1024_fullperf_20260707_082827`.
- Main xcu conclusion:
  `s_abarrier_try_wait -> s_xor_b32` is still the top issue bubble (`46.01%`),
  followed by `s_abarrier_try_wait -> s_waitcnt` (`6.60%`).  The next 40%
  route should redesign page/dS ownership and reduce ABarrier handoff bubbles,
  not only move local `wait_lgkm`.
- Rejected low-level candidate:
  merging the two worker dS store waits into one page-level wait passed
  correctness but regressed ticks to `33,729,150`; it was reverted from source.
- Rejected low-level candidate:
  switching PageFilled/DsFilled/PageUsed waits from inline-asm
  `abarrier_try_wait<true>` to builtin `abarrier_try_wait<false>` passed
  correctness/resource gates but regressed ticks to `33,754,630`; it was
  reverted.  The `s_xor_b32` hot row should be read as true ABarrier wait debt,
  not merely wrapper overhead.
- Rejected structural candidate:
  splitting `DsFilled` into pair0/pair1 was correct but did not improve
  performance.  Pair-all workers regressed to `33,548,970` ticks because both
  pairs were still ready together; two-worker sequential pair streaming
  regressed to `33,989,410` ticks because waves10-11 became thin.  The active
  source is restored to the accepted PageUsed consumer-owned baseline.

Current dKV fixed-env status for the 60% MMAC-active route:

- Active dKV code is the 16-wave Mq128/Nk128/D128 canonical dKV path:
  waves0-3 publish K + Q + sidecar, waves12-15 publish V + dO, and
  waves4-7 / waves8-11 are symmetric consumer groups over different K rows.
- The 2026-07-09 high-source split-wait patch is accepted only as a micro
  wait-late improvement.  It changes `dv_dk_mmac_owner16_read4x2` so high
  dV/dK source reads are issued before the first wait, then uses
  `wait_lgkm(8)` to run the low MMAC island while high reads remain in flight.
- Evidence:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_003152_dkv_splitwait_h1s1024_sqc7_fullperf`.
  H1/S1024 full perf improves `simTicks 48,274,135 -> 47,484,710`
  (`-1.64%`) versus the fixed-env current full-perf baseline, with correctness
  PASS, no spill/scratch, and `ldsBankConflict=0`.
- Boundary:
  full-perf MMAC active is essentially neutral/slightly down
  `32.9839% -> 32.9468%`; xcu wait rows improve slightly
  (`s_waitcnt 19.74% -> 19.55%`, normal matrix-read wait
  `3.50% -> 3.26%`).  This is not a structural 60% path.
- Current first-order dKV bottleneck remains ABarrier ownership:
  `s_abarrier_try_wait -> s_xor_b32` is still about `41.38%`, and
  `s_abarrier_try_wait -> s_waitcnt` remains about `8.59%`.
  Next serious work should reduce Q/Dout half-page ownership cliffs or
  increase useful MMAC per ownership epoch, not only polish local waits.

Current SQTT diagnosis for the dQ 40%+ MMAC-active route:

- See `results/dq_sqtt_bottleneck_20260708.md`.
- The current `dq_sidecar_soa_vec4` H1/S1024 full-perf artifact is
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_113438_dq_sidecar_soa_vec4_h1s1024_sqc7_fullperf`.
- dQ PMD: `kernel_ticks=35,382,165`, `MMAC active=25.35%`,
  `MMOP runtime share=41.86%`, `MMOP=55,296`, `VALU=138,208`,
  `SCA=87,176`, `coissue=17,446/16,910`, `ldsBankConflict=0`.
- dQ SQTT: top gaps are `s_abarrier_try_wait -> s_xor_b32 37.26%` and
  `s_abarrier_try_wait -> s_waitcnt 10.23%`; matrix read to wait/MMAC is not
  the first-order limiter in this capture.
- FWD H4/S1024 reference has `mmop_fp16 45.96%` SQTT hot-instruction share and
  only `salu_32 7.30%`, while dQ has `v_mmac 8.85%`, `s_xor_b32 35.36%`,
  and `s_waitcnt 16.44%`.
- Working diagnosis:
  dQ's three GEMM islands are being diluted by the PageFilled/PageUsed/
  QDoLatched ownership ledger.  The next useful optimization must increase
  useful MMAC per ownership epoch or reduce PageUsed critical-path exposure;
  wait/vmov cleanup is secondary until the ABarrier cliff moves.
- Evidence gap:
  the diagnostic is H1/S1024 for dQ versus H4/S1024/H4/S2048 for FWD.  Before
  claiming FWD-style acceptance, capture same-shape dQ H4/S1024 SQTT under
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.

Rejected dKV release-half Q-read-ahead candidate:

- Candidate:
  moved release-half Q normal reads before softmax/dS, then used
  `wait_lgkm(8)` before `DoutUsed` and `wait_lgkm(0)` before `QUsed`.
- Evidence:
  correctness and static gates passed, but consumer branch windows rose from
  `189/240` to `222/240`.  Full perf H1/S1024 regressed from
  `47,484,710` to `47,591,635` simTicks while MMAC active only nudged from
  `32.9468%` to `33.0627%`.
- Decision:
  `OBSERVE_ACTIVE_REJECT_TICKS`; local source is restored to the accepted
  `dkv_splitwait_highsrc` commit.  Do not reintroduce this live-range extension
  unless a later design also reduces Q/Dout ownership waits.
- Operational note:
  remote transfer to `10.59.41.48` became unstable while archiving/syncing.
  Before the next PMD run, confirm the container source matches local clean
  `d5878ae` plus no diff.

Next dKV structural candidate:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_12h_goal_plan_20260709.xlsx`,
  sheet `08_Raw2_Sidecar_Plan`.
- Thesis:
  the current `Mq128/RawBuffers=1` route is blocked by Q/Dout half-page
  `Used` waits.  `RawBuffers=2` would let producers prefetch `q_tile+1`, but
  raw Q+dO double buffering consumes the full 128KB LDS budget, so sidecar
  must leave LDS or be otherwise compressed/overlaid.
- First step:
  do a focused `Raw1 + global sidecar` probe before touching the performance
  route.  If direct/global sidecar is incorrect or clearly slower, reject
  `Raw2 + global sidecar` early.
- Boundary:
  do not add finer-grained PageUsed barriers or hold more operand fragments
  across softmax; both have already shown poor tradeoffs.

Rejected dKV causal invalid q-tile skip candidate:

- Candidate:
  skip q-loop iterations that are wholly invalid under causal masking for the
  current K/V tile.
- Evidence:
  runtime-causal skip failed metadata with SGPR spill.  Canonical causal-only
  skip passed static/resource and H1/S128 plus H1/S1024 correctness, but full
  perf regressed: `simTicks 47,484,710 -> 49,150,010`, `MMAC active
  32.9468% -> 28.7232%`, while `MMOP 131,072 -> 88,064`.
- Decision:
  `REJECT_PERF_REGRESSION_SOURCE_REVERTED`; local and remote sources are
  restored to accepted `dkv_splitwait_highsrc`.
- Lesson:
  in the current WASP schedule, eliminating invalid triangular work can reduce
  useful MMAC density and worsen wave progress if it does not also reduce the
  dominant Q/Dout ABarrier ownership bubble.  Do not retry isolated causal
  skip logic as the next dKV step.

Rejected dKV dO wait-under-softmax candidate:

- Candidate:
  move the ReleasePage dO-normal wait after softmax/dS so the independent VALU
  work can cover `ds_read_matrix` latency.
- Evidence:
  static/resource and H1/S128 plus H1/S1024 correctness passed, but H1/S1024
  full perf regressed: `simTicks 47,484,710 -> 48,067,565`.  xcu showed local
  wait improved (`s_waitcnt 19.55% -> 18.63%`) while the dominant ownership
  bubble worsened (`s_abarrier_try_wait -> s_xor_b32 41.38% -> 41.73%`).
- Decision:
  `REJECT_PERF_REGRESSION_SOURCE_REVERTED`; local and remote sources are
  restored to accepted `dkv_splitwait_highsrc`.
- Lesson:
  wait hiding is not automatically good when the wait also delays a page
  ownership release.  For this route, `QUsed`/`DoutUsed` arrival timing is on
  the critical path; future candidates must reduce exposed wait without making
  producer ownership waits worse.

Current next dKV plan:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_12h_goal_plan_20260709.xlsx`,
  sheet `16_FWD_BWD_Gap_Next`.
- FWD target:
  H4/S2048/SQC7 has `mmop_runtime_share=58.1159%`, stat-derived
  `MMAC active=45.0205%`, zero LDS bank conflict, and healthy TCC hit/reuse.
- Current dKV blocker:
  accepted H1/S1024 dKV has `MMAC active=32.9468%`, but xcu is dominated by
  `s_abarrier_try_wait -> s_xor_b32 41.38%`; local wait/MMOP reductions have
  not reduced this ownership bubble.
- Next code candidate when SSH returns:
  `dkv_q_used_release_before_softmax`, a narrow probe that releases `QUsed`
  before softmax/dS after reading Q-normal sources.  It must be rejected unless
  same-shape ticks fall and xcu shows a real ownership-bubble reduction.
- Remote status:
  `10.59.41.48` is currently unreachable through both 54 and 59 jump routes,
  so new PMD/xcu runs are paused rather than guessed.

Liuchang fallback status:

- On 2026-07-09 we returned to liuchang/zys1 while the new PMD/compiler route
  is paused.  Active remote source is `/zys/shaobo/fa3_bwd_wasp_clean`.
- dKV baseline rebuilt cleanly on liuchang: branch windows `14/16`,
  `222/240`, `222/240`, `8/16`; `s_trap=0`, `s_set_vgpr_size=4`,
  `v_mov_b32=539`, `v_mov_b64=139`, no spill/scratch.
- The score-zero-hoist probe reduced `v_mov_b64` to `111` but increased
  consumer windows to `226/240` and regressed xcu dispatch duration
  `94,728 -> 94,988`; it is recorded as
  `REJECT_PERF_REGRESSION_SOURCE_REVERTED` and removed from active source.
- The consumer half-order stagger probe made consumer1 process half1 -> half0
  while consumer0 kept half0 -> half1.  It passed static/resource and
  correctness gates but regressed H1/S1024 stats.  Treat this as proof that
  useful stagger must preserve early Q0/Dout0 release; otherwise it breaks the
  existing half-page conveyor.
- The global half1-first conveyor probe changed producers and both consumers
  to half1 -> half0.  It passed static/resource and correctness gates and
  stats-only looked like a tiny win, but full perf regressed:
  `simTicks 46,716,670 -> 46,947,355`, `kernel_ticks 43,103,060 -> 43,333,745`.
  It is removed from active source and recorded as
  `REJECT_FULLPERF_REGRESSION_SOURCE_REVERTED`.
- The Q-side sidecar prefetch-before-Used probe loaded sidecar triples before
  `wait_q_half_used` and stored them to LDS afterward.  It passed static and
  correctness gates and lowered the aggregate barrier counter, but H1/S1024
  stats regressed versus accepted `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 46,773,090`, `kernel_ticks 43,103,060 -> 43,159,480`.
  It is removed from active source and recorded as
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_REVERTED`.
- The merged-used-token probe replaced separate `QUsed`/`DoutUsed` arrivals
  with one `RawHalfUsed` arrival per half.  It passed static/resource and
  correctness gates, but H1/S1024 regressed:
  `simTicks 46,716,670 -> 47,066,110`,
  `kernel_ticks 43,103,060 -> 43,452,500`,
  `MMAC active 33.2391% -> 33.1006%`.  It is removed from active source and
  recorded as `REJECT_STATS_TICKS_REGRESSION_SOURCE_REVERTED`.
- The dP-before-Q first-pair probe split the first 32-row pair of each half
  into `dP MMAC` before `QFilled` and `score MMAC` after `QFilled`.  It passed
  static/resource and correctness gates, but H1/S1024 regressed:
  `simTicks 46,716,670 -> 48,090,770`,
  `kernel_ticks 43,103,060 -> 44,477,160`,
  `MMAC active 33.2391% -> 32.5023%`.  It is removed from active source and
  recorded as `REJECT_STATS_TICKS_REGRESSION_SOURCE_REVERTED`.
- `xcu` CLI is now available on liuchang via sidecar path:
  `XCU_ROOT=/zys/tools/xcompute_light_4.6.3/opt/XCompute-Light-4.6.3/XCompute`.
  Use `PATH=$XCU_ROOT/bin:$PATH` and
  `LD_LIBRARY_PATH=$XCU_ROOT/lib:$XCU_ROOT/lib/lib:$LD_LIBRARY_PATH` before
  `xcu status -P <perf> ...`.  The half1-first rejected perf has xcu outputs
  under `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dkv_global_half1_first_20260709_164425`.
- dKV perf path is valid on liuchang.  dQ stats-only correctness passes, but
  dQ with Perf/helper currently produces NaNs, so dQ perf evidence must be
  treated invalid until isolated.

BPS/vbcnt focused probe, 2026-07-11:

- Added opt-in macro `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE`, default off.
- Purpose:
  test whether explicit `s_waitcnt_vbcnt 0` before BPS-published Filled-token
  arrivals helps PMD readiness. This is not a new performance route and does
  not change canonical behavior unless built with
  `EXTRA_CXXFLAGS="-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=1"`.
- Current evidence:
  default and vbcnt variants both pass build/static gates and H1/S128 +
  H1/S1024 correctness. On H1/S1024, stats-only changed
  `kernel_ticks 43,523,025 -> 42,995,680` and
  `simTicks 47,136,635 -> 46,609,290`, with no spill/scratch and
  `ldsBankConflict=0`.
- Status:
  `OBSERVE_MICRO_WIN_NEEDS_XCU`. Before promotion, run same-env full perf and
  xcu detail/wavefronts/bubbles/pipeline for default versus vbcnt.

BPS/vbcnt promoted to default, 2026-07-11:

- `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE` now defaults to `1`.
- Disable only for A/B with:
  `EXTRA_CXXFLAGS="-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0"`.
- Default-enabled H1/S1024 correctness PASS under:
  `/zys/shaobo_runs/dkv_vbcnt_default_20260711/dkv_mmac_correctness_20260711_112221`.
- Stats:
  `simTicks=46,554,690`, `kernel_ticks=42,941,080`,
  `MMOP=131,072`, `VALU=168,514`, `SCA=114,520`, `LDS=79,360`,
  coissue `37,689/26,615`, `ldsBankConflict=0`.
- Treat this as a useful local BPS readiness fix, not as the top-level BWD
  solution. The next design must still target dKV fragmentation/ownership
  epochs, using Tri Dao FA3 BWD and Shaobo FWD as the structural references.

dQ dS->dQ ring2, 2026-07-11:

- The design remains on branch `exp/dq-ds-to-dq-ring2`, not the canonical
  source: P_K / C_dS / C_dQ / P_V; M128/N128 pages change from `K+V` to
  `K+dS`, and only C_dQ owns dQ output.
- A focused HCU builtin gate is unresolved. The first roundtrip probe and a
  second real `MMAC output -> fp16 ds_vec -> ds_write -> reader -> dQ MMAC`
  probe cover the HCU-exposed writer/read candidates that compile on the
  current toolchain. All candidate paths differ from direct dS@K despite no
  spill or LDS bank conflict.
- No canonical workaround is allowed. The current question is the supported
  ds_write producer/reader fragment ABI or PMD's `ds_write_matrix : testing`
  semantics; this must be resolved before integrating the otherwise viable
  two-consumer ring.
- Follow-up after rereading ISA/HCU docs:
  `DS_WRITE_MATRIX_FORMAT` and `DS_READ_MATRIX_FORMAT` do have documented B16
  page-format pairings.  The unresolved piece is not "hardware has no pair";
  it is the producer fragment ABI: how to turn a C_dS MMAC/VALU result into
  the exact 4-VGPR source expected by `ds_write_matrix_format_f16`.  Corrected
  M-pair probes with 2KB pages, LIT=0/1, and four simple fp16 pack orders still
  mismatch direct `dS@K`, with zero LDS bank conflict.
- Important prior proof:
  the Shaobo MLS layout reference records a passing 2026-07-05 operand probe:
  `VGPR(dS) -> ds_write_matrix_format(no t) -> ds_read_matrix_trans_format
  32x16 -> MMAC` with normal `32x16` K readers.  So the next task is not to
  abandon the native handoff; it is to make C_dS produce that accepted
  dQ-friendly layout directly.
- The follow-up q-owned chain probe rejects the easiest direct pack:
  `Q_trans x K32` q-owned score -> four simple fp16 pack orders ->
  `ds_write_matrix(no t)` -> dQ MMAC all fail (`any_pass=0`) while keeping
  `ldsBankConflict=0` and no scalar/permute workaround.  Next work must use the
  7/5 slot-map formula to generate the accepted source layout deliberately.
- Direct qK MMAC was tested in the same probe:
  variants 4-7 use direct `__builtin_hcu_mmac_f32_16x16x16_f16` for qK score
  instead of lit/4interleave.  The asm contains direct non-`lit` MMAC, but PMD
  still reports `any_pass=0`.  So this is not solved by changing qK MMAC form
  alone.

Slot-map reverse result, 2026-07-11:

- Focused probe:
  `probes/dq_dswrite_slotmap_reverse_probe.cpp`.
- Key inferred table for
  `ds_write_matrix(no t) -> ds_read_matrix_trans 32x16 -> K_normal MMAC`:
  `slot_k[group][word] = group * 4 + (word & 3)`.
- Meaning:
  C_dS must deliberately publish the dS source fragment in this slot layout.
  `word0..3` and `word4..7` are two half-regions with the same K-row labels,
  not arbitrary duplicate pack slots.  Do not "fix" this with scalar LDS
  gather, bpermute/mpermute, or another pack permutation hunt.
- Evidence:
  `/zys/shaobo_runs/dq_slotmap_reverse_probe_20260711_160921` and
  `/zys/shaobo_runs/dq_slotmap_reverse_probe_20260711_161646`;
  no spill/private segment, `ldsBankConflict=0`, and matrix path only.
- Follow-up:
  `/zys/shaobo_runs/dq_slotmap_reverse_split_probe_20260711_165032`
  proves the real lesson.  `pair_acc` fails, while `split_low`,
  `split_high`, and `split_combined` pass.  So C_dQ must keep the two
  half-regions as separate accumulator/update paths; do not use the old
  pair-accumulate helper for a dS handoff unless the math truly wants a K-step
  reduction.
- Compact source-slot map:
  `/zys/shaobo_runs/dq_slotmap_reverse_compact_probe_20260711_172345`.
  The identity write/read map has `mapped=504/512` and `unique_src=504/512`;
  the only unmapped destination slots are `(group=2,q=15,word=4..7)` and
  `(group=3,q=15,word=4..7)`.  This means the next C_dS publisher must use the
  compact dst->src slot table, not an assumed full 512-slot affine formula.
  The split-accumulator proof still passes in the same run.

dQ role topology note, 2026-07-11:

- A 12-wave single-producer dQ experiment removed producer1 and let waves0-3
  publish both Q/dO groups plus K/V.  It passed correctness/resource gates and
  reduced the top ABarrier bubble locally, but fullperf regressed:
  `simTicks 35,881,300 -> 36,049,650`, MMAC active stayed flat
  `27.4198% -> 27.4182%`, and XCU showed dispatch waves/avg active waves fell
  `128/79.17 -> 96/59.35`.
- Keep canonical dQ as 16 waves.  Producer-thinness should be solved by useful
  recurring producer work or shorter ownership lifetime, not by deleting the
  fourth 4-wave role.

dQ read scheduling note, 2026-07-12:

- Moving K-normal `ds_read_matrix` before softmax/dS reduced stats-only
  `waitLgkm`, but fullperf regressed and XCU still showed PageUsed ABarrier as
  the top bubble.  Do not promote read-placement-only changes unless they also
  reduce elapsed ticks and the ownership bubble in xcu.

dQ half-page PageUsed note, 2026-07-12:

- Splitting `PageUsed` into `HalfUsed + Used` passed correctness/resource
  gates but regressed H1/S1024 (`simTicks=36,033,725`, `MMAC active=27.3829%`)
  and raised barrier counters versus mainline fullperf.  The active source is
  restored to the canonical single-`PageUsed` ledger.  Do not add finer
  ownership tokens unless the design also removes another token or gives the
  producer useful recurring work.

dQ group1 reverse n_tile note, 2026-07-12:

- Reversing `n_tile` order for consumer group1 was a clean no-extra-token
  attempt at useful-work stagger, but it regressed H1/S1024
  (`simTicks=36,171,590`, `MMAC active=27.2470%`).  The active source is
  restored.  Pure chunk-order skew is not enough; future stagger must move
  different useful work, not just different addresses.

dQ native dS ring structural probe, 2026-07-12:

- Standalone probe:
  `probes/dq_native_ds_ring_structural_probe.cpp`.
- Evidence:
  `/zys/shaobo_runs/dq_native_ds_ring_structural_fix_20260712_024559`;
  workbook sheet
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `63_DQ_NativeRing_StructuralPrototype`.
- Result:
  `ACCEPT_PROBE_STRUCTURAL`.  Producer wave publishes K; publisher waves write
  deterministic dS slots with `ds_write_matrix_32x16_f16`; consumer waves read
  dS with `ds_read_matrix_trans` and K with normal `ds_read_matrix`, then run
  split MMAC.  All four slot low/high checks pass, no spill/scratch,
  `ldsBankConflict=0`, and no gather/permute/bpermute matrix workaround.
- Boundary:
  this proves the role-to-role native LDS handoff only.  It does not yet prove
  the full canonical softmax/dS arithmetic can be generated in source-slot
  order.  Canonical dQ remains unchanged.
- WDRA hygiene lesson:
  keep lane/threadIdx-derived VGPR setup inside each role branch after
  `s_set_vgpr_size`; a first version with pre-role `lane` and an ordinary LDS
  clear loop triggered PMD uninitialized-VGPR/LDS-index failure.

dQ dS@K read-batch wait collapse, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `64_DQ_DQGemm_Batch8_Wait0`.
- Result:
  `REJECT_PMD_REGISTER_INIT`.  Changing only `dq_update_from_ds_pair` from
  `read8 -> wait_lgkm(4) -> half MMAC -> wait_lgkm(0) -> half MMAC` to
  `read8 -> wait_lgkm(0) -> full MMAC island` passed static/resource gates
  but PMD aborted on H1/S128 with `vgpr81 is not init or has been freed` during
  MMOP.
- Lesson:
  keep the canonical split wait in dQ.  It appears to be required for current
  PMD/WDRA VGPR readiness tracking of the K-normal fragments, not merely a
  conservative scheduler wait.

dQ native dS ring formula probe, 2026-07-12:

- Standalone probe:
  `probes/dq_native_ds_ring_formula_probe.cpp`.
- Evidence:
  `/zys/shaobo_runs/dq_native_ds_ring_formula_20260712_030944`;
  workbook sheet
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `65_DQ_NativeRing_FormulaProbe`.
- Result:
  `ACCEPT_PROBE_FORMULA_SOURCE_SLOT`.  A C_dS publisher can compute synthetic
  softmax/dS formula values directly in `NativeDsSlotMap` source-slot order,
  write them with `ds_write_matrix_32x16_f16`, and have C_dQ consume them with
  `ds_read_matrix_trans` plus K-normal MMAC.  No gather/permute/bpermute or
  ordinary matrix `ds_read_b*`; `ldsBankConflict=0`; no spill/scratch.
- Boundary:
  qk/dP are still synthetic scalar formula inputs.  The next real blocker is
  MMAC output orientation: can score/dP MMAC produce the source-slot fragment
  natively, or does it require an expensive cross-lane/source-slot transform?

dQ tail accumulator keep-alive prune, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `66_DQ_Tail_KeepAlive_Prune`.
- Result:
  `REJECT_PMD_REGISTER_INIT`.  Removing only the post-store
  `keep_accumulator_live(dq_reg[d_idx])` loop passed static/source gates, but
  changed WDRA branch codegen materially: producer1 branch reported `38/40`
  VGPRs instead of the restored canonical `9/40`.  H1/S128 PMD aborted before
  correctness with `read vgpr70 before writing` and `VGPR index 85 is out of
  range: VGPR range=[0,40]` on `v_mov_b32`.
- Evidence:
  `/zys/shaobo_runs/dq_tail_keepalive_prune_20260712_031836/dq_correctness_20260712_031837`.
- Lesson:
  the keep-alive loop is a current WDRA/codegen liveness guard, not removable
  tail noise.  Canonical source is restored; do not delete this guard without
  a focused WDRA-exit proof.

dQ PageUsed early release, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `67_DQ_PageUsed_EarlyRelease`.
- Result:
  `OBSERVE_REJECT_SOURCE_RESTORED`.  Moving `dq_arrive_page_used` from after
  the `n_tile` loop to just after the last K-normal `wait_lgkm(0)` passed
  correctness/resource gates, but the evidence did not support promotion.
  Fullperf H1/S1024 had a tiny tick drop
  `36,094,240 -> 36,046,920`, while MMAC active fell
  `27.3254% -> 27.2589%` and barrier counter rose
  `50,779.75 -> 52,556.25`.
- XCU:
  top `s_abarrier_try_wait -> s_xor_b32` bubble worsened
  `1,140,988 -> 1,188,124` cycles.  The source is restored to canonical
  PageUsed placement.
- Lesson:
  PageUsed micro-placement is not enough.  The next route must reduce the
  ownership dependency or add real useful work during the PageUsed window; do
  not keep splitting/moving PageUsed as a standalone optimization.

dQ causal predicate minimalization, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `68_DQ_CausalMask_Minimal`.
- Result:
  `ACCEPT_PERF`.  In canonical dQ, `S%Mq==0` and `S%Nk==0`, and
  `active_k_tiles` guarantees every visited `krow` is in bounds.  The dS hot
  loop now checks only `krow <= qrow` instead of rechecking
  `krow < seqlen && qrow < seqlen` for every element.
- Evidence:
  H1/S128 and H1/S1024 correctness PASS; metadata `sgpr=65`, `vgpr=128`,
  `private=0`, no spill/scratch.  H1/S1024 fullperf improves
  `simTicks=36,094,240 -> 34,414,380`, MMAC active
  `27.3254% -> 29.2992%`, SCA `77,516 -> 58,940`, and VALU
  `121,632 -> 112,064`.  XCU dispatch duration improves
  `71,320 -> 67,628`.
- Boundary:
  this relies on the canonical fixed-shape/divisible-S contract.  Do not carry
  it to varlen or non-divisible shapes without reintroducing bounds proof.
- Next:
  PageUsed/ABarrier ownership remains the top xcu bubble.  The next path must
  reduce ownership dependence or give producer waves useful medium-weight work;
  pure PageUsed arrive-point motion was already rejected.

dQ tail second sync prune, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `69_DQ_Tail_SecondSync`.
- Result:
  `ACCEPT_SMALL_PERF`.  The active dQ source keeps the first terminal
  `__syncthreads()` and all `s_abarrier_inv` calls, but moves the second
  terminal sync inside `if (diag_store != 0)`.
- Evidence:
  H1/S128 and H1/S1024 correctness PASS; metadata `sgpr=65`, `vgpr=128`,
  `private=0`, no spill/scratch.  H1/S1024 fullperf improves
  `simTicks=34,414,380 -> 33,977,580`, MMAC active
  `29.2992% -> 29.4292%`, barrier counter
  `48,247.75 -> 46,545.75`, and waitLgkm `14,390.25 -> 14,068`.
- XCU:
  top `s_abarrier_try_wait -> s_xor_b32` bubble improves
  `1,115,944 -> 1,082,188` cycles.  `s_barrier -> s_cbranch_vccnz` remains
  large at `704,020` cycles, so this did not solve the terminal/ownership
  bottleneck.
- Next:
  continue toward 40% MMAC active by reducing PageUsed/ABarrier ownership
  dependence or adding useful producer work.  Do not spend more turns moving
  the same PageUsed arrive point unless the design removes a dependency/token.

dQ BPS vbcnt A/B, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `70_DQ_BPS_VBCNT_AB`.
- Result:
  `REJECT_CORRECTNESS`.  Built an isolated
  `build/fa3_bwd_dq_no_vbcnt` with
  `EXTRA_CXXFLAGS=-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0`; canonical source was
  not changed.
- Evidence:
  no-vbcnt asm has `s_waitcnt_vbcnt=0`, branch windows remain
  `8/40,161/216,161/216,9/40`, and static dQ gate PASS.  H1/S128 PMD
  completed but correctness failed with `pass=0`, `actual_nonfinite=8192`,
  `bad=8192`, first output `nan`.
- Lesson:
  in the current dQ BPS+ABarrier path, `s_waitcnt_vbcnt 0` is a correctness
  readiness boundary, not removable scheduler noise.  Future work must overlap
  or redesign this cost rather than deleting it.

dQ K-normal prefetch, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `71_DQ_KNormal_Prefetch`.
- Result:
  `REJECT_STATS_TICKS_REGRESSION`.  Temporarily prefetched K-normal fragments
  before score/dP so the dQ helper consumed prefetched `k_norm0/k_norm1`.
  Source was restored after the run.
- Evidence:
  resource and correctness were clean: branch windows
  `8/40,187/216,187/216,9/40`, metadata `private=0`, `sgpr=65`, `vgpr=128`,
  no spill/scratch, H1/S128 and H1/S1024 PASS, `ldsBankConflict=0`.
  But H1/S1024 stats regressed:
  `simTicks=33,529,405 -> 34,502,195`, MMAC active
  `29.5058% -> 28.5053%`, barrier counter
  `44,590.25 -> 49,150.25`.
- Lesson:
  local wait reduction is not enough.  This did reduce `waitLgkm`
  `14,146.75 -> 11,683.75`, but the +26 VGPR branch footprint and longer
  operand lifetime made the conveyor worse.

dQ final PageUsed tail wait, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `72_DQ_Tail_FinalUsed_Wait`.
- Result:
  `REJECT_PMD_REGISTER_INIT`.  Temporarily replaced the remaining terminal
  `__syncthreads()` before `s_abarrier_inv` with wave0 waits on the final
  `Page0Used/Page1Used` tokens.  Source was restored after the run.
- Evidence:
  build/static gates passed with unchanged branch windows
  `8/40,161/216,161/216,9/40`, but H1/S128 PMD aborted before correctness:
  `vgpr81 is not init or has been freed` during MMOP execution.
- Lesson:
  the first terminal sync before ABarrier invalidation is part of the current
  WDRA/PMD role-exit discipline.  Do not keep deleting tail barriers; move
  effort to mainloop ownership structure, such as group-level PageUsed or a
  native dS publisher/ring design.

dQ group-level PageUsed, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `73_DQ_GroupPageUsed`.
- Result:
  `REJECT_STATS_TICKS_REGRESSION`.  Temporarily changed PageUsed ownership
  from 8 per-wave consumer arrivals to 2 group-level arrivals using one
  EBarrier sync per consumer group plus a representative ABarrier arrive.
  Source was restored after the run.
- Evidence:
  static/source gates passed, branch windows stayed
  `8/40,161/216,161/216,9/40`, metadata `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS,
  `ldsBankConflict=0`.
  H1/S1024 regressed:
  `simTicks=33,529,405 -> 35,625,590`, MMAC active
  `29.5058% -> 28.0489%`.
- Lesson:
  PageUsed/ABarrier remains a bottleneck class, but reducing ABarrier arrival
  count through an extra EBarrier is not profitable.  Stop this direction;
  next work should change the page lifetime or add useful producer-side work,
  not add another synchronization layer.

dQ source-slot coordinate probe, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `83_DQ_SourceSlotCoord`.
- Result:
  `OBSERVE_LAYOUT_FACT_REJECT_DIRECT_SOURCE_SLOT`.  Added standalone
  `probes/dq_source_slot_coordinate_probe.cpp`; canonical dQ source was not
  changed.
- Evidence:
  PMD run `/zys/shaobo_runs/dq_source_slot_coord_probe_20260712_081829`
  shows canonical score MMAC natural coordinates are correct
  (`identity_errors=0`), but the same lane/word order is not the
  `ds_write_matrix` source-slot order (`source_slot_errors=502/504`), and
  direct readback fails (`read_identity_errors=510/512`).  Stats show no LDS
  bank conflict in the probe (`ldsBankConflict=0`).
- Lesson:
  the native dS ring blocker is layout ownership, not score arithmetic.  Do not
  directly write canonical MMAC output to LDS with `ds_write_matrix`; first
  prove a native MMAC/reader orientation that emits `NativeDsSlotMap` source
  order, or choose a different top-level dQ design.

dQ source-slot orientation probe, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `84_DQ_SourceSlotOrient`.
- Result:
  `REJECT_PROBE_CANONICAL_UNCHANGED`.  Extended only
  `probes/dq_source_slot_coordinate_probe.cpp`; canonical dQ source was not
  changed.
- Evidence:
  PMD run `/zys/shaobo_runs/dq_source_slot_orient_probe_20260712_083046`
  tested four native reader combinations.  Only canonical
  `q_trans_k_trans` preserves natural coordinates (`identity_errors=0`), but
  it still fails source-slot order (`source_slot_errors=502/504`).  The other
  three modes fail both identity and source-slot.  Final:
  `any_source_slot_pass=0`, `any_direct_read_pass=0`, `ldsBankConflict=0`.
- Lesson:
  simple normal/trans reader swaps are not the missing native dS-ring link.
  Further native dS work needs either a new instruction form or a measured
  source-slot rearrangement; otherwise optimization should return to canonical
  dQ ABarrier/page-ownership structure.

dQ boundary n_tile classify, 2026-07-12:

- Result:
  `ACCEPT_TICKS_ACTIVE_OBSERVE`.  Canonical dQ now skips fully invalid
  boundary n_tiles and uses no-mask dS for fully valid boundary n_tiles.
- Evidence:
  H1/S128 and H1/S1024 correctness PASS, no spill/scratch,
  `ldsBankConflict=0`.  Repeat H1/S1024 stats:
  `simTicks=29,706,495`, `MMAC active=32.0864%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`.
- Lesson:
  removing invalid causal MMAC work improves ticks but can lower aggregate
  MMAC active because the removed work was counted as MMOP.  Treat this as
  algorithm cleanup, not proof that the WASP pipeline is fixed.  Next focus is
  still ABarrier ownership/useful overlap or native dS source-slot design.

dQ q-tile split evidence, 2026-07-12:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `93_DQ_QTileSplit`.
- Result:
  `OBSERVE_QTILE_SPLIT_CAUSAL_FRONTLOAD`.  Running H1/S1024 causal with one
  q-tile per dispatch shows MMAC active rises from `11.045%` at tile0 to
  `40.815%` at tile7; tiles5-7 are already above `40%`.
- Lesson:
  current dQ's late causal steady region is not fundamentally stuck at 32%.
  The average is pulled down by early tiles where valid causal work is small
  but ABarrier/control/setup cost stays fixed.  Next work should address
  early-tile specialization or useful work per ownership epoch before more
  main-path matrix-read/MMAC micro-tweaks.

dQ conditional page barrier lifetime, 2026-07-12:

- Result:
  `REJECT_STATS_NO_BEST_WIN_SOURCE_RESTORED`.  Conditionalizing PageUsed and
  Page1Filled init/arrive/inv by `active_k_tiles` was correct and passed all
  gates, but H1/S1024 only reached `simTicks=30,037,735`,
  `MMAC active=32.1251%`, not better than the accepted repeat best
  `29,706,495`.
- Lesson:
  early causal ABarrier overhead is real, but tiny branch-level lifetime
  pruning is not enough.  Future early-tile work needs a bigger top-level
  change, such as changing useful work per CTA or a separate proven early-tile
  schedule, not another small PageUsed/init tweak.

dQ consumer1 reverse M16 mapping, 2026-07-12:

- Result:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.  Pairing same-SIMD consumer
  waves as `(0,7)/(1,6)/(2,5)/(3,4)` instead of canonical
  `(0,4)/(1,5)/(2,6)/(3,7)` passed correctness and gates, but H1/S1024 was
  `simTicks=30,142,840`, `MMAC active=32.2965%`, worse than the accepted
  repeat best.
- Lesson:
  causal row-work balance is a plausible concern, but this simple ownership
  remap does not move the critical path.  Continue focusing on larger
  ownership/softmax/control restructuring or native dS dependency design.

dQ tail no-invalidate, 2026-07-12:

- Result:
  `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`.  Removing normal-path terminal
  `__syncthreads()+s_abarrier_inv` passed static metadata and reduced SGPR
  `65 -> 63`, but H1/S128 PMD aborted with
  `vgpr81 is not init or has been freed` during MMOP.
- Lesson:
  the xcu tail bubble is real, but the current PMD/WDRA path still needs the
  terminal role-exit discipline.  Do not remove tail cleanup in canonical dQ
  unless a focused ABI/PMD proof makes it safe.

dQ dS-cache VUsed, 2026-07-12:

- Result:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.  This was the corrected
  version of VUsed early release: compute all page-local `score+dP+dS`, cache
  four dS fragment pairs in VGPR, release V with `VUsed`, then reread K and
  compute `dQ`.
- Evidence:
  gates and H1/S128/H1/S1024 correctness passed, no spill/scratch,
  `ldsBankConflict=0`.  H1/S1024 regressed to `simTicks=30,905,875`,
  `MMAC active=31.1624%`, `VALU=63,968`, `SCA=63,672`,
  `coissue=5,802/11,721`.
- Lesson:
  this proves the legal V-early lifetime is too expensive in the current
  canonical dQ loop.  Extra tokens, two-phase n-tile traversal, and dS cache
  live range dominate the overlap benefit.  Do not retry VUsed as a small
  token split; any next dQ improvement needs a larger ownership/dependency
  redesign or early-causal-tile specialization.

dQ QDoFilled group split, 2026-07-12:

- Result:
  `REJECT_STATS_NO_BEST_WIN_SOURCE_RESTORED`.  Splitting startup
  `QDoFilled` into group-local 4-wave tokens passed correctness/resources, but
  did not beat the accepted dQ best.
- Evidence:
  H1/S1024 first run `simTicks=29,853,915`, MMAC active `32.3773%`;
  repeat `simTicks=29,870,295`, MMAC active `32.0531%`;
  `ldsBankConflict=0`, no spill/scratch.  Accepted repeat best remains
  `29,706,495`.
- Lesson:
  the startup bubble is not just "consumer waiting for the other Q/dO group".
  Page0 K/V still cannot overwrite the shared sidecar LDS until all consumers
  have latched sidecar/QDo.  Do not continue QDoFilled token splitting as a
  standalone optimization.

dQ Nk256 single-page result, 2026-07-11:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `51_Nk256_SinglePage`.
- Result:
  `REJECT_STATS_TICKS_REGRESSION`.  H1/S256 and H1/S1024 correctness passed
  after fixing SGPR spill with `n_tile` `unroll 4`, but H1/S1024 regressed:
  `simTicks=41,586,545` vs mainline `35,704,760`, and MMAC active fell
  `27.3852% -> 24.3812%`.
- Lesson:
  reducing PageUsed epochs by making one 128KB K/V page is not enough.
  It removes K/V double buffering/prefetch and adds causal padding work; do
  not retry `Nk256` single-page unless the output/causal ownership design also
  changes.

dQ setprio MMAC islands, 2026-07-12:

- Result:
  `ACCEPT_MICRO_CANONICAL`.  FWD-style `s_setprio` is now wrapped around dQ
  score/dP and `dS @ K` MMAC islands.  Tile shape, math, LDS layout, ABarrier
  tokens, Q/dO latch, K/V page ownership, and store ownership are unchanged.
- Evidence:
  static/resource gates pass with `private=0`, `sgpr=65`, `vgpr=128`, no
  spill/scratch, and branch windows `8/40,159/216,159/216,9/40`.  H1/S128 and
  H1/S1024 causal correctness pass under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD:
  H1/S1024 improves from accepted best `29,706,495` ticks / `32.0864%`
  MMAC active to first run `29,145,480` ticks / `32.7016%`, repeat
  `29,438,955` ticks / `32.5598%`.  Instruction counts stay the same:
  `MMOP=50,688`, `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`;
  `ldsBankConflict=0`.  Coissue success improves from `6,280` to
  `10,706`/`11,366`.
- Lesson:
  FWD-style priority is a real micro-win for canonical dQ and should remain in
  the baseline.  It improves scheduler/coissue behavior but does not solve the
  larger ownership/wait/control bottleneck or reach the 40% MMAC active target.
- Fullperf/xcu:
  archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_143128_dq_setprio_h1s1024_sqc7_fullperf`.
  Fullperf gives `29,793,855` ticks / `32.2046%` MMAC active.  xcu shows the
  remaining top rows are control/ownership-heavy: `s_xor_b32 27.13%`,
  `s_cbranch_vccnz 17.20%`, `s_waitcnt_vbcnt 9.00%`, while `mmop_fp16` is
  `12.39%`.  Producer representative bubble is `98.78%`, consumer MMOP
  representative bubble is `61.42%`.

dKV/dQ ownership follow-up, 2026-07-12:

- dKV raw2 at current `Mq=128` was not implemented: Q+dO raw double buffering
  consumes the full 128KB LDS budget before sidecar, so it needs a separate
  sidecar lifetime/overlay design rather than a tile alias change.
- dQ short-causal Page1 init/invalidate pruning was tested and restored.
  H1/S128/H1/S1024 correctness and resource gates passed, but H1/S1024 was
  unstable (`29.242M` first, `29.175M` repeat) and did not reduce SCA/wait or
  explain the pipeline.  Keep canonical clean.
- Next useful directions:
  dQ producer global-load lookahead during PageUsed waits, or a native
  dS/ownership redesign with a focused resource proof.  Avoid more small
  Page1-token pruning and avoid raw2/Mq128 until sidecar fits under 128KB.
