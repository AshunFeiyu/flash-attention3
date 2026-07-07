# Source Status

## 2026-07-07 dQ 40% MMAC Active Target

Status: `DQ_MQ32_SIDECAR_LDS_BASELINE_ACTIVE`.

Current goal:

- Optimize the clean dQ path toward `MMAC active >= 40%`.
- Target shape is `B=1,H=1,S=1024,D=128,causal=true`,
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.
- Hard gates remain correctness PASS, `private=0`, no scratch/spill,
  `ldsBankConflict=0`, and main matrix path through MLS/BPS +
  `ds_read_matrix` + MMAC.

Current code state:

- Branch: `shaobo/7gemm-dq-bringup`.
- Canonical dQ tile: `Mq=32,Nk=64,D=128,12 waves`.
- Sidecar is staged by the producer into LDS; worker does not direct-load the
  three sidecar streams from global in the hot path.
- `PageUsed` is consumer-only: producer waits for 4 consumer arrivals before
  overwriting a page.  Worker-side `PageUsed` arrival was removed because
  workers finish page reads before `DsFilled`.
- Added a standalone measurement knob:
  `--tiles-per-dispatch` / `DQ_TILES_PER_DISPATCH`.  It controls how many q
  tiles the standalone harness packs into one dispatch and does not alter the
  kernel math or tile shape.

Latest target-shape evidence:

- Default two-dispatch H1/S1024:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063111`,
  aggregate `simTicks=52,082,485`, `MMOP=52,224`,
  `MMAC active=8.8385%`, `ldsBankConflict=0`, correctness PASS.
- One-dispatch measurement with `DQ_TILES_PER_DISPATCH=32`:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_063334`,
  `simTicks=34,346,130`, `MMOP=52,224`,
  `MMAC active=8.2338%`, `ldsBankConflict=0`, correctness PASS.
- XCU full perf on the same one-dispatch shape:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_072949`,
  helper perf
  `m5out/0/0/2737677_fa3_bwd_dq_clean.perf`.  The dominant bubble is
  `s_abarrier_try_wait -> s_xor_b32` on `DsFilled`; a representative window
  showed about `96%` bubble cost and no `MMAC+VALU` coissue.
- Rejected Kt-preread attempt:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_073822`,
  correctness/resource PASS but `simTicks=34,237,840` and
  `MMAC active=8.1943%` versus same one-dispatch baseline
  `34,215,090` / `8.2480%`; code removed from the active path.
- Accepted PageUsed consumer-only update:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082245`,
  correctness/resource PASS, `simTicks=33,372,430`,
  `MMAC active=8.44342%`, `SCA=212,520`, `ldsBankConflict=0`.

Decision:

- Keep `DQ_TILES_PER_DISPATCH=32` for S1024 perf capture because it avoids
  artificial dispatch splitting.
- Do not count it as an optimization; it lowers dispatch overhead but not core
  MMAC active.
- Direct Mq64 in the old q_subtile path remains rejected by hang; Mq64
  single-page direct and split variants are also rejected by perf because they
  lose overlap/coissue.
- Do not retry isolated Kt preread/code motion or Mq64 single-page designs.
  The next 40% path should stay on the legal two-page pipeline, preserve
  worker/consumer coissue, and continue removing unnecessary barrier/control
  debt before another larger-tile redesign.

## 2026-07-07 dQ dS Chunk Token Rejected

Status: `DQ_SOURCE_RESTORED_TO_SIDECAR_LDS_STAGING`.

The source was briefly modified to replace full-page `DsFilled(count=4)` with
per-worker dS chunk tokens.  Consumers waited chunk0+1 before consuming the
first 32-column dQ n-block, then chunk2+3 before consuming the second n-block.

Result:

- Static/resource PASS:
  `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- H1/S128 and H1/S1024 correctness PASS.
- H1/S1024 dispatch1 regressed:
  `kernel_ticks=31,380,440`, whole-active `MMAC=8.7319%`,
  compared with accepted `dq_sidecar_lds_staging`
  `kernel_ticks=28,114,905`, whole-active `MMAC=9.7068%`.
- `SCA=194,600` shows that finer-grain barriers added too much scalar/control
  debt.

Decision:

- Code was reverted and remote recertified to the accepted sidecar-LDS
  baseline:
  consumer branch `49/72`, worker `83/128`, metadata `private=0`,
  `sgpr=67`, `vgpr=168`, no spill.
- Do not retry finer dS chunking as an isolated optimization.  The next 40%
  MMAC-active route must either reduce token count/LDS footprint or increase
  useful MMAC work per ownership epoch.

## 2026-07-07 dQ Mq64 QDo Token Rejected

Status: `DQ_SOURCE_RESTORED_TO_MQ32_SIDECAR_LDS_STAGING`.

Tried to revive Mq64 with an explicit q-subtile lifetime proof:

- `ActiveDqTile = Mq64,Nk64`.
- Added `QDoUsed(count=4)` so worker waves release Q/dO/sidecar before
  producer overwrites that LDS region for q_subtile 1.
- Changed producer page reuse from `kt >= 2` to a cross-q_subtile
  `page_epoch >= 2` test.

Result:

- Static/resource PASS:
  `private=0`, `sgpr=69`, `vgpr=168`, no spill/scratch.
- H1/S128 PMD hung and was killed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_034822`.

Decision:

- Code was reverted and remote recertified to the accepted Mq32 sidecar-LDS
  baseline.
- Mq64 is still relevant for the 40% MMAC-active target, but the synchronization
  protocol must first be proven in a focused q_subtile ownership probe.  Do not
  retry direct Mq64 in the performance kernel.

Follow-up:

- Retried Mq64 with independent `page0_seen/page1_seen` tracking after finding
  that the first QDo attempt could still overwrite page0 in causal H1/S128.
- Static/resource remained clean, but H1/S128 hung again:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_035807`.
- Source was restored and recertified again.  The remaining issue is not just
  page reuse; it is an unresolved q_subtile ABarrier phase/lifetime protocol.
- Added focused probe `probes/dq_qsubtile_barrier_probe.cpp`.  It does not
  hang under PMD, but scalar LDS data checks fail (`errors=16`, `done=0`) even
  after adding `wait_lgkm(0)`.  Treat it as a synchronization smoke only, not as
  proof for the matrixized FA path.  A faithful follow-up must use
  `matrix_load ... bps lds` and `ds_read_matrix`.

## 2026-07-07 dQ Mq32 Double-Page Conveyor

Status: `DQ_MQ32_DOUBLEPAGE_CURRENT_BASELINE`.

Current dQ source on branch `shaobo/7gemm-dq-bringup` is a single canonical
MMAC path, not a phase stack:

- Tile: `Mq=32,Nk=64,D=128`, 12 waves.
- Producer waves0-3 load Q/dO once per q-subtile and stream K/V/Kt into two
  LDS pages.
- Worker waves8-11 wait `PageFilled`, compute score/dP/softmax/dS, write dS
  into the same page, then signal `DsFilled` and partial `PageUsed`.
- Consumer waves4-7 wait `DsFilled`, compute `dQ = dS @ K^T` with MMAC, store
  directly to global, and release `PageUsed`.
- Page protocol:
  `PageFilled(count=4)`, `DsFilled(count=4)`, `PageUsed(count=8)`,
  `AllDone(count=12)`.

Validation:

- Remote build/gate PASS:
  `TARGET_GFX=946 BUILD_ASM=1 SRC=src/dq_kernel.cpp BIN=build/fa3_bwd_dq_clean ASM=build/fa3_bwd_dq_clean.asm ./build.sh`.
- Symbol metadata PASS:
  `private=0`, `sgpr=61`, `vgpr=168`, `sgpr_spill=0`, `vgpr_spill=0`.
- WDRA branch windows:
  producer `1/40`, consumers `49/72`, worker `91/128`, tail `2/48`.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015920`;
  H1/S512 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015927`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_015941`.

Performance:

- H1/S1024 dispatch0:
  `kernel_ticks=21,420,035`, `MMAC active=5.8039%`,
  coissue `245/204`, `ldsBankConflict=0`.
- H1/S1024 dispatch1:
  `kernel_ticks=35,671,545`, `MMAC active=7.8501%`,
  coissue `751/665`, `ldsBankConflict=0`.
- Versus the serial Mq32 bringup dispatch1, ticks improved
  `36,587,005 -> 35,671,545` and MMAC active improved
  `7.6036% -> 7.8501%`.

Rejected follow-up:

- Direct `Mq=64,Nk=64` using the same two-page conveyor passed static/resource
  (`private=0`, `sgpr=75`, no spill) but hung in H1/S128 PMD and was reverted.
  Do not retry Mq64 by simply serializing two M32 q-subtiles; it needs explicit
  q-subtile token reset/lifetime design.

Conclusion:

This is a useful correctness-clean dQ conveyor baseline, but not a 40%
MMAC-active solution.  The next dQ design should increase useful MMAC island
size or rebalance worker/consumer roles while preserving no duplicate score/dP
inside one dQ tile.

## 2026-07-06 dQ Reopened

Status: `DQ_WORKBOOK_FIRST_BRINGUP`.

The preserved dKV source is not being rewritten in-place.  dQ work starts on
branch `shaobo/7gemm-dq-bringup` with a separate design contract:

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`.
- Preview directory:
  `/Volumes/172.20.68.76/共享/shaobo/dq_design_previews_20260706`.
- dQ ownership: Q tile owns dQ, loops over K/V tiles, and stores dQ once.
- First MMAC target after old Stage60 evidence review:
  `Mq=64,Nk=64,D=128,16 waves` with source-layout `K^T` ABI and two serial
  `M32` q-subtiles.  The earlier workbook `Nk=128` target is deferred until
  the same K LDS page can legally feed both normal and transpose dQ views
  without a duplicate Kt page.
- No atomic add, no phase stack, no direct consumer sidecar global reads in the
  intended hot path.

Bringup validation:

- dQ source/harness build PASS on remote:
  `SRC=src/dq_kernel.cpp BIN=build/fa3_bwd_dq_clean ASM=build/fa3_bwd_dq_clean.asm ./build.sh`.
- dQ gate PASS:
  `python3 scripts/check_dq_kernel_gate.py`.
- H1/S128 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260706_223246`.
- Numerical result:
  `dq_max_abs=5.82077e-11`, `dq_rel_l2=2.36625e-07`, `bad=0`, `pass=1`.
- Dispatch `simTicks` for the reference chain:
  softmax `2,108,136,030`, delta `1,400,708,855`, dP `12,980,695`,
  dQ output `22,555,715`.

This is accepted as `BRINGUP_ONLY`, not as a performance candidate.  The next
dQ step is to implement the canonical `Mq=64,Nk=64,D=128` MMAC kernel from the
revised workbook, using this reference path only for correctness.

## 2026-07-06 K/V Latch Wait-Prune Active

Status: `MQ128_R1_WAIT_PRUNE_MICRO_CURRENT`.

Current active source is still the clean W16/Mq128/Nk128/D128 canonical dKV
kernel with dQ frozen.  It starts from the accepted Sidecar Vec4 LDS baseline
and changes only the one-time resident K/V latch:

- Old K/V latch:
  four repetitions of K/V `ds_read_matrix_trans_pair` followed by
  `wait_lgkm(0)`.
- New K/V latch:
  issue all four K/V read groups first, then one `wait_lgkm(0)` before
  selecting the owner half.
- No q-loop ownership, math, sidecar path, output ownership, ABarrier ledger,
  API, or kernel count change.

Validation:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- WDRA branch windows:
  producer0 `14/16`, consumer0 `188/240`, consumer1 `188/240`,
  producer1 `8/16`.
- Static asm evidence:
  `s_waitcnt=347`, `ds_read_b32=0`, `ds_read_b128=96`,
  `ds_read_matrix=550`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_202609`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_202706`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_203150`.

Performance:

- Full perf `shaderActiveTicks=44,257,395`, `simTicks=47,871,005`.
- `MMAC active=32.7888%`.
- `MMOP=131,072`, `VALU=183,136`, `SCA=115,544`,
  `LDS=79,360`, `VMEM=4,352`.
- Coissue `36,652/26,646`.
- `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_203150_7gemm_wait_prune_kv_latch_h1s1024_sqc7_fullperf`.

XCU finding:

- Dispatch duration is effectively flat against Sidecar Vec4
  (`97,276 -> 97,268`).
- Top bottlenecks remain `s_abarrier_try_wait -> s_xor_b32` at `41.75%`
  and `s_waitcnt` at `19.99%` latency.

Conclusion:

Keep this as a safe micro cleanup because it removes a redundant latch wait
pattern without hurting resources or correctness.  It is not a 60% MMAC-active
solution; the next real optimization must attack ownership waits and useful
overlap in the steady q-loop.

## 2026-07-06 Sidecar Vec4 LDS Reads Active

Status: `MQ128_R1_SIDE_CAR_VEC4_CURRENT_BEST`.

Current active source keeps the clean W16/Mq128/Nk128/D128 canonical dKV
kernel and freezes dQ.  It starts from the read8 score/dP baseline and changes
only the sidecar reads inside the softmax/dS helpers:

- Old sidecar path:
  scalar `float` loads for row max-log2, inverse-sum, and delta.
- New sidecar path:
  one `Vec4F32` LDS load per sidecar family for the four rows owned by the
  lane group.
- Q0/Dout0/Q1/Dout1 half-page ownership tokens remain unchanged.
- K/V resident latch, dV/dK output ownership, score/dP read8 batching, and
  `AllDone` remain unchanged.
- No new kernel, no phase stack, no asm island, no dQ change.

Validation:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- WDRA branch windows:
  producer0 `14/16`, consumer0 `188/240`, consumer1 `188/240`,
  producer1 `8/16`.
- Static asm evidence:
  `ds_read_b32=0`, `ds_read_b128=96`, `ds_read_matrix=550`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_193914`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_193944`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260706_194400`.

Performance:

- Full perf `kernel_ticks=44,260,125`, `simTicks=47,873,735`.
- `MMAC active=32.6559%`.
- `MMOP=131,072`, `VALU=183,136`, `SCA=115,608`,
  `LDS=79,360`, `VMEM=4,352`.
- Coissue `36,479/26,644`.
- `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_194400_7gemm_sidecar_vec4_h1s1024_sqc7_fullperf`.

XCU finding:

- Dispatch duration improves from read8 `103,988` to `97,276`.
- Avg active waves improve from `115.47` to `120.93`.
- The old `ds_read_b32 -> s_waitcnt` sidecar bubble disappears.
- Top bottlenecks remain `s_abarrier_try_wait -> s_xor_b32` at about
  `41.86%` and `s_abarrier_try_wait -> s_waitcnt` at about `8.43%`.

Conclusion:

Promote this as the current best clean micro-baseline because it improves
read8 full-perf ticks by about `6.0%`, removes scattered sidecar LDS reads,
and preserves correctness/no-spill/no-bank-conflict.  It is not a final
FWD-style conveyor: the dominant ABarrier ownership bubble is still present,
VALU rose, and coissue-fail rose.  The next design must attack half-page
ownership waits and useful overlap, not another sidecar-only cleanup.

Rejected follow-up:

- Sheet `72_dout_preread` tried to pre-issue final M-pair dO normal source
  reads under the score/dP DBlock2/3 MMAC island.  It was static-clean and
  correct, but H1/S1024 stats-only regressed from sidecar Vec4
  `simTicks=48,445,215`, `MMAC active=32.6312%` to
  `simTicks=48,590,815`, `MMAC active=32.2856%`.  The source was restored to
  the sidecar Vec4 baseline.  Do not retry this as simple code motion; the next
  dO attempt needs a structural lifetime or producer1-work redesign.

## 2026-07-05 Score/dP Read8 Active

Status: `MQ128_R1_HALF_PAGE_SCORE_DP_READ8_CURRENT_BEST`.

Current active source keeps the clean W16/Mq128/Nk128/D128 canonical dKV
kernel and freezes dQ.  It starts from the half-page conveyor and changes only
the score/dP read/MMAC island:

- Old score/dP per M-pair:
  four repetitions of `4 ds_read_matrix_trans -> wait_lgkm(0) -> 8 MMAC`.
- New score/dP per M-pair:
  two repetitions of `8 ds_read_matrix_trans -> wait_lgkm(0) -> 16 MMAC`.
- Q0/Dout0/Q1/Dout1 half-page ownership tokens remain unchanged.
- K/V resident latch, sidecar layout, dV/dK output ownership, and `AllDone`
  remain unchanged.
- No new kernel, no phase stack, no asm island, no dQ change.

Validation:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- WDRA branch windows:
  producer0 `14/16`, consumer0 `180/240`, consumer1 `180/240`,
  producer1 `8/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_062922`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_062928`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_063337`.

Performance:

- Full perf `kernel_ticks=47,313,175`, `simTicks=50,926,785`.
- `MMOP=131,072`, `VALU=165,872`, `SCA=115,608`,
  `LDS=83,856`, `VMEM=4,352`.
- Coissue `36,333/25,091`.
- `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_063337_clean_read8_score_dp_h1s1024_sqc7_fullperf`.

XCU finding:

- Dispatch duration improves from half-page `106,108` to `103,988`.
- Avg active waves improve from `114.79` to `115.47`.
- `v_mmac -> s_waitcnt` gap improves from `3.92%` to `1.62%`.
- `s_waitcnt -> v_mmac` gap improves from `0.96%` to `0.60%`.
- `s_abarrier_try_wait -> s_xor_b32` remains dominant at about `40.93%`.

Conclusion:

Promote this as the current best clean micro-baseline because full-perf ticks
improve by about `2.00%` versus the half-page conveyor without changing
algorithmic work, instruction counts, resource gates, or LDS/global traffic.
This is not a 60% MMAC-active solution.  The next workbook-first design must
target ABarrier/consumer lockstep and useful overlap during half-token waits;
do not just batch more reads or jump to assembly.

## 2026-07-05 Half-Ring3 Slot Rejected

Status: `MQ128_R1_HALF_RING3_REJECT_REVERT_TO_HALF_PAGE`.

Sheet `70_mq128_half_ring3_design` was implemented in the single canonical
dKV kernel as a three-slot M64 half-packet ring:

- `Slot0Filled/Used = bar2/bar3`, `Slot1Filled/Used = bar4/bar5`,
  `Slot2Filled/Used = bar6/bar7`, `AllDone = bar8`.
- Each slot holds local M64 Q, dO, and sidecar.  Producers publish
  `half_packet % 3`; consumers process local `MBlockBase=0/2` and release the
  slot after both dO and Q source reads.
- No new kernel, no phase stack, no asm island, no output ownership change.

Validation:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=49`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- WDRA branch windows:
  producer0 `8/16`, consumer0 `189/240`, consumer1 `189/240`,
  producer1 `1/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_060908`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_060914`.

Performance:

- Ring3 stats-only H1/S1024:
  `kernel_ticks=50,617,385`, `simTicks=54,230,995`,
  `MMOP=131,072`, `MMAC active=30.2521%`, `VALU=163,682`,
  `SCA=213,896`, `LDS=83,920`, `VMEM=4,352`,
  coissue `28,645/18,016`, `ldsBankConflict=0`.
- Same-debug half-page baseline:
  `kernel_ticks=48,268,220`, `simTicks=51,881,830`,
  `MMAC active=31.6990%`, `SCA=115,608`,
  coissue `34,498/22,594`, `ldsBankConflict=0`.

Conclusion:

Reject before full perf/xcu.  The extra half-slot lookahead and lower token
family count did not compensate for paired Q/dO slot coupling and extra
slot-control/SCA.  The active source should be restored to the half-page
conveyor baseline.  Future topology work must preserve early Q/dO lifetime
benefit or create useful consumer work during ownership waits; ring depth alone
is not enough.

## 2026-07-05 Half-Page Conveyor Active

Status: `MQ128_R1_HALF_PAGE_CONVEYOR_CURRENT_BEST`.

Current active source keeps the clean W16/Mq128/Nk128/D128 canonical dKV
kernel and freezes dQ.  It refines the prior Q/dO lifetime split by cutting
the Mq128 page into two M64 semantic ownership halves on the same physical LDS
pages:

- `Q0Filled/Q0Used = bar2/bar3`, `Dout0Filled/Dout0Used = bar4/bar5`.
- `Q1Filled/Q1Used = bar6/bar7`, `Dout1Filled/Dout1Used = bar8/bar9`.
- `AllDone = bar10`; do not delete it by inspection because the previous
  cleanup changed WDRA metadata into private/VGPR spill.
- Producers publish half0 then half1.  Consumers consume MBlockBase `0/2`,
  release half0, then consume MBlockBase `4/6`, release half1.
- No new kernel, no new phase stack, no extra full LDS page, no output
  ownership change, and no asm island was added.

Validation:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=99`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- WDRA branch windows:
  producer0 `14/16`, consumer0 `180/240`, consumer1 `180/240`,
  producer1 `8/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_053013`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_053032`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_053321`.

Performance:

- Full perf `kernel_ticks=48,279,140`, `simTicks=51,892,750`.
- `MMOP=131,072`, `MMAC active=31.7858%`, `VALU=165,872`,
  `SCA=115,608`, `LDS=83,856`, `VMEM=4,352`.
- Coissue `33,962/22,131`.
- `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_053321_clean_half_page_conveyor_h1s1024_sqc7_fullperf`.

XCU finding:

- Dispatch aggregate:
  `s_abarrier_try_wait -> s_xor_b32` remains top at `40.42%`;
  `s_abarrier_try_wait -> s_waitcnt` is `9.07%`;
  `v_mmac -> v_mmac` is `7.50%`.
- Focused half-token windows:
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

Promote this as the current best clean baseline because it improves the Q/dO
split full-perf ticks from `51,238,915` to `48,279,140` and MMAC active from
`29.6586%` to `31.7858%` while preserving correctness and resource gates.
It is not a final FWD-style conveyor: the focused windows remain
`94-96%` bubble.  The next design must create real useful overlap during the
half-token waits, reduce handshakes without restoring a full-page cliff, or
lengthen useful MMAC islands without duplicate score/dP.

Next workbook design:

- Sheet `70_mq128_half_ring3_design` proposes a three-slot M64 half-packet
  ring.  It adds one extra half slot rather than a full Mq128 double buffer:
  q stream order becomes `q0h0 slot0 -> q0h1 slot1 -> q1h0 slot2 ->
  q1h1 slot0`.
- Estimated LDS after K/V latch is about `98.25KB` raw/sidecar, still below
  128KB.
- Preferred protocol is one paired `Filled/Used` counted token per slot
  (`Filled=8`, `Used=8`) to reduce steady token count versus separate
  Q/Dout half tokens.
- Main risk is SGPR/control pressure and losing early dO release due to Q/dO
  coupling.  Reject before PMD on any spill/private segment.

## 2026-07-05 Q/dO Lifetime Split Active

Status: `MQ128_R1_QDO_LIFETIME_SPLIT_MICRO_BASELINE`.

Current active source keeps the clean W16/Mq128/Nk128/D128 canonical dKV
kernel and freezes dQ.  It changes the 62C2 raw ownership contract from one
combined Q+dO token to two semantic lifetimes on the same physical LDS layout:

- `QFilled/QUsed = bar2/bar3`, published by waves0-3 with Q + sidecar.
- `DoutFilled/DoutUsed = bar4/bar5`, published by waves12-15 with dO.
- Consumers wait both Q and dO before score/dP.
- On the final M-pair, consumers read low/high dO normal sources, wait, arrive
  `DoutUsed`, run softmax/dS, read low/high Q normal sources, wait, arrive
  `QUsed`, then run the combined dV/dK MMAC island.
- No new kernel, phase, raw page, output ownership, or asm island was added.

Validation:

- Remote build/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=97`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- WDRA branch windows:
  producer0 `14/16`, consumer0 `180/240`, consumer1 `180/240`,
  producer1 `8/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_044322`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_044345`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_044647`.

Performance:

- Full perf `kernel_ticks=51,238,915`, `simTicks=54,852,525`.
- `MMOP=131,072`, `MMAC active=29.6586%`, `VALU=165,744`,
  `SCA=108,632`, `LDS=83,856`, `VMEM=4,352`.
- Coissue `27,090/16,944`.
- `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_044647_clean_qdo_split_h1s1024_sqc7_fullperf`.

XCU finding:

- Top bubble remains `s_abarrier_try_wait -> s_xor_b32`, `42.85%`.
- The old combined RawUsed bubble split into:
  `bar3 QUsed = 2,266,380 cycles` and
  `bar5 DoutUsed = 2,251,240 cycles`.
- Combined ownership wait is still about `4.52M` cycles, close to 62C2
  `bar3 Raw0Used = 4.62M` cycles.

Conclusion:

Keep this as a micro-positive baseline because it is correctness/resource clean
and improves 62C2 full-perf ticks and MMAC active.  It is not a qualitative
pipeline solution: the next workbook design must reduce or hide the combined
`QUsed + DoutUsed` ownership bubble instead of adding more token names.

Negative follow-up:

- Workbook sheet `67_mq128_prune_alldone` tried to remove the tail `AllDone`
  counted ABarrier and rely on the existing post-branch `__syncthreads()`.
- Remote build/source gate PASS, but symbol metadata failed:
  `private_segment_fixed_size=244`, `vgpr_spill_count=60`.
- Active source was restored to keep `AllDone`; do not retry this cleanup
  without a focused WDRA CFG/codegen probe.

Focused XCU follow-up:

- Workbook sheet `68_qdo_focused_xcu` drills into representative ownership
  windows from the Q/dO split perf.
- `bar3 QUsed` window `7600:21000`, `xcd0/se0/cu0/simd1/wave0`:
  producer pipeline bubble `99.60%`; same-SIMD mix bubble `95.46%`,
  MMAC `0.93%`, VALU `1.35%`.
- `bar5 DoutUsed` window `19400:31300`, `xcd0/se0/cu0/simd3/wave3`:
  producer pipeline bubble `99.70%`; same-SIMD mix bubble `94.97%`,
  MMAC `1.07%`, VALU `1.74%`.
- Coissue in the consumer slots is dominated by `v_mov_b32/e32` or
  `v_mov_b64/e32`, not the desired softmax/dS VALU covering MMAC.
- Next design must change the steady q-loop ownership/pipeline shape; token
  renaming and tail cleanup are not enough.

## 2026-07-05 62C2 RawUsed XCU Diagnosis

Status: `MQ128_R1_62C2_ACTIVE_AFTER_XCU_DIAGNOSIS`.

No kernel source change was made.  The active source remains 62C2:
W16/Mq128/Nk128/D128, `ActiveDkvTile=DkvTileD128MqNk128<128,1>`,
consumer VGPR window 240, exact causal predicate, no asm island.

New evidence:

- xcu top2000 output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/62c2_mq128_h1s1024_fullperf_20260705_033019_dispatch0_top2000`.
- Focused Raw0Used window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/62c2_mq128_h1s1024_raw0used_window_7000_22000`.
- Shared archive copy:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_033019_clean_62c2_mq128_h1s1024_sqc7_fullperf`.

Findings:

- `bar3 Raw0Used`, `s_abarrier_try_wait -> s_xor_b32`:
  total `4,623,276` cycles, count `448`, average `10,319.8`,
  max `13,427`.
- `bar6 AllDone` is also visible:
  total `1,238,870` cycles, count `110`, but this is tail/cleanup and should
  not be optimized before Raw0Used.
- Raw0Used window `7000:22000` on `xcd0,se0,cu0,simd3,wave3`:
  pipeline bubble `98.60%`; same-SIMD mix reports `Bubble=95.59%`,
  `MMAC=0.85%`, `VALU=1.19%`.
- Same-SIMD consumer slots still issue MMAC, but MMAC+VALU coissue is only
  `6.54%` and `8.27%`; producer slots are mostly waiting.

Conclusion:

Current 62C2 is the best clean baseline but not a 60% active design.  The next
candidate should be designed around raw Q/dO ownership lifetime, especially a
possible split between Q and dO half-page lifetimes.  Assembly is not the next
default move because the dominant evidence is an ownership/barrier bubble, not
a single compiler-generated hot island.

## 2026-07-05 Mq128 64 Rejected, 62C2 Still Active

Status: `MQ128_R1_62C2_ACTIVE_AFTER_64_REJECT`.

Sheet 64 tried a full-valid softmax/dS helper to remove per-element causal
branches for the 48.4375% full-valid M-pairs in H1/S1024.  The candidate was
reverted before PMD because it failed the no-spill static gate.

Evidence:

- 64A build/source gate PASS; branch windows `14/186/186/8`.
- 64A metadata FAIL:
  `private=0`, `sgpr=100`, `sgpr_spill=4`, `vgpr=128`,
  `vgpr_spill=0`.
- 64B removed saved `full_valid_*` booleans, but metadata stayed identical:
  `sgpr_spill=4`.

Conclusion:

Current active source remains 62C2.  Full-valid specialization is not promoted
until a lower-SGPR formulation exists.  Do not run PMD or perf on a spilling
variant.

## 2026-07-05 Mq128 63 Rejected, 62C2 Restored

Status: `MQ128_R1_62C2_ACTIVE_AFTER_63_REJECT`.

Sheet 63 tried to split sidecar lifetime from raw Q/dO lifetime and release
Raw0Used before final softmax.  The candidate was intentionally reverted from
the active source after correctness failure.

Evidence:

- Static/resource gate was clean:
  `private=0`, `sgpr=98`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`; branch windows `14/190/190/8`.
- 63A H1/S128 PASS, but H1/S1024 FAIL:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_035331`,
  `dk_rel_l2=0.0622111`, `dv_rel_l2=0.0326977`.
- 63B added a wait before RawUsed release.  H1/S128 PASS, but H1/S1024 still
  FAIL:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_035616`,
  `dk_rel_l2=0.0638349`, `dv_rel_l2=0.047026`.

Conclusion:

Current active source is restored to 62C2.  Do not keep release-before-softmax
code in the canonical route.  Future raw lifetime changes must prove
correctness on H1/S1024 or a focused instruction/lifetime probe before any
performance claim.

## 2026-07-05 Mq128 62C2 Candidate Active

Status: `MQ128_R1_62C2_ACTIVE_CANDIDATE`.

Current source state:

- Active dKV route is the clean W16/Mq128/Nk128/D128 canonical kernel:
  waves0-3 producer K+Q+sidecar, waves4-7 consumer0,
  waves8-11 consumer1, waves12-15 producer V+dO.
- `ActiveDkvTile = DkvTileD128MqNk128<128, 1>`.
- Consumer WDRA window is 240; producer windows remain 16.
- Canonical performance path requires exact causal tiles:
  `causal == 1`, `seqlen_q % Mq == 0`, `seqlen_k == seqlen_q`,
  `seqlen_k % Nk == 0`, `num_heads_kv == num_heads_q`.
- Hot softmax/dS helper uses the canonical exact-tile predicate
  `owner_krow <= qrow`; it no longer carries runtime seqlen bounds or
  full-valid control in the hot path.
- Main matrix path remains MLS/BPS + `ds_read_matrix` +
  `v_mmac_f32_16x16x16_f16`; no asm island is active.
- dQ remains frozen.

Validation:

- Static/source gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=96`, `sgpr_spill=0`, `vgpr=128`,
  `vgpr_spill=0`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_032159`.
- H1/S1024 correctness/stats PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_032222`.
- Full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_033019`.
- Shared perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260705_033019_clean_62c2_mq128_h1s1024_sqc7_fullperf`.

Metrics:

- H1/S1024 full-perf stats:
  `kernel_ticks=52,163,020`, `MMOP=131,072`,
  `MMAC active=29.2001%`, `VALU=167,536`, `SCA=106,968`,
  `LDS=83,856`, `VMEM=4,352`, coissue `25,179/15,960`,
  `ldsBankConflict=0`.
- Raw2 recert comparison:
  `kernel_ticks=53,008,410`, `MMAC active=27.7754%`,
  `VALU=181,980`, `SCA=296,328`.
- XCU dispatch0:
  duration `114,644`, avg active waves `117.91`.
- XCU top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `44.65%`,
  `s_abarrier_try_wait -> s_waitcnt` `9.57%`.
- Representative window
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/62c2_mq128_h1s1024_fullperf_20260705_033019_dispatch0_window_bar6`
  at `93000:113000`, `xcd0,se1,cu0,simd1,wave0`:
  `Bubble=96.51%`, `MMAC=0.70%`.

Conclusion:

62C2 becomes the active candidate baseline because it finally makes Mq128/R1
static/resource-clean and improves same-shape ticks/MMAC active versus raw2.
It is not a pipeline solution.  The next top-level design must attack the
steady Raw0Used/barId3-class ownership waits and make producer/consumer overlap
real; barId6 `AllDone` tail wait is visible but should not be optimized in
isolation.  Do not start by hand-writing the whole kernel in assembly.

Next:

- Keep 62C2 as the code baseline unless a new candidate fails and must be
  reverted.
- Update workbook before the next code change.
- Design from xcu evidence: reduce or hide `s_abarrier_try_wait` windows, then
  remeasure H1/S1024 with the same gates.
- Assembly remains a short-island fallback only after C++/topology work and
  xcu/asm prove a specific compiler-generated hot island is the blocker.

## 2026-07-05 Raw2 Canonical After ASM And Causal-Skip Negatives

Status: `RAW2_CANONICAL_ACTIVE`.

Current source state:

- Live source should remain the raw2 W16/Mq64/Nk128/D128 canonical dKV route:
  waves0-3 producer K+Q+sidecar, waves4-7 consumer0,
  waves8-11 consumer1, waves12-15 producer V+dO.
- dQ is frozen.
- No public phase stack, no alternate dKV kernel, and no asm island is active.
- Post-revert remote validation:
  - Build/static PASS on zys1, branch windows `6/198/198/1`.
  - Symbol metadata PASS: `private=0`, `sgpr=60`, `vgpr=112`,
    `sgpr_spill=0`, `vgpr_spill=0`.
  - Correctness PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_010339`
    for H1/S128 and
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_010344`
    for H1/S1024.
  - H1/S1024 recert stats:
    `kernel_ticks=53,008,410`, `MMOP=131,072`,
    `MMAC active=27.7754%`, coissue `32,341/18,768`,
    `ldsBankConflict=0`.

Latest rejected probes:

- `w16_score_dp_pair_asm_island` built cleanly, but failed H1/S128
  correctness.  The mistake was applying the resident K/V `+1024` pair-read
  relation to raw Q/dO M0/M1 score/dP reads, whose D128 raw-page separation is
  `4 * 1024` bytes.
- `w16_causal_prefix_page_skip` reduced H1/S1024 MMOP from `131,072` to
  `77,312`, but did not improve same-shape ticks and dropped MMAC active to
  `22.4979%`.  It is rejected even though it is correctness/resource clean.
- `w16_raw_release_before_softmax` tried to release RawUsed before softmax by
  pre-reading the final M-pair Q/dO sources.  Q/dO-only preread passed static
  but failed H1/S1024 correctness because sidecar rows were overwritten after
  release.  Q/dO+sidecar preread failed static with `private=52` and
  `vgpr_spill=24`.  Lesson: raw page ownership includes sidecar lifetime.
- `w16_sidecar_ring3_early_raw_release` moved sidecar into a three-page LDS
  ring so raw release-before-softmax was legal, but H1/S1024 regressed to
  `kernel_ticks=55,298,425`, `MMAC active=26.5015%`.
- `w16_producer_sidecar_rebalance` moved sidecar publication from producer0 to
  producer1 and passed correctness/resource gates, but H1/S1024 regressed to
  `kernel_ticks=53,558,960`, `MMAC active=27.5554%`.
- `w16_full_valid_mask_shrink` removed per-element valid-pair masking on
  full-valid owner16 pairs, but failed H1/S128 correctness with dV corruption
  (`dv_rel_l2=33.2914`) and PMD `read vgpr165 before writing`.
- `w16_sidecar_reg_prefetch_before_wait` loaded sidecar rows into producer VGPR
  before RawUsed wait and wrote to LDS after page ownership.  It passed
  correctness/resource gates, but H1/S1024 regressed to
  `kernel_ticks=53,658,605`, `MMAC active~=27.4726%`.
- `w16_half_page_raw_tokens` split each Mq64 raw page into M32 half tokens.
  It passed correctness/resource gates with branch windows `7/198/198/1`, but
  H1/S1024 regressed to `kernel_ticks=53,505,270`,
  `MMAC active=27.3801%`, and `SCA=330,730` versus raw2 `SCA=296,328`.
  Lesson: finer RawUsed granularity is not free; do not split ownership tokens
  again unless the design also removes protocol/control work or increases the
  useful MMAC island.
- `w16_mq128_singlebuf_static` tried the sheet 60 long-island route:
  `ActiveDkvTile=DkvTileD128MqNk128<128,1>` plus a static MBlockBase
  `0/2/4/6` chain.  It failed metadata before PMD with `private=8`,
  `sgpr=104`, `sgpr_spill=18`, `vgpr_spill=2`, and consumer branch windows
  `208/208`.  Lesson: Mq128 needs a resource redesign before code expansion;
  do not keep a dynamic Mq128 fallback as the performance route.
- `w16_mq128_vgpr240_retest` repeated the same static Mq128 route with
  consumer WDRA windows raised to 240 so the per-SIMD ledger exactly matched
  `P16+C240+C240+P16=512`.  It removed the private/VGPR spill but still failed
  metadata with `sgpr=100`, `sgpr_spill=18`, `vgpr=128`,
  `vgpr_spill=0`; branch windows were `15/16`, `209/240`, `209/240`,
  `9/16`.  Lesson: direct static Mq128 is blocked by scalar/control live
  range, not consumer VGPR window size alone.

Next:

- Keep raw2 canonical active.
- Treat assembly only as a later short-island fallback after a layout proof and
  xcu/asm evidence.
- Future causal optimization should change launch/tile ownership or the
  barrier-critical structure; do not keep adding consumer-side skip branches.
- Future RawUsed work must either move sidecar out of the raw page lifetime or
  reduce consumer live state before attempting release-before-softmax again.
- Future ABarrier work must reduce total ownership handshakes or hide them
  under larger useful work; half-page tokenization alone is now a recorded
  negative.
- Future larger-Mq work must first reduce helper live ranges or change phasing;
  local static expansion of the current Mq64 helper shape spills.  Raising
  consumer VGPR to 240 is not enough if SGPR/control state is unchanged.
- Next design basis is workbook sheet `62_mq128_sgpr_control_shrink`: start
  with 62A causal-true control-context shrinking, then only consider 62B
  two-half lexical scopes or 62C softmax helper split if static metadata proves
  the earlier step is insufficient.
- `w16_mq128_62a_causal_control_shrink` has now been tested as a static-only
  resource probe.  It improved consumer branch windows from `209/240` to
  `182/240` and reduced `sgpr_spill` from `18` to `14`, but metadata still
  failed (`private=0`, `sgpr=100`, `vgpr=128`, no VGPR spill).  No PMD was run.
  The code is reverted to raw2 and remote recert PASS.  Continue only with a
  stronger scalar/control split; 62A alone is rejected.

## 2026-07-04 Raw2 Canonical After Rejected Stagger

Status: `RAW2_CANONICAL_ACTIVE`.

Current source state:

- Active dKV kernel is the W16 K/V-resident Mq64 route with page-local raw
  ownership:
  waves0-3 producer K+Q+sidecar, waves4-7 consumer0,
  waves8-11 consumer1, waves12-15 producer V+dO.
- `ActiveDkvTile = DkvTileD128MqNk128<64>`.
- `kRawBuffers=2`.
- ABarrier ledger:
  `ResidentFilled=0`, `ResidentUsed=1`, `Raw0Filled/Used=2/3`,
  `Raw1Filled/Used=4/5`, `AllDone=6`.
- Consumer0 and consumer1 both use the canonical pair order:
  pair0 score/dP -> softmax/dS -> dV/dK, then pair2.

Latest rejected probe:

- `w16_raw2_g1_score_prefetch_stagger` made consumer1 compute score/dP for
  pair0 and pair2 before finishing pair0/pair2.
- Static/resource and correctness passed, but H1/S1024 regressed:
  `kernel_ticks=56,500,990` versus raw2 stats-only baseline
  `53,300,975`.
- MMAC active rose only slightly (`27.6518% -> 28.0755%`), so the probe was
  rejected and removed from live source.
- `w16_raw2_causal_true_specialize` required canonical `causal==1` and passed
  literal causal into consumers.  It reduced SGPR/SCA but regressed H1/S1024
  to `kernel_ticks=56,200,690`, so it was also rejected and removed.
- `w16_raw2_score_read_batch2` changed score/dP to
  `8 ds_read_matrix -> wait -> 16 MMAC` chunks.  It was resource/correctness
  clean, but H1/S1024 regressed to `kernel_ticks=56,275,310`, so it was
  rejected and removed.

Next:

- Keep raw2 canonical as the active route.
- Do not add another stagger variant that keeps two score/dP pairs live.
- Next candidates should either reduce softmax/control `v_mov` pressure or
  build a larger clean MMAC island without extra ABarrier generations or
  score/dP duplication.

## 2026-07-04 Single Raw Page + Producer Sidecar LDS

Status: `ACCEPT_PIPELINE_SIDECAR_LDS_SINGLEBUF`.

Current source state:

- Active dKV kernel remains the W16 K/V-resident Mq64 route:
  waves0-3 producer K+Q, waves4-7 consumer0, waves8-11 consumer1,
  waves12-15 producer V+dO.
- `kRawBuffers=1`; the active Q/dO raw page is a single page with sidecar
  metadata published into LDS by producer A.
- A compile-time page helper keeps the page arithmetic explicit:
  `raw_page_for_q_tile<Tile>(q_tile)`.
- Producer A writes a sidecar SoA page in LDS:
  `max_log2[64]`, `inv_sum[64]`, `delta[64]`.
- Consumers call `softmax_ds_owner16_from_lds_sidecar`; the active consumer
  path no longer reads `packed_sidecar` directly from global memory.
- Remote build/static/symbol gates pass:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `vgpr_count=112`; branch windows producer KQ `10/16`,
  consumer0 `196/208`, consumer1 `196/208`, producer VDout `4/16`.

Evidence:

- Double-buffer baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_153154`,
  `kernel_ticks=61582885`, `MMAC active=25.4747%`,
  coissue `26857/16837`, `ldsBankConflict=0`.
- Single-buffer temporary probe:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_153511`,
  `kernel_ticks=63344645`, `MMAC active=24.6431%`,
  coissue `19026/12647`, `ldsBankConflict=0`.
- Current single-buffer + LDS sidecar:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_154650`,
  `kernel_ticks=54539485`, `MMAC active=26.6693%`,
  `VALU=180570`, `SCA=215648`, `LDS=85822`, `VMEM=4352`,
  coissue `20030/11508`, `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_154650_clean_singlebuf_lds_sidecar_h1s1024_sqc7_fullperf`.
- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `35_singlebuf_lds_sidecar`.
- xcu top bubbles after the change:
  `s_abarrier_try_wait -> s_xor_b32` `47.39%`,
  `s_abarrier_try_wait -> s_waitcnt` `6.60%`,
  `ds_read_b32 -> s_waitcnt` `2.34%`.
- In the exported window `100000:118000`, SIMD bubble is `96.76%`, dominated
  by `s_abarrier_try_wait -> s_waitcnt`.

Conclusion:

Moving sidecar from consumer global reads into producer-published LDS is a real
win.  It beats the double-buffer global-sidecar baseline by about `11.44%`
kernel ticks and removes `global_load_dwordx3 -> s_waitcnt` from the top xcu
bubbles.  The price is that a single raw page exposes the page-control problem:
ABarrier/RawUsed-RawFilled serialization is now the dominant bottleneck.

Next:

- Keep sidecar LDS as the current active route.
- Do not interpret this as final FWD-style conveyor success; MMAC active is
  still only `26.6693%`.
- Next candidate should restore useful lookahead while keeping sidecar out of
  the consumer global path: either reintroduce two raw pages with sidecar pages,
  split sidecar/RawFilled ownership, or use the remaining LDS slack for a
  workbook-stressed third raw+sidecar page.

## 2026-07-03 BlockN / Owner-N32 Direct Expansion Probe

Status: `REJECT_STATIC_SPILL`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `33_blockn128_stress`

Clarification:

- Current CTA-level K/V resident width is already `kResidentNk=128`.
- The meaningful experiment was per-consumer owner-N expansion: N16 -> N32.

Temporary probe:

- A direct N32 implementation was compiled by giving each active consumer wave
  two N16 K/V halves and two dV/dK accumulator sets.
- Only waves4-7 owned output rows in the probe; waves8-11 were disabled for
  output to avoid duplicate stores.
- The code was removed after resource evidence was collected.

Evidence:

- Build completed, but symbol metadata failed:
  `private_segment_fixed_size=432`, `sgpr_count=104`,
  `sgpr_spill_count=22`, `vgpr_count=64`, `vgpr_spill_count=110`.
- Branch report showed consumer exactly at `208/208`, with spills/private
  segment already present.
- The live baseline was restored and remote gates passed:
  branch consumers `195/208`,
  `private_segment_fixed_size=0`, `sgpr_count=62`, `vgpr_count=112`,
  no SGPR/VGPR spill.

Conclusion:

Do not pursue direct owner-N32/N64/N128 by simply holding more long-lived
accumulators in the consumer.  Larger BlockN needs accumulator phasing or a
store-safe partial-reduction design before it can be a performance candidate.

## 2026-07-03 W16 K/V Resident Mq64

Status: `ACCEPT_PIPELINE_RESOURCE_WIN`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `32_kv_reg_resident`

Hypothesis:

- For a block-local fixed K/V tile, K/V should be read from LDS into each
  consumer wave's VGPR once, then reused across all q-loop score/dP work.
- This removes repeated K/V `ds_read_matrix` from the hot q-loop, reduces
  wait pressure, and makes K/V LDS lifetime available for later Q/dO buffering.

Current source shape:

- One canonical dKV kernel, no new public route.
- W16 CTA: waves0-3 producer K+Q, waves4-7 consumer0, waves8-11 consumer1,
  waves12-15 producer V+dO.
- Consumers latch `Nk=16,D=128` K/V into `Owner16KvRegs` after
  `ResidentFilled`.
- Active Mq64 loop consumes M rows as two half-sequential M32 groups to avoid
  holding both halves' score/dP state at once.

Evidence:

- Remote build/static/symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=62`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Branch windows:
  producer KQ `6/16`, consumer0 `195/208`, consumer1 `195/208`,
  producer VDout `1/16`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_170546`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_170901`,
  `kernel_ticks=61582430`, `MMOP=131072`, `VALU=181512`,
  `SCA=311168`, `LDS=66816`, `VMEM=4352`, coissue `26862/16883`,
  `MMAC active=25.4935%`, `VOP active=19.7415%`,
  `ldsBankConflict=0`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_170901_clean_kv_reg_resident_mq64_h1s1024_sqc7_fullperf`.

Conclusion:

K/V resident is the current clean W16 baseline and should be kept.  It is a
real resource and pipeline improvement, but the main xcu bottleneck remains
ABarrier/page-control plus sidecar/global wait, not K/V matrix-read latency.
Do not jump directly to Mq128 unless the workbook shows LDS/VGPR slack and a
clear expected reduction in `RawUsed`/sidecar bubbles.

## 2026-07-03 H19A Pre-Read All Source Before Softmax

Status: `REJECT_RESOURCE`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `18_barrier_audit`

Hypothesis:

- xcu top1000 showed producer-side `RawUsed0/1` waits dominate the barrier
  bubbles.
- Pre-read both low and high source-layout `dO^T/Q^T` operands before
  softmax/dS, issue one `wait_lgkm(0)`, then arrive `RawUsed` before running
  softmax and the dV/dK MMAC island.
- Expected benefit: shorten raw page ownership lifetime and reduce producer
  `RawUsed` wait.

Resource evidence:

- Remote build completed, but symbol metadata rejected the kernel:
  `private_segment_fixed_size=24`, `vgpr_spill_count=10`, `sgpr_count=80`,
  `vgpr_count=112`.
- The consumer branch hit the current 160 VGPR local window after keeping both
  low and high source fragments live across softmax/dS.

Conclusion:

Do not keep this in the active route.  The source-read-before-softmax idea is
only viable as a separately justified H19B with a wider consumer VGPR window
such as 208, or after shrinking the softmax/sidecar live range.  The H19A code
was removed from `src/dkv_kernel.cpp` before further optimization.

Restored baseline evidence:

- Remote build/static/symbol metadata PASS after revert:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`, branch consumers `150/160`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_033619`,
  `kernel_ticks=15342145`, `MMOP=2048`, `ldsBankConflict=0`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_033625`,
  `kernel_ticks=70444920`, `MMOP=131072`, `coissue=32369/20986`,
  `ldsBankConflict=0`.

## 2026-07-03 H19B 208 VGPR Pre-Read All Source

Status: `REJECT_PERF_STATS_ONLY`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `18_barrier_audit`

Hypothesis:

- Reuse H19A's earlier `RawUsed` release timing, but widen W12 consumer role
  from 160 to 208 VGPR.
- Resource estimate: one producer wave around 16 VGPR plus two consumer waves
  at 208 each is about `432 < 512` VGPR per SIMD, so this is a legal thing to
  test rather than an automatic rejection.

Evidence:

- Static/resource gate PASS:
  `private_segment_fixed_size=0`, `sgpr_count=76`, `vgpr_count=144`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`; branch consumers `164/208`.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_034717`,
  `kernel_ticks=15167425`, `MMOP=2048`, `ldsBankConflict=0`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_034723`,
  `kernel_ticks=77708085`, `MMOP=131072`, `coissue=31351/23990`,
  `ldsBankConflict=0`.
- Restored canonical comparison:
  `kernel_ticks=70444920`, `MMOP=131072`, `coissue=32369/20986`.

Conclusion:

208 VGPR makes the pre-read-all-source schedule resource-clean and correct, but
it regresses S1024 ticks by about `10.31%` and does not improve coissue.  Do
not promote wider consumer VGPR as a standalone fix for the RawUsed bubble.
The active route was reverted.  Next work should reduce the consumer
softmax/sidecar/mask work or change page ownership with less scheduling cost.

## 2026-07-03 H20A Owner16 Full-Valid Softmax Fast Path

Status: `REJECT_CORRECTNESS`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `18_barrier_audit`

Hypothesis:

- `softmax_ds_owner16_from_global_sidecar` already computes a uniform
  `full_valid_tile` predicate.
- Split full-valid and masked paths so full-valid causal tiles do not compute
  per-element `qrow/valid_pair` branches.
- Expected benefit: shrink consumer VALU/mask work without changing MMAC,
  ABarrier, LDS, or VGPR window.

Evidence:

- Static/resource gate PASS:
  `private_segment_fixed_size=0`, `sgpr_count=83`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`; branch consumers `155/160`.
- H1/S128 correctness failed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_035528`,
  `dk_rel_l2=0.000361379`, `dv_max_abs=0.518505`, `dv_rel_l2=14.4712`,
  `pass=0`.
- Retried with explicit `p_vec{}`/`ds_vec{}` initialization:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_035643`,
  same dV failure pattern.

Conclusion:

This matches the older full-valid fastpath negative: the full-valid split is
not safe in the main route with the current owner16 fragment/codegen shape.
The active route was reverted.  Do not retry this in the main kernel without a
focused fragment-layout/codegen probe that proves dV is preserved.

## 2026-07-03 H18A Packed Sidecar dwordx4 Probe

Status: `REJECT_PERF_STATS_ONLY`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `18_barrier_audit`

Hypothesis:

- xcu showed the second-largest bubble as
  `global_load_dwordx3 -> s_waitcnt`, 1.401M cycles / 11.23%, from
  `softmax_ds_owner16_from_global_sidecar`
- pad packed sidecar rows from 3 floats to 4 floats and force the fourth lane
  live so the compiler emits `global_load_dwordx4`
- expected benefit: a more regular 16B sidecar load shortens the consumer
  softmax/dS critical section and reduces producer `RawUsed` wait

Evidence:

- static/resource gate stayed clean:
  `private=0`, `sgpr=84`, `vgpr=112`, no scratch/spill
- asm changed as intended: hot sidecar loads became `global_load_dwordx4`
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_031739`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_031804`
- H1/S1024 stats:
  `simTicks=74270105`, `kernel_ticks=70656495`, `MMOP=131072`,
  `coissue=31641/20782`, `ldsBankConflict=0`

Comparison:

- canonical W12 companion stats:
  `kernel_ticks=70505435`, `coissue=31497/20470`
- H18A is slightly worse on ticks and coissue rate
- `VALU_cycles` stayed unchanged at `2289408`

Conclusion:

The sidecar 3-float load shape is not the main 60% MMAC-active blocker.  The
probe proves the opcode can be changed, but this does not shorten the measured
consumer page lifetime enough to reduce the dominant `RawUsed` bubble.  Revert
the ABI/code change and keep the conclusion: the next design must either reduce
the amount of consumer-side softmax/mask/sidecar work, or change the packet
lifetime so producer `RawUsed` waits are hidden by useful work.

## 2026-07-03 W16 WG-Local Nk64 Semantic Conveyor

Status: `REJECT_PERF_STATS_ONLY`

Workbook-first design sheet:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `17_wg_local_nk64_design`

Hypothesis:

- split the CTA into two independent WG-local semantic conveyors:
  waves0-3 producer WG0, waves4-7 consumer WG0, waves8-11 consumer WG1,
  waves12-15 producer WG1
- each WG owns private `K/V64` plus two semantic pages that first hold raw
  `Q/dO`, then are reloaded as source-layout `Q^T/dO^T`
- expected benefit: remove shared W12 RawUsed lockstep and let source MLS hide
  under softmax/peer WG MMAC

Resource stress correction:

- naive private raw double pages plus private source double pages would be
  `192KB` and illegal
- legal version reuses two semantic pages per WG:
  `K/V64 32KB + 2 pages * 16KB = 64KB/WG`, two WGs exactly `128KB`

Evidence:

- build/static resource PASS for `fa3_bwd_dkv_mmac_kernel`:
  `private=0`, `sgpr=86`, `vgpr=88`, no SGPR/VGPR spill
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_024846`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_024909`
- H1/S1024 stats:
  `kernel_ticks=80790710`, `MMOP=131072`, `ldsBankConflict=0`,
  `coissue=30775/21592`, `MMAC active=19.1856%`,
  `VOP active=27.6249%`

Conclusion:

This is a useful structural negative.  WG-local semantic pages preserve
correctness and avoid spill, but they do not move MMAC active toward 60%.
Duplicating Q/dO/source global loads and adding raw/source page epochs costs
more than the independence gained from removing the shared W12 page.  Do not
promote this route.  The next high-ceiling design should preserve W12's shared
double-buffer advantage or find a way to lengthen the consumer MMAC island
without source-epoch serialization.

## 2026-07-03 RawUsed Release After High-Read Candidate A

Status: `REJECT_XCU_PRIMARY_METRIC`

Candidate A moved `arrive_raw_used_page` before the low dV/dK MMAC island,
immediately after issuing the high source `ds_read_matrix`.  The intent was to
let the producer reuse raw pages earlier while the consumer still did useful
MMAC work.

Verified evidence:

- Static/resource gate stayed clean:
  `private=0`, `sgpr=84`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_020841`.
- H1/S1024 stats-only correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_020926`.
- H1/S1024 full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_021203`.
- Full perf stats:
  `kernel_ticks=69865705`, MMAC active `21.8495%`,
  `coissue=33894/21881`, `ldsBankConflict=0`.
- xcu evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/rawused_candidateA_h1s1024_20260703_021203`.

Comparison to canonical W12 early release:

- `s_abarrier_try_wait -> s_xor_b32` only moved from `3.576M/28.66%` to
  `3.538M/28.61%`.
- `s_waitcnt` hot latency worsened from `2.960M/22.04%` to `3.007M/22.57%`.
- full-perf MMAC active stayed flat to slightly worse.

Conclusion:

The code was reverted to the canonical release timing.  Candidate A is a useful
negative: advancing RawUsed by a few instructions is not enough to remove the
dominant ABarrier conveyor bubble.  The active route remains the W12 canonical
early-release kernel plus the W16 q-loop bound correctness fix.

## 2026-07-03 Mq64 Long-Island Reorder Negative

Status: `REJECT_CORRECTNESS`

A workbook-first experiment tried to make the existing Mq64 single-buffer dKV
path more FWD-like without adding a new kernel/path.  The hypothesis was that
larger high-cohesion helpers could create a longer MMAC island than the current
half-sequential Mq64 path:

- variant A: `score/dP half0 + score/dP half1 -> softmax half0 + softmax half1 -> dV/dK half0 + half1`
- variant B: `score/dP half0 + score/dP half1 -> softmax+dV/dK half0 -> softmax+dV/dK half1`
- variant C: full Mq64 helper: `score[4]/dp[4] -> softmax[4]/dS[4] -> dV/dK[4]`

Static/resource evidence was clean for all variants:

- no private segment, no scratch, no SGPR/VGPR spill
- Mq64 metadata stayed at `sgpr_count=100`, `vgpr_count=144`
- branch-local consumer pressure was `191/208`, `187/208`, and `179/208`

Correctness evidence rejected the direction:

- variant A: `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_012315`
- variant B: `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_012619`
- variant C: `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_012810`
- all three failed H1/S128 causal correctness with exact `dK` but bad `dV`:
  `bad=16384`, `dv_max_abs=2.73637e-05`, `dv_rel_l2=0.000267234`, `pass=0`

Conclusion:

The code was restored to the prior Mq64 half-sequential route.  Simple
live-range stretching across Mq64 halves is not a safe way to create a longer
MMAC island in this code shape.  The next 60% MMAC-active attempt needs a
topology/page-lifetime redesign, or a focused Mq64 dV seed/layout probe before
revisiting this helper.

## 2026-07-03 Legacy W16 Split-Producer Probe

Status: `REJECT_RUNTIME_HANG`

The repository still contains an older 16-wave dKV path,
`fa3_bwd_dkv_mmac_kernel`, where waves0-3 publish Q/K, waves12-15 publish
dO/V, and waves4-11 consume.  It resembles the desired FWD-style role split,
so it was run as a probe before changing the canonical W12 path.

Evidence:

- static/source gate PASS
- symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=78`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- H1/S128 causal PMD path:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_013236`
- run emitted `read vgpr112 before writing`
- run did not complete after more than `18B` simulated ticks and was killed

Conclusion:

Do not directly promote or resurrect the legacy W16 split-producer path.  A
future 16-wave/FWD-style dKV design still looks architecturally plausible, but
it needs a fresh barrier ownership protocol and H1/S128 correctness evidence
before performance work.

## 2026-07-02 Sidecar Lane-Broadcast Negative

Status: `REJECT_PERF_STATS_ONLY`

A workbook-first opt-in path tested whether sidecar global-read latency could
be reduced by loading each q-row sidecar triplet only on `lane_n == 0` and
broadcasting it with `__shfl`.

Verified evidence:

- Static/source gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=82`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_110454`.
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_110521`.
- Same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_110630`.
- Sidecar broadcast H1/S1024:
  `kernel_ticks=86765770`, MMAC active `15.8550%`, coissue `40501/30454`.
- Same-build baseline H1/S1024:
  `kernel_ticks=71209320`, MMAC active `21.5636%`, coissue `29217/22022`.

Conclusion:

The code was removed.  `__shfl`/bpermute broadcast is not a good sidecar fix in
the current dKV conveyor: it preserves correctness but greatly increases the
effective active-time cost and drops MMAC active share.  Future sidecar work
should change representation or page ownership, not broadcast each row inside
the consumer wave.

## 2026-07-02 Early RawUsed Release

Status: `ACCEPT_MICRO_OBSERVE_PIPELINE`

The clean repo now has an opt-in early page-release W12 path:

- API path: `kDkvPathWaspDkvMmac12WaveEarlyRelease`
- standalone flag: `--dkv-mmac12-early-release-check=1`
- script flag:
  `WAVES=12 EARLY_RELEASE=1 scripts/run_dkv_mmac_correctness.sh`
- kernel symbol: `fa3_bwd_dkv_mmac12_early_release_kernel`
- change: consumer releases RawUsed after high source operands are read and
  `wait_lgkm(0)` has passed, before the high dV/dK MMAC island

Verified evidence:

- Static/source gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_103428`.
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_103518`.
- Same-build stats-only delta versus W12 baseline:
  `kernel_ticks=70869890` versus `70883085`, MMAC active avg
  `21.9267%` versus `21.7746%`, coissue `31524/20411` versus
  `29244/21070`.
- Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_104215_clean_w12_early_release_h1s1024_sqc7_fullperf`.
- Full perf/xcu evidence:
  `kernel_ticks=70507255`, whole-dispatch MMAC active share `21.7393%`,
  `ldsBankConflict=0`; mid-loop producer bubble improves from `92.06%` to
  `90.45%`, but early-window SIMD MMAC slightly worsens and tail remains
  `99%+` bubble.

Conclusion:

Keep this as an opt-in micro path and evidence that page lifetime matters.  It
does not solve the main 60% MMAC-active gap.  The next optimization should
change page ownership/topology or reduce sidecar/global-read latency instead of
only moving consumer instructions around.

## 2026-07-02 Source-Score Layout Probe

Status: `REJECT_LAYOUT_PROBE`

A workbook-first focused probe tested whether existing `Q^T/dO^T`
source-layout LDS pages can replace raw `Q/dO` pages for score/dP:

- temporary source-score path was implemented as an opt-in diagnostic only
- score/dP read operands from `QtBase/DoutTBase`
- K/V resident pages, packed global sidecar, dV/dK read4x2, zero-seed
  accumulation, store ownership, and ABarrier protocol were unchanged

Static/resource evidence:

- remote build PASS
- symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- branch pressure: consumer `146/160`

Correctness evidence:

- H1/S128 causal PMD run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_092729`
- result:
  `dk_max_abs=0.000713147`, `dk_rel_l2=1.5564`,
  `dv_max_abs=0.000420369`, `dv_rel_l2=0.00432992`, `pass=0`

Conclusion:

Do not remove raw `Q/dO` pages for score/dP based on the current
source-layout ABI.  `Q^T/dO^T` pages remain valid for the existing dV/dK path,
but they are not a drop-in replacement for raw score/dP operands with the
current `ds_read_matrix` mapping.  The temporary code was removed; revisit only
with a smaller fragment-layout instruction probe.

## 2026-07-02 Causal Whole-Tile Skip

Status: `REJECT_PERF_H1S1024`

The clean repo now has an opt-in W12 causal whole-tile skip path:

- API path: `kDkvPathWaspDkvMmac12WaveCausalSkip`
- standalone flag: `--dkv-mmac12-causal-skip-check=1`
- script flag:
  `WAVES=12 CAUSAL_SKIP=1 scripts/run_dkv_mmac_correctness.sh`
- kernel symbol: `fa3_bwd_dkv_mmac12_causal_skip_kernel`
- skip rule:
  `causal && q_tile * Mq + Mq - 1 < k_base`
- packet page rule:
  `packet_idx` advances only for non-skipped q tiles, keeping producer and
  consumer double-buffer ownership aligned
- partial causal tiles are still handled by the existing softmax/dS mask path

Verified evidence:

- Static/source gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=88`, `vgpr_count=144`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Branch-local consumer pressure: `194/208`.
- H1/S128 noncausal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081433`.
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081455`.
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081517`.
- H1/S1024 stats:
  `kernel_ticks=72881900`, MMAC active share `16.7128%`, VOP share
  `29.4950%`, `MMOP=73728`, coissue `14760/11291`,
  `ldsBankConflict=0`.
- Same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_081604`
  with `kernel_ticks=71006845`, MMAC active share `21.6777%`,
  `MMOP=131072`, coissue `30195/21487`, `ldsBankConflict=0`.

Conclusion:

Do not promote this path.  It proves the whole-tile causal skip can be made
correct and resource-clean, but reducing MMOP alone makes H1/S1024 thinner and
more tail-limited.  The next optimization should improve the conveyor and MMAC
active share, not simply delete upper-triangle work in the current topology.

## 2026-07-02 Mq64 Semantic-Page Conveyor

Status: `REJECT_PERF_STATS_ONLY`

The clean repo now has an opt-in Mq64 semantic-page dKV path:

- API path: `kDkvPathWaspDkvMmac12WaveMq64Semantic`
- standalone flag: `--dkv-mmac12-mq64-semantic-check=1`
- script flag: `WAVES=12 MQ64_SEMANTIC=1 scripts/run_dkv_mmac_correctness.sh`
- kernel symbol: `fa3_bwd_dkv_mmac12_mq64_semantic_kernel`
- LDS layout: K/V resident 64KB plus two 32KB semantic pages
- page meaning:
  - raw generation: matrix0=`Q`, matrix1=`dO`
  - source generation: matrix0=`Q^T`, matrix1=`dO^T`
- main matrix path remains `matrix_load ... bps lds`, `ds_read_matrix`, and
  `v_mmac_*lit`; no raw LDS transpose writer was added.

Verified evidence:

- Static/source gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=90`, `vgpr_count=144`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- Branch-local compiler pressure:
  producer `1/16`, consumer `180/208`.
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_074704`.
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_074725`.
- H1/S1024 stats:
  `simTicks=76933675`, `kernel_ticks=73320065`, MMAC active avg `21.7509%`,
  VOP active avg `22.6370%`, coissue `23374/16882`, `MMOP=131072`,
  `ldsBankConflict=0`.
- Same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_075046`
  with `kernel_ticks=70974995`, MMAC active avg `21.7125%`,
  coissue `28477/21603`, `ldsBankConflict=0`.

Conclusion:

Do not promote this path.  It proves Mq64 semantic page reuse can be made
correct and resource-clean, but adding a second raw/source ABarrier generation
regresses ticks.  The next path must reduce barrier/control turns, not add
another page lifecycle.

## 2026-07-01 Clean dKV WASP Probe

Status: `BRINGUP_ONLY`

The clean repo owns a real FA3 BWD dKV kernel shape:

- thin C ABI in `include/shaobo_fa3_api.h`
- standalone HIP launcher in `src/dkv_kernel.cpp`
- 16-wave WDRA kernel with four explicit role branches
- branch-local producer/consumer VGPR windows
- ABarrier ownership ledger
- instruction helpers in `include/shaobo_instr.h`
- producer0 publishes Q + K via `matrix_load_32x32_b16 ... bps lds`
- producer1 publishes dO + V via `matrix_load_32x32_b16 ... bps lds`
- consumer groups wait Q, dO, K, V ownership tokens and run a score+dP
  `ds_read_matrix_trans_format` plus `v_mmac_*lit` probe
- PMD smoke wrapper treats model abort text as failure even if `run.py` exits 0

This is not a dKV performance candidate.  It proves the clean repo can emit and
run the WASP packet + matrixized score/dP path without scratch, spill, or LDS
bank conflict.  It does not compute dV/dK yet.

Verified evidence after the naming cleanup:

- Remote repo: `/zys/shaobo/fa3_bwd_wasp_clean`
- PMD smoke:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_211544`
- Metadata gate:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `sgpr_count=30`, `vgpr_count=84`
- Runtime signal:
  `fa3_bwd_dkv_probe status=success B=1 H=1 S=1024 D=128`
- Stats:
  `simTicks=7232680`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7232680`, `MMOP=2048`, `ldsBankConflict=0`,
  `mmop_active_share=6.4892%`
- PMD warning to investigate:
  `warn: read vgpr184 before writing`

Next implementation hypothesis:

Add the real q-loop, sidecar loading, softmax+dS, then dV/dK accumulation and
stores while preserving the clean file structure and evidence chain.

## 2026-07-01 MLS Publication Wait Cleanup

Status: `OBSERVE`

The producer packet publishers no longer execute `wait_lgkm(0)` immediately
after `matrix_load_32x32_b16 ... bps lds`.  The intended protocol is:

```text
producer matrix_load -> ABarrier Filled
consumer wait Filled -> ds_read_matrix -> wait before first MMAC use
```

Verified evidence:

- Remote repo: `/zys/shaobo/fa3_bwd_wasp_clean`
- PMD smoke:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_212516`
- Metadata gate:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `sgpr_count=30`, `vgpr_count=84`
- Runtime signal:
  `fa3_bwd_dkv_probe status=success B=1 H=1 S=1024 D=128`
- Stats:
  `simTicks=7250425`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7250425`, `MMOP=2048`, `ldsBankConflict=0`,
  `mmop_active_share=6.4353%`

Conclusion:

This validates the protocol on the current probe but does not prove a speedup.
The single-packet probe has little overlap opportunity, so performance should
be judged again after the real multi-packet q-loop exists.

## 2026-07-01 Coarse Packet Barrier Cut

Status: `OBSERVE_PROTOCOL`

The probe barrier ledger was collapsed from four fragment-level packet tokens
to two producer packet tokens:

```text
producer A: Q + K  -> PacketAFilled / PacketAUsed
producer B: dO + V -> PacketBFilled / PacketBUsed
consumer: wait PacketA + PacketB -> score+dP MMAC -> arrive PacketA/B Used
```

This directly targets the BWD SQTT symptom where the old path produced large
`abarrier -> abarrier` chains.  The cut is intentionally protocol-only: no
math ownership, tile shape, MLS/BPS, `ds_read_matrix`, or MMAC path was changed.

Evidence:

- `python3 scripts/check_dkv_kernel_gate.py` PASS.
- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=30`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD smoke PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_232821`
- Stats:
  `simTicks=7177170`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7177170`, `ldsBankConflict=0`,
  per-active-CU `MMOP=256`.
- Short TT/Perf capture completed:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_233020`
- XCU first-pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2010209_fa3_bwd_wasp_clean_20260701_233432`
- XCU top-bubble window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/packet_barrier_probe_window_20260701_233020`
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_233020_clean_packet_barrier_probe`
- XCU observation:
  dispatch duration `7852`, inst issues `20648`, MMAC share `18.05%`,
  SALU32 share `35.95%`, top bubble remains `abarrier -> abarrier` at
  `23.50%` with max duration `3696` cycles.  The selected top-bubble window is
  a 100% bubble window with no issued instructions.

Remaining before promotion:

- treat this as protocol evidence only; the one-shot probe still has unavoidable
  producer/consumer/all-done idle.  Promotion requires a multi-packet q-loop
  where producer work can continue during consumer MMAC/VALU islands.

## 2026-07-01 Stream q-loop Probe

Status: `OBSERVE_PIPELINE`

The single-packet probe was extended into a fixed `S=1024`, `Mq=32` q-loop:

```text
producer A: K resident once, then double-buffer Q raw pages
producer B: V resident once, then double-buffer dO raw pages
consumers: wait resident once, then stream 32 Q/dO pages through score+dP MMAC
```

This is still not full dKV.  It intentionally omits softmax/dS, dV/dK
accumulation, and stores so the packet conveyor can be measured without the old
phase-stack noise.

Evidence:

- Local source gate PASS:
  `python3 scripts/check_dkv_kernel_gate.py`
- Remote build/static/symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=38`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD stats smoke:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260701_235037`
- Stats:
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
- XCU observation:
  dispatch duration `54540`, inst issues `352336`, waves `128`, MMAC share
  `31.37%`, SALU32 `21.56%`, `lds_matrix` `15.68%`, `immed` `12.00%`.
  Top bubble changed from `abarrier -> abarrier` to
  `abarrier -> salu_32` at `43.80%`; `lds_matrix -> immed` remains visible at
  `6.74%`.

Conclusion:

The multi-packet conveyor is a better diagnostic shape than the one-shot probe:
MMAC active evidence improved and the dominant bubble changed.  The remaining
gap is not "missing MMAC"; it is control/barrier overhead around page ownership
and a fragmented `ds_read_matrix -> first MMAC` path.  The next code cut should
keep this q-loop but batch operand reads into longer FWD-style MMAC islands, then
add softmax/dS and dV/dK only after the read-to-use gap is measurable.

## 2026-07-02 dK/dV Reference Correctness

Status: `REFERENCE_CORRECTNESS_PASS`

A correctness-first dKV reference path now exists beside the WASP probe:

```text
P = softmax(QK * scale)
delta = sum(dO * (P @ V))
dP = dO @ V^T
dV = P^T @ dO
dK = (P * (dP - delta) * scale)^T @ Q
```

Scope:

- standalone flag: `--check=1`
- API path: `params.dkv_path = kDkvPathReferenceCorrectness`
- output dtype for this bring-up path: float `dK/dV`
- layout: BHSD, fp16 inputs, `B=1,H=1,S=128,D=128` smoke verified
- purpose: correctness oracle and output-ownership lock, not performance

Evidence:

- Remote build PASS with asm.
- Static gate PASS.
- Probe symbol metadata still PASS:
  `private_segment_fixed_size=0`, `sgpr_count=38`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_correctness_20260702_003625`
- Numeric result:
  `dk_max_abs=1.16415e-10`, `dk_rel_l2=3.53805e-07`,
  `dv_max_abs=3.72529e-09`, `dv_rel_l2=1.7753e-08`, `bad=0`, `pass=1`.
- WASP probe regression also PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260702_003118`

Boundary:

This does not mean the optimized WASP path computes/stores dK/dV yet.  It means
the repo now has a verified dK/dV formula path and a reusable golden harness.
The next step is to migrate `softmax/dS`, then dV MMAC, then dK MMAC and stores
into the WASP path while comparing against this reference.

## 2026-07-02 WASP Softmax/dS Sidecar

Status: `SIDECAR_CORRECTNESS_PASS`

The WASP probe now has an opt-in softmax/dS sidecar path:

```text
params.dkv_path = kDkvPathWaspSoftmaxDsSidecar
standalone: --probe-check=1
```

Scope:

- Keeps the existing MLS/BPS + `ds_read_matrix` + MMAC score/dP q-loop.
- Adds one scalar `(P,dS)` diagnostic per consumer wave inside the consumer
  role after the q-loop.
- Writes diagnostics to workspace slots:
  - `0..7`: score+dP probe diag
  - `8..15`: `P` sidecar
  - `16..23`: `dS` sidecar
- Compares `P/dS` against host CPU golden.
- This is correctness-only; the sidecar is scalar and intentionally not a perf
  candidate.

Evidence:

- Remote build PASS with asm.
- Static gate PASS.
- Probe symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=80`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD sidecar correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_sidecar_correctness_20260702_005025`
- Numeric result:
  `p_max_abs=0`, `p_rel_l2=0`, `ds_max_abs=2.18279e-11`,
  `ds_rel_l2=1.10949e-07`, `bad=0`, `pass=1`.
- Stats:
  `simTicks=1313879840`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=1313879840`, `ldsBankConflict=0`.
- Default WASP probe regression PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260702_005132`
  with `simTicks=28741440`, `ldsBankConflict=0`.

Important fix:

The first sidecar attempt produced zero diagnostics.  Moving `lane =
threadIdx.x % 64` into the consumer role after `s_set_vgpr_size` fixed it.
This reinforces the WDRA rule: values used for branch-local store predicates
should be initialized inside the role window that uses them.

Next:

Move from scalar sidecar to fragment-local `P/dS` in the consumer mainloop.
Only after that should dV and dK MMAC accumulation be connected to the store
epilogue.

## 2026-07-02 FWD-style dKV Workbook Gate

Status: `DESIGN_GATE_UPDATED`

The shared workbook is now the source of truth for the next dKV cut:

```text
/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx
```

Updated sheets:

- `FWD目标`
- `算法DAG`
- `资源预算`
- `流水设计`
- `指标门禁`
- `实验记录`

The design target is a FWD-style dKV path with no duplicate score/dP, explicit
output ownership, LDS budget `98816 B` plus about `28 KB` slack, and a T0-T5
expected conveyor that alternates score/dP MMAC, softmax/dS VALU, dV MMAC, and
dK MMAC across the two consumer groups.  The primary performance target remains
MMAC active share `>=60%`; ticks are supporting evidence, not the first tuning
axis for the small diagnostic shape.

Verification:

- workbook exported successfully
- backup written:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.codex_backup_20260702_fwdstyle_goal.xlsx`
- inspect file reports no formula-error matches:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx.inspect.ndjson`
- preview rendered:
  `/tmp/shaobo_wb_update_20260702/previews/流水设计.png`

Next:

Implement from the workbook, not from the old phase stack: keep the current
sidecar as oracle, then move fragment-local `P/dS`, dV MMAC, dK MMAC, and the
store epilogue into the clean WASP path.

## 2026-07-02 WASP Fragment Sidecar

Status: `FRAGMENT_SIDECAR_PASS`

The clean WASP path now has a fragment-level P/dS validation cut:

```text
params.dkv_path = kDkvPathWaspFragmentSidecar
standalone: --fragment-check=1
```

Implemented:

- Added `kDkvPathWaspFragmentSidecar = 3`.
- Reduced the main LDS allocation from full 128KB to actual `Layout::kBytes`,
  leaving room for shared sidecar pages.
- Added double-buffered sidecar rows:
  `max_log2`, `inv_sum`, and `delta`.
- Producer A publishes sidecar rows with the Q raw page and the existing raw
  page ABarrier ownership.
- Consumer score/dP now keeps four fragment accumulators instead of one
  checksum accumulator.
- The first MMAC for each score/dP fragment uses a FWD-style `mmac_zeros`
  accumulator seed; later MMACs accumulate into the fragment.
- Consumer computes fragment-local `P` and `dS` from the score/dP fragments and
  sidecar rows, then writes a diagnostic pair for correctness.

Evidence:

- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=88`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD fragment sidecar correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_fragment_sidecar_correctness_20260702_012111`
- Numeric result:
  `p_max_abs=6.34603e-06`, `p_rel_l2=6.92507e-06`,
  `ds_max_abs=2.66919e-08`, `ds_rel_l2=0.000359523`,
  `bad=0`, `pass=1`.
- Stats:
  `simTicks=10138765`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=10138765`, `MMOP=512` on active CU,
  `ldsBankConflict=0`, coissue `45/12`.
- Regression smoke PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_score_dp_probe_20260702_012136`
  with `simTicks=27633970`, `ldsBankConflict=0`.
- Scalar sidecar regression PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_sidecar_correctness_20260702_012150`.
- Workbook experiment ledger updated:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`.

Boundary:

This still does not store dK/dV.  It proves the FWD-style score/dP fragment
layout can feed sidecar-based fragment P/dS under PMD with no spill, no scratch,
and no LDS bank conflict.  Next, wire `p_frag` into dV MMAC and `ds_frag` into
dK MMAC.

## 2026-07-02 Full dK/dV MMAC Baseline

Status: `FULL_DKV_CORRECTNESS_PASS_BASELINE`

The clean path now computes and stores both dK and dV:

```text
params.dkv_path = kDkvPathWaspDkvMmac
standalone: --dkv-mmac-check=1
script: scripts/run_dkv_mmac_correctness.sh
```

Current design:

- 16-wave CTA, four explicit roles:
  waves0-3 producer Q/K/Q^T, waves4-7 consumer group0, waves8-11 consumer
  group1, waves12-15 producer dO/V/dO^T.
- Each consumer wave owns one `Nk=16,D=128` output slice and accumulates dV and
  dK across the q-loop.
- `score`, `dP`, `dV`, and `dK` each issue 16 MMAC per consumer wave per
  q tile; total 64 MMAC per consumer wave per q tile.
- No duplicate score/dP across D halves.
- LDS is fully budgeted at 128KB:
  resident K/V 64KB, raw Q/dO double buffer 32KB, source-layout Q^T/dO^T
  double buffer 32KB.
- Packed sidecar `[max_log2, inv_sum, delta]` comes from global memory through
  one pointer.  This removed SGPR spill but is now a visible flat-read/VALU
  hotspot.

Evidence:

- Remote build/static/symbol metadata PASS:
  `private=0`, `sgpr_count=76`, `vgpr_count=84`, no SGPR/VGPR spill.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_022300`
  with `dk_max_abs=1.49356e-07`, `dk_rel_l2=0.0025563`,
  `dv_max_abs=2.87902e-05`, `dv_rel_l2=0.000337571`, `pass=1`.
- H1/S1024 stats:
  `simTicks=86904090`, `kernel_ticks=83290480`,
  MMAC active share avg/min/max `18.7606%/16.4951%/21.4893%`,
  coissue `36070/20048`, `ldsBankConflict=0`.
- Causal=false diagnostic:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_022345`
  did not improve pipeline: MMAC active avg `15.7263%`,
  `kernel_ticks=87759035`.  It is diagnostic-only because dK relative L2 did
  not meet the formal gate despite tiny absolute error.
- Split raw/source ownership was tested and rejected:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_023309`
  regressed to `kernel_ticks=86912280` and MMAC active avg `18.1123%`.

Next:

Keep the coarse-packet baseline.  Use xcu on the accepted perf capture to
target the next bottleneck: packed-sidecar flat reads, necessary softmax/dS
VALU, and remaining ABarrier/control bubbles.  Do not reintroduce source
ownership tokens without a larger resource/pipeline redesign.

## 2026-07-02 12-Wave Single Producer Candidate

Status: `ACCEPT_CANDIDATE_BUT_LOW_ACTIVE`

The clean repo now has an opt-in 12-wave dKV kernel:

```text
params.dkv_path = kDkvPathWaspDkvMmac12Wave
standalone: --dkv-mmac12-check=1
script: WAVES=12 scripts/run_dkv_mmac_correctness.sh
```

Current design:

- 12-wave CTA, 768 threads, explicit WDRA branch windows.
- waves0-3 are the only producer group.  They publish resident K/V once and
  stream Q/dO/Q^T/dO^T packets for both consumer groups.
- waves4-7 and waves8-11 are heavy consumers.  Each group owns a different
  `Nk=16,D=128` output slice and computes score, dP, dV, and dK.
- The math, source-layout ABI, and coarse packet ownership are the same as the
  accepted 16-wave baseline.

Evidence:

- Build/static/symbol metadata PASS:
  `private=0`, `sgpr_count=82`, `vgpr_count=112`, no spill.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_024903`.
- PMD stats:
  `kernel_ticks=78751400`, MMAC active avg `20.2578%`,
  coissue `23301/12740`, `ldsBankConflict=0`.
- Compared to 16-wave baseline:
  ticks improved `5.45%`, active share improved `+1.50` percentage points.
- Full perf and xcu output:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_025324_clean_w12_dkv_mmac12_h1s1024_sqc7`.

Observed bottleneck:

xcu confirms the main remaining debt is not missing MMAC.  The hot mix is
`MMAC 22.67%`, `SALU32 20.30%`, `VALU32 17.49%`, `VALU64 11.80%`, and
`LDS_MATRIX 8.50%`.  The dominant issue bubble is still
`abarrier -> salu_32` (`38.85%`), with top `abarrier -> immed` gaps around
`15.8k` cycles.  Producer wavefronts are less numerous than in the 16-wave
kernel but still thin (`~2.5k` instructions versus `13k-16k` consumer
instructions).

Next:

Use W12 as the next evidence baseline, then redesign the barrier/pipeline
protocol toward the FWD pattern: fewer packet ownership turns, longer
continuous MMAC islands, and useful recurring producer work.  Do not treat W12
as the final FWD-style kernel; it only removes one visible source of dilution.

## 2026-07-02 W12 Sidecar Address Hoist

Status: `ACCEPT_MICRO`

The current source includes a small W12-compatible cleanup in
`softmax_ds_owner16_from_global_sidecar`: the q-tile sidecar base pointer is
computed once, and the inner vector loop indexes it by local row.  The same
patch hoists `krow` out of the `m_idx` loop in both sidecar helpers.

Evidence:

- Static/symbol metadata PASS for `fa3_bwd_dkv_mmac12_kernel`:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no spill.
- H1/S1024 causal=true correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_031634`.
- Full-perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_031634_clean_w12_sidecar_addr_h1s1024_sqc7`.
- Same full-perf baseline comparison:
  W12 `kernel_ticks=78625365`, active `19.9522%`;
  sidecar hoist `kernel_ticks=75964525`, active `20.3523%`.
- xcu still shows the dominant issue gap as `abarrier -> salu_32`
  (`38.33%`), so this is not a pipeline solution.

Next:

Treat the hoist as the current W12 micro-clean baseline.  The next real
FWD-style step must redesign the ABarrier/control protocol and consumer
compute islands; raising MMAC active from `~20%` to `60%` will not come from
more sidecar address cleanups.

## 2026-07-02 W12 Late-Source Conveyor Negative

Status: `REJECT_PERF_REVERT_CODE`

The late-source conveyor tried to publish raw `Q/dO` first, let consumers run
score/dP, then publish source-layout `Q^T/dO^T` into the same raw page during
consumer softmax/dS.  The path was resource-clean and correct, but slower:

- metadata: `private=0`, `sgpr=86`, `vgpr=112`, no spill
- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_033544`
- H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_033608`
- H1/S1024 result:
  `kernel_ticks=81165175`, MMAC active avg `19.1939%`
- current W12 sidecar baseline:
  `kernel_ticks=75964525`, MMAC active avg `20.3523%`

Conclusion:

Late-source added useful producer work but also added another exposed page
epoch.  The ownership latency outweighed the intended overlap.  The opt-in code
was reverted; keep only the result in the log/workbook.

## 2026-07-02 W12 Producer Early Exit Negative

Status: `REJECT_RUNTIME_PANIC_REVERT_CODE`

The producer early-exit attempt targeted the long thin producer wave tail seen
in xcu.  It changed only the tail protocol: producer waves returned after
`producer_all_loop`, while the eight consumer waves owned `AllDone` cleanup.

Results:

- Producer VGPR `80` failed compile due WDRA branch-average granularity.
- Producer VGPR `76` compiled and passed metadata:
  `private=0`, `sgpr=84`, `vgpr=132`, no spill.
- H1/S128 PMD aborted:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_034614`.
- Panic:
  `vgpr81 is not init or has been freed` during MMAC.

Conclusion:

This topology is not safe to keep.  Producer waves should not return early
while consumer waves continue MMAC in the same WDRA workgroup unless a focused
probe proves a supported cleanup protocol.  The opt-in code was reverted.

## 2026-07-02 W12 Full-Valid Softmax Fast Path Negative

Status: `REJECT_CORRECTNESS_REVERT_CODE`

The full-valid fast path tried to remove per-element `valid_pair` and causal
mask checks from `softmax_ds_owner16_from_global_sidecar` for tiles where every
owner16 pair should be valid.  Two variants were tested:

- early-return branch after filling `p_frag/ds_frag`
- structured `if/else` branch with the original slow path in the `else`

Both variants passed static metadata for `fa3_bwd_dkv_mmac12_kernel`
(`private=0`, `sgpr=80`, `vgpr=112`, no spill) but failed H1/S128 causal
correctness:

- `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_035820`
- `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_035954`

Representative result: `dK rel_l2=0.000361379`, but
`dV rel_l2=14.7566`.  The source is reverted to the W12 sidecar-address
baseline.

Rule: do not remove causal/predicate work from the global-sidecar owner16
helper by local branching alone.  Prove any future mask fast path with a
focused owner16 sidecar probe, because this branch specifically corrupts the
`P`/dV path while leaving dK close.

## 2026-07-02 W12 dV/dK Read-All MMAC Island

Status: `ACCEPT_MICRO_CURRENT_BASELINE`

The current source keeps the 12-wave single-producer topology and W12
sidecar-address cleanup, then changes only the dV/dK consumer island:

```text
old: repeat 4x { ds_read_matrix dO^T/Q^T pair -> wait -> small MMAC island }
new: issue all 8 dO^T/Q^T ds_read_matrix pairs -> wait once -> longer MMAC island
```

The code uses explicit `dout_t0..7` and `q_t0..7` registers plus a small
`dv_dk_mmac_one_out` helper, avoiding private memory arrays.

Evidence:

- Remote source: `/zys/shaobo/fa3_bwd_wasp_clean`
- Static metadata:
  `private_segment_fixed_size=0`, `sgpr_count=88`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_041233`
- H1/S1024 correctness/perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_041545`
- Perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_041545_clean_w12_dvdk_readall_h1s1024_sqc7`
- Metrics versus W12 sidecar-address:
  `kernel_ticks=72499700` versus `75964525`,
  MMAC active avg `21.3054%` versus `20.3523%`,
  coissue `30904/22164`, `ldsBankConflict=0`
- xcu detail:
  `MMAC=23.64%`, `lds_matrix -> immed=5.53%` versus previous `9.38%`,
  but top bubble remains `abarrier -> salu_32=38.64%`

Conclusion:

This is the current source baseline because it is correct, resource-clean, and
improves both ticks and MMAC active.  It is still a micro optimization.  The
remaining gap to the 60% MMAC active target is now clearly dominated by
ABarrier/control and phase alignment, not by missing dV/dK MMAC or LDS bank
conflict.

## 2026-07-02 W12 Pre-Softmax dV/dK Read Negative

Status: `REJECT_RESOURCE_REVERT_CODE`

The attempted schedule moved all dV/dK `dO^T/Q^T` `ds_read_matrix` operations
before softmax/dS so their LDS latency could overlap with VALU work:

```text
score/dP -> dO^T/Q^T reads -> softmax/dS -> wait -> dV/dK MMAC
```

The code built, but the metadata gate failed before PMD:

- `private_segment_fixed_size=24`
- `vgpr_spill_count=10`
- `sgpr_count=84`
- `vgpr_count=112`

Conclusion:

The schedule expands live ranges too much in the current helper structure.  The
source is reverted to the W12 dV/dK read-all baseline.  Future read-early
experiments must split the source operands into smaller groups or first shrink
the softmax/dS live range.

## 2026-07-02 W12 dV/dK 4+4 Read-Early Island

Status: `ACCEPT_MICRO_CURRENT_BASELINE`

The current source uses a bounded live-range retry of the rejected pre-softmax
read-all idea:

```text
score/dP -> read low dO^T/Q^T group -> softmax/dS
         -> wait low -> read high group -> MMAC low -> wait high -> MMAC high
```

Evidence:

- Static metadata:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_043459`
- H1/S1024 correctness/perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_043641`
- Perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_043641_clean_w12_dvdk_read4x2_h1s1024_sqc7`
- Metrics versus W12 read-all:
  `kernel_ticks=71508255` versus `72499700`,
  MMAC active avg `21.5678%` versus `21.3054%`,
  `lds_matrix -> immed=2.46%` versus `5.53%`,
  `ldsBankConflict=0`

Conclusion:

This is now the current source baseline.  It proves bounded read-early/wait-late
helps hide source operand LDS latency, but it also proves that the remaining
large gap to 60% MMAC active is not dV/dK read granularity.  The next target is
ABarrier/control serialization and consumer phase alignment.

ValuExec0 turnstile retry:

```text
group1 softmax arrive -> group0 wait before softmax -> both continue
```

The retry was correctness-clean and resource-clean, but performance regressed:

- H1/S1024 `kernel_ticks=73908835` versus read4x2 `71508255`
- MMAC active avg `20.8817%` versus read4x2 `21.5678%`
- coissue `29516/19583` versus read4x2 `30929/20971`

Decision: `REJECT_PERF`; the source was reverted to the read4x2 baseline.
Interpretation: explicit consumer phase gating adds control/wait cost unless it
also moves real independent work into the peer's MMAC window.  Future phase
alignment needs useful producer/helper work or a different ownership topology,
not another pure turnstile.

Rejected raw/source ownership split:

```text
producer: Raw(Q,dO) -> RawFilled -> Source(Q^T,dO^T) -> SourceFilled
consumer: RawFilled -> score/dP -> RawUsed -> SourceFilled/read4x2
          -> softmax/dS -> dV/dK -> SourceUsed
```

This targeted the current over-synchronization where raw score/dP is forced to
wait for source-layout operands that are only used later.  It did not change
the algorithm, tile, LDS bytes, output ownership, or matrixized main path.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_045658`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_045719`
- Static metadata:
  `private=0`, `sgpr_count=86`, `vgpr_count=112`, no SGPR/VGPR spill
- H1/S1024 `kernel_ticks=75607805` versus read4x2 `71508255`
- MMAC active avg `20.5505%` versus read4x2 `21.5678%`
- coissue `31554/23826` versus read4x2 `30929/20971`

Decision: `REJECT_PERF`; source reverted to read4x2.  The more precise packet
ownership did not pay for its extra ABarrier/control cost.

Current source delta:

`consumer_dkv_mmac_loop` is now specialized by template consumer group:

```text
consumer_dkv_mmac_loop<Tile, Bar, 0>(...)
consumer_dkv_mmac_loop<Tile, Bar, 1>(...)
```

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_050701`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_050727`
- Static metadata unchanged for the W12 kernel:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill
- H1/S1024 `kernel_ticks=71412705` versus read4x2 `71508255`
- MMAC active avg `21.5708%` versus read4x2 `21.5678%`

Decision: `ACCEPT_MICRO`.  This is the current source baseline, but it is only
a codegen/control cleanup; it does not materially change the pipeline gap to
60% MMAC active.

Current source delta:

dV/dK MMAC helpers now skip volatile zero seed for non-first q tiles:

```text
if constexpr (FirstQTile) {
    ins::zero_f16x8(zero_f16);
}
```

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_051325`
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_051343`
- Full perf/xcu archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_051513_clean_w12_zero_seed_h1s1024_sqc7`
- Static metadata unchanged:
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no SGPR/VGPR spill
- H1/S1024 stats-only `kernel_ticks=70604625` versus template baseline
  `71412705`
- MMAC active avg `21.7988%` versus `21.5708%`
- xcu full perf: `MMAC=23.68%`, `valu_32` hits `151648`,
  `abarrier -> salu_32=39.07%`, `flat_rd -> immed=15.03%`

Decision: `ACCEPT_MICRO`.  This is the current source baseline.  Remaining
dominant debts are barrier/control and sidecar global-read latency.

Rejected sidecar prefetch:

- candidate: prefetch all owner16 sidecar triplets into registers before
  `RawFilled`/score-dP, so softmax/dS would not issue `packed_sidecar` global
  reads at the use point
- evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_052854`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_052900`
- archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260702_052900_clean_w12_sidecar_prefetch_reject_h1s1024_sqc7`
- result: correctness PASS and static metadata clean, but consumer branch
  pressure rose to `158/160`, H1/S1024 `kernel_ticks` regressed to
  `75394410`, and MMAC active avg fell to `18.4182%`
- decision: `REJECT_PERF_STATS_ONLY`; code reverted

Rule: sidecar latency should still be addressed, but not by carrying all
24 sidecar floats across the score/dP MMAC island.  Future sidecar work must
shorten live range or reduce representation before adding registers.

Noncausal diagnostic boundary:

- after rebuilding the reverted zero-seed baseline, metadata returned to
  `private=0`, `sgpr_count=84`, `vgpr_count=112`, no spill
- `CAUSAL=0`, H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_053747`
- `CAUSAL=0`, H1/S1024 correctness FAIL:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_053811`
- failure signal:
  `dk_rel_l2=0.0199722`, `dv_rel_l2=0.002617`, `bad=0`, `pass=0`

Rule: do not use noncausal H1/S1024 performance counters to guide MMAC active
tuning until the noncausal correctness/tolerance boundary is resolved.  Continue
the primary mainline with `causal=true`.

Rejected sidecar pair-prefetch:

- candidate: inside `softmax_ds_owner16_from_global_sidecar`, group sidecar
  reads two q rows at a time instead of prefetching all 8 rows across score/dP
- static result: metadata clean, `private=0`, `sgpr_count=82`,
  `vgpr_count=112`, no spill; consumer branch pressure `144/160`
- correctness result: H1/S128 failed numerical comparison before stats/perf
  promotion
- failing run:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_055216`
- failure signal:
  `dk_rel_l2=8244.1`, `dv_rel_l2=30.6025`, `pass=0`, plus PMD
  `read vgpr111 before writing`
- decision: `REJECT_CORRECTNESS`; code reverted and remote source restored to
  the zero-seed baseline

Rule: sidecar read batching now needs a focused sidecar-fragment probe before
touching the full dKV mainline again.  The current source baseline remains
zero-seed cleanup, not any sidecar prefetch variant.

Rejected Mq64 single-buffer structural probe:

- design goal: double each consumer island from 64 to 128 MMAC per q tile and
  halve q-loop barrier/control turns by moving from `Mq=32` to `Mq=64`
- LDS arithmetic: `K/V 64KB + raw Q/dO 32KB + source Q^T/dO^T 32KB = 128KB`
- resource stress result:
  - four-fragment Mq64 spilled badly
  - half-sequential Mq64 with consumer 208 VGPR removed VGPR spill but still
    had SGPR spill
  - S1024/causal specialization finally reached `private=0`, `sgpr=100`,
    `sgpr_spill=0`, `vgpr=144`, `vgpr_spill=0`
- correctness result:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_062758`
  reported status success but `pass=0`, `dk_rel_l2=5680.1`,
  `dv_rel_l2=20.0452`, with PMD `read vgpr268 before writing`
- decision: `REJECT_CORRECTNESS`; do not capture perf because MMAC active is
  meaningless without correctness

Rule: Mq64 is still a possible future structure, but only after a focused
layout/correctness probe proves the Mq64 raw/source sidecar, dV/dK accumulator,
and store path.  Do not reintroduce the exact-128KB full-kernel Mq64 path as a
performance candidate directly.

Post-revert baseline check:

- remote build/static gate PASS after removing the Mq64 implementation
- W12 metadata PASS:
  `private=0`, `sgpr=84`, `sgpr_spill=0`, `vgpr=112`, `vgpr_spill=0`
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_063709`

Mq64 seed-fix reattempt:

- workbook updated first with `Mq64 seed-fix reattempt`
- restored the opt-in W12 Mq64 path and fixed high D-block accumulator seeding:
  the high dV/dK accumulator group now follows the same first-q-tile
  `SeedAccumulator` decision as the low group
- build/static/metadata PASS:
  `private=0`, `sgpr=100`, `sgpr_spill=0`, `vgpr=144`, `vgpr_spill=0`,
  branch-local consumer pressure `171/208`
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_064930`
- numerical signal:
  `dk_rel_l2=0.0025563`, `dv_rel_l2=0.000337571`, `bad=0`, `pass=1`
- stats signal:
  `simTicks=83209490`, `kernel_ticks=79595880`, MMAC active avg `19.8279%`,
  coissue success/fail `13713/7221`, `ldsBankConflict=0`
- comparison:
  zero-seed W12 baseline remains better at `kernel_ticks=70604625` and
  MMAC active avg `21.7988%`

Decision: `OBSERVE_CORRECTNESS_REJECT_PERF`.  Keep the seed lesson and the
opt-in path only as a diagnostic/future Mq64 basis; do not promote this
single-buffer exact-128KB topology.  The next FWD-style design needs LDS slack
or real producer/consumer overlap, not only a larger per-q-tile MMAC island.

Raw-page sidecar overlay diagnostic:

- workbook updated first with `Raw-page sidecar overlay` design, then with
  `Raw-page sidecar overlay result`
- opt-in path:
  `kDkvPathWaspDkvMmac12WaveSidecarOverlay` /
  standalone `--dkv-mmac12-overlay-check=1`
- producer publishes matrix packets, then after raw matrix use is released
  overlays sidecar values into the raw Q LDS page; consumers wait for this
  second generation before softmax/dS
- build/static/metadata PASS:
  `private=0`, `sgpr=86`, `vgpr=112`, no spills, branch pressure
  `producer=9/16`, `consumer=146/160`
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_070805`
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_070829`
- H1/S1024 stats:
  `simTicks=76727560`, `kernel_ticks=73113950`, MMAC active avg `18.9185%`,
  coissue `39007/29043`, total MMOP `131072`, `ldsBankConflict=0`
- comparison:
  zero-seed W12 baseline remains better at `kernel_ticks=70604625` and
  MMAC active avg `21.7988%`

Decision: `REJECT_PERF_STATS_ONLY`.  This path may remain as an opt-in
diagnostic, but it is not a mainline optimization.  The result confirms that
sidecar/global-read latency is real, yet adding another RawFilled/RawUsed
generation makes the ABarrier/control path heavier than the load it removes.

Score/dP read2x brick diagnostic:

- workbook updated first with `Score/dP read2x brick` design, then with
  `Score/dP read2x brick result`
- opt-in path:
  `kDkvPathWaspDkvMmac12WaveScoreDpBrick` /
  standalone `--dkv-mmac12-score-brick-check=1`
- change: the score/dP island reads two D-block operand families before a
  single `wait_lgkm(0)`, then issues two score+dP MMAC groups
- unchanged: producer packet protocol, global sidecar softmax/dS, dV/dK
  read4x2, zero-seed accumulation, and store ownership
- build/static/metadata PASS:
  `private=0`, `sgpr=84`, `vgpr=112`, no spills, branch pressure
  `consumer=150/160`
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_072317`
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_072323`
- H1/S1024 stats:
  `simTicks=74415250`, `kernel_ticks=70801640`, MMAC active avg `21.5465%`,
  coissue `34543/22997`, total MMOP `131072`, `ldsBankConflict=0`
- comparison:
  zero-seed W12 baseline remains better at `kernel_ticks=70604625` and
  MMAC active avg `21.7988%`

Decision: `REJECT_PERF_STATS_ONLY`.  Larger local score/dP read bricks do not
move the kernel toward 60% MMAC active.  The next main design must change the
packet ownership/conveyor or producer/consumer role utility instead of adding
more local read scheduling variants.

Mixed score/dP brick diagnostic:

- workbook updated first with `Mixed score brick` design and result rows
- opt-in path:
  `kDkvPathWaspDkvMmac12WaveMixedScoreBrick` /
  standalone `--dkv-mmac12-mixed-score-brick-check=1`
- design: group0 keeps the baseline consumer schedule while group1 uses the
  existing score/dP read2x brick schedule; producer, LDS pages, ABarrier ledger,
  dV/dK read4x2, zero-seed accumulation, and output ownership are unchanged
- build/static/metadata PASS:
  `private=0`, `sgpr=84`, `vgpr=112`, no spill, branch pressure
  group0 `150/160`, group1 `158/160`
- H1/S128 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_084709`
- H1/S1024 causal correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_084734`
- H1/S1024 stats:
  `kernel_ticks=71663865`, MMAC active avg `21.1732%`, coissue
  `33504/24695`, total MMOP `131072`, `ldsBankConflict=0`
- same-build W12 baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_084834`
  with `kernel_ticks=71312605`, MMAC active avg `21.5931%`, coissue
  `30130/20996`

Decision: `REJECT_PERF_STATS_ONLY`.  Keep only as an opt-in diagnostic.  The
result is another concrete reminder that higher coissue count is not enough:
MMAC active fell and ticks regressed by about `0.49%`.

Rule: local consumer schedule asymmetry alone does not solve the W12 lockstep
problem.  The next useful candidate should change the conveyor or operand
readiness/control path with workbook-backed resource reasoning, not just mix
two already-rejected score/dP read schedules.

Dedicated LDS sidecar resource rejection:

- workbook design attempted to move packed sidecar from consumer global reads
  into a dedicated packet-local LDS sidecar page under the existing
  `RawFilled` token
- intended benefit: reduce xcu `flat_rd -> immed` and give producer recurring
  useful work without adding a new ABarrier generation
- resource correction: current W12 `DkvLdsLayout` is already exactly 128KB:
  `Q 16KB + dO 16KB + K 32KB + V 32KB + Q^T 16KB + dO^T 16KB`
- adding two sidecar pages costs only 768B, but total becomes
  `131840B > 131072B`
- remote build failed at static resource gate before PMD; code was removed
- no correctness/perf row exists because the candidate never passed build

Decision: `REJECT_RESOURCE_DESIGN`.  Do not add sidecar/scratch LDS by
appending bytes to the current W12 layout.  A future sidecar-in-LDS design must
free or reuse an existing page/lifetime; raw-overlay is already a negative
example for adding an extra ownership generation.

Raw/source layout swap boundary:

- source-score probe:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_092729`
- raw-dVdK probe:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_094838`
- both temporary paths passed static/resource checks and then failed H1/S128
  correctness before any H1/S1024/perf run
- conclusion: current W12 layout still needs raw `Q/dO` pages for score/dP and
  source-layout `Q^T/dO^T` pages for dV/dK; neither page family is a drop-in
  replacement for the other with the current `ds_read_matrix_trans_pair`
  operand mapping
- both temporary code paths were removed from the clean repo
- next attempt to free LDS must change lifetimes/topology or start with a
  smaller documented instruction-layout probe, not another direct raw/source
  swap inside full dKV

Main-bottleneck RawUsed pass:

- canonical `fa3_bwd_dkv_mmac12_kernel` now keeps the micro early-release
  behavior: consumer groups call
  `consumer_dkv_mmac_loop<Tile, Bar, group, true>`
- remote build/static/metadata PASS:
  `private=0`, `sgpr=84`, `vgpr=112`, no spill/scratch
- H1/S1024 stats-only:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_235302`
  with `kernel_ticks=70505435`, MMAC active avg `21.8494%`
- H1/S1024 full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_235606`
  with `kernel_ticks=70658770`, MMAC active avg `21.8783%`,
  `coissue=31248/20588`, `ldsBankConflict=0`
- xcu detail still shows RawUsed as the dominant bubble:
  `s_abarrier_try_wait -> s_xor_b32`, `3.576M cycles`, `28.66%`
- decision: `ACCEPT_MICRO_CANONICAL`, not a pipeline solution

Split raw/source ABarrier retry:

- tried splitting raw `Q/dO` and source-layout `Q^T/dO^T` lifetimes into
  separate ABarrier token families without adding LDS bytes
- resource/correctness passed:
  - H1/S128:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_001233`
  - H1/S1024:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_001305`
- H1/S1024 stats rejected it:
  `kernel_ticks=75855325`, MMAC active avg `21.0489%`,
  `coissue=26810/18008`, `ldsBankConflict=0`
- code removed from active route; project gate again forbids source-layout
  split tokens in the canonical route
- rule: do not add more ABarrier generations to solve RawUsed unless xcu first
  proves the new wait is hidden by real consumer work

## 2026-07-03 Tile Ledger And 60% Active Redesign Target

- shared design workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- repo ledger:
  `results/tile_ledger_20260703.md`
- active source remains the single canonical `fa3_bwd_dkv_mmac12_kernel`
  route; no new phase stack was added for this ledger pass
- current source tile is `Mq=32,Nk=128,D=128,W12`
- current per-consumer-wave work per q tile is only `64` MMAC:
  `16 score + 16 dP + 16 dV + 16 dK`
- current `S1024` dispatch total is `131072` MMAC, matching PMD
- current LDS is exactly 128KB:
  `K/V resident 64KB + raw Q/dO pages 32KB + source-layout Q^T/dO^T pages 32KB`
- this closes the simple append-LDS options; sidecar or larger tiles must
  replace/reshape an existing lifetime
- next implementation candidate should increase effective MMAC-island length
  or hide RawUsed without extra ABarrier token families and without duplicate
  score/dP

## 2026-07-03 W16 Split-Producer Audit

Active code state:

- keep the `producer_qk_loop` runtime loop-bound fix:
  the producer now stops at `q_tiles` instead of fixed
  `Tile::kQTilesPerCta`
- keep W12 canonical early-release in `fa3_bwd_dkv_mmac12_kernel`
- do not keep W16 early-release in `fa3_bwd_dkv_mmac_kernel`; it was measured
  and rejected

Evidence:

- before the loop-bound fix, legacy W16 split-producer H1/S128 hung after more
  than `18B` simulated ticks and warned `read vgpr112 before writing`
- after the loop-bound fix, W16 H1/S128 and H1/S1024 pass correctness:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_013701`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_013725`
- W16 q-loop-fix H1/S1024 stats:
  `kernel_ticks=73333260`, MMAC active avg `20.5512%`, coissue
  `34952/26250`, `ldsBankConflict=0`
- W16 early-release H1/S1024 also passes correctness but regresses:
  `kernel_ticks=74882080`, MMAC active avg `20.3284%`, coissue
  `32395/24677`, `ldsBankConflict=0`

Decision:

`OBSERVE_CORRECTNESS_PASS` for the q-loop repair, `REJECT_PERF_STATS_ONLY` for
W16 early-release.  The current evidence says the 60% MMAC-active gap is not
mainly solved by increasing the role count to 16 waves or by copying W12's
early-release protocol.  Next work should attack useful MMAC density and
barrier/control bubbles in the canonical route.

## 2026-07-03 dV/dK Source Quad-Read Probe

Active code state:

- quad-read helper was tested and removed from the active route
- canonical dV/dK source reads are back to the previous pair-wrapper form
- q-loop fix and W12 canonical early-release remain

Evidence:

- FWD PMD path uses contiguous `ds_read_matrix` blocks without `s_nop` for
  some source reads
- BWD probe changed only `dv_dk_read_owner16_sources4` to read `dO^T` and
  `Q^T` as two contiguous 4-read blocks
- resource/correctness passed:
  `private=0`, `sgpr=82`, `vgpr=112`, no spill/scratch,
  `ldsBankConflict=0`
- H1/S1024 stats:
  `kernel_ticks=70560945`, MMAC active avg `21.7716%`,
  `mmop_runtime_share=45.5229%`, coissue `31952/20804`

Decision:

`OBSERVE_REJECT_PERF`.  The probe reduced local VOP/runtime pressure but did
not beat W12 canonical ticks or MMAC active.  The next 60% attempt must change
the packet/barrier conveyor or increase the effective MMAC island, not only
batch one operand-read family.

## 2026-07-03 H21A Q-pair Control-Only Boundary

Active code state:

- H21A q-pair code has been removed from the active route
- canonical `fa3_bwd_dkv_mmac12_kernel` is back to the accepted W12
  early-release loop
- remote build/static metadata after revert is clean:
  `private=0`, `sgpr=84`, `vgpr=112`, no spill/scratch

Evidence:

- H21A workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `19_qpair_design`
- helper-based q-pair consumer loop failed metadata:
  `sgpr_count=100`, `sgpr_spill_count=39`
- function-local macro q-pair loop also failed metadata:
  `sgpr_count=100`, `sgpr_spill_count=38`

Decision:

`REJECT_STATIC_SPILL`.  Reusing page0/page1 as a logical q-pair is still a
reasonable design target, but a direct `q_tile += 2` control-only rewrite is
not acceptable: it increases consumer SGPR live range before producing any PMD
evidence.  The next attempt should either reduce live variables in the
consumer body first or switch to a topology that lengthens useful MMAC without
duplicating the whole control body in one branch.

## 2026-07-03 H22 First-Tile Peel

Active code state:

- canonical `fa3_bwd_dkv_mmac12_kernel` keeps early RawUsed release
- `consumer_dkv_mmac_loop` now peels `q_tile=0` out of the hot loop
- the steady q-loop starts at `q_tile=1` and no longer carries the runtime
  first-tile accumulator-seed branch

Evidence:

- remote static/symbol gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no SGPR/VGPR spill, no scratch
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_042133`
- H1/S1024 stats/full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_042139`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_042626`
- full perf metrics:
  `kernel_ticks=67665325`, `MMOP=131072`, `ldsBankConflict=0`,
  `coissue=23064/18083`, `MMAC active share=23.0485%`
- shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_042626_clean_w12_h22_first_tile_peel_h1s1024_sqc7_fullperf`
- xcu still reports dominant bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `28.44%`,
  `global_load_dwordx3 -> s_waitcnt` `11.59%`

Decision:

`ACCEPT_MICRO_CONTROL`.  Keep this as the current canonical source baseline.
It is a useful cleanup and improves the H1/S1024 active share to about `23%`,
but it does not change the architectural bottleneck.  The next patch must
target exposed ABarrier/sidecar wait or longer useful MMAC islands; do not
reintroduce the rejected q-pair body duplication.

## 2026-07-03 H23 Remove ds_read_matrix Helper s_nop

Active code state:

- canonical `fa3_bwd_dkv_mmac12_kernel` keeps H22 first-tile peel
- `include/shaobo_instr.h::ds_read_matrix_trans_pair` no longer emits the
  fixed leading `s_nop 0`
- no new kernel/path/phase was added; this is an in-place helper cleanup for
  the current canonical dKV route

Evidence:

- remote static/symbol gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no SGPR/VGPR spill, no scratch
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_043940`
- H1/S1024 stats/full perf correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_043946`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_044220`
- full perf metrics:
  `kernel_ticks=67246725`, `MMOP=131072`, `ldsBankConflict=0`,
  `coissue=22768/18808`, `MMAC active share=23.2228%`
- shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_044220_clean_w12_h23_no_dsread_snop_h1s1024_sqc7_fullperf`
- xcu deltas versus H22:
  issue count `847944` vs `897096`,
  `ds_read_matrix` latency `475420` vs `557792`,
  hot `s_nop` row removed

Decision:

`ACCEPT_MICRO_SQTT`.  H23 is now the current canonical baseline.  The main
route is cleaner but still far from the `60%` MMAC active target.  The next
change should be driven by the remaining top xcu bubbles:
RawUsed/ABarrier around `28.48%` and sidecar `global_load_dwordx3 -> wait`
around `11.54%`.

## 2026-07-03 H24 Raw ABarrier Wait Builtin Boundary

Active code state:

- H24A/H24B code has been removed from the active route
- canonical source is restored to H23:
  raw Filled/Used waits use the asm `abarrier_try_wait<true>` wrapper, and
  `ds_read_matrix_trans_pair` still has no leading `s_nop`

Evidence:

- H24A changed both `wait_raw_used_page` and `wait_raw_filled_page` to the
  builtin wait wrapper.  Build completed, but symbol metadata failed:
  `private_segment_fixed_size=12`, `sgpr=80`, `vgpr=112`, no spills.
- H24B changed only `wait_raw_used_page` to builtin.  Build completed, but
  symbol metadata still failed:
  `private_segment_fixed_size=12`, `sgpr=82`, `vgpr=112`, no spills.
- Restored active H23 metadata PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- Workbook design/negative result sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `20_abarrier_wait_design`.

Decision:

`REJECT_STATIC_PRIVATE`.  The top SQTT ABarrier bubble cannot be solved by
blindly swapping raw wait asm to builtin in the canonical q-loop.  Future
ABarrier work should change page lifecycle/topology or add useful overlap; if
builtin wait is revisited, it belongs in a focused codegen probe rather than
the dKV performance kernel.

## 2026-07-03 H25 RawUsed Before Full dV/dK Island

Active code state:

- H25 code has been removed from the active route
- canonical source is restored to H23/H24-restored ordering:
  high source read is issued, low dV/dK MMAC runs while high source matures,
  then RawUsed is released before the high dV/dK MMAC group

Evidence:

- H25 reordered only `dv_dk_mmac_owner16_read4x2_early_release`:
  wait for high source and arrive RawUsed before both low/high dV/dK MMAC
  groups.
- Static/resource gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_050642`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_050705`.
- H1/S1024 stats regressed:
  `kernel_ticks=68373305`, `MMAC active share=22.8899%`,
  versus H23 `kernel_ticks=67246725`, `MMAC active share=23.2228%`.

Decision:

`REJECT_PERF_STATS_ONLY`.  Releasing RawUsed before the full dV/dK island is
not enough; it sacrifices the useful overlap where low dV/dK MMAC hides high
source-read latency.  Keep H23 ordering.  Future lifecycle attempts must
preserve high-read hiding or make producer work overlap with existing low/high
MMAC without adding pre-MMAC wait.

## 2026-07-03 H26 Causal Sidecar Split Boundary

Active code state:

- H26 code has been removed from the active route
- canonical source is restored to H23:
  the generic `softmax_ds_owner16_from_global_sidecar` is used in the
  consumer loop, and the only accepted code changes remain first-tile peel and
  the removed `ds_read_matrix` helper `s_nop`

Evidence:

- H26 added a causal-specific sidecar helper that removed the inner
  `(!causal || ...)` term while preserving the per-element `krow <= qrow`
  mask and all MMAC/ABarrier/LDS ownership.
- Workbook design/negative result sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `22_causal_sidecar_design`.
- Static/resource gate PASS for candidate:
  `private=0`, `sgpr=80`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_052600`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_052606`.
- H1/S1024 stats regressed:
  `kernel_ticks=70504980`, `MMAC active share=22.5343%`,
  `VOP active share=21.6469%`, `VALU=230108`,
  versus H23 `kernel_ticks=67246725`, `MMAC active share=23.2228%`,
  `VOP active share=20.9728%`, `VALU=213208`.
- Restored H23 metadata PASS after revert:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.

Decision:

`REJECT_PERF_STATS_ONLY`.  The runtime causal boolean is not the current
MMAC-active limiter.  The helper split is correct but increases SGPR/VALU and
regresses ticks/active.  Do not continue causal helper splitting unless a
future xcu trace proves a specific predicate row is again on the critical path.

## 2026-07-03 H27 RawUsed After High-Source Issue

Active code state:

- H27 is kept as the current W12 micro-improved route.
- The active ordering in `dv_dk_mmac_owner16_read4x2_early_release` is:
  wait low source, issue high-source `ds_read_matrix`, arrive `RawUsed`, run
  low dV/dK MMAC, wait high source, then run high dV/dK MMAC.
- The accepted prior changes remain in place:
  producer q-loop uses runtime `q_tiles`, the first q tile is peeled out of
  the hot consumer loop, and `ds_read_matrix_trans_pair` has no leading
  `s_nop`.

Evidence:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `23_rawused_after_high_issue`.
- Static/resource gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_053909`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_053933`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_054127`.
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/h27_rawused_after_high_issue_h1s1024_20260703_054127`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_054127_clean_w12_h27_rawused_after_high_issue_h1s1024_sqc7_fullperf`.
- H1/S1024 full-perf metrics:
  `kernel_ticks=66892280`, `MMAC active share=23.3787%`,
  `MMOP=131072`, `VALU=213208`, `SCA=289456`, `ldsBankConflict=0`.

Decision:

`ACCEPT_MICRO_OBSERVE_PIPELINE`.  H27 validates that a high-source
`ds_read_matrix` can be issued before releasing the raw page in this PMD run,
and it gives a small improvement over H23.  It does not change the main
bottleneck: RawUsed/ABarrier and sidecar global-load waits still dominate.
Future changes should not add more local reorder variants without first
showing in the workbook how they reduce those waits or lengthen a useful MMAC
window.

## 2026-07-03 H28 Producer Sidecar Cache-Warm

Active code state:

- H28 is now the best active W12 route.
- Producer `wave_local==0` cache-warms packed sidecar rows for q tiles
  `>=2` before waiting `RawUsed` and refilling that page.
- No value is handed from producer to consumer; consumer sidecar math and
  global sidecar loads remain unchanged.
- No LDS bytes, ABarrier tokens, output ownership, or MMOP count changed.

Evidence:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `24_sidecar_cache_prefetch`.
- Static/resource gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_060502`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_060528`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_060726`.
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/h28_sidecar_cache_prefetch_h1s1024_20260703_060726`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_060726_clean_w12_h28_sidecar_cache_prefetch_h1s1024_sqc7_fullperf`.
- H1/S1024 full-perf metrics:
  `kernel_ticks=66630200`, `MMAC active share=27.9004%`,
  `MMOP=131072`, `VALU=214952`, `SCA=292576`, `ldsBankConflict=0`.

Decision:

`ACCEPT_PIPELINE_OBSERVE`.  H28 keeps correctness/resource gates clean and
raises MMAC active from H27's `23.3787%` to `27.9004%`.  xcu shows a plausible
mechanism: RawUsed exposed wait falls from `28.41%` to `26.25%`, and sidecar
global wait falls from `11.69%` to `11.42%`.  It also introduces a new
producer prefetch `flat_load_dword -> s_waitcnt` bubble at `1.57%`, so the next
optimization should hide or batch that prefetch rather than adding more
independent memory probes.

## 2026-07-03 H29 Fire-And-Forget Sidecar Prefetch

Active code state:

- H29 was tested and then reverted.  The active source is back to the H28
  explicit sidecar prefetch form with empty-asm value consumption.
- H29 changed only `prefetch_packed_sidecar_tile`, replacing the explicit
  loaded values with volatile fire-and-forget reads.

Evidence:

- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_061743`.
- H1/S1024 stats-only PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_061810`.
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_062248`.
- Full-perf metrics:
  `kernel_ticks=66690260`, `MMAC active share=27.9272%`,
  `MMOP=131072`, `VALU=214952`, `SCA=292576`, `ldsBankConflict=0`.
- H28 comparison:
  `kernel_ticks=66630200`, `MMAC active share=27.9004%`.
- Restored H28 static metadata:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.

Decision:

`OBSERVE_ACTIVE_REJECT_TICKS`.  H29's active-share movement is too small and
does not shorten full-perf ticks, so H28 remains the best active route.  Future
work should not chase this exact fire-and-forget prefetch shape; the next
useful changes need to shorten RawUsed/sidecar waits or make longer MMAC
islands.

## 2026-07-03 H30 Future Sidecar Prefetch Placement

Active code state:

- H30 is now the best active W12 route.
- Producer still uses H28's explicit sidecar cache-warm helper, but the call is
  moved from "before waiting RawUsed for current q tile" to "after publishing
  q_tile, prefetch q_tile+2".
- The canonical kernel remains `fa3_bwd_dkv_mmac12_kernel`; no new performance
  path was added.

Evidence:

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  `26_sidecar_future_prefetch`.
- Static/resource gate PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill/scratch.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_063323`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_063344`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_063614`.
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/h30_future_sidecar_prefetch_h1s1024_20260703_063614`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_063614_clean_w12_h30_future_sidecar_prefetch_h1s1024_sqc7_fullperf`.
- Full-perf metrics:
  `kernel_ticks=66321255`, `MMAC active share=28.3952%`,
  `MMOP=131072`, `VALU=214984`, `SCA=296832`, `ldsBankConflict=0`.
- H28 comparison:
  `kernel_ticks=66630200`, `MMAC active share=27.9004%`.
- xcu comparison:
  duration `146440 -> 145764`, RawUsed bubble `26.25% -> 25.89%`,
  sidecar wait `11.42% -> 11.38%`, producer prefetch wait
  `1.57% -> 1.79%`.

Decision:

`ACCEPT_PIPELINE_MICRO`.  H30 should remain in the active route.  It confirms
the sidecar cache-warm placement matters, but the remaining gap to 60% is still
large.  Next work should treat `s_waitcnt` and RawUsed page lifetime as the
main bottleneck, not keep trying sidecar load opcode variants.

## 2026-07-03 Canonical dKV Route Convergence

Status: `CODE_GOVERNANCE_ACCEPT`

Tri Dao-style implementation rule:

- Active dKV optimization now has one supported hot route:
  `fa3_bwd_dkv_kernel`.
- Reference correctness remains available through `--check=1`.
- API/standalone/scripts no longer route to Mq64, overlay, score-brick,
  causal-skip, mixed-score, early-release, or generic probe paths.
- Historical kernels are still in the source for a later physical deletion
  pass, but they are unreachable from the supported route.

Evidence:

- Remote build PASS and no compile warnings after convergence.
- `scripts/check_dkv_kernel_gate.py` PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no SGPR/VGPR spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_090438`.
- H1/S1024 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_090629`.
- H1/S1024 stats-only:
  `simTicks=69908930`, `MMOP=131072`, `ldsBankConflict=0`,
  `coissue=22685/18256`, `MMAC active=23.5544%`.

Next:

- Do not add new public dKV paths.  Apply the next redesign inside
  `fa3_bwd_dkv_kernel`.
- Next structural target remains Tri Dao-style packet conveyor: reduce
  RawUsed/ABarrier and read-to-use gaps while preserving no duplicate score/dP,
  correct dK/dV ownership, no spill/scratch, and `ldsBankConflict=0`.

## 2026-07-03 Archive Unreachable Global Kernels

Status: `CODE_GOVERNANCE_ACCEPT`

Change:

- Old bring-up/rejected global kernels are compile-time archived.
- Build now instantiates only the canonical dKV performance kernel plus
  reference correctness kernels.
- This is a code-hygiene step; no performance claim is made.

Evidence:

- Remote build PASS.
- WDRA branch report shrank to the canonical kernel:
  producer `6/16`, consumer0 `159/160`, consumer1 `159/160`.
- `scripts/check_dkv_kernel_gate.py` PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, no spill.
- H1/S128 correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_091957`.

Next:

- Physical deletion of archived kernels can happen after we confirm their
  lessons are fully captured in workbook/ledger.
- The next real optimization should alter the RawUsed/source-read/softmax
  conveyor inside `fa3_bwd_dkv_kernel`, not create a new launch path.

## 2026-07-03 Canonical After-Convergence SQTT Rebaseline

Status: `PASS_XCU_BASELINE`

Workbook-first evidence:

- `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`
- sheet `28_current_xcu_rebaseline`

Shared perf archive:

- `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_093932_clean_canonical_after_convergence_h1s1024_sqc7_fullperf`
- open helper perf:
  `canonical_after_convergence_H1S1024_sqc7.perf`

Current active source state:

- canonical performance kernel: `fa3_bwd_dkv_kernel`
- shape: `B=1,H=1,S=1024,D=128,causal=true`
- env: `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`
- static/resource gate: `private=0`, `sgpr=78`, `vgpr=112`, no spills
- branch windows: producer `6/16`, consumer0 `159/160`, consumer1 `159/160`

PMD evidence:

- causal stats-only pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093524`,
  `kernel_ticks=66540565`, `MMAC active=23.4212%`.
- causal=false diagnostic:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093608`,
  `pass=0`, `dk_rel_l2=0.0199722`, `dv_rel_l2=0.002617`,
  `MMAC active=20.5948%`.
- full perf pass:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_093932`,
  `kernel_ticks=66411800`, `MMAC active=23.4288%`,
  `VOP active=21.3239%`, `coissue=23057/18211`, `ldsBankConflict=0`.

xcu evidence:

- dispatch 0 duration `145960`, inst issues `861816`, avg active waves
  `86.45`.
- top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `25.95%`,
  `global_load_dwordx3 -> s_waitcnt` `11.28%`,
  `v_mmac -> v_mmac` `6.55%`,
  `v_mmac -> s_waitcnt` `6.35%`,
  `s_abarrier_try_wait -> s_waitcnt` `5.59%`,
  `ds_read_matrix -> s_waitcnt` `2.38%`.

Conclusion:

This post-convergence run is clean and, after metric normalization, effectively
matches H30.  H30 recomputed from its archived `stats.txt` with the same
`sum(mmopRunTimeCounter)/sum(activeTimeCounter)` formula is `23.4386%`, while
current is `23.4288%`; the older `28.3952%` H30 value used a different active
metric and should not be compared directly.  The next step should not add
another public phase or chase generic read batching.  Continue modifying
`fa3_bwd_dkv_kernel` in place with a workbook-backed hypothesis that targets
raw-page ABarrier/control or exposed sidecar wait.

## 2026-07-03 Physical Removal Of Stale dKV Routes

Status: `CODE_GOVERNANCE_ACCEPT`

Change:

- Physically removed the archived `#if 0` bring-up/rejected global kernels.
- Removed old public dKV path constants; only
  `kDkvPathReferenceCorrectness=1` and `kDkvPathCanonicalDkv=5` remain.
- Removed stale Mq64, semantic-page, sidecar-overlay, causal-skip, score-brick,
  and split-producer helper bodies from `src/dkv_kernel.cpp`.
- Removed standalone probe/fragment scripts that targeted deleted
  `fa3_bwd_dkv_probe` behavior.
- Simplified the standalone executable to two modes:
  default canonical correctness and `--check=1` reference correctness.
- Tightened `scripts/check_dkv_kernel_gate.py`: it no longer strips archived
  blocks and now forbids stale experiment route symbols in active source.

Verification:

- Remote build PASS.
- dKV kernel gate PASS.
- Symbol metadata PASS:
  `private=0`, `sgpr=78`, `vgpr=112`, `sgpr_spill=0`, `vgpr_spill=0`.
- WDRA branch windows unchanged:
  producer `6/16`, consumer0 `159/160`, consumer1 `159/160`.
- H1/S128 PMD correctness PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_111624`.
- H1/S128 stats:
  `simTicks=17781855`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=17781855`, `MMOP=2048`, `ldsBankConflict=0`,
  coissue `351/238`.

Decision:

This is a code-governance checkpoint, not a performance promotion.  Future
dKV work should edit `fa3_bwd_dkv_kernel` directly and use git/workbook/ledger
for experiment history instead of adding phases or preserving rejected source
routes.

## 2026-07-03 Mq64 Same-LDS Experimental State

Status: `OBSERVE_CORRECTNESS_REJECT_PERF`

Current dirty source state:

- canonical dKV tile is temporarily `DkvTileD128Mq64Nk128W12`.
- raw `Q/dO` use the verified `matrix_load_32x16_b16` same-LDS contract.
- LDS budget is exactly 128KB:
  `K/V 64KB + Q/dO raw double buffer 64KB`.
- consumer WDRA window is `208`; branch report is producer `6/16`,
  consumer0 `204/208`, consumer1 `200/208`.
- symbol metadata is clean:
  `private=0`, `sgpr=80`, `vgpr=144`, `sgpr_spill=0`, `vgpr_spill=0`.

Correctness:

- Mq64-B half-sequential:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_150917`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_150946`,
  both PASS.
- Mq64-A full-score:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_151316`
  and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_151356`,
  both PASS.

Performance:

- canonical H1/S1024 rebaseline:
  `kernel_ticks=66411800`, `MMAC active=23.4288%`.
- A1 same-LDS Mq32:
  `kernel_ticks=67704000`, `MMAC active=22.8697%`.
- Mq64-B:
  `kernel_ticks=67825940`, `MMAC active=23.1120%`.
- Mq64-A:
  `kernel_ticks=67762240`, `MMAC active=23.0392%`.

xcu for Mq64-A:

- full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_151548`.
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/mq64A_full_h1s1024_20260703_151548`.
- top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `25.92%`,
  `global_load_dwordx3 -> s_waitcnt` `10.77%`,
  `s_abarrier_try_wait -> s_waitcnt` `7.92%`,
  `v_mmac -> v_mmac` `6.51%`,
  `ds_read_matrix_trans_format -> s_waitcnt` `3.32%`.

Conclusion:

`Mq64` proves the larger Wq can be made resource-clean with WDRA, but it is not
a performance promotion. The bigger page reduces q-tile count but does not
move the dominant RawUsed/control bubble; the source read wait grows and VOP
active rises. If work continues from this dirty state, the next edit must
directly attack RawUsed/control or useful producer/helper overlap. If the next
task is independent, revert to the canonical rebaseline first.

## 2026-07-03 W16 Split-Producer Mq64 Structural Probe

Status: `STRUCTURAL_PASS_REJECT_PERF`

Current source state:

- canonical dKV kernel is now a 16-wave structural probe.
- wave roles:
  waves0-3 producer K+Q, waves4-7 consumer0, waves8-11 consumer1,
  waves12-15 producer V+dO.
- PMD confirms `wg size=(1024,1,1)` and
  `16 waves using this aBarrier/eBarrier group`.
- static gate now requires the W16 annotation, the wave12-15 branch, and the
  two producer loops.

Correctness/resource evidence:

- Remote build/static/symbol gates PASS.
- Metadata:
  `private=0`, `sgpr=74`, `vgpr=112`, no spill/scratch.
- Branch windows:
  producer KQ `6/16`, consumer0 `204/208`, consumer1 `204/208`,
  producer VDout `1/16`.
- H1/S128 PASS and H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260703_161054`.

Performance:

- H1/S1024 full perf:
  `kernel_ticks=69039425`, `MMOP=131072`, `ldsBankConflict=0`,
  coissue `27214/18060`, `MMAC active=22.3357%`,
  `VOP active=23.7529%`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260703_161054_clean_w16_split_mq64_h1s1024_sqc7_fullperf`.
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/w16_split_mq64_h1s1024_sqc7_20260703_161054_dispatch0`.
- xcu top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `38.07%`,
  `s_abarrier_try_wait -> s_waitcnt` `9.89%`,
  `global_load_dwordx3 -> s_waitcnt` `8.18%`,
  `v_mmac -> v_mmac` `4.18%`,
  `ds_read_matrix_trans_format -> s_waitcnt` `2.12%`.

Conclusion:

This answers the 12wave objection but does not solve the pipeline.  The W16
split still has severe ABarrier/control gaps and consumer lockstep.  Producer B
is functionally present but remains too thin in the steady q-loop, so it does
not provide a forward-style producer/helper overlap window yet.

Next:

Treat W16 as the active structural constraint only if the next step keeps the
user's 16-wave requirement.  The next edit must target the ABarrier raw-page
lifetime and producer-B useful work, not just role count.

## 2026-07-04 Active Route: Raw2 Sidecar Overlay On K/V LDS

Status: `ACCEPT_PIPELINE_MICRO`

Current source state:

- active dKV route remains the single canonical `fa3_bwd_dkv_kernel`
- role topology remains 16-wave:
  waves0-3 producer K/Q/sidecar, waves4-7 consumer0,
  waves8-11 consumer1, waves12-15 producer V/dO
- K/V are resident-loaded once, latched into consumer VGPR, then the K/V LDS
  region is reused for sidecar metadata
- raw Q/dO is back to two pages (`kRawBuffers=2`)
- sidecar is in LDS, overlaid on dead K/V resident storage after
  `ResidentUsed`

Static/resource evidence:

- remote build PASS
- `scripts/check_dkv_kernel_gate.py` PASS
- symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=60`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- branch windows:
  producer KQ `6/16`, consumer0 `198/208`, consumer1 `198/208`,
  producer VDout `1/16`

Correctness/perf:

- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_161558`
- H1/S1024 full perf PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_161747`
- metrics:
  `kernel_ticks=53719120`, `MMOP=131072`, `MMAC active=27.7542%`,
  `VALU=181916`, `SCA=297480`, `LDS=85822`, `VMEM=4352`,
  coissue `32106/18911`, `ldsBankConflict=0`
- shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_161747_clean_raw2_sidecar_kv_overlay_h1s1024_sqc7_fullperf`

xcu:

- first-pass output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/raw2_sidecar_kv_overlay_h1s1024_20260704_161747_dispatch0`
- selected window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/raw2_sidecar_kv_overlay_h1s1024_20260704_161747_dispatch0_window_bar6`
- top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `39.79%`,
  `s_abarrier_try_wait -> s_waitcnt` `7.82%`,
  `ds_read_b32 -> s_waitcnt` `2.43%`

Conclusion:

- Sidecar-in-LDS remains the right direction, and overlaying it on dead K/V
  LDS lets us recover raw Q/dO lookahead without extra LDS capacity.
- Improvement is real but small versus the immediately previous LDS-sidecar
  route: `54539485 -> 53719120` kernel ticks, about `1.50%`.
- The current blocker is ABarrier ownership exposure.  The next edit should
  reduce `ResidentUsed`/Raw page over-synchronization before adding any new
  buffers or phase tokens.

## 2026-07-04 Current Active Route: Single Raw Buffer LDS Sidecar

Status: `CODE_CONVERGENCE_ACCEPT`

Current source state:

- active dKV route remains the single canonical `fa3_bwd_dkv_kernel`
- role topology remains 16-wave:
  waves0-3 producer K/Q/sidecar, waves4-7 consumer0,
  waves8-11 consumer1, waves12-15 producer V/dO
- K/V are resident-loaded once and latched into consumer VGPR
- sidecar is producer-published into LDS, but no longer overlays K/V
- raw Q/dO uses one page (`kRawBuffers=1`)
- active barrier ledger is only:
  `ResidentFilled`, `Raw0Filled`, `Raw0Used`, `AllDone`
- `Raw1` and `ResidentUsed` are removed from source/contract and forbidden by
  `scripts/check_dkv_kernel_gate.py`

Static/resource evidence:

- remote build PASS
- `scripts/check_dkv_kernel_gate.py` PASS
- symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- branch windows:
  producer KQ `10/16`, consumer0 `196/208`, consumer1 `196/208`,
  producer VDout `4/16`

Correctness/perf:

- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_164439`
- H1/S1024 stats PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_164444`
- metrics:
  `kernel_ticks=54818400`, `MMOP=131072`, `MMAC active=26.6857%`,
  `VALU=180570`, `SCA=215376`, `LDS=85822`, `VMEM=4352`,
  coissue `19747/11693`, `ldsBankConflict=0`

Conclusion:

- This is now the active route by user decision.  The raw2 overlay path was
  valid and faster, but only by about `1.5%` versus single-buffer LDS-sidecar
  while adding another ownership token and lifetime rule.
- Continue from this simpler single-buffer baseline.  Do not reintroduce Raw1
  or `ResidentUsed` without a new workbook-backed hypothesis.

## 2026-07-04 BlockMq Template Checkpoint

Status: `CODE_GOVERNANCE_ACCEPT`, with `Mq128` marked `REJECT_PERF`.

Current source state:

- `BlockMq` is now a tile template parameter:
  `DkvTileD128MqNk128<BlockMq>`.
- Active dKV still instantiates `ActiveDkvTile = DkvTileD128MqNk128<64>`.
- Active `Mq64` keeps the static two-Mpair consumer path and does not
  instantiate the K/V raw overlay or `ResidentUsed` path.
- The rejected `Mq128` stress is recorded in workbook sheet
  `38_mq128_template`; it is not the active alias.

Resource/correctness evidence:

- active `Mq64` template build PASS:
  `private_segment_fixed_size=0`, `sgpr_count=84`, `vgpr_count=112`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`
- branch windows:
  producer KQ `10/16`, consumer0 `196/208`, consumer1 `196/208`,
  producer VDout `4/16`
- H1/S128 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_190032`
- H1/S1024 PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_190101`

Latest H1/S1024 metrics:

- `kernel_ticks=54887105`
- `MMOP=131072`
- `MMAC active=26.5542%`
- coissue `20049/12067`
- `ldsBankConflict=0`

Mq128 stress result:

- static compile-time expansion failed resource gate:
  `private_segment_fixed_size=8`, `sgpr_spill_count=18`,
  `vgpr_spill_count=2`
- dynamic M-block loop fixed resource/admission:
  branch consumer `190/208`, `private=0`, `sgpr=62`, `vgpr=112`, no spill
- H1/S1024 correctness passed, but performance regressed:
  `kernel_ticks=62473320`, `MMAC active=23.2158%`

Conclusion:

- `BlockMq` templating is useful code governance, but `Mq128` as implemented
  is not a performance solution.
- The key lesson is subtle: for fixed S, `Mq128` does not increase total MMOP;
  it doubles per-tile MMAC while halving q-tile count.  The only possible win
  is amortizing control/wait under a longer island.  Our dynamic-loop version
  lost more to control and overlay lifetime than it gained.
- Future `BlockMq>64` attempts need a design that keeps compile-time MMAC
  islands without SGPR spill, or xcu proof that dynamic-loop overhead is fully
  hidden.

## 2026-07-04 Full Perf Rebaseline And Sidecar Prefetch Rejection

Status: active source restored to `f27ec64`-equivalent baseline after rejected
sidecar experiments.

Current baseline full perf:

- case:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411`
- helper perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411/m5out/0/0/2673937_fa3_bwd_wasp_clean.perf`
- xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/f27_h1s1024_fullperf_20260704_191411_dispatch0`
- focused xcu window:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/f27_h1s1024_fullperf_20260704_191411_dispatch0_window_barrier_swait`

Metrics:

- `simTicks=58424275`
- `MMOP=131072`
- `MMAC active=26.6364%`
- coissue `20088/11882`
- `ldsBankConflict=0`
- xcu top bubbles:
  `s_abarrier_try_wait -> s_xor_b32` `47.46%`,
  `s_abarrier_try_wait -> s_waitcnt` `6.64%`,
  `ds_read_b32 -> s_waitcnt` `2.53%`

Rejected experiments:

- `sidecar future prefetch without token`: `REJECT_CORRECTNESS`.
  H1/S128 failed even after explicit `lgkmcnt(0)` after sidecar write.
- `two-page current sidecar with four-wave writer`: correctness PASS but
  `REJECT_PERF`; H1/S1024 regressed to `simTicks=60289320` and
  `MMAC active=26.1523%`.

Lesson:

- Future sidecar prefetch cannot be treated as raw-packet-ready unless it has
  explicit readiness/ownership or is still inside the same raw epoch.
- Four-wave sidecar writing is correct, but in the current code shape it adds
  control/VALU/SCA cost without reducing the dominant ABarrier bubble.
- Keep active source simple; the next useful move must reduce raw/ABarrier
  lifetime itself, not only move sidecar work around.

Tail `AllDone` wave0-only wait probe:

- temporary change: only wave0 waited `AllDone` and invalidated abarriers at
  kernel tail; all waves still synchronized at the final `__syncthreads()`
- resource/correctness:
  - build/static gates PASS
  - metadata `private=0`, `sgpr=86`, `vgpr=112`, no spill
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_193951`
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_194010`
- performance:
  - baseline `simTicks=58424275`, `MMAC active=26.6364%`,
    coissue `20088/11882`
  - probe `simTicks=58700915`, `MMAC active=26.7200%`,
    coissue `18982/11595`
- decision: `REJECT_PERF`; code reverted
- lesson: tail-only ABarrier cleanup does not reduce the dominant raw-loop
  `s_abarrier_try_wait -> s_xor_b32` bubble and slightly hurts same-shape
  ticks.

Mq128 static-scoped direct probe:

- temporary change:
  - `ActiveDkvTile = DkvTileD128MqNk128<128>`
  - `BlockMq != 64` consumer path used compile-time
    `consume_mq_tile_owner16<Tile, Wdra, 0, FirstAccum>` instead of the
    dynamic M-loop
- intended benefit:
  - double per-packet MMAC island from `128` to `256` MMAC per consumer
  - halve raw Q/dO handshakes for fixed S
  - directly attack the dominant raw ABarrier/control bubble
- resource result:
  - build completed but metadata gate failed
  - `private_segment_fixed_size=8`
  - `sgpr_count=104`, `sgpr_spill_count=18`
  - `vgpr_count=112`, `vgpr_spill_count=2`
  - branch consumer windows exactly `208/208`
- decision: `REJECT_RESOURCE`; code reverted before PMD
- lesson: Mq128 remains the right class of algorithmic lever, but direct static
  expansion is not the implementation.  Future Mq128 work must first redesign
  lifetime/ownership so metadata is clean.

248-VGPR resource check:

- temporary change: same direct static Mq128 probe, but
  `WdraResourceWindows::kConsumerVgprs = 248`
- result:
  - `private_segment_fixed_size=0`
  - `sgpr_count=100`, `sgpr_spill_count=18`
  - `vgpr_count=132`, `vgpr_spill_count=0`
  - branch consumers `209/248`
- conclusion: widening the consumer VGPR window fixes the private/VGPR side,
  but the direct static Mq128 implementation is still blocked by SGPR spill.
  The next design must reduce scalar live state/control expansion, not only
  raise VGPR allocation.
- baseline Mq64 source and remote binary were restored after the check.

Further Mq128 resource probes:

- noinline M-pair helper:
  - `consume_mq_mpair_owner16` marked noinline under Mq128/consumer248
  - result: `private_segment_fixed_size=492`, `sgpr_spill_count=0`,
    `vgpr_spill_count=0`
  - decision: `REJECT_RESOURCE`
  - lesson: function boundary cuts SGPR lifetime but device call/private memory
    is unacceptable
- explicit four-Mpair sequence:
  - Mq128 first/rest q tile paths explicitly called MBlockBase
    `0,2,4,6`, without template recursion
  - result: `private=0`, `sgpr_count=100`, `sgpr_spill_count=18`,
    `vgpr_count=132`, `vgpr_spill_count=0`
  - decision: `REJECT_RESOURCE`
  - lesson: template recursion is not the root cause; the inlined four-Mpair
    static branch itself has too much scalar/control live state
- causal=true literal specialization:
  - Mq128/consumer248 static path passed literal `causal=1` into the consumer
    calls
  - result: `private=0`, `sgpr_count=100`, `sgpr_spill_count=16`,
    `vgpr_count=132`, `vgpr_spill_count=0`
  - decision: `REJECT_RESOURCE`
  - lesson: causal mask scalar work contributes, but only slightly; it is not
    enough to make Mq128 static resource-clean
- baseline Mq64 source and remote binary were restored and metadata PASS.

Mq128 dynamic causal=true probe:

- temporary change:
  - `ActiveDkvTile = DkvTileD128MqNk128<128>`
  - dynamic M-loop retained
  - consumer calls passed literal `causal=1`
- resource/correctness:
  - metadata PASS: `private=0`, `sgpr=58`, `vgpr=112`, no spill
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mq128_dyn_causal1_20260704_201845_s128`
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mq128_dyn_causal1_20260704_201914_s1024`
- performance:
  - `simTicks=65580515`, `MMAC active=23.6126%`
  - baseline Mq64 is `simTicks=58424275`, `MMAC active=26.6364%`
  - `VALU=269724` versus baseline `180570`
- decision: `REJECT_PERF`; code reverted
- lesson: dynamic Mq128 can be resource-clean, but its dynamic control/VALU
  overhead overwhelms the reduced raw barrier count.

## 2026-07-04 Q/dO Split Lifetime Probe

Design basis:

- Workbook sheet `43_q_dout_split_lifetime` records the dependency DAG:
  Q is consumed by score and dK, while dO is consumed by dP and dV.
- The probe tested whether Q and dO could use separate filled/used tokens so
  producer0 could start the next Q+sidecar packet before producer1's dO page
  was fully released.

Result:

- Full Q-first dK/dV split:
  - correctness PASS at H1/S128 and H1/S1024
  - metadata PASS: `private=0`, `sgpr=86`, `vgpr=112`, no spills
  - H1/S1024 `simTicks=63880635`, `MMAC active=24.4167%`
  - decision: `REJECT_PERF`
- Token-only split with combined dV/dK MMAC preserved:
  - correctness PASS at H1/S128 and H1/S1024
  - metadata PASS: `private=0`, `sgpr=86`, `vgpr=112`, no spills
  - H1/S1024 `simTicks=61733035`, `MMAC active=25.1155%`
  - decision: `REJECT_PERF`
- Baseline remains:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_191411`,
  `simTicks=58424275`, `MMAC active=26.6364%`.

Lesson:

- Separate Q/dO lifetimes are legal, but the current one-page Mq64 route does
  not gain enough usable overlap to pay for the extra token/SCA/control cost.
- Do not reintroduce Q/Dout split tokens as the next main step.  A future
  design must first create a larger useful-work window or larger GEMM island,
  then consider finer lifetime release inside that window.
- Active source and remote binary have been restored to the single
  `Raw0Filled/Raw0Used` baseline.

## Rejected Probe: Nk32 Packed Owner16x2

- Status: `REJECT_RESOURCE`; active code restored to baseline after this
  evidence.
- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dkv_fwdstyle_tile_design_20260703.xlsx`,
  sheet `44_nk32_four_consumer_probe`.
- The temporary source changed the canonical kernel so waves4-7 each processed
  two adjacent owner16 blocks, logically covering `Nk=32` per active consumer
  wave.
- `Raw0Used` arrival count changed from eight consumers to four active
  consumers; waves8-11 were inactive helper waves.
- First build with VGPR windows `16/248/16/16` failed compiler WDRA
  granularity.
- After balancing the inactive branch to `40` VGPRs, build completed, but the
  target kernel metadata failed:
  `private_segment_fixed_size=236`, `vgpr_spill_count=58`, `sgpr_count=96`,
  `vgpr_count=80`.
- Conclusion: owner16x2 packing keeps two dV/dK accumulator sets live at once
  and is not a viable resource shape.

## 2026-07-04 Raw2 Page-Local ABarrier Result

- Status: `ACCEPT_MICRO_OBSERVE`.
- Canonical code now uses `kRawBuffers=2` with page-local raw ownership
  barriers:
  - `Raw0Filled/Raw0Used = 2/3`
  - `Raw1Filled/Raw1Used = 4/5`
  - `AllDone = 6`
- A one-token raw2 attempt was rejected first because PMD aborted with
  `ABARRIER_CNT_ERROR` on `barId 2`; this established that two outstanding raw
  pages need separate ABarrier generations.
- Static/resource:
  - gate PASS
  - metadata `private=0`, `sgpr=60`, `vgpr=112`, no SGPR/VGPR spill
  - branch windows `producerKQ=6/16`, `consumer0=198/208`,
    `consumer1=198/208`, `producerVDout=1/16`
- Correctness:
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_221533`
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_221539`
- Full perf:
  - remote:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260704_221910`
  - shared:
    `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_221910_clean_raw2_tokens_h1s1024_sqc7_fullperf`
  - `simTicks=57,076,565`, `kernel_ticks=53,462,955`
  - `MMAC active=27.5982%`
  - `MMOP=131,072`, `VALU=181,980`, `SCA=296,328`, `LDS=85,822`
  - `coissue=30,829/18,010`, `ldsBankConflict=0`
- xcu:
  - `s_abarrier_try_wait -> s_xor_b32` dropped from `47.46%` to `40.24%`
    versus the single-buffer baseline.
  - Top remaining window is `Raw1Used` (`barId 5`) and is still almost pure
    producer idle: window `15252:23376`, `Bubble=98.19%`, `MMAC=0%`.
- Conclusion: keep raw2 page-local tokens as a small positive step, but do not
  mistake it for a pipeline solution.  The producer still catches the consumer
  after two pages; the next design needs either more independent producer work
  during RawUsed waits or a larger/cleaner consumer MMAC island.

## 2026-07-04 Raw3 Rejection And Current Optimization Focus

- Raw3 page-local token stress was tried and reverted.
- Temporary route:
  `kRawBuffers=3`, `Raw2Filled/Raw2Used=6/7`, `AllDone=8`.
- Static/resource PASS:
  `private=0`, `sgpr=68`, `vgpr=112`, no scratch/spill.
- H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- H1/S1024 stats regressed versus raw2:
  `kernel_ticks=56,111,055`, `MMAC active=26.5078%`
  versus raw2 stats-only `kernel_ticks=53,300,975`,
  `MMAC active=27.6518%`.
- Decision: `REJECT_PERF_STATS_ONLY`.

Current active route remains raw2 canonical:

- 16 waves.
- `ActiveDkvTile = DkvTileD128MqNk128<64>`.
- `kRawBuffers=2`.
- Page-local raw ownership:
  `Raw0Filled/Raw0Used=2/3`, `Raw1Filled/Raw1Used=4/5`,
  `AllDone=6`.

Current focus:

- Reduce or remove avoidable `v_mov`, especially the runtime-hot groups that
  XCU counts as VALU/coissue but do not represent useful pipeline overlap.
- Do not count `v_mov` + MMAC overlap as successful coissue.  Useful coissue
  must be softmax/dS, sidecar/predicate/address work, or another real BWD
  operation hidden under peer MMAC.
- Use asm and XCU evidence for every change.  Raw page depth alone is not the
  next route to 60% MMAC active.

## 2026-07-05 Sidecar Ring3 Early Release Rejection

- Workbook sheet: `55_sidecar_ring3_early_raw_release`.
- Goal: make ReleasePage `RawUsed` arrive before softmax/dS without spilling
  sidecar rows into consumer VGPR.
- Design:
  - raw Q/dO remained two pages
  - sidecar used three LDS pages
  - consumer pre-read Q/dO source, released raw page, then read sidecar ring
    for softmax/dS
- Result:
  - direct `%3` version: correctness/resource PASS, but
    `kernel_ticks=54,754,245`, `MMAC active=26.7523%`
  - static sidecar-page-counter version: correctness/resource PASS, but
    `kernel_ticks=55,298,425`, `MMAC active=26.5015%`
  - raw2 recert baseline remained better:
    `kernel_ticks=53,008,410`, `MMAC active=27.7754%`
- Decision: `REJECT_PERF_STATS_ONLY`; source reverted to raw2 canonical.
- Lesson: sidecar lifetime can be fixed with an LDS ring, but the control/SCA
  cost is visible and does not solve the RawUsed critical path.  Do not retry
  page/ring-depth changes without a top-level reason that also increases useful
  MMAC island length or removes ownership/control work.

## 2026-07-05 Producer Sidecar Rebalance Rejection

- Workbook sheet: `56_producer_sidecar_rebalance`.
- Goal: test whether producer imbalance could be improved without adding
  synchronization by moving sidecar publication from producer0 to producer1.
- Temporary ownership:
  - producer0: K + Q
  - producer1: V + dO + sidecar
  - same two raw pages and same RawFilled/RawUsed tokens
- Static/resource:
  - branch windows moved from `6/198/198/1` to `1/198/198/6`
  - metadata `private=0`, `sgpr=58`, `vgpr=112`, no scratch/spill
- Correctness:
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_014728`
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_014734`
- Performance:
  - `kernel_ticks=53,558,960`, `MMAC active=27.5554%`
  - raw2 recert baseline was better:
    `kernel_ticks=53,008,410`, `MMAC active=27.7754%`
- Decision: `REJECT_PERF_STATS_ONLY`; code reverted.
- Lesson: sidecar ownership rebalance can make producer VGPR windows look
  balanced and raise coissue, but it does not shorten the RawUsed/consumer
  critical path.  Future producer-thickening must either hide useful work under
  consumer MMAC or eliminate an actual wait.

## 2026-07-05 Full-Valid Mask Shrink Rejection

- Workbook sheet: `57_full_valid_mask_shrink`.
- Goal: shorten causal full-valid softmax/dS by skipping the per-element
  `valid_pair` checks inside the canonical raw2 LDS-sidecar route.
- Temporary source change:
  - added a no-mask `softmax_ds_owner16_full_valid_from_lds_sidecar` helper
  - selected it from `consume_mq_mpair_owner16` when the whole owner16 M-pair
    was full-valid
  - left MMOP count, output ownership, ABarrier tokens, and LDS pages unchanged
- Static/resource:
  - PASS
  - branch windows `6/197/197/1`
  - metadata `private=0`, `sgpr=66`, `vgpr=112`, no scratch/spill
- Correctness:
  - H1/S128 failed at
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_015917`
  - PMD warning: `read vgpr165 before writing`
  - `dk_rel_l2=0.000361379`
  - `dv_rel_l2=33.2914`, `dv_max_abs=1.3782`, `pass=0`
- Decision: `REJECT_CORRECTNESS`; source reverted to raw2 canonical.
- Guardrail: do not remove per-element `valid_pair` from the main dKV
  softmax/dS helper again without a focused fragment/codegen probe.  The same
  class of dV corruption seen in older full-valid fastpaths reproduces on the
  current LDS-sidecar raw2 route.

## 2026-07-05 Sidecar Register Prefetch Rejection

- Workbook sheet: `58_sidecar_reg_prefetch_wait`.
- Goal: hide producer0 sidecar global reads inside the measured RawUsed wait
  window without changing ABarrier ownership or consumer math.
- Temporary source change:
  - split sidecar publication into a producer register-load helper and an LDS
    store helper
  - loaded `row_max_log2`, `row_inv_sum`, and `row_delta` before
    `wait_raw_used_page`
  - wrote sidecar values to LDS only after the raw page was free
- Static/resource:
  - PASS
  - branch windows unchanged at `6/198/198/1`
  - metadata `private=0`, `sgpr=60`, `vgpr=112`, no scratch/spill
- Correctness:
  - H1/S128 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_021112`
  - H1/S1024 PASS:
    `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260705_021118`
- Performance:
  - H1/S1024 `kernel_ticks=53,658,605`,
    `MMAC active ~=27.4726%`, coissue `30,915/18,119`
  - raw2 recert baseline remained better:
    `kernel_ticks=53,008,410`, `MMAC active=27.7754%`
- Decision: `REJECT_PERF_STATS_ONLY`; source reverted to raw2 canonical.
- Guardrail: do not carry tiny producer prefetch work just because it is
  correct and resource-clean.  Future producer-thickening needs enough
  independent work to change the critical path or a consumer-release redesign.

## 2026-07-06 K/V Latch Uniform Half-Select Rejection

- Active code remains the accepted `w16_mq128_kv_latch_wait_prune` route.
- A temporary instruction-level candidate moved `owner_nblock & 1` selection
  outside the K/V latch `d_block` loop with `readfirstlane`.
- Static asm reduced `v_cndmask` by 64, but added vector move/control work.
- H1/S128 and H1/S1024 correctness passed, but H1/S1024 stats regressed:
  `simTicks=48,266,855` versus baseline `47,871,005`, and MMAC active fell
  `32.7888% -> 32.6821%`.
- Decision: `REJECT_STATS_ONLY`; candidate code was reverted locally and
  synced out of `/zys/shaobo/fa3_bwd_wasp_clean`.
- Guardrail: static instruction-count cleanup is not enough.  For dKV
  instruction-level tuning, keep only changes that improve same-shape PMD
  stats and do not reduce MMAC active.

## 2026-07-06 Sidecar Pair Read6 Accepted

- Active code now includes `w16_mq128_sidecar_pair_read6` in the canonical dKV
  kernel.
- Change: `softmax_ds_owner16_causal_exact_tile_ctx` reads both M rows'
  sidecar Vec4 triples first, then computes both rows' softmax/dS.
- Static/resource PASS:
  producer0 `14/16`, consumer0 `189/240`, consumer1 `189/240`,
  producer1 `8/16`; metadata `private=0`, `sgpr=99`, `vgpr=128`,
  no scratch/spill.
- Correctness PASS:
  H1/S128 and H1/S1024.
- Full perf H1/S1024:
  `simTicks=47,731,775`, `kernel_ticks=44,118,165`,
  `MMAC active=32.8831%`, `VALU=168,514`, `SCA=115,544`,
  `ldsBankConflict=0`.
- Previous accepted wait-prune full perf:
  `simTicks=47,871,005`, `kernel_ticks=44,257,395`,
  `MMAC active=32.7888%`, `VALU=183,136`.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260706_215636_7gemm_sidecar_read6_h1s1024_sqc7_fullperf`.
- XCU note: top bubble is still ABarrier/control (`s_abarrier_try_wait ->
  s_xor_b32`, about `41.91%`), so next instruction-level work should target
  ownership wait exposure or matrix-read/wait gaps, not another sidecar-only
  reshuffle.

## 2026-07-07 dQ Producer Sidecar LDS Staging Accepted

- Branch: `shaobo/7gemm-dq-bringup`.
- Canonical dQ source: `src/dq_kernel.cpp`.
- Active tile/topology: `Mq=32,Nk=64,D=128`, 12-wave CTA.
- Current accepted dQ change: producer-side sidecar LDS staging.
- Resource gate:
  `private=0`, `sgpr=67`, `vgpr=168`, `sgpr_spill=0`, `vgpr_spill=0`;
  branch windows producer `8/40`, consumers `49/72`, worker `83/128`.
- Correctness:
  H1/S128 PASS at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_022646`;
  H1/S1024 PASS at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_022713`.
- H1/S1024 stats:
  dispatch0 `kernel_ticks=17,546,620`, `MMAC active=6.707%`;
  dispatch1 `kernel_ticks=28,114,905`, `MMAC active=9.707%`;
  `ldsBankConflict=0`.
- Target gap:
  current dQ is correctness/resource clean and faster, but still far below the
  `MMAC active >= 40%` target.  Next work should reduce page ownership waits
  by removing or sharing the K^T LDS page, not by adding another cosmetic
  instruction shuffle.

### Same-K-LDS Probe Note

- Four quick attempts to consume dQ K^T from the same K LDS page were rejected
  on H1/S128 correctness:
  `matrix_load_32x32 t + normal` (`rel_l2=1.03597`),
  `matrix_load_32x32 t + trans` (`rel_l2=1.46283`),
  `matrix_load_32x16 pairs + normal` (`rel_l2=0.535917`),
  and `matrix_load_32x16 pairs + trans` (`rel_l2=1.45385`).
- The active source is restored to `dq_sidecar_lds_staging`.
- Any future same-K attempt needs an isolated fragment-layout probe before
  touching the canonical dQ kernel again.

## 2026-07-07 dQ ABarrier And Nk128 Static Rejection

- Active dQ source remains `dq_sidecar_lds_staging`.
- Full perf/xcu for the active baseline:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_030342`,
  xcu
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2732630_fa3_bwd_dq_clean_20260707_030530`.
- xcu dispatch1 bottleneck:
  `s_abarrier_try_wait -> s_xor_b32` is `53.17%`; matrix-read wait is much
  smaller (`ds_read_matrix_trans_format -> s_waitcnt = 3.94%`).
- Metric guardrail: the 40% goal uses whole-active MMAC
  `sum(mmopRunTimeCounter) / sum(activeTimeCounter)`, not the local
  non-empty-SIMD ratio `sum(mmopRunTimeCounter) / sum(runTimeCounter)`.
- PageUsed consumer-only test:
  correctness PASS, whole-active MMAC `9.7068% -> 9.9346%`, but dispatch1
  ticks regressed `28,114,905 -> 28,360,605`; rejected and reverted.
- Nk128 single-page test:
  workbook sheet `14_dq_nk128_single_page`; build completed but static
  metadata failed with `private=68`, `sgpr_spill=2`, `vgpr_spill=64`.
  Rejected before PMD and reverted.
- Remote recert after revert:
  dQ gate PASS, symbol metadata PASS with `private=0`, `sgpr=67`,
  `vgpr=168`, no spill/scratch.

## 2026-07-07 dQ Matrixized q_subtile Probe Accepted

- Active performance kernel remains `dq_sidecar_lds_staging`
  (`Mq=32,Nk=64,D=128`, 12-wave CTA).
- New focused probe:
  `probes/dq_qsubtile_matrix_probe.cpp`.
- Purpose:
  test the Mq64 prerequisite ownership pattern outside the full kernel:
  repeated page0, two q_subtiles, `QDoUsed`, `Page0Filled`,
  `Page0DsFilled`, `Page0Used`, and `AllDone`.
- Matrix path:
  producer waves use `matrix_load_32x32_b16 ... bps lds` for Q/dO/K;
  worker waves use `ds_read_matrix_32x16_trans` to verify Q/dO row ownership.
- Static/resource PASS:
  `private=0`, `sgpr=31`, `vgpr=64`, no spill/scratch.
  ASM evidence includes `matrix_load_32x32_b16=10`,
  `ds_read_matrix=10`, `s_abarrier=35`, `s_set_vgpr_size=4`.
- PMD PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_matrix_probe_20260707_042323`,
  `errors=0`, `done_waves=12`, `pass=1`,
  `simTicks=10,955,945`, `ldsBankConflict=0`.
- Decision:
  accept as protocol evidence only.  Mq64 is reopened as a plausible
  40%-MMAC-active route, but the next main-kernel edit must be surgical:
  mirror the probe's QDo/page release order and keep one canonical dQ path.

## 2026-07-07 dQ Mq64 Main-Kernel Retry Rejected

- Tried to translate the accepted matrixized q_subtile probe back into the
  canonical dQ kernel with `Mq=64,Nk=64`.
- Static/resource stayed clean in all variants:
  `private=0`, `vgpr=168`, `sgpr=69..72`, no spill/scratch.
- Correctness did not complete:
  H1/S128 hung at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_043215`;
  H1/S64 also hung at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_043539`.
- Diagnostic variants:
  disabling producer QDo wait did not unhang S64;
  worker self-wait plus single-transition QDo avoided one phase-wrap hazard
  but still hung;
  a BRINGUP_ONLY q_subtile `__syncthreads()` boundary also still hung.
- ABarrier evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_mq64_s64_abar_debug_20260707_044204`
  exposed QDo phase wrap from fast worker waves; later logs still showed the
  full kernel stalling around the consumer `DsFilled`/dS publication path.
- Decision:
  `REJECT_HANG`.  The active source is restored to Mq32 sidecar-LDS and the
  remote build is recertified:
  branch windows producer `8/40`, consumers `49/72`, worker `83/128`,
  metadata `private=0`, `sgpr=67`, `vgpr=168`, no spill/scratch.
- Next Mq64 work:
  do not add more tokens inside the full kernel directly.  First write a
  focused worker+consumer+dS publication probe with unequal worker progress,
  `DsFilled`, dQ-like consumer work/release, and `PageUsed`; then port only a
  passing protocol back to the canonical kernel.

## 2026-07-07 dQ q_subtile dS Consumer Probe Accepted

- Added `probes/dq_qsubtile_ds_consumer_probe.cpp`.
- It covers the previously missing q_subtile pieces:
  unequal worker progress, dS-like LDS publication, `DsFilled`,
  consumer `ds_read_matrix` and dQ-like MMAC, `PageUsed`, `QDoUsed`,
  and `AllDone`.
- Static/resource PASS:
  `private=0`, `sgpr=23`, `vgpr=48`, no spill/scratch.
- ASM evidence:
  `matrix_load_32x32_b16=6`, `ds_read_matrix=10`, `v_mmac=4`,
  `s_abarrier=34`.
- PMD PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/qsubtile_ds_consumer_probe_20260707_050746`,
  `errors=0`, `done_waves=12`, `consumer_epochs=8`, `pass=1`,
  `simTicks=12,012,455`, `ldsBankConflict=0`.
- Implication:
  abstract q_subtile ownership with worker/consumer/dS is viable.  The full
  Mq64 dQ hang is now more likely in the real `dq_publish_ds_chunk` or
  `dq_consume_ds_kt_full_dtile` helper path, source-layout assumptions, or the
  compiled control structure around those helpers.

## 2026-07-07 dQ 40%-Active Evidence Update

- Active source remains the accepted `dq_pageused_consumer_only` Mq32 two-page
  kernel.  Local and remote code were restored after rejecting the worker wait
  merge candidate.
- Accepted baseline full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_082827/m5out/0/0/2739404_fa3_bwd_dq_clean.perf`.
- xcu output:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_pageused_s1024_fullperf_20260707_082827`.
- xcu shows the dominant issue gap is still
  `s_abarrier_try_wait -> s_xor_b32 = 46.01%`, with
  `s_abarrier_try_wait -> s_waitcnt = 6.60%`; MMAC hot-row share is only
  `3.17%`.  This makes the `PageFilled/DsFilled/PageUsed` handoff the next
  design target for the 40% MMAC-active goal.
- Rejected candidate:
  merging worker dS store waits into one page-level wait passed correctness and
  resource gates but regressed S1024 one-dispatch ticks
  `33,372,430 -> 33,729,150`; code was reverted.
