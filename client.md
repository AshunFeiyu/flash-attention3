# Client

## Mission

Build clean Shaobo FA3 BWD kernels in the FA3 FWD style.  The current preserved
dKV baseline remains the 7-gemm focused dKV line; dQ is now reopened on a
separate branch and must follow the same workbook-first discipline.  The main
optimization target is MMAC active share, with FA3 FWD as the hard benchmark.
Correctness, no scratch/spill, `ldsBankConflict=0`, and explainable SQTT
evidence are required before any performance claim.

## dQ Reopen Contract

- Active branch: `shaobo/7gemm-dq-bringup`.
- Design workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`.
- Scope: implement a standalone dQ kernel after the dKV focused line.
- Output ownership: Q tile owns dQ and stores once after reducing across all
  K/V tiles; no atomic add in the first dQ path.
- Algorithm boundary: because dKV and dQ are separate kernels, dQ may recompute
  score/dP across kernels, but must not duplicate score/dP for the same
  `(Q tile, K tile)` inside dQ.
- Current correctness-clean MMAC tile: `Mq=32,Nk=64,D=128,12 waves`, using
  source-layout `K^T` ABI.
- Current dQ pipeline baseline is a two-page K/V/Kt/dS conveyor:
  producer publishes page input, worker publishes dS, consumer computes dQ and
  releases the page.  Q/dO are loaded once per q-subtile.
- Direct `Mq=64` by serially looping two `M32` q-subtiles hung at H1/S128.
  Revisit only with explicit q-subtile token reset or a new lifetime proof.
- `Nk=128` is a later upgrade only after the same K LDS page can feed both
  normal and transpose dQ views without duplicating Kt/dS LDS.
- Producer rule: producer publishes Q/dO plus packed sidecar to LDS, and streams
  K/V through LDS; consumer should not direct-load sidecar global in the hot
  path.
- Evidence flow: design workbook -> code -> static gates -> H1/S128
  correctness -> H1/S1024 PMD/xcu diagnosis -> update workbook/ledger/log.

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
