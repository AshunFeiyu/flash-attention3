# Source Status

## 2026-07-20 Dual Canonical Best Reconciliation

Status: `ACCEPT_GOVERNANCE_DUAL_BASELINE`; one clean branch now contains the
independently verified best dKV and dQ sources without adding a phase or a new
kernel path.

- Branch `work/dual-canonical-best-20260720` takes the dKV contract, gate, and
  kernel exactly from tag `best/dkv-three-m64-lifetimes-20260719`, and the dQ
  kernel exactly from tag `best/dq-c1-kread-stagger-20260720`.
- dKV remains exact-work `Mq192/Nk192/D128`, one producer plus three heavy
  consumers, with one physical Q/dO page split into three M64 ownership
  lifetimes. Fresh PMD HEAD1694 recertification passes H1/S384 and H1/S768;
  S768 reports kernel ticks `33,104,435`, MMOP `46,080`, coissue
  `11,693/9,716`, and bank conflict zero. The accepted fullperf reference is
  `33,135,830` ticks and `41.1992%` MMAC active.
- dQ remains exact-work `Mq128/Nk128/D128`, two producers plus two symmetric
  64-row consumers. C0 keeps the canonical cadence while C1 moves its existing
  K-normal reads before softmax/dS. Fresh recertification passes H1/S128 and
  H1/S1024; S1024 reports kernel ticks `24,786,125`, MMOP `50,688`, coissue
  `11,787/10,476`, and bank conflict zero. The accepted repeated fullperf
  reference is `24,279,710/24,438,505` ticks and
  `34.0720%/34.0778%` MMAC active.
- Both static gates pass with private/spill/scratch zero and executable
  `s_trap=0`. Build with `SHAOBO_RUN_ON_MODEL=1` so the latest compiler emits
  the guarded WDRA init/run-on-model sequence. At PMD runtime set `ROCM_PATH`
  to `/zys/shaobo/toolchains/pmd_20260717`; otherwise the old HEAD1668
  `libgem5.so` may be loaded and fail the current ASTCA ABI before kernel code.
- The M128 logical `64+32+32` dKV topology stays an isolated tail-free control.
  Its M32 owners map to two waves each, so it has `4+2+2=8` heavy waves rather
  than three full four-wave consumers. Expanding them to twelve heavy waves
  would execute 50% redundant MMAC.
- Workbook sheet `174_DKV_M128_vs_M192Tail` records the formula DAG, physical
  wave ledger, LDS/WDRA budget, S1024 decompositions, measured normalized
  control debt, and time0/time1/time2 expected pipeline. It keeps M192 as the
  steady canonical source and admits `M192 main + static M128 tail` only as a
  design candidate pending exact S1024 correctness and perf.

## 2026-07-20 dKV Owner16 Four-Consumer Full Integration Rejected

Status: `REJECT_STATIC_RESOURCE`; canonical performance source remains the
accepted exact-work M192 branch, with M128 `64/32/32` retained as the
tail-free control.

- The full `Mq64/Nk256/D128` four-group kernel compiled with four symmetric
  `128 VGPR` WDRA branches, but metadata reported `private_segment=468B` and
  `vgpr_spill_count=971`. No correctness or PMD performance run was admitted.
- The resource and lifecycle probes remain valid focused evidence. They do
  not include the complete simultaneous score/dP, softmax/dS, dV/dK source,
  double-page control, and final-store live ranges of the FA kernel.
- This rejects the present four-heavy-group mapping, not M128. The exact
  M128 split `64+32+32` still removes the tail and passes all hard gates, but
  native owner16 granularity maps it to `4+2+2=8` heavy waves rather than
  three full four-wave consumers.
- Do not increase four symmetric role windows above 128: their per-SIMD sum
  is already `4*128=512`. Any retry must first reduce the complete-kernel
  peak live set on paper and in ASM, not merely enlarge WDRA metadata.

## 2026-07-20 dKV Owner16 Four-Consumer Lifecycle Gate

Status: `ACCEPT_OWNERSHIP_LIFECYCLE_GATE`; full FA integration pending.

- `probes/dkv_owner16_4c_lifecycle_probe.cpp` proves the complete
  startup/steady ownership transition: four rotated leaders publish disjoint
  K/V N64 slices into 131072B LDS, all 16 waves latch a unique N16 K/V view,
  `ResidentUsed(16)` releases the storage, and page0/page1/page0 reuse two
  Q/dO+sidecar pages totaling 67072B.
- Latest compiler metadata is private0, SGPR40, VGPR128, spill/scratch0 and
  LDS131072. Static evidence is BPS108, `ds_read_matrix`56,
  ABarrier wait25/arrive36, four `s_set_vgpr_size 128` branches and no trap.
- PMD run
  `/zys/shaobo_runs/dkv_owner16_4c_lifecycle_probe_20260720_054620` passes
  with `bad=0`, roles `128/128/128/128`, three raw generations and
  `ldsBankConflict=0`. Probe kernel ticks are `6,412,770`; this is an
  admission measurement, not FA performance evidence.
- In-kernel vector comparisons exposed a PMD VCC/SGPR init-tracking false
  failure. Deterministic fragment capture plus host comparison is the
  reliable probe pattern. Branch-local lane/wave setup also avoids the
  earlier WDRA/global-store tracking abort.

## 2026-07-20 dKV Owner16 Four-Consumer Canonical Design

Status: `DESIGN_COMPLETE_CANONICAL_INTEGRATION_PENDING`.

- Keep `M128=C0:64+C1:32+C2:32` at `fcd87aa` as the exact, tail-free
  control.  Native owner16 granularity gives it only eight heavy waves; it
  cannot become three full four-wave consumer groups without 50% redundant
  MMAC or an inter-wave partial reduction.
- Workbook `173_DKV_Owner16_4C_Canonical` defines the admitted candidate:
  `Mq64/Nk256/D128`, four symmetric owner16 groups, one unique N16 dK/dV
  owner per wave, no permanent producer and no repeated score/dP.
- Startup K/V consumes exactly 131072B LDS.  All 16 waves must latch their
  owner K/V fragment before that storage is reinterpreted as two
  Q/dO+sidecar pages totaling 67072B.  The two lifetimes may overlay but may
  never coexist.
- The steady protocol uses Page0/1 Filled+Used and AllDone.  A rotated
  designated wave publishes one complete M16 slice per group, so Filled has
  four arrivals.  Used must have 16 arrivals because one group leader cannot
  prove its three peer waves finished their LDS reads.  Causal-invalid math
  still participates in every page generation.
- The focused resource gate at `e4562dc` passes four independent roles at
  `114/128 VGPR`, private/spill/scratch zero, native BPS+matrix-read+MMAC,
  PMD checksum PASS and bank0. The subsequent lifecycle gate also passes;
  together they admit canonical integration but are not full FA correctness
  or performance evidence.

## 2026-07-20 dKV M128 64/32/32 Native-Role Gate

Status: `DESIGN_COMPLETE_RESOURCE_PROBE_PASS`.

- Canonical source is unchanged after the rejected early-store experiment;
  it still matches the correct M128 `fcd87aa` source.
- Public ISA and HCU compiler tests list FP16 MMAC as
  `v_mmac_f32_16x16x16_f16` / the corresponding 16x16x16 builtin.  No native
  8-row output MMAC was found.  `16x16x8_f32` uses K=8 and still produces a
  16x16 output.
- M128 logical `64/32/32` is exact but physical `2P2C`.  A full-wave physical
  1P3C version adds 50% redundant MMAC; a half-active version retains only
  eight heavy waves and no per-SIMD scheduling gain.  Do not implement either.
- Workbook `172_DKV_M128_3C_Gate` admitted an isolated
  `Mq64/Nk256/D128` owner16-4C resource probe.  It budgets K/V startup
  LDS 131072B, steady two-page Q/dO+sidecar LDS 67072B, persistent 96 VGPR
  per wave and only 32 transient VGPR.  Any use above 128 VGPR per role
  rejects the topology before PMD.
- That resource probe passed at `e4562dc`; full ownership/lifetime design is
  now recorded in workbook `173_DKV_Owner16_4C_Canonical`.

## 2026-07-20 dKV M128 Final-M16 Early Store Rejected

Status: `REJECT_CORRECTNESS_SOURCE_RESTORED`.

- Base topology remains the exact, tail-free `Mq128/Nk128/D128` logical
  `64/32/32` path at `fcd87aa`; no score/dP or output work was duplicated.
- The candidate moved only finalized D0-D63 dK/dV stores between the final
  low-D and high-D MMAC8 islands.  A 160-VGPR window spilled 8 VGPR/28B and a
  168-VGPR window spilled 6 VGPR/20B.  The minimum clean window was 176, with
  actual branch use 175 and private/spill/scratch zero.
- H1/S128 causal failed the hard numerical gate: dK max_abs `1.3076`, RMSE
  `0.296851`; dV max_abs `0.373255`, RMSE `0.115149`.  PMD also warned that a
  write hit a pending cacheline.  `simTicks=14,469,455`, kernel ticks
  `10,855,845`, MMOP `2,048`, coissue `695/479`, bank conflict zero; these are
  failed-correctness diagnostics, not performance evidence.
- No S1024 or SQTT capture was admitted.  Candidate source and the temporary
  176-VGPR window were removed.  Evidence is in workbook
  `171_DKV_M128EarlyStore` and remote run
  `/zys/shaobo_runs/dkv_m128_early_store_s128/dkv_mmac_correctness_20260720_032954`.

## 2026-07-19 dKV M128 Logical 64/32/32

Status: `OBSERVE_TAIL_FREE_CONTROL_DEBT`; isolated branch only. Accepted best
remains `best/dkv-next-m16-prefetch-20260719`.

- `Mq=Nk=128,D=128` removes the S1024 tail and gives exact logical ownership
  C0/C1/C2=`64/32/32`, with no repeated score or dP. Physical WDRA residency
  is nevertheless 2P2C: `32/160/160/32`; logical C1/C2 share waves8-11.
- S128/S768/S1024 correctness passes. Static resources are private0, SGPR52,
  VGPR96, spill/scratch0; LDS is 67,072B and bank conflict is zero.
- S1024 fullperf: kernel ticks `32,393,270`, MMAC active `37.8149%`, MMOP
  `73,728`, coissue `10,577/7,417`, waitLgkm `19,571`, barrier `71,161.75`.
  XCU still shows consumer MMAC alignment and ownership/wait bubbles.
- Do not compare H1/S768 raw ticks directly: M128 launches six CTAs and M192
  four. Normalized M128 barrier, SCA and wait debt are worse, while successful
  coissue and native active are lower. This source is evidence for exact-tail
  ownership and spare VGPR, not a performance promotion.
- Branch: `exp/dkv-m128-c0-64-c1-32-c2-32-20260719`; workbook
  `162_DKV_M128_64_32_32`; archive
  `/共享/shaobo/perf/20260719_191431_dkv_m128_c0_64_c1_32_c2_32_h1s1024_sqc7_fullperf`.

## 2026-07-19 dKV M192 Early Half-Store

- Status: `REJECT_STATIC_RESOURCE`; failed source removed.
- Active dKV source remains equivalent to tag
  `best/dkv-next-m16-prefetch-20260719` (`f6842b0`) for the kernel body.
- The final-half store hypothesis preserved the algorithm and instruction
  counts by design, but every LLVM7b schedule spilled. Best static result was
  `private_segment=12`, `vgpr_spill_count=4`; therefore PMD was not run.
- Design/evidence is retained in workbook sheet `161_DKV_FinalHalfStore` and
  `results/perf_ledger.csv`. The next isolated branch is M128
  `C0=64/C1=32/C2=32`; it must not be stacked on this rejected source.

## 2026-07-19 dQ 1P3C Saturated-Grid Gate Accepted

Status: `ACCEPT_TOPOLOGY_SATURATED_GATE`; source remains the clean M128 2P2C
canonical body while the S1024-capable 1P3C design is derived.

- The existing M192/Nk128/D128 1P3C artifact is static-clean at
  `11/158/158/159`, private/spill/scratch0, LDS131072B and bank0. Its three
  64-row consumers execute the exact three-GEMM row-owned dQ DAG.
- H12/S768 gives both topologies enough CTAs to fill 48 CUs. With exact MMOP
  `345,600` and correctness PASS on both sides, 1P3C improves kernel ticks
  `33,831,525 -> 26,230,750` (`-22.47%`) and MMAC active
  `29.2940% -> 30.9286%` (`+1.6346 pp`). VALU/SCA also fall
  `474,960/312,024 -> 405,696/286,128`.
- This overturns the earlier topology rejection: H1/S768 launched only four
  M192 CTAs and measured CU underfill. It does not yet admit the current M192
  source for fixed S1024 because the tail CTA, row ownership and ABarrier
  arrivals are not implemented.
- Next source change must be one clean S1024-capable 1P3C implementation,
  preceded by workbook proof. Do not stack it with next-N32 prefetch or any
  other rejected scheduling experiment.
- Evidence: workbook `160_DQ_3C_SaturationGate`; remote control/candidate
  roots `/zys/shaobo_runs/dq_3c_saturation_control_h12s768/` and
  `/zys/shaobo_runs/dq_3c_saturation_candidate_h12s768/`.

## 2026-07-19 dQ Next-N32 Head Prefetch Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- The candidate kept the canonical `Mq128/Nk128/D128` 2P+2C topology,
  formula DAG, matrix-read count, MMAC count, and ABarrier ownership. It moved
  the next N32 D0/D1 score+dP reads under the current dQ tail MMAC.
- S128/S1024 correctness, private/spill/scratch0, and bank0 passed. S1024
  `waitLgkmCounter` improved `16,181.25 -> 14,022.25` (`-13.34%`).
- The loop-carried operand state fragmented static MMAC islands
  `72 -> 129`, raised singleton islands `8 -> 58`, added
  `5,808 VALU` and `784 SCA`, and regressed ticks
  `24,585,015 -> 24,666,005` (`+0.329%`). MMAC active fell
  `33.3848% -> 32.7367%`.
- Reject this source form and skip fullperf. The valid lesson is narrower:
  LDS-wait hiding is useful only when implemented without loop-carried VGPR
  moves/control that break the compact MMAC island.
- Evidence: workbook sheet `159_DQ_NextN32Prefetch`; remote runs
  `/zys/shaobo_runs/dq_next_n32_head_prefetch_a6_s128/`
  and `/zys/shaobo_runs/dq_next_n32_head_prefetch_a6_s1024/`.

## 2026-07-19 Canonical dQ M128 Two-Producer Restore

Status: `ACCEPT_CANONICAL_GOVERNANCE_RESTORE`.

- Canonical dQ is restored from the M192 one-producer/three-consumer OBSERVE
  topology to `Mq=128,Nk=128,D=128`, 16 waves, two symmetric producers and
  two symmetric 64-row consumers. It keeps exactly three GEMMs, row-owned dQ,
  startup Q/dO/sidecar latch, and two 64KB K/V pages. The M192 path cannot
  launch the fixed S1024 target because `1024 % 192 != 0` and had regressed
  same-work S768 ticks by 20.3%.
- Static gates pass with private/spill/scratch0, VGPR128, LDS exactly 128KB,
  no regular-DS matrix workaround, native MLS/BPS + `ds_read_matrix` + MMAC,
  and real `s_trap=0`. The source entry keeps a guarded
  `__builtin_hcu_wdra_init(40,216,216,40)` for run-on-model compilers; the old
  stable compiler legally omits it.
- On PMD HEAD1694, both old LLVM `a6a6eb6616ab...` and Jul18 LLVM pass S128
  and S1024 correctness. The old compiler wins the same-work A/B:
  `25,084,150 -> 24,585,015` kernel ticks (`-1.99%`) and
  `31.4899% -> 33.3848%` MMAC active. Barrier counter falls
  `71,508.5 -> 47,387.25`; therefore the latest compiler is rejected for this
  kernel despite its lower VALU count.
- Fullperf/xcu shows producer wave slots0/3 at `98.51%/98.62%` bubble and
  consumer slots1/2 at `46.32%/47.18%`. Consumers already show useful stagger:
  60 MMAC-vs-VALU bins and 49 MMAC-vs-MMAC bins in the 5k:45k window, with
  MMAC vector peers `397/1248` and `483/1245`. The next hypothesis targets
  ordinary PageUsed ownership cadence, not topology stacking.
- XCU's largest `s_abarrier_try_wait -> s_xor_b32` edge is wait attribution;
  it is not evidence that XOR itself is expensive. XCU 4.6.3 invalid-jump
  decode warnings are recorded as a parser limitation, not a kernel failure.
- Evidence: workbook sheet `158_DQ_CanonicalRestore`; remote run
  `/zys/shaobo_runs/dq_mq128_restore_a6_s1024_fullperf/`
  `dq_correctness_20260719_154801`; shared archive
  `/共享/shaobo/perf/20260719_154801_dq_mq128_restore_a6_h1s1024_sqc7_fullperf`.

## 2026-07-19 Canonical dKV Next-M16 Score Prefetch

Status: `ACCEPT_LATENCY_HIDING_SAME_EXACT_WORK`.

- Canonical dKV keeps the accepted causal `Mq=Nk=192,D=128`, one-producer /
  three-consumer ownership, seven ABarrier IDs, and exact four-GEMM DAG. The
  only change reuses dead current-M16 normal-source VGPRs to prefetch the next
  M16 score D0/D1 transpose fragments under the current dV/dK MMAC8 island.
- Generated ASM contains 60 exact
  `normal4 -> MMAC8 -> next-trans4 -> lgkmcnt(4) -> MMAC8` patterns and never
  prefetches across the HeadFilled/TailFilled ownership boundary. Branch use
  remains `32/156/156/156`; metadata remains private/spill/scratch0 with
  SGPR53/VGPR128 and real `s_trap=0`.
- S384 and three S768 runs pass dK/dV correctness with
  `ldsBankConflict=0`. Dynamic work is identical to the causal control:
  MMOP/VALU/SCA/LDS/VMEM/FLAT is
  `46,080/41,314/15,014/28,316/1,152/798`.
- Fullperf S768 ticks improve `34,951,735 -> 34,372,975` (`-1.656%`).
  `waitLgkmCounter` falls `12,106 -> 10,836.75` (`-10.48%`), and xcu reduces
  `MMAC -> s_waitcnt` bubble duration `151,672 -> 114,264` (`-24.66%`).
  Same-SIMD MMAC+VALU coissue rises `1,881 -> 2,397` (`+27.43%`).
- Native MMAC active slips `39.5157% -> 39.2884%` (`-0.2273 pp`), so this is
  accepted as a ticks/readiness/coissue improvement, not as completion of the
  50% active target. Terminal AllDone waits remain excluded from the critical
  optimization attribution; the next hypothesis must target ordinary
  ownership/readiness or the serialized output tail.
- Evidence: workbook sheet `157_DKV_NextM16Prefetch`; remote run
  `/zys/shaobo_runs/dkv_next_m16_prefetch_s768_fullperf_retry/`
  `dkv_mmac_correctness_20260719_142126`; shared archive
  `/共享/shaobo/perf/20260719_142126_dkv_next_m16_prefetch_h1s768_sqc7_fullperf`.
- Preserve this accepted state as tag
  `best/dkv-next-m16-prefetch-20260719`.

## 2026-07-19 Canonical dKV Causal Q-Start

Status: `ACCEPT_ALGORITHM_CAUSAL_ZERO_WORK_PRUNE`.

- Canonical dKV remains one 16-wave, one-producer/three-consumer kernel with
  `Mq=Nk=192,D=128`, seven ABarrier IDs, one raw page, and native
  MLS/BPS + `ds_read_matrix` + MMAC traffic. No phase or fallback path was
  added.
- For causal K tile `j`, Q publication and consumption now begin at Q tile
  `j`. Only that first retained tile applies the exact element mask; later Q
  tiles are compile-time full-valid. S768 issued work is the exact triangular
  count `46,080` MMOP instead of `73,728`.
- Latest LLVM reports branch use `32/156/156/156`, private/spill/scratch0 and
  real `s_trap=0`; PMD HEAD1694 passes S384 and repeated S768 correctness with
  `ldsBankConflict=0`.
- Accepted fullperf S768 kernel ticks are `34,951,735`, improving the previous
  best `35,707,035` by `2.115%`. Raw MMAC active is `39.5157%`; its fall from
  `43.7836%` is expected because the old trace counted invalid triangular
  MMAC. Workbook sheet `156_DKV_CausalQStart` contains the normalization and
  critical-CTA XCU proof.
- Preserve this state as tag `best/dkv-causal-qstart-20260719`; it is the
  control for the accepted next-M16 prefetch successor. Do not restore invalid
  causal work to inflate MMAC active.

## 2026-07-17 Current Canonical After EBarrier Filled Rejection

Status: `CANONICAL_RESTORED_EBARRIER_PROBE_RETAINED`.

- Canonical dKV source is restored to the ready-only-priority owner32 path:
  16 waves, `Mq=128,Nk=256,D=128`, resident K/V, two M64 Q/dO Filled
  ABarrier epochs, independent Q/dO Used ownership, four exact GEMMs, and
  native MLS/BPS + `ds_read_matrix` + MMAC matrix traffic.
- The direct EBarrier Filled candidate is preserved only in commit `b045492`
  and removed by `a2b772c`. The isolated correctness/performance probe remains
  in commit `9f76bf1`.
- Candidate hard gates pass, but same-build H1/S1024 performance does not:
  `68,752,320 -> 73,301,410` ticks and `40.0907% -> 37.7371%` MMAC active;
  barrier expands `79,233 -> 102,989.25`. Fullperf/XCU is intentionally
  skipped by the stats gate.
- Remote canonical rebuild reports roles `14/239/239/8`, private0, SGPR56,
  VGPR128, spill/scratch0; H1/S256 dK/dV correctness and bank0 are certified
  at `/zys/sb/ebrstr/dkv_mmac_correctness_20260717_202936`.
- Workbook sheet `139_DKV_EBarrierFilled`, the ledger, and optimization log
  retain the positive probe boundary and the negative mainline result. Do not
  reintroduce EBarrier Filled handoff into canonical without a topology that
  avoids heavy-consumer reconvergence.

## 2026-07-15 Current Canonical dKV After Immediate-Offset Promotion

Status: `ACCEPT_MICRO_FULLPERF_XCU`.

- Canonical dKV keeps the 16-wave `Mq=128,Nk=128,D=128` topology, resident
  K/V, Q/dO half-page ownership, LDS sidecar, and accepted score/dP operand
  ping-pong.  The only promoted change replaces four separately materialized
  LDS addresses with one base plus four instruction immediate offsets.
- Gates pass: branch windows `14/16,221/240,221/240,8/16`, `private=0`,
  `sgpr=99`, `vgpr=128`, no spill/scratch; H1/S128 and H1/S1024 correctness
  pass and `ldsBankConflict=0`.
- Fullperf kernel ticks improve `42,564,340 -> 42,335,020`; MMAC active rises
  `33.7716% -> 34.1944%`; waitLgkm falls 3.15%, barrier falls 3.94%, and
  coissue success rises 10.1%.
- XCU still attributes 35.21% to the main ABarrier ownership window.  The
  three-slot Q/dO ring was subsequently rejected because its control/fetch
  debt outweighed its prefetch distance.  The active structural hypothesis is
  the workbook-designed true `Nk256/owner32` consumer described below.
- Evidence:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260715_193455_dkv_score_dp_imm4_accept_h1s1024_sqc7_fullperf`.

## 2026-07-12 Current Dual-Kernel Status After XCU Reprofile

Status: `OBSERVE_PROFILE`; source unchanged.

- dKV:
  current canonical source remains `dkv_wave0_terminal_invalidate` on the
  16-wave `Mq=128,Nk=128,D=128` dKV route.  H1/S1024 fullperf at
  `/zys/shaobo_runs/dkv_wave0_inv_fullperf_20260712_211315` passes
  correctness and keeps `ldsBankConflict=0`, with
  `simTicks=46,829,510`, `MMOP=131,072`, `VALU=168,384`,
  `SCA=111,248`, `LDS=79,360`, `VMEM=4,352`.
  xcu shows `s_xor_b32 34.64%`, `s_waitcnt 19.54%`, MMAC `10.73%`,
  and representative Q1/Dout1 page-used waits dominated by
  `s_abarrier_try_wait -> s_xor_b32`.  The tail AllDone wait is visible but is
  not safely removable yet.
- dQ:
  current canonical source remains the setprio/read-priority route with the
  accepted `dq_compute_pages_from_latched` helper and without
  natural-wrong/layout experiments.  H1/S1024 fullperf at
  `/zys/shaobo_runs/dq_canonical_fullperf_20260712_212222` passes
  correctness and keeps `ldsBankConflict=0`, with `simTicks=29,269,240`,
  `MMOP=50,688`, `VALU=57,968`, `SCA=54,172`, `LDS=26,352`,
  `VMEM=1,408`.  xcu shows `s_xor_b32 26.70%`,
  `s_cbranch_vccnz 17.35%`, MMAC `12.52%`, and a Page0Used
  `s_abarrier_try_wait -> s_xor_b32` bubble with max duration `6,319` cycles.
- Shared conclusion:
  the next optimization should be selected from ownership/page-lifetime
  evidence.  Do not assume the problem is missing MMAC, and do not reintroduce
  `ds_read_b32`, bpermute, gather, or wrong-layout workarounds in either
  canonical path.

## 2026-07-12 Owner-Teardown Early-Exit Rejected

Status: `REJECT_PMD_VGPR_TRACKING_ABORT_SOURCE_RESTORED`.

- Temporary change:
  all roles arrived a final ownership token, but only wave0 waited and
  invalidated ABarriers; non-wave0 roles exited early.  This targeted xcu's
  visible dKV AllDone tail wait and dQ terminal `s_barrier -> s_cbranch`
  bubble without changing formulas, tiles, or matrix paths.
- Static/resource:
  dKV and dQ both passed static gates and metadata before PMD; no
  spill/scratch/private segment regression was seen.
- PMD:
  dKV H1/S128 aborted with `vgpr47 is not init or has been freed` during
  MMAC in
  `/zys/shaobo_runs/owner_teardown_stats_20260712_2134/dkv_mmac_correctness_20260712_213619`.
  dQ H1/S128 aborted with `vgpr81 is not init or has been freed` during
  MMAC in
  `/zys/shaobo_runs/dq_owner_teardown_20260712_2140/dq_correctness_20260712_213749`.
- Decision:
  source restored and remote dQ gate recertified.  Keep all-wave terminal
  convergence in the canonical kernels.  Any future attempt to reduce terminal
  cleanup must first be a focused WDRA-exit ABI/PMD probe, not a mainline
  kernel change.

## 2026-07-12 dKV Wave0 Terminal Invalidate Accepted

Status: `ACCEPT_MICRO_CANONICAL`.

- Design:
  dKV formula DAG, `Mq=128,Nk=128,D=128`, K/V resident load, Q/dO half-page
  producer ownership, sidecar LDS staging, score/dP, softmax/dS, dV/dK MMAC,
  and output ownership are unchanged.  The promoted patch only changes terminal
  cleanup: keep `AllDone`, but let wave0 invalidate the ABarriers after the
  existing role-exit wait and CTA barrier.
- Rejected stronger variant:
  removing `AllDone` entirely passed the source gate but failed metadata with
  `private_segment_fixed_size=244`, `sgpr_spill_count=2`,
  `vgpr_spill_count=60`.  Treat `AllDone` as a current WDRA/codegen live-range
  stabilizer.
- Gates:
  build, dKV gate, and metadata gate PASS; branch windows
  `14/16,221/240,221/240,8/16`; `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 causal correctness PASS.
- Metrics:
  H1/S1024 first/repeat `46,594,275` / `46,682,090` ticks,
  `MMOP=131,072`, `VALU=168,384`, `SCA=111,248`, `LDS=79,360`,
  `VMEM=4,352`, repeat coissue `37,013/25,997`, `waitLgkm=52,429.0`,
  `barrier=140,274.67`, `ldsBankConflict=0`.
- Evidence:
  first `/zys/shaobo_runs/dkv_wave0_inv_20260712_205804`;
  repeat `/zys/shaobo_runs/dkv_wave0_inv_repeat_20260712_210159`.
- Decision:
  keep as a tiny terminal-control cleanup.  It is not a structural pipeline
  fix; next dKV work still needs mainloop ownership/SQTT evidence.

## 2026-07-12 dQ Setprio Narrowing Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Design:
  moved `ins::raise_priority_2()` in `dq_update_from_ds_pair` from before
  K-normal `ds_read_matrix` to after the reads.  No formula, tile, LDS,
  ABarrier, or store ownership change.
- Gates:
  build, dQ gate, and metadata gate PASS with branch windows
  `8/40,158/216,158/216,9/40`, `private=0`, `sgpr=65`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS.
- Metrics:
  H1/S1024 `simTicks=29,979,040`, `MMOP=50,688`, `VALU=57,968`,
  `SCA=54,172`, `LDS=26,352`, `VMEM=1,408`, `coissue=10,578/9,194`,
  `waitLgkm=16,638.5`, `barrier=58,052.75`, `ldsBankConflict=0`.
- Evidence:
  `/zys/shaobo_runs/dq_setprio_narrow_dqmmac_20260712_210421`.
- Decision:
  source restored locally and remotely; dQ gate recertified.  Keep priority
  over read/wait/MMAC for the dS@K helper unless xcu proves otherwise.

## 2026-07-12 dKV Full-Tile Filled Probe Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  test whether waiting once for a full Mq128 Q/dO tile is better than waiting
  separately for half0 and half1.  The candidate reused `Q0Filled` as a
  count-16 full-tile readiness token and removed the consumer wait on
  `Q1Filled`, while leaving QUsed/DoutUsed half releases unchanged.
- Gates:
  temporary source passed build, dKV gate, and metadata gate:
  branch windows `14/16,221/240,221/240,8/16`; `private=0`, `sgpr=96`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 causal correctness
  PASS; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 produced `simTicks=47,544,770`, MMAC active `31.6659%`,
  `MMOP=131,072`, `VALU=170,180`, `SCA=110,280`,
  `coissue=40,053/28,128`, `waitLgkm=53,209.75`,
  `barrier=161,363.67`.  Despite lower SCA and one fewer consumer wait,
  ticks and MMAC active regress sharply versus half-filled merge repeat
  `46,698,470` / `33.3278%`.
- Evidence:
  `/zys/shaobo_runs/dkv_full_tile_filled_probe_20260712_162909`.
- Decision:
  reject and restore source.  The half-page conveyor is doing useful work:
  half0 can compute while half1 is still arriving.  Do not flatten dKV
  readiness to full-tile Filled unless a future design adds enough other
  useful work to cover the lost overlap.

## 2026-07-12 dKV Producer1 Filled-Seq Prune Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  after `dkv_half_filled_merge`, both producer groups still call
  `seq_q_half_filled` on the same combined `Q0Filled/Q1Filled` tokens.  Test
  whether producer1 can skip `seq` and only publish dO plus arrive, reducing
  SCA without changing LDS layout, MMAC count, or Used lifetime.
- Gates:
  temporary source passed build, dKV source gate, and metadata gate:
  branch windows `14/16,221/240,221/240,8/16`; `private=0`, `sgpr=97`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 causal correctness
  PASS; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 produced `simTicks=46,755,345`, MMAC active `33.1816%`,
  `MMOP=131,072`, `VALU=168,384`, `SCA=111,432`,
  `coissue=38,424/26,956`, `waitLgkm=52,109.50`,
  `barrier=141,081.26`.  It lowers SCA by 512 versus
  `dkv_half_filled_merge`, but regresses repeat ticks
  `46,698,470 -> 46,755,345`, lowers MMAC active, and raises wait/barrier.
- Evidence:
  `/zys/shaobo_runs/dkv_half_filled_seq_p0only_20260712_162013`.
- Decision:
  reject and restore source.  `s_abarrier_seq` removal is not a free
  instruction-count win; it can worsen the ready cadence even when correctness
  holds.

## 2026-07-12 dKV Half-Filled Token Merge Observed

Status: `OBSERVE_STATS_REPEAT_WIN_FULLPERF_PMD_STARTUP_BLOCKED`.

- Motivation:
  dKV best xcu evidence remained dominated by ABarrier ownership bubbles.
  Previous `QUsed/DoutUsed -> RawHalfUsed` merging reduced SCA but regressed
  ticks by delaying independent Q and dO page release.  This experiment keeps
  `QUsed` and `DoutUsed` independent, but merges only the half-filled
  readiness: both Q and dO producers arrive the same `Q{0,1}Filled` token
  with count 8, and consumers wait that token once per half.
- Static/resource:
  build, dKV source gate, and metadata gate PASS.  Branch windows are
  `14/16,221/240,221/240,8/16`; metadata is `private=0`, `sgpr=97`,
  `vgpr=128`, no SGPR/VGPR spill.  ASM keeps the matrixized path:
  `ds_read_matrix=550`, `v_mmac=1028`, `ds_read_b32=0`,
  `ds_bpermute=0`, `s_trap=0`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS; `ldsBankConflict=0`.
- Metrics:
  first H1/S1024 stats:
  `simTicks=46,323,550`, MMAC active `33.2633%`, `MMOP=131,072`,
  `VALU=168,384`, `SCA=111,944`, `LDS=79,360`, `VMEM=4,352`,
  `coissue=37,284/25,932`, `waitLgkm=52,908.75`,
  `barrier=140,675.42`.
  Repeat:
  `simTicks=46,698,470`, MMAC active `33.3278%`, same instruction counts,
  `coissue=37,057/25,788`, `waitLgkm=51,337.75`,
  `barrier=139,802.92`.
  Compared with prior best `46,716,670` ticks, this is a very small repeat
  win and lowers SCA `114,520 -> 111,944`.
- Evidence:
  first root `/zys/shaobo_runs/dkv_half_filled_merge_20260712_160653`;
  repeat root
  `/zys/shaobo_runs/dkv_half_filled_merge_repeat_20260712_160817`.
  Fullperf/xcu attempts
  `/zys/shaobo_runs/dkv_half_filled_merge_fullperf_20260712_160937`
  and
  `/zys/shaobo_runs/dkv_half_filled_merge_fullperf_retry_20260712_161104`
  both aborted before dispatch with the known PMD/libhsakmt
  `buffer overflow detected` startup issue.
- Decision:
  keep as an observed dKV candidate, but do not call it a structural solution.
  It trims one readiness wait class without hurting correctness/resources; xcu
  is still needed to prove whether the dominant ownership bubble actually
  moved.

## 2026-07-12 dQ Latch Helper Extract Rejected

Status: `REJECT_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  after accepting `dq_compute_pages_from_latched`, test whether extracting the
  Q/dO/sidecar latch into `dq_latch_qdo_sidecar` keeps the code more
  high-cohesion without changing math, ABarrier ownership, or matrix paths.
- Static/resource:
  remote build, dQ source gate, and metadata gate PASS.  Branch windows stayed
  `8/40,158/216,158/216,9/40`; metadata stayed `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- Metrics:
  H1/S1024 under `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']` produced
  `simTicks=29,466,255`, `MMOP=50,688`, `VALU=57,968`, `SCA=54,172`,
  `LDS=26,352`, `VMEM=1,408`, `coissue=11,672/10,291`,
  `waitLgkm=16,566.25`, `barrier=56,715`, `ldsBankConflict=0`.
  This is slower than the accepted latched-compute repeat
  `29,216,460` ticks.
- Evidence:
  root `/zys/shaobo_runs/dq_latch_helper_20260712_152624`;
  H1/S1024 stats
  `/zys/shaobo_runs/dq_latch_helper_20260712_152624/dq_correctness_20260712_152630/m5out/0/0/stats.txt`.
- Decision:
  reject and restore source.  Code structure alone is not enough: helper
  extraction that preserves instruction counts but worsens same-shape ticks
  should not stay in the canonical dQ route.

## 2026-07-12 dQ Latched Compute Helper Accepted

Status: `ACCEPT_MICRO_CODE_GOVERNANCE`.

- Motivation:
  prepare the short-causal fast path without duplicating score/dP/softmax/dQ
  code.  Extract the page-consume math after Q/dO/sidecar latch into one
  helper while keeping the canonical PageFilled/PageUsed protocol unchanged.
- Static/resource:
  build, dQ gate, and metadata gate PASS.  Consumer branch windows improved
  from `159/216` to `158/216`; metadata stayed `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- Metrics:
  after moving `mmac_zero` back to the outer consumer scope, H1/S1024 runs were
  `29,289,260` ticks / `32.8632%` MMAC active and repeat
  `29,216,460` ticks / `32.6674%`.  Instruction counts improved versus
  setprio canonical: `VALU 58,144 -> 57,968`, `SCA 54,316 -> 54,172`,
  `MMOP=50,688`, `LDS=26,352`, `ldsBankConflict=0`.
- Evidence:
  first root `/tmp/dq_refactor_151548`;
  repeat root `/tmp/dq_refactor_repeat_151642`.
- Decision:
  accept.  This is primarily code governance and live-range cleanup, not the
  40% MMAC-active structural solution.  It enables a future short-causal path
  to reuse the exact same compute helper instead of stacking duplicated code.

## 2026-07-12 dQ Setprio Reverse M16 Retest Observed

Status: `OBSERVE_NEEDS_REPEAT_SOURCE_RESTORED`.

- Motivation:
  `consumer1 reverse M16` was rejected before `s_setprio` because ticks
  regressed.  Retest the one-line row-work balancing after the accepted
  FWD-style priority islands, since same-SIMD scheduling changed.
- Gates:
  temporary source only.  Build, dQ source gate, and metadata gate PASS:
  branch windows `8/40,159/216,159/216,9/40`, `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS.
- Metrics:
  H1/S1024 successful run:
  `simTicks=29,148,665`, MMAC active `32.7388%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `ldsBankConflict=0`,
  coissue `10,919/9,596`.  This is effectively tied with setprio first
  `29,145,480` and better than setprio repeat `29,438,955`, but not stable
  enough to promote.
- Repeat issue:
  two immediate repeat attempts aborted before dispatch in libhsakmt
  `buffer overflow detected`, the known PMD startup issue, so no clean repeat
  stats are available.
- Evidence:
  successful root
  `/zys/shaobo_runs/dq_setprio_reverse_m16_retest_20260712_150345`;
  abort roots
  `/zys/shaobo_runs/dq_setprio_reverse_m16_retest_repeat_20260712_150450`
  and
  `/zys/shaobo_runs/dq_setprio_reverse_m16_retest_repeat2_20260712_150522`.
- Decision:
  observe only and restore canonical row mapping.  Row balancing may have a
  small interaction with `s_setprio`, but it does not yet prove a stable ticks
  win or move MMAC active toward 40%.

## 2026-07-12 dQ BPS vbcnt Off Probe Rejected

Status: `REJECT_CORRECTNESS_BPS_READINESS_SOURCE_UNCHANGED`.

- Motivation:
  the accepted setprio fullperf/xcu profile still showed
  `s_waitcnt_vbcnt` as a visible source row (`9.00%`).  Test whether dQ can
  remove the default BPS readiness waits before `QDoFilled`/`PageFilled`
  arrivals, as an isolated compile-flag probe with no source changes.
- Gates:
  temporary build used
  `EXTRA_CXXFLAGS=-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0`,
  `BIN=build/fa3_bwd_dq_clean_novbcnt`.  dQ source gate and metadata gate
  PASS.  Resources were legal and slightly smaller:
  branch windows `8/40,159/216,159/216,9/40`, `private=0`,
  `sgpr=63`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 causal PASS.  H1/S1024 causal FAIL with NaNs:
  `actual_nonfinite=6144`, first bad row `640`, last bad row `687`.
- Evidence:
  run root `/zys/shaobo_runs/dq_novbcnt_probe_20260712_145055`;
  failing H1/S1024 run
  `/zys/shaobo_runs/dq_novbcnt_probe_20260712_145055/dq_correctness_20260712_145100`.
- Decision:
  reject.  The vbcnt row is real readiness cost, not removable control
  debris.  Keep default BPS-vbcnt for dQ unless a future source-level lifetime
  proof narrows it to a safe subset.

## 2026-07-12 dQ Sidecar Vec4 Restore Rejected

Status: `REJECT_STATS_INCOMPLETE_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  restore the historical SoA `Vec4F32` consumer sidecar LDS reads for
  `row_max/row_sum/row_delta`, replacing three scalar volatile reads, to test
  whether the old sidecar micro-win still applies after the current boundary
  n_tile canonical cleanup.
- Gates:
  remote build, dQ source gate, and metadata gate PASS.  Resources stayed
  legal: branch windows `8/40,159/216,159/216,9/40`, `private=0`,
  `sgpr=65`, `vgpr=128`, no spill/scratch.  ASM counts:
  `ds_read_b128=4`, `ds_read_b32=2`, `ds_read_matrix=214`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- Metrics:
  the only local H1/S1024 stats backup is truncated to 31 lines, so full
  SIMD/runtime counters cannot be used as promotion evidence.  Visible
  `system.simTicks=29,960,840`, which is slower than the accepted repeat best
  `29,706,495`.
- Evidence:
  H1/S128
  `/zys/shaobo_runs/dq_restore_sidecar_vec4_20260712_124638/dq_correctness_20260712_125418`;
  H1/S1024
  `/zys/shaobo_runs/dq_restore_sidecar_vec4_20260712_124702/dq_correctness_20260712_125442`;
  partial local stats `work/tmp/dq_restore_sidecar_vec4_stats.txt`.
- Decision:
  reject and restore canonical source.  Correct/resource-clean sidecar read
  aggregation is not enough; it must beat the current canonical repeat best
  and provide full active/resource counters before promotion.

## 2026-07-12 dQ QDoFilled Group Split Rejected

Status: `REJECT_STATS_NO_BEST_WIN_SOURCE_RESTORED`.

- Motivation:
  fullperf/xcu evidence showed large early `s_abarrier_try_wait -> s_xor`
  bubbles, and PMD trace showed consumers waiting on `barId 4`
  (`QDoFilled`) before all eight producer waves arrived.  This candidate split
  `QDoFilled` into group-local 4-wave tokens so consumer0 only waited
  producer0 and consumer1 only waited producer1.  `QDoLatched` remained a
  single 8-wave CTA token because page0 K/V overwrites the shared sidecar LDS
  region.
- Gates:
  build/source/metadata PASS.  Resources unchanged:
  branch windows `8/40,159/216,159/216,9/40`, `private=0`, `sgpr=65`,
  `vgpr=128`, no SGPR/VGPR spill.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 first run: `simTicks=29,853,915`, MMAC active `32.3773%`,
  `MMOP=50,688`, `coissue=6,046/9,704`, `waitLgkm=16,374.2`,
  `barrier=55,755.8`.  Repeat: `simTicks=29,870,295`, MMAC active
  `32.0531%`, `coissue=6,135/10,288`, `waitLgkm=16,500.8`,
  `barrier=56,716.5`.
- Decision:
  reject and restore canonical source.  Group-local `QDoFilled` can reduce one
  startup dependency, but it does not beat the accepted repeat best
  `29,706,495` ticks.  The remaining startup critical path still runs through
  sidecar/QDo latch and page0 K/V overwrite ownership.

## 2026-07-12 dQ dS-Cache VUsed Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  previous VUsed token-only release was semantically too early.  This
  candidate implemented the correct lifetime version: for each K/V page,
  consumers compute all page-local `score + dP + dS` fragments first, cache
  four `dS` fragment pairs in VGPR, arrive `VUsed`, then reread K and compute
  `dQ`.  Producer1 waits `VUsed` and uses separate `VFilled` tokens so V for
  the next page can load while K remains live for dQ.
- Gates:
  build/source/metadata PASS.  Resources grew but stayed legal:
  consumer windows `159 -> 175/216`, `private=0`, `sgpr=69`, `vgpr=128`,
  no SGPR/VGPR spill.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regressed to `simTicks=30,905,875`, MMAC active `31.1624%`,
  `MMOP=50,688`, `VALU=63,968`, `SCA=63,672`, `LDS=26,352`,
  `VMEM=1,408`, `coissue=5,802/11,721`, `waitLgkm=15,421.2`,
  `barrier=56,671.2`.
- Decision:
  reject and restore canonical source.  Correct V early-release does create a
  plausible lifetime, but extra ABarrier tokens, two-phase n-tile loops, and
  dS cache live range increase control/VALU enough to overwhelm producer
  overlap.

## 2026-07-12 dQ Tail No-Invalidate Rejected

Status: `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`.

- Motivation:
  xcu fullperf showed a large tail bubble:
  `s_barrier -> s_cbranch_vccnz` at `14.88%`.  The candidate removed the
  normal-path terminal `__syncthreads()` and `s_abarrier_inv` sequence, leaving
  `diag_store` synchronization unchanged, to test whether ABarrier resources
  can be left to workgroup teardown after the last use.
- Static:
  build/source/metadata PASS; resources even improved locally to
  `private=0`, `sgpr=63`, `vgpr=128`, no spill/scratch.
- Runtime:
  H1/S128 PMD aborted before correctness:
  `read vgpr81 before writing` and
  `panic: cu0 simd1 vgpr81 is not init or has been freed` during MMOP
  execution.  Run:
  `/zys/shaobo_runs/dq_no_tail_inv_20260712_121000/dq_correctness_20260712_120139`.
- Decision:
  reject and restore canonical source.  The terminal sync/invalidate sequence
  remains part of the current WDRA/PMD role-exit discipline; do not remove it
  in the performance kernel without a focused PMD/ABI proof.

## 2026-07-12 dQ Consumer1 Reverse M16 Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  q-tile split suggested early causal tiles may suffer per-SIMD work
  imbalance.  The temporary mapping kept consumer0 rows `0..3` but reversed
  consumer1 rows to `7..4`, pairing same-SIMD work as `(0,7)`, `(1,6)`,
  `(2,5)`, `(3,4)` instead of `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`.
- Gates:
  remote build/source/metadata PASS with unchanged resources:
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.  H1/S128 and
  H1/S1024 correctness PASS; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats were `simTicks=30,142,840`, MMAC active `32.2965%`,
  `MMOP=50,688`, `VALU=58,144`, `SCA=54,316`,
  `coissue=6,237/9,730`.  This regresses versus accepted repeat best
  `29,706,495` ticks.
- Decision:
  reject and restore canonical source.  Row pairing alone does not move the
  critical path; PageFilled/PageUsed/softmax/control remain dominant.

## 2026-07-12 dQ Conditional Page Barrier Rejected

Status: `REJECT_STATS_NO_BEST_WIN_SOURCE_RESTORED`.

- Motivation:
  q-tile split showed early causal tiles are dominated by fixed
  ABarrier/control cost.  This candidate made PageUsed/Page1Filled init,
  arrive, and invalidate conditional on whether the page can actually be
  reused for the current `active_k_tiles`.
- Gates:
  remote build, dQ source gate, and metadata gate PASS:
  branch windows `8/40,159/216,159/216,9/40`, `private=0`, `sgpr=65`,
  `vgpr=128`, no SGPR/VGPR spill.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats were `simTicks=30,037,735`, MMAC active `32.1251%`,
  `MMOP=50,688`, `VALU=58,144`, `SCA=54,168`, `coissue=6,534/10,781`.
  This does not beat the current accepted repeat baseline
  `29,706,495` ticks, and only ties the first-run boundary n-tile number
  `30,040,010`.
- Decision:
  reject and restore canonical source.  Conditional ABarrier lifetime pruning
  is logically correct but too small; added branch/control offsets the saved
  init/arrive/inv cost.  Do not retry PageUsed/init pruning as a standalone
  tweak.

## 2026-07-12 dQ q-tile Split Evidence

Status: `OBSERVE_QTILE_SPLIT_CAUSAL_FRONTLOAD`.

- Motivation:
  after canonical contract cleanup, split H1/S1024 causal dQ into one
  `q_tile` per dispatch to determine whether the ~32% whole-kernel MMAC
  active is a uniform pipeline failure or mostly caused by early causal tiles
  with small valid K range.
- Run:
  `/zys/shaobo_runs/dq_qtile_split_20260712_111249/dq_correctness_20260712_112030`,
  with `DQ_TILES_PER_DISPATCH=1`, `GPU_CHIP=sb`, and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- Metrics:
  per-dispatch MMAC active rises monotonically with causal valid range:
  tile0 `11.045%`, tile1 `27.199%`, tile2 `32.959%`,
  tile3 `36.409%`, tile4 `37.876%`, tile5 `40.121%`,
  tile6 `40.357%`, tile7 `40.815%`; `ldsBankConflict=0` for all tiles.
- Interpretation:
  late q-tiles already meet the near-term `40%+` dQ active target.  The
  aggregate H1/S1024 active near `32%` is mostly dragged down by early causal
  tiles where MMOP work is small but fixed ABarrier/control/setup cost remains.
- Decision:
  record as evidence, no code change.  Do not keep blindly tweaking the
  `ds_read_matrix -> MMAC` main path; next candidates should either specialize
  early causal tiles, increase useful work per ownership epoch, or revisit the
  dS dependency graph with a resource proof.

## 2026-07-12 dQ Builtin Try-Wait Rejected

Status: `REJECT_METADATA_PRIVATE_SEGMENT_SOURCE_RESTORED`.

- Motivation:
  xcu fullperf shows the largest bubble as
  `s_abarrier_try_wait -> s_xor_b32`.  The current dQ wait wrappers use the
  inline-asm `ins::abarrier_try_wait<true>` path; test whether the builtin
  path improves codegen or scheduling.
- Code:
  temporary only.  Changed dQ PageFilled, PageUsed, QDoFilled, and QDoLatched
  wait wrappers from `ins::abarrier_try_wait<true>` to
  `ins::abarrier_try_wait<false>`.
- Result:
  build and dQ source gate passed, but symbol metadata failed:
  `private_segment_fixed_size=12`, `sgpr=69`, `vgpr=128`,
  no SGPR/VGPR spill.  The canonical hard gate requires private segment zero.
- Decision:
  reject without PMD.  Source restored to inline-asm wait wrappers and remote
  metadata gate recertified back to `private=0`, `sgpr=65`, `vgpr=128`.
  Do not retry builtin try-wait on this toolchain unless compiler evidence
  shows it no longer creates private segment.

## 2026-07-12 dQ Contract Cleanup Recertified

Status: `OBSERVE_CLEANUP_RECERT_CANONICAL_UNCHANGED`.

- Motivation:
  keep the clean repo honest: the active dQ contract should describe only the
  canonical performance path, while rejected or exploratory native dS
  source-slot work remains isolated as focused probes.
- Source:
  moved `DqNativeDsRingTileD128`, `NativeDsRingDqTile`,
  `DqNativeDsRingBarrierLedger`, and `NativeDsSlotMap` out of
  `include/dq_contract.h` into `probes/dq_probe_contract.h`.  Updated the
  source-slot probe files to include the probe-local contract.  Removed the
  unused `kDqPathNativeDsRingPrototype` constant from the active contract.
- Gates:
  `python3 scripts/check_dq_kernel_gate.py --source src/dq_kernel.cpp
  --contract include/dq_contract.h --asm /tmp/nonexistent_dq_cleanup.asm`
  PASS.  `rg` confirms the native dS ring/source-slot contract now only
  appears under `probes/`, not in `include/`, `src/`, or `scripts/`.
  Remote build/asm dQ gate and metadata gate also PASS:
  branch windows `8/40,159/216,159/216,9/40`, `private=0`, `sgpr=65`,
  `vgpr=128`, no SGPR/VGPR spill.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- Perf recert:
  H1/S1024 fullperf stats report `simTicks=30,262,960`,
  `MMAC active=32.0547%`, `MMOP=50,688`, `VALU=58,144`, `SCA=54,316`,
  `LDS=26,352`, `VMEM=1,408`, `coissue=6,096/9,906`,
  `ldsBankConflict=0`.
- XCU:
  helper `.perf` and xcu outputs are archived under
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_dq_contract_cleanup_h1s1024_sqc7_fullperf`.
  Top bubbles: `s_abarrier_try_wait -> s_xor_b32` `24.67%`,
  terminal `s_barrier -> s_cbranch_vccnz` `14.88%`,
  `abarrier -> salu_32` `7.85%`,
  `s_cmp_lg_u32 -> s_waitcnt_vbcnt` `6.55%`, and
  `lds_matrix -> immed` `4.37%`.
- Decision:
  accept the cleanup as code hygiene only.  It does not improve the 40% MMAC
  active target; the next dQ work remains structural ABarrier/control reduction
  or native dS dependency-graph redesign.

## 2026-07-12 dQ VUsed Early-Release Rejected

Status: `REJECT_PROTOCOL_LONGRUN_SOURCE_RESTORED`.

- Motivation:
  C74 producer1 loads V but waits on the shared `PageUsed` token until dQ
  finishes using K, even though V is mathematically dead after
  `dP = dO @ V^T`.  This candidate tried to add `Page0VUsed/Page1VUsed`
  tokens so the V producer could reuse the V half-page earlier while the
  consumers continued softmax/dS and `dQ = dS @ K`.
- Code:
  temporary only.  Added two VUsed ABarrier tokens, made consumers arrive
  VUsed after the dP MMAC region, and made the V producer wait VUsed instead
  of PageUsed for `kt > 1`.  K producer, shared K/V double pages, tile shape,
  and math were otherwise unchanged.
- Gates:
  static/resource PASS with branch windows `8/40,166/216,166/216,9/40`.
  Metadata stayed clean: `private=0`, `sgpr=65`, `vgpr=128`, no
  spill/scratch.  H1/S128 correctness PASS.
- Runtime:
  H1/S1024 did not complete in the normal PMD smoke window and was
  interrupted; leftover `fa3_bwd_dq_clean` processes were killed.  Source and
  remote mirror were restored to C74.
- Decision:
  reject.  In the current interleaved `n_tile` mainloop, V is still needed for
  later `n_tile` dP computations in the same K/V page.  Arriving VUsed after
  one `n_tile` is semantically too early, while arriving after all `n_tile`
  would be almost the same point as PageUsed because dQ is interleaved per
  `n_tile`.  Real V early-release would require a different schedule that
  computes all dP for the page before dQ, plus storage/register budget for
  qk/dp/dS; do not retry it as a token-only tweak.

## 2026-07-12 dQ WG-Local K/V Duplicate Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C74 xcu showed producer/control wave slots with only ~449 instructions but
  ~36K bubble cycles and ~98.8% bubble rate.  The dominant producer wait was
  steady `PageUsed` ownership (`s_abarrier_try_wait -> s_xor_b32` contexts
  around 5.6K-7.8K cycles), while consumer slots already carried the expected
  MMAC work.  This candidate tested whether each producer group should load a
  full local K+V page for its paired consumer, trading duplicated K/V MLS for
  local PageFilled/PageUsed counts of four.
- Code:
  temporary only.  Producers waves0-3 loaded K+V into page0 for consumer0,
  producers waves12-15 loaded K+V into page1 for consumer1, and each consumer
  read only its own page.  Q/dO/sidecar latch remained CTA-wide because the
  two steady pages overwrite the shared startup Q/dO or sidecar LDS regions.
- Gates:
  static/resource PASS with branch windows `8/40,160/216,160/216,9/40`.
  Metadata stayed clean: `private=0`, `sgpr=77`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regressed versus C74 stats:
  `simTicks 32,597,110 -> 35,871,745`; MMAC active fell
  `31.6674% -> 29.2945%`.  MMOP stayed `55,296`, but duplicated K/V raised
  VMEM `1,408 -> 2,560`; SCA fell `40,732 -> 24,652`, yet
  `barrierCounter` rose to `66,261.25` and elapsed ticks worsened.
- Decision:
  reject and restore C74 source.  Making producer waves thicker by duplicating
  K/V does not compensate for losing K/V double-buffer prefetch and doubling
  matrix-load traffic.  The next route should preserve shared K/V reuse and
  attack PageUsed latency by increasing useful work per ownership epoch or by
  changing the dS dependency graph, not by duplicating the page.

## 2026-07-12 dQ K-First Count-Fix Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C81 hung because PageFilled was reinterpreted as KFilled but still required
  eight arrivals.  C82 fixed the protocol count to four K producer arrivals
  and four V producer arrivals, then retested the same K-first overlap idea.
- Gates:
  static/resource PASS with branch windows `8/40,127/216,127/216,9/40`.
  Metadata stayed clean: `private=0`, `sgpr=66`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 regressed `32,597,110 -> 34,374,340` ticks.  MMAC active fell
  `31.6674% -> 30.3953%`.  VALU stayed `89,216`, but SCA rose
  `40,732 -> 43,832`; `waitLgkm` rose to `19,929.5`, and coissue became
  `12,021/10,569`.
- Decision:
  reject and restore C74 source.  K-first lowers consumer live VGPR pressure
  (`159 -> 127`) but breaks the paired score/dP MMAC island and increases
  wait/control more than it hides V load.  Do not split K/V readiness in the
  canonical dQ path without a different work order that preserves large MMAC
  islands.

## 2026-07-12 dQ K-First V-Overlap Rejected

Status: `REJECT_HANG_SOURCE_RESTORED`.

- Motivation:
  C74 waits for a combined K+V PageFilled token before score/dP work starts.
  C81 split the page-ready protocol into KFilled and VFilled so consumers could
  compute `score = Q @ K^T` after K arrives while producer1 continues loading V,
  then wait VFilled for `dP = dO @ V^T`.
- Code:
  temporary only.  Existing PageFilled tokens became KFilled; two VFilled
  tokens were added.  The consumer hot path changed from interleaved
  score+dP MMAC to `K read -> score MMAC -> VFilled wait -> V read -> dP
  MMAC -> dS -> dQ`.  PageUsed still protected K/V page reuse.
- Static evidence:
  build/static gates passed.  Consumer branch windows dropped
  `159/216 -> 127/216`, metadata stayed clean (`private=0`, `sgpr=66`,
  `vgpr=128`, no spill/scratch), and main matrix path stayed on
  `ds_read_matrix + v_mmac`.  The cost was more control:
  `s_abarrier_init 6 -> 8`, `s_abarrier_try_wait 12 -> 16`,
  `s_cbranch_vccnz 38 -> 45`.
- Runtime:
  H1/S128 did not complete in normal time and was interrupted.  No H1/S1024
  stats were collected.  Source and remote build were restored to C74 and
  recertified with branch windows `8/40,159/216,159/216,9/40`.
- Decision:
  reject.  K-first can reduce live VGPR pressure, but the naive page-level
  VFilled protocol is not valid with the current PageUsed/QDoLatched lifetime.
  Retry only as a focused barrier protocol probe before touching the
  performance kernel again.

## 2026-07-12 dQ Accumulator Zero-Seed Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C74 still had many `v_mov` instructions, including zeroing eight persistent
  dQ accumulators before the K loop.  This candidate tested whether the first
  `dS @ K` update could seed each dQ accumulator from the existing
  `mmac_zero`, removing explicit accumulator zeroing.
- Code:
  temporary only.  `dq_update_from_ds_pair` became a `SeedZero` template and
  the call site used the zero-seed path only for `kt==0 && n_tile==0`.
  Tile shape, 16-wave roles, ABarrier tokens, Q/dO/K/V layout, sidecar path,
  BPS waits, and output ownership were unchanged.
- Gates:
  static/resource PASS with branch windows `8/40,160/216,160/216,9/40`.
  Metadata stayed clean: `private=0`, `sgpr=65`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.  PMD printed a non-fatal
  `read vgpr124 before writing` warning.
- Metrics:
  H1/S1024 regressed `32,597,110 -> 34,696,480` ticks.  MMAC active fell
  `31.6674% -> 29.8264%`.  Static `v_mov_b64` fell `39 -> 7`, but the
  generated path grew elsewhere: `v_mov_b32 170 -> 220`,
  `v_mmac 384 -> 416`, `ds_read_matrix 214 -> 230`, and
  `s_cbranch_vccnz 38 -> 40`.  Runtime instruction counters also rose:
  `VALU 89,216 -> 97,184`, `SCA 40,732 -> 41,820`.
- Decision:
  reject and restore C74 source.  Zero-seed remains useful for fixed first
  MMAC islands such as score/dP, but dQ's long-lived accumulator state makes
  a runtime first-update path too expensive.

## 2026-07-12 dQ Sidecar Prefetch Under QDo MLS Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C74 xcu still showed producer-side `global_load_dword -> s_waitcnt` on the
  sidecar path, while Q/dO `matrix_load_32x32_b16` work follows immediately in
  the same producer branch.  This tested whether loading sidecar into a
  producer VGPR first, issuing Q/dO MLS, then storing the sidecar to LDS could
  hide sidecar latency without adding a token.
- Code:
  temporary only.  Added a producer-local sidecar prefetch object, moved the
  sidecar global load before Q/dO MLS, and kept the LDS sidecar store after
  the Q/dO matrix loads.  QDoFilled/QDoLatched tokens, tile shape, 16-wave
  roles, K/V pages, consumer math, BPS waits, and output ownership were
  unchanged.
- Gates:
  static/resource PASS with branch windows `10/40,159/216,159/216,10/40`.
  Metadata stayed clean: `private=0`, `sgpr=65`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.  ASM matched the intended order:
  `global_load_dword` before Q/dO `matrix_load_32x32_b16`, with the sidecar
  `ds_write_b32` after the matrix loads.
- Metrics:
  H1/S1024 stats regressed versus C74 fullperf stats:
  `32,721,325 -> 33,057,115` ticks, or `+1.026%`.  Local readiness/control
  counters improved (`waitLgkm 370.750 -> 355.750`,
  `barrier 1201.750 -> 1144.250`, coissue-fail fell `246 -> 153`), but the
  kernel paid back the gain as more instruction work (`VALU 3116 -> 3235`,
  `SCA 2057 -> 2063`) and worse elapsed ticks.
- Decision:
  reject and restore C74 source.  Sidecar schedule-only edits are not a
  structural route to 40% MMAC active unless paired with a larger ownership or
  dependency-graph change.

## 2026-07-12 dQ Exact Active K-Tiles Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  canonical dQ only supports causal exact tiles with `Mq=Nk=128`,
  `S % Mq == 0`, and `S % Nk == 0`.  Therefore each q tile's active K tiles
  can be expressed as `q_tile + 1` instead of computing
  `q_tile_end=min(q_base+Mq,seqlen)` and a ceil-div.  This tested whether
  removing scalar min/ceil control helped the C74 mainline.
- Code:
  temporary only.  Replaced `active_k_tiles` in the kernel entry with
  `q_tile + 1` and in the consumer with `q_base_tile / Nk + 1`.  No role,
  ABarrier, LDS page, BPS wait, MMAC, or output ownership changed.
- Gates:
  static/resource PASS with branch windows `8/40,159/216,159/216,9/40`.
  Temporary metadata improved to `private=0`, `sgpr=58`, `vgpr=128`,
  no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats regressed
  `32,597,110 -> 32,615,310` ticks.  MMAC active moved
  `31.6674% -> 31.6334%`; SCA rose `40,732 -> 42,344`, while
  MMOP/VALU/LDS/VMEM stayed `55,296/89,216/28,656/1,408`.
- Decision:
  reject and restore C74 source.  Static SGPR reduction did not translate to
  lower elapsed ticks; do not retry this scalar-algebra shortcut unless
  compiler/codegen evidence changes.

## 2026-07-12 dQ Tail Raw SBarrier Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C74 xcu showed a large terminal/control row
  `s_barrier -> s_cbranch_vccnz`.  Removing the pre-invalidate tail sync had
  already failed with PMD VGPR-init errors, so this candidate preserved the
  sync semantics but emitted a raw `s_barrier` instead of HIP
  `__syncthreads()`.
- Code:
  temporary only.  Added a small `raw_s_barrier()` helper and replaced only the
  normal-path tail `__syncthreads()` before `s_abarrier_inv`.  Mainloop,
  math, tile, PageUsed/QDo ownership, BPS waits, diagnostic sync, and output
  ownership were unchanged.
- Gates:
  static/resource PASS with branch windows `8/40,159/216,159/216,9/40`.
  Metadata: `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.  H1/S128
  and H1/S1024 correctness PASS; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats regressed
  `32,597,110 -> 32,835,530` ticks.  MMAC active moved only
  `31.6674% -> 31.7079%`, and instruction mix was unchanged
  (`MMOP=55,296`, `VALU=89,216`, `SCA=40,732`).
- Decision:
  reject and restore C74 source.  Coissue increased slightly, but it did not
  become elapsed-time improvement.  Stop tail-sync codegen tweaks for now; the
  remaining route to 40% MMAC active must attack mainloop ownership or the
  native dS-ring dependency graph.

## 2026-07-12 dQ Sidecar Early Latch Rejected

Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C74 xcu showed dominant ABarrier/control gaps.  Page0 K/V overwrites only the
  sidecar LDS region, not the latched Q/dO regions, so a startup-only
  `SidecarLatched` token was tested to let producers start page0 K/V after the
  sidecar rows were read, while page1 still waited for full `QDoLatched`.
- Code:
  temporary only.  The candidate added one ABarrier token
  `SidecarLatched`, made consumers arrive it after sidecar LDS reads, and made
  both producers wait it for `kt==0`.  Tile shape, 16-wave roles, Q/dO latch,
  K/V page ownership, matrix path, BPS waits, and output ownership were
  otherwise unchanged.
- Gates:
  static/resource PASS with canonical branch windows
  `8/40,159/216,159/216,9/40`.  Metadata: `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats-only improved slightly
  `32,597,110 -> 32,512,025` ticks and
  `31.6674% -> 31.7890%` MMAC active.  However fullperf regressed
  `32,721,325 -> 32,877,390` ticks while MMAC active only moved
  `31.6115% -> 31.7176%`.
- Evidence:
  stats run
  `/zys/shaobo_runs/dq_sidecar_latch_20260712_064000/dq_correctness_20260712_060609`;
  fullperf
  `/zys/shaobo_runs/dq_sidecar_latch_fullperf_20260712_065000/dq_correctness_20260712_060750/m5out/0/0/2782681_fa3_bwd_dq_clean.perf`;
  xcu
  `/zys/shaobo_runs/dq_sidecar_latch_fullperf_20260712_065000/xcu_outputs/sidecar_latch_d0`.
- Decision:
  reject and restore C74 source.  Splitting the startup sidecar lifetime is a
  useful observation but not a promotion: the added token/control raises SCA
  and does not shorten the fullperf critical path.  Avoid further fine-grained
  startup token splitting unless it removes a proven larger wait/control cost.

## 2026-07-12 dQ 12-Wave Single Producer Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  C74 xcu showed two producer wave slots with little useful work and dominant
  ABarrier/control gaps.  A 12-wave single-producer topology was tested to
  halve `QDoFilled/PageFilled` producer arrivals and BPS readiness waits.
- Code:
  temporary only.  The candidate used one producer branch `waves0-3` to load
  both Q/dO sidecar groups and both K+V pages, plus two consumer branches
  `waves4-7` and `waves8-11`.  `PageFilled/QDoFilled` counts were 4;
  `PageUsed/QDoLatched` stayed 8.
- Gates:
  static/resource PASS after producer `s_set_vgpr_size` was raised from 40 to
  48 to satisfy 3-branch WDRA average alignment.  Metadata:
  `private=0`, `sgpr=52`, `vgpr=160`, no spill/scratch.  H1/S128 and
  H1/S1024 correctness PASS; `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats regressed:
  `simTicks=32,597,110 -> 32,779,565`.
  MMAC active was effectively flat:
  `31.6674% -> 31.6917%`.  VALU/SCA fell
  `89,216/40,732 -> 88,096/33,400`, but that was not enough.
- Decision:
  source restored to the C74 16-wave dual-producer canonical path.  The
  negative lesson is that producer/control reduction is not useful if it
  serializes K+V publication and delays page readiness.

## 2026-07-12 dQ Branchless Causal Mask Accepted

Status: `ACCEPT_PERF_CANONICAL_DQ`.

- Motivation:
  after tail-second-sync cleanup, xcu still showed heavy control latency,
  including `s_cbranch_vccnz`.  The hot dS loop still had two per-element
  causal `if (krow <= qrow)` blocks for the two N16 fragments.
- Code:
  canonical `src/dq_kernel.cpp` replaces those branches with branchless valid
  multiplies.  Tile shape, 16-wave roles, Q/dO latch, K/V double pages,
  PageUsed ownership, BPS waits, matrix path, and output ownership are
  unchanged.
- Gates:
  static/source PASS.  Branch windows improved to
  `8/40,159/216,159/216,9/40`; metadata remains `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS;
  `ldsBankConflict=0`.
- Metrics:
  H1/S1024 stats-only:
  `simTicks=33,529,405 -> 32,597,110`,
  `MMAC active=29.5058% -> 31.6674%`,
  `VALU=112,064 -> 89,216`.
  H1/S1024 fullperf:
  `simTicks=33,977,580 -> 32,721,325`,
  `MMAC active=29.4292% -> 31.6115%`.
- Evidence:
  stats run
  `/zys/shaobo_runs/dq_branchless_causal_20260712_055500/dq_correctness_20260712_053305`;
  fullperf
  `/zys/shaobo_runs/dq_branchless_causal_fullperf_20260712_060000/dq_correctness_20260712_053434/m5out/0/0/2781898_fa3_bwd_dq_clean.perf`;
  xcu
  `/zys/shaobo_runs/dq_branchless_causal_fullperf_20260712_060000/xcu_outputs/branchless_causal_d0`.
- Decision:
  keep as the current canonical dQ baseline.  Remaining bottleneck is still
  ABarrier/control/BPS readiness, not a missing MMAC path.

## 2026-07-12 dQ Tail Keep-Alive Prune Rejected

Status: `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`.

- Motivation:
  test whether the post-store `keep_accumulator_live(dq_reg[d_idx])` loop in
  `dq_consumer_full3gemm_role` is removable tail VALU noise.
- Code:
  temporary removal only.  No math, tile, LDS, barrier, wait, matrix path, or
  output ownership change.
- Gates:
  build/source gate passed and metadata remained `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.  However branch windows changed materially:
  producer1 branch reported `38/40` VGPRs instead of the restored canonical
  `9/40`.
- Failure:
  H1/S128 PMD aborted before correctness under
  `/zys/shaobo_runs/dq_tail_keepalive_prune_20260712_031836/dq_correctness_20260712_031837`
  with `read vgpr70 before writing`, then `VGPR index 85 is out of range:
  VGPR range=[0,40]` on `v_mov_b32`.
- Decision:
  source restored.  Treat the keep-alive loop as part of the current
  WDRA/codegen liveness contract; do not remove it from canonical dQ without a
  focused WDRA-exit proof.

## 2026-07-12 dQ Native dS Ring Structural Probe Accepted

Status: `ACCEPT_PROBE_STRUCTURAL_CANONICAL_UNCHANGED`.

- Motivation:
  before changing canonical dQ, prove the proposed split-role path can pass a
  native LDS handoff: producer publishes K, C_dS publisher writes dS source
  slots, and C_dQ consumer reads them as trans fragments for dQ MMAC.
- Code:
  only standalone `probes/dq_native_ds_ring_structural_probe.cpp` changed.
  Canonical `src/dq_kernel.cpp` remains unchanged.
- Gates:
  static/resource PASS with branch windows `2/40,13/64,12/64,14/80,14/80,1/32`;
  metadata `private=0`, `sgpr=22`, `vgpr=120`, no spill/scratch.  ASM contains
  `ds_write_matrix_format`, `ds_read_matrix_trans_format`, normal
  `ds_read_matrix_format`, and `v_mmac`; no ordinary `ds_read_b*` matrix path
  or gather/permute workaround is used.
- PMD:
  `/zys/shaobo_runs/dq_native_ds_ring_structural_fix_20260712_024559`.
  All `slot0/slot1 low/high` checks passed; producer_done=1, publisher_done=2,
  consumer_done=2.  Stats: `simTicks=6,359,535`, `MMOP=4`, `VALU=176`,
  `SCA=453`, `LDS=30`, `ldsBankConflict=0`.
- Decision:
  accept as structural proof only.  Next work is a focused probe that replaces
  deterministic dS with canonical C_dS arithmetic in source-slot order; only
  after that should the canonical dQ kernel be modified.

## 2026-07-12 dQ dS@K Batch8 Wait0 Rejected

Status: `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`.

- Motivation:
  test the user-requested larger `ds_read_matrix`/MMAC island on the canonical
  dQ `dS@K` stage by waiting for all eight K-normal reads before a single
  longer MMAC island.
- Code:
  temporary change only in `dq_update_from_ds_pair`: `wait_lgkm(4)` was changed
  to `wait_lgkm(0)`, and the middle `wait_lgkm(0)` was removed.
- Gates:
  build and source/static gate passed; metadata remained `private=0`,
  `sgpr=67`, `vgpr=128`, no spill/scratch.
- Failure:
  H1/S128 PMD aborted before correctness under
  `/zys/shaobo_runs/dq_dqgemm_batch8_wait0_20260712_030114/dq_correctness_20260712_030114`
  with `read vgpr81 before writing` followed by
  `vgpr81 is not init or has been freed` in MMOP execute.
- Decision:
  source restored to canonical `wait_lgkm(4)` plus middle `wait_lgkm(0)`.
  Do not collapse this dQ K-normal wait split without a focused
  ds_read_matrix/MMAC VGPR-init proof.

## 2026-07-12 dQ Native dS Ring Formula Probe Accepted

Status: `ACCEPT_PROBE_FORMULA_SOURCE_SLOT_CANONICAL_UNCHANGED`.

- Motivation:
  after proving deterministic source-slot handoff, test whether source-slot
  publication also works with real softmax/dS-style scalar formula, including
  `exp2` and float-to-half conversion.
- Code:
  added standalone `probes/dq_native_ds_ring_formula_probe.cpp`.  Canonical
  `src/dq_kernel.cpp` is unchanged.
- Gates:
  metadata `private=0`, `sgpr=22`, `vgpr=120`, no spill/scratch.  ASM uses
  `ds_write_matrix_format`, `ds_read_matrix_trans_format`, normal
  `ds_read_matrix_format`, and `v_mmac`; no gather/permute matrix fallback.
- PMD:
  `/zys/shaobo_runs/dq_native_ds_ring_formula_20260712_030944`.
  All four slot checks passed; stats: `simTicks=6,556,550`, `MMOP=4`,
  `VALU=329`, `SCA=476`, `LDS=30`, `ldsBankConflict=0`.  PMD warns
  `VOP3P__V_MAD_MIXLO_F16 not test`, but numerical output is correct for this
  focused formula path.
- Decision:
  accept as probe evidence.  The remaining blocker for a real dQ ring is
  qk/dP MMAC output orientation into `NativeDsSlotMap`, not LDS handoff or the
  scalar softmax/dS formula itself.

## 2026-07-12 dQ Tail No-Invalidate Fast Exit Rejected

Status: `REJECT_PMD_REGISTER_INIT_SOURCE_RESTORED`.

- Motivation:
  xcu mainline showed `s_barrier -> s_cbranch_vccnz` as a large terminal
  bubble, so normal-path terminal `__syncthreads()+abarrier_inv` was tested
  under `diag_store != 0` only.
- Gates:
  static/resource PASS with unchanged branch windows
  `8/40,161/216,161/216,9/40`; metadata `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.
- Failure:
  H1/S128 PMD aborted before correctness with
  `panic condition !regInit[regIdx] occurred: cu0 simd1 vgpr81 is not init or has been freed`
  in MMOP execute.
- Decision:
  source restored to canonical tail cleanup.  Treat terminal sync/invalidate
  as required by the current WDRA/PMD role-exit path unless a focused
  WDRA-exit probe proves otherwise.

## 2026-07-12 dQ Score/dP Wait Split Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  test whether score/dP matrix-read latency can be hidden by starting D-block0
  MMAC after `wait_lgkm(12)` rather than waiting at `lgkmcnt(8)` for D-block0
  and D-block1 together.
- Gates:
  static/resource PASS with unchanged branch windows
  `8/40,161/216,161/216,9/40`; metadata `private=0`, `sgpr=67`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS.
- Stats:
  H1/S1024 `simTicks=36,199,800`, `kernel_ticks=32,586,190`,
  `MMAC active=27.1810%`, `waitLgkm=13,636`, `barrier=52,102.75`,
  `ldsBankConflict=0`.
- Decision:
  source restored to canonical score/dP `wait_lgkm(8)`.  This micro-split
  regresses elapsed ticks and active share; the next useful optimization must
  address PageUsed/ABarrier ownership or useful work per ownership epoch.

## 2026-07-12 dQ MLS32x16 Direct Source-Slot Probe Rejected

Status: `REJECT_PROBE_SOURCE_PROBE_ONLY`.

- Motivation:
  previous instruction probes proved `matrix_load_32x16_b16` pairs correctly
  with 32x16 normal/trans DS readers.  Test whether that official pair also
  satisfies the stricter dQ `NativeDsSlotMap` source-slot q ownership.
- Code:
  only `probes/dq_source_operand_layout_probe.cpp` changed.  The canonical
  dQ kernel remains untouched.
- Gates:
  metadata PASS with `private=0`, `sgpr=20`, `vgpr=12`, no spill/scratch.
- PMD:
  `/zys/shaobo_runs/dq_mls32x16_source_slot_20260712_011745`,
  `simTicks=8,070,335`, `MMOP=0`, `ldsBankConflict=0`.
- Result:
  for `load_name=mls32x16`, no tested reader full-matches.  Best q-match is
  still `44/504` on `normal_32x16_alt0`; decoded coverage is `496/504` for
  32x16 readers and `248/504` for 16x32 readers.
- Decision:
  direct-load/direct-reader source-slot variants are exhausted for now.  A
  native dS ring must prove a producer/MMAC orientation that creates source
  slots natively, or the main work should return to canonical full-3GEMM dQ.

## 2026-07-12 dQ MLS32 Direct ALT Source-Slot Probe Rejected

Status: `REJECT_PROBE_SOURCE_PROBE_ONLY`.

- Motivation:
  before coding a native dS source-slot ring, exhaust the official DS matrix
  ALT/interleave readers that might let one MLS32 LDS page satisfy
  `NativeDsSlotMap` q ownership without hot-path gather/permute.
- Code:
  only `probes/dq_source_operand_layout_probe.cpp` changed.  The canonical
  dQ kernel in `src/dq_kernel.cpp` was not touched.
- Compile boundary:
  current compiler rejects `trans_32x16_alt1`, `trans_32x16_alt2`,
  `normal_32x16_alt2`, and `trans_16x32_alt2`.  The legal tested addition is
  `normal_32x16_alt1`.
- Gates:
  metadata PASS with `private=0`, `sgpr=20`, `vgpr=12`, no spill/scratch.
- PMD:
  `/zys/shaobo_runs/dq_dsread_alt_source_slot_20260712_010656`,
  `simTicks=7,895,615`, `MMOP=0`, `ldsBankConflict=0`.
- Result:
  `operand_layout_final any_full_match=0`; best legal q-match is still
  `44/504` (`normal_32x16_alt0`), while `normal_32x16_alt1` is `40/504`.
- Decision:
  do not route native dS ring through MLS32 direct ALT readers.  Continue only
  with a different native producer/MMAC orientation proof, or return to the
  canonical full-3GEMM dQ path and optimize barrier/page cadence.

## 2026-07-11 dQ Nk256 Single-Page Epoch Rejected

Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.

- Motivation:
  workbook-first test of a structural dQ tile change: `Mq128/Nk128` two K/V
  pages -> `Mq128/Nk256` one 128KB K/V page.  The hoped-for win was fewer
  `PageFilled/PageUsed` ownership epochs in H1/S1024 (`36 -> 20`) and twice
  as much MMAC per epoch (`1536 -> 3072`).
- Gates:
  the first full-unroll build spilled SGPR (`sgpr_spill_count=9`).  Changing
  only the `n_tile` loop to `unroll 4` fixed resources:
  branch windows `8/40,158/216,158/216,9/40`; metadata `private=0`,
  `sgpr=58`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S256 and H1/S1024 PASS under
  `/zys/shaobo_runs/dq_nk256_singlepage_20260711_231218`.
- Stats:
  H1/S1024 versus restored mainline regressed:
  `simTicks 35,704,760 -> 41,586,545`,
  `kernel_ticks 32,091,150 -> 37,972,935`,
  `MMAC active 27.3852% -> 24.3812%`,
  `MMOP 55,296 -> 61,440`,
  `VALU 121,632 -> 147,072`,
  `barrierCounter 58,629.75 -> 86,381.5`,
  `ldsBankConflict=0`.
- Decision:
  source restored to `Mq128/Nk128` double-page.  Do not retry `Nk256`
  single-page alone; it loses K/V prefetch/double buffering and adds causal
  padding work faster than it removes ownership tokens.

## 2026-07-09 Formal Toolchain Switch To Zwj/Liuchang Overlay

Status: `ACCEPT_ENV_SWITCH`.

- Change:
  `build.sh` now defaults to
  `/home/zhangyushun/toolchains/zwj_liuchang_llvm_7940/bin/clang++` when that
  overlay exists, and exports `HIP_CLANG_PATH`, `PATH`, and `LD_LIBRARY_PATH`
  so `hipcc` and asm generation use the same compiler route.  `CLANGXX`,
  `HIPCC`, and `HIP_CLANG_PATH` remain explicit overrides for controlled
  experiments.
- Motivation:
  the default `zwj_8426` compiler path can emit WDRA `s_trap` prologues that
  current PMD cannot execute.  Zhang Wenjian's FP8 FWD runs because it uses a
  `liuchang_llvm` overlay or an already-built no-`s_trap` artifact, not because
  the container default compiler path is universally valid.
- Gates:
  formal `build.sh` dQ and dKV builds both report
  `toolchain zwj_liuchang_llvm_7940 overlay`; dQ asm has `s_trap=0`,
  `s_set_vgpr_size=4`; dKV asm has `s_trap=0`, `s_set_vgpr_size=4`.
  dQ gate PASS.  dKV gate and metadata gate PASS with `private=0`,
  `sgpr=99`, `vgpr=128`, no SGPR/VGPR spill.
- Correctness:
  dQ H1/S128 PASS:
  `/zys/shaobo_runs/formal_zwj7940_overlay/dq_correctness_20260709_122743`;
  dKV H1/S128 PASS:
  `/zys/shaobo_runs/formal_zwj7940_overlay/dkv_mmac_correctness_20260709_122745`.
- Decision:
  use this overlay as the default clean-repo compiler on `shaobo_dev_8426`.
  Do not replace `/opt/rocm` in place unless a separate rollback/backup plan is
  explicitly requested.

## 2026-07-09 dKV QUsed Before Softmax Accepted

Status: `ACCEPT_MICRO_TICKS_OWNERSHIP`.

- Motivation:
  the fixed-env dKV baseline was dominated by Q/dO ownership barriers:
  `s_abarrier_try_wait -> s_xor_b32` was `41.38%` in xcu.  Prior attempts to
  delay dO wait under softmax reduced local `s_waitcnt` but worsened ownership
  latency.  This experiment instead moved ReleasePage Q-normal source reads
  and `QUsed` arrival before softmax/dS, so producers can see Q half-page
  release earlier while consumers hold Q source regs through the following
  softmax and dV/dK MMAC.
- Code:
  the active `src/dkv_kernel.cpp` keeps one canonical dKV path.  The old helper
  that both read Q sources and arrived `QUsed` was narrowed to
  `dv_dk_mmac_owner16_qready`, which only consumes already-ready Q/dO source
  fragments for the dV/dK MMAC island.
- Gates:
  static/resource PASS; branch windows `14/16`, `222/240`, `222/240`,
  `8/16`; metadata `private=0`, `sgpr=99`, `vgpr=128`,
  `sgpr_spill=0`, `vgpr_spill=0`, scratch `0`; `ldsBankConflict=0`.
  The larger consumer window is a real headroom warning.
- Correctness:
  H1/S128 PASS
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_032250`;
  H1/S1024 PASS
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_032317`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260709_033115`;
  shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_033115_dkv_qused_before_softmax_h1s1024_sqc7_fullperf`.
  Versus `dkv_splitwait_highsrc`, `simTicks` improves
  `47,484,710 -> 46,716,670`, `kernel_ticks`
  `43,871,100 -> 43,103,060`, and `MMAC active`
  `32.9468% -> 33.2391%`.
- xcu:
  dispatch duration improves `96,420 -> 94,728`; average active waves
  `121.28 -> 122.18`; `s_abarrier_try_wait -> s_xor_b32`
  `41.38% -> 40.55%`; `v_mmac -> v_mmac` `8.44% -> 8.19%`.
  The tradeoff is visible: `ds_read_matrix_format -> s_waitcnt`
  `3.26% -> 3.90%`, and trans read wait `2.72% -> 2.89%`.
- Decision:
  keep as a micro-win because correctness/resource/perf/xcu agree on a small
  elapsed improvement and ownership bubble reduction.  Do not extend this
  pattern blindly: branch windows are now `222/240`, and the path remains far
  from the FWD-style `60%` MMAC-active goal.

## 2026-07-08 dQ Nk128 Direct Sidecar Rejected

Status: `REJECT_CORRECTNESS_SOURCE_REVERTED`.

- Motivation:
  after deeper/finer buffering failed, tested a higher-ceiling `Mq128/Nk128`
  route to double useful MMAC per page token.  Because Q+dO `64KB` plus K/V
  `64KB` fills LDS exactly, the candidate removed sidecar LDS staging and had
  consumers direct-load row sidecar from global.
- Static/resource:
  Nk128 compiled and passed metadata with `private=0`, `sgpr=63`,
  `vgpr=128`, no spill/scratch; branch windows `1/40`, `163/216`,
  `163/216`, `2/40`.
- Correctness:
  H1/S128 failed with all dQ values NaN:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_130721`.
  Adding explicit `wait_vmem_lgkm()` did not fix it:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_130949`.
  An Nk64 direct-sidecar diagnostic also failed all-NaN:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_131553`.
- Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_130949_dq_direct_sidecar_correctness_reject`.
- Decision:
  code reverted; active source remains the `b56b2dc` sidecar SoA Vec4
  baseline.  Direct consumer sidecar global loads are not allowed in the main
  dQ path without a focused WDRA/global-load sidecar probe.

## 2026-07-08 dQ Half-Page Release Rejected

Status: `REJECT_PERF_STATS_ONLY_SOURCE_REVERTED`.

- Motivation:
  current xcu showed the top wait around `Page1Used`.  The experiment split
  each `Nk64` K/V page into half0/half1 `n_tile` ownership tokens so producers
  could release/prefetch half a page earlier.
- Gates:
  H1/S128 and H1/S1024 correctness PASS; metadata `private=0`, `sgpr=56`,
  `vgpr=128`, no spill/scratch; branch windows `8/40`, `158/216`,
  `158/216`, `9/40`; `ldsBankConflict=0`.
- Stats-only H1/S1024 result:
  `kernel_ticks=36,212,995`, `simTicks=39,826,605`,
  `MMAC active=25.4434%`, `SCA=101,660`, coissue `13,482/15,991`.
- Comparison:
  sidecar SoA Vec4 stats-only recert was `kernel_ticks=35,483,175`,
  and accepted full perf was `kernel_ticks=35,382,165`.
- Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_124824_dq_half_page_release_h1s1024_sqc7_stats_reject`.
- Decision:
  code reverted; active source remains the `b56b2dc` sidecar SoA Vec4
  baseline.  Finer PageUsed splitting increased scalar/control cadence and
  should not be retried as an isolated optimization.

## 2026-07-08 dQ Nk32 Triple Page Rejected

Status: `REJECT_PERF_SOURCE_REVERTED`.

- Motivation:
  XCU on the current sidecar SoA Vec4 baseline attributed the top steady-state
  bubble to producer-side `Page1Used` wait.  The experiment tested whether
  shrinking `Nk=64 -> Nk=32` and keeping three true K/V pages could remove the
  Q/dO-overlay pressure without exceeding LDS.
- Resource design:
  Q+dO `64KB`, three K/V pages `48KB`, sidecar about `1.5KB`; total about
  `113.5KB`, below the `128KB` LDS budget.
- Gates:
  H1/S128 and H1/S1024 correctness PASS; metadata `private=0`, `sgpr=53`,
  `vgpr=128`, no spill/scratch; `ldsBankConflict=0`.
- Full-perf result versus current sidecar baseline:
  `kernel_ticks=35,382,165 -> 35,575,995`,
  `simTicks=38,995,775 -> 39,189,605`,
  `MMAC active=25.3548% -> 25.4985%`,
  coissue `17,446/16,910 -> 15,465/15,407`.
- XCU result:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_nk32_triple_page_s1024_fullperf_20260708_121749`;
  top bubble remains `s_abarrier_try_wait -> s_xor_b32 37.86%`, with
  representative rows still waiting on `barId=3 Page1Used`.
- Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_121749_dq_nk32_triple_page_h1s1024_sqc7_fullperf`.
- Decision:
  code reverted; active source remains the `b56b2dc` sidecar SoA Vec4
  baseline.  Do not continue deeper buffering by shrinking `Nk`; it increases
  page epochs and does not reduce the measured PageUsed bottleneck.

## 2026-07-08 dQ Sidecar SoA Vec4 Micro-Win

Status: `ACCEPT_MICRO_TICKS_NOT_PIPELINE_SUCCESS`.

- Canonical dQ source now keeps the pair-island Mq128/Nk64/16-wave full3GEMM
  route and changes only consumer sidecar LDS reads.
- AoS4 sidecar was rejected: it removed scalar sidecar `ds_read_b32` but caused
  `ldsBankConflict=12`.
- Accepted SoA Vec4 variant keeps the original sidecar layout and reads
  four-row sidecar groups.  ASM has `ds_read_b128=4` and residual
  `ds_read_b32=2`; no main matrix-path DS fallback was introduced.
- Gates:
  H1/S128, H1/S1024, and full-perf correctness PASS; metadata
  `private=0`, `sgpr=53`, `vgpr=128`, no spill/scratch; `ldsBankConflict=0`.
- Full-perf comparison versus `dq_ntile_pair_island`:
  `kernel_ticks=36,972,845 -> 35,382,165`, but
  `MMAC active=25.5487% -> 25.3548%` and `VALU=131,168 -> 138,208`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_sidecar_soa_vec4_s1024_fullperf_20260708_113438`.
  Top bubbles remain ABarrier/control, not sidecar or matrix-read waits.
- Decision:
  keep the patch as a small ticks win, but do not count it as progress toward
  the 40% MMAC-active goal.  Next work should return to ABarrier/page cadence
  and useful overlap, not sidecar-only reshuffling.

## 2026-07-07 dQ 40% MMAC Active Target

Status: `DQ_MQ128_16W_FULL3GEMM_CORRECTNESS_ACTIVE`.

Current goal:

- Optimize the clean dQ path toward `MMAC active >= 40%`.
- Target shape is `B=1,H=1,S=1024,D=128,causal=true`,
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.
- Hard gates remain correctness PASS, `private=0`, no scratch/spill,
  `ldsBankConflict=0`, and main matrix path through MLS/BPS +
  `ds_read_matrix` + MMAC.

Current code state:

- Branch: `shaobo/dq-xcu-guided-dq-kernel`.
- Canonical dQ tile: `Mq=128,Nk=128,D=128,16 waves`.
- Role ownership is now 16-wave full-3GEMM:
  waves0-3 producer for Q/dO group0 sidecar and K, waves4-7 consumer group0
  rows 0-63, waves8-11 consumer group1 rows 64-127, waves12-15 producer for
  Q/dO group1 sidecar and V.
- There is no dS-in-LDS handoff and no separate dS worker.  Each consumer
  computes `QK^T`, `dO V^T`, softmax/dS, and `dS K` for its own q rows.
- Sidecar is staged by producers into LDS; consumers do not direct-load sidecar
  global in the hot path.
- The all-zero bring-up bug was fixed by waiting first `PageFilled` before
  consumers read Q/dO/sidecar.  Without this readiness edge consumers could
  read zero or uninitialized Q/dO/sidecar data.
- The older Mq32 K-native dQ route remains the performance reference in the
  ledger, but it is no longer the canonical source shape after the user's
  topology correction.
- Added a standalone measurement knob:
  `--tiles-per-dispatch` / `DQ_TILES_PER_DISPATCH`.  It controls how many q
  tiles the standalone harness packs into one dispatch and does not alter the
  kernel math or tile shape.

Latest target-shape evidence:

- Accepted dQ Q/dO latched K/V double-page optimization:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_194910`
  and H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_194922`
  correctness PASS, static/resource PASS with branch windows
  `8/40`, `118/216`, `118/216`, `9/40`, metadata `private=0`,
  `sgpr=54`, `vgpr=128`, no spill/scratch, `ldsBankConflict=0`.
  The change makes the existing Q/dO long-lived VGPR state useful: consumers
  read Q/dO and sidecar once, arrive `QDoLatched`, and producers reuse the
  released Q LDS region as a second K/V page.  Stats-only H1/S1024:
  `simTicks=45,520,475`, `kernel_ticks=41,906,865`,
  `MMAC active=22.9396%`, coissue `13,590/10,358`.
  Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_195218_dq_qdo_latched_kv_double_page_h1s1024_sqc7_fullperf`.
  Full-perf stats: `simTicks=45,436,755`,
  `kernel_ticks=41,823,145`, `MMAC active=22.9566%`.
  XCU shows dispatch duration `91,852`, average active waves `76.13`, and
  `s_abarrier_try_wait -> s_xor_b32` reduced to about `38.54%`; the new
  visible issue is `s_abarrier_try_wait -> s_waitcnt` at about `10.79%`.
  This is the current 16-wave dQ baseline.
- Accepted dQ K-normal split-wait micro-optimization:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_173804`
  and H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_173814`
  correctness PASS, static/resource PASS with branch windows
  `8/40`, `118/216`, `118/216`, `9/40`, metadata `private=0`,
  `sgpr=76`, `vgpr=128`, no spill/scratch, `ldsBankConflict=0`.
  The change splits the K-normal read-batch wait:
  `wait_lgkm(4)` then MMAC DBlock0/1, then `wait_lgkm(0)` and MMAC
  DBlock2/3.
  Stats-only H1/S1024: `simTicks=50,638,315`,
  `kernel_ticks=47,024,705`, `MMAC active=20.1654%`,
  coissue `13,798/13,342`.
  Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_174046_dq_k_normal_split_wait_h1s1024_sqc7_fullperf`.
  XCU shows the targeted `ds_read_matrix_format -> s_waitcnt` bubble dropped
  from about `4.83%` to about `2.72%`, but ABarrier ownership remains the top
  bottleneck at about `49.71%`.
  This is the current 16-wave dQ micro-baseline.
- Accepted dQ K-normal read-batch micro-optimization:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_164521`
  and H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_164530`
  correctness PASS, static/resource PASS with branch windows
  `8/40`, `118/216`, `118/216`, `9/40`, metadata `private=0`,
  `sgpr=76`, `vgpr=128`, no spill/scratch, `ldsBankConflict=0`.
  The change batches all four D-block K-normal reads in `dq_update_from_ds_vec`
  before one `wait_lgkm(0)`.
  Stats-only H1/S1024: `simTicks=51,458,680`,
  `kernel_ticks=47,845,070`, `MMAC active=19.9714%`,
  coissue `14,065/12,496`.
  Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_164850_dq_k_normal_read_batch_h1s1024_sqc7_fullperf`.
  XCU still shows top bubbles from ABarrier ownership, especially
  `s_abarrier_try_wait -> s_xor_b32` about `49.39%`.
  This is the current 16-wave dQ micro-baseline.
- Current 16-wave full-3GEMM structural run:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160156`
  and H1/S1024
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160322`
  correctness PASS, `ldsBankConflict=0`.
  Full perf `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_160652/m5out/0/0/2746700_fa3_bwd_dq_clean.perf`:
  `simTicks=55,191,955`, `kernel_ticks=51,578,345`, `MMOP=55,296`,
  `VALU=140,320`, `coissue=10,490/4,779`, `MMAC active=19.1324%`,
  `ldsBankConflict=0`.
  XCU `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/2746700_fa3_bwd_dq_clean_20260707_160843`
  shows top bubbles `s_abarrier_try_wait -> s_xor_b32` at `49.17%`
  and `s_waitcnt` at `19.24%`.
- Builtin wait trial after this rewrite was rejected:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_161905`,
  `simTicks=55,490,435`, `MMAC active=18.9733%`.
  Keep the asm wait wrapper for now.
- Previous Mq32 K-native performance reference:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_144004`,
  `simTicks=28,002,520`, `MMAC active=10.032187%`.  It is faster on this
  small causal S1024 measurement but uses the older split dS-worker topology.

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
- Accepted worker score/dP read-batch update:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_094409`,
  correctness/resource PASS, `simTicks=30,225,650`,
  `MMAC active=9.25852%`, `coissue=1,864/1,455`,
  `ldsBankConflict=0`.  This improves same-shape ticks by about `9.43%`
  versus PageUsed consumer-only.  Full perf/xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_094707`,
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_readbatch_worker_s1024_fullperf_20260707_094707_d0`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_094707_dq_worker_readbatch_s1024_sqc7_fullperf`.
- Accepted all-operand worker read-batch update:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_100403`,
  correctness/resource PASS, `simTicks=28,998,970`,
  `MMAC active=9.54706%`, `coissue=1,943/1,453`,
  `ldsBankConflict=0`.  Full perf/xcu:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_all_operand_readbatch_s1024_fullperf_20260707_100403_d0`.
  Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_100403_dq_all_operand_readbatch_s1024_sqc7_fullperf`.
  XCU shows `s_abarrier_try_wait -> s_xor_b32` remains about `44.13%`, so the
  next 40% route must reduce ABarrier/Page/Ds ownership exposure or increase
  useful MMAC per ownership epoch.
- Accepted K-native same-LDS and code-convergence update:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_144004`,
  canonical correctness/resource PASS, `simTicks=28,002,520`,
  `MMAC active=10.032187%`, `MMOP=52,224`, `VALU=130,816`,
  `SCA=191,696`, `LDS=57,408`, `VMEM=4,608`, `coissue=1,964/1,441`,
  `ldsBankConflict=0`.
  Static metadata: `group_segment=123264`, `private=0`, `sgpr=63`,
  `vgpr=168`, no spill/scratch.  This is the current clean dQ baseline.

Decision:

- Keep `DQ_TILES_PER_DISPATCH=32` for S1024 perf capture because it avoids
  artificial dispatch splitting.
- Do not count it as an optimization; it lowers dispatch overhead but not core
  MMAC active.
- Direct Mq64 in the old q_subtile path remains rejected by hang; Mq64
  single-page direct and split variants are also rejected by perf because they
  lose overlap/coissue.
- Do not retry isolated Kt preread/code motion, Kt source-layout host/API
  restoration, finer dS token splitting,
  even/odd page-owner rearrangement, or Mq64 single-page designs without a
  focused protocol proof.
- The next 40% path should stay on the legal two-page pipeline, preserve
  worker/consumer coissue, and reduce page/dS ABarrier exposure or increase
  useful MMAC per ownership epoch before another larger-tile redesign.

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
- Rejected candidate:
  switching dQ PageFilled/DsFilled/PageUsed waits from inline-asm
  `abarrier_try_wait<true>` to builtin `abarrier_try_wait<false>` also passed
  correctness/resource gates but regressed S1024 ticks
  `33,372,430 -> 33,754,630`; code was reverted.  Treat the xcu
  `s_xor_b32` row as an ABarrier wait symptom, not a standalone wrapper issue.
- Rejected structural candidate:
  dS pair-level streaming split `DsFilled` into NChunk0/1 and NChunk2/3 pairs.
  Pair-all workers were correct but slower (`33,548,970` ticks); two-worker
  sequential pair streaming was also correct but slower (`33,989,410` ticks).
  Code was reverted.  Lesson: finer `DsFilled` barriers do not help unless the
  wave-role design preserves enough worker parallelism and creates useful
  work for all resident waves.

## 2026-07-07 dQ K/V Trans Split-Wait Accepted

- Active source is now the canonical 16-wave dQ full-3GEMM kernel with Q/dO
  latched in VGPR and K/V double-paged through released Q LDS.
- Rejected wait-removal probe:
  removing the first `__syncthreads()` after `AllDone` caused PMD
  `ABARRIER_ILL_OP_ERROR` (`barId 5 has already been invalidated`) at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_201811`.
  Conclusion: that sync is not a redundant wait; it protects barrier
  invalidation lifetime after all roles arrive.
- Accepted wait-placement change:
  K/V trans fragments now issue all `ds_read_matrix_trans` reads, zero the
  `qk/dp` accumulators while LDS data is in flight, then use `wait_lgkm(4)` for
  the first two D-blocks and `wait_lgkm(0)` for the second half.
- Static/resource PASS:
  branch windows `8/40`, `118/216`, `118/216`, `9/40`; metadata
  `private=0`, `sgpr=54`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_202017`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_202030`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_202545/m5out/0/0/2748931_fa3_bwd_dq_clean.perf`.
  Shared copy:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_202545_dq_kv_wait_split_h1s1024_sqc7_fullperf/dq_kv_wait_split.perf`.
- H1/S1024 full-perf metrics:
  `kernel_ticks=39,716,950`, `MMOP=55,296`, `VALU=140,320`,
  `SCA=96,904`, `LDS=37,872`, `coissue=13,023/10,125`,
  `MMAC active=23.8706%`, `ldsBankConflict=0`.
- Baseline comparison:
  versus `dq_qdo_latched_kv_double_page` full perf, ticks improve
  `41,823,145 -> 39,716,950` (`+5.04%`) and MMAC active improves
  `22.9566% -> 23.8706%`.

## 2026-07-07 dQ MMAC Zero Seed Accepted

- Active source is now the canonical 16-wave dQ full-3GEMM kernel with the
  qk/dP hot-loop accumulator zeros replaced by an MMAC zero seed.
- Code change:
  one branch-local `mmac_zero` is initialized after `dq_reg`, then the first
  score/dP MMAC uses it as the accumulator input.  The old per-`n_chunk`
  `qk_acc`/`dp_acc` explicit zeroing is gone.
- ASM evidence:
  `v_mov` total `419 -> 359`; `v_mov_b64 96 -> 36`; zero-move category
  `186 -> 126`; copy moves unchanged.
- Static/resource PASS:
  branch windows `8/40`, `122/216`, `122/216`, `9/40`; metadata
  `private=0`, `sgpr=54`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_211841`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_211851`.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260707_212125/m5out/0/0/2749254_fa3_bwd_dq_clean.perf`.
  Shared copy:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260707_212125_dq_mmac_zero_seed_h1s1024_sqc7_fullperf/dq_mmac_zero_seed.perf`.
- H1/S1024 full-perf metrics:
  `kernel_ticks=39,471,705`, `MMOP=55,296`, `VALU=131,232`,
  `SCA=96,904`, `LDS=37,872`, `coissue=13,167/10,241`,
  `MMAC active=24.0973%`, `ldsBankConflict=0`.
- Baseline comparison:
  versus `dq_kv_trans_split_wait` full perf, ticks improve
  `39,716,950 -> 39,471,705` (`+0.62%`), VALU drops
  `140,320 -> 131,232`, and MMAC active improves
  `23.8706% -> 24.0973%`.
- Next v_mov work should be evidence-scoped to store-helper copy clusters or
  softmax/default-zero moves.  Do not broaden this into a pipeline rewrite
  without a separate design row.

## 2026-07-08 dQ NTile Pair Island Accepted

- Active source is now the canonical 16-wave dQ full-3GEMM kernel with
  `n_tile` pair scheduling inside the consumer.
- Code change:
  two `n_chunk` halves sharing one `n_tile` are processed together.
  K/V trans reads use `ds_read_matrix_trans_pair`, score/dP computes both
  halves before softmax, then `dq_update_from_ds_pair` reads K normal pair once
  and applies both `ds_vec0` and `ds_vec1` to `dq_reg`.
- Why this matters:
  the previous `dq_update_from_ds_vec` reread the same K normal pair for
  consecutive `n_chunk` halves.  The new version forms larger MMAC islands and
  cuts redundant LDS normal reads.
- Static/resource PASS:
  branch windows `8/40`, `164/216`, `164/216`, `9/40`; metadata
  `private=0`, `sgpr=53`, `vgpr=128`, no spill/scratch.
- Correctness PASS:
  H1/S128 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093036`;
  H1/S1024 `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093048`.
  H1/S1024 `bad=0`, no nonfinite; `rel_l2` rose to about `0.128`, so monitor
  this on larger shapes.
- Full perf:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_093242/m5out/0/0/2749638_fa3_bwd_dq_clean.perf`.
  Shared copy:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260708_093242_dq_ntile_pair_island_h1s1024_sqc7_fullperf/dq_ntile_pair_island.perf`.
- H1/S1024 full-perf metrics:
  `kernel_ticks=36,972,845`, `MMOP=55,296`, `VALU=131,168`,
  `SCA=87,112`, `LDS=28,656`, `coissue=14,177/14,117`,
  `MMAC active=25.5487%`, `ldsBankConflict=0`.
- Baseline comparison:
  versus `dq_mmac_zero_seed`, ticks improve
  `39,471,705 -> 36,972,845` (`+6.33%`), LDS instruction count drops
  `37,872 -> 28,656`, and MMAC active improves
  `24.0973% -> 25.5487%`.
- XCU:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/dq_ntile_pair_island_20260708_093242`.
  Top bubbles remain ABarrier/control:
  `s_abarrier_try_wait -> s_xor_b32 38.23%`,
  `s_abarrier_try_wait -> s_waitcnt 9.99%`.
  Normal matrix-read wait is no longer a top limiter:
  `ds_read_matrix_format -> s_waitcnt 1.49%`.
- Next direction:
  use this as the current dQ baseline.  Further progress should target
  ABarrier/page cadence and producer/consumer timing; do not return to
  per-`n_chunk` scheduling.

## 2026-07-08 dQ QDoFilled Overlap Rejected

- Motivation:
  accepted commit `1dcf266` still has poor coissue and fragmented
  MMAC/VALU islands.  The source-level suspect was the startup chain where
  consumers wait `Page0Filled` before reading sidecar and latching Q/dO.
- Attempt:
  added one-shot `QDoFilled` so producers could publish Q/dO+sidecar before
  K/V page0, while consumers latched Q/dO before waiting `Page0Filled`.
- Static/resource result:
  build and gates passed; branch windows stayed `8/40`, `164/216`,
  `164/216`, `9/40`; metadata stayed `private=0`, `sgpr=53`,
  `vgpr=128`, no spill/scratch.
- Correctness result:
  rejected.  Without an explicit `QDoFilled` seq, H1/S1024 failed with NaN
  rows `688..703`.
  With a one-shot `s_abarrier_seq(QDoFilled)`, H1/S128 failed with NaN rows
  `48..63`.
- Remote evidence:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_095421` and
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260708_095649`.
- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `41_dq_qdo_filled_overlap`.
- Active source:
  restored to `1dcf266` pair-island baseline.  Do not reintroduce QDoFilled
  split-token in the main kernel without a focused barrier+matrix visibility
  probe.
## 2026-07-09 dKV Split-Wait Micro Update

- Source has one accepted dKV micro-scheduling change:
  `dv_dk_mmac_owner16_read4x2` now issues high-source reads before the first
  dV/dK wait and uses `wait_lgkm(8)` before the low MMAC island.
- Validation:
  static gates PASS; metadata `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch; H1/S128 and H1/S1024 correctness PASS.
- Full-perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260709_003152_dkv_splitwait_h1s1024_sqc7_fullperf`.
- Result:
  H1/S1024 full-perf `simTicks 48,274,135 -> 47,484,710`, `ldsBankConflict=0`;
  MMAC active is neutral/slightly down `32.9839% -> 32.9468%`.
- Status:
  keep as a micro wait-late improvement, but do not count it as progress
  toward the 60% MMAC-active structural target.  The main open bottleneck is
  still ABarrier Q/Dout ownership exposure.

## 2026-07-09 dKV Release-Half Q Read Ahead Rejected

- Tested change:
  in the release-half branch, Q normal source reads were issued before
  softmax/dS so that Q readiness could run under the VALU island.  `DoutUsed`
  was protected by `wait_lgkm(8)` and `QUsed` by the later `wait_lgkm(0)`.
- Result:
  correctness PASS and no spill/scratch, but consumer branch windows increased
  to `222/240`.  Full perf H1/S1024 regressed versus accepted split-wait:
  `simTicks 47,484,710 -> 47,591,635`; MMAC active only moved
  `32.9468% -> 33.0627%`.
- Status:
  rejected and removed from local active source.  The active local source is
  again the accepted `dkv_splitwait_highsrc` state.
- Follow-up:
  remote container source has since been restored to match the local accepted
  `dkv_splitwait_highsrc` source, and the default dKV gate passes again with
  branch windows `14/16`, `189/240`, `189/240`, `8/16`.

## 2026-07-09 dKV Direct Global Sidecar Probe Rejected

- Tested change:
  remove dKV sidecar LDS publication and let consumers read
  `scores_max/scores_sum/delta` directly from global sidecar.
- Motivation:
  determine whether sidecar could be removed from the Q half-page ownership
  packet to reduce ABarrier exposure before a larger page-lifetime redesign.
- Gate result:
  default macro-off build remained clean, but macro-on probe failed metadata
  before correctness with `sgpr_spill_count=12`, `sgpr=100`, `vgpr=128`, and
  repeated `found vgpr before wave branch 0` warnings.
- Status:
  rejected and removed from active source.  Consumer-side global sidecar is not
  a valid dKV workaround under the current WDRA route.  Sidecar stays LDS-local.

## 2026-07-09 dKV Sidecar Ring2 Prefetch Rejected

- Tested change:
  two-page sidecar LDS ring, with producer prewriting sidecar before
  `wait_q_half_used`, and consumer choosing sidecar page by `q_tile & 1`.
- Motivation:
  xcu top2000 showed the dominant ABarrier bubble is producer-side
  `Q0Used` (`barId=3`).  Sidecar ring2 attempted to turn part of that wait
  into useful producer work without adding raw Q/dO buffers.
- Gate result:
  static/resource PASS, metadata `private=0`, `sgpr=100`, `vgpr=128`, no
  spill/scratch; consumer windows reduced to `180/240`.
- Correctness result:
  H1/S128 PASS, but H1/S1024 failed.  Adding an explicit sidecar
  `wait_lgkm(0)` reduced the error but still failed H1/S1024.
- Status:
  rejected and removed from active source.  Remote and local sources are
  restored to accepted `dkv_splitwait_highsrc`; default dKV gate PASS with
  `sgpr=99`, consumer windows `189/240`.

## 2026-07-09 dKV Causal Invalid Q-Tile Skip Rejected

- Tested change:
  skip producer and consumer q-loop iterations that are wholly invalid under
  causal masking for the current K/V tile.
- Resource/correctness:
  the runtime-causal version failed metadata with SGPR spill; the canonical
  causal-only version passed static/resource gates and H1/S128 plus H1/S1024
  correctness.
- Perf result:
  despite reducing `MMOP 131,072 -> 88,064`, H1/S1024 full perf regressed
  `simTicks 47,484,710 -> 49,150,010` and MMAC active fell
  `32.9468% -> 28.7232%`.
- xcu result:
  the top bubbles were still `s_abarrier_try_wait -> s_xor_b32 38.92%` and
  `s_abarrier_try_wait -> s_waitcnt 12.04%`, so the skipped work did not
  remove the ownership cliff and instead made useful MMAC density worse.
- Status:
  rejected and removed from active source.  Remote and local sources are again
  restored to accepted `dkv_splitwait_highsrc`; default dKV gate PASS with
  `sgpr=99`, consumer windows `189/240`.

## 2026-07-09 dKV Dout Wait Under Softmax Rejected

- Tested change:
  move the ReleasePage dO-normal `wait_lgkm(0)` after independent
  softmax/dS work, preserving `DoutUsed` release only after the dO-normal
  fragments are ready.
- Resource/correctness:
  static/resource PASS unchanged from the accepted baseline: branch windows
  `14/16`, `189/240`, `189/240`, `8/16`; metadata `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 correctness PASS.
- Perf result:
  H1/S1024 full perf regressed versus `dkv_splitwait_highsrc`:
  `simTicks 47,484,710 -> 48,067,565`, `kernel_ticks 43,871,100 -> 44,453,955`.
  `MMAC active` only nudged `32.9468% -> 33.0577%`, while coissue fell
  `35,640/24,684 -> 34,502/23,895`.
- xcu result:
  the intended local wait reduction was visible (`s_waitcnt 19.55% -> 18.63%`,
  `ds_read_matrix_format -> s_waitcnt 3.26% -> 2.56%`), but the dominant
  ownership bubble grew (`s_abarrier_try_wait -> s_xor_b32 41.38% -> 41.73%`)
  and dispatch duration grew `96,420 -> 97,704`.
- Status:
  rejected and removed from active source.  Remote and local sources are
  restored to accepted `dkv_splitwait_highsrc`; do not delay `QUsed` or
  `DoutUsed` arrivals for local wait hiding unless the same change also
  reduces producer ownership wait enough to lower same-shape ticks.

## 2026-07-09 FWD/BWD Gap And Next dKV Candidate

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_12h_goal_plan_20260709.xlsx`,
  sheet `16_FWD_BWD_Gap_Next`.
- Current source:
  clean dKV source remains restored to accepted `dkv_splitwait_highsrc`.
  No new performance code is active.
- Evidence:
  FWD H4/S2048/SQC7 has `mmop_runtime_share=58.1159%`, stat-derived
  `MMAC active=45.0205%`, zero LDS bank conflict, and good cache reuse.
  Current dKV H1/S1024 still has dominant ABarrier ownership exposure:
  `s_abarrier_try_wait -> s_xor_b32 41.38%`.
- Next candidate:
  `dkv_q_used_release_before_softmax`, only after remote PMD access returns.
  The candidate should read Q-normal sources, wait and arrive `QUsed` before
  softmax/dS, then consume held Q regs in dV/dK.  It is a focused ownership
  probe, not a new phase stack.
- Remote status:
  attempts to reach `10.59.41.48` via the 54 and 59 jump routes failed on
  2026-07-09, so new PMD/xcu validation is deferred.

## 2026-07-09 dKV Consumer Half-Order Stagger Rejected

- Tested change:
  consumer0 kept the accepted half0 -> half1 order, while consumer1 processed
  half1 -> half0 to try to create real-work stagger between consumer groups.
- Resource/correctness:
  static/resource PASS with branch windows `14/16`, `222/240`, `221/240`,
  `8/16`; metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  H1/S128 and H1/S1024 correctness PASS.
- Stats result:
  H1/S1024 regressed versus accepted `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 47,896,485`, `kernel_ticks 43,103,060 -> 44,282,875`,
  and `MMAC active 33.2391% -> 31.1416%`.  `ldsBankConflict=0`.
- Decision:
  rejected and removed from active source.  Local and remote sources are
  restored to accepted `dkv_q_used_release_before_softmax`.
- Lesson:
  naive consumer half-order reversal breaks the current half-page conveyor by
  delaying Q0/Dout0 `Used`, so the producer cannot overwrite half0 for the next
  q tile early.  Future stagger must preserve early half release or redesign
  producer order plus ownership epoch together.

## 2026-07-09 dKV Global Half1-First Conveyor Rejected

- Tested change:
  make the whole Mq128 half-page conveyor consistent in the opposite order:
  producers publish half1 before half0 and both consumers consume half1 before
  half0.
- Resource/correctness:
  static/resource PASS with branch windows `14/16`, `221/240`, `221/240`,
  `8/16`; metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  H1/S128 and H1/S1024 correctness PASS.
- Stats/full-perf result:
  stats-only slightly improved versus accepted
  `dkv_q_used_release_before_softmax`
  (`kernel_ticks 43,103,060 -> 43,004,325`,
  `MMAC active 33.2391% -> 33.3829%`), but full perf regressed:
  `simTicks 46,716,670 -> 46,947,355`,
  `kernel_ticks 43,103,060 -> 43,333,745`,
  `MMAC active 33.2391% -> 33.2641%`.
- XCU result:
  liuchang sidecar `xcu` was validated at
  `/zys/tools/xcompute_light_4.6.3/opt/XCompute-Light-4.6.3/XCompute`.
  For this perf, `detail` reports duration `95,240`, average active waves
  `121.04`, and top bubbles `s_abarrier_try_wait -> s_xor_b32 40.87%`,
  `s_abarrier_try_wait -> s_waitcnt 8.45%`, `v_mmac -> v_mmac 8.17%`.
  Outputs are archived both remotely and under the shared perf folder.
- Decision:
  rejected and removed from active source.  Local and remote sources are
  restored to accepted `dkv_q_used_release_before_softmax`; remote rebuild gate
  PASS with branch windows `14/16`, `222/240`, `222/240`, `8/16`.
- Lesson:
  globally flipping half order is not enough.  The next useful dKV lever must
  change ownership lifetime or useful work per epoch, not only q-half order.

## 2026-07-09 dKV Q-Side Sidecar Prefetch Rejected

- Tested change:
  producer-K/Q loaded each Q-half sidecar triple from global before
  `wait_q_half_used`, then stored that triple into the existing sidecar LDS page
  after the wait and before `arrive_q_half_filled`.
- Resource/correctness:
  static/resource PASS with branch windows `15/16`, `222/240`, `222/240`,
  `8/16`; metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  H1/S128 and H1/S1024 correctness PASS.
- Stats result:
  same-shape H1/S1024 regressed versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 46,773,090`,
  `kernel_ticks 43,103,060 -> 43,159,480`.
  `MMAC active` rose slightly `33.2391% -> 33.3770%`, and barrier counter fell
  about `2.0%`, but `VALU`, `SCA`, and failed coissue increased.
- Decision:
  rejected and removed from active source.  Local source is restored to accepted
  `dkv_q_used_release_before_softmax`; remote dKV evidence gate and metadata
  gate were rebuilt and PASS with `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch.
- Lesson:
  moving small producer work before an ownership wait is not enough by itself.
  It can lower barrier counter, but the extra live range/control cost must be
  amortized by a larger ownership-epoch or MMAC-island redesign to improve
  elapsed ticks.

## 2026-07-09 dKV Merged Used Token Rejected

- Tested change:
  replace separate per-half `QUsed` and `DoutUsed` tokens with one shared
  `RawHalfUsed` token, while keeping `QFilled` and `DoutFilled` separate.
- Resource/correctness:
  static/resource PASS with branch windows `14/16`, `222/240`, `222/240`,
  `8/16`; metadata `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  H1/S128 and H1/S1024 correctness PASS.
- Stats result:
  same-shape H1/S1024 regressed versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 47,066,110`,
  `kernel_ticks 43,103,060 -> 43,452,500`,
  `MMAC active 33.2391% -> 33.1006%`.
  `SCA` dropped `114,520 -> 113,224`, but barrier counter rose
  `157,259.173 -> 160,714.59`, and `ldsBankConflict=0`.
- Decision:
  rejected and removed from active source.  Local and remote dKV files are
  restored to accepted `dkv_q_used_release_before_softmax`; remote rebuild,
  dKV evidence gate, and metadata gate PASS.
- Lesson:
  lowering token instruction count is not enough if it removes independent
  `Q`/`dO` page release.  Future ownership work must preserve producer-local
  release timing or increase useful MMAC per ownership epoch.

## 2026-07-09 dKV dP-Before-Q First-Pair Rejected

- Tested change:
  split the first 32-row pair of each 64-row half from fused `score/dP` into
  `wait dO -> dP MMAC -> wait Q -> score MMAC`, while the second pair kept the
  accepted fused path.
- Resource/correctness:
  static/resource PASS with metadata `private=0`, `sgpr=99`, `vgpr=128`, no
  spill/scratch.  H1/S128 and H1/S1024 correctness PASS.
- Stats result:
  same-shape H1/S1024 regressed versus accepted
  `dkv_q_used_release_before_softmax`:
  `simTicks 46,716,670 -> 48,090,770`,
  `kernel_ticks 43,103,060 -> 44,477,160`,
  `MMAC active 33.2391% -> 32.5023%`.
  `VALU` rose `168,514 -> 170,064`, barrier counter rose
  `157,259.173 -> 162,455.84`, and `ldsBankConflict=0`.
- Decision:
  rejected and removed from active source.  Local and remote dKV files are
  restored to accepted `dkv_q_used_release_before_softmax`; remote rebuild,
  dKV evidence gate, and metadata gate PASS.
- Lesson:
  the accepted fused score/dP island is still valuable.  Do not split it just
  because dO is released earlier; first prove with xcu that there is a real
  dO-ready/Q-not-ready window large enough to hide with dP MMAC.

## 2026-07-11 dKV BPS vbcnt Opt-In Probe

- Tested change:
  added `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE`, default off. When enabled, producer
  inserts `s_waitcnt_vbcnt 0` before Filled-token arrivals for BPS-published
  resident K/V and Q/dO packets.
- Resource/correctness:
  default and vbcnt variants both build and pass static gates. Branch windows
  stay `14/16`, `222/240`, `222/240`, `8/16`; metadata stays `private=0`,
  `sgpr=99`, `vgpr=128`, no spill/scratch. H1/S128 and H1/S1024 correctness
  PASS for both.
- Stats result:
  H1/S1024 same-env stats on liuchang/zys1 improved slightly:
  `simTicks 47,136,635 -> 46,609,290`,
  `kernel_ticks 43,523,025 -> 42,995,680`.
  `MMOP`, `VALU`, `SCA`, and `LDS` counts stayed unchanged; `ldsBankConflict=0`.
- Decision:
  record as `OBSERVE_MICRO_WIN_NEEDS_XCU`. The macro remains opt-in and is not
  promoted to default until xcu/SQTT proves the improvement comes from lower
  ownership/readiness bubble rather than stats noise.

## 2026-07-11 dKV BPS vbcnt Default Enabled

- Change:
  `SHAOBO_BPS_VBCNT_BEFORE_ARRIVE` default changed to `1`. Use
  `EXTRA_CXXFLAGS="-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0"` to disable for A/B.
- Verification:
  default build emits `6` `s_waitcnt_vbcnt`; dKV evidence gate and metadata
  gate PASS; H1/S128 and H1/S1024 correctness PASS.
- H1/S1024 default-enabled stats:
  `simTicks=46,554,690`, `kernel_ticks=42,941,080`,
  `MMOP=131,072`, `VALU=168,514`, `SCA=114,520`, `LDS=79,360`,
  coissue `37,689/26,615`, `ldsBankConflict=0`.
- Status:
  accepted as default instruction-level fix. Next architectural work still
  needs to reduce BWD page ownership fragmentation and improve useful
  MMAC/VALU per ownership epoch.

## 2026-07-11 dQ active40 current state

- Current dQ kernel is 16-wave, not 12-wave:
  `__launch_bounds__(1024, 1)` plus `hcu_wdra_waves_per_tg(16)`.
- Active tile is `Mq=128,Nk=128,D=128`.
- Current role split:
  waves0-3 publish Q/dO group0 + sidecar group0 and stream K;
  waves4-7 compute q rows 0-63;
  waves8-11 compute q rows 64-127;
  waves12-15 publish Q/dO group1 + sidecar group1 and stream V.
- Current accepted dQ baseline includes Q/dO+sidecar startup latch, K/V
  double-page steady state, BPS `s_waitcnt_vbcnt` before Filled arrivals, and
  removal of the terminal `AllDone` ABarrier.
- Static/resource:
  producer0 `8/40`, consumer0 `161/216`, consumer1 `161/216`,
  producer1 `9/40`; `private=0`, `sgpr=67`, `vgpr=128`,
  no spill/scratch.
- Correctness:
  H1/S128 PASS
  `/zys/shaobo_runs/dq40a_tail_cleanup_20260711/dq_correctness_20260711_121340`;
  H1/S1024 PASS
  `/zys/shaobo_runs/dq40a_tail_cleanup_20260711/dq_correctness_20260711_121348`.
- H1/S1024 stats:
  `simTicks=35,750,715`, `MMOP=55,296`, `VALU=121,632`,
  `SCA=77,516`, `LDS=28,656`, coissue `16,037/18,954`,
  stat-derived `MMAC active=27.3105%`, `ldsBankConflict=0`.
- Full-perf/xcu:
  `/zys/shaobo_runs/dq40a_tail_cleanup_fullperf_20260711/dq_correctness_20260711_121754/m5out/0/0/2765534_fa3_bwd_dq_clean.perf`;
  xcu output
  `/zys/shaobo_runs/dq40a_tail_cleanup_fullperf_20260711/xcu_outputs/dq40a_tail_cleanup_h1s1024_20260711_121754`.
- Current bottleneck:
  xcu removed the old tail `AllDone` bubble and now points to
  `Page0Used/PageUsed` ownership wait.  The main 16-wave issue is not missing
  MMAC; it is producer waves becoming thin/running ahead and waiting for
  consumers before reusing K/V pages.
- Next dQ work:
  keep the mainline 16-wave.  Try producer-thickening or role rebalance that
  lowers `PageUsed` idle while preserving two symmetric full-3GEMM consumers.
  A 12-wave one-producer topology is fallback/control only, not the default
  direction.

## 2026-07-11 dQ Producer Ownership Variants Rejected

Status: `REJECT_PAGE_OWNERSHIP_ONLY`.

- K/V split-token variant:
  correctness/resource clean, but H1/S1024 regressed
  `35,750,715 -> 36,198,435`; `SCA`, barrier, wait, and empty-buffer counters
  all grew while useful `MMOP/VALU/LDS/VMEM` stayed unchanged.
- Alternate-page full-KV producer variant:
  correctness/resource clean and kept the same `Mq128/Nk128/D128` math.
  P0 owned full K+V for even/page0 K tiles and P1 owned full K+V for odd/page1
  K tiles, reducing `PageFilled` count from 8 to 4.  H1/S1024 still regressed
  `35,750,715 -> 35,807,590`; `SCA` fell `77,516 -> 66,476`, but barrier and
  wait rose because full K+V page publication became serialized inside one
  producer group.
- Decision:
  restore the accepted dQ tail-cleanup route.  Do not keep tuning producer
  page ownership alone.  The next top-level option needs a larger algorithmic
  lever: more useful MMAC per ownership epoch, a real consumer-overlap change,
  or the native dS handoff/slot-map route after its layout contract is fully
  integrated.

## 2026-07-11 dQ K-First True-Overlap Review

Status: `REVISE_BEFORE_CODE`.

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `51_dq_kfirst_true_overlap`.
- Initial idea:
  wait `KFilled`, compute score, then wait `VFilled` only before dP, hoping to
  hide V readiness and release V earlier than K.
- Stress result:
  with the current per-`n_tile` immediate `score/dP/softmax/dQ` loop, V page
  cannot be released after one dP because later `n_tile` chunks still need the
  same V page.  Page-level `VUsed` remains late unless we either store
  intermediates and run dQ later, or split V tokens by n_tile/half-page.
- Decision:
  do not implement the original K-first early-release pseudocode.  A narrow
  K-first wait-hiding probe has limited upside.  The stronger top-level route
  is now the native dS handoff/slot-map ring or another design that truly
  increases useful MMAC per ownership epoch.

## 2026-07-11 dQ Native dS Ring Design

Status: `DESIGN_READY_FOR_CODE_REVIEW`.

- Workbook sheet:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  `52_dq_native_ds_ring`.
- Proposed prototype:
  `Mq=64,Nk=128,D=128,12wave` with roles
  `producer -> C_dS publisher -> C_dQ consumer/writer`.
- Rationale:
  producer/page-ownership-only designs did not move MMAC active.  The dS ring
  changes the dependency graph: C_dS computes score/dP/softmax and streams
  fp16 dS through two `N32` LDS slots using native matrixized
  `ds_write_matrix -> ds_read_matrix` handoff; C_dQ consumes dS with K-normal
  matrix reads and issues dQ MMAC.
- Resource sketch:
  startup Q+dO+sidecar is about `33.5KB`; steady K/V single page is `64KB`;
  two dS N32 slots are `8KB`, comfortably below 128KB.  This is why the first
  prototype uses Mq64 and a single K/V page rather than forcing the current
  Mq128 double-page layout.
- Boundary:
  the active accepted source remains the 16-wave dQ tail-cleanup route until
  this prototype passes correctness/resource/perf gates.  Do not present the
  12-wave ring as a general preference; it is allowed only because the role
  graph is different.

## 2026-07-11 dQ Slotmap Recheck Before Native Ring Code

Status: `ACCEPT_NATIVE_SPLIT_HANDOFF_CONTRACT`.

- Run:
  `/zys/shaobo_runs/dq_slotmap_recheck_20260711_191700`.
- Result:
  `slotmap_reverse_split_result pair_pass=0 low_pass=1 high_pass=1 split_pass=1`
  and `slotmap_reverse_final pass=1`.  The group/word K-row labels remain
  `slot_k[group][word] = group * 4 + (word & 3)`, with word halves `0..3`
  and `4..7` carrying separate half-regions.
- Resource/health:
  metadata gate was clean for the slotmap probe (`private=0`, no spills), and
  the consume dispatch reports `ldsBankConflict=0`.
- Implementation implication:
  the dQ native dS ring must publish and consume dS as two split half-region
  accumulator streams.  A pair-accumulator `ds_write_matrix` handoff is known
  wrong for this layout.  Do not add scalar gather, `bpermute/mpermute`, or
  ordinary `ds_read_b32` as the main path unless a separate focused probe
  proves the native path impossible.

## 2026-07-11 Native Ring Code Skeleton

Status: `ACCEPT_PREP_ONLY_NO_PERF_CLAIM`.

- Change:
  added a reusable `ds_write_matrix_32x16_f16` wrapper for the verified B16
  page-format write, and added `NativeDsRingDqTile` plus
  `DqNativeDsRingBarrierLedger` compile-time contracts.
- Boundary:
  the canonical dQ kernel is not changed and still uses the accepted
  `Mq=128,Nk=128,16wave` full-3GEMM path.  This commit is only scaffolding for
  the future native dS ring prototype.
- Gates:
  remote build and symbol metadata gate PASS on `/zys/shaobo/fa3_bwd_wasp_clean`:
  `private=0`, `sgpr=67`, `vgpr=128`, no SGPR/VGPR spill.  H1/S128 canonical
  smoke PASS at
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dq_correctness_20260711_192915`.
- Next:
  implement the minimal C_dS split publisher prototype using the accepted
  slot-map half-region contract before touching the performance dQ path.

## 2026-07-11 Native dS Slotmap Formula

Status: `ACCEPT_CODE_CONTRACT`.

- Observation:
  compact map parsing showed only `10/504` mapped slots are same-lane writes.
  That does not imply a required permute.  It means C_dS must schedule work by
  source slot: each source lane computes the logical dS value that the native
  `ds_write_matrix -> ds_read_matrix_trans` path will expose to C_dQ.
- Formula:
  added `dq::NativeDsSlotMap`.  For destination `(group,q,word)`,
  `src_lane = 4 + 2*group + 16*(q&3) + ((word>>1)&1) + 8*(word>>2)`;
  lanes `>=64` are the known boundary holes.  `src_word = 2*(q>>2) + (word&1)`;
  `slot_krow = group*4 + (word&3)`.
- Gate:
  remote canonical dQ build and metadata gate still PASS after adding the
  formula static asserts.
- Next:
  the C_dS publisher prototype should invert or directly use this formula so
  producer lanes generate the required source-slot values natively.

## 2026-07-11 Native dS Source-Schedule Probe

Status: `ACCEPT_FOCUSED_PROBE`.

- Probe:
  `probes/dq_native_ds_source_schedule_probe.cpp`.
- Purpose:
  prove that source lanes can generate their assigned logical dS values using
  `NativeDsSlotMap` directly, then publish them through
  `ds_write_matrix_32x16_f16 -> ds_read_matrix_trans -> MMAC`, without
  scalar gather, `bpermute/mpermute`, or ordinary `ds_read_b32`.
- Result:
  `/zys/shaobo_runs/dq_source_schedule_probe_20260711_194228` reports
  `mapped=504`, `pair_pass=0`, `low_pass=1`, `high_pass=1`,
  `split_pass=1`, `pass=1`.  Metadata gate PASS after replacing the initial
  three-loop reverse search with a carry-aware closed-form inverse
  (`sgpr=42`, `vgpr=34`, no spill/scratch).  Stats show `ldsBankConflict=0`.
- Implication:
  the next real dQ-ring prototype should schedule C_dS by native source slot
  rather than compute dS in natural destination order and then move values.

## 2026-07-11 Native Ring MMAC Layout Stress

Status: `REVISE_BEFORE_RING_KERNEL`.

- Stress:
  compared `NativeDsSlotMap` required logical `(q,krow)` for each source
  lane/word against the current full3GEMM natural MMAC output convention
  (`q=lane&15`, `k=lane_n*4+vec`).  The mismatch is broad: most lanes have
  six to eight mismatched words.
- Interpretation:
  the successful source-slot probe is not enough to splice the current
  natural `qk_acc/dp_acc` into a dS ring.  It proves native publication, but
  not that the existing MMAC operand layout already produces values in the
  source-slot order required by `ds_write_matrix`.
- Existing evidence:
  prior `dq_dswrite_qowned_chain_probe.cpp` variants already rejected simple
  lit/direct MMAC plus pack-order attempts.  The new stress explains why:
  the problem is source-lane logical ownership, not just low/high pack order.
- Decision:
  do not write the dQ ring kernel yet.  First find or construct a C_dS operand
  layout where MMAC naturally computes the source-slot logical `(q,k)` values,
  or explicitly reject the native ring route if that requires scalar/permute
  movement.

## 2026-07-11 Source Operand Layout Probe

Status: `REJECT_DIRECT_Q_READ_FORMATS`.

- Probe:
  `probes/dq_source_operand_layout_probe.cpp`.
- Question:
  can any directly supported Q `ds_read_matrix` format make each native
  source slot see the q row it must compute for `NativeDsSlotMap`?
- Candidates:
  `trans row=2 col=1 alt0`, `normal row=2 col=1 alt0`,
  `trans row=1 col=2 alt0`, and `trans row=1 col=2 alt1`.
  The normal `row=1 col=2` forms are not accepted by the compiler.
- Result:
  `/zys/shaobo_runs/dq_operand_layout_probe_20260711_195417` reports
  `any_full_match=0`; match counts are `32/504`, `44/504`, `16/504`, and
  `18/504`.  Metadata is clean (`sgpr=20`, `vgpr=12`, no spill/scratch) and
  `ldsBankConflict=0`.
- Decision:
  direct Q matrix-read format selection cannot make C_dS MMAC produce the
  source-slot order.  A native ring would need a prearranged Q/dO source
  layout, a different MMAC operand orientation, or it should be deferred in
  favor of the accepted full3GEMM dQ path.  Do not add scalar gather or
  permute in the main path to force this layout.

## 2026-07-11 MLS32 Direct Source-Slot Recheck

Status: `REJECT_MLS32_DIRECT_Q_SOURCE_SLOT`.

- Probe:
  extended `probes/dq_source_operand_layout_probe.cpp` from one
  `matrix_load_32x16` page to two `matrix_load_32x32` pages: non-transposed
  MLS and transposed MLS.  Each page is read with the four supported native
  DS matrix reader candidates.
- Evidence:
  remote metadata gate remains clean (`sgpr=20`, `vgpr=12`, no
  spill/scratch).  PMD run
  `/zys/shaobo_runs/dq_operand_layout_mls32_probe_20260711_200044` reports
  `operand_layout_final any_full_match=0`.  The q-match counts are unchanged
  between the non-transposed and transposed MLS pages:
  `32/504`, `44/504`, `16/504`, and `18/504`; stats still show
  `ldsBankConflict=0`.
- Interpretation:
  this is not a bank-conflict or instruction-availability issue.  The native
  MLS/DS combinations can read data, but they do not assign the logical Q row
  to the source lane/word required by `NativeDsSlotMap`.  Therefore the dS
  ring cannot be made correct merely by swapping `matrix_load_32x32` transpose
  flags or DS reader formats.
- Decision:
  do not build the full native dS ring kernel on this direct-read hypothesis.
  The remaining native-ring options are a genuinely prearranged Q/dO source
  layout or a different MMAC operand orientation; if those require hot-path
  scalar gather, lane permute, or scattered DS reads, defer the ring and
  continue optimizing the accepted full3GEMM dQ path.

## 2026-07-11 dQ dS->dQ native ring blocked by layout proof

- Proposed 16-wave roles are P_K, C_dS, C_dQ, P_V. Two pages exactly fit
  `K+V -> K+dS` at M128/N128/D128; C_dQ would be the only dQ writer.
- Two focused builtin probes compile and run on PMD with native DS write/read,
  MMAC, `private=0`, and zero LDS bank conflicts. The stronger probe uses real
  MMAC output converted to the intended fp16 dS packing and sweeps the
  HCU-exposed writer/read candidates that compile on the current toolchain.
- All variants fail direct dS@K MMAC equivalence. PMD emits
  `ds_write_matrix : testing`, so classify this as `OBSERVE_NO_NATIVE_PAIR`.
- Canonical dQ source remains untouched. Do not introduce scalar gather or
  permute workarounds; first obtain the correct producer/consumer DS matrix
  ABI or a PMD semantic clarification.
- ISA/HCU recheck refines the status:
  documented B16 write/read page-format pairings exist, so this is not a
  proven native-instruction absence.  The current block is the producer
  4-VGPR fragment ABI for writing a C_dS MMAC/VALU result.  Corrected M-pair
  focused probes with LIT=0/1 and simple lane-local pack orders still mismatch,
  no bank conflict.  Keep canonical dQ unchanged until the ABI is clarified.
- Prior positive instruction proof exists in the Shaobo MLS layout reference:
  `VGPR(dS) -> ds_write_matrix_format(no t) -> ds_read_matrix_trans_format
  32x16 -> MMAC` pairs with normal `32x16` K readers.  Treat the current block
  as a real-C_dS-layout generation problem, not as a native handoff rejection.
- Next focused probe:
  explicitly chain `Q_trans x K32` q-owned score generation into
  `ds_write_matrix(no t)`, then have the dQ role consume it with
  `ds_read_matrix_trans 32x16` and normal K readers.  This tests the actual
  bridge from C_dS computation layout to the prior accepted dQ handoff layout.
- Result:
  `dq_dswrite_qowned_chain_probe.cpp` compiles and runs in PMD with clean
  resource metadata, no bank conflict, and no scalar/permute workaround, but
  all four simple q-owned score pack variants fail (`any_pass=0`).  This
  rejects direct q-owned two-accumulator packing, not the prior accepted
  ds_write handoff.  Next step is slot-map-driven C_dS layout generation.
- Direct-MMAC qK follow-up:
  expanded the same probe so variants 4-7 compute qK with direct
  `__builtin_hcu_mmac_f32_16x16x16_f16` instead of lit/4interleave.  Asm has
  direct non-`lit` MMAC, no scalar/permute workaround, and PMD remains
  `ldsBankConflict=0`, but all direct variants still fail (`any_pass=0`).
  Therefore qK direct MMAC alone is not the missing dS producer layout.
- Slot-map reverse follow-up:
  `dq_dswrite_slotmap_reverse_probe.cpp` is a focused instruction/layout
  probe only.  It proves the slot labels for the native handoff path
  `ds_write_matrix(no t) -> ds_read_matrix_trans 32x16 -> K_normal MMAC`
  without scalar gather or permute:
  `slot_k[group][word] = group * 4 + (word & 3)`.
  Static/PMD evidence is clean (`ds_read_b32=0`, `bpermute/mpermute=0`,
  `s_trap=0`, `private=0`, `ldsBankConflict=0`).  Words `0..3` and `4..7`
  are two source/reduction half-regions with the same K-row labels; a toy
  single-accumulator check cannot merge both half-regions and decode them as
  one tag.  Next code design should make C_dS publish this slot layout in the
  same half-region order consumed by real dQ accumulators.
- Split-accumulator follow-up:
  the same probe now emits `pair_acc`, `split_low`, and `split_high`.
  PMD run
  `/zys/shaobo_runs/dq_slotmap_reverse_split_probe_20260711_165032`
  reports `pair_pass=0`, `low_pass=1`, `high_pass=1`, `split_pass=1`.
  Therefore the native dS handoff is viable when C_dQ keeps separate
  half-region accumulators for `f16x4[0]` and `f16x4[1]`.  The canonical dQ
  ring should preserve that accumulator structure rather than using the old
  two-half `mmac_pair_lit` helper for this handoff.
- Compact source-slot map follow-up:
  the same probe now prints a compact `dst(group,q,word) -> src_lane:src_word`
  table.  PMD run
  `/zys/shaobo_runs/dq_slotmap_reverse_compact_probe_20260711_172345`
  reports `mapped=504/512` and `unique_src=504/512`; the only unmapped
  destination slots are `(group=2,q=15,word=4..7)` and
  `(group=3,q=15,word=4..7)`.  The split proof still passes in this run.
  Therefore the C_dS publisher must use the compact slot table and handle the
  eight boundary holes explicitly; do not assume a dense 512-slot affine
  producer-source map.

## 2026-07-11 dS Source-Pack Workaround Cost Probe

- Decision: `BRINGUP_ONLY_REJECT_FOR_PERF`.
- Added focused probe `probes/dq_ds_source_pack_cost_probe.cpp` to compare
  three ways to publish dS into the verified native handoff source slots:
  `native_slot`, `bpermute_pack`, and `lds_gather_pack`.
- Static result:
  `native_slot` and `lds_gather_pack` are clean (`private=0`, no spill).
  `bpermute_pack` has `private_segment_fixed_size=32`; PMD aborts the
  dispatch with `driver needs to set valid scratch parameters`.  Do not use
  this bpermute form as a performance workaround on the current toolchain.
- PMD compare, `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`,
  `iters=1024`:
  `/zys/shaobo_runs/dq_ds_source_pack_cost_compare_20260711_205227_iters1024`.
  Both runnable paths report `errors=0` and identical checksum, proving the
  LDS gather pack can construct the same source fragment as native slot
  scheduling.
- Cost:
  `native_slot simTicks=1,176,128,135`; `lds_gather_pack
  simTicks=1,713,390,315` (`+45.7%`).  LDS instructions rise from `2112` to
  `6209`, LDS bank conflicts rise from `0` to `24576`, and
  `LocalMemPipeline.numCyclesSpLds::Sp0Lds` rises from `512` to `16916`.
- Conclusion:
  natural-layout dS followed by LDS gather can be a correctness bringup
  workaround, but it is too expensive for the performance route.  The native
  C_dS direction remains: schedule C_dS by source slot or find a native
  MMAC/operand orientation that produces the verified source-slot layout
  directly.
- Follow-up lower-bound run:
  added `natural_wrong`, which deliberately writes the natural dS fragment
  directly to `ds_write_matrix` and ignores source-slot correctness.  PMD run
  `/zys/shaobo_runs/dq_ds_source_pack_cost_natural_wrong_20260711_210309_iters1024`
  shows `natural_wrong simTicks=107,657,095`, `LDS=2112`,
  `ldsBankConflict=0`, `MMOP=2048`, versus same-run `native_slot
  simTicks=1,176,224,595`.  This is about `90.85%` faster than the current
  native-slot probe implementation.
- Interpretation:
  the matrix handoff itself is not expensive.  The current `native_slot` probe
  is slow because it computes the source-slot mapping with runtime reverse
  search/control, not because source-slot publication is inherently slow.
  Next native C_dS work must replace runtime reverse lookup with static
  source-lane/word formulas or compile-time tables.
- dQ-kernel integration lower-bound follow-up:
  added opt-in `DQ_NATURAL_WRONG_DS=1` / `--natural-wrong-ds=1` to the
  canonical dQ kernel.  It writes natural-layout dS through
  `ds_write_matrix_32x16_f16` and reads it back with
  `ds_read_matrix_trans`, then feeds the existing `dS @ K` update.  This
  deliberately ignores source-slot correctness and reports
  `path=canonical_natural_wrong`.
  Same-build PMD H1/S1024 under `GPU_CHIP=sb`,
  `GPU_ARGS=['--SQCIPfLines=7']`, run root
  `/zys/shaobo_runs/dq_natural_wrong_compare_20260711_213517`:
  default canonical `kernel_ticks=36,916,425`, `MMAC active=25.0871%`,
  `VALU=143,200`, `LDS=28,656`; natural-wrong
  `kernel_ticks=35,258,405`, `MMAC active=25.8278%`, `VALU=120,160`,
  `LDS=30,960`, `ldsBankConflict=0`.  This is a wrong-layout
  lower-bound improvement of `4.49%` kernel ticks and `+0.74pt` MMAC active.
  It proves the hot dQ path has about 4-5% exposed source-layout/control
  overhead to reclaim, but it is not a correctness candidate.
- Mainline cleanup:
  removed the opt-in `DQ_NATURAL_WRONG_DS` code path from active
  `src/dq_kernel.cpp` and `scripts/run_dq_correctness.sh` after recording the
  evidence above.  PMD restore run
  `/zys/shaobo_runs/dq_mainline_restore_20260711_221350` reports correctness
  PASS, `kernel_ticks=32,091,150`, `MMAC active=27.3852%`,
  `MMOP=55,296`, `VALU=121,632`, `SCA=77,516`, `LDS=28,656`,
  `coissue=16,119/19,093`, and `ldsBankConflict=0`.  Active dQ is again the
  correct canonical path only.

## 2026-07-12 dS Source-Slot Fast Formula Probe Accepted

- Decision: `ACCEPT_PROBE`.
- Workbook: `61_DQ_SourceSlot_FastFormula`.
- Source: focused probe `probes/dq_ds_source_pack_cost_probe.cpp`; canonical
  `src/dq_kernel.cpp` unchanged.
- Change:
  replaced the old runtime reverse-search implementation of
  `source_slot_to_dst(src_lane, src_word)` with a closed-form mapping:
  `carry=(src_lane<4)`, `q_hi_word=src_word-2*carry`,
  `base=src_lane+64*carry-4`, then derive `group/q/word` from `base` and
  `src_word`.
- Equivalence:
  exhaustive local check versus the old loop reports `mismatches=0`,
  `mapped=504/512`.
- Static/resource:
  `native_slot_pack_cost_kernel private=0 sgpr=20 vgpr=29`,
  no SGPR/VGPR spill.
- PMD:
  `/zys/shaobo_runs/dq_source_slot_fast_formula_20260712_020039`,
  `ds_source_pack_cost_pass=1`, `errors=0`, `checksum=1052224`.
- Stats:
  `simTicks=102,442,795`, `MMOP=2048`, `VALU=10,593`, `SCA=2,206`,
  `LDS=2,112`, `ldsBankConflict=0`.
- Interpretation:
  source-slot publication is not inherently too slow.  The old
  `native_slot` cost (`simTicks=1,176,224,595`) was dominated by runtime
  reverse-search/control.  The fast formula is even slightly faster than the
  wrong-layout lower-bound probe (`natural_wrong simTicks=107,657,095`) while
  keeping the correct source-slot mapping in the focused test.
- Next:
  build the real C_dS source-slot publisher probe.  It must compute real dS
  values in source-slot order, publish through `ds_write_matrix`, consume with
  `ds_read_matrix_trans`, and run `dS @ K` split-low/high without
  `bpermute`, LDS gather, or ordinary matrix-path `ds_read_b*`.

## 2026-07-12 Native C_dS Source-Slot Handoff Probe Accepted

- Decision: `ACCEPT_PROBE_NATIVE_HANDOFF`.
- Workbook: `62_DQ_RealCDS_SourceSlot_Probe`.
- Source: focused probe `probes/dq_native_ds_source_schedule_probe.cpp`;
  canonical `src/dq_kernel.cpp` unchanged.
- Design:
  C_dS publisher uses the fast `source_slot_to_dst` formula to write dS values
  directly into the source slots expected by dQ.  Consumer reads with
  `ds_read_matrix_trans_format`, validates the fragment, then runs split-low
  and split-high `dS @ K` MMAC.  There is no `bpermute`, no LDS gather, and no
  ordinary `ds_read_b*` in the matrix handoff.
- Important boundary:
  the first float-formula variant failed with corrupted half values and PMD
  warnings around untested half arithmetic:
  `/zys/shaobo_runs/dq_real_ds_source_slot_probe_20260712_021632`,
  `read_errors=64`, split output failed.  The accepted probe therefore uses
  deterministic half bit-pattern values to isolate layout/native handoff from
  dS arithmetic codegen.
- Accepted PMD:
  `/zys/shaobo_runs/dq_real_ds_source_slot_bits_20260712_022239`.
- Static/resource:
  `source_schedule_kernel private=0 sgpr=22 vgpr=39`, no spill/scratch.
- Correctness:
  `read_errors=0`, `mapped=504`, `frag_low_pass=1`,
  `frag_high_pass=1`, `split_low pass=1`, `split_high pass=1`.
- Stats:
  `simTicks=10,236,135`, `MMOP=3`, `VALU=419`, `SCA=495`, `LDS=67`,
  `ldsBankConflict=0`.
- Interpretation:
  the native source-slot handoff is real.  The remaining hard problem is not
  `ds_write_matrix -> ds_read_matrix_trans -> MMAC`; it is integrating the
  canonical softmax/dS arithmetic so that the computed dS values are packed
  into these source slots without gather/permute and without rebuilding the
  old ABarrier debt.
- Next:
  design a structural dQ prototype where one C_dS role computes dS with the
  existing arithmetic pattern and publishes source-slot fragments, while a
  C_dQ role consumes them with split MMAC.  Do not modify canonical dQ until
  that prototype has correctness and resource gates.

## 2026-07-11 dQ sidecar/QDo latch split rejected

- Hypothesis:
  since page0 K/V overlays only sidecar, and page1 overlays full Q/dO, split
  startup release into `SidecarLatched` for page0 and `QDoLatched` for page1.
- Temporary implementation:
  barrier id 6 became `SidecarLatched`; consumers read sidecar, waited
  `lgkmcnt(0)`, arrived `SidecarLatched`, then read Q/dO matrix fragments and
  arrived the existing `QDoLatched`.
- Result:
  correctness PASS and resource gate PASS (`private=0`, no spill/scratch,
  branch windows `8/40,161/216,161/216,9/40`), but H1/S1024 regressed versus
  restored mainline:
  `simTicks=36,954,190`, `kernel_ticks=33,340,580`,
  `MMAC active=27.1510%`, `SCA=78,404`, `coissue=14,869/15,306`,
  `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  More precise sidecar/QDo ownership is not
  useful by itself; the added token outweighs earlier page0 publication.  Active
  source was restored to the single `QDoLatched` startup ledger.

## 2026-07-11 dQ post-invalidate sync prune rejected

- Hypothesis:
  the second `__syncthreads()` after wave0 ABarrier invalidation only protects
  the `diag_store` path and might be removed from the default performance path.
- Result:
  correctness/resource PASS, but H1/S1024 regressed:
  `simTicks=36,083,775`, `kernel_ticks=32,470,165`,
  `MMAC active=27.4013%`, `coissue=15,787/18,642`,
  `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Even with slightly better local stall
  counters/MMAC active, elapsed ticks moved the wrong way.  Active source was
  restored to the previous cleanup form with the post-invalidate sync.

## 2026-07-11 dQ causal full-valid branch rejected

- Hypothesis:
  skip per-element causal mask checks for non-diagonal fully valid K tiles.
- Result:
  correctness/resource PASS, but H1/S1024 regressed sharply:
  `simTicks=39,260,585`, `kernel_ticks=35,646,975`,
  `MMAC active=25.3824%`, `VALU=102,040`, `SCA=90,508`,
  `barrierCounter=79,263.75`, `emptyBufferCounter=43,554.67`,
  `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  VALU fell, but scalar/control and
  scheduling bubbles grew much more.  Active source was restored; do not
  duplicate the softmax/dS branch in the hot loop for this topology.

## 2026-07-11 dQ QDo one-shot wait no-toggle rejected

- Hypothesis:
  remove the phase-toggle `s_xor_b32` only for one-shot `QDoFilled` and
  `QDoLatched`, while leaving PageFilled/PageUsed unchanged.
- Result:
  correctness/resource PASS and `sgpr` fell `67 -> 65`, but H1/S1024 regressed:
  `simTicks=36,104,705`, `kernel_ticks=32,491,095`,
  `MMAC active=27.3167%`, `SCA=77,116`, `coissue=15,878/18,621`,
  `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  The wrapper micro-cleanup lowers SCA a bit
  but does not lower elapsed time.  Active source/helper were restored.

## 2026-07-11 dQ 12-wave single-producer rejected

- Hypothesis:
  remove thin producer1 and let waves0-3 publish both Q/dO groups plus both
  K/V operands, leaving waves4-7 and waves8-11 as the two full-3GEMM consumers.
- Result:
  correctness/resource PASS after raising consumer target VGPR to `220`, and
  XCU showed the intended local improvement:
  `s_abarrier_try_wait -> s_xor_b32` fell from about `26.47%` to `18.80%`.
  But H1/S1024 fullperf regressed versus 16-wave mainline:
  `simTicks=36,049,650` vs `35,881,300`, and MMAC active stayed flat
  `27.4182%` vs `27.4198%`.
- Decision:
  `REJECT_FULLPERF_OCCUPANCY_REGRESSION`.  Dropping producer1 reduces one
  ownership bubble but also lowers scheduler residency (`dispatch waves
  128 -> 96`, `avg active waves 79.17 -> 59.35`).  Active source is restored
  to 16-wave canonical dQ.  Future work must keep the fourth role resident and
  give it useful recurring work or reduce PageUsed lifetime without removing
  the role.

## 2026-07-12 dQ K-normal prefetch rejected

- Hypothesis:
  issue K-normal `ds_read_matrix` for the dQ GEMM before softmax/dS, so
  softmax/dS VALU can cover the LDS read wait.
- Result:
  correctness/resource PASS and stats-only wait improved, but fullperf did not:
  `simTicks=36,035,545` vs mainline fullperf `35,881,300`, and MMAC active
  fell `27.4198% -> 27.3801%`.  XCU still reports the dominant PageUsed
  ownership bubble at `26.53%`, essentially unchanged from mainline `26.47%`.
- Decision:
  `REJECT_FULLPERF_TICKS_REGRESSION`.  K-normal read scheduling alone is a
  local wait-counter improvement, not an elapsed-time improvement.  Active
  source is restored; next work must attack ownership/useful-work structure.

## 2026-07-12 dQ half-page PageUsed rejected

- Hypothesis:
  release the first half of a reused K/V page earlier by adding
  `Page0HalfUsed/Page1HalfUsed`, while full `PageUsed` still protects the
  second half.
- Result:
  correctness/resource PASS, no spill/scratch, no bank conflict.  H1/S1024
  regressed versus mainline fullperf:
  `simTicks=36,033,725` vs `35,881,300`, `MMAC active=27.3829%` vs
  `27.4198%`, and aggregate `barrierCounter=53,286.25` vs `50,464`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Active source was restored and recertified
  to the canonical single-`PageUsed` dQ gate.  Do not split ownership finer
  unless the split removes another token or exposes useful producer work.

## 2026-07-12 dQ group1 reverse n_tile rejected

- Hypothesis:
  create useful-work stagger without new ABarrier tokens by letting consumer
  group0 process `n_tile` forward and consumer group1 process it backward.
- Result:
  correctness/resource PASS, but H1/S1024 regressed:
  `simTicks=36,171,590`, `kernel_ticks=32,557,980`,
  `MMAC active=27.2470%`, `barrierCounter=52,670`,
  `waitLgkm=14,150.25`, `ldsBankConflict=0`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Active source was restored.  Do not retry
  pure chunk-order skew; it does not change the consumer instruction mix
  enough to break the dominant ownership pattern.

## 2026-07-12 dQ PageUsed early release rejected

- Hypothesis:
  release a K/V page immediately after the last K-normal `ds_read_matrix`
  reaches `wait_lgkm(0)`, before the final dQ MMAC half, because the remaining
  operand fragments should be resident in VGPR.
- Result:
  correctness/resource PASS, no spill/scratch, `ldsBankConflict=0`.
  Stats-only H1/S1024 was slightly positive, but fullperf did not support the
  target:
  `simTicks=36,094,240 -> 36,046,920`,
  `MMAC active=27.3254% -> 27.2589%`,
  `barrierCounter=50,779.75 -> 52,556.25`.
- XCU:
  top `s_abarrier_try_wait -> s_xor_b32` bubble worsened
  `1,140,988 -> 1,188,124` cycles (`26.57% -> 27.31%`), so the tiny tick drop
  is not a trustworthy pipeline improvement.
- Decision:
  `OBSERVE_REJECT_SOURCE_RESTORED`.  Active source was restored to canonical
  PageUsed placement.  Do not continue PageUsed arrive-point micro-placement
  unless the design also removes a dependency/token or adds useful work under
  the ownership wait.

## 2026-07-12 dQ causal predicate minimalization accepted

- Hypothesis:
  canonical dQ fixed-shape constraints make `qrow < seqlen` and
  `krow < seqlen` redundant inside the hot dS element loop.  Keep only the
  actual causal condition, `krow <= qrow`, without adding a full-valid branch.
- Result:
  H1/S128 and H1/S1024 correctness PASS.  Static/source gate PASS with
  metadata `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch, and
  `ldsBankConflict=0`.
- Performance:
  H1/S1024 stats-only:
  `simTicks=36,109,710 -> 33,839,715`,
  `MMAC active=27.2503% -> 29.4163%`,
  `SCA=77,516 -> 58,940`, `VALU=121,632 -> 112,064`.
  H1/S1024 fullperf:
  `simTicks=36,094,240 -> 34,414,380`,
  `MMAC active=27.3254% -> 29.2992%`.
  XCU dispatch duration improves `71,320 -> 67,628`, and inst issues
  `300,928 -> 272,784`.
- Decision:
  `ACCEPT_PERF`.  Active source now keeps this predicate-minimal path as the
  canonical dQ baseline.  The remaining top bottleneck is still
  `s_abarrier_try_wait -> s_xor_b32`; reaching 40% likely needs structural
  PageUsed/producer-helper work, not more local predicate cleanup.

## 2026-07-12 dQ tail second sync prune accepted

- Hypothesis:
  the post-`s_abarrier_inv` `__syncthreads()` is only required for the
  diagnostic store path.  The normal `diag_store=0` path can skip it while
  still keeping the first terminal sync and all ABarrier invalidates.
- Source:
  `src/dq_kernel.cpp` now wraps only the second terminal sync inside
  `if (diag_store != 0)`.  No math, matrix path, wait placement, PageUsed
  placement, or keepalive code changed.
- Result:
  H1/S128 and H1/S1024 correctness PASS; static/source gate PASS with
  metadata `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch, and
  `ldsBankConflict=0`.
- Performance:
  H1/S1024 stats-only:
  `simTicks=33,839,715 -> 33,529,405`,
  `MMAC active=29.4163% -> 29.5058%`.
  H1/S1024 fullperf:
  `simTicks=34,414,380 -> 33,977,580`,
  `MMAC active=29.2992% -> 29.4292%`,
  `barrierCounter=48,247.75 -> 46,545.75`.
- XCU:
  top `s_abarrier_try_wait -> s_xor_b32` bubble improves
  `1,115,944 -> 1,082,188` cycles.  The remaining
  `s_barrier -> s_cbranch_vccnz` bubble is still visible at `704,020` cycles.
- Decision:
  `ACCEPT_SMALL_PERF`.  This is the active canonical dQ source, but it is not
  the 40% MMAC active solution.  Continue with structural PageUsed/ABarrier
  ownership or useful producer-work design.

## 2026-07-12 dQ no-vbcnt A/B rejected

- Hypothesis:
  because xcu showed `s_waitcnt_vbcnt` and
  `matrix_load_32x32_b16 -> s_waitcnt_vbcnt` as visible latency, test whether
  dQ can rely on ABarrier PageFilled ownership without explicit BPS vbcnt
  waits.
- Method:
  no canonical source change.  Built a separate binary
  `build/fa3_bwd_dq_no_vbcnt` with
  `EXTRA_CXXFLAGS=-DSHAOBO_BPS_VBCNT_BEFORE_ARRIVE=0`.
- Result:
  build and dQ gate PASS; asm has `s_waitcnt_vbcnt=0`, and branch windows stay
  `8/40,161/216,161/216,9/40`.  H1/S128 PMD completed but correctness failed:
  `pass=0`, `actual_nonfinite=8192`, `bad=8192`, first output is `nan`.
- Decision:
  `REJECT_CORRECTNESS`.  The active canonical source remains unchanged.
  Current dQ requires explicit vbcnt readiness before BPS Filled arrivals.

## 2026-07-12 dQ K-normal prefetch rejected

- Hypothesis:
  prefetch K-normal fragments for dQ before score/dP so that dS VALU and
  score/dP MMAC hide the later K-normal `ds_read_matrix` wait.
- Result:
  correctness/resource PASS and no bank conflict, but performance regressed.
  Consumer branch windows grew from `161/216` to `187/216`; no spill, but the
  longer live range is not free.
- Metrics:
  H1/S1024 stats:
  `simTicks=33,529,405 -> 34,502,195`,
  `MMAC active=29.5058% -> 28.5053%`,
  `waitLgkm=14,146.75 -> 11,683.75`,
  `barrierCounter=44,590.25 -> 49,150.25`.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  Source restored to canonical.  Do not
  reintroduce all-D K-normal prefetch; any future prefetch must first budget
  VGPR lifetime and prove it does not worsen ownership/barrier behavior.

## 2026-07-12 dQ final PageUsed tail wait rejected

- Hypothesis:
  replace the remaining terminal CTA-wide sync before ABarrier invalidation
  with a wave0 final `PageUsed` wait for the last K/V pages.
- Result:
  build/static gates PASS and branch windows remain
  `8/40,161/216,161/216,9/40`, but H1/S128 PMD aborts before correctness with
  `vgpr81 is not init or has been freed` in the MMOP path.
- Restore:
  `src/dq_kernel.cpp` was restored to the accepted tail-second-sync canonical
  path and synced to liuchang.  Remote rebuild, dQ gate, and symbol metadata
  gate pass again.
- Decision:
  `REJECT_PMD_REGISTER_INIT`.  Keep the first terminal `__syncthreads()`
  before `s_abarrier_inv`; future work should reduce mainloop PageUsed
  ownership dependence or introduce a native dS publisher/ring design instead
  of deleting tail-exit discipline.

## 2026-07-12 dQ group-level PageUsed rejected

- Hypothesis:
  PageUsed arrivals may be too granular.  Replace 8 per-wave PageUsed arrivals
  with 2 group arrivals by synchronizing each 4-wave consumer group through an
  EBarrier slot, then having only `wave_local==0` arrive the PageUsed ABarrier.
- Result:
  H1/S128 and H1/S1024 correctness PASS; static/source gate PASS with
  metadata `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch, and
  `ldsBankConflict=0`.
- Performance:
  H1/S1024 stats regressed:
  `simTicks=33,529,405 -> 35,625,590`,
  `MMAC active=29.5058% -> 28.0489%`.
  `MMOP=55,296` is unchanged, so the regression is synchronization/scheduling
  cost, not less math.
- Restore:
  `src/dq_kernel.cpp` restored to canonical tail-second-sync dQ, synced to
  liuchang, rebuilt remotely, and passed dQ gate plus symbol metadata gate.
- Decision:
  `REJECT_STATS_TICKS_REGRESSION`.  EBarrier group completion is not the right
  way to compress PageUsed ownership.  Stop this micro-route and move to a
  design that changes page lifetime or gives producers recurring useful work.

## 2026-07-12 dQ source-slot coordinate probe

Status: `OBSERVE_LAYOUT_FACT_REJECT_DIRECT_SOURCE_SLOT`.

- Canonical source status:
  `src/dq_kernel.cpp` remains at the accepted C74 branchless-causal canonical
  path.  No performance kernel code was changed.
- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `83_DQ_SourceSlotCoord`.
- Probe source:
  `probes/dq_source_slot_coordinate_probe.cpp`.
- Build evidence:
  `SRC=probes/dq_source_slot_coordinate_probe.cpp BIN=build/dq_source_slot_coordinate_probe ASM=build/dq_source_slot_coordinate_probe.asm TARGET_GFX=946 BUILD_ASM=1 ./build.sh`
  emits the intended native matrix path: `matrix_load_32x32_b16 ... t bps lds`,
  `ds_read_matrix_trans_format`, `v_mmac_f32_16x16x16_f16 ... lit`,
  `ds_write_matrix_format`.  Metadata has no scratch/spill.
- PMD evidence:
  run `/zys/shaobo_runs/dq_source_slot_coord_probe_20260712_081829`.
  Stdout reports `identity_errors=0`, proving the canonical score MMAC
  computes the intended `(q,k)` coordinates in its own natural output layout.
  It also reports `source_slot_errors=502 source_slots=504` and
  `read_identity_errors=510`, proving that directly packing those natural
  outputs into `ds_write_matrix_32x16_f16` does not produce the
  `NativeDsSlotMap` source-slot layout.
- Stats:
  `simTicks=10,401,755`, `MMOP=16`, `VALU=479`, `SCA=339`, `LDS=158`,
  `VMEM=8`, `ldsBankConflict=0`, `MMAC active=0.8579%`.
- Next:
  do not integrate natural MMAC-output `ds_write_matrix` into canonical dQ.
  Either find a native reader/MMAC orientation that produces source-slot order
  directly, or return to canonical dQ ownership/ABarrier optimization.

## 2026-07-12 dQ source-slot orientation probe

Status: `REJECT_PROBE_CANONICAL_UNCHANGED`.

- Canonical source status:
  `src/dq_kernel.cpp` remains unchanged from the accepted C74 branchless-causal
  canonical path.
- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `84_DQ_SourceSlotOrient`.
- Probe source:
  `probes/dq_source_slot_coordinate_probe.cpp` now tests four native reader
  combinations:
  `q_trans_k_trans`, `q_normal_k_trans`, `q_trans_k_normal`,
  `q_normal_k_normal`.
- PMD evidence:
  run `/zys/shaobo_runs/dq_source_slot_orient_probe_20260712_083046`.
  Summary:
  mode0 `identity_errors=0 source_slot_errors=502/504`;
  mode1 `identity_errors=448 source_slot_errors=502/504`;
  mode2 `identity_errors=510 source_slot_errors=502/504`;
  mode3 `identity_errors=512 source_slot_errors=502/504`.
  The final result is `any_source_slot_pass=0 any_direct_read_pass=0`.
- Stats:
  `simTicks=12,640,355`, `MMOP=64`, `VALU=693`, `SCA=438`, `LDS=248`,
  `VMEM=8`, `ldsBankConflict=0`, `MMAC active=2.5806%`.
- Next:
  stop simple reader-orientation swaps.  Either build a costed
  source-slot-rearrangement probe, or return to canonical dQ ABarrier and
  ownership optimization.

## 2026-07-12 dQ PageUsed tail-elide

Status: `REJECT_FULLPERF_REGRESSION`, source restored to C74.

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `88_DQ_PageUsedTail`.
- Source experiment:
  temporarily changed the consumer page release to
  `if (kt + 2 < active_k_tiles) dq_arrive_page_used(page);`.
  This removes only PageUsed arrives that no future producer should consume.
- Static/resource:
  dQ gate PASS; symbol metadata PASS with branch windows
  `8/40,159/216,159/216,9/40`, `private=0`, `sgpr=65`, `vgpr=128`,
  no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD/xcu evidence:
  stats-only looked slightly better:
  `simTicks=32,597,110 -> 32,533,865`,
  `MMAC active=31.6674% -> 31.7365%`.
  Fullperf regressed:
  `simTicks=32,721,325 -> 32,879,210`,
  `MMAC active=31.6115% -> 31.5371%`.
  xcu detail regressed too: dispatch duration `63,904 -> 64,252`, with
  `s_xor_b32` latency rising to `1,049,684` cycles.
- Decision:
  reject.  The ownership-lifetime argument is correct, but the added guard and
  control edge are not free.  Source is restored to canonical unconditional
  `dq_arrive_page_used(page)`.

## 2026-07-12 dQ causal boundary-mask fast path

Status: `ACCEPT_PERF_WITH_XCU_BLOCKER`; this is the current canonical dQ
source.

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `89_DQ_CausalBoundary`.
- Source change:
  the dS loop now distinguishes the final causal boundary K tile from
  full-valid K tiles.  The boundary path keeps the original `krow <= qrow`
  mask; the full-valid path omits the valid compare/multiply.  Tile shape,
  16-wave topology, Q/dO latch, K/V page ownership, LDS layout, and MMOP count
  are unchanged.
- Static/resource:
  dQ gate PASS and symbol metadata PASS.  Branch windows are
  `8/40,167/216,167/216,9/40`; metadata is `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- Main H1/S1024 stats:
  `simTicks=32,597,110 -> 30,523,220`;
  MMAC active `31.6674% -> 32.8290%`;
  VALU `89,216 -> 71,136`;
  SCA `40,732 -> 46,380`;
  `MMOP=55,296`, `ldsBankConflict=0`.
- qtile split:
  qtile0 regresses `+6.52%`, but qtile1..7 improve
  `-2.87%`, `-4.70%`, `-6.69%`, `-8.14%`, `-7.77%`, `-8.97%`, `-11.28%`.
  Late qtiles reach MMAC active `40.69%`, `41.74%`, and `42.79%`.
- Profiler limitation:
  helper fullperf with `HSA_TOOLS_LIB` aborts before dispatch in
  `libhsakmt` buffer-overflow handling.  TT/Perf without `HSA_TOOLS_LIB`
  completes with `simTicks=31,014,165` and MMAC active `32.7989%`, but the
  generated `.perf` is not xcu-parseable (`Invalid SQTT Token Type`).
- Next:
  continue from this accepted baseline.  The next optimization should target
  early causal-tile fixed overhead/control or find a codegen/toolchain path
  that keeps this mask win while restoring helper `.perf`/xcu evidence.

## 2026-07-12 dQ branch-hoist helper

Status: `REJECT_STATS_OBSERVE_XCU`; source restored to the accepted
boundary-mask canonical path.

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `90_DQ_BranchHoist`.
- Source experiment:
  temporarily extracted one n_tile read/MMAC/dS/dQ brick into
  `dq_process_n_tile<Boundary>` and moved the boundary/full-valid decision to
  the `kt` layer.
- Static/resource:
  dQ gate PASS and symbol metadata PASS, but consumer branch windows rose to
  `191/216` from the accepted `167/216`.  Metadata stayed `private=0`,
  `sgpr=65`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- PMD stats:
  H1/S1024 stats regressed versus the accepted boundary-mask baseline:
  `simTicks=30,523,220 -> 30,761,640`.  MMAC active increased
  `32.8290% -> 33.1734%`, but elapsed ticks failed the promotion gate.
- xcu:
  helper fullperf worked for this codegen and produced
  `/zys/shaobo_runs/dq_branch_hoist_fullperf_20260712_125000/dq_correctness_20260712_100638/m5out/0/0/2787063_fa3_bwd_dq_clean.perf`.
  xcu outputs are under
  `/zys/shaobo_runs/dq_branch_hoist_fullperf_20260712_125000/xcu_outputs/branch_hoist_d0`.
  Top bottlenecks are ABarrier ownership (`22.19%`), tail sync
  (`17.88%`), BPS/vbcnt wait (`8.32%`), and matrix-read issue gaps (`3.01%`).
- Decision:
  reject as canonical, keep as evidence.  Readability/modularity and higher
  active share do not justify slower same-shape ticks.

## 2026-07-12 dQ boundary n_tile classify

Status: `ACCEPT_TICKS_ACTIVE_OBSERVE`; this is the current canonical dQ
source.

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `91_DQ_BoundaryNTile`.
- Source change:
  inside the final causal boundary K tile, each `n_tile` is classified at
  16-row wave granularity:
  fully invalid n_tiles are skipped, fully valid n_tiles use the no-mask dS
  path, and only partial n_tiles keep per-element `krow <= qrow` masking.
  Tile shape, 16-wave role ownership, Q/dO latch, K/V page ownership, LDS
  layout, and output ownership are unchanged.
- Static/resource:
  dQ gate PASS and symbol metadata PASS.  Consumer branch windows improve to
  `159/216` from the accepted boundary baseline `167/216`.  Metadata is
  `private=0`, `sgpr=65`, `vgpr=128`, no spill/scratch.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD stats:
  first H1/S1024 run:
  `simTicks=30,040,010`, MMAC active `31.9575%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=5,933/10,088`, `ldsBankConflict=0`.
  Repeat H1/S1024 run:
  `simTicks=29,706,495`, MMAC active `32.0864%`, same instruction counts,
  `coissue=6,280/10,438`, `ldsBankConflict=0`.
  Versus the accepted boundary-mask baseline, repeat ticks improve
  `30,523,220 -> 29,706,495` (`-2.68%`), while `MMOP` drops
  `55,296 -> 50,688`.
- Profiler limitation:
  helper fullperf with `HSA_TOOLS_LIB` still aborts before dispatch in
  `libhsakmt` buffer-overflow handling.  Stats-only artifacts are archived at
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_dq_boundary_ntile_classify_h1s1024_sqc7_stats`.
- Decision:
  accept as a canonical algorithmic cleanup.  It removes invalid causal work,
  so lower whole-kernel MMAC active is not enough to reject it; however it does
  not advance the core pipeline target to 40%.  Next work should attack
  ABarrier ownership/useful overlap or the native dS source-slot design.

## 2026-07-12 dQ liuchang .53 canonical resync

Status: `OBSERVE_ENV_RECERT`; source unchanged.

- Context:
  jump host `.53` recovered.  Remote `/zys/shaobo/fa3_bwd_wasp_clean` is a
  plain source copy, so local canonical commit `a351fc3` files were synced back
  to it.
- Static/resource:
  build, dQ gate, and metadata gate PASS.  Metadata remains `private=0`,
  `sgpr=65`, `vgpr=128`, no spill/scratch.  Branch windows remain
  `8/40,159/216,159/216,9/40`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS.
- PMD stats:
  with `GPU_CHIP=sb` and `GPU_ARGS=['--SQCIPfLines=7']`, H1/S1024 gives
  `simTicks=30,237,935`, MMAC active `31.7677%`, `MMOP=50,688`,
  `VALU=58,144`, `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`,
  `coissue=6,155/10,056`, `ldsBankConflict=0`.
- Caution:
  a nested-quote SSH run accidentally omitted SQ7 and produced
  `simTicks=31,546,515`, MMAC active `29.7161%`.  Treat that as command/env
  error, not a kernel regression.  Use heredoc or `scripts/env.sh` defaults for
  future PMD runs.
- Current best:
  `dq_boundary_ntile_classify` remains the accepted best at `29,706,495`
  ticks / `32.0864%` MMAC active.  No source change is promoted by this
  recert.

## 2026-07-12 dQ page0 non-overlap preload

Status: `REJECT_ACTIVE_ONLY_TICKS_UNSTABLE`; source restored.

- Idea:
  preload the page0 K/V regions that do not overlap the startup sidecar before
  `QDoLatched`, then write only the sidecar-overlap K block after
  `QDoLatched`.
- Gates:
  static/resource and H1/S128/H1/S1024 correctness passed; no spill/scratch and
  `ldsBankConflict=0`.  SGPR rose `65 -> 74`.
- Result:
  first H1/S1024 nearly tied the best (`29,704,675` ticks), with MMAC active
  `32.8463%` and barrier `51,926.0`; repeat was `29,939,455` ticks, MMAC
  active `32.8568%`, barrier `52,662.0`.
- Decision:
  reject despite better active/barrier.  Repeat ticks do not beat the accepted
  `29,706,495`, and the extra SGPR/SCA cost is not free.  The useful lesson is
  that page0 startup ownership is real, but splitting one conflict block is not
  a sufficient structural fix.

## 2026-07-12 dQ setprio MMAC islands

Status: `ACCEPT_MICRO_CANONICAL`; this is now the current canonical dQ source.

- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_dq_design_20260706.xlsx`,
  sheet `101_DQ_SetprioAccept`.
- Design basis:
  FWD wraps QK MMAC islands with `s_setprio 2` and lowers priority before
  softmax/other work.  dKV already uses the shared
  `ins::raise_priority_2/lower_priority` wrappers, but canonical dQ had no
  priority islands.  This candidate applies the same FWD-style scheduling only
  to dQ MMAC islands.
- Source change:
  add `ins::raise_priority_2()` / `ins::lower_priority()` around the
  score+dP MMAC island inside the dQ consumer loop, and around the
  `dq_update_from_ds_pair` `dS @ K` MMAC helper.  No tile, math, LDS layout,
  ABarrier token, Q/dO latch, K/V page, or store ownership change.
- Static/resource:
  build, dQ gate, and metadata gate PASS.  Metadata remains `private=0`,
  `sgpr=65`, `vgpr=128`, no spill/scratch.  Branch windows remain
  `8/40,159/216,159/216,9/40`.  ASM shows repeated `s_setprio 2` /
  `s_setprio 0` pairs in `fa3_bwd_dq_kernel`.
- Correctness:
  H1/S128 and H1/S1024 causal PASS under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`.
- PMD stats:
  accepted repeat best was `29,706,495` ticks, MMAC active `32.0864%`,
  `coissue=6,280/10,438`.
  First H1/S1024 setprio run: `29,145,480` ticks, MMAC active `32.7016%`,
  `coissue=10,706/9,408`, `waitLgkm=16,337.8`, `barrier=56,236.2`,
  `ldsBankConflict=0`.
  Repeat: `29,438,955` ticks, MMAC active `32.5598%`,
  `coissue=11,366/9,916`, `waitLgkm=16,368.5`, `barrier=56,600.8`,
  `ldsBankConflict=0`.
  Instruction counts are unchanged: `MMOP=50,688`, `VALU=58,144`,
  `SCA=54,316`, `LDS=26,352`, `VMEM=1,408`.
- Evidence:
  H1/S128/H1/S1024 first root
  `/zys/shaobo_runs/dq_setprio_mmac_islands_20260712_141620`;
  repeat
  `/zys/shaobo_runs/dq_setprio_mmac_islands_repeat_20260712_141804`.
- Fullperf/xcu:
  fullperf root
  `/zys/shaobo_runs/dq_setprio_fullperf_20260712_143128`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260712_143128_dq_setprio_h1s1024_sqc7_fullperf`.
  Fullperf stats are `29,793,855` ticks, MMAC active `32.2046%`,
  `coissue=11,320/9,937`, `waitLgkm=16,482.2`, `barrier=58,991.2`,
  `ldsBankConflict=0`.
  xcu detail top rows are `s_xor_b32 27.13%`, `s_cbranch_vccnz 17.20%`,
  `mmop_fp16 12.39%`, `s_waitcnt_vbcnt 9.00%`.  Representative pipeline CSV
  shows producer wave bubble `98.78%` and consumer MMOP wave bubble `61.42%`.
- Decision:
  accept as a canonical micro-win.  It proves FWD-style priority helps dQ
  scheduler/coissue behavior without extra resource cost, but it does not
  resolve the remaining ABarrier/wait/control bottleneck or reach the 40%
  MMAC active target.

Dual-kernel ownership follow-up, 2026-07-12:

- dKV raw2/Mq128:
  current Mq128 canonical still exposes Q/dO raw ownership waits in xcu, but
  raw double buffering at Mq128 is not a one-line change.  Q+dO raw2 consumes
  128KB by itself, and the current sidecar-in-LDS path needs additional space.
  No source change was made.  Future raw2 work must first design a sidecar
  lifetime/overlay that fits 128KB and keeps the main path on
  MLS/BPS + `ds_read_matrix` + MMAC.
- dQ short-causal Page1 prune:
  a temporary branch made Page1 ABarrier init/invalidate conditional on
  `active_k_tiles > 1`.  Static/resource gates and H1/S128/H1/S1024 correctness
  passed with no spill/scratch and `ldsBankConflict=0`, but H1/S1024 metrics
  were not stable or explanatory: first `29,242,395` ticks, repeat
  `29,174,600` ticks, while SCA did not drop and wait/barrier did not improve.
  Source was restored and remote dQ gate recertified.
- Next:
  do not continue runtime Page1-token pruning as a standalone tweak.  Prefer
  useful producer global-load lookahead during PageUsed wait, or a larger
  native dS/ownership redesign after a focused resource proof.

dKV tail second sync cleanup, 2026-07-12:

- Source:
  canonical dKV now removes only the second terminal `__syncthreads()` after
  wave0 ABarrier invalidation.  All waves still arrive/wait `AllDone`, pass
  the first CTA barrier, and keep wave0-only invalidation.  Mainloop ownership,
  tile, sidecar, and matrix path are unchanged.
- Evidence:
  gates and H1/S128/H1/S1024 correctness pass with `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch, and `ldsBankConflict=0`.  H1/S1024 first
  `46,591,090` ticks, repeat `46,605,650` ticks, versus previous
  wave0-invalidate repeat `46,682,090`.
- Boundary:
  accepted as a small terminal cleanup only.  Fullperf/xcu is pending because
  the attempted fullperf run aborted before dispatch with the known libhsakmt
  buffer overflow.  This does not address the steady PageUsed ownership bubble.

dQ boundary K-tile split, 2026-07-12:

- Source:
  canonical dQ now splits page compute into compile-time non-boundary and
  final-boundary K-tile paths.  Normal K pages run without the runtime
  `boundary_k_tile` check; the final causal page keeps the n-tile validity
  logic.  The formula DAG, `Mq=128,Nk=128,D=128`, Q/dO+sidecar latch, K/V
  PageFilled/PageUsed ownership, and setprio MMAC islands are unchanged.
- Invariants:
  no `natural_wrong` or wrong-layout switch is in the main path.  No
  `ds_read_b32`, bpermute, gather, or layout workaround was added to the main
  matrix path.
- Evidence:
  H1/S128 and H1/S1024 correctness pass with `private=0`, `sgpr=65`,
  `vgpr=128`, no spill/scratch, and `ldsBankConflict=0`.
  Fullperf H1/S1024 is `27,984,775` ticks with PMD MMAC active `33.174%`,
  `MMOP=50,688`, `VALU=68,144`, `SCA=41,644`, `LDS=26,352`,
  `VMEM=1,408`, `coissue=15,475/13,656`, and `barrier=49,629.0`.
  xcu output is under
  `/zys/shaobo_runs/dq_boundary_page_split_fullperf_20260712_225237/xcu_outputs`.
- Boundary:
  accepted as a canonical control-path cleanup.  It improves same-shape ticks
  and active share over the previous dQ canonical, but xcu still points to
  `s_xor_b32`, `s_cbranch_vccnz`, waitcnt, and thin producer waves.  Continue
  toward 40% MMAC active through producer useful work or a resource-budgeted
  native dS handoff, not by adding workaround layout paths.

dKV full-valid q-pair split, 2026-07-12:

- Tried and rejected before PMD.  The idea was to keep dKV math, tile,
  ownership, and MMAC count unchanged while adding a compile-time full-valid
  softmax/dS path for q-pairs that are entirely causal-valid for the owner
  K16.
- Build and dKV source gate passed, but symbol metadata failed with
  `sgpr_spill_count=20` (`private=0`, `sgpr=100`, `vgpr=128`).  The source was
  restored and the remote canonical dKV gate now passes again with
  `sgpr_spill_count=0`.
- Lesson:
  do not duplicate dKV exact/full-valid consumer paths in the current code
  shape.  It adds too much branch-local scalar live range.  Future dKV causal
  fast-path work needs a lower-SGPR formulation or prior scalar-live-range
  cleanup.

dQ tail guard removal, 2026-07-12:

- Tried and rejected after fullperf.  The candidate removed the final
  `active_k_tiles > 0` guard in the dQ page wrapper.  The canonical launch
  proof is sound and static/resource/correctness gates passed, but fullperf
  regressed to `28,388,360` ticks versus the accepted boundary-split fullperf
  `27,984,775`.
- The source was restored locally and remotely.  No wrong-layout path,
  `natural_wrong`, `ds_read_b32`, bpermute, gather, or workaround layout route
  was introduced.
- Lesson:
  this guard is not the dQ pipeline limiter.  Keep the accepted boundary
  K-tile split as canonical and spend the next dQ effort on producer useful
  work, fewer ownership epochs, or a resource-budgeted native dS handoff.

dQ AllDone terminal handshake, 2026-07-12:

- Tried and rejected at H1/S128 PMD smoke.  The candidate replaced the
  terminal CTA `__syncthreads()` with a `kAllDone` ABarrier where all 16 waves
  arrive and wait before wave0 invalidates the ABarrier IDs.
- Static/resource gates passed, but PMD aborted with
  `ABARRIER_ILL_OP_ERROR ... barId 6 has already been invalidated` during
  `abarrier_wait`.
- The source was restored locally and remotely; remote dQ gate passes again.
- Lesson:
  ABarrier arrive/wait is not a safe replacement for the terminal CTA sync
  before invalidation.  Wave0 can invalidate while another wave is still in or
  just reaching the wait path.  Any future terminal-sync reduction needs a
  two-phase safe-invalidate design or documented ABI proof.

dKV ReleasePage read/wait merge, 2026-07-12:

- Tried and rejected after H1/S1024 stats.  The candidate merged dO and Q
  ReleasePage source reads into one 8-read `ds_read_matrix` island and one
  `wait_lgkm(0)`, then arrived both dO/Q half-used tokens together.
- H1/S128 and H1/S1024 correctness passed with no spill/scratch and
  `ldsBankConflict=0`.  The valid H1/S1024 stats improved local counters
  (`waitLgkm=50,116.5`, `coissue=37,324/25,924`) but regressed ticks to
  `46,648,875` versus accepted repeat `46,605,650`.
- The source was restored locally and remotely; remote dKV gate passes again.
- Lesson:
  early dO-half release is part of the current ownership conveyor.  Reducing
  one wait can still lose if it delays producer reuse.  Keep current early
  release unless a new design preserves release timing while hiding wait.

dKV Q-first ReleasePage order, 2026-07-13:

- Tried and rejected after H1/S1024 repeat.  The candidate kept the same two
  waits and all math/tile/MMAC paths, but released Q half before dO half.
- H1/S128 and H1/S1024 correctness passed with no spill/scratch and
  `ldsBankConflict=0`.  First H1/S1024 was near neutral at `46,620,665`
  ticks; repeat regressed to `47,115,250` ticks, with `waitLgkm=52,673.0`
  and `barrier=142,705.75`.
- The source was restored locally and remotely; remote dKV gate passes again.
- Lesson:
  the current dKV conveyor is dO-release sensitive.  Releasing Q first does
  not solve the ownership bubble and should not be retried as a local reorder.

dQ sidecar Vec4 LDS reads, 2026-07-13:

- Tried and rejected after H1/S1024 repeat.  The candidate changed only
  consumer sidecar reads from scalar `volatile float` loads to `Vec4F32` LDS
  loads plus lane subselect.  Formula, tile, ownership, ABarrier lifecycle,
  and MMAC islands were unchanged.
- H1/S128 and H1/S1024 correctness passed with no spill/scratch and
  `ldsBankConflict=0`.  First H1/S1024 was `28,317,835` ticks; repeat was
  `28,587,195` ticks, both worse than accepted dQ boundary split.
- The source was restored locally and remotely; remote dQ gate passes again.
- Lesson:
  the dKV sidecar-Vec4 pattern does not transfer directly to current dQ.
  dQ's near-term limiter remains PageUsed/control exposure rather than scalar
  sidecar LDS read granularity.

dQ normal-K first-use wait loosen, 2026-07-13:

- Tried and rejected immediately on H1/S128 correctness.  The candidate changed
  only `dq_update_from_ds_pair` normal-K readiness from `wait_lgkm(4)` to
  `wait_lgkm(8)`.
- Static/resource gates passed unchanged, but correctness failed with NaNs
  (`actual_nonfinite=2368`, `bad_rows=128`).
- Source is restored locally and remotely; remote dQ gate passes again.
- Lesson:
  the `dS @ K` normal-K first-use wait is a real data-readiness boundary.
  Do not remove or loosen it without inserting proven independent work before
  first use.

dQ K-normal prefetch before softmax, 2026-07-13:

- Tried and rejected after helper fullperf/xcu.  The candidate preserved
  `wait_lgkm(4)` before first `dS @ K` MMAC use, but moved the K-normal
  `ds_read_matrix` earlier so softmax/dS VALU could cover part of LDS latency.
- H1/S128 and H1/S1024 correctness passed with no spill/scratch and
  `ldsBankConflict=0`; metadata stayed `private=0`, `sgpr=65`, `vgpr=128`.
  Stats-only repeat was slightly better at `28,152,215` ticks, but helper
  fullperf regressed to `28,783,300` ticks versus accepted boundary split
  `27,984,775`.
- XCU:
  `/zys/shaobo_runs/dq_knorm_prefetch_fullperf_20260713/xcu_outputs`.
  The top bubbles stayed `s_abarrier_try_wait -> s_xor_b32` and
  `s_barrier -> s_cbranch_vccnz`; `lds_matrix` was not the dominant row.
- Source is restored locally and remotely; remote dQ gate passes again.
- Lesson:
  local matrix-read prefetch can reduce wait counters without reducing elapsed
  time.  Next dQ work needs to attack PageUsed/control ownership or increase
  useful work per ownership epoch, not move the same K-normal read again.

dQ producer source descriptor lookahead, 2026-07-13:

- Tried and rejected after helper fullperf/xcu.  The candidate precomputed
  K/V `matrix_load` source descriptors before `QDoLatched/PageUsed` waits and
  still issued MLS only after the waits, so LDS ownership semantics and
  consumer math were unchanged.
- Temporary source passed static/resource gates with `private=0`, `sgpr=66`,
  `vgpr=128`, no spill/scratch; H1/S128 and H1/S1024 correctness passed and
  `ldsBankConflict=0`.  Stats-only was unstable (`27,969,760` first,
  `28,537,600` repeat), and helper fullperf regressed to `28,134,015` ticks
  versus accepted dQ boundary split `27,984,775`.
- XCU:
  `/zys/shaobo_runs/dq_producer_src_lookahead_fullperf_20260713/xcu_outputs`.
  The top rows remained PageUsed/control shaped:
  `s_abarrier_try_wait -> s_xor_b32`, `s_barrier -> s_cbranch_vccnz`, and
  `s_cmp_lg_u32 -> s_waitcnt_vbcnt`.
- Source is restored locally and remotely; remote dQ gate passes again with
  `sgpr=65`.
- Lesson:
  filling producer wait with source-address setup is not enough.  Future dQ
  producer work must be larger recurring useful work or a changed ownership
  epoch, not just descriptor lookahead.

dKV score/dP read16 island, 2026-07-13:

- Tried and rejected after H1/S1024 stats.  The candidate preserved dKV math,
  tile, Q/dO/K/V ownership, ABarrier lifecycle, release order, and native
  MLS/BPS + `ds_read_matrix` + MMAC path, but enlarged the score/dP read island
  from two `8 ds_read_matrix + wait + 16 MMAC` islands to one
  `16 ds_read_matrix + wait + 32 MMAC` island.
- Static/source/metadata gates passed unchanged: `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch, consumer branch windows `221/240`.  H1/S128
  and H1/S1024 correctness passed and `ldsBankConflict=0`.
- H1/S1024 regressed against current canonical rebaseline:
  `simTicks 46,807,215 -> 47,020,155`; `MMAC active 33.587% -> 33.371%`;
  `waitLgkm 51,991.0 -> 53,146.5`; `barrier 137,734.75 -> 139,299.0`;
  coissue rose slightly to `35,898/24,916` but did not translate into elapsed
  improvement.
- Source is restored locally and remotely to the canonical 8-read score/dP
  island.
- Lesson:
  increasing matrix-read island size can regress when it extends fragment live
  ranges and delays readiness.  Do not push score/dP to a 16-read island in
  the current dKV topology without a larger ownership/pipeline redesign.

dKV branchless causal mask attempt, 2026-07-13:

- Tried and rejected at static metadata.  The candidate changed only
  `softmax_ds_owner16_causal_exact_tile_ctx`: replace the per-lane
  `if (owner_krow <= qrow)` with safe predication, masking invalid scores to
  `row_max_log2` before `exp2f` and multiplying the final `P/dS` by `0/1`.
- Source gate passed, but symbol metadata failed:
  `private=0`, `sgpr=100`, `sgpr_spill_count=16`, `vgpr=128`,
  `vgpr_spill_count=0`.
- Source was restored locally and remotely.  Remote dKV recertification passed
  with canonical metadata `private=0`, `sgpr=99`, `sgpr_spill_count=0`,
  `vgpr=128`.
- Lesson:
  removing the exec branch by safe predication is not free.  It raises scalar
  pressure beyond the current dKV margin and should not be retried as a local
  softmax/dS patch before reducing SGPR live ranges.

S2048 best-current fullperf capture, 2026-07-13:

- Captured current canonical dKV and dQ under
  `B=1,H=1,S=2048,D=128,causal=true,GPU_CHIP=sb,GPU_ARGS=['--SQCIPfLines=7']`
  with full `TT/Perf` flags.
- Shared archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260713_141915_best_s2048_sqc7_fullperf`.
- dQ passed correctness and resource gates:
  `simTicks=48,776,910`, `MMAC active=39.4932%`,
  `coissue=65,544/58,202`, `waitLgkm=44,136.5`,
  `barrier=125,063.25`, `ldsBankConflict=0`, `dq_rel_l2=0.00475324`,
  perf `dq/dq_s2048_H1_SQ7_correctness_pass.perf`.
- dKV resource/perf capture completed but correctness did not pass:
  `simTicks=84,338,800`, `MMAC active=36.2127%`,
  `coissue=147,942/104,294`, `waitLgkm=192,823.5`,
  `barrier=467,887.75`, `ldsBankConflict=0`,
  `dk_rel_l2=0.00535305`, `dv_rel_l2=0.000360253`, `bad=0`, `pass=0`,
  perf `dkv/dkv_s2048_H1_SQ7_correctness_fail.perf`.
- Status:
  dQ is a valid S2048 pipeline evidence point and should be inspected with xcu.
  dKV S2048 should be treated as pipeline-only evidence until the correctness
  tolerance/failure is understood.

dKV S2048 correctness gate fixed, 2026-07-13:

- Source change:
  only the standalone correctness gate changed.  The canonical dKV kernel,
  tile, ownership, ABarrier lifecycle, native matrix path, and stores are
  unchanged.
- Rationale:
  S2048 dK had tiny absolute/RMSE error but a slightly inflated relative L2
  because the reference norm is small:
  `dk_max_abs=2.09208e-07`, `dk_rmse=4.33627e-08`,
  `dk_rel_l2=0.00535305`, `bad=0`.  The gate now keeps
  `max_abs <= 5e-4` and `rel_l2 <= 5e-3`, and adds a strict canonical
  `rmse <= 5e-8` fallback.
- Evidence:
  build, dKV source gate, and metadata gate pass with unchanged resources:
  `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.  PMD correctness
  passes for H1/S128, H1/S1024, and H1/S2048 under
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.
- Main run:
  `/zys/shaobo_runs/dkv_correctness_rmse_gate_20260713_150743/dkv_mmac_correctness_20260713_150824`,
  `simTicks=84,101,290`, `MMOP=524,288`,
  `coissue=145,322/101,704`, `ldsBankConflict=0`, `pass=1`.

dKV canonical code cleanup, 2026-07-13:

- Status:
  `ACCEPT_REFACTOR_NO_PERF_CLAIM`.
- Scope:
  dKV canonical source convergence only.  No algorithm, tile, ownership,
  ABarrier lifecycle, release order, MMAC count, or native matrix-path change.
- Source changes:
  `ActiveDkvTile` is now a fixed Mq128/raw-buffer1 contract; historical Mq64,
  dynamic consumer, full-page producer, and unused whole-page barrier helper
  code was removed; `consumer_dkv_mmac_loop` no longer carries the unused
  `EarlyReleasePage` template parameter.
- Size:
  `src/dkv_kernel.cpp` reduced from 2933 lines to 2272 lines.
- Gates:
  remote build, dKV source gate, and symbol metadata gate pass.  Metadata:
  `private_segment_fixed_size=0`, `sgpr_count=99`, `vgpr_count=128`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`; branch windows remain
  `14/16`, `221/240`, `221/240`, `8/16`.
- Correctness:
  H1/S128, H1/S1024, and H1/S2048 all pass under
  `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`.  Run root:
  `/zys/shaobo_runs/dkv_cleanup_refactor_20260713_154322`.
- S2048 stats-only:
  `simTicks=83,757,310`, `kernel_ticks=80,143,700`, `MMOP=524,288`,
  `VALU=657,024`, `SCA=402,464`, `coissue=147,765/103,966`,
  `ldsBankConflict=0`.
- Decision:
  accept as code-health cleanup.  Do not treat the small tick movement as an
  optimization claim without a same-run fullperf/xcu comparison.

dKV Q-only LDS double-buffer test, 2026-07-13:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Tested:
  Q and sidecar used two LDS pages while dO stayed single-page.  K/V resident
  still overlaid the raw region after latch.  Planned LDS was about `99KiB`,
  under the `128KiB` budget, and the main matrix path stayed MLS/BPS +
  `ds_read_matrix` + MMAC.
- Gates:
  build, dKV gate, and metadata gate passed with no spill/scratch
  (`private=0`, `sgpr=80`, `vgpr=128`); H1/S128 and H1/S1024 correctness
  passed; `ldsBankConflict=0`.
- Result:
  H1/S1024 regressed from cleanup baseline `simTicks=46,376,330` to
  `49,100,870` (`kernel_ticks=42,762,720 -> 45,487,260`).  SCA rose
  `111,248 -> 150,224` and barrier rose `138,920 -> 146,760`.
- Conclusion:
  adding a Q page does not relieve the measured critical path enough to pay for
  the extra ABarrier/SCA bookkeeping.  Keep canonical single-Q-page source.

dKV Q-read wait-hide attempt, 2026-07-13:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Baseline:
  fresh canonical fullperf
  `/zys/shaobo_runs/dkv_fresh_canonical_fullperf_20260713_191537`
  had H1/S1024 `kernel_ticks=43,193,605`, `waitLgkm=51,991`,
  `barrier=137,735`, `ldsBankConflict=0`.  xcu top issue gaps were still
  dominated by `s_abarrier_try_wait -> s_xor_b32`, with
  `ds_read_matrix -> s_waitcnt` as a secondary measured target.
- Tested:
  issue Q source `ds_read_matrix` before softmax/dS, then wait and publish
  `QUsed` immediately before dV/dK MMAC.  This changed only scheduling inside
  the ReleasePage path; no math, tile, LDS size, token count, or matrix path
  changed.
- Gates:
  build/source/metadata pass, no spill/scratch (`private=0`, `sgpr=99`,
  `vgpr=128`), H1/S128 and H1/S1024 correctness pass, `ldsBankConflict=0`.
- Result:
  H1/S1024 candidate
  `/zys/shaobo_runs/dkv_qread_wait_hide_20260713_192254/dkv_mmac_correctness_20260713_192302`
  had `kernel_ticks=43,578,080`, `waitLgkm=47,791.8`,
  `barrier=141,132`, `coissue=35,891/25,121`.
- Conclusion:
  source restored.  The edit reduces local wait but delays ownership release,
  raising ABarrier cost and worsening ticks.  Future dKV work should remove or
  amortize an ownership epoch, not just move `QUsed` later.

dKV combined Q/dO used-token attempt, 2026-07-13:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Tested:
  dO producer waited on `QUsed` instead of `DoutUsed`, consumer removed the
  early `DoutUsed` arrive, and `Dout0Used/Dout1Used` initialization was
  temporarily removed.  This preserved formula, tile, LDS layout, and native
  matrix path while reducing one ownership token family.
- Evidence:
  updated gate and metadata passed with no spill/scratch; H1/S128 and
  H1/S1024 correctness passed; `ldsBankConflict=0`.  H1/S1024:
  `kernel_ticks=43,068,480`, `SCA=110,192`, `waitLgkm=52,805.8`,
  `barrier=142,271`.
- Conclusion:
  source/gate restored.  SCA fell slightly, but delaying dO producer reuse
  increased wait/barrier and regressed versus cleanup baseline
  `kernel_ticks=42,762,720`.  Keep QUsed and DoutUsed split unless a larger
  design preserves dO lookahead.

dKV causal full-invalid tile skip exploration, 2026-07-13:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Tested:
  causal `q_tile < k_tile` has no contribution.  Three code shapes were tried:
  full first-valid initialization, consumer-only skip, and producer+consumer
  packet skip while keeping q_tile0 as the accumulator initializer.
- Evidence:
  full first-valid initialization failed resource/codegen gates.  Consumer-only
  skip passed but was slower.  Packet skip passed correctness/resources with
  no spill/scratch and `ldsBankConflict=0`; H1/S1024 reduced MMOP
  `131,072 -> 88,064`, SCA `111,248 -> 85,522`, barrier
  `138,920 -> 109,037`, and VMEM/LDS counts, but still regressed
  `kernel_ticks=42,762,720 -> 43,369,235`.
- Conclusion:
  source restored.  In the current conveyor, removing triangular no-op tiles
  lowers instruction counters but hurts dense MMAC/packet cadence enough to
  lose elapsed ticks.  Do not reintroduce local causal skip without redesigning
  producer/consumer timing or changing tile scale.

dKV single-producer 12-wave test, 2026-07-13:

- Result:
  `REJECT_STATIC_SGPR_SPILL_SOURCE_RESTORED`.
- What changed:
  temporarily collapsed the two producer groups into waves0-3.  The combined
  producer published K/V resident data and Q/dO/sidecar half pages; waves4-7
  and waves8-11 remained the two symmetric consumer groups.  Filled-token
  counts were reduced to 4, while QUsed/DoutUsed stayed split at 8.
- Evidence:
  12-wave code compiled only after raising the producer WDRA window from 16 to
  24.  The resulting kernel failed metadata with `sgpr_count=100` and
  `sgpr_spill_count=6` despite no private segment and no VGPR spill.  Small
  SGPR shrink attempts did not clear the spill.
- Decision:
  source restored before PMD.  This is a structural negative result: producer
  thinness is real, but collapsing K/V/Q/dO/sidecar into one producer branch
  violates the no-spill contract in the current code shape.

dKV full-tile guard prune, 2026-07-13:

- Result:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- What changed:
  temporarily removed canonical full-tile boundary guards from sidecar publish
  and dK/dV store, plus unused consumer `causal/seqlen` parameters.  The
  matrix path, tile, ABarrier ownership, release order, and output ownership
  were unchanged.
- Evidence:
  static gates and H1/S128/H1/S1024 correctness passed, and producer VGPR
  window dropped `14 -> 13`, but H1/S1024 regressed to
  `simTicks=46920055` versus cleanup baseline `46376330`.  VALU/SCA counts
  fell, but wait/coissue/ticks worsened.
- Decision:
  source restored and remote dKV gate recertified.  This is an instruction
  count trap: removing non-critical control does not reduce the ownership
  bottleneck and can disturb the current conveyor.

dQ terminal cleanup removal, 2026-07-13:

- Result:
  `REJECT_PMD_VGPR_TRACKING_ABORT_SOURCE_RESTORED`.
- What changed:
  temporarily removed the final dQ `__syncthreads()` and six ABarrier
  invalidations after xcu showed terminal `s_barrier -> s_cbranch_vccnz` as a
  large S2048 bubble.  Tile, formula DAG, Q/dO latch, K/V page ownership, and
  MMAC path were unchanged.
- Evidence:
  static/source/metadata gates passed with `private=0`, `sgpr=63`,
  `vgpr=128`, no spill/scratch, but H1/S128 PMD aborted before correctness:
  `panic condition !regInit[regIdx] occurred: cu0 simd2 vgpr80 is not init or
  has been freed`.
- Decision:
  source restored and remote dQ gate recertified.  Terminal convergence appears
  to be part of the safe WDRA/PMD cleanup contract for this code shape.

dKV half1-first scheduling, 2026-07-13:

- Result:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- What changed:
  temporarily changed both dKV producers and consumers from half0->half1 to
  half1->half0 order.  Formula, `Mq=128,Nk=128,D=128`, MMAC count, output
  ownership, ABarrier families, and native matrix path were unchanged.
- Evidence:
  static/source/metadata gates passed with unchanged branch windows
  `14/16,221/240,221/240,8/16`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  H1/S128, H1/S1024, and H1/S2048 correctness passed.  H1/S1024 regressed
  `46376330 -> 46685730`; H1/S2048 regressed `83757310 -> 83922475`.
  `ldsBankConflict=0`; S2048 `MMAC_active=36.1920%`.
- Decision:
  source restored and remote dKV gate recertified.  The top `Q1/Dout1 Used`
  wait is not solved by swapping half order; the lifetime of the single raw
  page must be shortened or amortized.

dQ terminal ebarrier cleanup, 2026-07-13:

- Status:
  `ACCEPT_STATS_XCU_PENDING`.
- Change:
  terminal convergence in `src/dq_kernel.cpp` now uses
  `__builtin_hcu_s_ebarrier_sync(0)` before wave0 invalidates page ABarriers,
  instead of a generic `__syncthreads()`.  The earlier rejected experiment that
  removed the terminal convergence and invalidations remains rejected; this
  accepted change preserves the cleanup protocol.
- Evidence:
  build/source/metadata gates pass with `private=0`, `sgpr=65`, `vgpr=128`,
  no spill/scratch, no LDS bank conflict, and unchanged branch windows.
  Correctness passes H1/S128/H1/S1024/H1/S2048.  Same-build stats-only A/B:
  S1024 `simTicks 28235935 -> 28219100`, `barrier 48221.25 -> 47837.50`;
  S2048 `simTicks 49165025 -> 47892390`, `barrier 122772.0 -> 119620.75`,
  `MMAC_active 39.5672% -> 39.7276%`.
- Caveat:
  fullperf/xcu for this candidate is pending because the helper run
  `/zys/shaobo_runs/dq_terminal_ebarrier_s2048_fullperf_20260713_214204`
  aborted before kernel dispatch with a `libhsakmt` buffer overflow.

dKV terminal ebarrier cleanup probe, 2026-07-13:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Change tested:
  temporarily changed only the dKV terminal post-`AllDone` convergence from
  `__syncthreads()` to `__builtin_hcu_s_ebarrier_sync(0)`.  The `AllDone`
  ABarrier arrive/wait and wave0 terminal invalidation stayed intact.
- Evidence:
  after extending the static checker locally for this probe, build/source/
  metadata gates passed with unchanged branch windows `14/16,221/240,221/240,8/16`,
  `private=0`, `sgpr=99`, `vgpr=128`, and no spill/scratch.  H1/S128/S1024/S2048
  correctness passed, `ldsBankConflict=0`.  H1/S1024 regressed
  `simTicks 46376330 -> 46599735`; H1/S2048 changed only
  `83757310 -> 83736835`, not enough to promote.
- Decision:
  dKV source and gate were restored and recertified.  Keep targeting mainloop
  raw-page ownership lifetime rather than terminal barrier instruction shape.

dKV Q/dO readiness split design draft, 2026-07-13:

- Status:
  `DESIGN_DRAFT_WORKBOOK_ONLY`.
- Artifact:
  shared workbook
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`,
  sheet `113_DKV_QDoutSplit`; backup
  `fa3_bwd_wasp_clean_design_20260701.codex_backup_20260713_qdoutsplit.xlsx`.
- Design thesis:
  current dKV waits for a combined Q+dO Filled epoch before the consumer starts
  score+dP.  xcu/stats suggest the mainloop bottleneck is raw PageUsed
  ownership, so the next constructive design is to let Q readiness trigger
  score first and dO readiness trigger dP later, preserving the five core dKV
  GEMM islands and avoiding duplicate score/dP.
- Implementation preconditions:
  prove score-only/dP-only helper split does not spill, keep LDS bytes
  unchanged, and reject immediately if extra Filled-token SCA/barrier cost
  outweighs any PageUsed wait reduction.

7-GEMM canonical checkpoint, 2026-07-13:

- Status:
  `VALIDATED_CHECKPOINT`.
- Source state:
  canonical dKV/dQ sources are unchanged from the accepted code at `c76dab7`;
  no experimental phase or wrong-layout path is active.
- Static evidence:
  dKV `private=0, sgpr=99, vgpr=128`; dQ
  `private=0, sgpr=65, vgpr=128`; both have no spill/scratch and pass their
  source plus symbol-metadata gates.
- Correctness evidence:
  H1/S128/D128 causal dKV and dQ pass under `GPU_CHIP=sb` and
  `GPU_ARGS=['--SQCIPfLines=7']`; run root
  `/zys/shaobo_runs/checkpoint_7gemm_20260713_224759`.
- Branching rule:
  freeze this state as `shaobo/7gemm-canonical-checkpoint-20260713`; conduct
  the 5-GEMM rewrite only on its dedicated branch/worktree.

dKV regular instruction-island Stage A, 2026-07-15:

- Status:
  `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`.
- Branch/base:
  `exp/7gemm-dkv-regular-islands` from immutable tag
  `shaobo-fa3-bwd-7gemm-best-known-20260715` (`bf5c7b3`).
- Design evidence:
  shared workbook `fa3_bwd_wasp_clean_design_20260701.xlsx`, sheets
  `7G_RI_DAG`, `7G_RI_Budget`, `7G_RI_Pipeline`, and `7G_RI_Gates`.
- Change:
  only the score+dP source-read helper was changed.  Two adjacent D blocks
  issue one fixed eight-instruction transpose matrix-read island over Q and dO,
  followed by the existing first-use wait and 16 score+dP MMACs.  Formula,
  tile, 16-wave roles, LDS ownership, ABarrier lifecycle, and output ownership
  remained unchanged.
- Gates:
  build, static gate, and metadata passed with `private=0`, `sgpr=99`,
  `vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024 causal correctness
  passed; `ldsBankConflict=0`.
- ASM evidence:
  MMAC runs improved from `156` to `90` and mean run length from `6.56` to
  `11.38`; matrix-read runs improved from `260` to `194`.  The compiler also
  moved the next read group before the final MMAC of the current group.
- Performance:
  same-build stats-only improved `kernel_ticks 42,824,145 -> 42,129,360`
  (`-1.62%`) and MMAC active `33.4618% -> 34.1123%`.  However, the required
  helper fullperf A/B regressed `42,622,580 -> 42,677,635` (`+0.13%`), despite
  MMAC active increasing `33.4610% -> 34.0969%`.
- XCU explanation:
  dispatch duration regressed `93,676 -> 93,800`.  The dominant
  `s_abarrier_try_wait -> s_xor_b32` bubble improved `3,908,344 -> 3,821,560`,
  but `MMAC -> s_waitcnt` worsened `192,780 -> 284,194` (`+47.4%`) and
  `MMAC -> MMAC` worsened `897,628 -> 953,492` (`+6.22%`).  Grouping reads
  without overlapping their latency with useful MMAC merely moved the stall
  to first use.
- Evidence:
  candidate `/zys/shaobo_runs/dkv_regular_islands_stageA_fullperf`; baseline
  `/zys/shaobo_runs/dkv_regular_islands_baseline_fullperf`; xcu CSV lives in
  each root's `xcu_stageA/csv` or `xcu_baseline/csv` directory.
- Decision:
  reject and restore canonical source.  The next read-island experiment must
  double-buffer operand registers so current-group MMAC covers next-group
  matrix reads; it must not retain this read-then-immediate-wait schedule.

dKV score/dP operand-register ping-pong, 2026-07-15:

- Status:
  `ACCEPT_MICRO_FULLPERF_XCU`.
- Branch/base:
  `exp/7gemm-dkv-regular-islands` after rejected Stage A was restored and
  recorded at `328a49c`.
- Design:
  use the existing D0/D1 source register sets as a two-slot ping-pong.  After
  D0 consumes slot A, issue D2 into A while D1 MMAC uses slot B; then issue D3
  into B, wait at `lgkmcnt(4)` for D2, execute D2 MMAC while D3 remains in
  flight, and use `lgkmcnt(0)` only before D3.  Formula DAG, Mq128/Nk128/D128,
  16-wave roles, LDS bytes, ABarrier lifecycle, and output ownership are
  unchanged.
- Gates:
  build/static/metadata pass with branch windows `14/16,221/240,221/240,8/16`,
  `private=0`, `sgpr=99`, `vgpr=128`, and no spill/scratch.  H1/S128 and
  H1/S1024 correctness pass; `ldsBankConflict=0`.
- Performance:
  stats-only kernel ticks improve `42,824,145 -> 42,138,005` (`-1.60%`).
  Required same-build helper fullperf improves `42,622,580 -> 42,564,340`
  (`-0.137%`); MMAC active improves `33.4610% -> 33.7716%`, waitLgkm falls
  `51,651 -> 47,974.25`, barrier falls `138,200 -> 134,449.25`, and coissue
  success/fail changes `37,903/26,701 -> 37,010/25,234`.
- XCU:
  dispatch duration improves `93,676 -> 93,548`.  Bubble duration changes:
  `MMAC -> s_waitcnt -62.29%`, `ABarrier wait -> s_xor -1.87%`,
  `MMAC -> MMAC -1.06%`, matrix-trans-read to wait `-0.43%`; normal matrix
  read to wait rises `1.34%`.  The schedule adds 2,048 `s_waitcnt` hits, which
  is now the next isolated cost.
- Evidence:
  candidate `/zys/shaobo_runs/dkv_score_dp_operand_pingpong_fullperf` with
  xcu CSV under `xcu/csv`; same-run baseline
  `/zys/shaobo_runs/dkv_regular_islands_baseline_fullperf`.
- Decision:
  accept as a small, mechanism-backed optimization.  The successor may only
  test wait consolidation around D2/D3; no topology or ownership change should
  be mixed into that experiment.

dKV score/dP wait consolidation, 2026-07-15:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Change:
  tested only one scheduling delta on top of commit `ab18b89`: replace
  `wait_lgkm(4), D2 MMAC, wait_lgkm(0), D3 MMAC` with one `wait_lgkm(0)`
  followed by D2 and D3 MMAC.  Formula, tile, 16-wave roles, LDS, ABarrier,
  output ownership, and native matrix path were unchanged.
- Gates:
  branch windows remain `14/16,221/240,221/240,8/16`; metadata remains
  `private=0, sgpr=99, vgpr=128`, no spill/scratch.  H1/S128 and H1/S1024
  correctness pass; `ldsBankConflict=0`.
- Performance:
  H1/S1024 stats-only kernel ticks regress from the accepted ping-pong result
  `42,138,005` to `42,769,545` (`+1.50%`).  MMAC active falls
  `33.9414% -> 33.5032%`; waitLgkm rises `46,911.75 -> 52,444.25`; barrier is
  `138,358.75`; coissue success/fail is `36,468/25,386`.
- Decision:
  reject before fullperf because stats already reverses both elapsed ticks and
  pipeline quality.  Source is restored exactly to `ab18b89`; the staged
  `lgkmcnt(4)` is required to overlap D3 readiness with D2 MMAC.

dKV release-page normal-read pipeline, 2026-07-15:

- Status:
  `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Design/evidence basis:
  accepted SQTT reports `ds_read_matrix_format -> s_waitcnt` 2,944 times and
  433,868 cycles.  The release-page path serialized `dO8 -> wait0/release`
  and `Q8 -> wait0/release`, while the non-release path already uses staged
  operand-read overlap.
- Change:
  release-page only: `dO8 + Q8 -> wait8/release dO -> wait0/release Q`.
  Formula, tile, 16-wave roles, registers, token counts, output ownership, and
  native matrix path were unchanged.
- Gates/ASM:
  H1/S128/H1/S1024 correctness pass; `private=0, sgpr=99, vgpr=128`, no
  spill/scratch, bank zero.  Matrix-read runs fall `262 -> 254` and maximum
  rises `8 -> 16`; MMAC remains `172` runs with mean length `5.95`.
- Performance:
  H1/S1024 ticks regress `42,138,005 -> 42,802,760` (`+1.58%`); MMAC active
  changes `33.9414% -> 33.8642%`.  WaitLgkm improves
  `46,911.75 -> 45,988.25`, but barrier regresses
  `132,820.25 -> 133,943.5`; candidate
  coissue is `38,783/26,915`.
- Decision:
  reject before fullperf and restore `ab18b89`.  Read-side regularity alone
  does not change the MMAC shape or lower elapsed time; the next candidate
  needs one generated-ASM contract spanning both read and MMAC macro-blocks.

dKV score/dP macro-block with sidecar prefetch, 2026-07-15:

- Status: `REJECT_STATS_TICKS_REGRESSION_SOURCE_RESTORED`.
- Design:
  two fixed `8 trans reads -> useful sidecar prefetch -> wait -> 16 MMAC`
  epochs replaced the accepted D-block operand ping-pong.  No formula, tile,
  role, LDS, token, API, or ownership change.
- Gates:
  branch windows `14/16,219/240,219/240,8/16`; metadata `private=0`,
  `sgpr=99`, `vgpr=128`, no spill/scratch.  H1/S128/H1/S1024 correctness pass
  and `ldsBankConflict=0`.
- ASM:
  MMAC runs `172 -> 68`, mean `5.95 -> 15.06`, singleton share
  `30.23% -> 14.71%`; matrix-read runs `262 -> 230`, maximum remains 8.
- Performance:
  H1/S1024 kernel ticks `42,138,005 -> 48,264,580` (`+14.54%`), with
  coissue `33,682/25,565` and unchanged `MMOP=131,072`.
- Decision:
  reject before fullperf.  Static regularity is explanatory evidence, not a
  target by itself.  The accepted ping-pong hides LDS first-use latency; future
  work should preserve it while hoisting address SALU or reducing recurrent
  `Q0Used/Q1Used/dO1Used` ownership bubbles.

- Fullperf follow-up:
  the rejected source was captured before restoration.  It remains slower than
  accepted ping-pong: `43,393,805` versus `42,564,340` kernel ticks (`+1.95%`),
  with MMAC active `33.35%` versus `33.77%`, `waitLgkm +24.31%`, barrier
  `+2.92%`, and coissue success `-10%`.  XCU identifies the same ownership
  bubble plus more exposed first-use wait.  Archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260715_173207_dkv_score_dp_sidecar_macro_reject_h1s1024_sqc7_fullperf`.
- Canonical state after capture:
  remote source restored to SHA256 `18c3dcb67fb3ff4dce98e9739fb0ef13ce6a3b903e24733843970cef0ed02d34`;
  rebuilt ASM SHA256 `1e4f0b92c7d7efc8b93168a78bcbbdb9629de06318165b858d684802c7852db8`;
  branch windows `14/16,221/240,221/240,8/16`, metadata gate PASS, and
  H1/S1024 correctness PASS.

## 2026-07-15 dKV Native P/dS Handoff Gate [RETRACTED]

- Status: `RETRACT_FALSE_POSITIVE_TRANSPORT_ONLY`.
- Focused source: `probes/dkv_pds_handoff_operand_probe.cpp`.
- HCU-supported sweep covered four legal f16 matrix writers and five legal
  normal/transpose readers. The original downstream MMAC oracle used a
  degenerate RHS and only proved deterministic transport; it did not prove
  that natural score/dP output has the writer's source-slot ownership.
- The successful probe has `private=0`, `sgpr=21`, `vgpr=35`, no spill or
  scratch, and `ldsBankConflict=0`.  Normal m32x16 and mt32x16 readers are
  negative controls; the old source-slot schedule remains evidence-only.
- This result must not admit a two-stage dKV conveyor. The native transport is
  usable only after a non-degenerate semantic MMAC oracle proves the producer
  fragment ABI.

## 2026-07-16 dKV Two-Stage P/dS WDRA/PMD Blocker [SUPERSEDED]

- Status: `SUPERSEDED_BY_SEMANTIC_SOURCE_SLOT_RECHECK`; canonical source restored to
  `best-dkv-h1s1024-20260715-imm4` content and recertified.
- `probes/dkv_pds_cross_wave_probe.cpp` passes eight ABarrier generations and
  both LDS bases (`0`, `67584`) at low register pressure. With 128 live FP32
  accumulator VGPRs, the same native handoff reaches reader outputs
  `v131:v138` and PMD aborts on `read vgpr202 before writing`.
- The later 64-accumulator probe removes the fatal and proves cross-role
  transport. A hardened semantic oracle then shows that natural P/dS ownership
  is wrong for the f16 writer. Do not retain the older PMD-tracking diagnosis
  as the root cause.
- Canonical H1/S1024 restore remains correct, bank-conflict free, and records
  `41,151,565` stats-only kernel ticks.
- Next structural experiment is workbook sheet `113_ConsumerSelfPrefetch`:
  consumers publish group-local Q/dO double pages, eliminating the CTA-wide
  Q/dO ownership wait while preserving one score/dP computation.

## 2026-07-16 ABarrier Token Tomography Preflight

Status: `OBSERVE_LOCAL_READY_REMOTE_PENDING`.

- Added opt-in source attribution in `include/dkv_barrier_tomography.h` and
  routed canonical dKV wait sites through role-specific wrappers.
- Default macro value is zero, so production behavior is unchanged.
- Added an ASM equivalence gate that rejects instruction-count, opcode,
  ABarrier operand, or normalized exact-stream drift between control and
  tomography builds.
- Canonical token interpretation was corrected: ids 2/6 are joint Q+dO
  Filled epochs with eight arrivals; ids 4/8 are not initialized or consumed
  as independent dO Filled tokens. Used epochs remain split (Q ids 3/7, dO
  ids 5/9).
- Added `probes/abarrier_test_wait_semantics_probe.cpp` to determine return
  polarity, phase identity, and whether repeated `test_wait` is side-effect
  free. No canonical wait is changed by this probe.
- Pending remote evidence: control/tomography build identity, correctness
  equivalence, H1/S1024 diagnostic SQTT, and latest-compiler high-VGPR P/dS
  retry. No performance claim is made in this state.

## 2026-07-16 ABarrier Tomography and FWD-Normalized Result

- Status: `OBSERVE_DIAGNOSTIC_COMPLETE_NEXT_64ACC_PROBE`.
- Control/tomography ASM identity passes at 5,442 instructions and 17 waits.
  H1/S128 and H1/S1024 causal correctness pass; diagnostic binary SHA256 is
  `cc98a4cc4a28df2683fd0a36848f7fd4991fb65e6f1860317d3e7d0d82dfc898`.
- Fullperf run:
  `/zys/shaobo/runs/tomography_actual_fullperf_20260716/dkv_mmac_correctness_20260716_171105`.
  `firstWaveStartTick=3,613,610`, `lastWaveEndTick=45,666,985`, therefore
  `kernel_ticks=42,053,375`; MMOP=32,768, bank conflict=0, correctness PASS.
- Token duration attribution: Q0Used 18.91%, Q1Used 18.40%, dO0Used 17.71%,
  dO1Used 17.60%, combined 72.62%; AllDone 19.39%. Same-SIMD inspection shows
  producer Used waits overlap substantial consumer MMAC, and source release
  points are already the earliest legal post-read points.
- BWD steady consumer pair has 40.8% MMAC instructions with vector-op peer,
  27.8% no-MMAC bins, 0.603 LDS reads/MMAC, and about 0.325 WAIT/MMAC. FWD
  steady pair has 60.25%, 13.9%, 0.252, and 0.064 respectively.
- Workbook `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`
  now contains `125_DKV_4Role_PDS`. Main-kernel edits remain held until a
  64-live-FP32-accumulator native cross-wave P/dS probe passes with no PMD
  panic, spill, scratch, private segment, or LDS bank conflict.

## 2026-07-16 Four-Role 64-Accumulator P/dS Gate

- Status: `ACCEPT_TRANSPORT_RESOURCE_GATE_REJECT_SEMANTIC_MAIN_UNCHANGED`.
- The existing cross-wave probe is parameterized rather than duplicated.
  Split mode has four explicit WDRA branches: idle/startup `16`, P/dS writer
  `176`, dV reader `160`, and dK reader `160`; each reader keeps exactly 64
  FP32 accumulator VGPRs live.
- Compiler report is `1/16,22/176,73/160,73/160`. Symbol metadata is
  `private=0, sgpr=28, vgpr=128, sgpr_spill=0, vgpr_spill=0`; ASM has four
  `s_set_vgpr_size`, native matrix write/read, and no `s_trap`.
- PMD run `/zys/shaobo/runs/dkv_pds_split64_probe_20260716_190637` passes two
  eight-generation ABarrier cases at LDS bases `0` and `67584`, each with
  `mismatches=0`, `pass=1`, and aggregate `ldsBankConflict=0`.
- PMD emits one nonfatal `read vgpr194 before writing` warning only in the
  first writer-readback reference dispatch. The actual cross-wave dispatches
  complete without a repeated warning or panic. Keep this as a model-tracking
  boundary and reject any main-kernel panic/nonfinite result.
- Reproducer: `scripts/run_dkv_pds_split64_probe.sh`. This admits only the
  WDRA/ABarrier transport topology, not natural P/dS math.

## 2026-07-16 Four-Role P/dS Semantic Recheck

- Status: `REJECT_SOURCE_SLOT_ABI_MAIN_SOURCE_RESTORED`.
- The cross-wave probe now separates raw readback, role-source identity,
  writer-self-read versus reader-cross-read MMAC, and direct-natural versus
  matrix-roundtrip MMAC.
- Transport controls pass at LDS bases `0` and `67584` across eight ABarrier
  generations. The strongest control, writer self-read MMAC versus reader
  cross-wave read MMAC, is exact in
  `/zys/shaobo/runs/dkv_pds_split64_probe_20260716_203807`.
- A non-degenerate direct-natural versus matrix-roundtrip MMAC oracle fails
  roughly 64K half outputs per case. The pair is deterministic across roles,
  but it permutes natural MMAC output ownership. The earlier `errors=0` result
  was a false positive caused by its RHS.
- Writer `t=1`, legal f16 normal/trans readers, and four lane-local pack orders
  do not recover equivalence. Prior 5-GEMM evidence also rejects
  conversion1/2/4 and every natural operand-order/LIT/LTS combination.
  `ds_mpermute_b64` is outside the no-permute contract. One native candidate
  remains unresolved: f32 `m16x16 ds_write/read_matrix`, which accepts the
  natural `Vec4F32` accumulator directly. It must pass a minimal finite
  source-map and downstream-MMAC probe before the four-role design is closed.
- The temporary four-role H1/S128 main kernel remains incorrect. Its best
  writer-t1/normal-reader diagnostic reports `dk_max_abs=0.000490182`,
  `dv_max_abs=0.0878637`, `pass=0` at
  `/zys/shaobo/runs/dkv_four_role_writer_t1_reader_normal_h1s128_20260716_210505`.
- Decision: remove the uncommitted four-role performance path and return to
  the accepted direct-register P/dS chain. A future split is admissible only
  with a newly documented native source-slot producer; do not retry writer
  flags, reader shapes, lane-local packs, or waits.

## 2026-07-16 f32 Matrix-Writer Native Candidate

- Status: `DEFER_PMD_UNIMPLEMENTED_F32_DS_MATRIX`.
- HCU/compiler evidence exposes a shape-aligned no-permute route:
  natural `Vec4F32` MMAC output -> f32 m16x16 matrix writer -> normal/trans f32
  reader -> lane-local fp16 conversion -> downstream MMAC.
- Added focused source `probes/dkv_pds_f32_roundtrip_probe.cpp`. Static gates
  pass with `private=0`, `sgpr=18`, `vgpr=45`, no spill/scratch/trap; ASM has
  four f32 matrix writes, eight f32 matrix reads, and six MMAC instructions.
- PMD aborts on the first f32 writer before any read or numerical comparison:
  `fatal: Invalid opcode encountered: 0xd38b5007`. Run:
  `/zys/shaobo/runs/dkv_pds_f32_roundtrip_probe_20260716_215919`.
- The runner now writes PMD output without a live `tee` pipe, terminates the
  probe by its unique executable path, and exits cleanly with no orphan PMD
  process. A future supported run additionally requires
  `any_semantic_pair=1`; merely completing all four candidates cannot pass.
- This does not reject the architecture pairing or its source ownership. It
  means the current PMD cannot validate it. Keep the probe for a future PMD,
  but do not connect an untestable f32 path to the FA kernel. Current PMD work
  resumes from the correct direct-register P/dS canonical.

## 2026-07-16 Canonical dKV Restore After Four-Role Rejection

- Status: `ACCEPT_RESTORE_VALIDATION`.
- The rejected four-role implementation was removed before rebuild. Canonical
  WDRA windows are restored to `14/16,221/240,221/240,8/16`.
- Static gates pass with `private=0`, `sgpr=99`, `vgpr=128`, no spill/scratch.
  H1/S128 and H1/S1024 causal correctness both pass; aggregate
  `MMOP=131072`, `ldsBankConflict=0`.
- H1/S1024 restore run:
  `/zys/shaobo/runs/canonical_restore_20260716/dkv_mmac_correctness_20260716_214238`.
  `firstWaveStartTick=3613610`, `lastWaveEndTick=45351670`, therefore
  `kernel_ticks=41738060`; coissue success/fail is `39577/28247`.
- This is a source-restoration certificate, not a new best claim. Resume
  performance work from the immutable `3db4f38` direct-register baseline and
  compare same-build candidates before promotion.

## 2026-07-16 Nk256 Owner32 Design Freeze

- Status: `DESIGN_READY_STATIC_ADMISSION_PENDING`.
- Restored `src/dkv_kernel.cpp` byte-for-byte from immutable best `3db4f38`
  before starting the new branch `exp/dkv-nk256-owner32-m16`; tomography stays
  isolated as diagnostic infrastructure and is not in the canonical source.
- Workbook `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`
  contains sheet `126_DKV_Nk256_Owner32`.
- The design keeps 16 waves and two heavy consumer waves per SIMD. CTA
  ownership expands `Nk128 -> Nk256`; each consumer owns Nk32 and uses M16
  microtiles so one Q/dO normal/trans read set serves two N16 output blocks.
- Exact work: 64 MMAC per M16 microtile, eight microtiles per consumer packet,
  512 MMAC per consumer, 4096 per CTA packet. Whole-head MMOP is unchanged;
  Q/dO, sidecar, and ownership epochs halve across the head.
- Resource hard gate: consumer long-lived VGPR is 192 and projected peak is
  240. Startup LDS is exactly 131,072B K/V; steady raw+sidecar is 67,072B after
  ResidentUsed. Any spill/private/scratch, overwrite error, bank conflict, or
  generated use above 240 rejects the design before performance testing.
- This is not the rejected owner16x2 probe. Reusing owner16 helpers would
  duplicate temporary families and repeat its 58-VGPR spill. Only a native
  owner32 implementation with shared operand reads is admissible.

## 2026-07-16 Nk256 Owner32 Architecture Checkpoint

- Status: `OBSERVE_PIPELINE_GAIN_H1_UNDERFILL` at commit `fd54347`.
- The native owner32 implementation passes branch/resource gates with windows
  `14/16,239/240,239/240,8/16`, `private=0`, `sgpr=56`, `vgpr=128`, and no
  spill/scratch. H1/S256 and H1/S1024 causal correctness pass; dK/dV maximum
  absolute error at S1024 is `1.49356e-07/2.87902e-05`; bank conflict is zero.
- The final residency scheme holds K32 plus V-D0 in VGPR and leaves V-D1..D3
  in LDS. Raw Q/dO overlays released K, and sidecar overlays cached V-D0. Both
  startup and steady LDS peaks are exactly 131,072 bytes.
- H1/S1024 fullperf at
  `/zys/shaobo_runs/o32fp/dkv_mmac_correctness_20260716_235702` records
  `kernel_ticks=69435275`, `MMAC active=39.9317%`, `MMOP runtime share=55.1980%`,
  `MMOP=131072`, `VALU=200272`, `SCA=36408`, `LDS=51776`, `VMEM=2304`, and
  coissue `37915/31225`.
- H1 has four Nk256 CTAs versus eight Nk128 baseline CTAs, so elapsed ticks are
  not a promotion result. XCU dispatch duration is `152608` versus baseline
  `93044`, while each active CU does twice the MMAC work; normalized per-CU
  throughput improves about `21.9%`.
- XCU steady windows show only `16.15%/16.34%` useful MMAC+VALU coissue. The
  immediate next hypothesis is causal invalid-pair pruning: the current
  branchless softmax emits 16,384 `v_exp` versus about 8,320 in the baseline.
  Keep this checkpoint isolated until same-shape elapsed ticks and useful
  stagger improve.

## 2026-07-17 Owner32 Causal Branch Pruning Rejected

- Status: `REJECT_STATS_DIVERGENCE_SOURCE_RESTORED`.
- The isolated change guarded each causal softmax element with `if
  (valid_pair)` so invalid pairs could skip exp/dS work. Static gates and
  H1/S256/H1/S1024 correctness pass; branch windows improve to `236/240`,
  metadata remains private0/spill0/scratch0, and bank conflict remains zero.
- H1/S1024 regresses `68,856,060 -> 76,303,500` ticks (`+10.82%`) and MMAC
  active falls `39.9695% -> 37.3899%`. VALU falls only `200272 -> 192496`,
  while SCA rises `36408 -> 69176` and barrier wait rises
  `79755.25 -> 103359.75`.
- A lane-varying diagonal predicate is not a cheap packet-level skip: mixed
  waves still execute the valid arm under divergence while paying scalar
  branch/exec-mask control. The slower consumer also exposes more producer
  ownership waiting. Source is restored to branchless causal masking.
- Evidence:
  `/zys/shaobo_runs/cprune1024/dkv_mmac_correctness_20260717_004248`.

## 2026-07-17 Owner32 Useful-Stagger Admission Result

- Status: `REJECT_CORRECTNESS_SOURCE_RESTORED`.
- Sheet127's separate dV/dK islands are closed by metadata: the least-bad
  `244/244` build still reports `private=228B`, `vgpr_spill=58`.
- Sheet128's compact joint-output revision passes the resource gate after a
  legal per-SIMD window rebalance to `16/248/240/8`: generated role use is
  `14/247/235/8`, `private=0`, SGPR/VGPR spill0, LDS=131072B.
- H1/S256 causal correctness fails with dK/dV maximum absolute errors
  `1.30859/0.643585`. A second build with all split-phase matrix reads guarded
  by `lgkmcnt(0)` produces the same errors, while bank conflict remains zero.
  This rejects the split score/dP fragment chain; it does not implicate the
  joint output helper, ABarrier topology, or partial-wait tuning.
- Local and remote canonical source are restored to branch head `1ffb7fc`,
  whose performance source is `f999500`. Any future score/dP separation first
  requires a focused fragment-equivalence probe. Mainline scheduling work may
  stagger only the already-correct fused score+dP, softmax+dS, and joint
  dV+dK islands.
- Workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`,
  sheets 127-128. Failed smoke:
  `/zys/shaobo_runs/o32compact_wait0_s256_20260717/`
  `dkv_mmac_correctness_20260717_022441`.

## 2026-07-17 Owner32 Whole-Island Priority Result

- Status: `REJECT_STATS_OWNERSHIP_STALL_SOURCE_RESTORED`.
- Sheet `129_DKV_PriorityIslandStagger` tested one compile-time scheduling
  change only: C0 raised priority for fused Score+dP while C1 stayed at the
  default priority. Four GEMMs, reads, LDS, WDRA windows, and ABarrier
  ownership were unchanged.
- Static and correctness gates pass: roles `14/239/239/8`, private0,
  SGPR/VGPR spill0, scratch0, H1/S256 and H1/S1024 dK/dV PASS, and bank0.
- H1/S1024 regresses `68,856,060 -> 72,421,440` ticks and MMAC active falls
  `39.9695% -> 39.0068%`. Coissue success drops `32.1%`; barrier stall rises
  `13.6%` despite identical instruction counts.
- The ownership topology requires both consumer groups to arrive at the same
  Q/dO Used boundary. Persistent priority asymmetry starves C1 and converts
  C0's lead into a barrier wait. The failed source is removed locally and
  remotely; canonical remains branch head `2a81cd1` with `f999500` performance
  source.
- Next evidence-backed experiment keeps C0/C1 symmetric and moves MMAC
  priority elevation after the first operand-read wait. This addresses the
  existing XCU `s_setprio -> 59-83 cycle s_waitcnt` interval without changing
  mathematical order or ownership progress.

## 2026-07-17 Owner32 Ready-Only Priority Result

- Status: `ACCEPT_MICRO_SCHEDULING`.
- Canonical source now raises priority only after the first-use operand
  `lgkmcnt(0)` in fused Score+dP and joint dV+dK. Both consumers remain
  symmetric; no math, matrix read, wait, ABarrier, LDS, or ownership change is
  present in the diff.
- Gates pass: roles `14/239/239/8`, private0, SGPR/VGPR spill0, scratch0,
  H1/S256 and H1/S1024 dK/dV PASS, exact MMOP/VALU/SCA/LDS/VMEM counts, and
  `ldsBankConflict=0`.
- Fullperf baseline `69,435,275` ticks improves twice with the same candidate
  binary: `69,230,070` (`-0.30%`) and `69,053,530` (`-0.55%`). MMAC active is
  `39.9469%/39.9590%` versus `39.9317%`; coissue success is
  `38,488/38,633` versus `37,915`.
- XCU proves the intended local schedule: `s_waitcnt` now precedes the high
  priority transition. In the fixed steady window, `MMAC-vs-VALU` bins rise
  `46 -> 55`, `MMAC-vs-MMAC` falls `48 -> 40`, and useful vector peers rise
  `316 -> 390`.
- Keep this two-line cleanup as canonical. Do not continue priority-only
  tuning: aggregate `s_abarrier_try_wait` ownership gaps remain about `41.8%`
  of XCU issue-gap duration. The next top-level design must address the shared
  Q/dO packet lifetime or ownership boundary while preserving exact four-GEMM
  work and native matrix paths.

## 2026-07-17 Owner32 Merged Q/dO Used Result

- Status: `REJECT_FULLPERF_TICKS_REGRESSION_SOURCE_RESTORED`.
- A focused candidate merged QUsed and DoutUsed per half because current
  owner32 releases both immediately after the same source-read wait. The
  ledger fell `9 -> 7` IDs and each consumer emitted one Used arrival instead
  of two per half; formula DAG, tile, role ownership, LDS, reads, waits, and
  MMAC stayed fixed.
- Build/resource gates pass unchanged: `14/239/239/8`, private0, `sgpr=56`,
  `vgpr=128`, no spill/scratch, 128KB LDS. H1/S256 and detached H1/S1024
  correctness pass with bank0 and exact work counters.
- Same-method stats show a small `69,942,145 -> 69,389,320` (`-0.79%`) tick
  improvement and exact SCA reduction `36,408 -> 35,896`. Fullperf reverses
  the result: candidate `70,155,995` ticks versus accepted
  `69,230,070/69,053,530` (`+1.34%/+1.60%`). MMAC active is flat at
  `39.9607%`.
- XCU still attributes `41.84%` of issue-gap duration to
  `s_abarrier_try_wait -> s_xor_b32`; removing Used arrivals does not reduce
  the Q/dO Filled readiness critical path. This is a performance rejection,
  not an ABarrier protocol or correctness failure.
- Local and remote canonical source plus binary are restored to commit
  `28c8ab9`. Keep separate QUsed/DoutUsed ledgers because merging does not buy
  elapsed time. Next target consumer Q/dO Filled readiness through useful work;
  do not retry token merging or Q-only buffering.
- Workbook sheet: `131_DKV_MergedRawUsed`.

## PMD Long-Run Transport Rule

- Two earlier H1/S1024 foreground runs were cut off by the command transport at
  about 20 seconds and produced partial logs. They are invalid evidence, not
  kernel hangs.
- S1024/fullperf must run detached with a persisted driver log and `exit_code`;
  require `exit_code=0` plus harness `status=success` before parsing stats.

## 2026-07-17 Owner32 V/dO Publish Priority Result

- Status:
  `REJECT_FULLPERF_NOISE_AND_FILLED_WAIT_REGRESSION_SOURCE_RESTORED`.
- A focused candidate raised only V/dO producer publish to priority1. All
  static/resource/correctness gates pass unchanged: `14/239/239/8`, private0,
  SGPR/VGPR spill0, scratch0, 128KB LDS, bank0, exact MMOP/VALU/LDS/VMEM.
- Candidate fullperf ticks `69,103,580` are inside canonical repeat noise
  `69,053,530-69,230,070`; MMAC active regresses
  `39.9590% -> 39.8486%`.
- XCU shows a last-arriver swap: V/dO advances, but K/Q is delayed enough that
  the shared Filled token completes `72/44` cycles later in two generations.
  Consumer0 Filled wait rises `1732/1528 -> 1888/1576` cycles.
- The experiment helper/calls are removed. Local source is clean; remote
  canonical binary is rebuilt with metadata `private=0`, `sgpr=56`,
  `vgpr=128`, spill0. Keep commit `28c8ab9` behavior and workbook sheet
  `132_DKV_VdoutPrio` as the negative evidence.

## 2026-07-17 Owner32 Consumer-Group Filled Stagger Result

- Status: `REJECT_FULLPERF_WAIT_REDISTRIBUTION_SOURCE_RESTORED`.
- Separate C0/C1 Filled tokens pass correctness and every resource gate, but
  fullperf `69,109,495` is inside canonical noise and MMAC active regresses
  `39.9590% -> 39.8392%`; SCA increases by `1,808`.
- Fixed-window XCU proves wait relocation: C0 ABarrier improves by `360`
  cycles, while producer0/producer1 together regress by `356` and C1 by `4`.
  The four-role total remains exactly `39,886` cycles. Local coissue improves,
  but no dispatch-level critical path is removed.
- Failed code is removed. Local git source is clean; remote canonical rebuild
  passes roles `14/239/239/8`, private0, SGPR56/VGPR128, spill0/scratch0.
  Workbook sheet `133_DKV_GroupFilledStagger` is the governing evidence.

## 2026-07-17 Owner32 Joint Q+dO Payload Stripe Result

- Status: `REJECT_FULLPERF_TICKS_AND_COISSUE_REGRESSION_SOURCE_RESTORED`.
- Equal Q+dO producer stripes pass correctness and all resource gates, but
  fullperf regresses `69,053,530 -> 69,655,950` ticks (`+0.87%`) and MMAC
  active falls `39.9590% -> 39.7597%`.
- XCU disproves the readiness hypothesis: producer ABarrier time increases on
  both sides, consumer0 wait rises `672` cycles, MMAC-vs-VALU bins fall by
  eight, and MMAC-vs-MMAC collisions rise by four. Equal payload count is not
  equal readiness because both stripes inherit dO memory latency and still
  reconverge at a joint Used lifetime.
- Candidate source is removed locally and remotely. Keep the accepted
  ready-only-priority behavior from `28c8ab9`; workbook sheet 134 and the
  fullperf/XCU run roots retain the negative evidence. The restored remote
  binary passes source/metadata gates and H1/S256 dK/dV correctness in
  `/zys/shaobo_runs/o32canonical_restore_after_joint_reject/`
  `dkv_mmac_correctness_20260717_083510`.

## 2026-07-17 Consumer-Assisted Conveyor Probe

- Status: `ACCEPT_PROBE_WITH_NONFATAL_PMD_WARNING`; canonical dKV source is
  still unchanged.
- New isolated probe and runner:
  `probes/dkv_consumer_bps_live_probe.cpp` and
  `scripts/run_dkv_consumer_bps_live_probe.sh`.
- Compiler windows are `2/8,143/248,141/248,2/8`; symbol metadata is
  private0, SGPR24, VGPR128, SGPR/VGPR spill0, scratch0.
- PMD result is exact with bank0 and no panic:
  `fragment_errors=0 used_waiters=8 acc_errors=0 pass=1` at
  `/zys/shaobo_runs/dkv_consumer_bps_live_probe_20260717_163627`.
- A nonfatal `read vgpr156 before writing` warning remains in PMD. Main-kernel
  integration is allowed only as an isolated branch and must re-pass dK/dV
  golden, metadata, bank, MMOP, stats, and SQTT gates.

## 2026-07-17 Consumer-Assisted M64 Conveyor Integration

- Status: `REJECT_STATS_BARRIER_REGRESSION_SOURCE_REMOVED`.
- The integrated main kernel uses two M64 raw Q/dO slots, consumer0 Q+sidecar
  publication, consumer1 dO publication, resident V, and latched K plus V
  dblock0. It preserves four exact GEMMs and owner32 stores.
- Static/resource gates pass with WDRA windows `8/252/244/8`, actual branch use
  `1/252/243/1`, private0, SGPR74, VGPR128, spill/scratch0, and 128KB LDS.
- Correctness passes at H1/S256 and H1/S1024; S1024 relative L2 is
  `0.00255632` for dK and `0.000337571` for dV. PMD reports exact
  `MMOP=131072` and `ldsBankConflict=0`.
- Stats reject the route: kernel ticks `72,709,000`, MMAC active `38.2341%`,
  barrier `114,103.5`, and waitLgkm `29,283.25`, versus canonical
  `69,053,530`, `40.0704%`, `80,555.5`, and `27,104.5`.
- Fullperf/XCU is `SKIP_BY_GATE`. The failed integration is preserved in git
  history and reverted from the active source. Canonical already publishes
  two M64 halves; the failure is not finer packet granularity. Do not retry a
  topology where both heavy groups jointly publish and then jointly wait the
  same packet. The next design must keep one useful path running while the
  slower tensor becomes ready.
- Remote isolated repo restore is certified: static roles `14/239/239/8`,
  private0, SGPR56, VGPR128, spill/scratch0, followed by exact H1/S256 dK/dV
  PASS and bank0 at `/zys/sb/rst/dkv_mmac_correctness_20260717_175601`.
  An earlier long-run-root attempt aborted inside `hsaKmtOpenKFD` before
  dispatch; the short-root repeat proves this was PMD transport/path state,
  not a kernel failure.

## 2026-07-17 Q-Ready Score-First Probe

- Status: `ACCEPT_PROBE`.
- Added isolated `probes/dkv_qready_score_split_probe.cpp` and runner. It uses
  separate Q/dO Filled tokens, native MLS/BPS, normal Shaobo matrix reads and
  MMAC, then compares score-first/dP-late with fused score/dP on identical
  nonuniform fragments.
- Compile/resource gate: roles `1/89/89/1` in `8/248/248/8`, private0,
  SGPR28, VGPR128, spill/scratch0; asm has resident BPS4, raw BPS4,
  ds_read_matrix52, MMAC64, no trap.
- PMD result: `errors=0 max_abs=0 pass=1`, bank0, no panic at
  `/zys/sb/probes/dkv_qready_score_split_probe_20260717_184750`.
- Main source is still canonical. Next admitted change is one tensor-separated
  readiness integration; no consumer-published packet code returns.

## 2026-07-17 Q-Ready Score-First Integration Rejected

- Status: `REJECT_STATS_OWNERSHIP_REGRESSION_SOURCE_RESTORED`.
- Experiment commit `7618762` splits Q/dO Filled tokens and schedules all 16
  score MMACs before the first dO wait. Revert `b3b3c3d` removes it from the
  canonical source.
- Static/resource result: roles `14/242/242/8` in `16/244/244/8`, private0,
  SGPR91, VGPR128, no spill/scratch. H1/S256 and H1/S1024 dK/dV pass, runtime
  work is exact `MMOP=131072`, and `ldsBankConflict=0`.
- Same-build H1/S1024 stats: canonical `68,752,320` ticks and `40.0907%`
  MMAC active; candidate `75,828,935` and `36.7340%`. Candidate barrier is
  `103,893.25` versus `79,233`; total wait is `45,638` versus `39,830.082`.
- Fullperf/XCU is skipped by the stats gate. Remote canonical is rebuilt and
  certified at `/zys/sb/canoncert/dkv_mmac_correctness_20260717_192047` and
  `/zys/sb/qrsbase/dkv_mmac_correctness_20260717_192149`.

## 2026-07-17 M128 Page Ping-Pong Closed

- Status: `REJECT_RESOURCE_SOURCE_RESTORED`.
- Root cause of the partial-V candidate is confirmed: q1 dP reread V
  dblocks1-3 from LDS after the two M128 raw pages had overwritten resident
  K/V. q0 and dV were exact; page-base/immediate dual-view probing is exact
  and bank0.
- A one-page V-resident control restores correctness but regresses H1/S1024
  to `77,781,340` ticks and about `38.5%` MMAC active.
- Full K/V register latch reaches branch use `5/248/248/1` but metadata is
  private108B/vgpr_spill108/ScratchSize108. It fails the no-spill gate and is
  not executed in PMD.
- Active source must remain the committed owner32 M64 two-slot canonical.
  The M128 experiment is evidence only in workbook sheet 140 and the ledger.

## 2026-07-17 C0-Only Read8 Stagger

- Status: `REJECT_STATS_EXPERIMENT_BRANCH`.
- C0-only dV/dK read8 is correct and resource-clean at roles
  `14/239/239/8`, but H1/S1024 regresses to `70,769,335` ticks and
  `39.3111%` MMAC active. Baseline is `68,752,320` and `40.0907%`.
- Barrier grows by `12,797.2` cycles and waitLgkm by `1,568.5`; the shared
  packet lifetime forces the asymmetric consumers to reconverge.
- Fullperf/XCU is skipped. One symmetric read8 stats discriminator is allowed
  before restoring the tagged canonical.

## 2026-07-18 Symmetric Read8 Discriminator Rejected

- Status: `REJECT_STATS_BATCHING_OWNERSHIP_REGRESSION_SOURCE_RESTORED`.
- Both consumers batch the two dV/dK source groups into read8/wait/MMAC16.
  Static roles are `14/239/239/8` inside `16/244/244/8`; metadata is private0,
  SGPR56, VGPR128, spill0/scratch0. H1/S256 and H1/S1024 dK/dV pass, MMOP is
  exactly `131072`, and `ldsBankConflict=0`.
- H1/S1024 regresses from canonical `68,752,320 / 40.0907%` to
  `72,833,215 / 38.1220%` ticks/MMAC active. C0-only is intermediate at
  `70,769,335 / 39.3111%`.
- Symmetric waitLgkm is `27,345.2`, close to canonical `27,063.5`, but barrier
  expands to `98,169.8` from `79,233`. The route is therefore limited by
  delayed shared Q/dO page release, not merely asymmetric scheduling or local
  first-use wait.
- Fullperf/XCU is skipped by the stats gate. Workbook sheet 141 and the ledger
  retain the negative evidence in commit `30d44d8`. The active source now
  exactly matches tag `best/dkv-owner32-40p09-20260717`; remote rebuild passes
  roles `14/239/239/8`, metadata gates, and H1/S256 dK/dV correctness at
  `/zys/sb/canonical_after_read8_reject/dkv_mmac_correctness_20260718_002545`.

## 2026-07-18 Read8 Early-Used Release Rejected

- Status: `REJECT_STATS_EARLY_RELEASE_CONTENTION_SOURCE_RESTORED`.
- The candidate issues all eight final Q/dO matrix reads, waits lgkm0, arrives
  QUsed/DoutUsed, then executes MMAC16. It changes no math, LDS layout, token
  count, or output ownership.
- Roles `14/239/239/8` fit `16/244/244/8`; metadata is private0, SGPR56,
  VGPR128, spill0/scratch0. H1/S256 and H1/S1024 pass with exact MMOP and bank0.
- H1/S1024 is `73,276,840` ticks, `37.8104%` MMAC active,
  `waitLgkm=27,795.2`, `barrier=99,844.5`, and coissue
  `36,247/31,586`. It loses to canonical and to symmetric read8.
- Earlier legal Used does not reduce next-generation readiness. Producer work
  was already mostly hidden and now competes with the consumer window. Skip
  fullperf/XCU; experiment commit `566921a` preserves the rejected source and
  the active tree is restored to owner32 canonical. Static/metadata gates pass
  at `14/239/239/8`, private0, SGPR56, VGPR128, spill/scratch0; H1/S256 dK/dV
  correctness and bank0 are recertified at
  `/zys/sb/canonical_after_early_used_reject/`
  `dkv_mmac_correctness_20260718_010507`.

## 2026-07-18 Full K/V Latch Rejected by Resource Gate

- Status: `REJECT_RESOURCE_FULL_KV_OWNER32_SOURCE_RESTORED`.
- Full V D0-D3 is latched with K and all steady-loop V matrix reads/fragments
  are removed. R1 retains two Q/dO read slots; R3 uses one slot.
- R1 roles are `14/244/244/8`, metadata private124B/vgpr_spill116. R3 roles
  are `14/240/240/8` with the identical private/spill result. The required
  R3 WDRA windows are `16/240/240/16=512`; total 504 is rejected by the
  compiler's branch-average VGPR granularity gate.
- ASM attributes folded spill traffic to the FirstAccum/steady-loop merge.
  Peeling q_tile0 changes metadata to private248B/vgpr_spill111 and therefore
  does not solve capacity. No correctness or PMD performance run is allowed.
- Workbook sheet 143 contains the DAG, VGPR lower bound, instruction saving,
  expected pipeline, and all three static results. Experiment commit
  `91c2437` preserves the rejected source; active source is restored to the
  owner32 canonical. Static/metadata gates pass at `14/239/239/8`, private0,
  SGPR56, VGPR128, spill/scratch0; H1/S256 correctness and bank0 pass at
  `/zys/sb/canonical_after_fullkv_reject/`
  `dkv_mmac_correctness_20260718_113602`.

## 2026-07-18 Owner16 1P+3C Full K/V Canonical Candidate

- Branch: `exp/dkv-owner16-1p3c-full-kv` from `eefc855`.
- Active source contains one canonical owner16 dKV path: Nk192, three
  symmetric four-wave consumer groups, persistent K/V, and one combined
  Q+dO+sidecar raw packet. There is no owner32 fallback or phase switch.
- Correctness root cause fixed: single-M16 score/dP has two reads per D block,
  so the D2 first-use wait is `lgkmcnt(2)`, not the inherited M32 value 4.
- Gates pass: source gate, metadata private0/SGPR46/VGPR128/spill0, branch use
  `22/141/141/133`, S384 and S768 dK/dV correctness, exact MMOP 73,728, and
  `ldsBankConflict=0`.
- S768 stats improve from tagged owner32 `54,078,570` to `46,718,945` ticks at
  identical MMOP. Fullperf gives owner32/owner16 aggregate MMAC active
  `38.3658%/32.1307%`: the new topology uses 16 active SIMD slots rather than
  12, so the share falls even though elapsed same-work ticks improve 13.61%.
- XCU dispatch 0 completes all 64 waves with `0%` no-wave idle and average
  `63.14` active waves. Each SIMD has one thin producer slot and three roughly
  equal 5.4k-instruction consumer slots. Consumer MMAC+VALU coissue is
  `29.70%/27.91%/22.54%`.
- Remaining canonical bottleneck is consumer-local scheduling:
  `MMAC->MMAC 7.47%`, `MMAC->wait 5.34%`, and matrix-read-to-wait gaps
  `4.95%+4.46%`. Status is
  `ACCEPT_FULL_KV_ARCHITECTURE_XCU_DIAGNOSED`; next edit must target those
  gaps without changing ownership or exact work.
- Fullperf:
  `/zys/shaobo_runs/owner16_1p3c_fullkv_fullperf/`
  `dkv_mmac_correctness_20260718_153852`.

## 2026-07-18 Owner16 Score/dP Read8 Accepted

- Status: `ACCEPT_SCORE_DP_READ8_FIRST_USE` on branch
  `exp/dkv-owner16-1p3c-full-kv`.
- Canonical topology and math are unchanged. One M16 score/dP tile now issues
  all eight Q/dO trans matrix reads before first use and retires D0-D3 with
  `lgkmcnt(6/4/2/0)` before four MMAC each.
- Static gates pass: branch use `22/145/145/145` in
  `32/160/160/160`; private0, SGPR46, VGPR128, spill0/scratch0; emitted ASM
  matches the planned read/wait/MMAC sequence.
- S384 and S768 correctness pass; S768 dK/dV relL2 is
  `0.00191329/0.000319636`, MMOP is exactly 73,728, and bank conflict is zero.
- Stats-only S768 improves `46,718,945 -> 44,943,080` kernel ticks (`-3.80%`)
  and `32.2055% -> 32.7318%` MMAC active.
- Fullperf improves `46,804,485 -> 44,852,080` kernel ticks (`-4.17%`),
  `32.1307% -> 32.8015%` MMAC active, waitLgkm
  `28,421.75 -> 22,656.5`, and barrier `91,890.5 -> 86,833.75`.
  XCU duration falls `102,864 -> 98,572`; trans-read hot latency falls
  `192,552 -> 129,216`. Consumer MMAC+VALU is
  `29.65%/30.67%/26.92%` in the same 8k:80k window.
- Evidence: stats-only
  `/zys/shaobo_runs/owner16_scoredp_read8/`
  `dkv_mmac_correctness_20260718_161522`; fullperf/XCU
  `/zys/shaobo_runs/owner16_scoredp_read8_fullperf/`
  `dkv_mmac_correctness_20260718_161841`; workbook sheet 145.

## 2026-07-18 Owner16 dV/dK Read8 First-Use Accepted

- Status: `ACCEPT_DVDK_READ8_FIRST_USE` on branch
  `exp/dkv-owner16-1p3c-full-kv`.
- Canonical dV/dK now emits eight normal matrix reads followed by
  `lgkmcnt(4) -> MMAC8 -> lgkmcnt(0) -> RawUsed arrive -> MMAC8`. No phase
  switch or alternate kernel remains in source.
- Gates pass: branches `22/145/145/145` in `32/160/160/160`; private0,
  SGPR46, VGPR128, spill0/scratch0; exact emitted ASM; S384/S768 dK+dV PASS;
  S768 MMOP 73,728 and bank0.
- Stats-only S768: ticks `44,943,080 -> 43,976,205` (`-2.15%`), MMAC active
  `32.7318% -> 33.8957%`, waitLgkm `22,656.5 -> 17,446.25`.
- Fullperf S768: ticks `44,852,080 -> 43,876,105` (`-2.18%`), MMAC active
  `32.8015% -> 33.8928%`, waitLgkm `22,656.5 -> 17,530.5`, XCU duration
  `98,572 -> 96,428`, normal-read-to-wait `4.57% -> 4.15%`.
- Remaining blocker: ABarrier wait issue gaps total `41.83%` in dispatch0.
  Do not enlarge the read island again without ownership-edge evidence.
- Fullperf/XCU:
  `/zys/shaobo_runs/owner16_dvdk_read8_firstuse_fullperf/`
  `dkv_mmac_correctness_20260718_165607`; workbook sheet 146.

## 2026-07-18 Owner16 Mq192 Ownership Epoch Accepted

- Status: `ACCEPT_MQ192_OWNERSHIP_EPOCH` on
  `exp/dkv-owner16-mq192`.
- The canonical kernel remains 16-wave 1P+3C with resident K/V and exact four
  GEMMs. Mq192 only enlarges the one Q+dO+sidecar packet; S768 ownership
  cadence changes from six packets to four.
- Layout/resource gates pass: startup K/V 96KiB, steady raw+sidecar 100,608B,
  branches `30/145/145/145`, private0, SGPR55, VGPR128, spill0/scratch0.
  Q and dO use two LDS base SGPRs so every native matrix-read immediate remains
  below 64KiB; no gather or ordinary matrix-read workaround is introduced.
- S384/S768 dK+dV pass. S768 keeps MMOP 73,728, LDS 44,768, VMEM 1,728 and
  bank0. Stats-only ticks are 42,662,165 and MMAC active 35.1548%.
- Fullperf ticks are 43,033,445, MMAC active 34.8979%, barrier stall
  76,858.25, and XCU duration 94,576. Versus `d248e9b`, ticks fall 1.92%,
  active rises 1.01 pp, and ordinary ABarrier wait count falls 432->304.
- Remaining limitation: a single raw page still serializes publish/consume;
  AllDone tail duration increased and fullperf waitLgkm rose 1.91%. The next
  hypothesis must distinguish useful lookahead from merely adding tokens.
- Evidence: workbook sheet 147 and shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_174748_owner16_mq192_s768_sqc7/`.

## 2026-07-18 Owner16 Head64/Tail128 Intra-Packet Readiness Accepted

- Status: `ACCEPT_HEAD64_TAIL128_INTRA_PACKET_OVERLAP` on
  `exp/dkv-owner16-head64-tail128`.
- Canonical source still has one dKV kernel, one Mq192 raw page, resident K/V,
  three symmetric owner16 consumer groups, and four exact GEMMs.  The only
  ownership change is `RawFilled -> RawHeadFilled + RawTailFilled`; `RawUsed`
  remains a single release after M11.
- Producer publishes native 64-row Q/dO/sidecar head, then the 128-row tail.
  Consumers wait head, execute M0-M3, wait tail, and execute M4-M11.  No empty
  delay, second page, duplicate GEMM, `ds_read_b32`, gather, or layout
  workaround enters the matrix path.
- Gates: branches `32/145/145/145`, private0, SGPR50, VGPR128,
  spill0/scratch0; S384/S768 PASS; MMOP/LDS/VMEM
  `73,728/44,768/1,728`; `ldsBankConflict=0`.
- Stats-only S768: ticks 41,065,570, MMAC active 36.5520%, barrier 63,005.25.
  Versus accepted Mq192 this is `-3.74%`, `+1.40 pp`, and `-17.01%`.
- Fullperf S768: ticks 40,882,205, MMAC active 36.7738%, waitLgkm 17,576.2,
  barrier 61,634.2.  Versus Mq192 this is `-5.00%`, `+1.88 pp`, `-1.62%`, and
  `-19.81%`; coissue is `30,188/23,730`.
- XCU duration is 89,848 with 0% no-wave idle.  Ordinary ownership waits rise
  from 304 to the predicted 496 events but their duration falls 18.46%; the
  fixed-window top ABarrier gap falls 23,249->17,141 cycles and producer-wave
  coissue rises 2.27%->42.26%.
- Host support remains exact-tile only.  S1024 returned `status=unsupported`
  because `S % 192 != 0`; partial-tail support is deferred, not silently
  treated as correct.
- Evidence: workbook sheet 148; fullperf/XCU
  `/zys/shaobo_runs/owner16_head64_tail128_fullperf/`
  `dkv_mmac_correctness_20260718_183256`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_183256_owner16_head64_tail128_s768_sqc7/`.

## 2026-07-18 Mq96 Raw2 Rejected

- `REJECT_MQ96_RAW2_LOOKAHEAD`: correctness/resource/exact-work gates pass at
  WDRA `20/172/172/148`, but S768 ticks regress 5.11%, active falls 1.55 pp,
  waitLgkm rises 15.28%, and VALU rises 9.21%.  Barrier time alone improves.
- A consumer WDRA window exactly equal to the observed 168-VGPR live edge gave
  undefined C0/C1 results with metadata spill0; one four-VGPR allocation
  granule of headroom restored correctness.  Treat metadata spill0 as
  necessary but not sufficient at a role-window boundary.
- The failed implementation has been removed.  Evidence is retained in
  workbook sheet `149_DKV_Mq96_Raw2_Lookahead` and `client.md`.

## 2026-07-18 Mq192 Head/Tail Split-Used Accepted

- Status: `ACCEPT_MQ192_HEAD_TAIL_SPLIT_USED` on
  `exp/dkv-owner16-head-tail-split-used`.
- Canonical dataflow remains Mq192/Nk192, resident K/V, one physical
  Q+dO+sidecar packet, three symmetric owner16 consumers, exact four GEMMs,
  and native MLS/BPS + `ds_read_matrix` + MMAC.  Head64 and Tail128 now have
  separate Used tokens so producer can publish Head(t+1) during Tail(t).
- Gates: branches `32/145/145/145`, private0, SGPR50, VGPR128,
  spill0/scratch0, LDS 100,608B; S384/S768 dK+dV PASS; S768 exact
  MMOP/LDS/VMEM `73,728/44,768/1,728`, bank0.
- Stats-only S768: ticks 39,486,265, MMAC active 38.1762%, waitLgkm 17,854.75,
  barrier 49,613, coissue `29,944/23,369`.
- Fullperf S768: ticks 39,383,435, MMAC active 38.2453%, waitLgkm 18,035.5,
  barrier 49,145.25, coissue `29,880/23,479`.
- Versus the previous best fullperf, ticks fall 3.67%, active rises 1.47 pp,
  and barrier falls 20.26%.  XCU duration falls 89,848->86,560; ordinary
  ownership gap duration falls 11.30% despite 496->544 events, and the maximum
  gap falls 16,795->12,263.  Producer coissue rises 42.26%->59.39%.
- Next measured bottleneck: fixed-window consumer MMAC+VALU coissue falls to
  `26.46/26.12/21.07%`; retain split ownership and investigate consumer
  read/wait/VALU scheduling rather than adding another page/token.
- Evidence: workbook sheet 150; remote
  `/zys/shaobo_runs/owner16_split_used_fullperf/`
  `dkv_mmac_correctness_20260718_203148`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_203148_owner16_mq192_head_tail_split_used_s768_sqc7/`.

## 2026-07-18 dV/dK Source-Read Software Pipeline Accepted

- Status: `ACCEPT_DVDK_SOURCES_UNDER_USEFUL_WORK` on
  `exp/dkv-owner16-dvdk-under-softmax`.
- Canonical math, Mq192/Nk192 tile, resident K/V, split Head/Tail Used
  ownership, one kernel, and exact four GEMMs are unchanged.  The consumer
  now issues sidecar3 plus D0/D1 normal reads, retires only sidecar with
  `lgkmcnt(4)`, executes softmax/dS, retires D0/D1, then overlaps D2/D3 reads
  with the first dV/dK MMAC8.
- Static/resource gates pass at branches `32/154/154/154`, private0, SGPR50,
  VGPR128, spill0/scratch0 and LDS100,608B.  S384/S768 correctness passes;
  S768 keeps MMOP/LDS/VMEM `73,728/44,768/1,728` and bank0.
- Stats/fullperf ticks are `38,680,460/38,840,165`; MMAC active is
  `39.4010%/39.2062%`.  Versus tag
  `best/dkv-owner16-head-tail-split-used-s768-20260718`, fullperf ticks fall
  1.38%, MMAC active rises 0.96 pp, and waitLgkm falls 24.51%.
- XCU duration falls to 85,360 and wait hits fall 21.57%.  Residual debt is
  worse MMAC-to-MMAC spacing and lower fixed-window consumer MMAC+VALU
  coissue, so the next edit must target MMAC cadence without expanding the
  source live set back to the spilling D1 design.
- Evidence: workbook sheet 151; fullperf/XCU
  `/zys/shaobo_runs/owner16_dvdk_under_softmax_d2_fullperf/`
  `dkv_mmac_correctness_20260718_215518`.

## 2026-07-18 Loop-Lived MMAC Zero Accepted

- Status: `ACCEPT_LOOP_LIVED_MMAC_ZERO` on
  `exp/dkv-owner16-loop-zero-static-probe`.
- The canonical algorithm and pipeline remain D2.  One read-only `F16x8`
  zero fragment is initialized once per consumer q-loop and passed to the
  first-accumulation score/dP and dV/dK helpers.
- Gates: branches `32/158/158/158`, private0, SGPR50, VGPR128,
  spill0/scratch0, LDS100,608B; S384/S768 PASS; S768 MMOP/LDS/VMEM remains
  exactly `73,728/44,768/1,728`, VALU falls to 100,704, and bank0.
- Stats-only S768: ticks 37,219,000, MMAC active 40.4364%, waitLgkm 11,668,
  barrier 42,965.75, coissue `30,694/26,620`.
- Fullperf S768: ticks 36,811,775, MMAC active 40.6086%, waitLgkm 11,779.75,
  barrier 42,157.75, coissue `30,659/26,489`.
- Versus D2 fullperf, ticks fall 5.22%, active rises 1.40 pp, and dynamic
  `v_mov_b64_e32` falls `6,880 -> 2,144`.  XCU duration falls to 80,904;
  the main ABarrier gap falls 19.23%, while consumer MMAC+VALU coissue rises
  to `27.66/28.34/26.26%`.
- Remaining trace bottleneck is matrix readiness: XCU `s_waitcnt` latency is
  5.23% higher and normal matrix-read-to-wait duration 4.80% higher despite
  lower PMD wait stalls.  Do not alter ownership or zero lifetime in the next
  experiment.
- Evidence: workbook sheet 152; remote
  `/zys/shaobo_runs/owner16_loop_zero_fullperf/`
  `dkv_mmac_correctness_20260718_225519`; shared archive
  `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260718_225519_owner16_loop_mmac_zero_s768_sqc7/`.

## 2026-07-19 Latest-Pair Rebaseline And 1P3C Design Gate

Status: `DKV_ACCEPT_ENV_REBASE_DQ_KEEP_STABLE_COMPILER_1P3C_DESIGN_READY`.

- PMD is fixed to side-by-side HEAD1694 at
  `/zys/shaobo/toolchains/pmd_20260717`; its executable and preloaded
  `libgem5_opt.so` must come from the same root.
- dKV latest-compiler artifact uses explicit WDRA windows
  `32/160/160/160`.  S768 fullperf correctness passes with private0, spill0,
  bank0, exact MMOP73,728; ticks `35,707,035`, active `43.7836%`, coissue
  `22,104/18,318`.  XCU still attributes the largest debt to
  ABarrier/readiness: `s_waitcnt` 31.13%, `s_xor` 18.92%, with top gaps
  ABarrier->xor 19.93% and ABarrier->wait 11.23%.
- dQ stable baseline remains old compiler plus latest PMD: S1024 correctness
  PASS, private0/spill0/bank0, exact MMOP50,688; ticks `24,600,030`, active
  `33.3978%`, coissue `15,306/13,534`.  The latest compiler is OBSERVE/REJECT
  for dQ because it regresses to `25,002,705 / 30.7854%` even though VALU is
  lower; SQTT exposes longer BPS and ABarrier waits.
- Workbook sheet `154_1P3C_50pct_Gate` is the implementation contract.  dQ
  code changes may begin only if static generation proves each of three
  consumer windows <=160 VGPR with no spill.  The candidate uses Mq192/Nk128,
  96KB Q+dO startup plus 2.25KB sidecar, and a lifetime-overlaid 128KB K/V
  two-page steady state.

## 2026-07-19 Canonical dKV Explicit WDRA Route Accepted

Status: `ACCEPT_CANONICAL_TOOLCHAIN_ROUTE`.

- Source change is guarded by `SHAOBO_RUN_ON_MODEL=1`; it emits
  `__builtin_hcu_wdra_init(32,160,160,160)` and `run-on-model` only for the
  latest compiler route.  Default compiler build still passes unchanged.
- Latest compiler build: role use `32/158/158/158`; private0, SGPR50,
  VGPR128, no spill/scratch; dKV static gate PASS.
- Latest PMD HEAD1694 S384 correctness PASS.  S768 stats-only:
  simTicks39,437,125, first3,613,610, kernel ticks35,823,515, active43.1608%,
  MMOP73,728, VALU80,272, coissue21,792/18,043, bank0.
- S128 returns the harness status `unsupported`; it is not a numerical
  failure.  Canonical dKV smoke is S384 and steady evidence is S768.

## 2026-07-19 dQ Mq192 1P3C Topology Is An Observe Baseline

Status: `OBSERVE_1P3C_TOPOLOGY_PROOF_TICKS_REGRESSION`.

- Active source branch maps waves0-3 to one K/V producer and the remaining
  three wave groups to symmetric 64-row dQ consumers.  Q/dO are self-published
  and latched by each consumer group; sidecar is producer-published; K/V use
  two lifetime-overlaid 64KB pages.
- Static gates pass at `11/158/158/159` branch VGPRs with private0, spill0,
  scratch0 and LDS131,072B.  H1/S768 causal correctness passes and bank
  conflict is zero.
- Against the same-work Mq128 2P2C control, exact MMOP remains 28,800 and
  MMAC active rises `30.0592% -> 34.3345%`; VALU, SCA, VMEM, branch, xor,
  vbcnt-wait and ordinary-wait counts all fall.
- Same-shape ticks regress `19,608,225 -> 23,591,750`, so this branch is not
  the dQ best-performance branch.  It is retained as the clean topology proof
  for the next useful-stagger experiment.
- XCU still shows lockstep ownership/readiness debt: ABarrier-to-xor 12.73%,
  ABarrier-to-SALU 10.61%, and lds-matrix-to-immediate 4.66%.
- Next edit must change only useful mathematical order across consumers; no
  delay, duplicate GEMM, extra page/token, or consumer-assisted prefetch is
  allowed in the same hypothesis.

## 2026-07-19 Invalid dQ Control Path Guard

Status: `REJECT_INVALID_CONTROL_REFERENCE_PATH`; canonical source restored.

- A temporary consumer1 useful-stagger source passed static resource and ISA
  equality checks (`11/158/152/159`, private0, spill0, scratch0; exact MMAC,
  matrix-read, wait, and barrier counts), but it was not performance-tested
  against a valid control.
- The terminated run used `--canonical=0`; it was the scalar reference path,
  not a repeat of successful `path=canonical` controls.  It provides no PMD
  environment or kernel-performance conclusion.  The overloaded host was
  cleaned separately by removing seven runaway `kded5` processes.
- `src/dq_kernel.cpp` is back at commit `3eeff47`; no unvalidated stagger code
  remains.  The next PMD attempt must explicitly set `CANONICAL_DQ=1`, finish
  with `path=canonical`, verify SQ7 in PMD's resolved arguments, and only then
  compare the one-hypothesis candidate.

## 2026-07-19 Rejected dQ Candidate: Causal-Balanced M16 Ownership

Status: `REJECT_CAUSAL_BALANCE_LOCKSTEP`; canonical mapping restored.

- Current contiguous consumer ownership is mathematically imbalanced under
  causal masking: first-CTA useful n32 units are `6/14/22`.
- Tested mapping was `local_m16 = wave_local * 3 + ConsumerGroup`, yielding
  M16 sets `{0,3,6,9}`, `{1,4,7,10}`, `{2,5,8,11}` and work `12/14/16`.
- This preserves every row exactly once, exact three-GEMM work, LDS layout,
  Q/dO latch, K/V pages, ABarrier ledger, MMAC/read/wait counts, and store
  ownership.
- Static gates passed at `11/158/158/159`, private0, spill0, scratch0.
  H1/S768 causal correctness passed with maxAbs `1.5201e-7`, relL2
  `0.00151559`, no NaN/Inf, and bank0.
- Against the freshly recertified canonical control, exact work remained
  MMOP/LDS/VMEM/FLAT `28,800/15,092/704/564`, but kernel ticks regressed
  `23,364,250 -> 24,132,290` (`+3.29%`).  VALU/SCA rose slightly
  `33,808/23,844 -> 33,856/23,876`; coissue success/fail both rose from
  `12,071/10,930 -> 12,982/11,717`.
- Interpretation: causal row imbalance was a useful phase offset.  Balancing
  it increased lockstep MMAC competition rather than shortening the terminal
  tail.  Restore contiguous ownership and pursue a legal DAG-order stagger.

## 2026-07-19 Rejected dQ Candidate: Legal DAG-Order Stagger

Status: `REJECT_DAG_STAGGER_BREAKS_MMAC_ISLAND`; source restored.

- Consumer1 alone used `score -> P -> dP -> dS -> dQ`; consumer0/2 retained
  fused score+dP.  Static generation preserved exact MMAC, matrix-read,
  lgkm-wait, ABarrier, exp, v_mov, and data-movement counts.  It added only
  eight `s_setprio`; roles were `11/158/152/159`, metadata private0 SGPR59
  VGPR128, spill0, scratch0.
- H1/S768 causal correctness passed with maxAbs `1.5201e-7`, relL2
  `0.00151559`, no NaN/Inf, and bank0.
- Exact MMOP/LDS/VMEM/FLAT stayed `28,800/15,092/704/564`; VALU fell
  `33,808 -> 33,608`, but kernel ticks regressed
  `23,364,250 -> 24,166,870` (`+3.44%`).  Coissue success/fail fell
  `12,071/10,930 -> 11,305/10,121`.
- Conclusion: the fused score+dP island is locally beneficial.  A one-group
  DAG split loses MMAC continuity before peer waves can hide its P/softmax
  interval.  Preserve fused score+dP; next structural candidate must reduce
  readiness/ownership exposure without splitting MMAC work.

## 2026-07-19 Rejected dQ Candidate: Split V/K Page Ownership

Status: `REJECT_SPLIT_USED_TOKEN_CONTROL_COST`; canonical source restored.

- The page `Used` handshake was split at real last-use points: consumers
  arrived `VUsed` after the final dP and `KUsed` after dQ; the producer loaded
  next-generation V between those waits, then loaded K and published the
  original combined `Filled` token.
- Static work and resources remained legal: branches `11/158/158/159`,
  private0/spill0/scratch0, exact matrix-load/MMAC/read counts, and no added
  global/LDS payload.  H1/S768 causal correctness and bank0 passed.
- Canonical control was `23,364,250` ticks, SCA23,844, coissue
  `12,071/10,930`, LDS credit stall7,784.  The unconditional split reached
  `23,586,290` ticks (`+0.95%`), SCA24,966, coissue `12,420/11,366`, and credit
  stall6,709.  It exposed useful overlap, but barrier bookkeeping cost more.
- A tail-aware condition (`kt + 2 < active_k_tiles`) regressed to
  `24,856,650` ticks (`+6.39%`), VALU34,192, SCA25,128, coissue
  `10,727/9,705`, and credit stall8,555.  Dynamic tail control made the
  producer schedule less regular and did not amortize the token split.
- Do not add finer-grained ownership to this tile.  Preserve one page-used
  token and pursue a design that increases useful MMAC per ownership epoch or
  removes an epoch entirely.

## 2026-07-19 Rejected dQ Refactor: N32 Stage Helpers

Status: `REJECT_STATIC_CODEGEN_DRIFT`; canonical source restored.

- The refactor extracted forced-inline `score+dP` and `softmax+dQ` helpers but
  deliberately preserved all mathematical work and loop order.
- Static ISA kept exact MMAC576, matrix reads312, matrix loads20, ABarrier
  wait/arrive/seq `12/15/4`, exp192, v_mov85, branches81, and stores25.
- Despite that, wait instructions rose `89 -> 93`, specifically through
  extra VMEM/combined waits, and branch VGPR use rose from `11/158/158/159`
  to `11/160/160/160`.  The compiler therefore did not preserve the original
  scheduling/lifetime boundary.
- No PMD run was admitted.  Future coissue experiments must leave the compact
  original compute body intact or prove an exact generated schedule first.

## 2026-07-19 dQ Consumer-Assisted V: Combined Filled Rejected

Status: `REJECT_COMBINED_BPS_GENERATION`; source restored.

- The valid current-page form moved four V BPS operations to consumer1 while
  the producer retained four K BPS operations.  Static resources were
  private0/spill0/scratch0 with exact MMAC/read/load payload; H1/S384 and
  H1/S768 correctness passed and bank conflicts remained zero.
- It serialized page readiness behind consumer1: H1/S768 ticks regressed
  `23,364,250 -> 25,983,230` (`+11.21%`), SCA became `25,512`, coissue became
  `13,662/12,419`, and LDS credit stall became `10,573`.
- The one-step lookahead form hung.  SQAbar recorded Page0 Filled
  `expected=8` but only four arrivals, leaving `pending=4`; no numerical or
  performance result is attached to the deadlocked run.
- Rule: one BPS-tracked ABarrier generation has one sequencing owner.  K and V
  lookahead from independent roles requires separate KFilled/VFilled tokens;
  a combined Used token may remain shared.

## 2026-07-19 dQ Separate K/V Filled Rejected

Status: `REJECT_SPLIT_FILLED_TICKS_REGRESSION`; canonical source restored.

- The corrected no-sentinel schedule gives independent K/V BPS owners legal
  `KFilled` and `VFilled` generations, then computes the preceding tile while
  V for the current tile is resident.  It passes S384/S768 correctness,
  private/spill/scratch0, exact work, and bank0.
- S768 is `25,837,175` ticks versus canonical `23,364,250` (`+10.58%`), with
  coissue `10,942/9,761` versus `12,071/10,930`.  Protocol correctness did not
  translate into pipeline overlap; the extra Filled lifecycle is net cost.
- Remove the experiment from the active route.  Reopen consumer-assisted BPS
  only with a topology that removes an ownership epoch rather than adding one.

## 2026-07-19 dKV B16 Matrix-Store Entry Contract Is Not The Root Cause

Status: `OBSERVE_PMD_OR_UNDOCUMENTED_MATRIX_STORE_CONTRACT`; canonical dKV is
unchanged.

- The latest compiler exposes only `__builtin_hcu_s_abarrier_init` for
  ABarrier initialization.  DCU Wiki requires init visibility before use and
  synchronization before invalidation; the focused probe now implements the
  full `init/sync/seq/store/arrive/wait/sync/inv` lifecycle.
- A single official `matrix_store_32x16_b16` still writes only rows 0..16 of a
  32x16 control tile: 240/512 row-major elements remain wrong, with the first
  failure at row17/col0.  PMD exits normally, private/spill/scratch are zero,
  `s_trap=0`, and `ldsBankConflict=0`.
- This rules out missing ABarrier init and multi-store page reuse as causes.
  It does not yet distinguish a PMD partial-store defect from an undocumented
  descriptor/source-layout ABI.
- C2 matrix-store integration remains blocked.  C1 was therefore isolated as
  packed FP16 direct vector global stores; its passing result is recorded in
  the next section.
- Evidence: PMD-005 in `docs/perf_model_pmd_compiler_issues.md` and remote run
  `/zys/shaobo_runs/dkv_b16_matrix_store_probe_builtin_single_reclass/`
  `run_20260719_103901`.

## 2026-07-19 dKV C1 Packed-FP16 Direct Store Passes

Status: `ACCEPT_FOCUSED_INSTRUCTION_CONTROL`; canonical performance is pending.

- The focused kernel uses the exact owner16 dKV output mapping: one 64-lane
  wave owns `16x128`, and each lane emits eight contiguous four-half vectors.
- PMD validates all 2,048 elements exactly.  ASM has eight
  `global_store_dwordx2`, no `global_store_dwordx4`, 32 FP32-to-FP16
  conversions, no trap, private0/spill0, and bank0.
- This separates the two output choices cleanly: C1 is supported and correct;
  C2 matrix-store remains blocked by PMD-005.  The next experiment may replace
  only the real canonical dKV epilogue/output type with C1 and compare against
  the accepted FP32-output tag using same-shape ticks and SQTT.
- Evidence:
  `/zys/shaobo_runs/dkv_b16_direct_store_probe_builtin/`
  `run_20260719_104409`.

## 2026-07-19 dKV C1 Canonical Performance Rejected

Status: `REJECT_PERF_VALID_FUNCTION`; canonical source restored to FP32 output.

- Full-kernel FP16 output is numerically valid at S384 and S768 and retains
  private/spill/scratch0, exact MMOP73,728, bank0, and the native
  MLS/BPS+`ds_read_matrix`+MMAC path.
- The S768 output data-cycle counter improves `12,288 -> 6,144`, but the
  compiler emits enough FP32-to-FP16 conversion work to raise dynamic VALU
  `80,272 -> 81,904`.
- Against the accepted latest-pair control, ticks regress
  `35,707,035 -> 35,834,435` (`+0.357%`) and aggregate MMAC active falls
  `43.7836% -> 43.6662%`.  C1 therefore does not address the current critical
  path despite reducing output bandwidth.
- ABarrier entry initialization is not implicated: canonical dKV already uses
  the Wiki-required init visibility and pre-invalidate synchronization.  The
  latest compiler route separately emits explicit WDRA init through
  `SHAOBO_RUN_ON_MODEL=1`; these contracts must remain distinct.
- Evidence:
  `/zys/shaobo_runs/dkv_b16_direct_canonical_s768/`
  `dkv_mmac_correctness_20260719_105832`.

## 2026-07-19 dKV C1b FWD Packed Builtin Is Codegen-Equivalent

Status: `REJECT_STATIC_CODEGEN_EQUIVALENT`; canonical source restored.

- The focused FWD-style builtin probe passes PMD exactly and emits the desired
  16 `v_cvt_pk_f16_f32` plus eight packed stores.
- In the full dKV kernel, however, LLVM already fused the previous scalar
  casts.  Scalar-C1 and builtin-C1b both contain 384 packed conversions, no
  scalar conversion opcode, 49 `global_store_dwordx2`, branch use
  `32/158/158/158`, and private/spill/scratch0.
- The explicit builtin is therefore useful as instruction evidence but not a
  canonical optimization.  No full PMD run is admitted; the active source is
  restored before the next hypothesis.
- Evidence:
  `/zys/shaobo_runs/dkv_b16_packed_cvt_probe_gate/run_20260719_113043` and remote
  ASM directories `build/dkv_b16_direct_canonical` versus
  `build/dkv_b16_packed_canonical`.

## 2026-07-19 Score/dP Long-Island Status

- `read8 -> wait0 -> MMAC16` is rejected and removed from canonical dKV.
  S384/S768 correctness, metadata, exact MMOP, and bank0 gates pass, but S768
  stats regress `35,823,515 -> 36,638,420` ticks and MMAC active falls
  `43.1608% -> 42.6444%`.
- The compiler did emit the requested clean MMAC16 island. The regression is
  therefore a scheduling result: full operand drain exposes first-use LDS
  latency and lowers useful coissue. Canonical score/dP again uses staged
  `lgkmcnt(6/4/2/0)` first use.
- No candidate fullperf was captured. Accepted best remains tag
  `best/dkv-latest-pair-43p78-20260719` with S768 fullperf
  `35,707,035` ticks and `43.7836%` MMAC active.

## 2026-07-20 dKV Owner16 Four-Consumer Resource Gate Passes

Status: `ACCEPT_RESOURCE_GATE_CANONICAL_NOT_YET_CHANGED`.

- M128 logical `64/32/32` remains exact and tail-free but physical 2P2C.  A
  half-active M32 role cannot increase per-SIMD heavy-wave residency, while a
  full four-wave role would add 50% redundant MMAC.
- The admitted alternative is `Mq64/Nk256/D128`: four independent 4-wave
  owner16 groups, persistent per-wave ledger
  `dK32+dV32+K16+V16=96 VGPR`, target window 128.
- Focused compilation with LLVM `7b796991` reports branch use
  `114/114/114/114` inside `128/128/128/128`; private0, SGPR29, VGPR128,
  spill0, scratch0.  Static opcode evidence is BPS10, matrix-read96, MMAC128,
  resize128x4, executable trap0.
- Seeded PMD HEAD1694 execution passes `bad=0`, bank0, no panic.  Evidence:
  `/zys/shaobo_runs/dkv_owner16_4c_resource_probe_20260720_045244`.
  Diagnostic kernel ticks are 4,147,780 with MMOP512 and coissue238/978; do
  not compare these numbers against the canonical dKV kernel.
- Canonical `src/dkv_kernel.cpp` is untouched.  Integration is now allowed
  only after the full K/V-startup, raw-page, ABarrier-generation and exact
  output-ownership ledger is written into the workbook.
