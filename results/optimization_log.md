# Optimization Log

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
