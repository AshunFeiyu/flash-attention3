# Optimization Log

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
