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
