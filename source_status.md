# Source Status

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
