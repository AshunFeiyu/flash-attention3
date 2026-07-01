# Source Status

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
