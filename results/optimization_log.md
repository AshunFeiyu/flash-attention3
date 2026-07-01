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
