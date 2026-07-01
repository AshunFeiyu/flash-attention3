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
