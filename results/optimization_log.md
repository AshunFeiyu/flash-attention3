# Optimization Log

## 2026-07-01 Stage61 Clean Scaffold

Decision: `BRINGUP_ONLY`

Hypothesis:

`A clean FA3 FWD-style execution shell with explicit WDRA role branches and
ABarrier ownership gates should be buildable and PMD-runnable before any dKV
math body is ported.`

Implemented:

- Added a standalone clean Stage61 scaffold kernel.
- Added build, static gate, and PMD smoke wrapper.
- Kept the historical Cxx phase stack out of the clean repo.
- Removed scaffold-only post-role diagnostic stores after asm showed they caused
  `private_segment_fixed_size=32` and `vgpr_spill_count=17`.
- Removed scaffold barrier invalidation after PMD reported
  `ABARRIER_ILL_OP_ERROR` on `kAllDone`; bring-up barriers are now allowed to
  retire at kernel end until a verified quiescence pattern is introduced.
- Hardened the smoke wrapper to fail when PMD stdout contains
  `panic`, `Program aborted`, `core dumped`, or `Aborted`, because `run.py` can
  return success after a model abort.
- Scaffold smoke now requires stdout
  `stage61_fwdstyle_scaffold status=success`; its `stats.txt` can be empty and
  is not treated as performance evidence.

Required evidence before moving math:

- build/asm PASS
- scaffold static gate PASS
- symbol metadata no private segment, no SGPR/VGPR spill
- PMD scaffold smoke reports `stage61_fwdstyle_scaffold status=success`

Promotion note:

This is not a performance candidate and must not get a perf-ledger row.  The
first real candidate begins when producer packet publication and one consumer
MMAC island are ported with correctness evidence.

Verified:

- Remote build/gates PASS in `/zys/shaobo/fa3_bwd_wasp_fwdstyle_clean`.
- PMD smoke PASS at
  `/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean/stage61_clean_scaffold_20260701_191726`.
- Symbol metadata:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `sgpr_count=8`, `vgpr_count=84`.

## 2026-07-01 Stage61 Clean S2 Score/Dp Probe

Decision: `BRINGUP_ONLY`

Hypothesis:

`The clean repo should use FA FWD-style structure directly: small instruction
helpers, explicit producer loops, explicit consumer loop, ABarrier phase
ledger, and a real matrixized score+dP island before full dKV math.`

Implemented:

- Added `include/stage61_fwdstyle_instr.h` with minimal FWD-style wrappers for
  ABarrier phase ops, `s_setprio`, `matrix_load_32x32_b16 ... bps lds`,
  `ds_read_matrix_trans_format`, and `v_mmac_*lit`.
- Replaced the empty scaffold body with S2:
  producer0 publishes Q+K, producer1 publishes dO+V, consumers run a score+dP
  MMAC probe and write a diagnostic workspace value.
- Updated the static gate to require real matrix path evidence rather than only
  role branches.
- Updated standalone to allocate real Q/K/V/dO device buffers before PMD.

Verified:

- Remote build PASS with asm.
- Static gate PASS.
- Symbol metadata PASS:
  `private_segment_fixed_size=0`, `sgpr_count=30`, `vgpr_count=84`,
  `sgpr_spill_count=0`, `vgpr_spill_count=0`.
- PMD smoke PASS:
  `/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean/stage61_clean_s2_score_dp_probe_20260701_204940`
- PMD stats:
  `simTicks=7237685`, `firstWaveStartTick=3613610`,
  `lastWaveEndTick=7237685`, `MMOP=2048`, `ldsBankConflict=0`,
  `mmop_active_share=6.48%`.

Notes:

- This is not promoted as a performance candidate because it is only a
  score+dP probe, not full dKV.
- PMD reports `warn: read vgpr184 before writing`; this did not abort, but it
  should be investigated before relying on the probe for SQTT conclusions.
- Next candidate should keep the clean FWD-style structure and add sidecar +
  real q-loop, then softmax+dS, then dV/dK accumulation and stores.
