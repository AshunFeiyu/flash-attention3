# Source Status

## 2026-07-01 Clean Scaffold

Status: `BRINGUP_ONLY`

The clean repo now owns a real Stage61 dKV FWD-style scaffold:

- thin C ABI in `include/shaobo_fa3_api.h`
- standalone HIP launcher in `src/stage61_dkv_fwdstyle.cpp`
- 16-wave WDRA kernel with four explicit role branches
- branch-local producer/consumer VGPR windows
- ABarrier ownership ledger
- static scaffold gate and PMD smoke wrapper
- PMD smoke wrapper treats model abort text as failure even if `run.py` exits 0
- scaffold smoke uses stdout `stage61_fwdstyle_scaffold status=success` as the
  pass signal; empty scaffold `stats.txt` is not performance evidence

This scaffold intentionally does not compute dV/dK yet.  It is the clean
execution shell that the score/dP, softmax/dS, dV/dK, and store blocks will be
migrated into one at a time.

Bring-up note:

`s_abarrier_inv` is intentionally absent in this scaffold.  PMD showed that a
single wave can invalidate `kAllDone` after its wait returns while peer waves
have not reached the same wait yet.  Reintroduce invalidation only with a
separately verified quiescence pattern.

Verified evidence:

- Remote repo: `/zys/shaobo/fa3_bwd_wasp_fwdstyle_clean`
- PMD smoke:
  `/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean/stage61_clean_scaffold_20260701_191726`
- Metadata gate:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `sgpr_count=8`, `vgpr_count=84`
- Runtime signal:
  `stage61_fwdstyle_scaffold status=success B=1 H=1 S=1024 D=128`
- Design workbook:
  `/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_stage61_cleanrepo_s0_20260701.xlsx`

Next implementation hypothesis:

`Port the producer packet publishers first, preserving the current scaffold's
four-role WDRA topology and keeping all generated run artifacts outside git.`

## 2026-07-01 S2 Score/Dp MMAC Probe

Status: `BRINGUP_ONLY`

The clean repo now has a real FA FWD-style code shape instead of an empty
barrier scaffold:

- instruction helpers live in `include/stage61_fwdstyle_instr.h`
- `src/stage61_dkv_fwdstyle.cpp` is organized as producer loops, consumer loop,
  score/dP MMAC probe, and execution kernel
- producer0 publishes Q + K via `matrix_load_32x32_b16 ... bps lds`
- producer1 publishes dO + V via `matrix_load_32x32_b16 ... bps lds`
- consumer groups wait Q, dO, K, V ownership tokens and run
  `ds_read_matrix_trans_format` plus `v_mmac_*lit`
- standalone allocates real Q/K/V/dO buffers before PMD

Verified evidence:

- Remote repo: `/zys/shaobo/fa3_bwd_wasp_fwdstyle_clean`
- PMD smoke:
  `/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean/stage61_clean_s2_score_dp_probe_20260701_204940`
- Metadata gate:
  `private_segment_fixed_size=0`, `sgpr_spill_count=0`,
  `vgpr_spill_count=0`, `sgpr_count=30`, `vgpr_count=84`
- Runtime signal:
  `stage61_fwdstyle_s2 status=success B=1 H=1 S=1024 D=128`
- Stats:
  `simTicks=7237685`, `MMOP=2048`, `ldsBankConflict=0`,
  `mmop_active_share=6.48%`
- PMD warning to investigate:
  `warn: read vgpr184 before writing`

This is not a dKV performance candidate.  It proves the clean repo can emit
and run the FWD-style packet + matrixized score/dP path without scratch, spill,
or LDS bank conflict.  Next implementation should add sidecar and a real q-loop
while preserving this file structure.
