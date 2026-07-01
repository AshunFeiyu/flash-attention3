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
