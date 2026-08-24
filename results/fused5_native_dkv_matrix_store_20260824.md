# Fused5 Native dK/dV Matrix Store

Status: `ACCEPT_NATIVE_DKV_MATRIX_STORE_NEW_GLOBAL_BEST`.

## Change

- Parent is C83 commit `73e8119`; implementation commit is `3388f47`.
- dK/dV accumulate with native FP16-output MMAC fragments.
- At the end of the q-loop, the kernel reuses 16 KiB of the released V LDS
  region for four independent 32-row owner pairs.
- Each pair writes dK/dV fragments with
  `ds_write_matrix_32x16_trans_f16`, then one wave issues
  `matrix_store_32x32_b16`.
- GQA partial dK/dV workspace and its reduction output are FP16. The reducer
  still accumulates each head group in FP32.
- C83's causal specialization, tile, five-GEMM DAG, q-loop schedule and dQ
  reduction are unchanged.

## Gates

- Static resources: causal SGPR70/VGPR128; noncausal SGPR71/VGPR128;
  private segment, SGPR spill and VGPR spill are all zero.
- WDRA role use: causal `9/141/87/130`, within `16/204/88/204`.
- Correctness PASS: causal H1/S128, causal H1/S1024, noncausal H1/S128,
  and GQA Hq4/Hkv2/S128.
- PMD: no panic, no VGPR warning, `ldsBankConflict=0`.
- Generated ISA contains FP16 MMAC, `ds_write_matrix_format` and
  `matrix_store_32x32_b16`.

## Performance

Same compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7,
H1/S1024/D128 causal:

| Metric | C83 | Native store | Delta |
| --- | ---: | ---: | ---: |
| stats-only fused ticks | 40,884,935 | 40,252,940 | -1.546% |
| stats-only lifecycle ticks | 44,969,470 | 44,272,865 | -1.549% |
| fullperf fused ticks | 41,167,035 | 39,706,485 | -3.548% |
| MMAC active | 36.579709% | 38.403324% | +1.823615 pp |
| MMOP | 88,064 | 88,064 | 0 |
| FLAT | 3,616 | 2,592 | -1,024 |
| VALU | 90,032 | 89,168 | -864 |
| SCA | 35,592 | 39,176 | +3,584 |
| LDS | 61,056 | 61,568 | +512 |

XCU shows the new tail is not free: terminal `s_ebarrier_sync` contributes a
visible issue gap. The native FP16 accumulation/store chain nevertheless wins
because it removes the old per-lane FP32 global-store drain. The next bounded
hypothesis is to reduce only this terminal writer-page synchronization.

Evidence:

- Remote run:
  `/zys/sb/runs/c83_native_dkv_matrix_store_3388f47`
- Shared perf:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260824_213327_C84_C83_native_dkv_matrix_store_H1S1024_causal_SQ7`
