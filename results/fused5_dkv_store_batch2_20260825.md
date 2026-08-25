# Fused5 dK/dV Two-D-Block Store Batch

Status: `ACCEPT_SMALL_TICKS_AND_ACTIVE_MMAC50_OPEN`.

## Hypothesis

C84 reuses 16 KiB of released V LDS but publishes and stores one D32 block per
owner pair at a time. Reuse the full 32 KiB V region so each owner pair can
publish two D32 blocks before one store epoch. This halves terminal ownership
generations without changing the five-GEMM mainloop.

## Change

- Commit: `0f3527e`.
- Tile, MMOP work, 16-wave roles, ABarrier ledger and output ownership unchanged.
- Terminal writer scratch: 16 KiB -> 32 KiB, still inside released V LDS.
- Four D blocks are stored in two batches instead of four independent epochs.

## Gates

- Static: causal/noncausal SGPR70/71, VGPR128; private/spill/scratch all zero.
- WDRA role usage unchanged at `9/141/87/130` for causal.
- Correctness PASS: causal S128/S1024, noncausal S128 and GQA Hq4/Hkv2/S128.
- PMD warning/panic zero; `ldsBankConflict=0`; dynamic MMOP remains 88,064.

## Performance

Same compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7,
H1/S1024/D128 causal:

| Metric | C84 control | C85 batch2 | Delta |
| --- | ---: | ---: | ---: |
| stats-only fused mean, 3 runs | 39,558,003 | 39,318,977 | -0.604% |
| stats-only fused median | 39,594,555 | 39,301,535 | -0.740% |
| fullperf fused ticks | 39,785,200 | 39,308,360 | -1.198% |
| MMAC active | 38.392786% | 39.054060% | +0.661274 pp |
| barrier share | 15.518838% | 15.267943% | -0.250895 pp |
| SCA | 39,176 | 38,344 | -832 |

XCU confirms the intended mechanism:

- dynamic `s_ebarrier_sync`: 1,408 -> 896;
- consecutive ebarrier gap count: 480 -> 224;
- consecutive ebarrier gap duration: 267,496 -> 193,100;
- ebarrier hot-instruction latency: 279,628 -> 201,256.

The aggregate ABarrier gap grows slightly in relative and absolute SQTT
attribution. This is not a structural ownership fix. Promote the bounded
epilogue improvement, then stop changing the store path.

Evidence:

- Candidate run: `/zys/sb/runs/c85_dkv_store_batch2_0f3527e`.
- Control run: `/zys/sb/runs/c85_control_c84_3388f47`.
- Shared perf: `/Volumes/172.20.68.76/\u5171\u4eab/shaobo/perf/`
  `20260825_134415_C85_dkv_store_batch2_H1S1024_causal_SQ7`.
