# Fused5 dV Read Under P Writer

Date: 2026-08-12  
Decision: `ACCEPT_CANONICAL_READINESS_WIN`

## Hypothesis

The dV path does not need to wait for the P writer before issuing independent
dO reads. Issue four normal dO matrix reads first, retire the older P writer
with `lgkmcnt(4)`, then read P and wait once before the dV MMAC island.

## Verification

- H1/S128 causal and H1/S1024 causal correctness: PASS.
- Five GEMMs and MMOP `92,160`: unchanged.
- LDS `131,072B`; bank conflict `0`.
- No private segment, scratch, VGPR spill, or SGPR spill.
- WDRA role sizes: `16/204/204/88`.

## Performance Evidence

Fullperf on `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`:

| Metric | Accepted raw-page baseline | Candidate |
|---|---:|---:|
| complete ticks | 53,714,570 | 52,466,960 |
| compute ticks | 50,863,995 | 49,835,240 |
| MMAC active | 31.3305% | 31.5782% |
| XCU duration | 111,792 | 109,528 |
| ABarrier gap | 29.23% | 28.45% |
| P writer readiness gap | 2.64% | 0.69% |

Two paired same-environment stats repeats report `-0.9743%` complete ticks,
`+0.5798pp` MMAC active, `-0.8383pp` LGKM wait, and `-0.5753pp` barrier
share. XCU reports consumer peer-VALU MMAC counts changing from `395/641` to
`511/525` for C0/C1, while total direct overlap remains `1,036/4,096`.

## Interpretation

This is a useful canonical micro-optimization. It removes a real P-writer
readiness edge and balances the consumer groups, but it does not solve the
larger MMAC-vs-MMAC lockstep. The next experiment is a per-group completion
token, not another read-placement tweak.

Evidence paths:

- local perf: `work/perf/20260812_072621_fused5_dv_read_under_writer_h1s1024_sqc7/`
- remote fullperf: `/zys/shaobo_runs/fused5_dv_read_under_writer_20260812/fullperf/`
- remote xcu: `/zys/shaobo_runs/fused5_dv_read_under_writer_20260812/xcu/`
