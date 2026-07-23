# Fused5 Useful Stagger Result

Date: 2026-07-23

Status: `ACCEPT_MICRO_TICKS_USEFUL_STAGGER / MMAC50_OPEN`

## Change

- Keep the accepted 16-wave topology, output ownership, LDS map, ABarrier
  tokens, and exactly five GEMMs.
- Split the coupled score/dP block into explicit score and dP matrix-product
  islands.
- C0 executes `score -> P -> dV -> dP -> dS`.
- C1 executes `dP -> score -> P -> dS -> dV`.
- Each score or dP island batches four transposed matrix reads, performs one
  first-use wait, and issues eight MMAC.
- Retain FP32 P until dS while also keeping the FP16 P fragment for dV.

No delay, duplicate score/dP, new kernel, phase flag, page, token, scalar
matrix read, gather, or wrong-layout path was added.

## Gates

- Source gate: PASS.
- Compile role usage: `8/165/168/84` inside WDRA `8/200/200/88`.
- Metadata: SGPR60, VGPR124, private0, SGPR spill0, VGPR spill0.
- LDS: 115,456 B.
- H1/S128 causal and noncausal: PASS.
- H1/S1024 causal: PASS.
- Dynamic MMOP: 92,160.
- LDS bank conflicts: 0.
- PMD panic and uninitialized-VGPR warnings: 0.

## Same-Shape Result

Compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`,
`GPU_ARGS=['--SQCIPfLines=7']`.

| Metric | Native lag-one | Useful stagger | Change |
|---|---:|---:|---:|
| kernel ticks | 73,280,025 | 73,016,580 | -0.3595% |
| MMAC active | 21.809889% | 21.641706% | -0.168183 pp |
| VALU instructions | 176,896 | 169,376 | -4.2511% |
| SCA instructions | 39,376 | 37,808 | -3.9821% |
| LDS instructions | 64,096 | 64,096 | 0 |
| successful coissue | 14,399 | 22,895 | +59.0041% |
| failed coissue | 14,962 | 15,533 | +3.8163% |
| waitLgkm share | 8.302109% | 7.568883% | -0.733226 pp |
| ABarrier share | 22.489394% | 23.680675% | +1.191281 pp |

The ticks improvement repeats in stats-only (`73,046,155`) and fullperf
(`73,016,580`), so it is accepted as a micro scheduling improvement. It is
not an MMAC-active winner and does not satisfy the 50% goal.

## SQTT Explanation

In the same `0:159000`, XCD0/SE0/CU0/SIMD0 window:

- C0 MMAC+VALU coissue rises from `10.91%` to `12.79%`.
- C1 MMAC+VALU coissue rises from `11.24%` to `12.69%`.
- Dynamic `s_waitcnt` issues fall from 28,480 to 23,872.
- Transposed matrix-read-to-wait bubble falls from `6.58%` to `5.11%`.
- The top ABarrier-following bubble grows from `3,484,240` to `3,546,516`
  cycles (`+1.79%`).
- Atomic-to-atomic bubble grows from `1,585,892` to `1,678,392` cycles.
- Final ebarrier-to-branch bubble grows from `1,174,488` to `1,214,636`
  cycles.

The legal order change creates real MMAC/VALU overlap and read batching
removes wait debt. The groups then reach shared ownership and output
boundaries earlier, where ABarrier and atomic tails absorb nearly all of the
saved time. This explains lower ticks together with slightly lower aggregate
MMAC active.

## Evidence

- Build:
  `/zys/sb/fa3b/build_fused5_useful_stagger_20260723`
- S128 causal:
  `/zys/sb/fa3b/fused5_useful_stagger_correctness/5gemm_symmetric_s128_c1_20260723_190606`
- S128 noncausal:
  `/zys/sb/fa3b/fused5_useful_stagger_correctness/5gemm_symmetric_s128_c0_20260723_190606`
- S1024 stats:
  `/zys/sb/fa3b/fused5_useful_stagger_stats/5gemm_symmetric_s1024_c1_20260723_190644`
- Fullperf:
  `/zys/sb/fa3b/fused5_useful_stagger_fullperf/5gemm_owner_s1024_c1_fullperf_20260723_190815`
- Perf:
  `3347219_fused_bwd_correctness.perf`
- Perf SHA256:
  `53d9dcc1ed2768a49f25ea23742819ddc8e6b69e057bd4fc9e124194c86e91b4`
- XCU first pass:
  `/zys/sb/fa3b/xcu_outputs/fused5_useful_stagger_s1024_20260723_d0`
- XCU same-window exports:
  `/zys/sb/fa3b/xcu_outputs/fused5_useful_stagger_s1024_20260723_d0_window`
- Shared archive:
  `/共享/shaobo/perf/20260723_190815_fused5_useful_stagger_h1s1024_sqc7_fullperf`

## Next Gate

Preserve the batched reads and legal C0/C1 order. Do not add another local
schedule permutation. Map the dominant ABarrier wait to its ownership edge,
then redesign that page lifetime so the earlier-arriving group can perform
useful next-generation work without waiting for the other group. Any token
split must remain within the eight hardware ABarrier IDs and 128 KiB LDS.
