# Fused5 dK/dV Four-D-Block Store Batch

Status: `ACCEPT_TICKS_AND_ACTIVE_MMAC50_OPEN`.

## Change

Commit `34d5f39` changes only the terminal dK/dV matrix-store epilogue. After
the existing all-role mainloop rendezvous, K and V are dead, so their
contiguous 64 KiB LDS range is reused to publish all four D32 dK/dV blocks in
one store epoch instead of two epochs of two blocks.

The five-GEMM DAG, tiles, output ownership, mainloop reads, ABarrier protocol,
matrix-store count and external ABI are unchanged. Generated ISA reduces
static `s_ebarrier_sync` sites from 44 to 28 across causal and noncausal
symbols, and combined wait sites from 14 to 10.

## Evidence

Same compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7,
H1/S1024/D128 causal:

| Metric | C109 | C111 | Delta |
| --- | ---: | ---: | ---: |
| stats-only fused mean | 38,956,038 | 38,601,442 | -0.910% |
| stats-only lifecycle mean | 43,065,447 | 42,772,882 | -0.679% |
| same-time fullperf fused | 38,948,455 | 38,870,195 | -0.201% |
| same-time fullperf lifecycle | 42,987,035 | 42,928,795 | -0.135% |
| MMAC active | 39.442415% | 39.792603% | +0.350188 pp |
| barrier latency share | 14.438399% | 14.095966% | -0.342433 pp |
| wait-LGKM latency share | 8.480431% | 8.448171% | -0.032260 pp |

Role-filtered SQTT shows consumer0/consumer1 total bubbles fall by roughly
787/791 cycles. Their terminal matrix-store-to-wait gaps fall from
2,804/2,828 to 2,168/2,268 cycles. The dQ writer's terminal
ebarrier-to-ebarrier gap falls from 3,241 to 931 cycles.

H1/S128 causal/noncausal and H1/S1024 causal complete golden correctness
pass. VGPR roles are `9/142/87/130` within `16/204/88/204`; private segment,
spill, scratch, PMD warning and LDS bank-conflict counts are zero.
MMOP/VALU/LDS/VMEM/FLAT remain exact at
`88064/89040/61568/1664/2592`.

Evidence:

- `/zys/sb/fa3b/c111_dkv_store_batch4_fullperf/`;
- `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260827_113114_C111_dkv_store_batch4_H1S1024_causal_SQ7`.

The dK/dV store-batching tier is now closed. The remaining MMAC-active debt is
inside the steady mainloop, not the terminal store epilogue.
