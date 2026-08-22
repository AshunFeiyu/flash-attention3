# Fused5 dQ Writer MMAC Zero Seed

## Hypothesis

The canonical writer explicitly clears eight `Vec4F32` accumulators for every
q tile.  Replace those clears with one branch-local zero seed: the first C0
MMAC defines each accumulator and the remaining C0/C1 MMACs accumulate in
place.  Formula, M64/N128/D128 tile, five-GEMM work, LDS, ABarrier ownership,
group order, and stores remain unchanged.

## Static Result

The original `16/204/204/88` WDRA split left no allocator headroom.  The first
build used 88 writer VGPRs and spilled 43 VGPRs.  Repartitioning the same 512
physical VGPR pool to `16/196/196/104` gives actual role use
`9/187/91/182` and restores private/spill/scratch to zero.  Kernel metadata is
SGPR82/VGPR128.

The intended ISA change is exact:

| Opcode family | Control | Candidate |
|---|---:|---:|
| `v_mov_b64` | 116 | 70 |
| MMAC | 832 | 832 |
| ABarrier waits | 31 | 31 |
| matrix reads | 840 | 840 |
| global stores | 56 | 56 |

## Validation

S128 causal and noncausal pass complete CPU-golden correctness.  All six
S1024 control/candidate runs pass delta, dK, dV and dQ correctness with bank0
and no PMD warning.

Three interleaved pairs are not stable:

| Pair | Control fused ticks | Candidate fused ticks | Delta |
|---|---:|---:|---:|
| 1 | 45,237,920 | 44,678,270 | -1.237% |
| 2 | 44,898,490 | 44,991,765 | +0.208% |
| 3 | 44,822,960 | 44,959,460 | +0.305% |
| mean | 44,986,457 | 44,876,498 | -0.244% |

Lifecycle mean moves `49,094,803 -> 49,023,368` (`-0.146%`).  The first pair
alone creates the mean benefit; the following two pairs regress.  Fewer moves
are real, but they are hidden well enough that they do not produce repeatable
ticks.

## Decision

Status: `OBSERVE_ISA_WIN_TICKS_UNSTABLE_CANONICAL_RESTORED`.

Keep MMAC zero seed as an admission-time heuristic for new kernels, but do not
promote it into this fused canonical path.  No fullperf/xcu capture is admitted
after the repeated ticks gate fails.  The next hypothesis must target the
writer's measured C0-ready wait and global-store issue tail rather than another
isolated move reduction.

Evidence: `/zys/sb/fa3b/writer_zero_seed_20260822/paired`.

