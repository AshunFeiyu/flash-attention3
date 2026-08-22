# Fused5 Compile-Time-Peeled dKV Zero Seed

Status: `ACCEPT_MICRO_TICKS_AND_CODEGEN / MMAC50_OPEN`.

## Evidence Boundary

The accepted `9ef0704` kernel explicitly clears eight FP32 dV and eight FP32
dK accumulators before its q-loop. A prior runtime `qi/m_block` zero-seed was
correct but duplicated MMAC/control and regressed ticks. The current loop
already peels q tile 0, so its first accumulator definition can be selected at
compile time.

The first implementation of this experiment also failed the static gate: it
specialized the complete C1 panel body and expanded waits `340 -> 422`,
branches `46 -> 138`, and code size by about 3.6%. It was never run on PMD.

## Admitted Implementation

- C0 dV panel 0 uses the existing zero fragment as the MMAC accumulator.
- C0 and C1 dK panel 0 use the zero fragment as the MMAC accumulator.
- C1 dV retains explicit initialization because C1 consumes panels through a
  runtime loop; specializing that loop duplicates its softmax/control body.
- Every later panel and q tile uses the live FP32 accumulator normally.

This removes 48 static `v_mov_b64` instructions without adding MMAC, matrix
reads, waits, branches, LDS traffic, tokens, or runtime first-update tests.

## Correctness And Resource Proof

- Every admitted first MMAC writes all eight accumulator destinations.
- Arithmetic order after the first contribution is unchanged.
- S128 causal and noncausal, S1024 causal, and S2048 causal complete backward
  CPU-golden checks pass for delta, dK, dV, and dQ.
- `SGPR=82`, `VGPR=128`, private/spill/scratch are zero, and LDS bank conflict
  is zero.

## Static Evidence

| Metric | Control | Candidate |
| --- | ---: | ---: |
| static MMAC | 1,472 | 1,472 |
| `ds_read_matrix` | 840 | 840 |
| `s_waitcnt` | 340 | 340 |
| scalar branches | 46 | 46 |
| ABarrier opcodes | 102 | 102 |
| `v_mov_b64` | 116 | 68 |
| fused symbol bytes | 711,576 | 703,278 |
| role VGPR use | 9/171/87/164 | 9/171/87/164 |

## Performance Evidence

Three interleaved H1/S1024 pairs:

```text
control:   44,516,290  44,431,660  44,841,615
candidate: 44,343,390  44,166,395  44,146,830
mean:      44,596,522 -> 44,218,872  (-0.847%)
```

The complete lifecycle mean improves `48,731,258 -> 48,391,828` (`-0.697%`).
One H1/S2048 scaling pair improves fused ticks
`82,868,695 -> 82,479,215` (`-0.470%`) and complete ticks
`90,620,075 -> 89,715,080` (`-0.999%`).

Candidate S1024 fullperf keeps exact `MMOP=92,160`, matrix LDS `63,872`,
VMEM `1,408`, and FLAT `3,616`; VALU falls `119,744 -> 118,304`.
MMAC active is effectively neutral (`34.701% -> 34.686%`), so this is a
repeatable codegen/ticks micro-win, not the structural path to 50%.

## Decision And Next Boundary

Promote the partial compile-time seed. Do not specialize the runtime C1 panel
loop or claim that move removal solves the main pipeline limit. XCU still
attributes the dominant debt to ABarrier ownership (`try_wait -> s_xor`) and
matrix-read first-use waits. The next hypothesis must change useful work
available during those waits without adding token cadence or fake MMAC.

Evidence:

- A/B: `/zys/sb/runs/fused5_peeled_zero_seed_ab`
- S2048: `/zys/sb/runs/fused5_peeled_zero_seed_s2048`
- fullperf: `/zys/sb/f5zfull/b1_hq1_hkv1_s1024_d128_c1_fullperf_perfonly_20260823_032215`
