# dot_do_o Top-Level Design

## Objective

Optimize the required preprocessing stage without changing the external FA BWD
contract:

```text
dot_share = dot_ticks / (dot_ticks + dKV_ticks + dQ_ticks) <= 5%
```

The locked H1/S1024 baseline is:

| Stage | Kernel ticks |
|---|---:|
| dot_do_o | 12,398,295 |
| dKV | 30,654,715 |
| dQ | 21,011,445 |

Holding dKV/dQ fixed, the dynamic target is:

```text
dot_ticks <= 0.05 / 0.95 * (dKV_ticks + dQ_ticks)
          <= 2,719,272 ticks
```

This requires at least a 4.56x dot speedup and a 78.07% tick reduction.

## Math And Data Contract

For every BHSD row `r`:

```text
delta[r]       = sum_d fp32(dO[r,d]) * fp32(O[r,d])
packed[r,0]    = scores_max[r] * softmax_scale * log2(e)
packed[r,1]    = scores_sum[r] != 0 ? 1 / scores_sum[r] : 0
packed[r,2]    = delta[r]
```

At D=128, one row reads 512 bytes of FP16 O/dO plus 8 bytes of FP32
max/sum, and writes 16 FP32 bytes. The total is 536 bytes for about 255 FP
operations, or roughly 0.476 FP operations/byte. This is a memory/latency and
reduction kernel; MMAC active is not a relevant objective.

## Baseline Failure Mode

The baseline maps one thread to one row. A wave therefore handles 64 rows, and
at a fixed D position adjacent lanes access rows 256 bytes apart. Each thread
also serializes the complete D=128 loop. H1/S1024 launches only eight
128-thread CTAs. PMD shows only 8 of 48 CUs and 16 of 192 SIMDs active.

The dominant problems are:

1. Strided, poorly coalesced wave memory access.
2. A 128-iteration dependency chain per lane.
3. Too few CTAs and waves to hide global-memory latency.

## Canonical Architecture

Use one wave64 per row and four independent waves per 256-thread CTA:

```text
wave_in_cta = threadIdx.x >> 6
lane        = threadIdx.x & 63
row         = blockIdx.x * 4 + wave_in_cta
```

For D=128, every lane loads scalar elements `lane` and `lane+64` from O and
dO, computes two FP32 products, then participates in a six-step wave
reduction:

```text
partial = fp32(dO[lane]) * fp32(O[lane])
partial += fp32(dO[lane+64]) * fp32(O[lane+64])
for offset in [32, 16, 8, 4, 2, 1]:
    partial += shfl_down(partial, offset, 64)
lane0 publishes delta and packed sidecar
```

The existing focused probe proves that `__shfl_down(..., 64)` is correct on
`GPU_CHIP=sb` and lowers to six `ds_bpermute_b32` instructions.

### Launch Geometry

| Property | Baseline | Canonical |
|---|---:|---:|
| Threads/CTA | 128 | 256 |
| Waves/CTA | 2 | 4 |
| Rows/CTA | 128 | 4 |
| CTAs at H1/S1024 | 8 | 256 |
| Active CU / SIMD | 8 / 16 | 48 / 192 |
| Row owner | thread | wave |
| LDS | 0 | 0 |
| Barrier | 0 | 0 |

Four waves fit the Shaobo four-SIMD scheduling unit while each wave owns an
independent row. No CTA join, LDS reduction, ABarrier, EBarrier, WDRA or
local-wave mode is required.

## Expected Pipeline

```text
time0  all lanes: dO[lane], dO[lane+64] loads
time1  all lanes: O[lane], O[lane+64] loads
time2  all lanes: convert and two products
       lane0:     max/sum prefetch when scheduling permits
time3  shuffle/add offset 32
time4  shuffle/add offset 16
time5  shuffle/add offset 8
time6  shuffle/add offset 4
time7  shuffle/add offset 2
time8  shuffle/add offset 1
time9  lane0: reciprocal, scale, delta and packed stores
```

The main latency hiding comes from many resident independent waves, not from
intra-CTA buffering. The six-step reduction is a bounded dependency chain.

## Resource Budget

- LDS: 0 bytes.
- Barrier slots: 0.
- Private/scratch/spill: exactly 0.
- VGPR target: at most 32 per wave; generated metadata is authoritative.
- SGPR target: at most 32 per wave; generated metadata is authoritative.
- Main loads: two scalar FP16 values from each tensor per lane. A packed
  `half2` refinement remains optional because A1 already passes the hard goal.
- Bank conflict: exactly 0 because there is no LDS path.
- Build: ordinary translation unit without local-wave/WDRA flags.

## Candidate Ladder

1. **A1 wave scalar-pair -- ACCEPT**: use the proven `lane` and `lane+64`
   mapping. This alone fixes launch parallelism and passes the 5% target.
2. **A2 packed half2 -- DEFER**: use one aligned dword load per tensor/lane.
   Reopen only if a stricter target justifies another correctness/codegen risk.
3. **A3 native dot2 partial**: only after a focused builtin/ASM probe proves a
   better instruction chain and full correctness.
4. **A4 sidecar scheduling**: move only existing max/sum work under the shuffle
   tree if Source/xcu proves it is on the critical path.
5. Fusion is last resort and requires a new full DAG/resource design.

A 128-thread block reduction is rejected at design time because it adds LDS
and barriers to a problem with a proven wave-native reduction.

## Validation And Promotion

Every candidate must pass, in order:

1. Normalized ASM: coalesced packed loads, six wave shuffles, no barrier/LDS.
2. Metadata: no private segment, scratch, SGPR spill or VGPR spill.
3. H1/S128 cached-golden full lifecycle correctness.
4. H1/S1024 cached-golden full lifecycle correctness and bank conflict zero.
5. Three same-binary PMD repetitions against the locked baseline artifact.
6. Per-dispatch tick decomposition and active-CU distribution.
7. Winner-only xcu if the remaining gap needs instruction-level attribution.

Promotion requires stable same-run improvement and the exact dynamic share
formula at or below 5%. CPU wall time is never used.

## Accepted Result

`A1` passes every hard gate on LLVM `e0f10535` and PMD HEAD1694:

| Metric | Baseline | A1 result |
|---|---:|---:|
| dot ticks | 12,398,295 | 2,447,900 median |
| dot speedup | 1.00x | 5.065x |
| dot tick reduction | 0% | 80.26% |
| end-to-end dot share | 19.353% | 4.618% median |
| active CU / SIMD | 8 / 16 | 48 / 192 |
| active-SIMD time CV | 0.11% | 2.93% |
| SGPR / VGPR | not gated | 22 / 12 |
| spill / scratch / private | not gated | 0 / 0 / 0 |
| bank conflict | 0 | 0 |

The three A1 dot runs are `2,429,245`, `2,449,720`, and `2,447,900` ticks;
their full range is 0.84% of the mean. End-to-end shares are `4.593%`,
`4.618%`, and `4.624%`. S128 and every S1024 run pass the cached CPU golden;
S1024 delta max absolute error is `1.86e-9` with no non-finite values.
The control was independently rebuilt from commit `84a46e3` with the same
LLVM/PMD lock; all three control dot runs are exactly `12,398,295` ticks and
pass the same lifecycle correctness gate.

The accepted static shape contains six `ds_bpermute_b32`, no barrier, no
WDRA resize and no LDS allocation. Dynamic dot counts are VALU `71,680`, SCA
`5,120`, LDS-class shuffle `6,144`, FLAT `8,192`, and MMOP `0`. These counts
are higher than the serialized baseline because the model counts useful
wave-level instructions; the 5.065x speedup comes from exposing all 48 CUs and
192 SIMDs, not from reducing total wave instructions.

A TT/Perf full-lifecycle attempt remained numerically correct and measured
`2,443,805` dot ticks / `4.592%`, but the helper emitted no `.perf` for the
three-HSACO, three-dispatch harness. XCU remains optional follow-up through a
single-dispatch dot harness; it is not needed to establish the hard share,
resource, correctness or balance gates.

## Evidence

- Baseline run:
  `/zys/sb/fullbwd_final/full_bwd_correctness_20260722_112827/m5out`
- Rebuilt control repeats:
  `/zys/sb/dot_control_repeat2/full_bwd_correctness_20260722_131305/m5out`
  and `/zys/sb/dot_control_repeat3/full_bwd_correctness_20260722_131419/m5out`
- A1 repeated runs:
  `/zys/sb/dot_do_o_a1/full_bwd_correctness_20260722_123052/m5out`,
  `/zys/sb/dot_do_o_a1_repeat2/full_bwd_correctness_20260722_123217/m5out`,
  and `/zys/sb/dot_do_o_a1_repeat3/full_bwd_correctness_20260722_123320/m5out`
- TT/Perf stats-only output:
  `/zys/sb/dotfp/full_bwd_correctness_20260722_124146/m5out`
- Wave reduction probe:
  `remote_src/shaobo_instruction_probes/hip/wave_dot_reduce_probe.hip`
- Shared workbook sheet: `DOT_TopDesign` in
  `/Volumes/172.20.68.76/` shared `shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx`.
