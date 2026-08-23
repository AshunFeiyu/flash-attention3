# Fused5 dQ Fragment-Native Workspace Design

Status: `REJECT_SCALE_REGRESSION / CANONICAL_SOURCE_RESTORED`.

## Measured Trigger

Accepted `58e90fc` fullperf attributes `270,196` hot-instruction latency cycles
to `global_store_dwordx2` and another `139,280` issue-gap cycles to consecutive
instances. The fused symbol dynamically issues 2,304 dQ-partial FP16 stores:
each writer lane stores two four-half fragments for the same logical row.

The existing workspace is row-major `[BH,Ktile,S,D]`. This makes reduction
simple, but the two fragments owned by one lane are separated by 16 columns and
cannot be combined into one vector transaction.

## One Hypothesis

Keep the same workspace byte count but store it in the dQ MMAC fragment's
native lane-major order:

```text
workspace[BH,Ktile,row,d_owner,lane_group,fragment_element]
shape tail = [4,4,8] = 128 fp16 values per row

lane_group = lane >> 4
packed_group = d_owner * 4 + lane_group
packed_offset = row * 128 + packed_group * 8

packed[0:4] = dq_acc[d_half=0]  # output columns d_owner*32 + lane_group*4 + 0:4
packed[4:8] = dq_acc[d_half=1]  # output columns d_owner*32 + 16 + lane_group*4 + 0:4
```

The writer converts both FP32 fragments to FP16 and emits one
`global_store_dwordx4`. The reducer owns one packed group per thread, loads one
eight-half vector per K tile, accumulates two Vec4F32 sums, and performs two
row-major final stores.

## Invariants

- Exact five GEMMs, MMOP92,160 at H1/S1024 causal.
- M64/N128/D128, 16-wave roles, writer G1-first order, all 12 ABarrier IDs.
- LDS128KiB, BPS/MLS + `ds_read_matrix` + MMAC matrix path, and all ownership
  events unchanged.
- Workspace bytes remain `BH * Ktiles * S * D * sizeof(fp16)`.
- No duplicate dQ, dK, dV, score, or dP work.
- Final dQ remains row-major FP16 and complete CPU-golden comparable.

## Resource And Traffic Budget

| Item | Accepted | Candidate | Gate |
|---|---:|---:|---|
| fused dQ stores per writer lane/panel | 2 x 8B | 1 x 16B | one `global_store_dwordx4` |
| fused dynamic dQ store instructions | 2,304 | about 1,152 | exact stats |
| workspace bytes | unchanged | unchanged | byte formula |
| reducer outputs per thread | 4 | 8 | two Vec4F32 sums |
| reducer CTAs at H1/S1024 | 128 | 64 | all 48 CUs still coverable |
| reducer final stores per 8 outputs | 2 | 2 | row-major ABI |
| fused/reducer private or spill | 0 | 0 | hard gate |

The previous generic two-output/thread reducer failed because it retained two
separate row-major loads. This design is not the same instruction shape: one
16-byte packed load replaces two 8-byte loads for the same eight outputs. The
64-CTA coverage risk remains real and is part of the end-to-end admission.

## Expected Pipeline

```text
writer time0: finish G1/G0 dQ MMAC; two FP32 fragments already resident
writer time1: convert two Vec4F32 -> one Vec8F16; one 16B workspace store

reducer time0: load packed Ktile0/Ktile1 Vec8F16 pair
reducer time1: accumulate low/high Vec4F32 sums while next pair ages
reducer time2: convert each sum; two row-major 8B final stores
```

No new ABarrier, LDS page, matrix read, or global byte is introduced.

## Admission

1. Workbook A0 must precede source changes; implementation stays in the two
   cohesive workspace writer/reducer functions.
2. A1 generated ISA must halve fused dQ partial-store sites without changing
   MMAC, matrix reads, ABarrier, or dKV stores.
3. S128 causal/noncausal and S1024 full CPU-golden must pass with warning0,
   no private/spill/scratch and bank0.
4. Three interleaved S1024 pairs compare fused, reducer and full-lifecycle
   ticks. Promotion requires lower fused and lower total mean; reducer may not
   erase the fused gain.
5. S2048 scaling is mandatory before promotion. Fullperf/xcu follows only if
   the paired-ticks gate passes.

Workbook: section65 in the 2026-08-23 fused5 design workbook.

## Result

The implementation passed the intended static and semantic gates. Fused
partial stores changed from two `global_store_dwordx2` transactions per lane
to one `global_store_dwordx4`; dynamic fused FLAT fell `3,616 -> 2,464` while
MMOP stayed `92,160`, LDS stayed `63,872`, bank conflicts stayed zero, and no
private/spill/scratch appeared. Full CPU-golden correctness passed S128 causal
and noncausal plus every S1024/S2048 A/B run.

Three interleaved S1024 pairs initially looked positive:

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| fused ticks | 43,128,540 | 42,326,072 | -1.861% |
| dQ reducer ticks | 1,717,170 | 2,015,043 | +17.347% |
| full lifecycle ticks | 47,289,060 | 46,784,010 | -1.068% |
| fused MMAC active | 35.265% | 36.188% | +0.923 pp |
| fused wait-VM | 3.442% | 3.049% | -0.393 pp |
| fused wait-LGKM | 7.564% | 7.949% | +0.385 pp |
| fused barrier | 13.273% | 12.975% | -0.298 pp |

The mandatory S2048 pair reverses the result:

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| fused ticks | 79,983,995 | 80,258,815 | +0.344% |
| dQ reducer ticks | 4,259,483 | 4,597,093 | +7.926% |
| full lifecycle ticks | 87,528,578 | 88,131,908 | +0.689% |

The fused store reduction is real, but the reducer's 128-to-64 CTA collapse
reduces latency hiding and load balance. The S1024 fused tail gain therefore
does not scale. No fullperf candidate is admitted. Both production source
files are restored to accepted `58e90fc` behavior; only this evidence remains.

Boundary: a future fragment-native workspace is admissible only if it keeps
the reducer at 128 CTAs (or otherwise proves equal-or-better coverage) while
retaining the single fused store. Do not retry the same eight-output/thread
reducer topology.

Evidence: `/zys/sb/runs/f5dqfrag_ab_20260823` and
`/zys/sb/runs/f5dqfrag_s2048_ab_20260823`.
