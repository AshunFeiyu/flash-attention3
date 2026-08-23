# Fused5 Sidecar Pair-Read Design

Status: `REJECT_LGKM_BARRIER_MIGRATION_CANONICAL_RESTORED`.

## Measured Trigger

The accepted `58e90fc` SQTT attributes repeated consumer bubbles to scalar
sidecar LDS reads:

```text
C0 ds_read_b32(max/invsum) -> wait   2,505 cycles
C0 ds_read_b32(delta)      -> wait   1,256 cycles
C1 sidecar read/wait families        about 2,300 cycles
```

The producer already stores max and inverse-sum with one
`ds_write2st64_b32`; the two fields are separated by exactly 64 dwords. The
consumer source expresses them as unrelated C++ loads, so LLVM emits separate
`ds_read_b32` operations and first-use waits.

## One Hypothesis

Read row max and inverse-sum with the supported builtin
`__builtin_hcu_ds_read2_f32(base, 0, 64, false)`. It emits one
`ds_read2_b32 offset1:64`; the builtin's final boolean is the GDS selector and
must remain false. Keep delta as an independent
scalar read because its first use is later in the dS stage.

```text
sidecar max/invsum:  ds_read2_b32 offset1:64 -> one first-use wait -> probability
sidecar delta:       ds_read_b32      -> existing late first-use wait -> dS
```

This is an instruction-granularity change. It does not alter the algorithm,
ownership graph, or matrix path.

## Invariants

- Exactly five GEMMs and dynamic MMOP 92,160 at H1/S1024 causal.
- M64/N128/D128, 16 waves, G1-first writer order and all 12 ABarrier IDs.
- LDS remains 128 KiB; sidecar offsets remain field-major at 0/64/128 dwords.
- Matrix operands remain MLS/BPS + `ds_read_matrix` + MMAC.
- No ordinary DS read enters a matrix operand path; this helper is sidecar
  metadata only.
- No new wait, barrier, global transaction, page, or output store.
- No private segment, spill, scratch, PMD warning, or LDS bank conflict.

## Resource And Pipeline Budget

The pair result occupies two VGPRs, equal to the two scalar values it replaces.
No new long-lived value crosses an MMAC island. Expected role windows remain
`16/204/204/88`, metadata SGPR82/VGPR128, and measured branch usage should not
increase materially.

```text
time0  read max+invsum as one LDS pair; score operands are already available
time1  retire the pair once; probability VALU consumes both values
time2  read delta independently while dV operands/MMAC age
time3  retire delta at dS first use; all ownership events remain unchanged
```

## Admission

1. A1 generated ISA must contain `ds_read2_b32 offset1:64` without `gds` in
   the fused symbol and
   reduce scalar sidecar read sites without changing MMAC, matrix-read, wait,
   ABarrier, VMEM, or global-store counts unexpectedly.
2. S128 causal/noncausal full CPU golden must pass, followed by S1024 causal.
3. Static metadata, warning, bank and exact-work gates must pass.
4. Three interleaved S1024 pairs decide promotion. Lower LDS/SCA count or
   higher MMAC active alone is insufficient.
5. Fullperf/xcu is captured only after the paired-ticks gate. It must show the
   sidecar read-to-wait family shrinking without migration into ABarrier or
   matrix-read readiness.

Workbook: section 63 in the 2026-08-23 fused5 design workbook.

## Result

The focused instruction path is legal. Compiler `e0f10535` emits 40
`ds_read2_b32 offset1:64` operations and no accidental `gds` operation. Static
matrix work is exact at MMAC1472 and matrix-read840; ABarrier stays102,
private/spill/scratch remain zero, and both S128 causal/noncausal full-golden
runs pass with warning0 and bank0. Dynamic S1024 work changes as intended:
LDS instructions fall `63,872 -> 61,568` and VALU falls `117,156 -> 116,900`.

Three interleaved S1024 pairs nevertheless reject the candidate:

| Metric | Control mean | Candidate mean | Delta |
|---|---:|---:|---:|
| fused ticks | 43,286,880 | 44,325,038 | +2.398% |
| MMAC active | 35.155497% | 34.891421% | -0.264076 pp |
| wait-VM | 3.449782% | 2.519882% | -0.929900 pp |
| wait-LGKM | 7.448527% | 8.952652% | +1.504125 pp |
| barrier | 13.340440% | 14.248019% | +0.907579 pp |
| coissue success | 24,377 | 23,879 | -2.042% |

The pair read reduces instruction count but couples max and inverse-sum to one
LDS transaction/readiness edge. The compiler schedule exposes more LGKM wait,
which then delays ownership release and raises barrier share. This is a
critical-path regression, not an instruction-support failure. No S2048 or
fullperf is admitted; canonical scalar panel-local reads are restored.

Evidence: `/zys/sb/runs/f5sr2_ab_20260823`.
