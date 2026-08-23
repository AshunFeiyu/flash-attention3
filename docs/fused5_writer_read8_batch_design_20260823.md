# Fused5 dQ Writer Read8 Batch Design

Status: `REJECT_OUTSTANDING_LDS_PRESSURE_CANONICAL_RESTORED`.

## Measured Trigger

The accepted causal-boundary build reaches 36.659% fullperf MMAC active. XCU
shows the dQ writer's largest repeated non-ownership bubble is
`ds_read_matrix_trans_format -> s_waitcnt`: 24,116 cycles over 128 waits on a
representative full-length writer wave. The writer currently repeats:

```text
4 dS matrix reads -> lgkmcnt(0) -> 8 dQ MMAC
```

for each of four M16 panels and both source groups.

## One Hypothesis

Batch two adjacent M16 panels without changing the formula or ownership:

```text
8 dS matrix reads -> one first-use lgkmcnt(0) -> 16 dQ MMAC
```

This keeps each MMAC island's inputs independent, halves writer first-use wait
sites per q tile, and matches the regular read8/MMAC-island issue shape proven
useful in the reference SQTT. No read, MMAC, store, LDS page, token, barrier or
output owner is added.

## Fixed Work And Resource Ledger

| Item | Accepted | Candidate gate |
|---|---:|---:|
| tile / roles | M64/N128/D128, 4P+4C0+4C1+4W | exact |
| logical GEMMs | 5 | exact |
| MMAC per q tile | 1,280 | exact |
| writer dS reads per q tile | 32 | exact |
| writer wait sites per q tile | 8 | 4 |
| writer MMAC per q tile | 256 | exact |
| writer live dS fragments | 4 x Vec8F16 | 8 x Vec8F16 |
| extra writer live VGPR estimate | 0 | +16 |
| WDRA windows | 16/204/204/88 | expected 16/204/188/104 |
| total physical VGPR/SIMD | 512 | exact 512 |
| LDS / ABarrier | 128KiB / 12 IDs | exact |
| dynamic MMOP/LDS/VMEM/FLAT | 92160/63872/1408/3616 | exact |

The current writer is effectively at its 88-VGPR boundary, so batching is not
admitted with the old ledger. The only legal resource change is to move one
16-VGPR quantum from C1's measured headroom to the writer. If generated C1 or
writer use exceeds 188/104, or any private/spill/scratch appears, reject before
PMD. Do not reduce the accumulator tile or duplicate work to make it fit.

## Expected Pipeline

```text
time0  writer: wait Filled1 -> read dS M0+M1 (8 reads)
time1  peer C0/C1: score/dP/dV/dK MMAC or softmax/dS
       writer: one readiness wait -> dQ M0+M1 (16 MMAC)
time2  writer: read dS M2+M3 (8 reads)
time3  peer C0/C1: useful work
       writer: one readiness wait -> dQ M2+M3 (16 MMAC)
time4  writer: release source group, repeat for the second group
```

Expected XCU effect: writer trans-read-to-wait count `128 -> 64`, longer
contiguous MMAC islands, lower absolute writer read-wait cycles, and no growth
in producer/C0/C1 ownership waits.

## Admission

1. Compile and static gates first: exact MMAC/read/barrier counts, WDRA
   `16/204/188/104`, no private/spill/scratch.
2. Full golden S128 causal/noncausal and S1024 causal; warning0 and bank0.
3. Three interleaved S1024 pairs must lower fused ticks and lifecycle ticks.
4. S2048 must not regress; fullperf/xcu must prove lower absolute writer
   read-wait debt, not merely a higher MMAC-active percentage.
5. If the compiler interleaves reads/MMAC irregularly, adds waits, or spills,
   reject and restore commit `36af512` behavior.

Workbook: section69 in the 2026-08-23 fused5 design workbook.

## Result

The resource proof succeeds. WDRA `16/204/188/104` produces branch use
`9/173/162/99` in source-role order, SGPR71/VGPR128, with no
private/spill/scratch. Generated code retains exact MMAC1472, symbol matrix
reads840 and ABarrier102; static waits fall `312 -> 300`. Full CPU-golden
correctness passes S128 causal/noncausal and S1024 causal with warning0 and
bank0.

Runtime rejects the hypothesis. Three S1024 pairs produce:

| Metric | Accepted | Read8 | Delta |
|---|---:|---:|---:|
| fused ticks | 42,164,243 | 42,919,392 | +1.791% |
| lifecycle ticks | 46,435,328 | 47,148,768 | +1.536% |
| MMAC active | 36.584% | 36.243% | -0.341 pp |
| wait-LGKM | 8.855% | 9.028% | +0.173 pp |
| barrier | 14.087% | 14.254% | +0.167 pp |
| wait-VM | 2.956% | 3.260% | +0.304 pp |
| coissue success | 20,032 | 19,785 | -1.230% |

VALU/SCA/LDS/FLAT remain exact, proving the regression is scheduling pressure,
not extra algorithmic work. Doubling the writer packet increases outstanding
LDS reads and live operand lifetime; the larger packet delays first use and
ownership release enough to erase the reduction in wait sites. The candidate
loses all three pairs, so S2048/fullperf are not admitted. Production source
and WDRA return exactly to commit `36af512`; retain only this evidence.

Evidence: `/zys/sb/runs/f5writer_read8_correctness_20260823` and
`/zys/sb/runs/f5writer_read8_ab_20260823`.
