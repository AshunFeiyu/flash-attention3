# FA3 BWD dKV Clean WASP Design Contract

## Algorithm

For each `Mq x Nk` score tile:

```text
S  = Q @ K^T
P  = exp2((S - m_i) * softmax_scale_log2) / l_i
D  = sum(dO * O)
dP = dO @ V^T
dS = P * (dP - D) * softmax_scale
dV += P^T @ dO
dK += dS^T @ Q
```

The clean design must not duplicate score or dP across D-split owners unless a
workbook row proves the resource/performance tradeoff.

## Initial Tile Thesis

```text
Mq = 32
Nk per consumer wave = 16
consumer waves = 8
resident Nk = 128
D = 128
threads = 16 waves * 64 lanes
```

Per consumer wave, expected MMAC count:

```text
score: (32/16) * (16/16) * (128/16) = 16
dP:    (32/16) * (16/16) * (128/16) = 16
dV:    (16/16) * (128/16) * (32/16) = 16
dK:    (16/16) * (128/16) * (32/16) = 16
total: 64 MMAC / q-tile / consumer wave
```

This keeps all four GEMM islands balanced.  That is the first algorithm-level
lesson from the old repo: smaller dV/dK islands or D-split score/dP duplication
can make local traces look busy while lowering useful MMAC density.

## Resource Budget

Target LDS plan:

```text
K/V resident, Nk=128,D=128:       2 * 128 * 128 * 2 = 65536 B
raw Q/dO double buffer:           2 * 2 * 32 * 128 * 2 = 32768 B
dO^T source-layout double buffer: 2 * 32 * 128 * 2 = 16384 B
sidecar double buffer:            2 * 3 * 32 * 4 = 768 B
planned total:                    115456 B
slack under 128KB:                15616 B
```

`Q^T` should reuse the raw Q semantic pages after raw consumers release them,
instead of allocating another full trans double buffer.  `Q^T` and `dO^T` are
loaded from source-layout ABI via MLS/BPS; do not do LDS raw-to-trans scatter in
the main path.

## Wave Roles

```text
waves 0-3:   producer A, K/V rows 0..63, Q raw, Q^T source-layout reuse
waves 4-7:   consumer group 0, Nk rows 0..63, dV+dK full D ownership
waves 8-11:  consumer group 1, Nk rows 64..127, dV+dK full D ownership
waves 12-15: producer B, K/V rows 64..127, dO raw, dO^T source-layout
```

This is the clean WASP goal: two recurring producers, two heavy consumer groups,
no one-time-only thin producer after startup.

## Pipeline Target

```text
time0:
  P0 loads K/V0 + Q_raw(0)
  P1 loads K/V1 + dO_raw(0)

time1:
  C0/C1 score+dP MMAC on raw(0)
  P0 prepares Q_raw(1) or Q^T(0) after raw release
  P1 prepares dO_raw(1) or dO^T(0)

time2:
  C0/C1 softmax+dS VALU for tile0
  P0/P1 publish trans operands for tile0/1

time3:
  C0/C1 dV+dK MMAC for tile0
  producers prefetch next useful packet
```

Expected XCompute pattern:

- MMAC islands are long enough to expose a real conveyor, not just a busy local
  trace.
- `ds_read_matrix -> s_waitcnt -> MMAC` gaps shrink because read batches and
  useful VALU are placed between issue and first use.
- Producer waves have recurring work after K/V startup.
- Consumer groups should not show long lockstep wait bands on the same trans
  ownership token.

## SQTT Evidence Contract

Use XCompute CLI as the default Wavefronts/SQTT path.  GUI inspection is a
fallback or human cross-check, not the primary evidence source.

Required CLI sequence for a perf candidate:

```bash
xcu status -P case.perf --sqtt-sections detail
xcu status -P case.perf --sqtt-sections wavefronts,bubbles --sqtt-dispatches 1 --sqtt-top 50
xcu status -P case.perf --sqtt-sections pipeline --sqtt-dispatches 1 --sqtt-time-range <start:end> --sqtt-location <loc> -F csv -D <out>/pipeline
xcu status -P case.perf --sqtt-sections simd --sqtt-dispatches 1 --sqtt-time-range <start:end> --sqtt-location <simd-loc> -F csv -D <out>/simd
```

The conclusion must identify the dominant bubble pattern and map it to the
pipeline design, for example `ds_read_matrix -> s_waitcnt -> v_mmac`, VALU
work not hidden by peer MMAC, barrier waits, instruction fetch/no-v gaps, or
SIMD imbalance.

## Bring-Up Sequence

The clean implementation moves in guarded cuts:

```text
Cut A, launch shell:
  real HIP launch, 16-wave WDRA, four role branches, ABarrier ledger
  no dV/dK math, no perf promotion

Cut B, producer packets:
  K/V resident load, raw Q/dO, source-layout Q^T/dO^T publication
  no consumer math promotion until packet correctness is probed

Cut C, first consumer island:
  score+dP MMAC only, with MLS/BPS + ds_read_matrix + v_mmac evidence

Cut D, full dKV:
  softmax+dS, dV+dK, store publication, correctness and perf gates
```

Every cut must keep the source readable enough to map Wavefronts/SQTT rows
back to a producer loop, consumer MMAC island, softmax/dS island, or epilogue.

## Done Metrics

Hard gates:

- correctness pass for delta, dV, dK
- no private segment, no scratch, no SGPR/VGPR spill
- LDS <= 128KB
- `ldsBankConflict=0`
- no ordinary DS main matrix path

Performance gates:

- primary: MMAC active share approaches or exceeds the same-run FA3 forward
  reference
- supporting: target-shape dKV ticks decrease, wait/barrier/no-v gaps reduce,
  `xcu` pipeline/SIMD CSV explains the pipeline
