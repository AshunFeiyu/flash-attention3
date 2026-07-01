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
sidecar/slack seed:               about 512 B initially
planned total:                    98816 B
slack under 128KB:                about 28 KB
```

Do not allocate another full LDS raw-to-trans scratch in the main path.  If
`Q^T` or `dO^T` source-layout operands are needed for MMAC, they must be loaded
through MLS/BPS into pages whose ownership/lifetime is explicit in the workbook,
or reuse released raw-page capacity.  LDS raw-to-trans scatter remains rejected
until a focused probe proves bank-conflict-free behavior and a resource row pays
for it.

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
T0:
  P0 loads K resident + Q page0
  P1 loads V resident + dO page0

T1:
  C0 score+dP MMAC on page0
  C1 reads operands for page0
  P0/P1 prefetch Q/dO page1 and sidecar metadata

T2:
  C0 softmax+dS VALU for page0 plus dV operand reads
  C1 score+dP MMAC on page0
  P0/P1 prepare next useful raw/source-layout pages

T3:
  C0 dV MMAC page0
  C1 softmax+dS VALU page0

T4:
  C0 dK MMAC page0
  C1 dV MMAC page0

T5:
  C0 releases page0 and advances
  C1 dK MMAC page0 and releases page0
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

The current FWD/BWD SQTT gap analysis is recorded in
`docs/sqtt_fwd_bwd_gap_20260701.md`.  Its main conclusion is that the BWD dKV
trace already contains MMAC and `ds_read_matrix`, but fails to match FWD because
barrier/control serialization and read-to-use latency break the conveyor.

The current workbook source of truth is:

```text
/Volumes/172.20.68.76/共享/shaobo/fa3_bwd_wasp_clean_design_20260701.xlsx
```

The workbook must be updated before changing the tile shape, output ownership,
ABarrier ledger, or source-layout operand budget.

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

MLS/BPS publication rule:

```text
producer: matrix_load_32x32_b16 ... bps lds for a whole producer packet
producer: ABarrier PacketFilled
consumer: ABarrier wait PacketFilled -> ds_read_matrix -> wait before first MMAC use
```

Do not put `wait_lgkm(0)` immediately after producer `matrix_load`; that drains
the producer instead of letting the ABarrier token carry packet ownership.
Use explicit waits only before a wave consumes its own LDS read result or before
an actual overwrite/reuse hazard that the ownership ledger does not cover.
Avoid fragment-level barrier ledgers such as separate Raw, Trans, K, and V
tokens when the fragments are always consumed together.  That shape can create
`abarrier -> abarrier` chains in SQTT and should be replaced by coarse producer
packet ownership unless a later multi-packet proof needs finer lifetimes.

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
