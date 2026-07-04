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

## Current Tile Thesis

```text
Mq = 64 experimental, Mq = 32 canonical rebaseline
Nk per consumer wave = 16
consumer waves = 8
resident Nk = 128
D = 128
threads = 16 waves * 64 lanes
```

Per consumer wave, expected MMAC count for the current Mq64 experiment:

```text
score: (64/16) * (16/16) * (128/16) = 32
dP:    (64/16) * (16/16) * (128/16) = 32
dV:    (16/16) * (128/16) * (64/16) = 32
dK:    (16/16) * (128/16) * (64/16) = 32
total: 128 MMAC / q-tile / consumer wave
```

This keeps all four GEMM islands balanced.  The Mq64 same-LDS experiment is
resource-clean but not promoted yet: H1/S1024 remains slower than the Mq32
canonical rebaseline because the dominant RawUsed/control bubble is unchanged.

## Resource Budget

Target LDS plan:

```text
K/V resident, Nk=128,D=128:       2 * 128 * 128 * 2 = 65536 B
raw Q/dO single buffer, Mq64:     1 * 2 * 64 * 128 * 2 = 32768 B
sidecar LDS, Mq64:                1 * 3 * 64 * 4 = 768 B
planned total:                    99072 B
slack under 128KB:                32000 B
```

Do not allocate another full LDS raw-to-trans scratch in the main path.  The
verified current contract is `matrix_load_32x16_b16` plus normal/trans
`ds_read_matrix` from the same raw `Q/dO` LDS page.  This removes the external
source-layout ABI.  The current promoted sidecar route additionally moves
`max_log2/inv_sum/delta` into producer-published LDS so consumer softmax/dS no
longer performs direct sidecar global loads.

The remaining LDS slack is not free performance.  It should be spent only after
a workbook-stressed ABarrier/page-lifetime design proves that extra raw/sidecar
pages reduce token waits more than they add control cost.

## Wave Roles

```text
waves 0-3:   producer A, K resident + raw Q same-LDS pages + future sidecar prefetch
waves 4-7:   consumer group 0, Nk rows 0..63, dV+dK full D ownership
waves 8-11:  consumer group 1, Nk rows 64..127, dV+dK full D ownership
waves 12-15: producer B, V resident + raw dO same-LDS pages
```

This is the current W16 structural probe shape.  It is correctness/resource
clean, but the first H1/S1024 full perf shows it is not yet a performance
promotion: `kernel_ticks=69039425`, `MMAC active=22.3357%`, and xcu still shows
`s_abarrier_try_wait -> s_xor_b32` as the dominant bubble.  The next edit must
reduce the raw-page ABarrier/control lifetime or create real useful producer
work; simply adding the second producer group is not enough.

## Pipeline Target

```text
T0:
  P0 loads K resident + Q page0 + sidecar page0
  P1 loads V resident + dO page0

T1:
  C0 score+dP MMAC on page0
  C1 reads operands for page0
  P0/P1 prefetch Q/dO page1 and sidecar metadata

T2:
  C0 softmax+dS VALU for page0 plus dV operand reads
  C1 score+dP MMAC on page0
  P0 prepares next useful raw page and sidecar prefetch

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

Current actual H1/S1024 single-buffer sidecar-LDS result:

- `kernel_ticks=54539485`, `MMAC active=26.6693%`,
  `ldsBankConflict=0`.
- The old consumer `global_load_dwordx3 -> s_waitcnt` top bubble is removed.
- xcu now shows `s_abarrier_try_wait -> s_xor_b32` `47.39%` and
  `s_abarrier_try_wait -> s_waitcnt` `6.60%`; selected window
  `100000:118000` has `96.76%` SIMD bubble.  This means sidecar LDS is useful,
  but the single raw-page protocol is still not a long conveyor.

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

Semantic page reuse boundary:

```text
raw generation:    page = Q/dO
  trans view:        same raw page read by trans ds_read_matrix
```

The old semantic-page source generation is no longer the current contract.  The
same-LDS 32x16 contract is correctness-proven, but Mq64 A/B show that larger Wq
still needs a separate RawUsed/control solution before promotion.

Sidecar LDS boundary:

```text
K/V resident generation:
  producer publishes K/V into LDS
  consumers latch K/V into VGPR exactly once

sidecar generation:
  producer writes max_log2 / inv_sum / delta into a small dedicated LDS region
  RawFilled covers Q, dO, and sidecar readiness for that q tile
```

This is the active contract after the 2026-07-04 single-buffer convergence
pass.  Sidecar stays in LDS because that removed the old consumer global-load
wait, but it no longer overlays K/V and does not require `ResidentUsed`.
`Raw1` is also removed from the active route.  Reintroducing either token needs
fresh workbook and xcu evidence.

Causal skip boundary:

```text
whole-tile skip condition:
  causal && q_tile * Mq + Mq - 1 < k_base
```

This is algorithmically valid and was verified in the W12 causal-skip path, but
it regressed H1/S1024 performance.  MMOP dropped from `131072` to `73728`,
while `kernel_ticks` worsened from `71006845` to `72881900` and MMAC active
share fell from `21.6777%` to `16.7128%`.  The lesson is that deleting
upper-triangle work can make the conveyor thinner and more tail-limited.  Any
future causal pruning must prove that SIMD balance and MMAC active share
improve, not merely that total instruction count decreases.

## Bring-Up Sequence

The clean implementation moves in guarded cuts:

```text
Cut A, launch shell:
  real HIP launch, 16-wave WDRA, four role branches, ABarrier ledger
  no dV/dK math, no perf promotion

Cut B, producer packets:
  K/V resident load, raw Q/dO same-LDS publication
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
