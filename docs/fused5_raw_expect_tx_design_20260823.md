# Fused5 Raw BPS ABarrier EXPECT_TX

## Evidence And Hypothesis

The accepted G1-first kernel spends most producer time waiting for raw-page
reuse, then serializes each refill as:

```text
RawUsed -> 4 Q/dO BPS per producer wave -> sidecar -> vm/lgkm wait
        -> vbcnt drain -> RawFilled arrive
```

The registered `expect_tx_wiring_probe` proves on compiler `e0f10535` and PMD
HEAD1694 that `matrix_load_32x32_b16 ... bps lds` decrements ABarrier
transactions when payload bytes land in LDS. The unit is bytes, not operation
count. The multi-generation `expect_tx_page_cadence_probe` adds a stricter
rule: a transaction-tracked wave must not issue ordinary memory operations
between `expect_tx` and arrival. Mode18 passes only after BPS and sidecar work
are assigned to different waves.

## Transaction And Ownership Proof

For M64/D128 one raw page contains 16 BPS blocks:

```text
2 row blocks * (Q + dO) * 4 D blocks = 16 blocks
16 * 32 * 32 * sizeof(f16) = 32768 bytes
```

The producer role is deliberately asymmetric inside this ownership edge:

```text
wave0: BPS tasks 0,3,6,9,12,15 -> expect_tx(12288), no ordinary VMEM
wave1: BPS tasks 1,4,7,10,13   -> expect_tx(10240), no ordinary VMEM
wave2: BPS tasks 2,5,8,11,14   -> expect_tx(10240), no ordinary VMEM
wave3: all 64 sidecar rows, no expect_tx, plain arrival after VM/LDS wait
```

The transaction total is exactly `12288 + 10240 + 10240 = 32768` bytes.
All four waves contribute one arrival, so the barrier completes only when:

```text
pending_arrivals == 0 && transaction_bytes == 0
```

Consumers keep the same `RawFilled` wait, so they cannot read Q/dO before all
BPS bytes land or sidecar before wave3 arrives. `RawUsed`, page generation,
overwrite points and all downstream dS/dQ ownership remain unchanged.

## Expected Pipeline

```text
time0 P0-2: wait RawUsed0 -> expect_tx0(12K/10K/10K) -> Q/dO BPS0
      P3:   wait RawUsed0 -> sidecar0 -> vm/lgkm wait -> arrive
      P0-2: arrive RawFilled0 immediately after BPS issue; no vbcnt drain
      C: computes prior page

time1 P0-2: expect_tx1(12K/10K/10K) -> Q/dO BPS1 -> arrive
      P3:   sidecar1 -> vm/lgkm wait -> arrive RawFilled1
      C: RawFilled0 completes when all 32768 BPS bytes land, then consumes

time2 P: wait RawUsed0 while page1 transactions may finish
      C: consumes page0/page1 with unchanged five-GEMM schedule
```

The target is not fewer ABarrier events. It is allowing two page generations
of BPS traffic to be outstanding without a producer-side full VBCNT drain
between them.

## Invariants And Resource Budget

- Exact five GEMMs, M64/N128/D128, 16 waves and 1,280 useful MMAC per tile.
- Same 128KiB LDS, two raw pages, 12 ABarrier IDs, arrival counts and phase
  transitions.
- Same BPS count and bytes; no extra global/LDS transaction.
- Sidecar VM/LDS wait remains; resident K/V remains on the accepted VBCNT path.
- No tracked wave may execute sidecar or any other ordinary memory operation
  between its raw-page `expect_tx` and arrival.
- Main consumer path remains MLS/BPS + `ds_read_matrix` + MMAC.
- SGPR/VGPR metadata, role windows, private/spill/scratch and bank0 are hard
  gates.

Static control cost replaces two raw-page VBCNT wait sites with six immediate
`s_abarrier_expect_tx` sites: three producer-wave paths for each page. Any
extra MMAC, matrix read, BPS, token, page or output work rejects the candidate.
The producer uses one non-unrolled round-robin task loop to avoid duplicating
16 BPS instruction bodies. Static BPS and branch growth are explicit gates.

## Prior Evidence And Boundary

The same 6/5/5 pure-BPS plus one-sidecar wiring previously passed correctness
on the older Gen6 topology and was performance-neutral. Pairwise page rotation
then regressed by 18.5%, so rotation is explicitly forbidden here. This A5
operator test is admitted only because the accepted G1-first topology now
exposes `RawFilled` as a measured C1 wait edge. If current-topology ticks are
neutral or worse, EXPECT_TX is closed for this canonical path as well.

## Admission

1. Generated ASM must contain raw-page `s_abarrier_expect_tx` values
   `12288/10240/10240` before BPS, and no raw-page VBCNT wait before
   `RawFilled` arrival on waves0-2.
2. Re-run the exact fused gate and metadata gate; no resource debt.
3. Full CPU-golden correctness: S128 causal/noncausal and S1024 causal; bank0.
4. Three interleaved S1024 pairs against
   `best/fused5-writer-g1-first-20260823` decide ticks.
5. Only a ticks winner proceeds to S2048 and fullperf/XCU. XCU must show that
   producer page0/page1 BPS publication overlaps without moving the debt into
   consumer `RawFilled`, wait-VM, or matrix first-use bubbles.

This is workbook Section57.

## Result

Status: `REJECT_NEUTRAL_CONTROL_OVERHEAD_CANONICAL_RESTORED`.

- Static/resource gates pass: useful MMAC `1472`, symbol matrix reads `840`,
  SGPR82/VGPR128, and private/spill/scratch all zero. Runtime payload remains
  16 BPS blocks per page and LDS remains 128KiB.
- The compact round-robin publisher changes generated control shape:
  instruction BPS sites `20 -> 11`, waits `340 -> 335`, VBCNT waits `4 -> 1`,
  ABarrier instructions `102 -> 106`, and scalar branches `73 -> 97`.
- Full CPU golden correctness passes S128 causal/noncausal and S1024 causal;
  there is no panic, VGPR warning, or LDS bank conflict.
- Three interleaved S1024 fused means are `43,434,755` control versus
  `43,377,425` candidate, only `-0.132%`. Pair directions are mixed
  (`-1.080%`, `+0.746%`, `-0.058%`), so the candidate is noise-neutral.
- A representative pair shows why: SCA grows `46,656 -> 68,880` and VALU
  grows `117,156 -> 119,700`; coissue rises `24,164 -> 25,366`, but ticks do
  not improve stably. The dynamic 6/5/5 task loop trades VBCNT drain for too
  much producer scalar control.
- No S2048 or fullperf/XCU run is admitted. The experiment reproduces the
  older Gen6 neutral result, so EXPECT_TX raw publication is closed for the
  current G1-first canonical topology. Production source is restored exactly
  to commit `58e90fc`.

Evidence:

- `/zys/sb/runs/fused5_raw_expect_tx_correctness_20260823`
- `/zys/sb/runs/fused5_raw_expect_tx_ab_20260823`
- `/zys/sb/experiments/fused5_raw_expect_tx_20260823`
