# Fused5 C1 Early dS Publication

## Evidence

The accepted `58e90fc` writer order releases the shallow C1 dS queue first.
Fresh XCU then attributes `21,928` writer ABarrier cycles to the C1 Filled
stream, versus `6,824` for C0. C1 is still executing one complete dV GEMM
before its batch Filled event:

```text
C1 panel x4: dP -> score -> P/dS -> dV
publish dS batch -> dK
```

For each q tile and C1 wave, dV is 32 MMAC. It is useful work, but it delays
the dS ownership edge needed by the independent dQ writer.

## Formula And Ownership Proof

The five logical GEMMs remain exact:

```text
score = Q @ K^T
dP    = dO @ V^T
dV   += P^T @ dO
dK   += dS^T @ Q
dQ    = dS @ K
```

Only C1 scheduling changes:

```text
current:   score/dP/P/dS x4 -> dV x4 -> publish dS -> dK
candidate: score/dP/P/dS x4 -> publish dS -> dV x4 -> dK
```

C1 retains four FP16 P fragments until dV. The dQ writer consumes exactly the
same four native dS panels and signals the same DqDone token. RawUsed remains
after dV and dK, so producer overwrite safety does not change.

## Resource Budget

- Tile remains M64/N128/D128, 16 waves, exactly 1,280 useful MMAC per logical
  tile.
- Per C1 wave: score32 + dP32 + dV32 + dK32 = 128 MMAC, unchanged.
- Per writer wave: G1 dQ32 + G0 dQ32 = 64 MMAC, unchanged.
- C1 retains four `F16x8` P fragments, at most 16 extra VGPR. Accepted actual
  C1 use is 87 in a 204-VGPR role window.
- LDS remains 128KiB; 12 ABarrier IDs, all pages and all transaction counts
  remain unchanged.
- SGPR/VGPR metadata, private/spill/scratch and bank0 remain hard gates.

## Expected Pipeline

```text
time0  C1: dP/score/P/dS x4 -> publish G1
       C0: score/P/dP/dS x4

time1  W : wait G1 -> dQ(G1) MMAC32 -> release G1
       C1: dV MMAC32
       C0: publish G0 -> dV MMAC32

time2  W : dQ(G0) MMAC32 -> store partial dQ
       C1: dK MMAC32 -> RawUsed
       C0: dV/dK MMAC -> RawUsed
```

The intended gain is earlier dQ work and overlap with existing dV/dK, not
fewer GEMMs or fake delay. The main risk is that simultaneous writer/C1 MMAC
islands contend for the same SIMD MMAC issue slots or expose unhidden C1 dV
matrix-read waits.

## Admission

1. Generated ASM: MMAC1472, symbol matrix-read840, ABarrier102, and main
   matrix opcodes exact; no extra branch/token/page.
2. C1 role <=204, private/spill/scratch0, LDS128KiB.
3. Full cached CPU-golden PASS at S128 causal/noncausal and S1024 causal;
   warning0 and bank0.
4. Three interleaved S1024 pairs against `58e90fc` decide ticks.
5. A winner must reduce writer G1 Filled cycles and fused ticks. Higher MMAC
   active without lower ticks is not sufficient.
6. Only a winner proceeds to S2048 and candidate fullperf/XCU.

This is workbook Section56.

## Result

Status: `REJECT_LOCAL_DV_OVERLAP_LOSS_CANONICAL_RESTORED`.

The generated kernel preserves MMAC1472, matrix-read854 file-wide,
ABarrier102, branches46 and `v_mov_b64`68. Static waits grow only
`340 -> 342`; metadata remains SGPR82/VGPR128 with no private segment,
spill or scratch. Full cached CPU-golden correctness passes S128
causal/noncausal and S1024 causal, with warning0 and bank0.

Three interleaved S1024 pairs measure fused means
`43,371,207 -> 43,561,093`, a `+0.438%` regression. Pair results are mixed:
the candidate wins one pair, loses one, and is effectively flat in the third.
Publishing C1 dS before dV therefore does not create a stable net overlap on
this topology. It removes C1's proven dS/next-dO-read/dV local schedule and
moves dV matrix readiness into a later island; any earlier writer start is
offset by this consumer-local loss and MMAC contention.

No S2048 or candidate fullperf is admitted. Canonical source is restored to
tag `best/fused5-writer-g1-first-20260823` behavior.

Evidence:
`/zys/sb/runs/fused5_c1_early_ds_publish_correctness_20260823`,
`/zys/sb/runs/fused5_c1_early_ds_publish_ab_20260823`, local A/B script under
`outputs/019ea61f-c117-76b2-abad-e776092d47a0/fused5_c1_early_ds_publish_ab_20260823`,
and workbook section56.
