# Fused5 dQ Writer G1-First Release

## Evidence

Accepted S1024 SQTT was exported for all four wave slots on one SIMD. The
writer has one startup `ResidentFilled` wait followed by a stable per-tile
pattern:

```text
wait G0 Filled: 1.3K-4.2K cycles
consume G0 dS
arrive G0 Done
wait G1 Filled: approximately 3 cycles
consume G1 dS
arrive G1 Done
```

C0 has two dS generations (`G0` and `G0Alt`) and its steady reuse waits are
normally approximately 3 cycles. C1 has one dS page and repeatedly waits for
`DqDone1` before republishing; representative waits reach 1.1K cycles. G1 is
already ready by the time the writer finishes G0.

## Hypothesis

Consume the single-buffered G1 page first, then the double-buffered G0 page:

```text
wait G1 Filled
consume G1 dS
arrive G1 Done
wait G0 Filled
consume G0 dS
arrive G0 Done
store the same dQ partial
```

The writer starts at most a few cycles later because G1 currently follows G0
readiness by only about three cycles. It releases C1 one complete G0 dQ MMAC
island earlier. Delayed G0 release is expected to be absorbed by C0's
alternate generation.

## Invariants

- Exact five logical GEMMs and unchanged MMAC count.
- Unchanged M64/N128/D128 tile, 16-wave roles, dQ ownership and result order.
- Unchanged 128KiB LDS layout and all matrix/global transactions.
- Same 12 ABarrier IDs, participant counts, generations and phase variables.
- No new waits, branches, stores, pages, tokens, or duplicate work.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- No private segment, spill, scratch, or LDS bank conflict.

## Expected Pipeline

```text
time0: writer waits G1; C0/G0 is already ready or nearly ready
time1: writer G1 dQ MMAC; C1 receives Done1 and starts next publication
time2: writer waits G0 (expected ready); writer G0 dQ MMAC
time3: writer stores dQ; C0 uses alternate generation; C1 advances dK/RawUsed
```

The intended gain is not a lower first writer wait by itself. It is shorter
single-page C1 backpressure, earlier C1 dK/RawUsed progress, and a more regular
producer/consumer conveyor.

## Admission

1. Generated ISA must preserve MMAC/read/wait/ABarrier counts and only reverse
   the writer G0/G1 consume blocks.
2. Role VGPR/SGPR and all spill/private/scratch gates must match the accepted
   `799a255` baseline.
3. Full golden correctness must pass S128 causal/noncausal and S1024 causal;
   LDS bank conflicts must remain zero.
4. Three interleaved S1024 A/B pairs decide ticks. Higher active/coissue alone
   is not promotion.
5. Winner SQTT must show lower C1 `DqDone1` wait without a larger net writer,
   producer, or terminal bubble. S2048 must not regress.

The corresponding workbook design is Section 54.

## Result

Status: `ACCEPT_STRUCTURAL_BARRIER_RELEASE_MMAC50_OPEN`.

- Generated ISA and resources remain exact: MMAC1472, symbol matrix-read840,
  wait340, ABarrier102, `v_mov_b64`68, roles `9/176/87/164`, SGPR82/VGPR128,
  private/spill/scratch0.
- Full CPU-golden correctness passes S128 causal/noncausal, S1024 and S2048;
  PMD reports warning0 and `ldsBankConflict=0`.
- Three paired S1024 runs improve fused mean
  `44,078,732 -> 43,251,693` (`-1.876%`). Two paired S2048 runs improve
  `82,494,685 -> 79,781,520` (`-3.289%`).
- Fullperf fused ticks improve `44,274,685 -> 43,318,730` (`-2.159%`),
  MMAC active rises `34.821% -> 35.037%`, barrier share falls
  `14.140% -> 13.263%`, and wait-LGKM falls `7.820% -> 7.428%`.
- XCU dispatch duration falls `97,308 -> 95,204`. Consumer0, consumer1 and
  writer MMAC+VALU coissue events rise by `107/236/200` on the sampled SIMD.

The mechanism is cost migration, not disappearance. C1's steady `DqDone1`
wait falls `7,561 -> 45` cycles, after which it reaches the next `RawFilled`
earlier (`45 -> 7,901` cycles). This advances dK/RawUsed enough that producer
ABarrier cycles fall `65,239 -> 49,679`; writer ABarrier cycles also fall
`32,643 -> 29,571`. C0 remains protected by its alternate generation. The
remaining debts are exposed wait-VM (`2.845% -> 3.325%`), terminal ebarrier,
and the still-open 50% MMAC-active target.

Evidence:

- `/zys/sb/runs/fused5_writer_g1_first_ab_20260823`
- `/zys/sb/runs/fused5_writer_g1_first_s2048_ab_20260823`
- `/zys/sb/runs/fused5_writer_g1_first_fullperf_20260823`
- workbook Section 54 results
