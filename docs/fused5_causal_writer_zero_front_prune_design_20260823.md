# Fused5 Causal Writer Zero-Front Prune Design

Status: `DESIGN_ADMITTED / STATIC_GATE_PENDING`.

## Algebraic Proof

For every causal CTA, the first retained Q tile and the two N64 consumer
groups own:

```text
Q0 rows   = [k_base + 0,  k_base + 63]
C0 K rows = [k_base + 0,  k_base + 63]
C1 K rows = [k_base + 64, k_base + 127]
```

The causal predicate is `k_row <= q_row`, so every C1 element in Q0 is
invalid. C83 already removes C1's first score/dP/dV/dK work, but preserves a
native zero-dS publication and makes the dQ writer read that zero page and
execute `dQ += 0 @ K1`. Those writer reads and MMACs are also provably zero.

This is not an accumulator zero-seed experiment. The writer keeps its accepted
explicit accumulator initialization and C0 accumulation order.

## One Hypothesis

For the causal symbol only, and only when `Generation==0 && qi==0`:

1. C1 waits `RawFilled0`, but publishes no zero dS payload;
2. C1 still arrives `BatchDsFilled1`, `RawUsed0` and `DqDone1`;
3. the writer still waits `BatchDsFilled1` and arrives `DqDone1` so barrier
   phases and the next reuse epoch remain exact;
4. the writer skips the G1 `ds_read_matrix` packet and its 32 dQ MMACs;
5. the writer executes the unchanged G0 read/MMAC and stores the final tile.

The implementation may add one runtime `qi==0` branch inside the causal
Generation0 writer body. It may not instantiate a second G0 body, duplicate
the store body, add a token/page, or change the noncausal symbol.

## Work Ledger

| Item | Accepted C83 | Candidate causal delta |
| --- | ---: | ---: |
| logical GEMMs | exact five formulas | exact, invalid domain removed |
| tile / roles | M64/N128/D128, 4P+4C0+4C1+4W | exact |
| C1 first-tile consumer MMAC | already removed | exact |
| writer first-tile G1 MMAC | 4 waves x 32 = 128 per CTA | -128 per CTA |
| C1 zero dS writes | 4 waves x 4 = 16 per CTA | -16 per CTA |
| writer G1 matrix packets | 4 waves x 4 panels = 16 per CTA | -16 per CTA |
| ABarrier events | existing Filled1/Done1 cadence | exact |
| dQ stores | 4 panels x 2 vectors x 4 waves | exact |

H1/S1024 causal launches eight K-tile CTAs. The dispatch must therefore remove
1,024 zero writer MMOP issues, 128 zero dS writes and 128 zero-page matrix
read packets. Dynamic MMOP is expected to move `88,064 -> 87,040`; this is a
useful-work reduction, so same-shape ticks and effective FLOPs are primary.

## Ownership And Phase Proof

`BatchDsFilled1` has four C1 arrivals and four writer waiters. `DqDone1` has
four C1 arrivals plus four writer arrivals. The zero-front epoch must keep all
eight Done arrivals even though no payload is consumed. Therefore the next
C1 `ReuseDs` wait cannot observe an early or stale phase, and the writer's
next Filled1 wait advances from the same phase as the producer group.

`RawUsed0` also remains unchanged: C0 is still the last real raw-page user,
while C1 retains its existing arrival. Producer page reuse is not accelerated
or weakened by this experiment.

## Resource Budget

| Resource | C83 | Candidate gate |
| --- | ---: | ---: |
| LDS | 128 KiB | exact |
| barrier IDs | 12 | exact |
| WDRA | 16/204/204/88 | exact |
| causal role use | 9/173/87/162 | no growth |
| metadata | SGPR70/VGPR128 | private/spill/scratch0 |
| matrix path | MLS/BPS + ds_read_matrix + MMAC | exact |

The causal symbol should retain the accepted static MMAC body because the
Generation0 G1 body is still needed at `qi>=2`. Admission therefore checks
for one small branch and no increase in static MMAC/read/store sites. A second
inlined G0/store body rejects the experiment before PMD.

## Expected Pipeline

```text
time0  P: publish Raw0
       C0: diagonal score/P/dP/dS -> dV/dK
       C1: wait Raw0 -> signal zero epoch, no payload write
       W : wait Filled1 -> signal Done1, no G1 read/MMAC

time1  W : wait Filled0 -> G0 dS read -> dQ MMAC -> store
       P : prepares Raw1 while C0 remains the Raw0 pace setter

time2+ accepted G1-first/G0-second steady pipeline, unchanged
```

This does not claim to eliminate the 22.13% aggregate ABarrier gap. It removes
real zero work from the writer critical role while preserving the ownership
DAG, then measures whether the shorter first epoch lowers end-to-end ticks.

## Admission

1. Workbook proof is complete before source editing.
2. Static causal/noncausal symbols retain exact five-GEMM declarations,
   matrix paths, LDS, token IDs, stores and no spill/private/scratch.
3. Causal static MMAC/read/store counts do not grow; only a bounded branch is
   allowed. Noncausal ISA is byte-equivalent or count-equivalent to C83.
4. Full S128 causal/noncausal correctness passes with warning0 and bank0.
5. Three interleaved H1/S1024 pairs decide promotion. Candidate dynamic MMOP
   must be 87,040 and fused/lifecycle mean ticks must both improve.
6. Only a repeated ticks win is admitted to S2048 and fullperf/xcu. Otherwise
   restore C83 and retain only this design/result evidence.

Workbook: sections87-88 in the 2026-08-23 fused5 design workbook.

## Result

Status: `REJECT_S2048_SCALING_RUNTIME_BRANCH_OVERHEAD`.

- Static resource gates pass. Causal role use becomes `9/173/85/162` from
  `9/173/87/162`; SGPR70/VGPR128 and private/spill/scratch0 remain exact.
  Noncausal role use and opcode counts are unchanged.
- The causal symbol has no MMAC/read/store body growth. It removes four static
  zero dS writes and one wait, but compiler control flow adds five scalar
  branches and 28 `v_mov_b64` sites.
- Full S128 causal/noncausal and all S1024/S2048 runs pass dot/dK/dV/dQ CPU
  golden checks with warning0, nonfinite0 and bank0.
- S1024 dynamic work matches the proof exactly: MMOP `88,064 -> 87,040` and
  LDS `61,056 -> 60,416`; VMEM/FLAT remain `1,408/3,616`. Compiler overhead
  raises VALU `90,032 -> 91,600` and SCA `35,592 -> 35,688`.
- Three interleaved S1024 pairs all improve. Fused mean moves
  `41,261,220 -> 40,761,023` (`-1.212%`) and lifecycle mean moves
  `45,444,187 -> 44,971,138` (`-1.041%`).
- Two interleaved S2048 pairs do not scale. Fused mean moves
  `76,429,308 -> 76,611,535` (`+0.238%`) and lifecycle mean moves
  `83,722,958 -> 84,084,000` (`+0.431%`).
- The zero-domain saving happens once per CTA, while the compiler's branch/phi
  move cost recurs in the longer q loop. No fullperf is admitted. Production
  source is restored byte-for-byte to C83, and this runtime-branch topology is
  closed. A future retry must peel the epoch without duplicating G0/store and
  without adding recurring accumulator moves.

Evidence:

- remote experiment:
  `/zys/sb/experiments/fused5_causal_writer_zero_front_prune_20260823_c87`
- S1024 A/B:
  `/zys/sb/runs/fused5_c87_ab_control` and
  `/zys/sb/runs/fused5_c87_ab_candidate`
- S2048 A/B:
  `/zys/sb/runs/fused5_c87_s2048_control` and
  `/zys/sb/runs/fused5_c87_s2048_candidate`
