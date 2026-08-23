# Fused5 dQ Writer First-MMAC Zero Seed

Status: `REJECT_PIPELINE_SKEW_AND_BARRIER_EXPOSURE_CANONICAL_RESTORED`.

## Measured Trigger

The accepted G1-first SQTT has 256 dynamic `v_mov_b64` instructions in one
representative writer wave. The writer clears eight independent dQ
accumulators at the start of every q tile before executing the same 64 dQ
MMACs.

## Exact Change

- Keep one four-VGPR zero fragment outside the writer q loop.
- For each M-panel and D-half, let G1 writer 0's MMAC define the accumulator
  from that zero fragment.
- Accumulate G1 writers 1--3 and all G0 writers exactly as before.
- Remove the per-tile dQ accumulator clear.

Per q tile, the G1 dQ GEMM stays at 32 MMACs: eight seed MMACs and 24 normal
accumulations. G0 stays at 32 normal accumulations. The five-GEMM work and
reduction order are unchanged after replacing `zero + first_term` with the
existing MMAC's zero accumulator input.

## Invariants

- M64/N128/D128, 16 waves and role ownership are unchanged.
- LDS remains 131,072 bytes; matrix reads and stores are unchanged.
- All 12 ABarrier IDs, phases, participant counts and event order are exact.
- Static MMAC1472, matrix-read840 and ABarrier102 must remain exact.
- The writer role may retain four additional live zero VGPRs but must remain
  inside WDRA budget with no private/spill/scratch.
- Draft0 kept the accepted `16/204/204/88` WDRA windows and failed the static
  gate with private160B and 43 VGPR spill operations: the backend writer branch
  was already `87/88`. Draft1 redistributes the same 512 physical registers as
  `16/204/200/92`; the C1 branch remains safely below its reduced 200 window
  at measured use164. No physical capacity is added.
- Full lifecycle golden and bank0 are mandatory.

## Expected Pipeline

```text
time0  writer waits G1; one persistent zero fragment is already ready
time1  first G1 writer MMAC seeds each of eight dQ accumulators
time2  remaining G1 and all G0 MMACs accumulate in the original order
time3  unchanged FP32-to-FP16 partial store
```

Promotion requires lower paired ticks at S1024 with no S2048 loss. Fullperf
must show lower writer `v_mov_b64` and no migration into ABarrier, wait-LGKM,
VGPR or store debt.

## Result

Draft0 failed the static resource gate with the accepted `16/204/204/88`
WDRA split: the writer was already at `87/88`, so retaining a zero fragment
caused `private_segment=160B` and 43 VGPR spill operations. Draft1 reused the
same 512 physical registers as `16/204/200/92`; it passed with writer91/92,
C1 164/200, private/spill/scratch0.

The intended code-generation change was achieved. Static MMAC1472,
matrix-read840 and ABarrier102 remain exact; MMAC islands improve `203 -> 194`,
singleton islands fall `19 -> 10`, and the static zero/move class falls by 46.
Full CPU-golden correctness passes S128 causal/noncausal and S1024 causal with
bank0.

Three interleaved S1024 pairs nevertheless regress fused mean
`43,065,447 -> 43,394,867` (`+0.765%`) and win only one pair. Dynamic VALU
falls `117,156 -> 112,612`, but MMAC active falls `35.304% -> 35.177%`,
coissue success falls `24,309 -> 22,986`, wait-LGKM rises `0.306 pp`, and
ABarrier rises `0.551 pp`. The removed writer clear was hidden beneath peer
MMAC and also provided useful phase skew; deleting it lets the writer reach
closed ownership waits earlier. Fewer instructions did not shorten the
critical path.

The S1024 primary gate failed, so S2048 and fullperf/xcu were not admitted.
Canonical source and WDRA are restored exactly to `58e90fc`. Do not retry this
zero-seed shape unless useful work replaces the lost skew or the ownership
topology changes.

Workbook: section59 in
`fa3_bwd_5gemm_clean_design_20260823_writer_zero_seed_rejected.xlsx`.
