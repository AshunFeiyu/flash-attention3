# Client

## Mission

Build one clean Shaobo FA3 BWD dKV kernel in the FA3 FWD style. dQ is frozen.
The main optimization target is MMAC active share, with FA3 FWD as the hard
benchmark. Correctness, no scratch/spill, `ldsBankConflict=0`, and explainable
SQTT evidence are required before any performance claim.

## Current Canonical State

- Repo focus: `/zys/shaobo/fa3_bwd_wasp_clean`.
- Active tile: `Mq=64, Nk=128, D=128`, 16 waves, `GPU_CHIP=sb`,
  `GPU_ARGS="['--SQCIPfLines=7']"`.
- Wave roles:
  - waves0-3: producer K + Q + sidecar
  - waves4-7: consumer group 0, owns `Nk=0..63`
  - waves8-11: consumer group 1, owns `Nk=64..127`
  - waves12-15: producer V + dO
- Main path: `matrix_load_32x16/32x32 ... bps lds` +
  normal/trans `ds_read_matrix` + `v_mmac_*lit`.
- Raw Q/dO use two page-local ABarrier generations. K/V is latched into
  consumer VGPR, then raw pages overlay the K/V LDS region.
- Output ownership is unique: every consumer owns one `Nk16 x D128` dV/dK
  slice. Do not duplicate score/dP for the same owner.

## Latest Evidence

- Canonical raw2 H1/S1024 stats-only baseline:
  `kernel_ticks=53,300,975`, `MMAC active=27.6518%`, `MMOP=131,072`,
  `ldsBankConflict=0`.
- Full perf archive:
  `/Volumes/172.20.68.76/共享/shaobo/perf/20260704_221910_clean_raw2_tokens_h1s1024_sqc7_fullperf`.
- XCU evidence: top bubble is still
  `s_abarrier_try_wait -> s_xor_b32` at about `40.24%`; the selected
  Raw1Used SIMD window is about `93.94%` bubble and only `1.05%` MMAC.
- Rejected after raw2: raw3 page depth, consumer1 score-prefetch stagger,
  causal=true specialization, score read batch2, WG-local duplicate Q/dO, and
  causal invalid-prefix page skip, raw release before softmax, and sidecar
  ring3 early release, and producer sidecar rebalance.  All
  resource/correctness-clean candidates regressed or failed to improve
  same-shape performance.

## Current Diagnosis

The kernel does have MMAC and the matrix path is not the primary missing piece.
The active limiter is packet ownership, topology, and causal redundant work:

- producer1 is still thin after V startup plus dO publication;
- both consumer groups share CTA-wide raw Q/dO page ownership;
- RawUsed/ABarrier wait is not hidden by useful MMAC/VALU work;
- local instruction tweaks raise coissue counters but do not shorten the
  critical path.
- for causal=true, pages with `q_tile_end < k_base` are mathematically all
  invalid for the CTA, but the current baseline still computes the GEMMs and
  masks the result to zero.

## Next Experiment

Workbook sheet `51_structural_pivot` records the rejected WG-local duplicate
Q/dO structural probe:

```text
WG0: waves0-3 producer K0/V0 + Q/dO + sidecar
     waves4-7 consumer0, owns Nk0..63

WG1: waves12-15 producer K1/V1 + Q/dO + sidecar
     waves8-11 consumer1, owns Nk64..127
```

It passed correctness/resource gates but regressed H1/S1024 full perf to
`simTicks=58,310,070`, `MMAC active=26.7125%`, and doubled VMEM. Do not pursue
independent warpgroup ownership by duplicating shared Q/dO.

Workbook sheet `52_causal_page_skip` records a rejected algorithm candidate:

```text
If causal && q_tile_end < k_base:
  producers advance the raw token without Q/dO/sidecar MLS publication;
  consumers wait the raw token, immediately arrive RawUsed, and skip
  score/dP/softmax/dV/dK for that page.
```

The resource-clean prefix-only implementation passed H1/S128 and H1/S1024
correctness and reduced H1/S1024 MMOP from `131,072` to `77,312`, but
`kernel_ticks` stayed flat/slightly worse at `53,474,330` and MMAC active
dropped to `22.4979%`. Do not keep this consumer-side skip in the canonical
route; future causal work needs launch/tile ownership or critical-path
restructuring.

Workbook sheet `53_score_dp_pair_asm` records the first asm-island negative:

```text
Keep the C++ clean kernel and replace only the hot score/dP read island:
  four independent ds_read_matrix_trans reads
    -> two ds_read_matrix_trans_pair asm helper calls
```

It built cleanly and produced the intended pair-read asm, but H1/S128
correctness failed. Root cause: the existing `ds_read_matrix_trans_pair`
assumes the second fragment is `+1024` bytes from the first, while raw Q/dO
M0/M1 score/dP reads are separated by `4 * 1024` bytes in the D128 raw page.
The source is reverted to the raw2 baseline. Future asm islands must first
prove the exact LDS adjacency relation for the target layout.

Workbook sheets `54_raw_release_before_softmax` and
`55_sidecar_ring3_early_raw_release` record two raw-lifetime negatives:

```text
54: pre-read Q/dO source and arrive RawUsed before softmax
    -> H1/S1024 correctness fails because sidecar is still live.

54b: pre-read sidecar rows to VGPR too
    -> static fails with private=52, vgpr_spill=24.

55: keep raw2 but add sidecar ring3, then release raw before softmax
    -> correctness/resource pass, but H1/S1024 kernel_ticks=55,298,425,
       MMAC active=26.5015%, worse than raw2 recert 53,008,410 / 27.7754%.
```

Do not retry page/ring-depth work in isolation.  A future raw-lifetime design
must also remove ownership/control cost or enlarge a useful MMAC island.

Workbook sheet `56_producer_sidecar_rebalance` records a producer topology
negative:

```text
producer0: K + Q + sidecar   ->   K + Q
producer1: V + dO            ->   V + dO + sidecar
```

It passed correctness/resource gates and moved branch windows from
`6/198/198/1` to `1/198/198/6`, but H1/S1024 regressed to
`kernel_ticks=53,558,960`, `MMAC active=27.5554%`.  Coissue rose, but the
RawUsed/consumer critical path did not shorten.  Do not equate producer visual
balance with useful pipeline overlap.

## Workflow Rules

1. Update the shared workbook before changing tile shape, output ownership,
   ABarrier ledger, or operand lifetime.
2. Keep one canonical dKV kernel. Rejected experiments are reverted and recorded
   in `results/perf_ledger.csv` and the workbook.
3. Run static gates, H1/S128 correctness, H1/S1024 correctness, PMD stats, then
   XCU CLI. GUI Wavefronts is optional human review only.
4. Judge by MMAC active share first, then same-shape ticks and SQTT explanation.
   Coissue dominated by `v_mov` is not a success.
5. Commit every accepted or rejected evidence checkpoint before starting the
   next hypothesis.
