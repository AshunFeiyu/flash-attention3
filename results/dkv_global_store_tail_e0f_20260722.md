# dKV Global-Store Tail: LLVM47a7 vs e0f10535

Date: 2026-07-22

## Verdict

The longer-looking global-store tail under LLVM `e0f10535` is not a compiler
code-generation regression.  The old and new compiler builds emit the same
target instruction stream for the canonical dKV kernel, including the same
`global_store_dwordx4` sequence, registers, address arithmetic and ordering.
The visible difference comes from PMD VMEM arbitration and the phase at which
the two heavy consumer groups reach the terminal AllDone convergence.

Keep the latest compiler and the canonical direct-vector-store epilogue.  Do
not replace it with an LDS writer/matrix-store path, and do not split stores
into the final MMAC island.  The actionable optimization boundary is consumer
completion skew before the epilogue, not the store opcode sequence itself.

## Evidence Chain

Environment is fixed to PMD HEAD1694, config seed `c22d6a42`, `GPU_CHIP=sb`,
`GPU_ARGS=['--SQCIPfLines=7']` and the same canonical source.

| Evidence | LLVM47a7 | e0f10535 run 1 | e0f10535 run 2 | Interpretation |
|---|---:|---:|---:|---|
| fullperf kernel ticks | 35,077,315 | 34,625,955 | 34,692,840 | new compiler is not slower in these trace runs |
| aggregate global-store latency | 217,728 | 221,452 | 218,652 | run 2 is only 0.42% above old |
| aggregate store issue gaps | 187,312 | 191,192 | 188,128 | run 2 is only 0.44% above old |
| SpTaData cycles | 16,384 | 16,384 | 16,384 | exact data traffic |
| SpTaAddr cycles | 4,384 | 4,384 | 4,384 | exact address traffic |
| representative slot1 MMAC-to-last-store | 2,904 | 2,888 | - | new is 16 cycles shorter |
| representative slot2 MMAC-to-last-store | 4,872 | 4,672 | - | new is 200 cycles shorter |

Normalized assembly is identical after removing HIP CUID/debug-path/DWARF
noise.  Dynamic `MMOP`, `VALU`, `SCA`, `LDS`, `VMEM`, `FLAT` and output bytes
are also exact for the compiler A/B.  Therefore a compiler-generated store
sequence cannot explain the GUI-level difference.

The matched-wave timeline makes the boundary unambiguous:

| Physical consumer slot | Compiler | last MMAC -> first store | store issue span | last store -> wave end |
|---|---:|---:|---:|---:|
| slot1 / consumer0 | LLVM47a7 | 252 | 2,652 | 4,081 |
| slot1 / consumer0 | e0f10535 | 272 | 2,616 | 4,085 |
| slot2 / consumer1 | LLVM47a7 | 1,288 | 3,584 | 1,761 |
| slot2 / consumer1 | e0f10535 | 1,024 | 3,648 | 1,761 |

The new compiler is 16 cycles shorter from the last MMAC to the last store on
slot1 and 200 cycles shorter on slot2.  The four-cycle slot1 post-store delta
is below PMD issue granularity, while slot2 is exact.  What looks like a long
`global_store` tail in the GUI is therefore mostly time attributed to the
store-adjacent issue gap plus `AllDone` convergence; it is not extra store
instructions or a new store dependency emitted by e0f10535.

## Rejected Workarounds

### Reversing C1 dV/dK Store Order

C0 kept `dK -> dV`; C1 used `dV -> dK` to stripe peer streams.  Correctness,
resources and exact work passed, but three interleaved S1024 pairs were mixed.
The paired median was about `+0.12%` slower.  The PMD VMEM pipe did not gain a
stable scheduling benefit, so canonical ordering was restored.

### Final-Q-Tile Early Drain

The low D0-D63 outputs were stored after their final MMAC and before the final
D64-D127 MMAC, leaving only half the outputs in the terminal store epoch.

- Compile-time whole-q-tile specialization was numerically correct but doubled
  static MMAC code from `1028` to `2052`, increasing instruction-footprint
  pressure.  It was not promotable.
- A compact version kept static MMAC at `1028`, passed S128 correctness, used
  branch VGPR `162/168`, and had no spill/private/scratch or bank conflict.
- All three interleaved S1024 pairs regressed: `+1.452%`, `+1.306%`, and
  `+1.300%`; paired median regression was `+1.306%`.
- Median MMAC active fell from `38.4912%` to `37.5363%`.  Dynamic FLAT/VMEM
  stayed exact (`1096/2560`), but SCA rose `38048 -> 39800` and the split store
  interrupted the last MMAC island.

This proves that shortening the visually terminal store block does not shorten
the CTA critical path when the added control/address work and VMEM issue occupy
the final useful-compute window.

### Consumer-Assisted Resident Publication

The next structural candidate removed the producer-side serialized
`K/V -> ResidentUsed -> Q/dO` startup chain.  Consumers published and latched
their own K/V while the two producers published independent M64 Q/dO pages;
sidecar reused the released K/V LDS region.  It preserved exact four-GEMM work,
direct FP32 stores and seven ABarrier IDs.

- Static gates passed at exactly 128KB LDS with SGPR55/VGPR112, role use
  `8/152/15/152`, and no private/spill/scratch.
- H1/S128 and all six interleaved H1/S1024 control/candidate runs passed
  correctness with exact `MMOP=73728`, `LDS=45200`, `VMEM=2560`, `FLAT=1096`,
  SpTaData/Addr `16384/4384` and `ldsBankConflict=0`.
- Paired S1024 tick deltas were `+0.0366%`, `+0.5723%`, and `+0.1675%`; paired
  median was a `+0.1675%` regression.  Median MMAC active fell
  `38.5302% -> 38.3739%` (`-0.1562pp`).
- Median ABarrier stall fell about `3.5%`, but `waitVb` rose about `8.7%`,
  `waitVm` rose about `3.5%`, SCA increased `38048 -> 38624`, and VALU
  increased `60752 -> 60824`.  Coissue was effectively flat.

This is the decisive counterexample to treating fewer barrier cycles as an
end in itself: the candidate moved readiness debt from ABarrier into BPS/VMEM
and added scalar setup, so the terminal critical path did not shrink.  The
experiment is preserved by commits `3c0b5b2` and `4502a29`; current source is
restored.

## Why Matrix Store Is Not The Fix

Canonical dK/dV outputs are FP32.  The currently verified Shaobo
`ds_write_matrix + matrix_store` path is a B16-oriented producer/consumer
handoff.  Converting FP32 accumulators to B16 would change the output contract;
an FP32 writer probe currently hits PMD/compiler encoding uncertainty.  Even a
working FP32 writer would add LDS traffic, a handoff and another readiness
boundary.  FWD also uses direct vector global stores for its output, so direct
stores are not intrinsically contrary to the FWD-style design.

## Next Action

1. Retain the canonical direct `global_store_dwordx4` epilogue on e0f10535.
2. Treat sub-percent aggregate store-latency differences as PMD run variance
   unless paired fullperf plus same-wave SQTT both reproduce them.
3. Optimize the work immediately before store: reduce C0/C1 completion skew
   without moving work into BPS/VMEM readiness or adding tokens, output
   traffic, LDS footprint, or branch predicates across WDRA/MMAC regions.
4. Keep store-tail experiments isolated.  Promotion still requires repeated
   same-build ticks, exact dynamic work, correctness, bank0 and no spill.
5. Do not retry consumer-assisted K/V publication in this form.  The next
   admissible tail hypothesis must preserve the canonical 67,072-byte LDS
   layout and BPS ownership, then use SQTT to target the specific C0/C1
   pre-store completion skew rather than the visually attributed store block.
