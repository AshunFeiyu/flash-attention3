# Fused5 M128 Double-Raw and Dead-dO dS Reuse Design Gate

Status: `REJECT_DESIGN_SIDE_CAR_AND_DS_LIFETIME`

## Hypothesis

Increase the fused five-GEMM ownership epoch from `M64` to `M128`, retain two
raw `Q/dO` generations after resident `K/V` have been latched, and reuse the
dead `dO` half of the current raw page for the native dS writer. The intended
benefit is to double each useful GEMM island from 256 to 512 MMAC instructions
and amortize `RawFilled/RawUsed` control over twice the arithmetic.

No production source is changed until the formula, LDS, VGPR and ABarrier
ledgers all pass.

## Formula and Useful Work

The exact DAG remains:

```text
score = Q @ K^T
dP    = dO @ V^T
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

For `M=N=D=128` and a `16x16x16` MMAC, each logical GEMM contains
`8*8*8 = 512` MMAC instructions. The five-GEMM epoch contains 2560 useful
MMAC instructions and introduces no duplicate score or dP.

## LDS Pressure Test

```text
startup resident K/V:       32 + 32 = 64 KiB
startup raw Q/dO page 0:    32 + 32 = 64 KiB
startup total:                         128 KiB

steady raw Q/dO page 0:                64 KiB
steady raw Q/dO page 1:                64 KiB
steady total:                         128 KiB
```

The apparent fit is incomplete:

- Logical FP16 dS is 32 KiB for `M128xN128`.
- The verified Shaobo matrix-writer ABI occupies twice the logical dS bytes,
  so a full M128 dS publication needs 64 KiB.
- The dead dO half of one raw page releases only 32 KiB.
- Publishing one M64 half at a time fits, but restores the existing M64 dS
  ownership cadence and therefore does not amortize the critical handshake.
- Sidecar requires `3*128*4 = 1536` bytes per raw generation. Two exact 64-KiB
  raw pages leave no LDS capacity for it.

## Rejected Escape Routes

1. Let consumers load sidecar directly from global memory.

   This restores duplicate VMEM requests across eight heavy waves and the
   global-load wait already removed by the accepted sidecar-LDS design.

2. Write sidecar into a freed old raw page, make consumers latch it, then
   overwrite that page with the next raw matrix.

   This adds a sidecar-filled/latch sub-phase to every raw generation. Reusing
   `RawFilled/RawUsed` doubles their cadence; adding a token family adds the
   same control debt under another name. Both erase the M128 epoch advantage.

3. Serialize the two heavy consumer groups so one full-group M128 dS page fits
   in the dead dO half.

   This trades ABarrier pressure for loss of the two-consumer overlap and is
   incompatible with the FWD-style SIMD/coissue target.

## Direct-register Boundary

The K/V-left isolated dKV oracle has a valid direct `dS-reg -> dK` contract.
The current fused Q-left canonical orientation does not: the production-like
probe in `docs/fused5_dk_register_ds_design_20260812.md` fails only dK while
dV and dQ remain correct. Do not reuse the oracle result outside its operand
ownership contract, and do not add permute/gather workarounds.

## Decision

`REJECT_DESIGN_SIDE_CAR_AND_DS_LIFETIME`. The arithmetic tile is attractive,
but the complete storage and ownership graph cannot preserve the intended
barrier-frequency reduction. No kernel patch, build, PMD run or perf capture is
admitted for C91. The next structural hypothesis must keep sidecar in LDS,
avoid a new token family, and reduce an existing ownership epoch rather than
renaming or subdividing it.
