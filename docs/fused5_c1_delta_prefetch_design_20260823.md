# C1 Delta Prefetch Before Score

## Hypothesis

Move the existing C1 `row_delta` LDS read before the four score matrix reads.
Retire the old dO packet with `lgkmcnt(5)`, execute dP, then consume score and
delta after the compiler-generated full LDS wait. The intent was to remove the
late sidecar wait without adding matrix traffic, MMAC work, LDS storage,
ABarrier tokens, or output ownership.

## Static Result

- Exact work is unchanged: MMOP 92,160, matrix reads 840, LDS 63,872.
- Static `s_waitcnt` falls from 336 to 316.
- Static `v_mov_b32_e32` rises from 174 to 178.
- Role use remains `9/171/87/164`; SGPR82/VGPR128.
- Private segment, scratch, SGPR spill and VGPR spill remain zero.

## Correctness

H1/S128/D128 causal and non-causal complete lifecycle tests pass for delta,
dK, dV and dQ. PMD reports no panic, no VGPR warning and zero LDS bank
conflicts.

## Performance Result

Three interleaved H1/S1024/D128/causal pairs reject the candidate:

- fused ticks: `44,335,048 -> 44,384,795` (`+0.112%`)
- lifecycle ticks: `48,515,892 -> 48,613,868` (`+0.202%`)
- MMAC active: `34.8523% -> 34.7050%`
- wait-LGKM: `7.8593% -> 8.1336%`
- barrier: `13.7303% -> 13.6466%`
- coissue success: `20,433 -> 21,719`
- dynamic VALU: `119,744 -> 119,808`

Fewer static waits and more successful coissue do not shorten the critical
path. The extra outstanding sidecar read increases LDS readiness pressure and
the compiler's `lgkmcnt(5)` schedule raises dynamic wait share. This is a
measured scheduling rejection, not a correctness or resource failure.

## Decision

`REJECT_OUTSTANDING_LDS_PRESSURE_CANONICAL_RESTORED`.

Do not preload C1 row delta ahead of the score packet on this topology. The
canonical late sidecar read plus dV-under-dS schedule remains the accepted
path. Evidence is under `/zys/sb/fa3b/c1_delta_prefetch_ab` and workbook
section 48.
