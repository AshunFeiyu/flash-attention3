# Fused5 M128 Packed VALU And Prelude Result

Date: 2026-08-12

Classification: `OBSERVE_ISA_WIN_TICKS_REGRESSION`, then
`REJECT_TICKS_REGRESSION / CANONICAL_RESTORED`.

## Packed VALU Recovery

Explicit two-float vectors recover the intended packed softmax/dS codegen:

- scalar `v_mul_f32_e32`: `194 -> 2`
- `v_fma_mix_f32`: `64 -> 0`
- scalar `v_sub_f32_e32`: `80 -> 0`
- `v_pk_mul_f32`: `32 -> 130`
- `v_pk_fma_f32`: `0 -> 32`

The candidate passes H1/S128 causal/noncausal correctness, exact MMOP92,160,
bank0, SGPR64/VGPR124 and private/spill/scratch0. Dynamic VALU falls below
the M64 baseline (`120,800 -> 118,032`). However, repeated H1/S1024 fused
ticks average 72,462,617.5, 0.575% slower than the accepted M64 mean. MMAC
active averages 22.211146%, barrier share 28.720295%, and waitLgkm 7.691578%.

This admits the expression/codegen technique as evidence, not the M128
topology.

## Half1 Useful Prelude

The second experiment moves only legal half1 score/dP/probability/dS work
before the half0 `KvDsUsed` wait. It does not change formulas, MMOP, pages,
barrier counts or output ownership.

Static role use is `8/169/169/84`; H1/S128 causal/noncausal correctness,
no-spill resources and bank0 pass. H1/S1024 runs are 72,787,715 and
72,917,845 ticks, averaging 72,852,780. MMAC active averages 22.063894%,
waitLgkm 7.136310%, and barrier share 29.019252%.

The useful prelude is 0.539% slower than packed M128 and 1.117% slower than
accepted M64. The mandatory half-page P/dS ownership handoff remains exposed;
larger live state does not turn it into useful overlap.

## Evidence And Decision

- Packed runs: `/zys/shaobo_runs/fused5_m128_packed_valu_20260812`
- Prelude runs: `/zys/shaobo_runs/fused5_m128_half1_prelude_20260812`
- Prelude ASM: `/zys/shaobo_runs/fused5_m128_half1_prelude_20260812/asm_audit/fused_bwd_kernel.asm`
- ASM SHA256: `94be2fd393a481b75811d0fd59fa37d8e70dcb9e1178aa77e99e63fcf4f2b828`
- Compiler e0f10535, PMD HEAD1694, `GPU_CHIP=sb`, SQ7.

Close M128 on this toolchain after two same-tier failures. Restore
`b28e73d`; the next experiment must change a higher-share operator boundary
or ownership topology rather than move another wait or half-tile schedule.
