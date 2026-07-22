# MMAC C to DS Matrix Writer Source ABI

Date: 2026-07-22

Status: `OBSERVE_NO_F32_C_NATIVE_SOURCE_MATCH`

## Question

Can a real FP32 MMAC C fragment be downcast lane-locally to FP16 and passed
directly to `ds_write_matrix_format_f16`, so a later matrix reader sees the
canonical dS operand layout without a runtime permute or gather?

## Method

- Reused `dq_source_slot_coordinate_probe.cpp`; no canonical FA kernel change.
- Replaced the old incomplete 504/512 analytical slot map with the exact
  512-slot permutations measured by `ds_matrix_reg_roundtrip_probe`.
- Verified the six bit-permutation formulas against all 6144 rows in
  `ds_matrix_slot_map.csv`: formula errors `0`.
- Tested 16 MMAC modes (`Q/K` normal or transpose reader x `LIT/LTS`) for both:
  - N-pair: two adjacent score fragments along `seqlen_k`.
  - M-pair: two adjacent score fragments along `seqlen_q`.
- Compared each natural MMAC C coordinate map against four writer modes and
  three measured reader modes: 384 native combinations, no runtime transform.

## Evidence

PMD run:

`/zys/sb/fa3b/layout_probes/dq_source_slot_20260722_224840`

Toolchain:

- compiler LLVM `e0f10535a0d681bcf3885ea2c398cc494bf6e332`
- PMD `HEAD_1694(lib_ini_opt)`
- `GPU_CHIP=sb`

Static/resource gate:

- `SGPR=31`, `VGPR=79`
- private segment `0`
- SGPR/VGPR spill `0/0`
- ordinary `ds_read_b*` `0`
- permute/permlane `0`
- `ldsBankConflict=0`

Semantic result:

- N-pair canonical `qT_kT_lit1_lts0`: identity errors `0`.
- M-pair canonical `qT_kT_lit1_lts0`: identity errors `0`.
- N-pair best writer/reader mismatch: `384/512`.
- M-pair best writer/reader mismatch: `384/512`.
- `any_exact_source_slot_pass=0`.

## Interpretation

The matrix writer transport is not losing data: the prior writer dump and
roundtrip probes proved complete, reversible 512-element transport. The new
result narrows the remaining gap to the producer register ABI:

```text
FP32 MMAC C
  -> lane-local FP16 downcast
  != ds_write_matrix_format_f16 producer source slots
```

This does not contradict the compiler guidance that matrix writer output and
MLS output share the Shaobo LDS swizzle. Unified LDS swizzle does not imply
that an FP32 MMAC C fragment, after a lane-local cast, already has the writer's
input register ownership.

## Next Native Test

Test the HCU FP16-output MMAC form
`__builtin_hcu_mmac_16x16x16_f16_lit_lts`. Its C/D register ABI may pair with
the FP16 matrix writer even though FP32-C plus lane-local downcast does not.
This remains a focused instruction probe; do not add runtime permute/gather to
the FA main path.
