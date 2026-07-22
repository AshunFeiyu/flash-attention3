# MMAC C to DS Matrix Writer Source ABI

Date: 2026-07-22

Status: `ACCEPT_F16_C_NATIVE_SOURCE_MATCH / F32_TO_DS_SEMANTIC_PENDING`

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

## FP16-Output MMAC Follow-up

The same probe was extended with the native HCU FP16-output MMAC form:

`__builtin_hcu_mmac_16x16x16_f16_lit_lts`

PMD run:

`/zys/sb/fa3b/layout_probes/dq_source_slot_20260722_225346`

The exhaustive source-slot audit found eight exact native combinations. The
minimal distinct contracts are:

- `qT_kT_lit0_lts0 -> trans writer -> trans_m32_alt0 reader`
- `qT_kT_lit0_lts1 -> trans writer -> normal_m32_alt0 reader`

Both contracts pass for adjacent-N and adjacent-M fragment pairs with
`mismatch=0/512`. Writer `alt0` and `alt1` have the same register source ABI,
so each distinct contract appears twice.

Static/resource evidence:

- `SGPR=31`, `VGPR=85`
- private segment `0`
- SGPR/VGPR spill `0/0`
- `ldsBankConflict=0`
- `v_mmac_16x16x16_f16=384`
- ordinary `ds_read_b*=0`, permute/permlane `0/0`

## Revised Interpretation

The native chain exists. The cleanest proven register-layout contract is:

```text
FP16-output MMAC(qT, kT, lit=0, lts=0)
  -> ds_write_matrix_format_f16(trans)
  -> ds_read_matrix_trans_format_f16(m32x16, alt0)
```

This closes the instruction-layout existence question, but it does not yet
close the FA dS handoff. Production score and dP normally use FP32
accumulation, followed by FP32 softmax/dS arithmetic and an FP16 conversion.
The previously tested lane-local conversion from FP32 C does not have this
writer source ABI. The next focused semantic test must therefore prove one of
these without runtime permutation:

1. the final dS arithmetic can remain in the FP16-output MMAC source slots; or
2. a native instruction mode converts the FP32 dS ownership into that ABI.

Do not integrate the FP16 MMAC path into the canonical FA kernel until its
numeric error and downstream MMAC semantics pass a dense CPU oracle.
