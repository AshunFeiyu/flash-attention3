# Explicit Vec4 dS Packing

## Hypothesis

Represent the four useful probability values as `Vec4F32`, convert probability
and dS to FP16 with one vector conversion each, and evaluate
`dS = P * (dP - D) * scale` as an explicit vector expression. Keep causal
predicates, source slots, upper-half zeroing, MMAC, LDS requests, waits,
ABarriers and output ownership unchanged.

## Static Result

The generated ISA rejects the premise:

- total semantic instructions: `6,532 -> 6,644`
- `v_mov_b32_e32`: `174 -> 212`
- `v_mov_b64_e32`: `42 -> 148`
- `v_pk_mul_f32`: `120 -> 160`
- `v_pk_add_f32`: `40 -> 0`
- waits, matrix reads and MMAC remain exact
- SGPR count: `82 -> 86`
- role use: `9/171/87/164 -> 9/169/87/165`
- private segment, scratch and spills remain zero

The accepted scalar source is already packed more effectively by the compiler.
Explicit vector temporaries extend live ranges and create substantially more
register moves. This fails the stated ISA objective before PMD correctness or
performance work is admitted.

## Decision

`REJECT_STATIC_CODEGEN_EXPANSION_CANONICAL_RESTORED`.

Close dS source-expression micro-tuning on this compiler. The next hypothesis
must come from whole-kernel ownership or issue-gap analysis rather than source
syntax. Workbook section 49 contains the formula, slot and resource proof;
candidate build evidence is under
`/zys/sb/experiments/fused5_explicit_vec4_ds_20260823`.
