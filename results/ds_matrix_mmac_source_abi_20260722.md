# MMAC C to DS Matrix Writer Source ABI

Date: 2026-07-22

Status: `OBSERVE_COORDINATE_MATCH / REJECT_DENSE_DUAL_CONSUMER_CHAIN`

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

## Coordinate-Level Interpretation

The coordinate-label probe found this candidate register-layout contract:

```text
FP16-output MMAC(qT, kT, lit=0, lts=0)
  -> ds_write_matrix_format_f16(trans)
  -> ds_read_matrix_trans_format_f16(m32x16, alt0)
```

This is only a candidate because the coordinate input is separable and sparse:
only two D positions encode the Q and K row labels. It can hide a D-fragment
or ownership permutation. Production score and dP also require FP32
accumulation followed by FP32 softmax/dS arithmetic and FP16 conversion.

## Dense Dual-Consumer Stress

The non-symmetric dense probe was extended to publish a real FP16-output MMAC
fragment once, then read the same LDS page through the trans dQ view and the
normal dK view. It also reconstructs both reader tensors on the host before
the downstream MMAC, so the failure can be localized before output stores.

Valid runs use writer byte offset zero:

- LTS0: `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_232115`
- LTS1: `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_232230`
- FP32-downcast control:
  `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_232322`

Results:

| Published source | Trans tensor | Normal tensor | dQ | dK |
|---|---:|---:|---:|---:|
| FP16 MMAC LTS0 | 496 mismatches | 496 | 493 | 911 |
| FP16 MMAC LTS1 | 488 mismatches | 488 | 488 | 882 |
| FP32 dS downcast | 488 mismatches | 488 | 462 | 887 |

All runs pass transport and resource gates. The native FP16 cases use
`SGPR44/VGPR62`, private/spill0, `ldsBankConflict=0`, ordinary matrix-path DS
read0 and permutation0. Their asm contains
`ds_write_matrix_format ... element:2 row:2 col:1 t` with no `offset:16`.

The dense failure supersedes the coordinate-only promotion. The matrix writer
and readers remain proven lossless permutations, but neither FP16 MMAC LTS
candidate produces a general dense source fragment that serves both required
views. The five-GEMM gate remains closed on this toolchain.

The next native solution must provide one of these without runtime
permutation:

1. the final dS arithmetic can remain in the FP16-output MMAC source slots; or
2. a native instruction mode converts the FP32 dS ownership into that ABI.

Do not integrate the FP16 MMAC path into the canonical FA kernel from a sparse
coordinate probe. A dense, non-symmetric CPU oracle is the promotion gate.
