# MMAC C to DS Matrix Writer Source ABI

Date: 2026-07-22

Status: `ACCEPT_DENSE_D32_D128_NATIVE_DS_CHAIN / REAL_FA_PENDING`

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

The first dense runs were invalid: writer offset was corrected, but the reader
still added byte offset 16 after its pointer already named the page, and the dQ
oracle mixed N-half and D-half K fragments. Their mismatch counts are retained
in git history only and must not classify the ISA.

After fixing both controls, the score-only chain passes:

- `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260722_235908`
- trans tensor, normal tensor, dQ and dK: all `max_abs=0`, mismatch `0`.

The final dS chain computes score and dP with the same FP16-output MMAC ABI,
performs dS VALU in the native source slots, publishes once, then consumes the
same LDS page through trans dQ and normal dK views:

- `/zys/sb/fa3b/layout_probes/dq_native_ds_dense_20260723_000722`

| Check | Result |
|---|---:|
| score / dP / dS CPU oracle | all exact |
| trans reader tensor | 0 mismatch |
| normal reader tensor | 0 mismatch |
| dQ downstream MMAC | 0 mismatch |
| dK downstream MMAC | 0 mismatch |
| SGPR / VGPR | 44 / 68 |
| private / spill / bank conflict | 0 / 0 / 0 |
| scalar matrix read / permute | 0 / 0 |

The exact native tuple is:

```text
FP16 MMAC lit0/lts0 source slots
  -> ds_write_matrix_format_f16(t1, alt0, offset0)
  -> trans-m32-alt0 for dQ
  -> normal-m32-alt0 for dK
```

The D128 streaming replay passes the same contract without retaining all
operand slices: `/zys/sb/fa3b/layout_probes/`
`dq_native_ds_dense_20260723_002608`. Score/dP/dS, both reader tensors and
dQ/dK are exact; FP16 accumulation versus FP32 is `max_abs=0, rel_l2=0`.
Metadata is SGPR40/VGPR53 with private/spill/bank0. The native D128 handoff
gate is open; real FA softmax/causal and five-output correctness remain.
