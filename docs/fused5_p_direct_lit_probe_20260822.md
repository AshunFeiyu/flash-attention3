# Fused5 P-to-dV Native LIT ABI Probe

Status: `REJECT_NATIVE_DIRECT_P_ABI / CANONICAL_UNCHANGED`.

## Question

Can a different native FP16 score-MMAC output mode remove the canonical
`P -> ds_write_matrix -> ds_read_matrix -> dV MMAC` ownership bridge without
ordinary LDS reads, gather or permutation?

## Evidence

The prior LTS probe is a negative control. On the locked compiler and PMD,
FP32 `lit1/lts1` produces the same dump as `lit1/lts0`; it is not a transpose
or ownership switch:

`/zys/sb/fa3b/layout_probes/lts_pr1_20260822`

The exhaustive coordinate probe was rerun at:

`/zys/sb/fa3b/layout_probes/dq_source_slot_20260822_201008`

Among the 16 FP16-output operand/LIT/LTS modes, the only N-pair mode with the
canonical logical score coordinates is `qT_kT_lit1_lts0` (mode 36). The sweep
ends with `any_direct_read_pass=0`.

The dense downstream differential is:

`/zys/sb/fa3b/layout_probes/p_direct_lit_final_20260822`

It compares the only logical candidate against the canonical bridge before
the same dV MMAC:

```text
score1_vs_bridge      7.42603
score0_vs_score1      6.99805
candidate_vs_control 15.697
pass                  0
```

Static gates pass: MLS3, matrix writer1, normal reader2, trans reader2,
FP16-output MMAC4, downstream FP32 MMAC2, scalar DS read0, permute0,
SGPR28/VGPR21, private/spill0 and `ldsBankConflict=0`.

## Decision

The failure is a native register-ownership mismatch, not a PMD panic or an
operand-readiness bug. Keep the canonical bridge. Do not retry LTS, lit1, an
ordinary `ds_read_b32` gather, `bpermute`, or a wrong-layout direct path.

The next admitted optimization must preserve fragment semantics and overlap a
real future operand read with existing useful MMAC work.
