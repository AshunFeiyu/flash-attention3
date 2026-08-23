# Matrix Store Component Attribution

## Question

For the failing Shaobo B16 matrix-store path, which layer is most likely at
fault: the FA kernel, builtin/code generation, PMD, or the instruction
contract itself?

The minimal contract is:

```text
aligned A-global[32x16]
  -> matrix_load_32x16_b16 BPS
  -> matching swizzled LDS page
  -> matrix_store_32x16_b16 LDS-source
  -> aligned A1-global[32x16]
```

Expected: all 512 unique FP16 bit patterns are preserved. The probe uses one
full wave, all-one EXEC, descriptor stride 16 elements, byte offset 0, no
WDRA, no MMAC, no FA formula, and no register-layout conversion.

## A0-A6 Evidence

| Gate | Evidence | Result |
| --- | --- | --- |
| A0 contract | Matching 32x16 B16 load/store, aligned buffers, stride16, offset0, 512 unique finite tags | PASS |
| A1 compile/ISA | LDS-source store emitted; SGPR23/VGPR5, private/spill/scratch0, no trap | PASS |
| A2 transport | Rows0..16 correct; rows17..31 remain poison: 240/512 mismatches | FAIL |
| A3 dense oracle | Direct bitwise oracle fails before any downstream MMAC | BLOCKED BY A2 |
| A4 lifecycle | `vbcnt0`, `lgkmcnt0`, compiler-emitted `vmcnt0`; ABarrier and cache-policy sweeps unchanged | FAIL NOT EXPLAINED BY LIFECYCLE |
| A5 operator consumer | Canonical dKV epilogue must not use matrix-store while A2 fails | REJECT |
| A6 performance | Not admissible because the transport contract is incorrect | DEFER |

## Isolation Matrix

### Compiler And PMD Cross Product

Four compiler revisions and two usable PMD revisions give the same result:

| LLVM | PMD | Direct mismatch | First failure | Writer-chain mismatch |
| --- | --- | ---: | --- | ---: |
| 7b796991 | HEAD1668 | 240 | row17/col0 | 503 |
| 7b796991 | HEAD1694 | 240 | row17/col0 | 503 |
| 47a7d59a | HEAD1668 | 240 | row17/col0 | 503 |
| 47a7d59a | HEAD1694 | 240 | row17/col0 | 503 |
| e0f10535 | HEAD1668 | 240 | row17/col0 | 503 |
| e0f10535 | HEAD1694 | 240 | row17/col0 | 503 |
| a2724117 | HEAD1668 | 240 | row17/col0 | 503 |
| a2724117 | HEAD1694 | 240 | row17/col0 | 503 |

The relevant generated instruction sequence is equivalent across the four
compiler snapshots. This strongly disfavors a single compiler regression and
a HEAD1694-only PMD regression.

### Completion And Cache Policy

The following LDS-source modes all produce the identical 240 mismatches and
first failure at row17/col0:

- compiler-emitted `vmcnt(0)` plus an explicit VM/LGKM drain;
- documented ABarrier `seq -> store -> arrive -> wait` lifecycle;
- `glc`, `slc`, and `glc+slc`;
- `buffer_wbinvl1_vol` after the completed store.

IT trace proves full EXEC and this order:

```text
matrix_load_32x16_b16 ... lds:1 bps
s_waitcnt_vbcnt 0
s_waitcnt lgkmcnt(0)
matrix_store_32x16_b16 ... lds:1
s_waitcnt vmcnt(0)              # inserted by compiler
s_waitcnt vmcnt(0) & lgkmcnt(0) # probe drain
```

The `_rtn` builtin is not an LDS-source completion-return variant. It lowers
to the VGPR-source form `matrix_store ... v[1:4] ...`, so it is not a valid
control for LDS-source completion.

### Shape Family

Matching direct load/store controls on PMD HEAD1694 produce:

| Shape | Expected elements | Mismatches | Poison left | First failure |
| --- | ---: | ---: | ---: | --- |
| 32x16 | 512 | 240 | 240 | row17/col0 |
| 64x16 | 1024 | 1024 | 1024 | row0/col0 |
| 32x32 | 1024 | 1024 | 1024 | row0/col0 |

All three runs exit normally without a PMD panic. The instruction family is
modeled far enough to decode and execute, but its LDS-source global-write
behavior is incomplete under this documented contract.

### Writer Isolation

The earlier writer-only probe writes the complete LDS page through each
`ds_write_matrix_format_f16` mode, scans it with ordinary LDS reads, and
preserves all 512 unique values. The current loss boundary is therefore not
the FP16 matrix writer itself; it is at matrix-store/descriptor/model.

### Compiler Surface Gap

All four compiler snapshots reject `s_waitcnt_vwcnt 0` for gfx946 with
`instruction not supported on this GPU`. This is a separate compiler/ISA
exposure gap. It does not explain the direct failure because the LDS-source
matrix store is followed by compiler-generated `s_waitcnt vmcnt(0)` and the IT
trace reaches that wait normally.

## Attribution

| Candidate | Verdict | Reason |
| --- | --- | --- |
| FA algorithm or fused kernel | Ruled out for this failure | One-wave direct roundtrip has no FA math, roles, WDRA, MMAC, or output ownership. |
| Probe/source usage | Strongly disfavored | Matching documented shapes, aligned descriptors, full EXEC, offset0, lifecycle and cache sweeps all reproduce. An undocumented model-only ABI cannot be excluded. |
| Compiler code generation | Unlikely primary cause | Four revisions emit equivalent relevant ISA and fail identically on both PMDs. Compiler still has a separate `vwcnt` exposure issue. |
| PMD | Primary suspect, high confidence | Two PMD revisions share the exact 32x16 truncation; HEAD1694 silently drops all 64x16/32x32 writes. |
| Hardware instruction | Unverified | No Shaobo silicon or independent RTL/reference emulator is available. PMD behavior is not proof of hardware behavior. |

Current status:

```text
PROBABLE PMD COMMON IMPLEMENTATION DEFECT
OR REMOTE UNDOCUMENTED MODEL ABI REQUIREMENT
```

Do not report this as a confirmed hardware instruction bug. The decisive next
control is the same binary on silicon or an independent RTL/reference model,
or a PMD-owner statement of the required descriptor/source-layout ABI.

## Evidence Paths

- Compiler/PMD matrix:
  `/zys/sb/runs/matrix_store_compiler_pmd_matrix_20260823/summary.csv`
- Completion/cache sweep:
  `/zys/sb/runs/matrix_store_completion_policy_head1694_20260823/`
- IT trace:
  `/zys/sb/runs/matrix_store_completion_it_head1694_20260823/`
- Shape family:
  `/zys/sb/runs/matrix_store_shape_family_head1694_20260823/`
- `vwcnt` compiler gate:
  `/zys/sb/runs/matrix_store_vwcnt_compiler_gate_20260823/`
- Earlier exhaustive direct/full-chain result:
  `results/matrix_global_roundtrip_20260722.md`

## Owner Questions

1. Does PMD support the LDS-source forms of B16 `matrix_store_32x16`,
   `matrix_store_64x16`, and `matrix_store_32x32` on `GPU_CHIP=sb`?
2. If yes, what exact M# descriptor fields, source swizzle, stride unit, LDS
   base alignment, wave count, and completion counter are required?
3. Why does the 32x16 form commit exactly 17 rows while the other tested forms
   commit none?
4. Is the compiler expected to expose `s_waitcnt_vwcnt` on gfx946, and is
   LDS-source matrix-store completion governed by VM or VW in the intended
   compiler/PMD pair?
