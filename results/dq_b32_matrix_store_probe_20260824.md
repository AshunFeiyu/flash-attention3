# dQ FP32 B32 Matrix-Store Probe

Status: `REJECT_DIRECT_CANONICAL_LAYOUT`.

## Question

Can the canonical dQ FP32 accumulator replace lane-wise global stores with a
precision-preserving native chain?

```text
FP32 accumulator bits
  -> ds_write_matrix_format_u32
  -> LDS
  -> matrix_store_16x16_b32
  -> FP32 global output
```

This route does not downcast dQ. `u32` is used only as a bit-preserving
transport because PMD does not decode the native f32 DS writer.

## Static Evidence

- Compiler: locked `e0f10535` toolchain.
- PMD: `CoreArch:HEAD_1694`.
- Symbol resources: SGPR13, VGPR5, private/spill/scratch0.
- ISA: two B32 DS-writer forms and four `matrix_store_16x16_b32` forms.
- Forbidden fallback: scalar matrix DS read0, permute0.

## Result

The B32 writer and global matrix-store instructions execute with
`ldsBankConflict=0`, but none of the eight writer/store T/R combinations is an
exact logical 16x16 identity:

| DS writer | Best mismatch count | Store T/R sensitivity |
|---|---:|---|
| `t=0` | 252/256 | none |
| `t=1` | 192/256 | none |

The best output has the same fixed 4x4 component transpose on every row:

```text
0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
```

Therefore the current dQ accumulator ownership is not the native B32 writer
source ABI. Direct integration would require a register rearrangement, unless
a different native MMAC output mode is proved to generate this source layout.

## Performance Boundary

`matrix_store_16x16_b32` is the only B32 global matrix-store shape exposed by
the locked compiler. The current M64xD32 dQ writer owns eight 16x16 fragments,
so it still needs eight matrix-store instructions, the same fragment count as
the existing vector stores, while adding DS writes, LDS ownership and store
completion waits. The canonical LDS allocation also has only about 6.5 KiB
free during its released-K/V steady state, not the 32 KiB needed for a private
M64xD128 FP32 dQ page.

Do not add a permute/gather workaround and do not lower dQ to FP16. Keep the
accepted direct FP32 global-store path until a native MMAC-output-layout probe
passes and an LDS lifetime budget shows net value.

## Reproduction

- Source: `probes/dq_b32_matrix_store_probe.cpp`
- Runner: `scripts/run_dq_b32_matrix_store_probe.sh`
- Remote run:
  `/zys/sb/dqb32_head1694_map/dq_b32_matrix_store_20260824_225346`

The runner intentionally returns failure when no exact combination exists;
the negative result is the oracle, not a PMD crash.
