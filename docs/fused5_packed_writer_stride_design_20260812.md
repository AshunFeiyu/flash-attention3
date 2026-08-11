# Fused5 Packed Writer-Stride Probe Design

Date: 2026-08-12

Status: `A0_A4_PASS / OPERATOR_INTEGRATION_PENDING`.

## Trigger

The accepted fused kernel spends 32KB on raw Q/dO and 64KB on four physical
dS panels after resident K/V are latched. A second raw page would remove the
dominant single-page ownership chain, but the 128KB LDS limit leaves no room.

The locked writer-footprint probe already proves that
`ds_write_matrix_32x16_trans_f16` touches exactly `[base, base+1024)` and leaves
`[base+1024, base+2048)` unchanged. Production nevertheless spaces each writer
by 2048 bytes. This probe asks whether the reader ABI also permits a 1024-byte
writer stride.

## A0 Contract

Two writer waves publish different deterministic fragments to:

```text
packed pages:  base 0,    1024
control pages: base 4096, 6144
```

After a CTA barrier, two reader waves consume corresponding packed and control
pages through both native views:

```text
ds_read_matrix_32x16_normal
ds_read_matrix_32x16_trans
```

Admission requires:

1. every normal fragment bit equals its 2KB-stride control;
2. every trans fragment bit equals its 2KB-stride control;
3. one dense MMAC result from packed fragments equals the control MMAC result;
4. no private/spill/scratch and no PMD panic.

The probe compares packed against a known-safe layout in the same dispatch, so
it does not assume a lane-linear logical order.

## Resource Consequence If Admitted

With a 1024-byte writer stride:

```text
8 dS writers * 1024B * 4 M16 panels = 32KB
8 P writers  * 1024B                 =  8KB
```

The fused steady-state budget becomes:

| Region | Current | Packed |
|---|---:|---:|
| raw Q/dO page0 | 32KB | 32KB |
| resident K/V startup | 64KB | 64KB |
| batch dS after latch | 64KB | 32KB |
| P scratch | 16KB | 8KB |
| sidecar | 0.75KB | 0.75KB |

The upper 32KB of released K/V LDS can then hold raw Q/dO page1 without
increasing the 128KB startup allocation. Operator integration is forbidden
until this focused A0-A3 probe passes.

## Measured Probe Result

The same-dispatch packed/control probe passes on compiler `e0f10535` and PMD
HEAD1694:

```text
writer0 normal=0 trans=0 mmac=0 mismatches
writer1 normal=0 trans=0 mmac=0 mismatches
metadata SGPR18 / VGPR26 / private0 / spill0 / scratch0
```

Generated ASM contains two native writers, two normal readers, two trans
readers and two MMAC instructions. This admits 1024-byte writer stride through
A4. The next step is one operator-consumer integration that changes only P/dS
page stride and LDS accounting; double buffering remains a later hypothesis.
