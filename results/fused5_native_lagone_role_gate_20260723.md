# Fused5 Native Lag-One 16-Wave Role Gate

Status: `ACCEPT_FOCUSED_GATE / PRODUCTION_INTEGRATION_PENDING`

## Contract

- Four WDRA roles:
  - waves0-3: 32-VGPR loader role
  - waves4-7: 176-VGPR dKV group0
  - waves8-11: 176-VGPR dKV group1
  - waves12-15: 96-VGPR D32 dQ writer
- Exact useful work per M64/N128/D128 tile:
  - eight dKV waves x 128 MMAC = 1024
  - four dQ waves x 64 MMAC = 256
  - total = 1280 MMAC, exactly five logical GEMMs
- One native dS generation is reused. Consumers perform useful next-panel
  MMAC and prefetch the next dS fragment before waiting for `DsUsed`; the
  fragment remains in VGPR until the current LDS generation is reusable.
- The dS handoff stays native:
  `ds_write_matrix_f16 -> normal dK read + trans dQ read`.

## Result

- Compiler branch usage: `1 / 159 / 159 / 32` VGPR inside
  `32 / 176 / 176 / 96`; physical target is 480/512.
- Metadata: SGPR22, VGPR120, private0, SGPR spill0, VGPR spill0.
- ASM: writer2, trans-read8, normal-read2, MMAC80 static,
  four `s_set_vgpr_size`, regular `ds_read_b*`0, permute0, trap0.
- PMD dense result: dQ mismatch0, dK mismatch0, pressure mismatch0.
- Dynamic stats: simTicks15444520, MMOP1280, coissue205/230,
  `ldsBankConflict=0`.
- Evidence:
  `/zys/sb/fa3b/layout_probes/fused5_native_lagone_20260723_172647`.

## WDRA Finding

The first run reproduced a deterministic PMD panic:

`VGPR index 158 is out of range: VGPR range=[0,96]`.

ASM showed that a common `lane = threadIdx.x % 64` expression was emitted as
`v_and_b32 v158, 63, v1` before the role branch. Moving lane acquisition into
each branch removed the pre-role high-VGPR instruction; the unchanged PMD
then passed without uninitialized-VGPR warnings. This is a source/codegen
placement issue, not a total VGPR-budget failure.

## Promotion Rule

The gate proves role resources, exact work, native dS handoff and
single-generation lag-one ownership. It does not prove full FA correctness or
performance. Canonical integration must retain the current Q-latch CPU-golden
path and pass H1/S128, H1/S1024, metadata, bank0 and xcu evidence before
promotion.
