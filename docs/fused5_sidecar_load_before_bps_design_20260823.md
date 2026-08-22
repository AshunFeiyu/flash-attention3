# Fused5 Producer Sidecar Load Before BPS

## Evidence

The accepted G1-first fullperf exposes wait-VM while reducing ownership debt:

- wait-VM share: `2.845% -> 3.325%`.
- XCU global issue gap `global_load_dwordx3 -> s_waitcnt`: 280,476 cycles,
  3.91% of the dispatch issue-gap ledger, max 1,471 cycles.
- The source maps to the producer's three-float sidecar row load.
- Producer uses 9 of its 16 WDRA VGPR budget, so a three-float packet is
  feasible without changing the role allocation.

The current order is:

```text
Q/dO BPS -> sidecar global_loadx3 -> wait_vmem_lgkm -> RawFilled
```

## Hypothesis

Split the existing sidecar copy into issue and publish helpers:

```text
sidecar global_loadx3 -> Q/dO BPS -> wait_vmem_lgkm
-> sidecar LDS stores -> BPS vbcnt wait -> RawFilled
```

The global packet can age under the unchanged Q/dO matrix loads. This adds no
global/LDS transaction, ownership event, barrier, formula work or output.

## Invariants

- Exact five logical GEMMs and unchanged M64/N128/D128 tile.
- Unchanged 16-wave roles, 12 ABarrier IDs and 128KiB LDS layout.
- Same Q/dO BPS and sidecar global/LDS transactions.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- No private segment, spill, scratch or LDS bank conflict.
- Producer role must remain within 16 VGPR; other role budgets stay exact.

## Expected ISA

The static gate must prove a real schedule:

```text
global_load_dwordx3
matrix_load Q/dO packet
s_waitcnt vmcnt/lgkm
sidecar ds_write
```

If the compiler inserts a VM wait before the first matrix load because of the
inline-assembly memory boundary, reject before PMD. Do not add scheduling
NOPs, a new token, or extra buffering to force the experiment.

## Admission

1. Build and inspect generated ASM before runtime.
2. Preserve MMAC/read/wait/ABarrier and global/LDS dynamic work; no spill.
3. Full golden correctness: S128 causal/noncausal and S1024 causal; bank0.
4. Three interleaved S1024 pairs against tag
   `best/fused5-writer-g1-first-20260823` decide ticks.
5. A winner must lower paired ticks and the XCU
   `global_load_dwordx3 -> s_waitcnt` gap without growing net ABarrier debt.

This is workbook Section 55.

## Result

Status: `REJECT_STATIC_CONTROL_COST_AND_TICKS_CANONICAL_RESTORED`.

The generated ISA does place `global_load_dwordx3` before the Q/dO BPS
packet, so the scheduling premise is real. It also grows static waits
`340 -> 342`, scalar branches `46 -> 49`, `v_mov_b64` `68 -> 72`, and the
producer role from 9 to 12 VGPR. MMAC1472, matrix reads854 file-wide,
ABarrier102, SGPR82/VGPR128 and private/spill/scratch0 remain clean.

Full cached CPU-golden correctness passes S128 causal/noncausal and S1024
causal with warning0 and bank0. Three interleaved S1024 pairs measure fused
means `43,129,905 -> 43,610,992`, a `+1.115%` regression. The extra
cross-BPS packet lifetime therefore costs more control/register movement than
the VMEM aging saves. No S2048 or candidate fullperf is admitted, and the
canonical source is restored to `58e90fc` behavior.

Evidence:
`/zys/sb/runs/fused5_sidecar_before_bps_ab_20260823`, local A/B scripts under
`outputs/019ea61f-c117-76b2-abad-e776092d47a0/fused5_sidecar_before_bps_ab_20260823`,
and workbook section55.
