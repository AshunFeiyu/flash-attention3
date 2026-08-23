# Fused5 Causal C1 First-Valid dKV Zero Seed

## New Premise

Accepted C1 zero-front removes the complete causal `qi0` mathematical body.
Therefore `qi1` is now C1's compile-time-known first valid accumulation tile.
The canonical code nevertheless clears all eight dV and eight dK FP32
accumulators at role entry.

This premise did not exist in the earlier peeled-zero-seed experiment. That
experiment retained C1 dV clears because its first valid panel lived in the
generic path. The accepted zero-front now gives one fixed call site without
adding an ownership or loop split.

## Exact Design

```text
causal C1:
  qi0: publish native zero dS; no dV/dK work
  qi1 panel0: first dV MMAC and first dK MMAC define accumulators from mmac_zero
  later panels/tiles: accumulate in place

noncausal C1:
  explicitly clear dV/dK as before
  qi0 and later tiles accumulate in place
```

No accumulator can be read before its defining first MMAC. The first dV/dK
islands cover every accumulator destination exactly once before later use.

## Draft / Stress / Revise

Draft: condition C1 explicit clears on `!causal`, set `FirstAccum=true` on the
already-peeled C1 `qi1` call, and seed only panel0 dV plus the existing first
dK update.

Stress:

- S128 causal has exactly the zero-front plus one diagonal tile; all outputs
  must still be defined.
- Noncausal must execute the original explicit initialization.
- Do not seed every panel; dV/dK accumulators span q tiles.
- Do not add a runtime `first_update` test inside the steady loop.
- Reject if compiler duplicates the full C1 panel body, grows static waits or
  branches, or exceeds the C1 WDRA window.

Revise: use the existing compile-time `FirstAccum` template only at the peeled
call site. A small panel0 compile-time branch is allowed only if generated ISA
contains no dynamic branch and keeps exact MMAC/read/wait counts.

## Resource And Pipeline Budget

| Item | Accepted | Candidate gate |
| --- | ---: | ---: |
| Tile / roles | M64/N128/D128, 16 waves | unchanged |
| MMOP S1024 causal | 88,064 | exact |
| C1 explicit zero moves | 32 `v_mov_b64` per wave | removed on causal path |
| LDS / ABarrier | 128 KiB / 12 IDs | unchanged |
| Metadata | SGPR71 / VGPR128 | no spill/private/scratch |
| Role use | 9/173/87/162 | same WDRA windows |

Expected cadence:

```text
time0: C1 zero-front publishes dS and releases existing tokens
time1: C1 qi1 score/dP/P/dS with first dV/dK MMAC seeded from zero
time2: writer consumes G1 first; later C1 tiles accumulate normally
```

## Admission

1. Static MMAC, matrix-read, wait, ABarrier and store counts remain exact.
2. `v_mov_b64` must fall without code-size/control duplication.
3. S128 causal/noncausal, S1024 causal and S2048 causal full golden pass.
4. No private/spill/scratch, PMD warning or LDS bank conflict.
5. Three S1024 pairs and two S2048 pairs decide promotion; fullperf/xcu only
   follows a repeatable ticks win.

## Static Result

Status: `REJECT_C1_BODY_DUPLICATION_SPILL_CANONICAL_RESTORED`.

The correct causal/noncausal implementation needs two `qi1` instantiations:
causal seeds its first valid tile, while noncausal must accumulate on the
already-live `qi0` result. Compiler `e0f10535` does not merge these bodies.

Compared with accepted `6f445c0`:

- static MMAC sites grow `1472 -> 1600`;
- waits grow `305 -> 324` and branches `76 -> 77`;
- `v_mov_b64` grows `84 -> 116` instead of falling;
- C1 role use grows `162 -> 203` in a 204-VGPR window;
- metadata reports `private_segment=20` and `vgpr_spill=4`.

The experiment therefore fails both the no-body-duplication and resource
gates before PMD. Repartitioning WDRA would only hide the allocator symptom;
it would not remove the duplicated 128 MMAC sites or instruction footprint.
Canonical source is restored. Future C1 zero seeding requires a compile-time
causal kernel specialization or a native accumulator-select form proven not
to duplicate MMAC bodies; neither belongs in this canonical runtime-causal
kernel as a local patch.
