# Fused5 M32 Raw Conveyor Result

Date: 2026-08-12

Decision: `REJECT_STATIC_RESOURCE / CANONICAL_RESTORED`.

## Hypothesis

Split the M64 raw ownership epoch into two M32 releases. Heavy consumers would
capture two Q panels, signal `RawUsed`, and execute 16 dK MMAC while the
producer refilled that Q/dO/sidecar half for the next tile.

## Static Result

| Expression | WDRA windows | Metadata result |
|---|---|---|
| paired Q+dS | `8/200/200/88` | private16B, VGPR spill3 |
| paired Q; dS read later | `8/200/200/88` | private16B, VGPR spill3 |
| paired Q; intermediate quota | `8/204/204/88` | compiler rejects VGPR granularity |
| paired Q; full pool | `8/208/208/88` | private16B, VGPR spill3 |

The full-pool build still reports branch use `8/163/166/86`, but generated
metadata is authoritative for the no-spill gate. The extra simultaneous Q
panel is not representable spill-free by this source expression on compiler
`e0f10535`.

## Decision

Do not run PMD or claim pipeline improvement. Restore the accepted workspace
reduction source at `197e5d9`. This rejects paired-panel early release under
the present VGPR lifetime, not the general idea of producer refill under dK.
A future retry needs a layout/instruction expression that captures an M32 Q
half with fewer live registers, proven first in an isolated A4 resource probe.

Evidence directories:

- `/zys/shaobo/fa3_bwd_5gemm_toolkit/build/fused5_m32_raw_conveyor`
- `/zys/shaobo/fa3_bwd_5gemm_toolkit/build/fused5_m32_raw_conveyor_v2`
- `/zys/shaobo/fa3_bwd_5gemm_toolkit/build/fused5_m32_raw_conveyor_v4`
