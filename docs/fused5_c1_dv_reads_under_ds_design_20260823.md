# Fused5 C1 dV Reads Under dS

Decision: `ACCEPT_WAIT_REDUCTION_AND_TICKS / PACKED_DS_DEBT_OPEN`.

## Hypothesis

The accepted C1 schedule already carries four next-panel dO-trans reads across
the current dV MMAC. SQTT still showed a repeated current-panel edge:

```text
P-normal + dO-normal + next-dO-trans reads -> wait -> dV MMAC
```

Current dS is mathematically independent of those operand reads. Issue the
same packet before dS and use the existing dS VALU as readiness age:

```text
sidecar ready -> 9 matrix reads -> dS VALU -> lgkmcnt(4) -> 8 dV MMAC
```

No matrix transaction, MMAC, LDS byte, ownership token, output owner, or API
is added. C0, producer, writer, dK, and all 12 ABarrier lifecycles remain
canonical.

## Implementation Proof

- `DvOperandPacket` separates issue from first use without duplicating work.
- `row_delta` is made ready before the packet, so dS does not force an
  `lgkmcnt(0)` that drains the matrix reads.
- Two compiler-order dependencies preserve the intended ISA order. They do
  not create a hardware instruction or an uninitialized VGPR definition.
- The first empty-output implementation was rejected before performance
  testing because PMD reported 40 VGPR read-before-write warnings.
- Final generated ISA contains the target read/dS/wait/MMAC order and no
  extra pre-wait `v_mov` island.

## Hard Gates

| Gate | Result |
| --- | --- |
| Exact five GEMMs / dynamic MMOP | PASS; S1024 MMOP 92,160 |
| Main operand path | PASS; MLS/BPS + ds_read_matrix + MMAC |
| LDS / ABarrier | 128 KiB / 12 IDs, unchanged |
| WDRA role use | 9 / 171 / 87 / 164 |
| Metadata | SGPR82, VGPR128, private0, spill0, scratch0 |
| Correctness | S128 causal and noncausal, S1024, S2048 PASS |
| PMD lifecycle | panic0, VGPR warning0 |
| LDS bank conflict | 0 |

## Paired Performance

| Shape | Runs | Control ticks | Candidate ticks | Delta | MMAC active | waitLgkm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1/S1024/D128 causal | 3 pairs | 45,014,667 | 44,387,525 | -1.393% | 34.579% -> 34.629% | 8.814% -> 7.918% |
| H1/S2048/D128 causal | 2 pairs | 83,965,245 | 83,212,448 | -0.897% | 38.655% -> 38.664% | 9.846% -> 8.805% |

S2048 coissue success rises `76,766.5 -> 78,815` (`+2.67%`). The candidate
keeps exact MMOP/LDS/VMEM/FLAT counts and bank0. Compiler scheduling expands
S1024 VALU `118,880 -> 119,744` and SCA `46,744 -> 46,776`; this limits the
MMAC-active gain and is the next debt.

## SQTT Explanation

The same H1/S1024 capture changes:

- dispatch duration `99,820 -> 97,376`;
- `s_waitcnt` latency share `22.99% -> 20.96%`;
- trans-matrix-to-wait bubble `10.93% -> 8.50%`;
- normal-matrix-to-wait bubble `4.26% -> 3.58%`;
- remaining dS-pack-to-wait bubble is about `1.20%`.

This is a real first-use-distance improvement. It is not a barrier-count,
work-count, or occupancy artifact. MMAC active stays nearly flat because the
compiler emits more scalar dS VALU while the wait debt falls.

## Evidence And Next Step

- Control: `/zys/sb/experiments/fused5_ihv_baseline` (`b402ffd` behavior).
- Candidate: `/zys/sb/experiments/fused5_c1_dv_reads_under_ds`.
- S1024 A/B: `/zys/sb/fa3b/c1_dv_under_ds_ab`.
- S2048 A/B: `/zys/sb/fa3b/c1_dv_under_ds_s2048_ab`.
- Perf: `/zys/sb/fa3b/c1_dv_under_ds_fullperf`.
- XCU: `/zys/sb/fa3b/c1_dv_under_ds_xcu_20260823`.
- Workbook: section 47 in `fa3_bwd_5gemm_clean_design_20260822.xlsx`.

Next, keep this accepted readiness schedule and recover canonical packed dS
code generation. Do not add matrix reads, barriers, empty delay, or a new
kernel phase.
