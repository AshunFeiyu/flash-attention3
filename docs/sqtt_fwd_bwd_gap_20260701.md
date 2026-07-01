# FWD vs BWD SQTT Gap, 2026-07-01

## Setup

- Perf root: `/Volumes/172.20.68.76/共享/shaobo/perf/20260701_193535_fwd_bwd_sqtt_h4s1024_sqc7_xcu`
- Shape: `B=1,H=4,S=1024,D=128,causal=true,fp16`
- Runtime knobs: `GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`
- FWD perf: `FWD_H4S1024_SQC7.perf`, XCU dispatch 0
- BWD perf: `BWD_C125C_H4S1024_SQC7_dKV.perf`, XCU dispatch 1

## Dispatch Evidence

| Metric | FWD | BWD C125C dKV | Meaning |
|---|---:|---:|---|
| Duration | 60,608 | 171,620 | BWD is much longer even with more active CUs. |
| CUs | 16 | 32 | BWD occupies more CUs, but not with useful MMAC work. |
| Waves | 256 | 512 | BWD has twice the wave count. |
| Inst issues | 549,136 | 5,161,952 | BWD has about 9.4x instruction volume. |
| MMAC latency share | 45.96% | 18.44% | Main gap: BWD fails to form long MMAC-active regions. |
| SALU32 latency share | 7.30% | 25.52% | BWD scalar/control stream dominates. |
| LDS matrix share | 5.89% | 6.92% | BWD does use `ds_read_matrix`; the issue is scheduling/hiding. |
| ABarrier opcode share | 1.39% | 1.73% | ABarrier instructions are not many, but their wait chains are costly. |

## Bubble Evidence

| Bubble | FWD | BWD C125C dKV | Meaning |
|---|---:|---:|---|
| Top bubble | `abarrier -> salu_32`, 45.04% | `abarrier -> abarrier`, 50.02% | BWD has barrier-chain serialization, not a smooth ownership handoff. |
| Matrix-read bubble | `lds_matrix -> mmop`, 1.18% | `lds_matrix -> immed`, 10.44% | BWD read latency is not hidden before useful MMAC. |
| Branch/control bubble | `jump -> salu_32`, 2.34% | `jump -> salu_32`, 9.39% | BWD inner protocol has too much control flow. |
| MMAC self gap | `mmop -> mmop`, 2.68% | not a top BWD bubble | BWD is not limited by MMAC issue spacing; it is not reaching enough MMAC islands. |

The selected BWD barrier hotspot window
`remote_tree/bwd_c125c_h4s1024_sqc7/short_tt_perf/xcu_dispatch1_bubble_abarrier`
is especially clear: 90 instructions, 20 bubbles, 96.01% bubble cycles, and
top bubble `abarrier -> abarrier` at 8004 cycles.

## Expected Code-Level Pipeline

The clean dKV kernel should implement this steady pipeline:

```text
time0:
  producer A publishes K/V0 + Q raw packet
  producer B publishes K/V1 + dO raw packet

time1:
  consumer group 0/1 run long score + dP MMAC islands
  producer A/B prepare next raw/source-layout packet

time2:
  consumer group 0/1 run softmax + dS VALU
  producer A/B publish Q^T / dO^T source-layout operands or next packet

time3:
  consumer group 0/1 run long dV + dK MMAC islands
  producer A/B prefetch the next useful packet
```

The intended coissue pattern is same-SIMD, different-wave overlap: one wave is
in a long MMAC island while another wave performs useful VALU/SALU/producer
work.  `s_abarrier` should be a low-frequency packet ownership fence, not a
per-substage baton.

## Actual BWD Pattern

The C125C trace does not match the expected conveyor:

- Consumers and producers frequently rendezvous on barrier chains; the top
  bubble is `abarrier -> abarrier`, not `mmop -> mmop`.
- `ds_read_matrix` is followed by immediate/control work often enough for
  `lds_matrix -> immed` to become the second-largest bubble class.
- The scalar/control stream is too large: SALU32 is 25.52%, and branch
  instructions are visible in both hot instructions and bubbles.
- Flat/global traffic is not the dominant problem in this trace.
- The trace proves there is MMAC, but the MMAC work is chopped into islands
  separated by barrier/control/read-latency gaps.

## Diagnosis

We should redesign the BWD dKV pipeline protocol, not the math ownership:

- Keep the algorithm goal: no duplicate score/dP and no wrong dV/dK ownership.
- Rewrite the packet ownership ledger so a producer can advance to useful
  next-packet work instead of entering `abarrier -> abarrier` chains.
- Batch `ds_read_matrix` operands and delay `s_waitcnt lgkmcnt(0)` until first
  use; do not put producer-side waits after MLS publication.
- Collapse inner-loop scalar state: hoist predicates, page ids, role constants,
  and branch decisions out of the fragment loop where possible.
- Split consumer code into explicit long islands: score/dP MMAC, softmax/dS
  VALU, dV/dK MMAC, store epilogue.

This is a pipeline redesign, but not a restart from scratch.  The next code
work should preserve the clean repo contract and replace the old C125C-style
barrier-heavy protocol with a FWD-style packet conveyor.
