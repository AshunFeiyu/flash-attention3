# Fused5 1P3C Resource And Native-Handoff Gate

## Hypothesis

The accepted M64/N128 fused kernel assigns one SIMD the following roles:

```text
producer + dKV-C0 + dKV-C1 + dQ-writer
MMAC/SIMD/tile = 0 + 128 + 128 + 64 = 320
```

The dQ writer is dependent on a completed dS batch and cannot contribute
score, dP, dV or dK work. Local read scheduling has reduced individual waits,
but it cannot change this role-level arithmetic ceiling.

The structural candidate is one producer group plus three symmetric heavy
consumer groups. Each group owns one disjoint N64 slice. Every consumer wave
owns one N16 dK/dV slice and one D32 dQ partial:

```text
producer + C0 + C1 + C2
MMAC/SIMD/M64xN192 tile = 0 + 160 + 160 + 160 = 480
```

The candidate remains exactly five logical GEMMs. Score and dP are evaluated
once for each `(q,k)` pair; dV/dK have unique N ownership; dQ is split over
the three disjoint N64 groups and combined by the existing partial-output ABI
or native fp32 accumulation after the resource gate.

## Tile And MMAC Ledger

| Item | Candidate |
| --- | ---: |
| Mq / Nk / D | 64 / 192 / 128 |
| Producer waves | 4 |
| Consumer groups | 3 x 4 waves |
| N ownership per group / wave | 64 / 16 |
| Score MMAC | 384 |
| dP MMAC | 384 |
| dV MMAC | 384 |
| dK MMAC | 384 |
| dQ MMAC | 384 |
| Total useful MMAC | 1,920 |
| Consumer-wave score+dP+dV+dK | 128 |
| Consumer-wave dQ partial | 32 |
| Total per consumer wave | 160 |

## Resource Draft

The first production draft retains the 48 KiB K source layout in LDS so dQ
can read a normal view without keeping a second K register layout alive.
V is latched in transposed register fragments and its released 48 KiB region
is reused by the streamed raw packet and dS handoff.

| Lifetime | Bytes |
| --- | ---: |
| startup K+V source layout | 96 KiB |
| steady retained K source layout | 48 KiB |
| one raw Q+dO M64 packet | 32 KiB |
| one dS panel per three groups | 12 KiB |
| two dS panel generations | 24 KiB |
| packed sidecar | 0.75 KiB |
| steady total with two dS generations | 104.75 KiB |

The target WDRA ledger is `32 + 160 + 160 + 160 = 512` VGPR/SIMD. No role
has spare physical VGPR capacity, so the gate models 112 persistent VGPRs
per consumer: dK+dV accumulators (64), K/V trans fragments (32), and the K
normal view required by dQ (16). Native dS normal/trans reads, dQ
accumulators and addresses are then added as real transient state.

## Ownership And Expected Pipeline

```text
time0  P: MLS K/V192 -> ResidentFilled
       C0/C1/C2: latch K/V trans; retain K LDS normal source

time1  P: MLS raw Q/dO packet t -> RawFilled
       C0: score/dP/softmax/dV/dK panel m
       C1: score/dP/softmax/dV/dK panel m
       C2: score/dP/softmax/dV/dK panel m

time2  Cg: write its four dS N16 fragments -> DsFilled[g][page]
       Cg: read four dS fragments + retained K normal view
       Cg: dQ partial MMAC for one D32 owner

time3  Cg: release dS page; continue next useful panel
       P: after RawUsed, MLS next raw packet while consumers finish dQ
```

ABarrier is only an ownership ledger. Matrix operand readiness still uses a
counted `lgkmcnt` immediately before first use. No ordinary `ds_read_b*`,
permute, gather or wrong-layout path is admitted.

## Admission Gate

Before changing the canonical kernel, the focused probe must prove:

1. 16 waves with `32/160/160/160` role windows compile with no private,
   scratch, SGPR spill or VGPR spill.
2. Three independent consumer groups publish dS with native
   `ds_write_matrix` and read the same page through normal/trans views.
3. Dense dK and dQ consumer oracles pass under persistent accumulator
   pressure, with dQ emitted by native FP32 global atomic add.
4. `ldsBankConflict=0`, no PMD panic and no uninitialized-VGPR warning.

Failure at this gate rejects the topology before a production rewrite.
Passing it only admits the resource/layout contract; it is not a performance
claim.

## Gate Result

Status: `ACCEPT_RESOURCE_LAYOUT_GATE`.

The high-pressure native-atomic probe passes on compiler `e0f10535`, PMD
`HEAD1694`, `GPU_CHIP=sb`:

```text
roles                  1/153/153/153 inside 32/160/160/160
metadata               SGPR28, VGPR128, private0, spill0, scratch0
native path             ds_write_matrix=6, trans read=24, normal read=6
compute/output          MMAC=270, native FP32 atomic=48
forbidden path          scalar matrix read=0, permute=0, permlane=0
correctness             dQ=0 mismatches, dK=0, pressure=0
PMD                     panic=0, VGPR warning=0, ldsBankConflict=0
```

Evidence:

`/zys/sb/race_codex/layout_probes/fused5_1p3c_native_handoff_20260818_201421`

Only seven branch VGPRs remain. The production implementation must reuse the
accepted dKV 1P3C register schedule and keep dQ accumulation panel-local; it
must not retain a full M tile of dQ accumulators or add a second address
pipeline before the metadata gate.

## Production Result

Status: `REJECT_BARRIER_AND_CAUSAL_OVERCOMPUTE`.

The first full implementation initially spilled because it retained K/V
trans, K normal, dK/dV accumulators and raw operand ping-pong together. A
resource-clean variant retained the 48 KiB K source layout in LDS, latched
only V, read K through native normal/trans matrix views, and used one 32 KiB
raw Q/dO page plus a 48 KiB dS batch. It compiled at branch usage
`1/153/153/153`, with private/spill/scratch0, passed S384 causal and
noncausal full dQ/dK/dV correctness, and had bank0.

The same-toolchain causal S384 result rejects the topology:

| Metric | a427 M64/N128 2C+writer | 1P3C M64/N192 |
| --- | ---: | ---: |
| fused ticks | 21,430,500 | 92,823,640 |
| MMAC active | 26.7021% | 10.2845% |
| barrier share | 19.9126% | 35.6276% |
| wait VM share | 3.0250% | 9.7453% |
| executed MMOP | 15,360 | 17,280 |

The single raw page serializes every q tile behind `RawUsed`; direct sidecar
loads add exposed VMEM, and N192 executes 12.5% more masked MMAC on the causal
triangle than N128 at S384. The probe remains valid capability evidence, but
this production realization must not replace `a427be9`.
