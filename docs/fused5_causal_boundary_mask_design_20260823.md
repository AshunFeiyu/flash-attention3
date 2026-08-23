# Fused5 Causal Boundary-Mask Design

Status: `ACCEPT_CAUSAL_STEADY_MASK_ELISION_MMAC50_OPEN`.

## Measured Trigger

Accepted `58e90fc` executes the exact triangular q-tile domain, but every
retained q tile still passes runtime `causal` into the per-score-word
probability stage. The fullperf fused dispatch dynamically issues 117,156
VALU instructions and reaches 35.037% MMAC active. Source and SQTT show the
probability/dS VALU island is repeated in every q-loop epoch.

## Formula Proof

One CTA owns K rows:

```text
[k_base, k_base + Nk - 1], Nk = 128
```

Its first retained Q tile is selected by:

```text
q_tile_begin = k_base / Mq, Mq = 64
q_base(qi) = k_base + qi * 64
```

Therefore:

```text
qi = 0: Q=[k_base+0,   k_base+63]   -> partial causal tile
qi = 1: Q=[k_base+64,  k_base+127]  -> partial causal tile
qi >=2: Q row >= k_base+128 > every owned K row -> fully valid
```

Only `qi=0/1` may require `krow <= qrow`. For `qi>=2`, setting the local mask
mode to false is algebraically exact for score, P, dS, dV, dK and dQ. In
noncausal mode all q tiles remain unmasked.

## One Hypothesis

Keep the existing two peeled q tiles as boundary-mask calls. Compile the
steady `qi>=2` loop and its tail with mask disabled at the call site. Do not
change tile size, MMAC, ownership, LDS, ABarrier, matrix traffic, output
mapping, or launch order.

Expected generated-code effect:

- remove repeated K/Q row comparisons and conditional probability selects
  from the steady q-loop;
- preserve the same exp/scale/dS arithmetic for valid elements;
- reduce VALU/SCA and shorten the consumer path to RawUsed/dS publication;
- indirectly reduce producer ABarrier waits and raise MMAC active.

## Fixed Budget

| Item | Accepted | Candidate gate |
|---|---:|---:|
| tile | M64/N128/D128 | exact |
| logical GEMMs | 5 | exact |
| H1/S1024 MMOP | 92,160 | exact |
| LDS | 128KiB | exact |
| ABarrier IDs/sites | 12 / 102 | exact |
| branch resources | 9/176/87/164 | no growth beyond WDRA |
| SGPR/VGPR metadata | 82/128 | no spill/private/scratch |
| matrix path | MLS/BPS + ds_read_matrix + MMAC | exact |

## Expected Pipeline

```text
time0  producer: boundary raw0/1 publication
       C0/C1: score -> causal predicate -> P -> dP -> dS

time1+ producer: steady raw page refill
       C0/C1: score -> P -> dP -> dS (no causal compare/select)
       writer: peer dQ MMAC

effect producer RawUsed waits shorten only if consumer steady VALU was on the
       ownership critical path; no artificial delay or extra work is added.
```

## Admission

1. Generated ASM must retain exact MMAC/read/ABarrier counts and reduce the
   steady probability predicate/select instruction class.
2. Full CPU-golden correctness must pass S128 causal/noncausal, S1024 causal,
   and S2048 causal with warning0 and bank0.
3. Three interleaved S1024 pairs must reduce fused and full-lifecycle mean
   ticks. MMAC active, VALU/SCA, wait/barrier and coissue explain the result.
4. Two S2048 pairs must not regress. Fullperf/xcu follows only after scaling.
5. If the compiler merges masked/unmasked paths or adds branch/resource debt,
   reject before PMD and restore canonical source.

Workbook: section67 in the 2026-08-23 fused5 design workbook.

## Result

All admission gates pass. Generated code preserves MMAC1472, matrix reads847
(840 inside the fused symbol), ABarrier102, LDS/VMEM/FLAT traffic and all five
logical GEMMs. Role VGPR use is `9/173/87/162`, metadata is SGPR71/VGPR128,
and private/spill/scratch remain zero. Static waits fall `340 -> 312`,
`v_cmp` falls `162 -> 66`, and `v_cndmask` falls `160 -> 64`.

Full CPU-golden correctness passes S128 causal/noncausal, S1024 causal and
S2048 causal with warning0 and bank0. Paired runtime results are:

| Shape | Metric | Accepted | Candidate | Delta |
|---|---|---:|---:|---:|
| H1/S1024 | fused ticks, 3-pair mean | 43,207,255 | 42,109,643 | -2.540% |
| H1/S1024 | lifecycle ticks, 3-pair mean | 47,381,122 | 46,217,687 | -2.455% |
| H1/S2048 | fused ticks, 2-pair mean | 79,778,790 | 78,068,445 | -2.144% |
| H1/S2048 | lifecycle ticks, 2-pair mean | 87,210,760 | 85,557,290 | -1.896% |

Fullperf SQTT keeps MMOP92,160 exact while dynamic VALU falls
`117,156 -> 98,032` and SCA falls `46,656 -> 38,984`. MMAC active rises
`35.037% -> 36.659%`; dispatch issues fall `369,884 -> 342,752` and duration
falls `95,204 -> 93,128`. The producer ABarrier issue gap falls 4.53% and the
sidecar global-load wait falls 35.70%.

The remaining critical path is now more visibly readiness/synchronization
bound: wait-LGKM rises `7.428% -> 8.811%`, barrier share rises
`13.263% -> 14.245%`, and transpose-read-to-wait bubble cycles rise 22.45%.
This is not added work; it is exposed debt after removing steady causal VALU.
The next experiment must cover or shorten this existing readiness edge without
adding tokens, reads, barriers or fake delay.

Evidence: `/zys/sb/runs/f5causal_boundary_correctness_20260823`,
`/zys/sb/runs/f5causal_boundary_ab_20260823`,
`/zys/sb/runs/f5causal_boundary_s2048_ab_20260823`, and
`/zys/sb/runs/f5causal_boundary_fullperf_20260823`.
