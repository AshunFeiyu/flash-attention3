# Fused5 C109 Sidecar Readiness Split

Date: 2026-08-27

Status: `ACCEPT_TICKS_AND_ACTIVE`.

## One Hypothesis

C85 publishes one `RawFilled` token only after both the Q/dO BPS loads and the
sidecar global-load-to-LDS path complete. SQTT shows about 32,644 producer
bubble cycles on the sidecar path. Roughly half overlap peer work, but the
remaining half delays the consumer's score MMAC even though score needs only
Q and K.

Split readiness without changing page ownership:

```text
producer:
    wait RawUsed(page) before reuse
    sequence RawFilled(page), SidecarFilled(page)
    BPS-load Q/dO
    wait BPS readiness
    arrive RawFilled(page)
    publish sidecar global -> LDS
    arrive SidecarFilled(page)

consumer:
    wait RawFilled(page)
    score MMAC; group1 also performs dP MMAC
    wait SidecarFilled(page) immediately before first softmax use
    continue the unchanged C85 arithmetic
    arrive RawUsed(page) after all matrix and sidecar uses
```

`RawUsed` remains the sole lifetime guard. The producer therefore cannot
overwrite either the raw matrices or sidecar page until both consumer groups
finish. No `SidecarUsed` token is needed.

## Invariants

- Formula, MMOP count, M64/N128/D128 tile, 16-wave roles and output ownership
  are unchanged.
- Main matrices remain BPS/MLS + `ds_read_matrix` + MMAC.
- Sidecar remains three scalar LDS reads; it does not enter the matrix path.
- LDS remains exactly 128 KiB and the sidecar does not alias either raw page.
- Producer WDRA remains 16 VGPR. Unlike the rejected prefetch-before-BPS
  experiment, sidecar values are not kept live across BPS instructions.
- Two producer-filled ABarrier tokens are added, one per page. The ledger grows
  from 12 to 14 contiguous IDs.

## Expected Effect

The score island can overlap the producer's sidecar global request and LDS
publication. Group1 can additionally cover it with its dP island. Expected
fused-kernel improvement is 0.5-2%; no improvement is acceptable evidence to
reject this ownership split.

## Gates

1. Static build: exact MMOP, LDS 131,072 B, no private segment, no spill or
   scratch, and all role VGPR windows respected.
2. ISA: `RawFilled` publication precedes the sidecar completion path; the
   first consumer sidecar wait is after score MMAC, not before matrix reads.
3. Correctness: H1/S128 causal and noncausal, then H1/S1024 causal.
4. Dynamic: `ldsBankConflict=0`, same PMD/compiler/SQ7 as C85.
5. Performance: repeated same-build C85/C109 lifecycle and fused ticks. Any
   accepted gain must be repeatable and explained by SQTT; otherwise restore
   C85 source and record `REJECT`.

## Measured Result

Compiler `e0f10535`, PMD `HEAD1694`, `GPU_CHIP=sb`, SQ7,
H1/S1024/D128 causal:

| Metric | C85 | C109 | Delta |
| --- | ---: | ---: | ---: |
| stats-only fused mean | 39,690,332.5 (2 runs) | 38,729,600 (3 runs) | -2.421% |
| stats-only lifecycle mean | 43,758,715 | 42,822,173.3 | -2.140% |
| fullperf fused ticks | 39,308,360 | 38,771,915 | -1.365% |
| fullperf lifecycle ticks | 43,553,965 | 42,819,140 | -1.687% |
| MMAC active | 39.054060% | 39.563669% | +0.509609 pp |
| `s_waitcnt` hot latency share | 28.46% | 23.90% | -4.56 pp |
| post-ABarrier `s_xor` latency share | 22.59% | 20.37% | -2.22 pp |

S128 causal/noncausal and S1024 causal full lifecycle correctness pass with
`ldsBankConflict=0`. Causal/noncausal metadata is private/spill/scratch zero;
producer use remains 9/16 VGPR and heavy consumers remain within 204 VGPR.
MMOP, VALU, LDS, VMEM and FLAT counts are unchanged. SCA rises
38,344 -> 41,048 and failed coissue rises, which prices the two extra tokens,
but the shorter readiness critical path wins overall.

Generated ISA preserves the intended order: four BPS loads, `RawFilled`
arrive, sidecar `global_load_dwordx3` plus LDS stores, then `SidecarFilled`
arrive. Each consumer's first sidecar wait follows its first score MMAC; group1
also covers it with dP MMAC. This is the mechanism supported by the tick and
SQTT changes.

Evidence:

- correctness/stats: `/zys/sb/fa3b/c109_sidecar_ready_runs` and
  `/zys/sb/fa3b/c109_ab_runs`;
- fullperf/xcu: `/zys/sb/fa3b/c109_sidecar_ready_fullperf/fused5_full/`
  `b1_hq1_hkv1_s1024_d128_c1_fullperf_20260827_102500`;
- shared archive: `/Volumes/172.20.68.76/共享/shaobo/perf/`
  `20260827_102500_C109_sidecar_ready_split_H1S1024_causal_SQ7`.
