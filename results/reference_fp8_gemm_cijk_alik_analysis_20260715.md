# Shaobo FP8 GEMM Cijk/Alik reference analysis

Status: `OBSERVE`

## Evidence

- Perf: `/Volumes/172.20.68.76/共享/shaobo/参考perf/540084_fp8_gemm.perf`
- Source CSV: `/Volumes/172.20.68.76/共享/shaobo/参考perf/1.csv`
- Kernel: `Cijk_Alik_Bljk_F8BS_MT256x256x128_TT16_8_WG16_32_1_MFWGS768_WGM1_bias_channelwise`
- The kernel name records `MFWGS768`; with wave64 this is a 12-wave CTA.
- The CSV contains 48 selected wavefronts. Hit counts are aggregate execution
  counts and are not interpreted as wave counts by themselves.

## Reconstructed role topology

The active control flow separates a 4-wave-equivalent streaming path from an
8-wave-equivalent compute path:

- `0x003ac94c..0x003ac9bc`: eight
  `buffer_load_dwordx4 ... lds` instructions form the producer BPS packet.
- `0x003ac9c4`, `0x003aca08`: CTA barriers bracket producer visibility and
  buffer reuse.
- `0x003aca1c`: the narrow path terminates after the producer loop.
- `0x003acaa0..0x003acc9c`: the compute path initializes 128 accumulators.
- `0x003accc0..0x003accd8` and later four-load groups prefetch the other GEMM
  operand directly into consumer VGPRs.
- `s73 & 7` control at `0x003acd08` selects staggered compute subgroups.

The important property is not the exact source-level role name: producer work
continues across the main loop, while consumers also prefetch an independent
operand. The design does not leave all recurrent data movement on one role.

## Mainloop cadence

There are eight active MMAC islands. Each island contains 64 consecutive
`v_mmac_f32_16x16x32_fp8_fp8` instructions. A representative epoch is:

```text
producer: BPS global -> LDS for the next packet
consumer: 16 x ds_read_b128
consumer: s_waitcnt lgkmcnt(0)
consumer: 1 x MMAC, s_setprio 1, 63 x MMAC, s_setprio 0
all roles: one coarse CTA barrier / ownership boundary
```

Some epochs split the sixteen LDS reads into `8 reads + 4 direct global loads
+ 8 reads`, so useful consumer prefetch is placed between read groups. The
active mainloop has:

| Item | Dynamic hits | Source latency sum |
| --- | ---: | ---: |
| MMAC | 32,768 | 303,944 |
| `s_waitcnt` | 800 | 183,572 |
| `ds_read_b128` | 8,192 | 113,812 |
| direct consumer global loads | 1,152 | 45,852 |
| `s_barrier` | 576 | 2,848 |

Within this source range, MMAC accounts for 44.38% of the exported latency
sum, waits 26.80%, and LDS reads 16.62%. These Source sums are attribution
evidence, not elapsed time or MMAC-active share.

## Zero initialization

`0x003acaa0..0x003acc9c` contains 128 consecutive `v_mov_b32_e32 vN, 0`
instructions. Their latency sum is 22,452, or 1.41% of the full exported
latency sum. This reference therefore does not prove that zero moves are free;
it shows that a large and regular MMAC island can amortize a one-time 128-VGPR
initialization. Removing moves is secondary to fixing small MMAC islands and
recurrent synchronization.

## Comparison with current seven-GEMM dKV

The rejected consumer-owned-ring capture has the same MMOP count as its
accepted baseline but regresses from 43,103,060 to 53,225,445 ticks and from
33.24% to 26.82% MMAC active. Its SQTT is dominated by terminal role imbalance
and recurrent ownership/control:

- `s_ebarrier_sync -> s_cbranch_vccnz`: 50.64% aggregate issue-gap evidence.
- `s_abarrier_try_wait -> s_xor_b32`: 4.39%.
- matrix read to wait gaps: 2.76% normal plus 1.89% transpose.
- SCA rises from 114,520 to 155,360 while MMOP remains 131,072.

The FP8 GEMM reference suggests the next architectural experiment should:

1. use one 4-wave producer group and two symmetric 4-wave consumer groups;
2. let the producer stream the recurrent Q/dO packet, while consumers keep
   useful operand prefetch/latching work;
3. merge Q and dO readiness into a coarse packet-generation boundary instead
   of independent tensor/half ownership chains where correctness permits;
4. target a long MMAC island between ownership boundaries and use `s_setprio`
   around the island;
5. judge producer thinness from the producer/consumer work ratio, not from an
   expectation that every role must have equal instruction count.

This does not justify mechanically replacing dKV ABarrier with `s_barrier`.
BWD has Q/dO normal/trans views, sidecar, softmax/dS, four GEMMs, and output
accumulator lifetimes. A 12-wave coarse-packet design needs a fresh LDS/VGPR
budget and correctness proof.

## Tool compatibility and remaining evidence

The jump route was restored through `172.20.32.79`, and the complete artifact
was uploaded to zys1 as
`/zys/shaobo/reference_perf_cijk/540084_fp8_gemm.perf`. `xcu` 4.6.3 accepted
the file and section arguments, but printed only section headers and no SQTT
rows for `detail`, `wavefronts`, `bubbles`, or `coissue`. Treat this as an
artifact/tool schema compatibility issue, not evidence that the capture has no
SQTT data.

XCompute Ultra 4.5.1 opens the same artifact successfully. Its Summary identifies
dispatch 6 as this kernel with grid `(16,16,1)` and block `(768,1,1)`, directly
proving a 12-wave CTA. The exported Source CSV is therefore the primary machine-
readable evidence for this report. Exact timeline overlap, SIMD balance, and
coissue still require a compatible CLI build or GUI Wavefronts inspection.
