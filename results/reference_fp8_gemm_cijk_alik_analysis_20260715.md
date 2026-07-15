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

## Instruction grammar and regularity

The reference is regular at the instruction-stream level, not only in the
Wavefronts visualization. Its eight compute epochs alternate between two
nearly fixed templates:

```text
Template A:
  16 x ds_read_b128
  1 x s_waitcnt lgkmcnt(0)
  1 x MMAC, s_setprio 1, 63 x MMAC, s_setprio 0
  1 x coarse barrier

Template B:
  8 x ds_read_b128
  4 x buffer_load_dwordx4 for a future operand
  8 x ds_read_b128
  5 x scalar address/loop instructions
  1 x s_waitcnt lgkmcnt(0)
  1 x MMAC, s_setprio 1, 63 x MMAC, s_setprio 0
  1 x coarse barrier
```

All eight semantic MMAC islands contain exactly 64 MMAC instructions. The 128
LDS reads form four runs of 16 and eight runs of 8; no MMAC or LDS-read run is
a singleton. The four global loads in Template B are useful prefetch work, not
arbitrary instructions inserted into the MMAC body. The five scalar
instructions update the next global address and loop predicate before the
first-use wait.

The latest generated seven-GEMM dKV assembly has a sharply different static
shape. This comparison uses the generated kernel assembly only; it does not
equate static instruction counts with dynamic runtime:

| Structural metric | FP8 GEMM reference | Current dKV assembly |
| --- | ---: | ---: |
| MMAC instructions | 512 | 512 |
| MMAC runs | 8 semantic islands | 112 strict runs |
| Mean MMACs per run | 64 | 4.57 |
| Singleton MMAC runs | 0 | 62 / 112 (55.4%) |
| Matrix/LDS read instructions | 128 | 288 |
| Matrix/LDS read runs | 12 | 176 |
| Mean reads per run | 10.67 | 1.64 |
| Singleton read runs | 0 | 134 / 176 (76.1%) |

A representative dKV score/dP region starts with a read bundle and wait, but
then becomes `MMAC -> scalar LDS-address setup -> several MMAC -> one or two
matrix reads -> MMAC`. This matches the irregular pattern seen in Wavefronts.
The compiler is not acting randomly: each standalone matrix-read helper
materializes a separate scalar LDS address, and the scheduler moves those
address operations and future reads between MMACs to hide local latency. The
source exposes many small scheduling units, so the compiler produces a locally
plausible but globally fragmented stream.

The important distinction is that reference Template B also interleaves work,
but only at a deliberate macro boundary: `8 reads / 4 useful loads / 8 reads`.
It does not alternate one read with one or two MMACs throughout the compute
body.

## What to reuse

1. Define source-level packet helpers whose generated unit is a complete
   instruction block: precompute LDS addresses, issue 8 or 16 matrix reads,
   wait once at first use, then execute a fixed MMAC block.
2. Keep address calculation and loop predicates outside the MMAC body. Prefer
   a shared base plus immediate offsets over one SGPR materialization per read.
3. Use `s_setprio` to bracket a real MMAC island, rather than a broad helper
   whose body still contains reads, address setup, and waits.
4. Interleave only useful future work at a stable boundary. For dKV this can be
   the next Q/dO packet or independent sidecar preparation, not arbitrary
   instruction-by-instruction mixing.
5. Let peer waves provide coissue: one consumer should expose a sufficiently
   long MMAC island while the other consumer performs softmax/dS or packet
   preparation. Fine-grained read/MMAC alternation inside both consumers makes
   them converge on the same dependency rhythm.
6. Add an assembly regularity gate. The first practical dKV target is at least
   `8 matrix reads -> at most one first-use wait -> 16 or more MMACs`, with no
   address SALU, barrier, or unrelated VALU inside the MMAC block. SQTT, ticks,
   and correctness still decide whether a larger island is beneficial.

The reusable checker is `scripts/analyze_asm_islands.py`. The current assembly
is expected to fail the proposed structural target:

```bash
scripts/analyze_asm_islands.py \
  --asm build/fa3_bwd_wasp_clean.asm \
  --symbol fa3_bwd_dkv_kernel \
  --min-mean-mmac 16 \
  --max-singleton-mmac-pct 10 \
  --max-singleton-read-pct 10
```

This is initially an analysis gate, not a promotion gate. It may be tightened
only after a correctness-passing version demonstrates that the regular shape
also lowers same-shape ticks and improves SQTT MMAC active.

## What not to copy

1. The reference uses `ds_read_b128`, not Shaobo's normal/transpose
   `ds_read_matrix` pair. Its exact read instruction and layout are not a dKV
   implementation template.
2. Its coarse `s_barrier` works for a regular GEMM epoch. Replacing dKV
   ABarrier ownership with CTA barriers would serialize independent Q/dO
   generations and consumers.
3. Its single `lgkmcnt(0)` still creates a hard read-to-MMAC dependency. The
   cost is amortized by 64 MMACs; dKV must verify the same tradeoff rather than
   copying the wait count blindly.
4. It initializes 128 accumulator VGPRs with 128 `v_mov` instructions. That is
   tolerated by the long steady loop, not an initialization pattern to adopt.
5. Direct consumer global prefetch is useful in this GEMM, but sidecar or
   transposed-source loads in dKV may require producer ownership and LDS
   visibility. The destination and dependency chain matter more than matching
   the opcode count.
6. Visual regularity alone is not success. A perfectly regular stream can
   still have poor occupancy, cache behavior, coissue, or correctness. The
   regularity gate is explanatory evidence, subordinate to same-shape ticks and
   SQTT MMAC-active results.

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
