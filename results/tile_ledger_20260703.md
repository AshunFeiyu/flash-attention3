# dKV Tile Ledger, 2026-07-03

## Current Mainline

- Kernel: `fa3_bwd_dkv_mmac12_kernel`
- Route: one canonical W12 dKV route, dQ frozen
- Shape for design diagnosis: `B=1,H=1,S=1024,D=128,causal=true`
- Tile: `Mq=32,Nk=128,D=128`
- CTA waves: 12 waves
- Producer: waves 0-3 load resident K/V once, then stream Q, dO, Q^T source-layout, and dO^T source-layout for each q tile
- Consumers: waves 4-7 and waves 8-11, each consumer wave owns `Nk=16` rows and accumulates full `D=128` dK/dV

Current accepted evidence:

- PMD stats/full perf path:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/dkv_mmac_correctness_20260702_235606`
- XCU path:
  `/zys/shaobo_runs/fa3_bwd_wasp_clean/xcu_outputs/main_bottleneck_w12_earlycanon_h1s1024_20260702_235606`
- Resource gate: `private=0`, `sgpr=84`, `vgpr=112`, no spill/scratch
- Bank conflict: `ldsBankConflict=0`
- Total MMOP: `131072`
- MMAC active: about `21.88%`
- Top SQTT bubble: `s_abarrier_try_wait -> s_xor_b32`, about `28.66%`

## MMAC Count

For one consumer wave and one q tile:

```text
score = Q[Mq=32,D=128] x K^T[D=128,Nk_wave=16]
      = (32/16) * (16/16) * (128/16) = 16 MMAC

dP    = dO[Mq=32,D=128] x V^T[D=128,Nk_wave=16]
      = 16 MMAC

dV    = P^T[Nk_wave=16,Mq=32] x dO[Mq=32,D=128]
      = (16/16) * (128/16) * (32/16) = 16 MMAC

dK    = dS^T[Nk_wave=16,Mq=32] x Q[Mq=32,D=128]
      = 16 MMAC

total per consumer wave per q tile = 64 MMAC
total per CTA per q tile = 8 consumer waves * 64 = 512 MMAC
q tiles for S1024 = 1024 / 32 = 32
total per CTA = 512 * 32 = 16384 MMAC
total H1/S1024 dispatch = 8 K CTAs * 16384 = 131072 MMAC
```

This matches PMD `MMOP=131072`.

## LDS And Stream Bytes

Current W12 LDS plan is already exactly 128KB:

```text
K resident  = 128 * 128 * 2 = 32KB
V resident  = 128 * 128 * 2 = 32KB
Q raw       = 2 pages * 32 * 128 * 2 = 16KB
dO raw      = 2 pages * 32 * 128 * 2 = 16KB
Q^T source  = 2 pages * 32 * 128 * 2 = 16KB
dO^T source = 2 pages * 32 * 128 * 2 = 16KB
total       = 128KB
```

Lower-bound matrix movement per CTA for S1024:

```text
K/V resident load        = 64KB
Q/dO/source stream       = 32 q tiles * 32KB = 1024KB
dK/dV fp32 final stores  = 2 * 128 * 128 * 4 = 128KB
sidecar unique lower bound = 32 q tiles * 32 rows * 3 floats * 4 = 12KB
```

The low arithmetic-intensity number is not the main problem. The current
dominant loss is packet overhead: 32 q-packet turns, short 64-MMAC islands per
consumer wave, sidecar/global control, and RawUsed ABarrier bubbles.

## Why SIMD/MMAC Active Is Low

The current mainline is compute-correct and MMAC-backed, but it is not a long
FWD-style conveyor:

- each consumer wave only gets 64 MMAC before another q-packet control block;
- producer and consumer handshake every `Mq=32` rows;
- the source-layout requirement forces both raw and source pages to remain in
  LDS, leaving no spare LDS for a dedicated sidecar page;
- xcu shows RawUsed waits as the largest bubble, not LDS bank conflict;
- direct Mq64 attempts doubled the MMAC island but lost overlap because the
  128KB LDS budget forced single-buffer or extra ownership protocol.

## Larger Tile Options

| Option | Effective Mq | Resident Nk | Raw/source buffering | LDS | Consumer wave MMAC/q packet | Packet turns S1024 | Status |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Current W12 | 32 | 128 | raw double + source double | 128KB | 64 | 32 | canonical, active about 21.9% |
| Mq64 single buffer | 64 | 128 | raw single + source single | 128KB | 128 | 16 | correctness fixed in seedfix, perf rejected |
| Mq64 double buffer | 64 | 128 | raw double + source double | 192KB | 128 | 16 | impossible under 128KB |
| Mq128 single buffer | 128 | 128 | raw single + source single | 192KB | 256 | 8 | impossible under 128KB |
| Nk64 + Mq64 single | 64 | 64 | raw single + source single | 96KB | 128 | 16 | repeats Q/dO across more K CTAs, lower global reuse |

## Design Consequence

To approach 60% MMAC active, the next mainline cannot be another small local
edit. It must increase effective MMAC island length or hide packet waits while
preserving these hard constraints:

- no duplicate score/dP;
- no extra ABarrier token generation unless xcu proves it is hidden;
- no dedicated LDS append beyond the existing 128KB;
- raw Q/dO remains for score/dP and source-layout Q^T/dO^T remains for dV/dK,
  unless a focused MLS/ds_read_matrix probe proves a new legal pairing;
- main matrix path stays MLS/BPS + `ds_read_matrix` + `v_mmac`;
- correctness, no spill/scratch, and `ldsBankConflict=0` are non-negotiable.

## Next Candidate Direction

Preferred next design work:

1. Re-derive an Mq64-equivalent conveyor using existing two Mq32 pages, without
   appending LDS or adding source/raw token families.
2. Compare against a clean Mq64 single-buffer reference to decide whether the
   lost overlap is caused by page lifetime, source-layout read timing, or
   consumer live range.
3. If neither can move MMAC active above the low-20% band, switch to a
   FWD-style 16-wave role split only after the workbook proves producer1 has
   recurring useful work and the same 128KB LDS budget still holds.

