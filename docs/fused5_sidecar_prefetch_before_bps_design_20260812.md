# Fused5 Sidecar Prefetch Before BPS Design

Date: 2026-08-12

Status: `REJECT_TICKS_REGRESSION / CANONICAL_RESTORED`.

## One Hypothesis

The accepted raw Q/dO double-page kernel publishes each packet with this
producer sequence:

```text
four Q/dO BPS matrix loads
global_load_dwordx3(sidecar)
wait sidecar
three LDS sidecar stores
wait BPS/vbcnt
RawFilled arrive
```

SQTT repeatedly attributes about 500-1367 issue-gap cycles to the sidecar
`global_load_dwordx3`; the representative producer wave accumulates about
13,048 cycles on this opcode. Move only that global load before the four BPS
matrix loads, keep its three values in producer VGPRs, and publish them after
the matrix loads. The BPS issue window should age the sidecar request before
its first use.

This round must not change:

- the five-GEMM formula, exact MMOP92,160, or output ownership;
- M64/N128/D128, 16-wave roles, or consumer arithmetic order;
- two complete 32KB raw pages, LDS131,072B, or the eight ABarrier tokens;
- the main matrix path, which remains BPS/MLS + `ds_read_matrix` + MMAC.

## Pseudocode

```text
producer_fill_raw(page, tile):
    sequence RawFilled(page)
    sidecar = global_load_dwordx3(tile rows owned by this wave)
    BPS MLS Q(tile, page)
    BPS MLS dO(tile, page)
    if first tile:
        wait KvDsUsed
    wait sidecar before first use
    publish sidecar to LDS(page)
    wait BPS/vbcnt before publication
    arrive RawFilled(page)
```

The implementation should express this as two cohesive helpers:

- `producer_prefetch_raw_sidecar`: global load only, returns three FP32 values;
- `producer_publish_raw_sidecar`: LDS stores only, consumes those values.

No page specialization, phase switch, duplicate producer loop, or second
production path is permitted.

## Resource And Dependency Budget

| Item | Accepted baseline | Candidate budget | Gate |
|---|---:|---:|---|
| producer actual VGPR | 9 | at most 16 | WDRA role window |
| sidecar live values | 0 across BPS | 3 FP32 | no spill/private |
| LDS | 131,072B | 131,072B | exact |
| barriers | 8 | 8 | unchanged |
| MMOP | 92,160 | 92,160 | exact |
| static kernel body | 2,076 instruction lines | no material duplication | inspect ASM |

The sidecar load has no data dependency on Q/dO BPS. Its only ordering
requirement is completion before the three LDS sidecar stores. BPS completion
and `vbcnt` ordering remain required before `RawFilled` publication. The first
packet's `KvDsUsed` handoff remains in its original location after the BPS
loads and before sidecar publication.

## Expected ISA And SQTT

Expected producer issue order:

```text
global_load_dwordx3
matrix_load_32x32_b16_bps_lds x4
s_waitcnt ...              # sidecar first use / existing BPS readiness
ds_write_b32 x3
RawFilled arrive
```

The compiler may conservatively move the load or wait. Promotion therefore
requires ISA/SQTT evidence, not source-order inference. Success should reduce
sidecar-attributed issue-gap without increasing ABarrier, BPS/vbcnt, fetch, or
consumer wait debt.

## Admission Gates

1. A1/static: one canonical body, exact MMOP92,160, LDS131,072B,
   private/spill/scratch0, producer VGPR at most16.
2. A2-A5 correctness: H1/S128 causal and noncausal, then repeated H1/S1024
   causal complete lifecycle.
3. Dynamic health: `ldsBankConflict=0`; no extra matrix-path DS/gather.
4. Performance: compare repeated complete and compute ticks with accepted
   means 53,207,245 and 50,290,467.5. Any repeatable positive delta may be
   accepted; ticks remain the deciding metric.
5. A6 only for an admitted candidate: xcu must show the sidecar global request
   issued before BPS and a smaller exposed gap, while MMAC active does not
   regress without a stronger complete-tick benefit.

If producer resource use exceeds its WDRA window, correctness fails, or ticks
do not improve, restore the accepted source and close this sidecar scheduling
tier.

## Measured Result

The compiler emits the intended order: `global_load_dwordx3`, four BPS matrix
loads, `s_waitcnt vmcnt(0)`, and three `ds_write_b32`. Producer VGPR rises only
9 -> 12 within its 16-register role window. S128 causal/noncausal and repeated
S1024 causal correctness pass with exact MMOP92,160, LDS131,072B,
private/spill/scratch0 and bank0.

The latency hiding is real but not profitable:

- wait-VM share falls 2.589% -> 1.633%;
- dynamic VALU rises 127,352 -> 128,280 and SCA 58,336 -> 59,424;
- static kernel instruction lines rise 2,076 -> 2,114;
- wait-LGKM rises 11.572% -> 12.402%;
- compute mean rises 50,290,467.5 -> 51,558,552.5 (+2.522%);
- complete mean rises 53,207,245 -> 54,279,907.5 (+2.016%);
- MMAC active falls 31.476233% -> 31.059025%.

Splitting one lane-predicated load-and-publish block into two blocks adds a
second exec-mask/address-control path and extends three VGPR values across the
BPS sequence. The observed lower VM wait is outweighed by added instruction
and LGKM/control pressure. No fullperf/SQTT capture is admitted after the
same-shape tick regression. Production source is restored to `d62c645`.
