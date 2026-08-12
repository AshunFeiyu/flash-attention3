# Fused5 Consumer0 Raw-Page Prefetch

Date: 2026-08-12

## One hypothesis

Move only the recurring Q/dO/sidecar packet publication after the first raw
page from producer waves 0-3 to consumer0 waves 4-7. Consumer0 publishes the
next page while consumer1 computes the current page. The existing two physical
raw pages and `RawFilled0/1` plus `RawUsed0/1` generations remain unchanged.

This targets the measured chain:

```text
dK(t) -> RawUsed(t) -> producer BPS(t+1) -> RawFilled(t+1)
```

The exact five-GEMM DAG, M64/N128/D128 tile, dS publication, dQ writer and
output ownership do not change.

## Ownership protocol

```text
producer:  resident K/V + raw page 0 -> RawFilled0 -> wait final RawUsed
consumer0: wait page t; BPS Q/dO/sidecar page t+1 -> RawFilled(t+1)
           then compute page t and arrive RawUsed(t)
consumer1: wait RawFilled(t), compute page t, arrive RawUsed(t)
```

Page reuse remains guarded by both consumer groups. Consumer0 is the sole
publisher after page 0, so there is no producer/consumer double writer or new
barrier token. Consumer0's BPS addresses use the same four-wave stripe and the
same source-layout path as the old producer helper.

## Resource budget

- LDS: unchanged at 131072 bytes.
- MMOP: unchanged at 92160 for H1/S1024.
- WDRA windows: unchanged target `16/204/204/88`; consumer0 must absorb the
  temporary BPS source state without private/spill/scratch.
- Matrix path: unchanged MLS/BPS + `ds_read_matrix` + MMAC; no ordinary DS,
  permute or gather path.

## Expected pipeline

```text
time0: P loads raw0; C0/C1 latch K/V and consume raw0
time1: C0 publishes raw1 while C1 executes score/dP/softmax/dV/dS(raw0)
time2: C0 executes score/dP/softmax/dV/dS(raw0); C1 consumes aged raw1
time3: C0 publishes raw2 while C1 executes raw1; both retire RawUsed by page
```

The candidate is useful only if the producer `RawUsed` gap shrinks without
creating a larger consumer0 first-use gap or BPS/vbcnt debt. A higher coissue
counter alone is insufficient.

## Gates

1. Static build, exact five GEMMs, no spill/private/scratch, and bank0.
2. H1/S128 complete correctness, then H1/S1024 complete correctness.
3. H1/S1024 fused and full ticks no worse than the wait-pruned canonical
   (`47.430868M` fused, `52.596180M` full mean).
4. Stats must explain `RawFilled/RawUsed`, MMAC active, wait-LGKM, barrier,
   VMEM/BPS and per-SIMD balance. Full SQTT remains pending while PMD ASTCA
   config generation is broken.
