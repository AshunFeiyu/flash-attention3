# Five-GEMM Output Ownership Redesign

## Baseline Diagnosis

The canonical M64/N128/D128 kernel is a correctness baseline, not a viable
performance schedule. H1/S1024 executes the exact 92,160 useful MMOPs and has
zero LDS bank conflicts, but its partial FP32 atomic epilogue dominates:

```text
kernel_ticks       3,074,860,880
useful MMAC active 0.589255%
waitVm share       61.1656%
barrier share      32.7436%
```

The measured result rejects waitcnt micro-tuning as the next action. The
output owner must change before SQTT scheduling work is meaningful.

## Canonical Conveyor Contract

```text
waves0-3   producer: publish K/V once, then Q/dO/sidecar generations
waves4-7   dKV owners: one N32 each; persist dK+dV across the q-loop
waves8-11  dQ owners: one D32 each; consume one M16 dS generation at a time
```

The arithmetic remains exactly five GEMMs and 1,280 useful MMACs for every
M64/N128/D128 tile. dKV writes four native N32 dS pages. dQ reads their trans
view and contributes one dQ atomic per K tile. dK and dV store once at CTA
completion. Estimated atomic element traffic falls from about
`147,456 * q_tiles` to `8,192 * q_tiles`, roughly 18x.

LDS remains 115,456 bytes: 64 KiB resident K/V, 32 KiB Q/dO, 16 KiB for two
four-page dS generations, and 768 bytes sidecar. The target WDRA ledger is
`24/240/96 = 360`, not the correctness baseline's symmetric
`24/240/240 = 504`.

## 128-Live Resource Gate

The focused high-pressure probe now checks all 128 persistent FP32 VGPRs with
a host checksum and treats any PMD `read VGPR before writing` warning as a
failure. LLVM `e0f10535` plus PMD HEAD1694 passes:

```text
run              /zys/sb/fa3b/layout_probes/dkv_pds_split64_probe_20260723_022728
branch use       1/22/141/1 inside 16/176/248/8
metadata         SGPR29/VGPR112, private/spill0
ABarrier cases   2/2 pass across low and high LDS bases
pressure checksum mismatches 0
ldsBankConflict  0
PMD panic/warning 0/0
```

This opens only the 128-live VGPR resource gate.

## Two-Generation Conveyor Gate

The exact 12-wave `24/240/96` topology now passes its isolated structural
gate. Four dKV waves publish two alternating generations of four native dS
pages; four dQ waves read every page through the trans view, release the page
after the reads retire, and execute downstream MMAC while the next generation
can be produced.

```text
run              /zys/sb/fa3b/layout_probes/fused5_ds_conveyor_20260723_031356
branch use       producer/dQ/dKV = 1/49/136 inside 24/96/240
metadata         SGPR25/VGPR120, private/spill0
semantic oracle  view/output/128-live mismatches = 0/0/0
native path      writer t1/alt0 + trans reader + MMAC; scalar read/permute0
ldsBankConflict  0
PMD panic/warning 0/0
```

The decisive code-shape rule is to schedule `generation0` and `generation1`
as a fixed pair. A runtime `(iteration & 1)` branch caused LLVM to create a
whole-array PHI copy of the 128 persistent accumulators, driving branch usage
to 256 and spilling eight VGPRs. The paired loop lowers real dKV usage to 136
with the ordinary MMAC builtin; no inline-assembly workaround is required.

Both resource and native-conveyor gates are open. The next step is the single
canonical production rewrite with persistent dK/dV ownership and one minimal
dQ atomic contribution per K tile.
