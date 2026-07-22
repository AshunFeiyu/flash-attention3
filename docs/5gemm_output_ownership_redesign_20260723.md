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

This opens only the 128-live VGPR resource gate. The next mandatory gate is a
12-wave, `24/240/96`, two-generation native dS conveyor probe. Production is
not rewritten until that probe proves phase ownership, native reader/writer
semantics, checksum correctness, and clean PMD execution.
