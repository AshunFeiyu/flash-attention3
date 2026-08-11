# Fused5 Q-Double / Sidecar Overlay Result

Date: 2026-08-12

Classification:
`REJECT_A5_DK_LIFETIME / NO_PERF / RESTORE_B28E73D`.

## Hypothesis

Use a dedicated second 16KB Q page and place the 768-byte sidecar in the
proven unused half of a P/dS writer page. After both heavy consumer groups
finish score/dP/P/dV/dS, signal `EarlyUsed` so the producer can publish the
next Q/dO/sidecar packet while current dK and dQ continue.

## Reusable Probe

`ds_write_matrix_32x16_trans_f16` was tested on a sentinel-filled 2KB page.
With compiler e0f10535 and PMD HEAD1694 it touched exactly `[0,1024)` and left
the aligned `[1024,2048)` bytes unchanged. Metadata was SGPR13/VGPR17 with no
private/spill/scratch and no PMD panic.

Evidence:
`/zys/shaobo_runs/fused5_padding_probe_20260812/layout_probes/ds_write_matrix_padding_20260812_012617`.

## Integration Result

- Static/source gate: PASS.
- Role use: `8/163/166/84` within `8/200/200/88`.
- Metadata: SGPR60/VGPR124, LDS131072B, private/spill/scratch0.
- Dynamic work: exact MMOP2,560; `ldsBankConflict=0`.
- Causal overlap: dV/dQ PASS, dK FAIL `rel_l2=0.650218`.
- Noncausal overlap: dV/dQ PASS, dK FAIL `rel_l2=1.01811`.
- Serial-QUsed diagnostic: dK restored to `rel_l2=0.00123603`.

Evidence:

- overlap causal:
  `/zys/shaobo_runs/fused5_qdouble_sidecar_overlay_20260812/smoke/b1_h1_s128_d128_c1_20260812_014828`
- overlap noncausal:
  `/zys/shaobo_runs/fused5_qdouble_sidecar_overlay_20260812/smoke/b1_h1_s128_d128_c0_20260812_015003`
- serial diagnostic:
  `/zys/shaobo_runs/fused5_qdouble_serial_q_used_diag_20260812/b1_h1_s128_d128_c1_20260812_015404`

## Decision

No performance claim is admissible. A dedicated Q page removes the explicit
address alias, but does not make next-generation MLS/BPS safe while current
dK consumes LDS. The exact reason remains PMD-versus-undocumented lifecycle
behavior; it must not be presented as a hardware fact.

The prior correct split-lifetime experiment already proved that prefetching
only dead dO/sidecar moves the saved ABarrier debt into BPS readiness and
regresses ticks. Close this ownership tier, restore `b28e73d`, and redesign
the tile so each raw ownership epoch carries more useful MMAC work.
