# dQ FP32 Native Matrix-Store Operator A/B

Status: `REJECT_PRODUCTION / ACCEPT_INSTRUCTION_ABI`.

## Question

Can the terminal dQ GEMM replace its FP16 direct partial store with:

```text
plain FP32 MMAC C -> ds_write_matrix_format_f32(trans)
                  -> matrix_store_16x16_b32 -> FP32 workspace
```

The instruction probe already proves this tuple is numerically exact. This
experiment measures its complete FA backward cost against commit `3388f47`.

## Variants

All measurements use compiler `e0f10535`, PMD `CoreArch:HEAD_1694`,
`GPU_CHIP=sb`, `GPU_ARGS=['--SQCIPfLines=7']`, and causal H1/S1024/D128.

| Variant | dQ MMAC/store | Fused ticks | dQ reduce ticks | Total ticks | Result |
|---|---|---:|---:|---:|---|
| Accepted baseline | `lit1/lts0`, FP16 direct | 40,252,940 | 1,589,315 | 44,272,865 | control |
| FP32 direct | `lit1/lts0`, FP32 vector direct | 39,794,300 | 2,113,475 | 44,355,675 | `OBSERVE`, +0.187% total |
| FP32 native | `lit0/lts0`, trans writer + B32 matrix store | 44,711,940 | 1,957,410 | 49,122,710 | `REJECT`, +10.954% total |

The FP32 direct control improves fused ticks by 1.139%, but the doubled
workspace slows reduction by 32.980%, leaving total time essentially flat.
The native path is 12.358% slower than the FP32 direct control inside the
fused dispatch and 10.747% slower end to end. Its eight per-writer-wave
16x16 stores serialize `DS write -> LDS readiness -> matrix store -> page
reuse`; the only admitted B32 store builtin is 16x16, so no wider native B32
store amortizes this sequence.

## `mmac_4interleave` Decision

FWD `mmac_4interleave` is `lit1/lts0`. It is already the accepted canonical
dQ MMAC and is the correct layout for direct dQ partial stores. The focused
probe proves it does not match either FP32 DS-writer orientation. The exact
native writer/store chain requires `lit0/lts0` plus the transposed FP32
writer.

A plain-MMAC direct-store diagnostic is therefore intentionally invalid: it
produces `dQ rel_l2 ~= 1.25` and `cosine_error ~= 0.79`. At the original WDRA
split it also spills 32 VGPRs; reallocating the writer window to 96 removes
the spill but cannot repair the layout. Its ticks are not an optimization
result.

## Correctness Gate Finding

The diagnostic exposed a harness loophole: low-amplitude wrong-layout dQ
passed because the old gate accepted either relative L2 or RMSE. The canonical
gate now requires both limits, so this failure class cannot be admitted by a
small absolute magnitude.

## Evidence

- Exact ABI probe:
  `/zys/sb/dq_f32_writer_store_test/layout_probes/dq_f32_dswrite_store_20260825_112154`
- FP32 native operator:
  `/zys/sb/runs/fused5_dq_fp32_native/b1_hq1_hkv1_s1024_d128_c1_20260825_113437`
- FP32 direct control:
  `/zys/sb/runs/fused5_dq_fp32_direct/b1_hq1_hkv1_s1024_d128_c1_20260825_114146`
- Plain direct negative:
  `/zys/sb/runs/fused5_dq_fp32_plain_direct_wdra96/b1_hq1_hkv1_s1024_d128_c1_20260825_114625`

Canonical source remains on the accepted FP16 direct dQ epilogue. A future
FP32 native attempt must first provide either a wider B32 matrix-store shape
or a proven multi-page schedule that hides page-reuse waits without consuming
active P/dS/raw LDS lifetimes.
