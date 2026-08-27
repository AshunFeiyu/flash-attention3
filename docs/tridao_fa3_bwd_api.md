# Tri Dao FA3 backward API alignment

## Source contract

The interface follows `Dao-AILab/flash-attention` Hopper FA3 backward at local
upstream commit `2402cb0bed7a2185cb9ddbe88fb998656cf73066`.

The public backward inputs are:

```text
dout, q, k, v, out, softmax_lse
optional preallocated dq, dk, dv
cu_seqlens_q/k, seqused_q/k, max_seqlen_q/k
softmax_scale, causal, window_left/right, softcap
deterministic, sm_margin
```

Hopper FA3 backward does not expose dropout, ALiBi, `num_splits`, `pack_gqa`,
KV-cache, rotary, or attention-chunk parameters. Those belong to another
FlashAttention path or to forward only.

## Shaobo C ABI

Use `shaobo_fa3_params_init` before assigning fields. The integration entry is:

```c
size_t shaobo_fa3_bwd_v2_workspace_bytes(const ShaoboFa3Params* params);

int shaobo_fa3_bwd_v2(
    const void* dout,
    const void* q,
    const void* k,
    const void* v,
    const void* out,
    const void* softmax_lse,
    void* dq,
    void* dk,
    void* dv,
    void* softmax_d,
    const ShaoboFa3Params* params);
```

`softmax_lse` is natural-log LSE, matching Tri Dao. `softmax_d` is optional
FP32 storage for `sum(dout * out)`. If it is null, the value remains internal
to the workspace. Set `softmax_scale_is_set=0` to use `1/sqrt(head_dim)` or set
it to 1 and provide `softmax_scale` explicitly.

Tri Dao may allocate `dq/dk/dv` when its optional output tensors are absent.
This low-level C ABI does not own an allocator, so all three output pointers
must be supplied by the caller; a future PyTorch binding may allocate them
before invoking v2.

The old `shaobo_fa3_bwd` and component entry points remain available while the
five-GEMM integration path is validated. New framework bindings should target
v2 rather than the legacy two-auxiliary interface.

## Capability matrix

| Parameter or mode | Tri Dao FA3 BWD | Shaobo v2 status |
| --- | --- | --- |
| FP16 | yes | supported |
| BF16 | yes | unsupported |
| Fixed length | yes | supported |
| Varlen `cu_seqlens_q/k` | yes | rejected explicitly |
| `seqused_q/k` | yes | rejected explicitly |
| MHA | yes | supported |
| MQA/GQA | yes | supported through deterministic workspace reduction |
| Causal | yes | supported |
| Full attention | yes | supported |
| Local window | yes | rejected explicitly |
| Natural-log `softmax_lse` | yes | supported |
| Optional softmax scale | yes | supported with `softmax_scale_is_set` |
| Softcap | yes | only `0.0` accepted |
| Deterministic backward | yes | supported; current ownership is deterministic |
| `sm_margin` | yes | only `0` accepted |
| Fixed BSHD tensor layout | yes | unsupported |
| Contiguous BHSD tensor layout | wrapper-specific | supported |
| Dqk/Dv multiples of 8 up to 256 | yes | only Dqk=Dv=128 supported |
| Sq different from Sk | yes | unsupported |
| Dropout | not in Hopper FA3 BWD | only `0.0` accepted |

`shaobo_fa3_bwd_get_capabilities` is the machine-readable source of truth.
Fields that exist in the ABI but are not in the returned feature mask must not
be submitted to the kernel. `shaobo_fa3_bwd_v2_validate` returns
`SHAOBO_FA3_STATUS_UNSUPPORTED` before any dispatch for those combinations.

## Current execution boundary

The v2 call submits the complete five-GEMM lifecycle:

```text
dot_do_o(LSE -> packed sidecar)
fa3_bwd_5gemm(dK, dV, dQ partial)
dq_reduce(dQ partial -> FP16 dQ)
optional GQA dK/dV reduction
```

The LSE adapter uses the exact identity:

```text
P = exp(score * scale - LSE)
  = exp2(score * scale * log2(e) - LSE * log2(e))
```

It therefore packs `row_max_log2=LSE*log2(e)` and `row_inv_sum=1` without
reconstructing forward max/sum.

## Verification gate

```bash
scripts/build_fused5_full_bwd_correctness.sh
SKIP_BUILD=1 S=128 scripts/run_fused5_full_bwd_correctness.sh
SKIP_BUILD=1 S=1024 scripts/run_fused5_full_bwd_correctness.sh
SKIP_BUILD=1 S=128 H=4 HKV=2 scripts/run_fused5_full_bwd_correctness.sh
```

Promotion requires the API contract test, static metadata gates, full golden
checks for delta/dQ/dK/dV, bank conflict zero, and an unchanged normalized
instruction hash for `fa3_bwd_5gemm_kernel`.
