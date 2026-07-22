# Full FA BWD Correctness Chain

## Scope

The harness verifies one causal FP16 BHSD backward lifecycle with three GPU
dispatches:

1. `dot_do_o`: compute `D_i = sum(dO_i * O_i)` and publish the packed dKV
   sidecar `[max(QK)*scale*log2(e), 1/sum, D_i]`.
2. `dKV`: consume the GPU-produced sidecar and write FP32 dK and dV.
3. `dQ`: consume the same GPU-produced `D_i` plus forward max/sum and write
   FP32 dQ.

dKV and dQ are mathematically sibling branches, but they are launched in this
order on the default stream so one run covers the complete operator lifecycle.

## CPU Golden Cache

`scripts/generate_full_bwd_golden.py` computes Q, K, V, O, dO, forward
softmax auxiliaries, D, dQ, dK and dV with NumPy. Inputs and O are stored as
little-endian FP16; auxiliaries and gradients are little-endian FP32.

The cache key includes schema, deterministic-input formula, B/H/S/D, causal
mode and the exact FP32 softmax-scale bits. `manifest.json` records every
payload's dtype, shape, byte count and SHA256 plus a combined payload digest.

- Missing key: generate once through a temporary directory and atomically
  rename it into place (`golden_cache_status=MISS`).
- Existing valid key: read only (`golden_cache_status=HIT`).
- Missing, corrupt or incompatible payload: fail closed. Use `--regenerate`
  explicitly to replace it.

The default cache root is `/zys/shaobo_golden/fa3_bwd_7gemm`, outside git and
outside PMD run directories.

## Build Contract

`scripts/build_full_bwd_correctness.sh` builds and gates standalone dKV and dQ
ASM, then links a multi-object harness. dKV/dQ retain the locked local-wave
WDRA flags. The ordinary 128-thread `dot_do_o` kernel and host harness are
compiled without local-wave; otherwise PMD sees an uninitialized WDRA mode
before compiler-generated `s_set_vgpr_size`.

Current hard gates:

- dKV metadata: SGPR52/VGPR96, private/spill/scratch0.
- dQ metadata: SGPR60/VGPR128, private/spill/scratch0.
- Existing dKV/dQ static instruction gates pass.
- PMD emits exactly three non-empty dispatch stats files.
- All four comparisons report max/mean absolute error, RMSE, relative L2,
  cosine error and non-finite count.

## Commands

```bash
cd /zys/shaobo/fa3_bwd_wasp_7gemm_consumer_conveyor_20260717
scripts/build_full_bwd_correctness.sh
S=128 SHAOBO_RUN_ROOT=/zys/sb/fullbwd scripts/run_full_bwd_correctness.sh
S=1024 SKIP_BUILD=1 SHAOBO_RUN_ROOT=/zys/sb/fullbwd \
  scripts/run_full_bwd_correctness.sh
```

The scripts inherit the locked compiler, PMD HEAD1694, `GPU_CHIP=sb`, SQ7 and
the audited PMD config seed from `scripts/env.sh`.

## Verified Evidence

- H1/S128: PASS, cache HIT on repeat, exactly three dispatches. Maximum errors:
  D `3.73e-9`, dK `6.02e-8`, dV `2.74e-5`, dQ `4.55e-8`.
- H1/S1024: PASS, exactly three dispatches. Maximum errors: D `3.73e-9`,
  dK `1.48e-7`, dV `2.88e-5`, dQ `1.86e-7`.
- Evidence roots:
  `/zys/sb/fullbwd_final/full_bwd_correctness_20260722_112808/m5out` and
  `/zys/sb/fullbwd_final/full_bwd_correctness_20260722_112827/m5out`.

This harness is a correctness gate, not a fused-performance measurement. The
simple dot stage is intentionally separate and its ticks must not be compared
with the optimized dKV/dQ kernels.
