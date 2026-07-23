# Fused5 Useful Stagger Design

Status: `ACCEPT_MICRO_TICKS_USEFUL_STAGGER / MMAC50_OPEN`

Workbook: `32 Useful Stagger` in
`/共享/shaobo/fa3_bwd_5gemm_clean_design_20260723.xlsx`.

## Hypothesis

The accepted 16-wave kernel has correct ownership and 21.81% useful MMAC
active, but SQTT still shows C0/C1 MMAC-vs-MMAC alignment. Keep every output
owner, LDS region, and ABarrier unchanged. Reorder only independent work in
the per-M16 DAG.

```text
score = Q @ K^T
dP    = dO @ V^T
P     = softmax(score)
dV   += P^T @ dO
dS    = P * (dP - D) * scale
```

Legal orders:

```text
C0: score(M) -> P(V) -> dV(M) -> dP(M) -> dS(V)
C1: dP(M) -> score(M) -> P(V) -> dS(V) -> dV(M)
```

After the first unavoidable MMAC/MMAC beat, the expected pairings are:

```text
C0 P VALU   | C1 score MMAC
C0 dV MMAC  | C1 P VALU
C0 dP MMAC  | C1 dS VALU
C0 dS VALU  | C1 dV MMAC
```

No arithmetic is duplicated.

## Code Shape

- Replace coupled `score_dp_stage` with one compile-time matrix-product helper.
- Each call emits four trans `ds_read_matrix`, one first-use wait, and eight
  MMAC.
- Split probability and dS into separate VALU helpers.
- Retain one FP32 probability fragment plus one FP16 MMAC fragment.
- Keep `Group` compile-time specialized; do not add a runtime phase or a new
  kernel.
- Keep dS publication, dK, dQ, atomics, LDS, and all ABarrier slots unchanged.

## Resource Estimate

| Role | Accepted use | Estimated peak | Window |
|---|---:|---:|---:|
| P0 | 8 | 8 | 8 |
| C0 | 168 | 184 | 200 |
| C1 | 169 | 185 | 200 |
| P1 | 84 | 84 | 88 |

LDS stays at 115,456 B. Useful work stays at 1,280 MMAC/tile and dynamic
MMOP 92,160 for H1/S1024 causal.

## Gates

- Exact five-GEMM static contract and dynamic MMOP.
- S128 causal/noncausal and S1024 causal correctness.
- Role use within `8/200/200/88`; private/spill/scratch 0.
- LDS 115,456 B and bank conflict 0.
- Same-shape ticks lower than commit `8b3f820`.
- XCU shows more C0/C1 MMAC-vs-VALU bins and fewer MMAC-vs-MMAC bins.

Reject if FP32 P lifetime spills, generated work changes, a new page/token is
needed, or same-shape ticks do not improve.

## Actual Result

- Compile role use is `8/165/168/84`; private/spill/scratch remain zero.
- S128 causal/noncausal and S1024 causal pass; MMOP remains 92,160 and LDS
  bank conflicts remain zero.
- Fullperf ticks improve `73,280,025 -> 73,016,580` (`-0.3595%`).
- Useful MMAC active changes `21.809889% -> 21.641706%` (`-0.168183 pp`).
- C0/C1 MMAC+VALU coissue improves from `10.91%/11.24%` to
  `12.79%/12.69%`.
- Transposed matrix-read-to-wait bubble falls from `6.58%` to `5.11%`.
- ABarrier and atomic tail bubbles grow, absorbing nearly all of the local
  schedule gain.

Decision: keep the clean stage split, batched matrix reads, and useful group
ordering as a micro-ticks improvement. It is not an MMAC-active promotion.
The next design gate is ownership lifetime, not another consumer-order tweak.
Detailed evidence is in
`results/fused5_useful_stagger_20260723.md`.
