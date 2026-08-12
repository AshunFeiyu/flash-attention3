# Fused5 dV dO Read Under P Writer Design

Date: 2026-08-12

Status: `ACCEPT_CANONICAL_READINESS_WIN / MMAC50_OPEN`.

## One Hypothesis

After the accepted raw Q/dO double-page redesign, the dominant ABarrier gaps
belong mostly to the thin producer and dQ writer. The two heavy consumer waves
instead report normal `ds_read_matrix -> s_waitcnt` as their largest local
gap: 18,132 and 18,744 cycles in the representative SIMD window. The current
dV bridge serializes three LDS readiness groups per M16 panel:

```text
write P -> wait0 -> read P -> wait0 -> read dO x4 -> wait0 -> dV MMAC x8
```

Issue the independent dO reads after the P writer and before its readiness
wait. `lgkmcnt(4)` then retires the oldest P writer while preserving up to four
younger dO reads; the P reader is issued only after its writer is ready:

```text
write P -> read dO x4 -> wait4 -> read P -> wait0 -> dV MMAC x8
```

This round changes only dV LDS scheduling. It must not change formulas,
GEMM count, tile, pages, barriers, output ownership, C0/C1 stage order, or the
main native matrix path.

## Dependency Proof

1. `dO` resides in the raw page and is independent of the local P writer.
2. P normal read depends on P writer completion, so it cannot cross `wait4`.
3. With one oldest writer followed by four dO reads, `lgkmcnt(4)` permits at
   most those four younger operations to remain outstanding.
4. The final `lgkmcnt(0)` retires P plus dO reads before dV first use.
5. `RawUsed` remains after dK, so earlier dO reads do not shorten or violate
   raw-page ownership.

## Resource And Pipeline Budget

| Item | Accepted baseline | Candidate | Gate |
|---|---:|---:|---|
| dV LDS ops/panel | writer + P read + dO read4 | unchanged | exact |
| dV waits/panel | 3 | 2 | ASM/dynamic |
| consumer VGPR | 161/163 | at most 204 | no spill/private |
| LDS | 131,072B | 131,072B | exact |
| MMOP | 92,160 | 92,160 | exact |
| barriers | 8 | 8 | unchanged |

C0 still executes `score -> P -> dV -> dP -> dS`; C1 still executes
`dP -> score -> P -> dS -> dV`. The existing useful stagger therefore remains
intact. No new fragment is introduced; the four dO fragments merely become
live before the P normal read instead of after it.

## Expected ISA And Evidence

Expected dV island:

```text
ds_write_matrix_format(P)
ds_read_matrix_format(dO) x4
s_waitcnt lgkmcnt(4)
ds_read_matrix_format(P)
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 x8
```

Promotion requires the compiler to retain this order. A source-only claim is
insufficient. Dynamic LDS and MMOP counts must remain exact; wait instruction
count should fall by four per M16 panel per consumer wave.

## Admission Gates

1. A1/static: canonical source gate, expected ISA, exact LDS/MMOP, no
   private/spill/scratch, role use inside `16/204/204/88`.
2. A2-A5: H1/S128 causal/noncausal and repeated H1/S1024 causal dQ/dK/dV
   correctness; bank0.
3. Performance: compare with accepted means compute50,290,467.5 and
   complete53,207,245. Any repeatable tick reduction is admissible.
4. A6 for a stats winner: xcu must show lower consumer matrix-read/write
   first-use debt without reducing useful C0/C1 MMAC-vs-VALU overlap.

Reject and restore canonical source if `lgkmcnt(4)` does not preserve P writer
readiness, resources grow beyond the WDRA window, correctness fails, or
same-shape ticks regress.

## Evidence And Decision

The candidate passed the existing static/resource gates and repeated S1024
correctness checks. It keeps five GEMMs, MMOP `92,160`, LDS `131,072B`,
bank-conflict `0`, and the admitted role window `16/204/204/88`.

Fullperf candidate versus the accepted raw-page baseline:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| complete ticks | 53,714,570 | 52,466,960 | -2.3227% |
| compute ticks | 50,863,995 | 49,835,240 | -2.0226% |
| MMAC active | 31.3305% | 31.5782% | +0.2477pp |
| xcu duration | 111,792 | 109,528 | -2.024% |
| ABarrier gap | 29.23% | 28.45% | -0.78pp |
| P writer -> wait | 2.64% | 0.69% | -1.95pp |

Same-environment paired stats repeats are smaller but consistent: complete
ticks fall `0.9743%`, MMAC active rises `0.5798pp`, `waitLGKM` falls
`0.8383pp`, and barrier share falls `0.5753pp`. The consumer read waits become
slightly larger, while the former P-writer readiness debt is removed. XCU
also shows the direct MMAC/VALU overlap count remains `1,036/4,096`, but the
two groups become more balanced (`511/525` peer-VALU MMACs instead of
`395/641`). This is a readiness and balance win, not yet a coissue win.

Evidence:

- perf: `/Users/zhangyushun/Documents/Codex/2026-06-08/shaobo-hip-shaobo-demo/work/perf/20260812_072621_fused5_dv_read_under_writer_h1s1024_sqc7/3565581_fused5_dv_read_under_writer_h1s1024_sqc7.perf`
- remote xcu: `/zys/shaobo_runs/fused5_dv_read_under_writer_20260812/xcu`
- remote fullperf: `/zys/shaobo_runs/fused5_dv_read_under_writer_20260812/fullperf`

Promotion is limited to the canonical branch after the source and evidence
commit. The next hypothesis is runtime ownership topology: replace the
post-startup global `KvDsUsed` rendezvous with one completion token per
consumer group. It must preserve the single physical dS page generation and
therefore cannot claim a dS double-buffer benefit.
