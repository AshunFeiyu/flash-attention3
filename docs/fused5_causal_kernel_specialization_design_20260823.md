# Fused5 Compile-Time Causal Kernel Specialization

Status: `DESIGN_ADMITTED / STATIC_GATE_PENDING`.

## Evidence Trigger

The accepted causal C1 zero-front makes `qi1` C1's first valid dV/dK update.
Trying to express that fact inside the runtime-causal kernel duplicated the
complete q-tile body: static MMAC sites grew `1472 -> 1600`, C1 reached
203/204 VGPR and the symbol spilled. The mathematical seed point is valid;
the runtime-mode code shape is not.

Fresh SQTT also shows that the current causal dispatch pays for a broad
four-role symbol while only a small boundary subset needs causal predicates.
At 128-cycle resolution, 128/472 bins have no heavy-role MMAC and the sampled
CTA spends about 4.16K cycles before its first MMAC. A smaller causal symbol
may reduce branch/front-end debt, but this is an inference until static and
PMD evidence pass.

## One Structural Hypothesis

Keep one templated canonical body and compile two mode-specialized entry
symbols. The host API keeps its existing `causal` argument and selects one
symbol before launch.

```text
causal symbol:
  C0 qi0: diagonal body, first dV/dK MMAC seeds accumulators
  C1 qi0: native zero dS publication, no GEMM
  C1 qi1: diagonal body, first dV/dK MMAC seeds accumulators
  later tiles: unmasked accumulation

noncausal symbol:
  C0/C1 qi0: first dV/dK MMAC seeds accumulators
  later tiles: unmasked accumulation
```

The implementation must not retain two hand-written kernel bodies. Formula,
tile, LDS layout, ABarrier ledger, output ownership and matrix helpers remain
single-source templates.

## Formula And Work Invariants

The logical DAG remains exactly:

```text
score = Q @ K^T
dP    = dO @ V^T
P,dS  = softmax(score), P * (dP-D) * scale
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

There are exactly five logical GEMMs. Causal H1/S1024 must retain dynamic
`MMOP=88,064`; noncausal work must match the accepted noncausal oracle. The
specialization removes only explicit accumulator clears and runtime mode
control. It may not restore causal-invalid MMAC work.

## Resource And ABI Budget

| Item | Accepted | Candidate gate |
| --- | ---: | ---: |
| Tile / roles | M64/N128/D128, 16 waves | exact |
| LDS / ABarrier | 128 KiB / 12 IDs | exact |
| WDRA windows | 16/204/204/88 | exact unless a smaller legal ledger is proven |
| Causal role use | 9/173/87/162 | no growth |
| Metadata | SGPR71/VGPR128 | private/spill/scratch0 for both symbols |
| External API | runtime `causal` argument | unchanged |
| Production source | one canonical body | two generated specializations |

Total binary text may grow because both modes are emitted, but each symbol
must be no larger than the accepted runtime-mode symbol in MMAC/read/wait
sites. A per-symbol body duplication rejects the experiment.

## Expected Pipeline

```text
time0  producer: resident K/V + raw packet publication
       causal C1: zero-front publication, no accumulator clears
time1  C0: score/P/dP/dS and early dS publication
       C1: first valid score/dP/P/dS with native dV/dK MMAC seed
       W : consume G1 first, then G0
time2+ existing raw double-page and dS-generation pipeline, unchanged
```

This does not claim to solve ABarrier ownership by itself. The expected direct
gain is fewer zero moves, branches and front-end instructions; the current
ownership/readiness pipeline remains the next debt if ticks do not improve.

## Admission Gates

1. Static gate recognizes exactly the causal/noncausal specialization pair
   generated from one canonical body.
2. Each symbol preserves exact matrix-read/MMAC/ABarrier/store structure and
   has private/spill/scratch zero.
3. Causal `v_mov_b64` and branch counts fall; no q-tile body duplication.
4. S128 causal and noncausal plus causal S1024/S2048 full CPU golden pass,
   warning0 and bank0.
5. Three paired S1024 and two paired S2048 runs decide promotion. Lower ticks
   are mandatory; raw MMAC-active ratio is interpreted with exact work.
6. A repeatable win receives fresh fullperf/xcu. Otherwise restore the
   accepted source and retain only this evidence.

## Result

Status: `ACCEPT_COMPILE_TIME_MODE_SPECIALIZATION_MMAC50_OPEN`.

- The implementation keeps one templated canonical body and emits exactly two
  entry symbols. The public launch API still takes runtime `causal` and selects
  the matching symbol on the host.
- Causal static ISA shrinks from 1,472 to 1,344 MMAC sites, matrix reads from
  840 to 768, waits from 301 to 268, and `v_mov_b64` sites from 42 to 10.
  Noncausal retains 1,472 MMAC and 840 matrix-read sites but also has only 10
  `v_mov_b64` sites. Both symbols are private/spill/scratch free.
- Resource gates pass at causal roles `9/173/87/162`, SGPR70/VGPR128 and
  noncausal roles `9/173/85/162`, SGPR71/VGPR128. LDS remains 128 KiB.
- Full CPU-golden lifecycle correctness passes S128 causal/noncausal and
  causal S1024/S2048, with warning0, nonfinite0 and bank0.
- Interleaved S1024 A/B fused means improve `41,372,468 -> 40,846,943`
  (`-1.270%`); lifecycle means improve `45,496,588 -> 45,074,348`
  (`-0.928%`). The S2048 pair improves fused `76,976,900 -> 76,047,790`
  (`-1.207%`) and lifecycle `84,637,735 -> 83,486,585` (`-1.360%`).
- Fullperf fused ticks improve `41,686,645 -> 41,167,035` (`-1.247%`).
  MMAC active rises `36.407955% -> 36.579709%`; VALU/SCA fall
  `91,248/37,544 -> 90,032/35,592`; waitVM/LGKM/barrier all fall slightly.
  Dynamic MMOP/LDS/VMEM/FLAT remain `88,064/61,056/1,408/3,616`.
- XCU issues fall `325,952 -> 321,648`. The top ABarrier issue gap remains the
  dominant debt at `22.13%`; transpose-read readiness is `7.10%` and terminal
  ebarrier is `6.52%`. The specialization is promoted, but it does not close
  the ownership/readiness problem.

Evidence:

- Remote experiment:
  `/zys/sb/experiments/fused5_causal_kernel_specialization_20260823_c83`
- Fullperf:
  `/zys/sb/runs/fused5_c83_fullperf/b1_hq1_hkv1_s1024_d128_c1_fullperf_perfonly_20260823_150530`
- Local perf/xcu archive:
  `outputs/019ea61f-c117-76b2-abad-e776092d47a0/c83_fullperf`
