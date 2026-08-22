# Fused5 C1 Q Lag-One Under dV

Status: `REJECT_DV_FIRST_USE_DELAY / CANONICAL_RESTORED`.

## Evidence

The promoted invalid-half prune removes 15.21% of dynamic VALU and raises
S2048 MMAC active by 1.237 pp, but SQTT shows that C1 loses useful peer-VALU
coverage and reaches operand readiness sooner:

- C1 MMAC-with-vector-peer falls from about `664/2048` to `436/2048`.
- C0/C1 256-cycle no-MMAC bins rise `72 -> 84`.
- Dynamic `s_waitcnt` hits rise `18,272 -> 19,712`.
- Consumer role pressure falls to about C0=171 and C1=168 under the 204-VGPR
  WDRA windows.

The current accepted C1 pipeline already prefetches dO-trans(m+1) before
dV(m), but Q-trans(m+1) is not issued until the next loop body. The remaining
VGPR headroom can carry one future Q packet without changing work or LDS.

## Single Hypothesis

Move only the existing Q-trans(m+1) packet to the current dV window. Keep the
maximum live LDS read count at eight by issuing Q only after the existing
partial wait has retired P-normal and dO-normal for current dV:

```text
panel m:
  P writer/read + dO-normal(m) + dO-trans(m+1)
  wait lgkmcnt(4)                 # current P+dO ready; next dO remains
  issue Q-trans(m+1) x4           # total outstanding returns to eight
  dV(m) MMAC x8                   # ages next dO and next Q

panel m+1:
  wait lgkmcnt(4)                 # older dO-trans ready; Q remains
  dP(m+1) MMAC x8
  wait lgkmcnt(0)                 # Q-trans ready
  score(m+1) MMAC x8
```

The first panel keeps the canonical startup sequence. The final panel keeps
the canonical no-next-packet dV helper.

## Invariants

- Exactly five logical GEMMs and dynamic MMOP92,160.
- No new matrix read, global request, LDS byte, ABarrier ID or ownership edge.
- Main matrix path remains MLS/BPS + `ds_read_matrix` + MMAC.
- P/dS source layouts, dQ writer order, raw-page release and output ownership
  remain canonical.
- No empty delay and no dead VALU is reintroduced.

## Resource Budget

| Role | Promoted base | Candidate delta | Expected | Cap |
| --- | ---: | ---: | ---: | ---: |
| Producer | 9 | 0 | 9 | 16 |
| C0 | 171 | 0 | 171 | 204 |
| dQ writer | 87 | 0 | 87 | 88 |
| C1 | 168 | +16 Q packet | <=184 | 204 |

LDS remains exactly 128KiB. Peak outstanding matrix reads remain eight: four
next-dO reads survive the partial wait, then four next-Q reads are issued.

## Time0 / Time1 / Time2

```text
time0  C0 score/P/dP/dS       | C1 dP/score/P/dS
time1  C0 useful MMAC/VALU    | C1 wait4 -> Qnext read4 -> dV MMAC
time2  C0 next panel work     | C1 wait4 -> dP MMAC -> wait0 -> score MMAC
```

Expected result: Q readiness is aged under real dV MMAC, C1 wait/no-MMAC
windows fall, and the useful C0/C1 stagger lost by the VALU prune is restored.

## Gates

1. History de-dup must prove this is not the rejected same-epoch dP+score
   read8, C0 dO-under-score, or whole-Q latch design.
2. ASM must show `wait4 -> Qnext read4 -> dV MMAC`, followed next panel by
   `wait4 -> dP MMAC -> wait0 -> score MMAC`.
3. Roles must fit `16/204/204/88`; private/spill/scratch must remain zero.
4. H1/S128 causal/noncausal and H1/S1024 causal golden correctness PASS;
   MMOP92,160 and bank0.
5. Paired S1024 and S2048 must not regress. SQTT must lower C1 wait/no-MMAC
   debt without increasing RawFilled/RawUsed or global-store tail debt.

If generated ISA moves Q before the partial wait, or S2048 repeats the prior
outstanding-read-pressure regression, reject and restore `b402ffd`.

## History De-dup

This combined schedule is new. Commit `549cf0a` previously accepted a
future-Q lag-one path, but it did not carry future dO across dV. Commit
`a427be9` accepted the current future-dO-across-dV path after commit `2510617`
replaced the earlier Q lookahead with a same-panel dO+Q selective-wait packet.
No historical commit combines both future packets under current dV.

The candidate is also distinct from the rejected C0 dO-under-score schedule:
Q is issued only after the current dV operands retire, so peak outstanding
matrix reads stay at eight rather than twelve. It is distinct from whole-Q
latch because raw-page ownership and release do not move.

## Result

The implementation generated the requested order exactly. C1 used `184/204`
VGPR; other roles were `9/171/87`, metadata remained SGPR87/VGPR128, and
private/spill/scratch were zero. H1/S128 causal and noncausal complete golden
correctness passed with bank0 and exact five-GEMM output.

Three interleaved H1/S1024 pairs all regress:

| Pair | Control fused ticks | Candidate fused ticks | Delta |
| --- | ---: | ---: | ---: |
| 1 | 45,246,110 | 45,847,165 | +1.328% |
| 2 | 45,596,915 | 45,612,385 | +0.034% |
| 3 | 45,461,325 | 45,584,630 | +0.271% |

Means move `45,434,783 -> 45,681,393` (`+0.543%`). MMAC active falls
`34.456% -> 34.285%`; wait-LGKM rises `8.847% -> 9.027%`. Dynamic MMOP,
VALU, SCA, LDS, VMEM and FLAT are exact, and barrier share slightly falls.

The early Q packet does not remove work. Its four read issues sit directly
between the current dV readiness wait and the current dV MMAC, delaying the
critical MMAC island. The saved next-panel Q age is smaller than that issue
delay and also disturbs the accepted C1 cadence. This is a scheduling failure,
not a resource, correctness, compiler or PMD failure.

Reject before S2048/fullperf and restore `b402ffd`. Preserve future-dO under
dV, but do not add future-Q before the current dV MMAC on this topology.

Evidence: `/zys/sb/fa3b/c1_q_lagone_20260822`.
