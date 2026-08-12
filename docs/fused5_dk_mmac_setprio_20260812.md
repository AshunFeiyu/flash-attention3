# Fused5 dK MMAC Setprio Audit

Date: 2026-08-12

## Result

The dK read-ahead island was wrapped with `s_setprio 2/0`, matching the
existing dV priority wrapper. H1/S128 and H1/S1024 full lifecycle correctness
passed, and resources stayed spill-free. Three valid H1/S1024 runs produced
fused ticks `48,061,650`, `46,484,620`, `47,679,450` and full ticks
`53,157,195`, `51,551,955`, `52,785,915`. Means were about `47.41M/52.50M`
versus the wait-pruned canonical `47.43M/52.60M`, within model noise.

Parsed MMAC active values were `33.2063%`, `33.4764%`, and `33.0167%`; the
mean is below the canonical 33.49% representative result. SCA and MMOP were
unchanged, and no SQTT `.perf` was available because of the PMD ASTCA warning.

Decision: `REJECT_OBSERVE_NEUTRAL`. Restore the unwrapped dK island. The
priority hint alone does not create useful coissue or a stable tick win.

## Boundary

Keep `s_setprio` around fixed MMAC islands only when Source/SQTT shows a
priority inversion. It is not a substitute for enlarging a useful read/MMAC
island or changing ownership cadence.
