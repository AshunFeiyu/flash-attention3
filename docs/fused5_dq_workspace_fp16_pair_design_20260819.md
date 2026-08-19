# Fused5 dQ FP16 Workspace Pair Load

Status: `ACCEPT_KERNEL_LOCAL_SCALE_END_TO_END`

## Hypothesis

The accepted FP16-workspace reducer still attributes 31.75% of issue-bubble
duration to `global_load_dwordx2 -> s_waitcnt`. Its metadata is only
SGPR20/VGPR13, so issuing two adjacent k-tile vectors before first use may
amortize the wait without repeating the FP32 candidate's VGPR25-to-36
pressure regression.

## Boundary And Gates

- Change only the reducer's k-tile loop; preserve its 2D ownership and FP32
  accumulation.
- Require two consecutive `global_load_dwordx2` in ISA before conversion.
- Require private/spill/scratch0, S128 c0/c1 and S1024 correctness, and bank0.
- Compare three alternating S1024 pairs against commit `784546e`.
- Accept only if reducer ticks fall without a full-lifecycle regression.

## Result

- ISA is the intended partial pipeline:
  `load0 -> load1 -> vmcnt(1) -> consume0 -> vmcnt(0) -> consume1`.
- Reducer resources are SGPR20/VGPR19, private/spill/scratch0; correctness
  passes S128 c0/c1, S1024 c1 and S2048 c1; bank0.
- Three alternating S1024 pairs improve reducer mean
  `1,941,333 -> 1,645,887` (`-15.22%`). Full-lifecycle mean is noise-flat
  `49,481,402 -> 49,471,695` (`-0.02%`).
- At S2048, where reduction is a larger fraction of the lifecycle, reducer
  ticks improve `6,450,535 -> 4,208,295` (`-34.76%`) and total ticks improve
  `93,688,140 -> 92,370,915` (`-1.41%`) despite fused-run noise.
- Fresh SQTT reduces the former single-load bubble but still exposes a 20.32%
  first-load readiness gap, 5.13% odd-tail load gap, and 18.70% kernel-start
  argument readiness gap. Further reducer batching is lower priority than the
  fused kernel's ABarrier/LDS-read critical path.
