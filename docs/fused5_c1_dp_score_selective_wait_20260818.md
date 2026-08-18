# Fused5 C1 dP/Score Selective Wait

## Hypothesis

Consumer group1 already executes `dP -> score`. Issue the four dO-trans
matrix reads and four Q-trans matrix reads as one FIFO packet, retire only the
older dO packet with `lgkmcnt(4)`, execute the complete eight-MMAC dP island,
then retire Q and execute the score island. The math DAG, output ownership,
ABarrier protocol, and group0 schedule remain unchanged.

## Static Result

Generated ISA has the required order:

```text
ds_read_matrix_trans x4 dO
ds_read_matrix_trans x4 Q
s_waitcnt lgkmcnt(4)
v_mmac x8 dP
s_waitcnt lgkmcnt(0)
v_mmac x8 score
```

The tied wait follows the admitted packet8 probe. Fused5 gates pass with
branch VGPR use `9/178/175/86` inside `16/204/204/88`, metadata
SGPR60/VGPR128, and no private segment or spill.

## Verification

- H1/S128 causal and noncausal full lifecycle: PASS.
- H1/S1024 causal full lifecycle: PASS.
- Exact five GEMMs, MMOP 92,160, LDS bank conflict 0.
- Three interleaved S1024 pairs:
  - control: 46,151,105 / 46,181,590 / 46,797,660 ticks
  - candidate: 45,843,070 / 46,047,365 / 46,137,910 ticks
  - means: 46,376,785 -> 46,009,448, improvement 0.792%
- Exact fullperf comparison:
  - control: 46,149,740 ticks, MMAC active 33.794843%
  - candidate: 46,197,060 ticks, MMAC active 33.981919%
  - fullperf tick difference is noise-sized; MMAC active rises 0.187076 pp.
- XCU candidate versus control:
  - trans-read to wait: 9.16% -> 9.11%
  - FP32 MMAC-to-MMAC gap: 5.95% -> 5.74%
  - FP16 MMAC-to-MMAC gap: 3.28% -> 3.15%
  - ABarrier-to-XOR: 26.96% -> 26.59%

## Decision

`ACCEPT_MICRO_TICKS_AND_ACTIVE`. The repeated stats-only result and SQTT
mechanism agree, while the fullperf tick delta stays inside noise. This is a
schedule primitive, not the 40% MMAC-active breakthrough. The next admitted
hypothesis is dO lag-one: keep the next dO panel in flight across current
softmax/dS/dV, then pair it with current Q at the next dP/score boundary.

