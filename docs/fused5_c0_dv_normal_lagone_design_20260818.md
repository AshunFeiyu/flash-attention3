# Fused5 C0 dV Normal-Read Lag-One Design

## Evidence

After promoting `2b6efe5`, xcu attributes the largest C0 bubble family to
`ds_read_matrix_format -> s_waitcnt` (`12,392` cycles on the representative
wave).  C0's four dV panels are now contiguous after dS publication, so the
next panel's dO-normal operand has an entire eight-MMAC island in which to
retire.

## Single Hypothesis

Use two dO-normal fragment buffers.  Prime panel0 once, then for each dV panel
issue the P layout read and the next panel's four dO-normal reads before a tied
`lgkmcnt(4)` first-use wait.  The wait admits current P/current dO while keeping
only next dO outstanding:

```text
read dO0
for panel m:
  write/read Pm; read dO(m+1)
  wait current operands, leave next dO alive
  8 dV MMAC
```

The final panel drains to `lgkmcnt(0)`.  Arithmetic, dS publication, LDS,
ABarrier generations and output ownership remain unchanged.

## Gates

- C0 must remain <=204 VGPR with private/spill/scratch0.
- S128 c0/c1 and S1024 c1 full correctness PASS; bank0.
- Paired S1024 fused ticks improve and xcu reduces C0 normal-read wait without
  increasing writer barrier debt.

## Result

Status: `REJECT_COISSUE_REGRESSION_CANONICAL_RESTORED`.

Static resources remain `9/172/179/86` with no spill/scratch.  S128 c0/c1 and
S1024 correctness pass with bank0.  However paired S1024 fused ticks regress
`45,258,395 -> 45,774,365` (`+1.14%`), MMAC active falls
`34.9557% -> 34.6057%`, and failed coissue rises `21,106 -> 24,120` while
wait-LGKM also rises slightly.  The next dO packet enlarged the outstanding
read/issue competition but did not extend useful overlap.  Restore `2b6efe5`;
the dV normal-read lag-one tier is closed.
