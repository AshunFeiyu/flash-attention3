# Fused5 Score/dP Pair Read Rejection

Date: 2026-08-12  
Decision: `REJECT_TICKS_AND_WAIT_REGRESSION`

## Hypothesis

Issue the four score matrix reads and four dP matrix reads as one operand
island, then use one `lgkmcnt(0)` before the two MMAC islands.

## Evidence

- H1/S128 and H1/S1024 full lifecycle correctness: PASS.
- Exact MMOP: 92,160; LDS bank conflict: 0.
- No private segment, scratch, SGPR spill, or VGPR spill.
- H1/S1024 candidate: fused `49,330,645` ticks, complete lifecycle
  `54,762,890` ticks, MMAC active `31.995634%`, wait-LGKM `12.648060%`,
  barrier `15.726154%`, coissue `21,999/23,304`.
- Accepted dK read-ahead baseline: fused mean `47,559,633` ticks and MMAC
  active about `33.377%`.

## Conclusion

`ds_read_matrix` batching alone did not create a useful conveyor. The single
wait exposed a longer combined dependency/read-pressure interval and joined
two MMAC islands that should have been available for peer VALU overlap.
Restore the canonical source and do not retry score/dP pairing without a new
ownership or layout proof.
