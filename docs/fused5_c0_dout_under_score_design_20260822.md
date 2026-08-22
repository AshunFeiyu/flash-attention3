# C0 dO matrix read under score/softmax

## Target

The canonical H1/S1024 SQTT assigns 2,907 timestamp units, about 12.2% of
C0 issue-gap duration, to repeated matrix-read first-use debt. The candidate
kept all formula, MMAC, LDS, ABarrier, ownership and output paths unchanged and
attempted to issue the current dO packet before dP so score MMAC and probability
VALU could age that packet.

## Candidate A: twelve outstanding transactions

The first schedule retained the canonical next-Q prefetch and additionally
issued current dO before score:

`Q(current) + dO(current) + Q(next) -> lgkmcnt(8) -> score/P -> lgkmcnt(4) -> dP`

It passes S128 causal/non-causal and S1024 full correctness, bank0 and all
resource gates. Actual roles are `9/189/87/182` under WDRA
`16/212/196/88`, with private/spill/scratch zero. Static and dynamic MMOP,
matrix reads, ABarrier and stores remain exact.

It is rejected: fused ticks regress `44,976,750 -> 48,752,795`
(`+8.396%`). `noVALUready` rises `307,486 -> 346,731`; coissue success falls
`25,111 -> 17,835`. Keeping twelve LDS transactions live collapses issue
readiness even though the arithmetic is unchanged.

## Candidate B: strict maximum of eight

The bounded revision waits for current Q first, then issues current dO and
next Q together:

`wait Q(current) -> dO(current) + Q(next) -> score/P -> lgkmcnt(4) -> dP`

This also passes correctness, bank and resource gates with exact dynamic work.
It still regresses fused ticks `45,184,685 -> 47,221,720` (`+4.508%`) and
full lifecycle `+4.112%`. `noVALUready` rises by 13,155 cycles and coissue
success falls by 2,847.

## Conclusion

Decision: `REJECT_C0_DOUT_READ_SCHEDULING_TIER_CANONICAL_RESTORED`.

The canonical order is a deliberate trade: next-Q is issued before the Q
first-use wait, so its latency ages across score, probability, dP and dS.
Hiding current dO without delaying next-Q requires twelve outstanding LDS
transactions and overloads this pipeline; bounding the queue to eight moves
next-Q later and loses more than dO hiding gains. Do not retry another C0
Q/dO read ordering on this tile topology. The next hypothesis must change a
structural producer/consumer publication boundary or useful work per ownership
epoch, not another wait placement.

Evidence:

- `/zys/sb/fa3b/c0_dout_under_score_20260822/paired`
- `/zys/sb/fa3b/c0_dout_under_score_20260822/max8_paired`
