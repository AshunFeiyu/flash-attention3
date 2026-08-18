# Fused5 Writer dS Panel Lag-One Design

## Evidence and Hypothesis

Canonical `2b6efe5` writer SQTT still spends `19,832` representative cycles in
`ds_read_matrix_trans -> s_waitcnt`.  Each source group has four independent
dS panels, currently consumed serially.  Use two four-fragment buffers: issue
panel `m+1`, wait only for panel `m` with `lgkmcnt(4)`, then execute its eight
dQ MMAC.  This mirrors the accepted dK panel read-ahead and changes no
ownership or arithmetic.

Writer needs one extra four-fragment packet.  Rebudget WDRA from
`16/204/204/88` to `16/188/204/104`; measured C0 use is 172, so the physical
sum remains 512 with explicit branch headroom.

## Gates

- All roles fit their new pools; private/spill/scratch0.
- Full correctness PASS and bank0.
- Same-build S1024 ticks improve; writer trans-read wait falls without raising
  dS-token barrier debt.

## Result

Status: `REJECT_NONCRITICAL_WRITER_WAIT_CANONICAL_RESTORED`.

Roles `9/172/179/100` fit `16/188/204/104`, with no spill/scratch. S128 and
S1024 full correctness pass, bank0. Wait-LGKM falls `7.566% -> 6.999%`, but
S1024 fused ticks regress `45,258,395 -> 46,264,400` (`+2.22%`), MMAC active
falls, and barrier/VMEM shares rise. The writer-local wait was not the CTA
pace setter; restore `2b6efe5` pools and code.
