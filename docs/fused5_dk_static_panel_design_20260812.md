# Fused5 dK Static Panel Read

Status: `ACCEPT_MICRO_TICKS_CANONICAL_CANDIDATE`.

## Hypothesis

The accepted dK lag-one conveyor used runtime `m_block` address selection for
panels one through three. The candidate emits four explicit compile-time panel
helpers while preserving the same two-slot read-ahead, MMAC count, ownership,
and waits:

```text
read Q0/dS0 -> wait
read Q1/dS1 -> dK0 -> wait
read Q2/dS2 -> dK1 -> wait
read Q3/dS3 -> dK2 -> wait -> dK3
```

This removes runtime panel address work and regularizes the source islands. It
does not change dK lifetime or release `RawUsed` early.

## Result

H1/S128 and two H1/S1024 full lifecycle runs passed. Static metadata remained
clean at roles `9/163/165/86`, with no private segment, spill, scratch or LDS
bank conflict; exact `MMOP=92160` was preserved. H1/S1024 fused ticks were
`47,651,240` and `47,624,850` versus canonical `47,757,255`; complete
lifecycle ticks were `52,770,900` and `52,746,330`. MMAC active was
`33.349778%` and `33.370119%`, versus canonical `33.152365%`.

The repeat's wait shares were `waitVm=2.245216%`, `waitLgkm=9.802545%`, and
`barrier=15.120667%`; coissue was `21,346/24,516`. This is a stable small
win from compile-time regular panel addresses, not a structural route to
50%. Keep it as the new canonical micro baseline and move the next hypothesis
to ownership/control topology.

Decision: `ACCEPT_MICRO_TICKS_CANONICAL_CANDIDATE`.

## Gates

- exact five GEMMs and canonical 16-wave roles;
- no new barrier ID, LDS byte, or output path;
- no spill/private/scratch and bank conflict zero;
- H1/S128 and H1/S1024 full correctness.
