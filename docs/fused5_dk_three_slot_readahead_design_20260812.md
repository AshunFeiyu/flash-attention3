# Fused5 dK Three-Slot Read-Ahead Design

Status: `EXPERIMENTAL_RESOURCE_GATE`

## Single hypothesis

The accepted dK tail overlaps one next Q/dS panel with the current dK MMAC.
This experiment asks whether issuing two future panels before the first dK
MMAC can remove the repeated read-to-use bubbles without changing the five-GEMM
DAG, LDS layout, ABarrier ledger, or output ownership.

Expected schedule:

```text
time0: read Q0/dS0, Q1/dS1, Q2/dS2 -> wait
time1: dK(Q0,dS0) MMAC while Q3/dS3 reads
time2: dK(Q1,dS1) MMAC
time3: dK(Q2,dS2) MMAC
time4: dK(Q3,dS3) MMAC
```

The extra live state is two additional Q fragments (four matrix fragments
each) and two dS fragments. The first gate is therefore compilation and WDRA
resource metadata, not PMD timing.

Admission gates:

- role-local consumer VGPR remains at or below 204;
- private/scratch/spills remain zero;
- exact MMOP92,160, bank0 and native MLS/BPS + `ds_read_matrix` + MMAC;
- H1/S128 full lifecycle correctness before H1/S1024 timing.

If the role exceeds 204 or spills, reject immediately. If it builds and is
correct but does not lower same-shape ticks, remove it and retain the two-slot
canonical implementation.
