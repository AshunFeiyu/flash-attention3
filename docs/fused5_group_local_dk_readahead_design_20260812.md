# Fused5 dK Read-Ahead Design

After dS publication, each dKV consumer accumulates `dK=dS^T@Q` over four
M16 panels. The accepted schedule keeps one current and one next Q/dS pair:

```text
time0: read Q0+dS0 -> wait -> dK(M0) MMAC
time1: read Q1+dS1 while dK(M0) MMAC -> wait -> dK(M1) MMAC
time2: read Q2+dS2 while dK(M1) MMAC -> wait -> dK(M2) MMAC
time3: read Q3+dS3 while dK(M2) MMAC -> dK(M3) MMAC -> RawUsed
```

No new ABarrier is required. Existing group-local `DqDone` and
`BatchDsFilled` tokens protect the dS page, while the local read-ahead only
changes operand readiness. Exact MMOP, source layout, bank-zero, correctness,
and no-spill gates remain mandatory.
