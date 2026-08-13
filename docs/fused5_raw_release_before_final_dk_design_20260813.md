# Fused5 Raw Release Before Final dK Island

Status: `REJECT_TICKS_CANONICAL_RESTORED`

## Hypothesis

The canonical dK read-ahead already completes the final raw Q read before the
last dK MMAC island:

```text
read Q0/dS0 -> wait -> read Q1/dS1 -> dK0 -> wait
read Q2/dS2 -> dK1 -> wait -> read Q3/dS3 -> dK2 -> wait -> dK3
```

After the wait immediately following `read Q3/dS3`, no instruction in the
remaining `dK3` island reads the raw Q/dO page. Release `RawUsed(page)` at that
point, then execute the final dK MMAC from VGPR fragments. This lets the
producer refill the raw page while the final useful dK work is running.

## Invariants

- Keep exactly five logical GEMMs and the existing M64/N128/D128 tile.
- Keep MLS/BPS + `ds_read_matrix` + MMAC as the matrix path.
- Do not add a barrier, read a raw page after release, or change dS/dQ
  ownership.
- No additional live fragments: the existing two-slot Q/dS read-ahead is
  unchanged.

## Expected Pipeline

```text
time t:     final Q3/dS3 LDS reads complete
            C0/C1: arrive RawUsed(page)
            P:     may refill the released raw Q/dO/sidecar page
time t+1:   C0/C1: final dK MMAC from already-latched Q3/dS3
            P:     useful MLS/BPS work overlaps final dK
```

## Admission

Static resources, complete H1/S128 and H1/S1024 correctness, exact MMOP and
bank-conflict-free LDS are mandatory. Promote only if repeated H1/S1024 fused
ticks improve and XCU shows a shorter producer `RawUsed` gap without a new
first-use wait or barrier debt.

## Result

Static gates passed with producer/consumer/dQ-writer branch usage `9/174/176/86`,
metadata `private=0`, spill `0`, scratch `0`, and `ldsBankConflict=0`. The
complete S128 and S1024 CPU-golden lifecycle passed (`dot/dKV/dQ=PASS`).

At H1/S1024 the candidate measured `46,957,820` fused ticks versus the
canonical `46,637,955`, a `+0.686%` regression. The early release did not
translate into producer/consumer overlap under the current scheduler. Restore
the canonical release after the final dK island and do not repeat this exact
release point without new SQTT evidence.
