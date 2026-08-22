# Fused5 cross-group alternate dS page

## Hypothesis

Keep the canonical `P4 + C0x4 + C1x4 + W4` topology and exact five GEMMs.
Rotate the existing 16 KiB alternate dS physical page between C0 and C1,
instead of dedicating both dS generations to C0. This gives both heavy groups
a two-q-tile reuse distance for their base page while preserving the two
independent dKV issuers.

This is not the rejected full-token merge, C0 half publication, C1-first
writer, dead-dO alias, or one-heavy-group N32 topology. LDS, matrix traffic,
MMAC count, output ownership, producer path and writer order remain unchanged.

## Formula and ownership

The tile remains `M64/N128/D128`, 16 waves, with N16 output ownership per dKV
wave and D32 ownership per dQ writer wave:

| GEMM | Formula | MMAC/CTA | Owner |
| --- | --- | ---: | --- |
| score | `Q @ K^T` | 256 | C0/C1 unique N16 |
| dP | `dO @ V^T` | 256 | C0/C1 unique N16 |
| dV | `P^T @ dO` | 256 | C0/C1 unique N16 |
| dK | `dS^T @ Q` | 256 | C0/C1 unique N16 |
| dQ | `dS @ K` | 256 | W unique D32 |

Total work remains 1,280 MMAC per CTA and 92,160 dynamic MMOP in the H1/S1024
causal diagnostic. No score, dP, dS, dK, dV or dQ is duplicated.

## Physical pages and token ledger

The three 16 KiB dS pages are unchanged:

- base0: C0 on odd q tiles;
- base1: C1 on even q tiles;
- alternate: C0 on even q tiles, C1 on odd q tiles.

The alternate physical page has two logical ownership token pairs because its
producer alternates. The next owner waits for the previous owner's complete
event:

| q parity | C0 page | C0 wait before reuse | C1 page | C1 wait before reuse |
| --- | --- | --- | --- | --- |
| even | alternate/C0Alt | prior C1Alt Done | base1 | prior C1Base Done |
| odd | base0 | prior C0Base Done | alternate/C1Alt | prior C0Alt Done |

`Done` remains count 8: four dKV waves arrive after dK has retired from the
page and four writer waves arrive after dQ has retired. `Filled` remains count
4. Barrier IDs rise from 12 to 14; no token is CTA-wide.

## Resource budget

| Resource | Canonical | Candidate | Budget |
| --- | ---: | ---: | ---: |
| startup LDS | 131,072 B | 131,072 B | 131,072 B |
| steady released-K/V use | 58,880 B | 58,880 B | 65,536 B |
| alternate dS bytes | 16,384 B | 16,384 B | 16,384 B |
| ABarrier IDs | 12 | 14 | 16 |
| WDRA windows | 16/204/204/88 | unchanged target | total 512 |
| MMAC | 92,160 | 92,160 | exact five GEMMs |

The implementation may not add another page, matrix transaction, global
transaction, accumulator, permutation or gather. Any spill/private/scratch or
bank conflict rejects the experiment before PMD performance runs.

## Expected pipeline

```text
time0  C0: q0 -> Alt(C0)  | C1: q0 -> Base1 | W: C0Alt dQ -> C1Base dQ
time1  C0: q1 -> Base0    | C1: wait C0Alt, q1 -> Alt(C1)
       W: C0Base dQ -> C1Alt dQ
time2  C0: wait C1Alt, q2 -> Alt(C0) | C1: q2 -> Base1
       W: C0Alt dQ -> C1Base dQ
time3  C0: q3 -> Base0    | C1: wait C0Alt, q3 -> Alt(C1)
       W: C0Base dQ -> C1Alt dQ
```

Canonical C1 waits on its own late writer release every q tile. The candidate
moves the late cross-page wait to C0 every second tile; on that tile C1 can
continue useful base-page work. The critical success condition is lower total
`DqDone` gap time without reducing C0/C1 peer MMAC issue or increasing writer
Filled waits.

## Admission and rejection

Run static gates, S128 causal/noncausal correctness, then same-build paired
H1/S1024 causal PMD runs. Promote to S2048/SQTT only if fused ticks improve.

Reject if any of the following occurs:

- barrier phase/count mismatch, PMD timeout, stale read or correctness error;
- private segment, scratch, spill or LDS bank conflict;
- exact MMOP or dynamic matrix/VMEM/FLAT work changes;
- total `DqDone` ownership debt merely moves from C1 to C0 with no ticks gain;
- coissue/MMAC active falls because C0 becomes the new pacing group.

## Result

Status: `REJECT_NO_CRITICAL_PATH_GAIN_CANONICAL_RESTORED`.

The 14-token schedule compiles at actual role use `9/171/87/168`, SGPR93,
VGPR128, with no private segment, spill or scratch. S128 causal/noncausal and
all three S1024 causal runs pass the complete dot/dK/dV/dQ CPU golden; bank
conflict remains zero. Dynamic MMOP, VALU, LDS, VMEM and FLAT work is exact.

Three same-environment H1/S1024 pairs are mixed: one win and two losses.
Canonical fused mean is `45,123,563` ticks and candidate mean is `45,211,682`
ticks (`+0.195%`). Mean MMAC active falls from `34.723%` to `34.688%`.
Barrier share falls about `0.266 pp`, but wait-LGKM rises about `0.205 pp`,
SCA rises `46,744 -> 46,980`, and coissue success falls about `20,396 ->
19,984` per run. The ownership wait moved without shortening the MMAC critical
path. No S2048 or fullperf capture is admitted; canonical source is restored.
