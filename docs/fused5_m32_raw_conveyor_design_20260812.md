# Fused5 M32 Raw Conveyor Design

Date: 2026-08-12

Status: `REJECT_STATIC_RESOURCE / CANONICAL_RESTORED`.

## Measured Trigger

Accepted commit `197e5d9` removes the dQ atomic tail and exposes the next
ownership bottleneck. Representative same-SIMD SQTT at H1/S1024/D128 causal
shows:

| Role | Repeated wait | Count | Mean gap | Aggregate gap |
|---|---|---:|---:|---:|
| producer waves0-3 | `RawUsed` before next raw packet | 14 steady waits | 5,820 cycles | 81,486 cycles |
| consumer0 waves4-7 | `RawFilled` before next tile | 15 steady waits | 1,510 cycles | 22,653 cycles |
| consumer1 waves8-11 | `RawFilled` before next tile | 15 steady waits | 1,021 cycles | 15,309 cycles |
| dQ writer waves12-15 | first dS group ready | 16 waits | 4,221 cycles | 67,528 cycles |

The old producer waits until both heavy groups finish all four dK panels, then
loads the complete next M64 Q/dO packet. This serializes
`dK(t) -> RawUsed(t) -> Q/dO MLS(t+1) -> RawFilled(t+1)`.

## One Hypothesis

Release and refill the raw page in two M32 halves. For each half, every heavy
consumer reads two Q panels and two dS panels into VGPRs, executes
`s_waitcnt lgkmcnt(0)`, and arrives at `RawUsed`. The producer may then overwrite
that M32 Q/dO/sidecar half for tile `t+1` while consumers execute the 16 dK MMAC
instructions that use only the captured fragments.

No GEMM, arithmetic order, output ownership, LDS address, or role is changed.
The hypothesis is that one extra ordered `RawUsed` generation per tile costs
less than the currently exposed full-packet wait and MLS readiness chain.

## Formula And Exact Work

The five logical GEMMs remain:

```text
score = Q @ K^T
dP    = dO @ V^T
dV   += P^T @ dO
dK   += dS^T @ Q
dQ   += dS @ K
```

M64/N128/D128 remains 256 MMAC per GEMM and 1,280 MMAC per tile. H1/S1024
causal must remain exactly MMOP92,160 across the fused compute dispatch.

## Resource Budget

LDS remains byte-for-byte identical to `197e5d9`:

| Region | Bytes |
|---|---:|
| raw Q | 16,384 |
| raw dO | 16,384 |
| resident K/V, then batch dS | 65,536 |
| P scratch | 16,384 |
| sidecar | 768 |
| total | 115,456 |

The paired dK helper holds two panels simultaneously:

```text
Q fragments:  2 panels * 4 fragments * 4 VGPR = 32 VGPR
dS fragments: 2 panels * 1 fragment  * 4 VGPR =  8 VGPR
increment over one-panel helper                     about 20 VGPR
```

Accepted role use is `8/163/166/86`; the expected heavy-role high-water marks
are about `183/186`, below the fixed `8/200/200/88` WDRA windows and 504
physical guard. Generated metadata and role evidence remain authoritative.

## Ownership Ledger

The seven barrier IDs are unchanged. Only barrier 4 changes cadence:

```text
RawUsed generation 2*t+0: Q/dO rows 0..31 of tile t are captured
RawUsed generation 2*t+1: Q/dO rows 32..63 of tile t are captured
RawFilled generation t+1: both halves of tile t+1 are published
```

An arrival is legal only after consumer `lgkmcnt(0)`, so no next-generation
MLS may overlap an outstanding read from the same LDS half. This is the key
difference from the rejected Q-double A5 integration.

## Expected Pipeline

```text
time0:
  P0       publish raw(t)
  C0/C1    score/dP/P/dV/dS(t)
  WQ       wait dS(t)

time1:
  C0/C1    read Q+dS half0(t), wait LDS, arrive RawUsed(2t)
           execute dK half0: 16 MMAC
  P0       wait RawUsed(2t), MLS Q/dO/sidecar half0(t+1)
  WQ       dQ MMAC(t)

time2:
  C0/C1    read Q+dS half1(t), wait LDS, arrive RawUsed(2t+1)
           execute dK half1: 16 MMAC
  P0       wait RawUsed(2t+1), MLS Q/dO/sidecar half1(t+1)
           publish RawFilled(t+1)
  WQ       partial store(t)

time3:
  C0/C1    consume already-aged raw(t+1)
```

Expected SQTT change: producer `RawUsed` gaps and consumer `RawFilled` gaps
fall; MMAC-vs-VALU overlap need not change in this experiment.

## Gates

1. Source/ASM: one canonical kernel, exactly five GEMMs, MMOP92,160, no atomic,
   no scalar matrix workaround.
2. Resource: fixed WDRA windows pass, LDS115,456B, no private/spill/scratch.
3. Correctness: H1/S128 causal and noncausal, then H1/S1024 causal; bank0.
4. Performance: complete compute+reduction ticks below the `197e5d9` repeated
   mean 58,696,137.5 ticks.
5. SQTT on a promoted candidate: producer `RawUsed` and consumer `RawFilled`
   issue-gap cycles must fall without equivalent vbcnt, matrix first-use, or
   barrier expansion.

If static resource use exceeds the fixed windows or either S128 case fails,
remove the source change and retain this document as `REJECT` evidence.

## Measured Result

The first paired-Q expression used the canonical `8/200/200/88` WDRA windows.
The compiler reported the same branch-role counts as the accepted source, but
kernel metadata contained `private_segment_fixed_size=16` and
`vgpr_spill_count=3`. Delaying dS reads until after `RawUsed` reduced semantic
live state but produced the same spill result.

The only intermediate physical allocation, `8/204/204/88=504`, is illegal on
this compiler because the branch-averaged VGPR size does not meet the target
granularity. The next legal allocation, `8/208/208/88=512`, consumes the full
physical pool and still produces the same 16-byte private segment and three
VGPR spills.

The source therefore fails the resource gate before PMD. No correctness or
performance claim is made. Production kernel and contract are restored
byte-for-byte to accepted commit `197e5d9`; only this evidence remains.
