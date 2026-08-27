# C111 dK/dV Four-D-Block Store Batch

## Evidence

C109 role-filtered SQTT shows the largest non-ownership bubbles in both dKV
consumer roles at the terminal matrix-store path:

- consumer0: `matrix_store_32x32_b16 -> s_waitcnt` is 2,804 cycles;
- consumer1: the same pair is 2,828 cycles;
- each role also pays two store epochs because C85 batches only two D32 blocks.

The existing code executes a CTA-wide `s_ebarrier_sync` after every q tile,
dS handoff, dQ partial store, and dKV accumulation has completed. Resident
K/V and both raw Q/dO pages are dead after this rendezvous. C85 reused only the
32 KiB V region; C111 may therefore reuse the contiguous `K + V` 64 KiB
region for the complete D128 dK/dV store scratch.

## Change

Set `kDkvStoreDblocksPerBatch` from two to four. The existing templated store
loop then publishes all four D32 dK/dV blocks before one matrix-store epoch.
No source algorithm, output owner, store count, tile, wave role, matrix
instruction, external ABI, or mainloop ownership token changes.

## Resource Proof

```text
4 D blocks * (dK + dV) * 2 KiB/page * 4 N32 pairs = 64 KiB
```

The scratch begins at `LdsLayout::kKBase` (32 KiB) and ends at 96 KiB. This
covers K and V only after the existing pre-store CTA rendezvous makes both
regions dead. The physical kernel LDS allocation remains 128 KiB.

The first bring-up placed the scratch at `V + Raw1` (64--128 KiB). dQ stayed
correct but dK/dV failed, isolating the error to the store source path. That
high-address layout is not admitted here; C111 uses only the already exercised
32--96 KiB K/V address range.

## Expected Pipeline

```text
time0  all roles finish mainloop/dQ store -> CTA rendezvous
time1  dKV waves publish four D32 dK/dV pages
time2  pair-half0 waves issue 8 matrix stores -> one wait
time3  all roles finish one store epoch
```

C85 executes time1-time3 twice. C111 should remove one pair of CTA-wide store
barriers and one terminal wait boundary per dKV wave.

## Gates

- H1/S128 causal/noncausal and H1/S1024 causal full golden PASS.
- no private segment, spill, scratch, warning, or LDS bank conflict;
- exact five GEMMs and exact MMOP/VALU/mainloop matrix work;
- generated store loop has one epoch with eight matrix-store instructions per
  active pair-half wave;
- repeated same-time H1/S1024 fused and lifecycle ticks must not regress.

Accept a small repeatable gain. If matrix-store queue pressure makes one large
epoch slower, restore C109 and close this store-batching tier.

## Result

Accepted at commit `34d5f39`. Three interleaved stats pairs improve fused and
lifecycle means by `0.910%` and `0.679%`. Same-time fullperf improves by
`0.201%` and `0.135%`, while MMAC active rises by `0.350188 pp` to
`39.792603%`. Correctness, resources, bank conflicts and exact-work gates all
pass. Role-filtered SQTT confirms smaller terminal matrix-store and CTA
rendezvous gaps, so the gain matches the design hypothesis.

The rejected 64--128 KiB bring-up layout is not retained. Its dK/dV failure
does not prove a hardware limit; it is only an unverified address/layout
observation and requires an isolated probe before reuse.
