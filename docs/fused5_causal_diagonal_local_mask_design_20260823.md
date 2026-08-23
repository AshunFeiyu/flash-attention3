# Fused5 Causal Diagonal Local-Mask Design

## Classification

Instruction/control specialization on the accepted `6f445c0` algorithm and
ownership topology. This is not a new pipeline, tile, role, or ABarrier design.

## Formula Proof

For both remaining causal diagonal tiles, the CTA's global K base cancels from
`krow <= qrow`.

- C0 diagonal: `q_base = k_base`, `n_owner = owner`.
- C1 diagonal: `q_base = k_base + 64`, `n_owner = 4 + owner`.

Therefore both reduce to the same group-local predicate:

```text
owner * 16 + local_k <= m_block * 16 + local_q
```

`local_q/local_k` are the existing native score source-slot coordinates. The
predicate changes no score, P, dS, dV, dK, or dQ value and does not alter the
already-accepted C1 zero-front or C0 second-tile fully-valid specialization.

## Draft / Stress / Revise

Draft: introduce a diagonal-only probability helper using group-local owner,
panel, lane, and score-word coordinates.

Stress:

- The helper must be used only where the Q and K tile bases are equal after
  removing the consumer-group 64-row offset.
- Non-causal and fully-valid calls must retain their current compile-time
  behavior.
- Causal ragged tails, arbitrary local windows, dropout, and a future changed
  M/N ownership map are outside this proof.
- A branchless predication experiment on an older dKV kernel spilled SGPRs;
  this candidate keeps the existing masked select and changes only coordinate
  construction. Current fused5 has SGPR71, so static metadata decides before
  PMD.

Revise: use a compile-time causal mode (`full` or `diagonal`) rather than a
runtime boolean. Do not add a third runtime branch or duplicate the consumer
loop. If generated ISA does not remove global-base/address arithmetic or
reduce SCA/VALU while preserving MMAC/read/wait counts, reject before PMD.

## Resource And Pipeline Ledger

| Item | Accepted | Candidate gate |
| --- | ---: | ---: |
| Tile / roles | M64/N128/D128, 2P2C+writer | unchanged |
| Logical GEMMs | exactly 5 | exact MMOP 88,064 at S1024 causal |
| LDS | 128 KiB | unchanged |
| ABarrier IDs | 12 | unchanged |
| SGPR / VGPR metadata | 71 / 128 | no increase causing spill |
| Role use | 9/173/87/162 | within the same WDRA windows |
| Matrix / global traffic | accepted counts | exact |

Expected schedule:

```text
time0: score MMAC; peer waves execute existing MMAC/VALU work
time1: diagonal local compare + exp/cndmask; no global q/k coordinate chain
time2: dP/dS and unchanged dS publication; writer consumes existing token
```

The gain cap is small because only two diagonal regions remain. Admission is
still justified because the accepted causal pruning exposed ownership and
readiness, and this specialization can shorten C0's remaining diagonal dS
publication without adding a token or transaction.

## Admission Gates

1. Generated ASM preserves MMAC, matrix-read, ABarrier, wait, and store counts.
2. Static SCA/VALU or coordinate instructions must shrink; otherwise stop.
3. No private segment, SGPR/VGPR spill, scratch, or LDS bank conflict.
4. Full cached CPU-golden correctness: S128 causal/non-causal, S1024 causal,
   then S2048 causal.
5. Three interleaved S1024 A/B pairs and two S2048 pairs decide promotion.
6. Fullperf/xcu is admitted only after ticks improve; it must show that the
   removed control does not migrate into larger LGKM or ABarrier debt.

## Static Result

Status: `REJECT_COMPILER_PACKING_REGRESSION_CANONICAL_RESTORED`.

The formula reduction is exact, but the C++ specialization changes the
compiler's vectorization and predicate packing. Compared with accepted
`6f445c0`, the generated symbol moves from SGPR71 to SGPR82 and heavy-role use
from `173/162` to `176/164`. Private/spill/scratch remain zero, so this is not
an allocator failure.

Whole-symbol opcode deltas expose the real rejection:

- `s_and_b64 +116`, `v_cndmask_b32 +116`;
- `v_mul_f32 +142`, `v_fma_mix_f32 +90`;
- compare instructions increase by 114 across the emitted variants;
- `s_waitcnt +33`;
- packed operations shrink (`v_pk_mul_f32 -71`, `v_pk_fma_f32 -45`).

The candidate therefore fails the static admission gate before correctness or
PMD. Canonical production source is restored. Evidence is preserved in local
`static_c79/control.asm` and `static_c79/candidate.asm`; do not retry this C++
boolean/template form. A future diagonal-mask attempt requires a focused
vector ABI/builtin form that proves the packed ISA first.
