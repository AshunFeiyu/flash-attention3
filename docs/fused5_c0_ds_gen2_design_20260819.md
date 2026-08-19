# Fused5 C0 Two-Generation dS Design

## Evidence

H1/S1024 SQTT identifies the dQ writer as the final CTA pace-setter.  On one
representative SIMD the terminal ebarrier arrival timestamps are consumer0
`89868`, producer `92224`, consumer1 `95316`, and dQ writer `97280`.  The writer
waits twice per q tile.  Its group0 wait is normally exposed for roughly
`1.5K--3.8K` cycles, while the following group1 wait is usually already ready.

The accepted C0-early-publication schedule has exhausted useful work that can
be moved behind one dS publication.  The remaining serialization is physical:
group0 cannot publish q tile `i+1` until the writer releases the single group0
dS page for q tile `i`.

## Single Hypothesis

Give only group0 a second 16 KiB dS page and a separate Filled/Done token pair.
Alternate the two generations by q-tile parity:

```text
time0  C0 publish dS(q0,g0)       W wait/consume q0
time1  C0 publish dS(q1,g1)       W consume q0 then q1
time2  C0 waits Done(g0), reuses  W consume q1
```

Group1 retains its canonical single page and token pair.  Formula, exact five
GEMMs, M64/N128/D128 tile, raw Q/dO pages, matrix layouts, output ownership,
and MMOP count are unchanged.

## Resource And Lifetime Proof

- The alternate C0 page occupies 16 KiB in the released V interval at
  `[VBase+16KiB, VBase+32KiB)`.
- Startup K/V still owns that interval.  All consumers latch resident K/V
  before the producer reuses it, so the steady-state overlay is legal.
- Scratch plus two sidecar pages occupy less than the low 16 KiB of the released
  V interval and do not overlap the alternate page.
- Steady released-K/V reuse is approximately `32KiB batch dS + 8KiB scratch +
  1.5KiB sidecar + 16KiB alternate C0 = 57.5KiB`, below 64 KiB.
- New ABarrier IDs are dedicated to C0 generation1.  This fixes the prior
  rejected experiment, which alternated addresses but incorrectly reused one
  phase/token pair for two outstanding generations.

## Admission Gates

1. Build and fused symbol metadata: no private segment, spill, or scratch.
2. Exact MMOP remains 92,160 at H1/S1024; matrix path remains native.
3. H1/S128 causal and noncausal, then H1/S1024 causal correctness PASS.
4. `ldsBankConflict=0` and no PMD VGPR warning/panic.
5. Paired H1/S1024 fused ticks improve and SQTT shows the writer's group0
   dS-wait bubble shrinking without moving the same debt to group1 or VMEM.

If the extra token control regresses ticks or PMD cannot support barrier IDs
10/11, reject and restore the accepted pair-load branch.

## Result

Status: `ACCEPT_TICKS_SCALE_SQTT`

The first runtime-parity implementation was rejected: dynamic selection around
the heavy body increased SCA/VALU and regressed S1024 fused ticks by `4.84%`.
The admitted implementation instead unrolls a fixed generation0/generation1
pair. Group0 alternates its two physical dS pages and two token pairs at
compile time; group1 remains on its canonical single generation.

Static gates pass with SGPR76/VGPR128 and no private segment, spill, or scratch.
H1/S128 causal and noncausal, H1/S1024 causal, and H1/S2048 causal pass the
cached CPU-golden lifecycle with `ldsBankConflict=0`. Exact dynamic work is
unchanged: S1024 has MMOP92,160/LDS63,872/VMEM1,408 and S2048 has
MMOP348,160/LDS239,360/VMEM4,864.

Three alternating S1024 pairs improve fused mean ticks
`45,347,575 -> 44,799,907` (`-1.208%`) and complete-lifecycle mean ticks
`49,436,357 -> 48,940,558` (`-1.003%`). The S2048 scale gate improves fused
ticks `84,345,170 -> 82,777,695` (`-1.858%`) and lifecycle ticks
`91,961,870 -> 90,170,080` (`-1.948%`).

SQTT confirms the intended ownership mechanism. On the representative SIMD,
the dQ writer's ABarrier bubbles fall `33,219 -> 21,983` cycles (`-33.8%`) and
its terminal ebarrier issue moves `97,280 -> 96,028`. Consumer0 does more work
and reaches the terminal barrier later (`89,868 -> 91,436`), but it no longer
forces the writer to release and reacquire the same group0 page every q tile.
At S2048, wait-LGKM share falls `8.121% -> 7.882%` and barrier share falls
`10.925% -> 10.709%`.

MMAC active decreases slightly at S2048 (`38.692% -> 38.099%`) while exact
MMOP is unchanged. This is accepted because same-work fused and lifecycle
ticks both improve, the gain scales with q-loop length, and SQTT directly
shows the targeted writer wait shrinking. The next optimization must target
the remaining matrix-read/first-use gaps or terminal store tail; do not add a
third dS generation.
