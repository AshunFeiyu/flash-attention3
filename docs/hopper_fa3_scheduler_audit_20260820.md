# Hopper FA3 scheduler audit

## Source lock

- Repository: `Dao-AILab/flash-attention`
- Local commit: `b54df166ebb69b896892826014759d09b9c3c9c6`
- Commit date: 2026-07-22
- Primary files: `hopper/tile_scheduler.hpp`,
  `hopper/flash_fwd_launch_template.h`, `hopper/flash_bwd_launch_template.h`,
  `hopper/flash_fwd_kernel_sm90.h`, `hopper/flash_bwd_kernel_sm90.h`

## Forward causal/local scheduling

Hopper forward selects `DynamicPersistentTileScheduler` for fixed-length
causal/local attention. Its implementation states the policy directly:

1. Launch exactly `num_sm` persistent CTAs.
2. Map logical blocks in longest-processing-time-first order.
3. When one CTA finishes a tile, its producer obtains the next global tile
   with `atomicAdd(tile_count_semaphore, 1)`.
4. Transfer the new work index from producer to consumer through two named
   barriers and scheduler shared storage.
5. Restrict scheduling to `(batch,head)` L2 sections. The section size is the
   largest power of two whose K/V working set fits a modeled 32MB L2 budget.

This is true dynamic load balancing: the first SM to become free receives the
longest remaining tile. Static CUDA block-to-SM order is not relied upon.

## Backward causal scheduling

Hopper backward selects `SingleTileBwdLPTScheduler` when `Is_causal`; the
noncausal path uses `SingleTileScheduler`.

The backward scheduler is static rather than persistent:

1. Grid size remains the complete number of `(n_block,head,batch)` work
   items; each CTA processes one work item.
2. `(batch,head)` owners are grouped into an L2 section. The section size is
   the largest power of two whose Q, dO, and FP32 dQ accumulator working set
   fits a modeled 40MB L2 budget.
3. Within each section, all owners of `n_block=0` are dispatched first, then
   `n_block=1`, and so on. For causal BWD, lower `n_block` has the longer Q
   loop, so this is LPT: heavy, then medium, then light.
4. `Deterministic` enables the opposite block direction (`SPT` in the
   template), reflecting an ordering constraint rather than the default
   performance schedule.

The code proves the static mapping and L2 swizzle. It does not state why BWD
does not use the forward persistent queue. dQ accumulation/semaphores,
multi-output ownership, deterministic ordering, and a much heavier CTA
lifecycle are reasonable explanations, but remain inference.

## Relevance to Shaobo

The portable principles are:

- Causal work must be modeled from exact q-loop length, not CTA count.
- Long jobs should start early so the final scheduler tail contains light
  work.
- Equal-work `(b,h)` owners should be contiguous only within a cache-locality
  section, not necessarily across the whole grid.
- Load balance and cache locality are joint objectives; a globally perfect
  permutation can lose by destroying Q/dO/K/V reuse.

The hardware-dependent parts are:

- Hopper forward uses a persistent global atomic queue and named-barrier work
  handoff inside a CTA. Shaobo's current fused5 kernel initializes twelve
  ABarriers and carries four WDRA roles for one logical tile; looping it over
  multiple work items requires an explicit multi-generation lifecycle proof.
- Hopper's 32MB/40MB L2 constants cannot be copied to Shaobo, especially for
  multi-die where cache ownership is per die.

## Candidate schedules for H16/S8192

With 48 CUs and 16 `(b,h)` owners, one dispatch round spans three K tiles.

| Policy | Modeled final CU work | Worst completed-round spread | Comment |
| --- | ---: | ---: | --- |
| Original LPT order | 1344--1430 | 86 | Light tail, fixed CU residue imbalance |
| Heavy/light pairing | 1364--1430 | 66 | Middle/tail problem; no max improvement |
| LPT serpentine | 1384--1390 | 6 | Heavy-to-light globally, lane direction alternates |
| Rotated heavy/middle/light bands | 1386--1388 | 84 | Best final sum, but large transient round imbalance |

The conservative first Shaobo candidate is LPT serpentine. It preserves the
Hopper property that the global tail is light while compensating Shaobo's
fixed CU round assignment. Rotated heavy/middle/light remains a measured
competitor, not an assumed winner: its final static sum is slightly better,
but each round intentionally mixes very different CTA lengths.

## Decision ladder

1. Extend the host planner score from final maximum work to both final maximum
   and completed-round prefix spread.
2. Compare identity LPT, LPT serpentine, and rotated heavy/middle/light on at
   least H16/S1024, H16/S2048, and H16/S8192 model shapes.
3. Keep the best static scheduler only if full-operator ticks improve and the
   SQTT tail matches the model.
4. Consider a 48-CTA persistent work queue only as a separate topology
   experiment. It first needs a focused WDRA/ABarrier multi-work probe and a
   dQ partial ownership design; it must not be layered onto the static branch.

