# Fused 5-GEMM Single-dS Publication Gate

## Decision

Status: `ACCEPT_CANONICAL_SINGLE_DS_PUBLICATION_MMAC50_OPEN`.

Keep the existing q-owned score/dP orientation and its FP16-output MMAC
source slots. Do not switch the production kernel to K/V-left score ownership.
The next minimal canonical change is:

1. Compute score, dP, P and dS once.
2. Keep the existing local P writer/read bridge for dV.
3. Retain dS in VGPR instead of writing it to the local scratch page.
4. After the previous final dS generation is free, write dS once to the final
   batch page.
5. Read that same page through the normal view for dK.
6. Read it later through the transposed view for dQ.

This preserves exactly five GEMMs and removes one dS R2S, one dS S2R and the
associated local readiness wait. It does not yet reduce the 115,456-byte LDS
peak because the 16 KiB P scratch remains live.

## Evidence

### K/V-left publication format sweep

- Run:
  `/zys/sb/fa3b/layout_probes_kv_left_sweep/dq_native_ds_dense_sweep_20260723_132254`.
- Twenty native writer/reader combinations were tested at D128.
- All twenty fail dQ publication, with max-absolute error around
  `0.057-0.089` and thousands of mismatches.
- Result: `REJECT_WRITER_READER_OWNERSHIP_FORMAT`.

### K/V-left adjacent-M source-slot join

- Run:
  `/zys/sb/fa3b/layout_probes_kv_left/dq_mpair_20260723_133734`.
- No native pair is exact. The best case still has `448` mismatches:
  MMAC `lit0/lts1`, writer `t1/alt0`, normal `alt1` reader.
- Static resources are SGPR14/VGPR37, no private/spill and bank0.
- Result: `REJECT_MPAIR_SOURCE_SLOT_ABI`.

### q-left FP16 source-slot stress

- Run:
  `/zys/sb/fa3b/layout_probes_d128_f16/dq_native_ds_dense_20260723_134433`.
- Native writer/read transport is exact for score, dP, dS, downstream dK/dQ
  and both normal/transposed views.
- FP16 accumulation versus the FP32 CPU path:
  - score `rel_l2=4.26009e-4`
  - dP `rel_l2=4.09753e-4`
  - causal P `rel_l2=1.16166e-4`
  - causal dS `rel_l2=5.46788e-4`
- Result: `OBSERVE_NUMERIC_CANDIDATE_SINGLE_PUBLISH`.
- Boundary: this is not a full FA golden. S128 and S1024 fused correctness
  remain mandatory before any performance claim.

## Resource and Lifetime Proof

- Startup peak stays at `115,456 B`:
  K/V 64 KiB + Q 16 KiB + dO 16 KiB + P scratch 16 KiB + sidecar 768 B.
- After resident K/V fragments are latched, the dead 64 KiB K/V region is the
  final dS batch page.
- During final publication the same peak remains:
  Q 16 KiB + dO 16 KiB + P scratch 16 KiB + final dS 64 KiB + sidecar 768 B.
- A two-Q-stage design with the current P bridge would require `132,608 B`;
  therefore Q2/dO1 is not admitted until P or dS storage is compacted.

## Promotion Gate

- H1/S128 causal and noncausal correctness pass.
- H1/S1024 causal correctness pass.
- Exact dynamic MMOP remains unchanged.
- Private segment, scratch, SGPR spill and VGPR spill are zero.
- `ldsBankConflict=0`.
- Main matrix path remains native MLS/BPS + `ds_read_matrix` + MMAC.
- Same-shape ticks decrease and SQTT attributes the change to the removed dS
  local bridge rather than shifted ownership debt.

## Canonical Integration Result

- Static build passes at role `8/190/189` inside WDRA `24/240/240`,
  SGPR100/VGPR168, LDS115,456 B, private/spill/scratch0.
- H1/S128 causal and noncausal pass; H1/S1024 causal passes with exact
  `MMOP=92,160`, bank0 and no PMD VGPR warning.
- Fullperf versus the locked same-toolchain control:
  - kernel ticks `103,895,610 -> 102,105,640` (`-1.722854%`)
  - MMAC active `16.480234% -> 16.817606%` (`+0.337372 pp`)
  - dynamic LDS `75,808 -> 73,504`
  - `s_waitcnt` issues `34,976 -> 30,656`
  - ABarrier share `27.558803% -> 23.536976%`
- XCU attributes the remaining top bubble to `RawUsed` barrier id4. The
  producer waits `11,687` cycles in the selected window with `99.99%`
  pipeline bubble; the locked control maximum was `12,395` cycles.
- The hypothesis is accepted because ticks, matrix traffic and exposed
  ownership wait all improve. It is not the 50% target: failed coissue rises
  `6,621 -> 10,224`, VM wait rises `1.768743% -> 3.374652%`, and the combined
  Q/dO `RawUsed` lifetime remains the dominant structural blocker.
- Fullperf:
  `/zys/sb/fa3b/single_ds_fullperf/5gemm_owner_s1024_c1_fullperf_20260723_142126`.
- XCU:
  `/zys/sb/fa3b/xcu_outputs/5gemm_single_ds_s1024_20260723`.
- Shared archive:
  `/共享/shaobo/perf/20260723_142126_fused5_single_ds_publish_h1s1024_sqc7_fullperf`.

Next gate: split Q and dO release without adding an LDS stage. dO is dead
after dV; Q is needed through dK. First test a post-dV Q-fragment latch against
the 240-VGPR role budget, then release the raw page before dK. Do not implement
Q2/dO1 LDS buffering because its current budget is 132,608 B.
