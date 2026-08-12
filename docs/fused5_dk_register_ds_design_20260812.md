# Fused5 Register dS to dK Probe

Status: `REJECT_CORRECTNESS_CANONICAL_RESTORED`

## Hypothesis

The current dK tail publishes dS to LDS for the dQ writer, then reads the
same dS back with four normal `ds_read_matrix` operations before `dS^T@Q`.
Tri Dao's backward mainloop keeps dS in registers for dK and writes it once
to shared memory only for dQ. Test the same lifetime split in Shaobo:

```text
score -> P -> dV -> dP -> dS(reg)
                  |             |
                  |             +-> dK MMAC with Q read from raw LDS
                  +-> ds_write_matrix once for dQ
```

This is a source-layout contract test, not an assumption that a Shaobo
register fragment is automatically a normal dS reader fragment.

## Gates

- Keep the five-GEMM DAG, M64/N128/D128 tile, roles, LDS map and ABarrier
  tokens unchanged.
- Retain the dS LDS publication and dQ writer path.
- Remove only the dS normal LDS reads on the dK path.
- H1/S128 causal full lifecycle must pass before S1024 timing.
- Exact MMOP92,160, no spill/private/scratch, bank0, and native matrix path
  remain required.

If dK correctness fails, restore the two-slot canonical implementation. A
failure means the produced dS register layout is not the dK MMAC source view;
it does not justify adding permute/gather/bpermute to the production path.

## Result

The candidate compiled with consumer roles `163/165`, lower than the
canonical `176/175`, and passed static no-spill/no-scratch gates. H1/S128
full lifecycle correctness failed only dK: `rel_l2=1.33595` and
`cosine_error=0.847635`; dV and dQ remained correct. This proves the direct
register fragment is not the normal dK source-layout view on the current
Shaobo path. The source was restored without a permutation workaround.
