# dot_do_o Packed FP16 Pair Design

Status: `REJECT_NEUTRAL_TICKS_CANONICAL_RESTORED`.

## Trigger

The accepted wave-per-row kernel reduced dot preprocessing from 12.398M to
about 2.44M ticks. As fused5 improved, its H1/S1024 lifecycle share rose to
about 5.4%, slightly above the 5% target. The current ISA issues four scalar
`global_load_ushort` instructions per wave for Q-row inputs:

```text
dout[lane], out[lane], dout[lane+64], out[lane+64]
```

## One Hypothesis

Remap each lane to the contiguous pair `d=2*lane,2*lane+1` and vector-load one
FP16 pair from dO plus one FP16 pair from O. Convert both pairs to FP32 and
compute the same two products before the unchanged six-stage wave reduction.

```text
dout_pair = load32(dout[row, 2*lane : 2*lane+2])
out_pair  = load32(out [row, 2*lane : 2*lane+2])
partial   = fp32(dout_pair[0]) * fp32(out_pair[0])
          + fp32(dout_pair[1]) * fp32(out_pair[1])
delta     = wave_reduce_sum(partial)
```

Every row still covers each D128 element exactly once. Lane0 sidecar loads,
reciprocal, delta and packed stores remain unchanged.

## Resource And Work Ledger

| Item | Accepted A1 | Packed A2 gate |
| --- | ---: | ---: |
| row owner | one wave | exact |
| lanes / elements per lane | 64 / 2 | exact |
| input bytes per row | 512 | exact |
| scalar input loads | 4 ushort sites | 2 dword sites |
| products per row | 128 | exact |
| shuffle/add stages | 6 | exact |
| sidecar loads/stores | 2 / 4 logical values | exact |
| LDS / barrier / WDRA | 0 / 0 / none | exact |
| SGPR/VGPR | 22 / 12 | no growth beyond 32 |
| private/spill/scratch | 0 | exact |

The mapping changes coalescing granularity, not total traffic. Expected benefit
comes from fewer VMEM instructions and one packed conversion path. Reject if
the compiler scalarizes the pair, adds private storage, or changes the six
shuffle instructions.

## Expected Pipeline

```text
time0  two packed dword loads: dO pair and O pair
time1  packed FP16->FP32 conversion and two FMA terms
time2  six ds_bpermute/add reduction stages
time3  lane0 max/sum loads, reciprocal, delta and packed stores
```

## Admission

1. Static ASM must contain two input `global_load_dword` sites and no four-site
   ushort fallback; six `ds_bpermute_b32` remain.
2. Metadata stays SGPR/VGPR <=32 with private/spill/scratch0 and no LDS/barrier.
3. Full S128 causal/noncausal and S1024 causal CPU golden pass with warning0,
   nonfinite0 and bank0.
4. Three interleaved S1024 A/B pairs decide promotion. Absolute dot ticks and
   lifecycle ticks must both improve; fused5 ticks/MMOP must remain unchanged
   within PMD run noise.
5. A winner is checked at S2048. No xcu capture is required unless static and
   PMD counters disagree about the load reduction.

Workbook: sections89-90 in the 2026-08-23 fused5 design workbook.

## Result

Static and resource gates pass. The candidate replaces four input ushort
loads with two dword loads, keeps six `ds_bpermute_b32` stages, and remains
SGPR22/VGPR12 with no private/spill/scratch. Full S128 causal/noncausal,
S1024 causal and S2048 causal lifecycle correctness pass with warning0 and
bank0.

The reduced load count is offset by packed conversion work. Dynamic S1024
FLAT falls `8192 -> 6144`, while VALU rises `71680 -> 74752`. Three
interleaved S1024 pairs improve dot ticks only `0.211%`; three S2048 pairs are
neutral at `0.023%`. The apparent lifecycle changes are dominated by fused5
run noise because that dispatch is source-identical between A/B.

The candidate is rejected and production source is restored. Do not retry
pair packing unless a native packed FP16 dot/conversion path removes the extra
VALU rather than merely exchanging load instructions for conversion
instructions.
