#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Vec4F16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;

constexpr int kWaveSize = 64;
// `t=1` m32x16 reader uses a 64x16 LDS address footprint even though one
// writer transaction carries a 32x16 f16 fragment.  Keep every candidate page
// 2KB apart so the trans form cannot alias or overrun the next candidate.
constexpr int kReaderPageElems = 64 * 16;
constexpr int kPackCountPerLayout = 4;
constexpr int kPackCount = 2 * kPackCountPerLayout;

__device__ __forceinline__ _Float16 f16_from_bits(uint16_t bits) {
    return *reinterpret_cast<const _Float16*>(&bits);
}

__device__ __forceinline__ ins::Vec4F32 mmac_f16_plain(
    ins::Vec4F16 lhs, ins::Vec4F16 rhs, ins::Vec4F32 acc) {
#if defined(__gfx946__) || defined(__gfx92a__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(lhs, rhs, acc, 0, 0);
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(lhs, rhs, acc);
#endif
}

__device__ __forceinline__ void make_pack_candidates(
    const Vec4F16& m0,
    const Vec4F16& m1,
    Vec8F16* mmajor,
    Vec8F16* component_interleave,
    Vec8F16* even_then_odd,
    Vec8F16* component_even_then_odd) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        (*mmajor)[i] = m0[i];
        (*mmajor)[4 + i] = m1[i];
        (*component_interleave)[2 * i] = m0[i];
        (*component_interleave)[2 * i + 1] = m1[i];
    }
    (*even_then_odd)[0] = m0[0];
    (*even_then_odd)[1] = m0[2];
    (*even_then_odd)[2] = m1[0];
    (*even_then_odd)[3] = m1[2];
    (*even_then_odd)[4] = m0[1];
    (*even_then_odd)[5] = m0[3];
    (*even_then_odd)[6] = m1[1];
    (*even_then_odd)[7] = m1[3];
    (*component_even_then_odd)[0] = m0[0];
    (*component_even_then_odd)[1] = m1[0];
    (*component_even_then_odd)[2] = m0[2];
    (*component_even_then_odd)[3] = m1[2];
    (*component_even_then_odd)[4] = m0[1];
    (*component_even_then_odd)[5] = m1[1];
    (*component_even_then_odd)[6] = m0[3];
    (*component_even_then_odd)[7] = m1[3];
}

__device__ __forceinline__ void count_dq_mmac_mismatch(
    const Vec4F16& expected_m0,
    const Vec4F16& expected_m1,
    const Vec8F16& candidate,
    const Vec4F16& k0,
    const Vec4F16& k1,
    int* out,
    int slot) {
    Vec4F16 candidate_m0{};
    Vec4F16 candidate_m1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        candidate_m0[i] = candidate[i];
        candidate_m1[i] = candidate[4 + i];
    }

    ins::F32x4 expected{};
    ins::F32x4 actual{};
    expected.f32 = ins::mmac_f16_lit(expected_m0, k0, expected.f32);
    expected.f32 = ins::mmac_f16_lit(expected_m1, k1, expected.f32);
    actual.f32 = ins::mmac_f16_lit(candidate_m0, k0, actual.f32);
    actual.f32 = ins::mmac_f16_lit(candidate_m1, k1, actual.f32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (expected.scalar[i] != actual.scalar[i]) {
            atomicAdd(out + slot, 1);
        }
    }
}

// DS_WRITE_MATRIX_FORMAT f16 group4 writes a 32x16 tile from one wave's
// Vec8F16.  The two F32 MMAC outputs therefore have to be adjacent along M,
// not two adjacent N tiles.  This is the exact dS(M,K) -> dQ(M,D) handoff
// shape required by the proposed C_dS/C_dQ ring.
__global__ void __launch_bounds__(kWaveSize, 1)
    dq_native_ds_write_mmac_mpair_probe_kernel(int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ _Float16 lds[kPackCount * kReaderPageElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    Vec4F16 q_m0{};
    Vec4F16 q_m1{};
    Vec4F16 score_k{};
    Vec4F16 dq_k0{};
    Vec4F16 dq_k1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_m0[i] = f16_from_bits(static_cast<uint16_t>(0x3400 + lane * 4 + i));
        q_m1[i] = f16_from_bits(static_cast<uint16_t>(0x3800 + lane * 4 + i));
        score_k[i] = f16_from_bits(static_cast<uint16_t>(0x3a00 + lane * 4 + i));
        dq_k0[i] = f16_from_bits(static_cast<uint16_t>(0x3600 + lane * 4 + i));
        dq_k1[i] = f16_from_bits(static_cast<uint16_t>(0x3e00 + lane * 4 + i));
    }

    ins::F32x4 zero{};
    ins::F32x4 score_lit_m0{};
    ins::F32x4 score_lit_m1{};
    ins::F32x4 score_plain_m0{};
    ins::F32x4 score_plain_m1{};
    score_lit_m0.f32 = ins::mmac_f16_lit(q_m0, score_k, zero.f32);
    score_lit_m1.f32 = ins::mmac_f16_lit(q_m1, score_k, zero.f32);
    score_plain_m0.f32 = mmac_f16_plain(q_m0, score_k, zero.f32);
    score_plain_m1.f32 = mmac_f16_plain(q_m1, score_k, zero.f32);

    Vec4F16 ds_lit_m0{};
    Vec4F16 ds_lit_m1{};
    Vec4F16 ds_plain_m0{};
    Vec4F16 ds_plain_m1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // The real softmax/dS step changes values but keeps MMAC output
        // ownership.  Keep the conversion explicit so the probe isolates the
        // native MMAC-result -> DS-write fragment contract.
        ds_lit_m0[i] = static_cast<_Float16>(score_lit_m0.scalar[i]);
        ds_lit_m1[i] = static_cast<_Float16>(score_lit_m1.scalar[i]);
        ds_plain_m0[i] = static_cast<_Float16>(score_plain_m0.scalar[i]);
        ds_plain_m1[i] = static_cast<_Float16>(score_plain_m1.scalar[i]);
    }
    Vec8F16 pack_lit_mmajor{};
    Vec8F16 pack_lit_component_interleave{};
    Vec8F16 pack_lit_even_then_odd{};
    Vec8F16 pack_lit_component_even_then_odd{};
    Vec8F16 pack_plain_mmajor{};
    Vec8F16 pack_plain_component_interleave{};
    Vec8F16 pack_plain_even_then_odd{};
    Vec8F16 pack_plain_component_even_then_odd{};
    make_pack_candidates(ds_lit_m0, ds_lit_m1, &pack_lit_mmajor,
                         &pack_lit_component_interleave, &pack_lit_even_then_odd,
                         &pack_lit_component_even_then_odd);
    make_pack_candidates(ds_plain_m0, ds_plain_m1, &pack_plain_mmajor,
                         &pack_plain_component_interleave,
                         &pack_plain_even_then_odd,
                         &pack_plain_component_even_then_odd);

    // ISA Delta group4: t=0/alt0 writer directly feeds normal group4 reader.
    // Sweep only the unreported MMAC-F32 -> eight-f16 source packing ABI.
    // This preserves a native matrix path: there is no scalar LDS access or
    // lane-permute instruction in any candidate.
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_lit_mmajor, lds + 0 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_lit_component_interleave, lds + 1 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_lit_even_then_odd, lds + 2 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_lit_component_even_then_odd, lds + 3 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_plain_mmajor, lds + 4 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_plain_component_interleave, lds + 5 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_plain_even_then_odd, lds + 6 * kReaderPageElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        pack_plain_component_even_then_odd, lds + 7 * kReaderPageElems, 16, 2, 1, 0, 0);
    ins::wait_lgkm(0);

    const Vec8F16 lit_mmajor = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 0 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 lit_component_interleave = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 1 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 lit_even_then_odd = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 2 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 lit_component_even_then_odd =
        __builtin_hcu_ds_read_matrix_format_f16(
            lds + 3 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 plain_mmajor = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 4 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 plain_component_interleave =
        __builtin_hcu_ds_read_matrix_format_f16(
            lds + 5 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 plain_even_then_odd = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 6 * kReaderPageElems, 16, 2, 1, 0);
    const Vec8F16 plain_component_even_then_odd =
        __builtin_hcu_ds_read_matrix_format_f16(
            lds + 7 * kReaderPageElems, 16, 2, 1, 0);
    ins::wait_lgkm(0);

    count_dq_mmac_mismatch(ds_lit_m0, ds_lit_m1, lit_mmajor, dq_k0, dq_k1, out,
                           0);
    count_dq_mmac_mismatch(ds_lit_m0, ds_lit_m1, lit_component_interleave,
                           dq_k0, dq_k1,
                           out, 1);
    count_dq_mmac_mismatch(ds_lit_m0, ds_lit_m1, lit_even_then_odd, dq_k0,
                           dq_k1, out, 2);
    count_dq_mmac_mismatch(ds_lit_m0, ds_lit_m1, lit_component_even_then_odd,
                           dq_k0, dq_k1, out, 3);
    count_dq_mmac_mismatch(ds_plain_m0, ds_plain_m1, plain_mmajor, dq_k0,
                           dq_k1, out, 4);
    count_dq_mmac_mismatch(ds_plain_m0, ds_plain_m1,
                           plain_component_interleave, dq_k0, dq_k1, out, 5);
    count_dq_mmac_mismatch(ds_plain_m0, ds_plain_m1, plain_even_then_odd,
                           dq_k0, dq_k1, out, 6);
    count_dq_mmac_mismatch(ds_plain_m0, ds_plain_m1,
                           plain_component_even_then_odd, dq_k0,
                           dq_k1, out, 7);
#else
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace

int main() {
    int* d_out = nullptr;
    int h_out[kPackCount]{};
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset");
    hipLaunchKernelGGL(dq_native_ds_write_mmac_mpair_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy");
    check_hip(hipFree(d_out), "hipFree");

    const char* names[kPackCount] = {
        "lit_mmajor", "lit_component_interleave", "lit_even_then_odd",
        "lit_component_even_then_odd", "plain_mmajor",
        "plain_component_interleave", "plain_even_then_odd",
        "plain_component_even_then_odd",
    };
    bool any_packing_pass = false;
    for (int i = 0; i < kPackCount; ++i) {
        std::printf("%s_dq_mmac_mismatch=%d\n", names[i], h_out[i]);
        any_packing_pass |= h_out[i] == 0;
    }
    std::printf("native_ds_write_mmac_mpair_pass=%d\n",
                any_packing_pass ? 1 : 0);
    return any_packing_pass ? 0 : 2;
}
