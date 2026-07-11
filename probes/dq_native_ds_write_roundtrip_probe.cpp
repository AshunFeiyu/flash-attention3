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
constexpr int kMatrixElems = 32 * 16;
constexpr int kResultCount = 16;

enum ResultSlot : int {
    kNormalWriteNormalReadFrag = 0,
    kNormalWriteTransReadFrag = 1,
    kTransWriteNormalReadFrag = 2,
    kTransWriteTransReadFrag = 3,
    kNormalWriteNormalReadMmac = 4,
    kNormalWriteTransReadMmac = 5,
    kTransWriteNormalReadMmac = 6,
    kTransWriteTransReadMmac = 7,
    kNormalWriteTrans16Alt0Frag = 8,
    kNormalWriteTrans16Alt0Mmac = 9,
    kNormalWriteTrans16Alt1Frag = 10,
    kNormalWriteTrans16Alt1Mmac = 11,
    kTransWriteTrans16Alt0Frag = 12,
    kTransWriteTrans16Alt0Mmac = 13,
    kTransWriteTrans16Alt1Frag = 14,
    kTransWriteTrans16Alt1Mmac = 15,
};

__device__ __forceinline__ _Float16 f16_from_bits(uint16_t bits) {
    return *reinterpret_cast<const _Float16*>(&bits);
}

__device__ __forceinline__ uint16_t f16_bits(_Float16 value) {
    return *reinterpret_cast<const uint16_t*>(&value);
}

__device__ __forceinline__ void count_fragment_mismatch(
    const Vec8F16& expected,
    const Vec8F16& actual,
    int* out,
    int slot) {
    const uint16_t* expected_bits =
        reinterpret_cast<const uint16_t*>(&expected);
    const uint16_t* actual_bits = reinterpret_cast<const uint16_t*>(&actual);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (expected_bits[i] != actual_bits[i]) {
            atomicAdd(out + slot, 1);
        }
    }
}

__device__ __forceinline__ void count_mmac_mismatch(
    const Vec8F16& expected_lhs,
    const Vec8F16& actual_lhs,
    const Vec4F16& rhs,
    int* out,
    int slot) {
    Vec4F16 expected_lo{};
    Vec4F16 actual_lo{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        expected_lo[i] = expected_lhs[i];
        actual_lo[i] = actual_lhs[i];
    }
    ins::F32x4 zero{};
    ins::F32x4 expected{};
    ins::F32x4 actual{};
    expected.f32 = ins::mmac_f16_lit(expected_lo, rhs, zero.f32);
    actual.f32 = ins::mmac_f16_lit(actual_lo, rhs, zero.f32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (expected.scalar[i] != actual.scalar[i]) {
            atomicAdd(out + slot, 1);
        }
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dq_native_ds_write_roundtrip_probe_kernel(int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ _Float16 lds[4 * kMatrixElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    Vec8F16 producer_frag{};
    Vec4F16 rhs{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        producer_frag[i] = f16_from_bits(
            static_cast<uint16_t>(0x3400 + lane * 8 + i));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rhs[i] = f16_from_bits(static_cast<uint16_t>(0x3800 + lane * 4 + i));
    }

    // Four HCU-documented 32x16 f16 writer/reader pairings. The probe checks
    // the actual MMAC result, not merely whether LDS values appear populated.
    __builtin_hcu_ds_write_matrix_format_f16(
        producer_frag, lds + 0 * kMatrixElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        producer_frag, lds + 1 * kMatrixElems, 16, 2, 1, 0, 1);
    ins::wait_lgkm(0);

    const Vec8F16 normal_from_normal =
        __builtin_hcu_ds_read_matrix_format_f16(
            lds + 0 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 trans_from_normal =
        __builtin_hcu_ds_read_matrix_trans_format_f16(
            lds + 0 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 normal_from_trans =
        __builtin_hcu_ds_read_matrix_format_f16(
            lds + 1 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 trans_from_trans =
        __builtin_hcu_ds_read_matrix_trans_format_f16(
            lds + 1 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 trans16_alt0_from_normal =
        __builtin_hcu_ds_read_matrix_trans_format_f16(
            lds + 0 * kMatrixElems, 16, 1, 2, 0);
    const Vec8F16 trans16_alt1_from_normal =
        __builtin_hcu_ds_read_matrix_trans_format_f16(
            lds + 0 * kMatrixElems, 16, 1, 2, 1);
    const Vec8F16 trans16_alt0_from_trans =
        __builtin_hcu_ds_read_matrix_trans_format_f16(
            lds + 1 * kMatrixElems, 16, 1, 2, 0);
    const Vec8F16 trans16_alt1_from_trans =
        __builtin_hcu_ds_read_matrix_trans_format_f16(
            lds + 1 * kMatrixElems, 16, 1, 2, 1);
    ins::wait_lgkm(0);

    count_fragment_mismatch(producer_frag, normal_from_normal, out,
                            kNormalWriteNormalReadFrag);
    count_fragment_mismatch(producer_frag, trans_from_normal, out,
                            kNormalWriteTransReadFrag);
    count_fragment_mismatch(producer_frag, normal_from_trans, out,
                            kTransWriteNormalReadFrag);
    count_fragment_mismatch(producer_frag, trans_from_trans, out,
                            kTransWriteTransReadFrag);
    count_mmac_mismatch(producer_frag, normal_from_normal, rhs, out,
                        kNormalWriteNormalReadMmac);
    count_mmac_mismatch(producer_frag, trans_from_normal, rhs, out,
                        kNormalWriteTransReadMmac);
    count_mmac_mismatch(producer_frag, normal_from_trans, rhs, out,
                        kTransWriteNormalReadMmac);
    count_mmac_mismatch(producer_frag, trans_from_trans, rhs, out,
                        kTransWriteTransReadMmac);
    count_fragment_mismatch(producer_frag, trans16_alt0_from_normal, out,
                            kNormalWriteTrans16Alt0Frag);
    count_mmac_mismatch(producer_frag, trans16_alt0_from_normal, rhs, out,
                        kNormalWriteTrans16Alt0Mmac);
    count_fragment_mismatch(producer_frag, trans16_alt1_from_normal, out,
                            kNormalWriteTrans16Alt1Frag);
    count_mmac_mismatch(producer_frag, trans16_alt1_from_normal, rhs, out,
                        kNormalWriteTrans16Alt1Mmac);
    count_fragment_mismatch(producer_frag, trans16_alt0_from_trans, out,
                            kTransWriteTrans16Alt0Frag);
    count_mmac_mismatch(producer_frag, trans16_alt0_from_trans, rhs, out,
                        kTransWriteTrans16Alt0Mmac);
    count_fragment_mismatch(producer_frag, trans16_alt1_from_trans, out,
                            kTransWriteTrans16Alt1Frag);
    count_mmac_mismatch(producer_frag, trans16_alt1_from_trans, rhs, out,
                        kTransWriteTrans16Alt1Mmac);

    // Keep a raw producer fragment live so an optimizing compiler cannot
    // collapse the comparison into a compile-time constant.
    if (lane == 0 && f16_bits(producer_frag[0]) == 0) {
        atomicAdd(out, 1);
    }
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
    int h_out[kResultCount]{};
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset");
    hipLaunchKernelGGL(dq_native_ds_write_roundtrip_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy");
    check_hip(hipFree(d_out), "hipFree");

    const char* names[kResultCount] = {
        "normal_write_normal_read_fragment",
        "normal_write_trans_read_fragment",
        "trans_write_normal_read_fragment",
        "trans_write_trans_read_fragment",
        "normal_write_normal_read_mmac",
        "normal_write_trans_read_mmac",
        "trans_write_normal_read_mmac",
        "trans_write_trans_read_mmac",
        "normal_write_trans16_alt0_fragment",
        "normal_write_trans16_alt0_mmac",
        "normal_write_trans16_alt1_fragment",
        "normal_write_trans16_alt1_mmac",
        "trans_write_trans16_alt0_fragment",
        "trans_write_trans16_alt0_mmac",
        "trans_write_trans16_alt1_fragment",
        "trans_write_trans16_alt1_mmac",
    };
    bool any_candidate_pass = false;
    for (int i = 0; i < kResultCount; ++i) {
        std::printf("%s_mismatch=%d\n", names[i], h_out[i]);
    }
    const int frag_slots[] = {kNormalWriteNormalReadFrag,
                              kNormalWriteTransReadFrag,
                              kTransWriteNormalReadFrag,
                              kTransWriteTransReadFrag,
                              kNormalWriteTrans16Alt0Frag,
                              kNormalWriteTrans16Alt1Frag,
                              kTransWriteTrans16Alt0Frag,
                              kTransWriteTrans16Alt1Frag};
    const int mmac_slots[] = {kNormalWriteNormalReadMmac,
                              kNormalWriteTransReadMmac,
                              kTransWriteNormalReadMmac,
                              kTransWriteTransReadMmac,
                              kNormalWriteTrans16Alt0Mmac,
                              kNormalWriteTrans16Alt1Mmac,
                              kTransWriteTrans16Alt0Mmac,
                              kTransWriteTrans16Alt1Mmac};
    for (int candidate = 0; candidate < 8; ++candidate) {
        if (h_out[frag_slots[candidate]] == 0 &&
            h_out[mmac_slots[candidate]] == 0) {
            any_candidate_pass = true;
        }
    }
    std::printf("native_ds_write_roundtrip_pass=%d\n",
                any_candidate_pass ? 1 : 0);
    return any_candidate_pass ? 0 : 2;
}
