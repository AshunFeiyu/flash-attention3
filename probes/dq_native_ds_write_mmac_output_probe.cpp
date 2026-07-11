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
    kN0N = 0,
    kN0T = 1,
    kN0T16A0 = 2,
    kN0T16A1 = 3,
    kN1N = 4,
    kN1T = 5,
    kN1T16A0 = 6,
    kN1T16A1 = 7,
    kT0N = 8,
    kT0T = 9,
    kT0T16A0 = 10,
    kT0T16A1 = 11,
    kT1N = 12,
    kT1T = 13,
    kT1T16A0 = 14,
    kT1T16A1 = 15,
};

__device__ __forceinline__ _Float16 f16_from_bits(uint16_t bits) {
    return *reinterpret_cast<const _Float16*>(&bits);
}

__device__ __forceinline__ void count_dq_mmac_mismatch(
    const Vec4F16& expected_ds0,
    const Vec4F16& expected_ds1,
    const Vec8F16& candidate,
    const Vec4F16& k_norm0,
    const Vec4F16& k_norm1,
    int* out,
    int slot) {
    Vec4F16 candidate_ds0{};
    Vec4F16 candidate_ds1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        candidate_ds0[i] = candidate[i];
        candidate_ds1[i] = candidate[4 + i];
    }

    ins::F32x4 zero{};
    ins::F32x4 expected{};
    ins::F32x4 actual{};
    expected.f32 = ins::mmac_f16_lit(expected_ds0, k_norm0, zero.f32);
    expected.f32 = ins::mmac_f16_lit(expected_ds1, k_norm1, expected.f32);
    actual.f32 = ins::mmac_f16_lit(candidate_ds0, k_norm0, zero.f32);
    actual.f32 = ins::mmac_f16_lit(candidate_ds1, k_norm1, actual.f32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (expected.scalar[i] != actual.scalar[i]) {
            atomicAdd(out + slot, 1);
        }
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dq_native_ds_write_mmac_output_probe_kernel(int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ _Float16 lds[4 * kMatrixElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    // This is the producer shape used by C_dS: two MMAC result fragments are
    // converted to fp16 and packed as ds_vec0/ds_vec1 before dQ consumes them.
    Vec4F16 q_frag0{};
    Vec4F16 q_frag1{};
    Vec4F16 score_k0{};
    Vec4F16 score_k1{};
    Vec4F16 dQ_k0{};
    Vec4F16 dQ_k1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_frag0[i] = f16_from_bits(static_cast<uint16_t>(0x3400 + lane * 4 + i));
        q_frag1[i] = f16_from_bits(static_cast<uint16_t>(0x3800 + lane * 4 + i));
        score_k0[i] = f16_from_bits(static_cast<uint16_t>(0x3a00 + lane * 4 + i));
        score_k1[i] = f16_from_bits(static_cast<uint16_t>(0x3c00 + lane * 4 + i));
        dQ_k0[i] = f16_from_bits(static_cast<uint16_t>(0x3600 + lane * 4 + i));
        dQ_k1[i] = f16_from_bits(static_cast<uint16_t>(0x3e00 + lane * 4 + i));
    }

    ins::F32x4 zero{};
    ins::F32x4 score0{};
    ins::F32x4 score1{};
    score0.f32 = ins::mmac_f16_lit(q_frag0, score_k0, zero.f32);
    score1.f32 = ins::mmac_f16_lit(q_frag1, score_k1, zero.f32);

    Vec4F16 ds_vec0{};
    Vec4F16 ds_vec1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // The nonlinearity does not change fragment ownership. Use a finite
        // deterministic fp16 conversion in place of full softmax for layout.
        ds_vec0[i] = static_cast<_Float16>(score0.scalar[i]);
        ds_vec1[i] = static_cast<_Float16>(score1.scalar[i]);
    }
    Vec8F16 producer_ds{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        producer_ds[i] = ds_vec0[i];
        producer_ds[4 + i] = ds_vec1[i];
    }

    __builtin_hcu_ds_write_matrix_format_f16(
        producer_ds, lds + 0 * kMatrixElems, 16, 2, 1, 0, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        producer_ds, lds + 1 * kMatrixElems, 16, 2, 1, 1, 0);
    __builtin_hcu_ds_write_matrix_format_f16(
        producer_ds, lds + 2 * kMatrixElems, 16, 2, 1, 0, 1);
    __builtin_hcu_ds_write_matrix_format_f16(
        producer_ds, lds + 3 * kMatrixElems, 16, 2, 1, 1, 1);
    ins::wait_lgkm(0);

    const Vec8F16 n0n = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 0 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 n0t = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 0 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 n0t16a0 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 0 * kMatrixElems, 16, 1, 2, 0);
    const Vec8F16 n0t16a1 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 0 * kMatrixElems, 16, 1, 2, 1);
    const Vec8F16 n1n = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 1 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 n1t = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 1 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 n1t16a0 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 1 * kMatrixElems, 16, 1, 2, 0);
    const Vec8F16 n1t16a1 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 1 * kMatrixElems, 16, 1, 2, 1);
    const Vec8F16 t0n = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 2 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 t0t = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 2 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 t0t16a0 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 2 * kMatrixElems, 16, 1, 2, 0);
    const Vec8F16 t0t16a1 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 2 * kMatrixElems, 16, 1, 2, 1);
    const Vec8F16 t1n = __builtin_hcu_ds_read_matrix_format_f16(
        lds + 3 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 t1t = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 3 * kMatrixElems, 16, 2, 1, 0);
    const Vec8F16 t1t16a0 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 3 * kMatrixElems, 16, 1, 2, 0);
    const Vec8F16 t1t16a1 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        lds + 3 * kMatrixElems, 16, 1, 2, 1);
    ins::wait_lgkm(0);

    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n0n, dQ_k0, dQ_k1, out, kN0N);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n0t, dQ_k0, dQ_k1, out, kN0T);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n0t16a0, dQ_k0, dQ_k1, out, kN0T16A0);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n0t16a1, dQ_k0, dQ_k1, out, kN0T16A1);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n1n, dQ_k0, dQ_k1, out, kN1N);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n1t, dQ_k0, dQ_k1, out, kN1T);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n1t16a0, dQ_k0, dQ_k1, out, kN1T16A0);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, n1t16a1, dQ_k0, dQ_k1, out, kN1T16A1);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t0n, dQ_k0, dQ_k1, out, kT0N);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t0t, dQ_k0, dQ_k1, out, kT0T);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t0t16a0, dQ_k0, dQ_k1, out, kT0T16A0);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t0t16a1, dQ_k0, dQ_k1, out, kT0T16A1);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t1n, dQ_k0, dQ_k1, out, kT1N);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t1t, dQ_k0, dQ_k1, out, kT1T);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t1t16a0, dQ_k0, dQ_k1, out, kT1T16A0);
    count_dq_mmac_mismatch(ds_vec0, ds_vec1, t1t16a1, dQ_k0, dQ_k1, out, kT1T16A1);
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
    hipLaunchKernelGGL(dq_native_ds_write_mmac_output_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy");
    check_hip(hipFree(d_out), "hipFree");

    const char* names[kResultCount] = {
        "normal_alt0_write_normal_read", "normal_alt0_write_trans_read",
        "normal_alt0_write_trans16_alt0", "normal_alt0_write_trans16_alt1",
        "normal_alt1_write_normal_read", "normal_alt1_write_trans_read",
        "normal_alt1_write_trans16_alt0", "normal_alt1_write_trans16_alt1",
        "trans_alt0_write_normal_read", "trans_alt0_write_trans_read",
        "trans_alt0_write_trans16_alt0", "trans_alt0_write_trans16_alt1",
        "trans_alt1_write_normal_read", "trans_alt1_write_trans_read",
        "trans_alt1_write_trans16_alt0", "trans_alt1_write_trans16_alt1",
    };
    bool any_candidate_pass = false;
    for (int i = 0; i < kResultCount; ++i) {
        std::printf("%s_dq_mmac_mismatch=%d\n", names[i], h_out[i]);
        any_candidate_pass |= h_out[i] == 0;
    }
    std::printf("native_ds_write_mmac_output_pass=%d\n",
                any_candidate_pass ? 1 : 0);
    return any_candidate_pass ? 0 : 2;
}
