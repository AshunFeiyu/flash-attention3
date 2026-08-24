#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 32;
constexpr int kCols = 32;
constexpr int kDim = 32;
constexpr int kElems = kRows * kCols;
constexpr int kMatrixHalfs = kRows * kDim;
constexpr int kResultPageHalfs = 4096;
constexpr int kLdsHalfs = 2 * kMatrixHalfs + kResultPageHalfs;

union Frag8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__device__ __forceinline__ Frag8 pack_pair(
    ins::Vec4F16 first, ins::Vec4F16 second) {
    Frag8 packed{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed.scalar[i] = first[i];
        packed.scalar[4 + i] = second[i];
    }
    return packed;
}

__device__ __forceinline__ void compute_and_write_half(
    const Frag8& a, const Frag8& b0, const Frag8& b1,
    _Float16* result_lds, int byte_offset) {
    ins::Vec4F16 c0{};
    ins::Vec4F16 c1{};
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        c0 = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
            a.f16x4[k], b0.f16x4[k], c0, 0, 0);
        c1 = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
            a.f16x4[k], b1.f16x4[k], c1, 0, 0);
    }
    const Frag8 packed = pack_pair(c0, c1);
    _Float16* const page = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(result_lds) + byte_offset);
    __builtin_hcu_ds_write_matrix_format_f16(
        packed.f16x8, page, 0, 2, 1, 0, 1);
}

__global__ void __launch_bounds__(2 * kWaveSize, 1)
matrix_store_32x32_two_wave_kernel(
    const _Float16* a, const _Float16* b, _Float16* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) _Float16 lds[kLdsHalfs];
    _Float16* const a_lds = lds;
    _Float16* const b_lds = a_lds + kMatrixHalfs;
    _Float16* const result_lds = b_lds + kMatrixHalfs;
    const int wave = static_cast<int>(threadIdx.x / kWaveSize);

    if (wave == 0) {
        ins::matrix_load_32x32_b16_bps_lds(
            reinterpret_cast<__half*>(a_lds),
            ins::prepare_matrix_src(
                reinterpret_cast<const __half*>(a), kDim),
            0, true);
        ins::matrix_load_32x32_b16_bps_lds(
            reinterpret_cast<__half*>(b_lds),
            ins::prepare_matrix_src(
                reinterpret_cast<const __half*>(b), kDim),
            0, true);
        ins::wait_vbcnt0();
    }
    __syncthreads();

    Frag8 a0{};
    Frag8 a1{};
    Frag8 b0{};
    Frag8 b1{};
    ins::ds_read_matrix_trans_pair(
        reinterpret_cast<const __half*>(a_lds), 0,
        a0.f16x8, a1.f16x8);
    ins::ds_read_matrix_trans_pair(
        reinterpret_cast<const __half*>(b_lds), 0,
        b0.f16x8, b1.f16x8);
    ins::wait_lgkm(0);

    if (wave == 0) {
        compute_and_write_half(a0, b0, b1, result_lds, 0);
    } else {
        compute_and_write_half(a1, b0, b1, result_lds, 1024);
    }
    ins::wait_lgkm(0);
    __syncthreads();

    if (wave == 0) {
        const ins::Vec4U32 dst = ins::prepare_matrix_src(
            reinterpret_cast<const __half*>(output), kCols);
        __builtin_hcu_matrix_store_32x32_b16(
            dst, reinterpret_cast<short*>(result_lds), 0,
            false, false, false, false);
        ins::wait_vmem_lgkm();
    }
#else
    (void)a;
    (void)b;
    (void)output;
#endif
}

float input_a(int row, int dim) {
    return static_cast<float>(((row * 5 + dim * 3) % 11) - 5);
}

float input_b(int row, int dim) {
    return static_cast<float>(((row * 7 + dim * 2) % 13) - 6);
}

}  // namespace

int main() {
    std::vector<_Float16> a(kRows * kDim);
    std::vector<_Float16> b(kRows * kDim);
    std::vector<_Float16> expected(kElems);
    std::vector<_Float16> output(kElems);
    for (int row = 0; row < kRows; ++row) {
        for (int dim = 0; dim < kDim; ++dim) {
            a[row * kDim + dim] = static_cast<_Float16>(input_a(row, dim));
            b[row * kDim + dim] = static_cast<_Float16>(input_b(row, dim));
        }
    }
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            float sum = 0.0f;
            for (int dim = 0; dim < kDim; ++dim) {
                sum += static_cast<float>(a[row * kDim + dim]) *
                       static_cast<float>(b[col * kDim + dim]);
            }
            expected[row * kCols + col] = static_cast<_Float16>(sum);
        }
    }

    _Float16* device_a = nullptr;
    _Float16* device_b = nullptr;
    _Float16* device_output = nullptr;
    check_hip(hipMalloc(&device_a, a.size() * sizeof(_Float16)), "malloc a");
    check_hip(hipMalloc(&device_b, b.size() * sizeof(_Float16)), "malloc b");
    check_hip(hipMalloc(&device_output, output.size() * sizeof(_Float16)),
              "malloc output");
    check_hip(hipMemcpy(device_a, a.data(), a.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy a");
    check_hip(hipMemcpy(device_b, b.data(), b.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy b");

    hipLaunchKernelGGL(matrix_store_32x32_two_wave_kernel,
                       dim3(1), dim3(2 * kWaveSize), 0, 0,
                       device_a, device_b, device_output);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(_Float16),
                        hipMemcpyDeviceToHost), "copy output");

    int mismatches = 0;
    float max_abs = 0.0f;
    int first = -1;
    for (int index = 0; index < kElems; ++index) {
        const float error = std::fabs(
            static_cast<float>(output[index]) -
            static_cast<float>(expected[index]));
        if (error != 0.0f) {
            ++mismatches;
            if (first < 0) first = index;
        }
        if (error > max_abs) max_abs = error;
    }

    check_hip(hipFree(device_output), "free output");
    check_hip(hipFree(device_b), "free b");
    check_hip(hipFree(device_a), "free a");

    std::printf(
        "matrix_store_32x32_two_wave status=%s mismatches=%d "
        "max_abs=%g first_row=%d first_col=%d\n",
        mismatches == 0 ? "PASS" : "FAIL", mismatches, max_abs,
        first < 0 ? -1 : first / kCols,
        first < 0 ? -1 : first % kCols);
    return mismatches == 0 ? 0 : 1;
}
