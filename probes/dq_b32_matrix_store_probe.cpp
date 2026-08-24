#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kElements = kRows * kCols;
constexpr int kLdsBytes = 2 * 1024;
constexpr int kStoreBarrier = 0;
constexpr int kCandidates = 8;

union F32x4 {
    ins::Vec4F32 f32;
    ins::Vec4U32 u32;
    float scalar[4];
};

uint32_t float_bits(float value) {
    union {
        float f32;
        uint32_t u32;
    } bits{value};
    return bits.u32;
}

__device__ __forceinline__ ins::Vec4U32 matrix_dst(float* output) {
    return ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output), kCols);
}

__device__ __forceinline__ void write_u32(int writer_t,
                                          ins::Vec4U32 value,
                                          uint32_t* lds) {
    if (writer_t == 0) {
        __builtin_hcu_ds_write_matrix_format_u32(
            value, lds, 0, 1, 1, 0, 0);
    } else {
        __builtin_hcu_ds_write_matrix_format_u32(
            value, lds, 0, 1, 1, 0, 1);
    }
}

__device__ __forceinline__ void store_b32(int store_mode,
                                          ins::Vec4U32 dst,
                                          uint32_t* lds) {
    switch (store_mode) {
        case 0:
            __builtin_hcu_matrix_store_16x16_b32(
                dst, reinterpret_cast<int*>(lds), 0, false, false, false,
                false);
            break;
        case 1:
            __builtin_hcu_matrix_store_16x16_b32(
                dst, reinterpret_cast<int*>(lds), 0, true, false, false,
                false);
            break;
        case 2:
            __builtin_hcu_matrix_store_16x16_b32(
                dst, reinterpret_cast<int*>(lds), 0, false, true, false,
                false);
            break;
        default:
            __builtin_hcu_matrix_store_16x16_b32(
                dst, reinterpret_cast<int*>(lds), 0, true, true, false,
                false);
            break;
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dq_b32_matrix_store_probe_kernel(float* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) uint32_t lds[kLdsBytes / sizeof(uint32_t)];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int candidate = static_cast<int>(blockIdx.x);
    const int row = lane & 15;
    const int lane_col = (lane >> 4) * 4;

    if (lane == 0) {
        __builtin_hcu_s_abarrier_init(kStoreBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    F32x4 value{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = lane_col + i;
        value.scalar[i] = static_cast<float>(row * kCols + col) + 0.25f;
    }

    write_u32(candidate >> 2, value.u32, lds);
    ins::wait_lgkm(0);

    int store_phase = 0;
    ins::abarrier_seq<false>(kStoreBarrier);
    store_b32(candidate & 3,
              matrix_dst(output + candidate * kElements), lds);
    ins::abarrier_arrive_cnt<false>(kStoreBarrier, 1);
    ins::abarrier_try_wait<false>(kStoreBarrier, store_phase);
    ins::wait_vmem_lgkm();

    __builtin_hcu_s_ebarrier_sync(0);
    if (lane == 0) {
        __builtin_hcu_s_abarrier_inv(kStoreBarrier);
    }
#else
    (void)output;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
        std::exit(2);
    }
}

}  // namespace

int main() {
    std::vector<float> output(kCandidates * kElements, -1.0f);
    float* output_device = nullptr;
    check_hip(hipMalloc(&output_device, output.size() * sizeof(float)),
              "hipMalloc output");
    check_hip(hipMemcpy(output_device, output.data(),
                        output.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "initialize output");

    hipLaunchKernelGGL(dq_b32_matrix_store_probe_kernel,
                       dim3(kCandidates), dim3(kWaveSize), 0, 0,
                       output_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(output.data(), output_device,
                        output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipFree(output_device), "hipFree output");

    int passing = 0;
    for (int candidate = 0; candidate < kCandidates; ++candidate) {
        int mismatches = 0;
        int first = -1;
        for (int i = 0; i < kElements; ++i) {
            const float expected = static_cast<float>(i) + 0.25f;
            const float actual = output[candidate * kElements + i];
            if (float_bits(actual) != float_bits(expected)) {
                ++mismatches;
                if (first < 0) {
                    first = i;
                }
            }
        }
        passing += mismatches == 0;
        std::printf(
            "dq_b32_matrix_store writer_t=%d store_t=%d store_r=%d "
            "mismatches=%d first=%d pass=%d\n",
            candidate >> 2, candidate & 1, (candidate >> 1) & 1,
            mismatches, first, mismatches == 0 ? 1 : 0);
    }
    constexpr int kMapCandidate = 4;
    for (int row = 0; row < kRows; ++row) {
        std::printf("dq_b32_slot_map row=%d", row);
        for (int col = 0; col < kCols; ++col) {
            const float value =
                output[kMapCandidate * kElements + row * kCols + col];
            std::printf(",%d",
                        static_cast<int>(value - 0.25f));
        }
        std::printf("\n");
    }
    std::printf("dq_b32_matrix_store passing=%d/%d\n", passing,
                kCandidates);
    return passing == 0 ? 1 : 0;
}
