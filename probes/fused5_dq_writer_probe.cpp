#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kWaves = 12;
constexpr int kIterations = 8;
constexpr int kDqWaves = 4;
constexpr int kPageFloats = 16 * 128;
constexpr int kPageBytes = kPageFloats * sizeof(float);
constexpr int kLdsBytes = 2 * kPageBytes;
constexpr int kOutputValues = kIterations * kDqWaves * kWaveSize * 8;

struct Barrier {
    static constexpr int kFilled0 = 0;
    static constexpr int kUsed0 = 1;
    static constexpr int kFilled1 = 2;
    static constexpr int kUsed1 = 3;
};

__host__ __device__ __forceinline__ float expected_value(int iteration,
                                                         int d_block,
                                                         int lane,
                                                         int element) {
    return static_cast<float>(iteration * 100000 + d_block * 10000 +
                              lane * 100 + element);
}

__device__ __forceinline__ float* page_ptr(float* lds, int generation) {
    return lds + generation * kPageFloats;
}

__device__ __forceinline__ int fragment_offset(int d_block,
                                               int lane,
                                               int d_half) {
    const int row = lane & 15;
    const int lane_group = lane >> 4;
    // Swizzle each b128 lane group over all 64 banks: row chooses four
    // consecutive banks while d_half/lane_group select disjoint planes.
    return d_block * 512 + d_half * 256 + lane_group * 64 + row * 4;
}

template <int Generation>
__device__ __forceinline__ void publish_dq_fragment(float* lds,
                                                    int iteration,
                                                    int d_block,
                                                    int lane,
                                                    int& used_phase) {
    static_assert(Generation == 0 || Generation == 1);
    constexpr int kFilled =
        Generation == 0 ? Barrier::kFilled0 : Barrier::kFilled1;
    constexpr int kUsed = Generation == 0 ? Barrier::kUsed0 : Barrier::kUsed1;
    if (iteration >= 2) {
        ins::abarrier_try_wait<true>(kUsed, used_phase);
    }
    ins::abarrier_seq<false>(kFilled);

    ins::Vec4F32 fragment0{};
    ins::Vec4F32 fragment1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        fragment0[i] = expected_value(iteration, d_block, lane, i);
        fragment1[i] = expected_value(iteration, d_block, lane, i + 4);
    }
    float* page = page_ptr(lds, Generation);
    *reinterpret_cast<ins::Vec4F32*>(
        page + fragment_offset(d_block, lane, 0)) = fragment0;
    *reinterpret_cast<ins::Vec4F32*>(
        page + fragment_offset(d_block, lane, 1)) = fragment1;
    ins::wait_lgkm(0);
    ins::abarrier_arrive_cnt<false>(kFilled, 1);
}

template <int Generation>
__device__ __forceinline__ void consume_dq_fragment(float* output,
                                                    const float* lds,
                                                    int iteration,
                                                    int d_block,
                                                    int lane,
                                                    int& filled_phase) {
    static_assert(Generation == 0 || Generation == 1);
    constexpr int kFilled =
        Generation == 0 ? Barrier::kFilled0 : Barrier::kFilled1;
    constexpr int kUsed = Generation == 0 ? Barrier::kUsed0 : Barrier::kUsed1;
    ins::abarrier_try_wait<true>(kFilled, filled_phase);

    const float* page = page_ptr(const_cast<float*>(lds), Generation);
    const ins::Vec4F32 fragment0 = *reinterpret_cast<const ins::Vec4F32*>(
        page + fragment_offset(d_block, lane, 0));
    const ins::Vec4F32 fragment1 = *reinterpret_cast<const ins::Vec4F32*>(
        page + fragment_offset(d_block, lane, 1));
    ins::wait_lgkm(0);
    ins::abarrier_arrive_cnt<false>(kUsed, 1);

    const int base =
        ((iteration * kDqWaves + d_block) * kWaveSize + lane) * 8;
    *reinterpret_cast<ins::Vec4F32*>(output + base) = fragment0;
    *reinterpret_cast<ins::Vec4F32*>(output + base + 4) = fragment1;
}

__device__ __forceinline__ void run_dq(float* lds, int d_block, int lane) {
    int used_phase0 = 0;
    int used_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int pair = 0; pair < kIterations / 2; ++pair) {
        publish_dq_fragment<0>(lds, pair * 2, d_block, lane, used_phase0);
        publish_dq_fragment<1>(lds, pair * 2 + 1, d_block, lane, used_phase1);
    }
}

__device__ __forceinline__ void run_writer(float* output,
                                           const float* lds,
                                           int d_block,
                                           int lane) {
    int filled_phase0 = 0;
    int filled_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int pair = 0; pair < kIterations / 2; ++pair) {
        consume_dq_fragment<0>(output, lds, pair * 2, d_block, lane,
                               filled_phase0);
        consume_dq_fragment<1>(output, lds, pair * 2 + 1, d_block, lane,
                               filled_phase1);
    }
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_dq_writer_probe_kernel(float* output) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(64, 16, 64);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) float lds[kLdsBytes / sizeof(float)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Barrier::kFilled0, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kUsed0, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kFilled1, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kUsed1, 4);
    }
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave < 4) {
        __builtin_hcu_s_set_vgpr_size(64);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        run_writer(output, lds, wave, lane);
    } else if (wave < 8) {
        __builtin_hcu_s_set_vgpr_size(16);
    } else {
        __builtin_hcu_s_set_vgpr_size(64);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        run_dq(lds, wave - 8, lane);
    }
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Barrier::kFilled0);
        __builtin_hcu_s_abarrier_inv(Barrier::kUsed0);
        __builtin_hcu_s_abarrier_inv(Barrier::kFilled1);
        __builtin_hcu_s_abarrier_inv(Barrier::kUsed1);
    }
#else
    (void)output;
#endif
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

}  // namespace

int main() {
    float* output_device = nullptr;
    std::vector<float> output(kOutputValues, -1.0f);
    check_hip(hipMalloc(&output_device, output.size() * sizeof(float)),
              "hipMalloc output");
    check_hip(hipMemset(output_device, 0, output.size() * sizeof(float)),
              "hipMemset output");
    hipLaunchKernelGGL(fused5_dq_writer_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, output_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(output.data(), output_device,
                        output.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipFree(output_device), "hipFree output");

    int mismatches = 0;
    int reported = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int d_block = 0; d_block < kDqWaves; ++d_block) {
            for (int lane = 0; lane < kWaveSize; ++lane) {
                const int base =
                    ((iteration * kDqWaves + d_block) * kWaveSize + lane) * 8;
                for (int element = 0; element < 8; ++element) {
                    const float expected =
                        expected_value(iteration, d_block, lane, element);
                    if (output[base + element] != expected) {
                        ++mismatches;
                        if (reported < 32) {
                            std::printf(
                                "mismatch iter=%d dblock=%d lane=%d elem=%d "
                                "actual=%.0f expected=%.0f\n",
                                iteration, d_block, lane, element,
                                output[base + element], expected);
                            ++reported;
                        }
                    }
                }
            }
        }
    }
    std::printf(
        "fused5_dq_writer config waves=12 roles=64/16/64 iterations=%d "
        "pages=2 page_bytes=%d lds_bytes=%d\n",
        kIterations, kPageBytes, kLdsBytes);
    std::printf("fused5_dq_writer mismatches=%d pass=%d\n", mismatches,
                mismatches == 0 ? 1 : 0);
    return mismatches == 0 ? 0 : 3;
}
