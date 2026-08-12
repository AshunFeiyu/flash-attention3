#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kWaves = 16;
constexpr int kProducerWaves = 4;
constexpr int kDkvConsumerBegin = 4;
constexpr int kDkvConsumerEnd = 12;
constexpr int kDqPublisherBegin = 12;
constexpr int kRawGenerations = 2;
constexpr int kMatrixRows = 32;
constexpr int kMatrixCols = 32;
constexpr int kMatrixElements = kMatrixRows * kMatrixCols;
constexpr int kMatrixBytes = kMatrixElements * sizeof(__half);
constexpr int kPublisherRegionBytes = kProducerWaves * kMatrixBytes;
constexpr int kRawPageBytes = 2 * kPublisherRegionBytes;
constexpr int kLdsBytes = kRawGenerations * kRawPageBytes;
constexpr int kConsumerWaves = kDkvConsumerEnd - kDkvConsumerBegin;

static_assert(kWaves * kWaveSize == 1024,
              "mixed publisher probe must use one 16-wave CTA");
static_assert(kPublisherRegionBytes == 8192 && kRawPageBytes == 16384 &&
                  kLdsBytes == 32768,
              "raw page geometry changed");

struct Barrier {
    static constexpr int kRawFilled = 0;
    static constexpr int kRawUsed = 1;
    static constexpr int kAllDone = 2;
};

union Fragment {
    ins::Vec8F16 f16x8;
    _Float16 scalar[8];
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x % kWaveSize);
}

__device__ __forceinline__ int raw_page_base(int generation) {
    return generation * kRawPageBytes;
}

__device__ __forceinline__ int source_matrix_base(int generation,
                                                  int wave_local) {
    return (generation * kProducerWaves + wave_local) * kMatrixElements;
}

__device__ __forceinline__ void load_matrix_page(
    const __half* source, __half* lds, int generation, int wave_local,
    int region_offset) {
    const int source_offset = source_matrix_base(generation, wave_local);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, ins::prepare_matrix_src(source + source_offset, kMatrixCols),
        raw_page_base(generation) + region_offset +
            wave_local * kMatrixBytes,
        true);
}

template <int RegionOffset>
__device__ __forceinline__ float read_region_sum(const __half* lds,
                                                 int generation) {
    Fragment fragments[kProducerWaves];
    const __half* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + raw_page_base(generation) +
        RegionOffset);
    ins::ds_read_matrix_32x16_trans_imm4<0, kMatrixBytes,
                                         2 * kMatrixBytes,
                                         3 * kMatrixBytes>(
        base, fragments[0].f16x8, fragments[1].f16x8,
        fragments[2].f16x8, fragments[3].f16x8);
    ins::wait_lgkm(0);

    float sum = 0.0f;
#pragma unroll
    for (int page = 0; page < kProducerWaves; ++page) {
#pragma unroll
        for (int word = 0; word < 8; ++word) {
            sum += static_cast<float>(fragments[page].scalar[word]);
        }
    }
    return sum;
}

template <int RegionOffset>
__device__ __forceinline__ void run_mixed_publisher(
    const __half* source, __half* lds, float* q_sums, float* dout_sums,
    int* stats, int wave) {
    int used_phase = 0;
#pragma unroll
    for (int generation = 0; generation < kRawGenerations; ++generation) {
        if (generation != 0) {
            ins::abarrier_try_wait<false>(Barrier::kRawUsed, used_phase);
        }
        // The producer opens one transaction interval for the whole raw page.
        // The dO publisher joins that interval with its own MLS packet and
        // arrival; it must not issue a second seq for the same generation.
        if (wave == 0) {
            ins::abarrier_seq<false>(Barrier::kRawFilled);
        }
        load_matrix_page(source, lds, generation, wave & 3, RegionOffset);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Barrier::kRawFilled, 1);
        if (lane_id() == 0) {
            atomicAdd(stats + (RegionOffset == 0 ? 0 : 1), 1);
        }
    }
    ins::abarrier_arrive_cnt<false>(Barrier::kAllDone, 1);
    (void)q_sums;
    (void)dout_sums;
}

__device__ __forceinline__ void run_mixed_consumer(
    const __half* lds, float* q_sums, float* dout_sums, int* stats,
    int wave) {
    int filled_phase = 0;
#pragma unroll
    for (int generation = 0; generation < kRawGenerations; ++generation) {
        ins::abarrier_try_wait<false>(Barrier::kRawFilled, filled_phase);
        const float q_sum = read_region_sum<0>(lds, generation);
        const float dout_sum = read_region_sum<kPublisherRegionBytes>(
            lds, generation);
        const int output =
            (generation * kConsumerWaves + (wave - kDkvConsumerBegin)) *
                kWaveSize +
            lane_id();
        q_sums[output] = q_sum;
        dout_sums[output] = dout_sum;
        ins::abarrier_arrive_cnt<false>(Barrier::kRawUsed, 1);
        if (lane_id() == 0) {
            atomicAdd(stats + 2, 1);
        }
    }
    ins::abarrier_arrive_cnt<false>(Barrier::kAllDone, 1);
}

__global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_mixed_raw_publisher_probe_kernel(const __half* __restrict__ q_source,
                                        const __half* __restrict__ dout_source,
                                        float* __restrict__ q_sums,
                                        float* __restrict__ dout_sums,
                                        int* __restrict__ stats) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(32, 128, 128, 32);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());

    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Barrier::kRawFilled, 8);
        __builtin_hcu_s_abarrier_init(Barrier::kRawUsed, 8);
        __builtin_hcu_s_abarrier_init(Barrier::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < kDkvConsumerBegin) {
        __builtin_hcu_s_set_vgpr_size(32);
        run_mixed_publisher<0>(q_source, lds, q_sums, dout_sums, stats,
                               wave);
    } else if (wave < 8) {
        __builtin_hcu_s_set_vgpr_size(128);
        run_mixed_consumer(lds, q_sums, dout_sums, stats, wave);
    } else if (wave < kDqPublisherBegin) {
        __builtin_hcu_s_set_vgpr_size(128);
        run_mixed_consumer(lds, q_sums, dout_sums, stats, wave);
    } else {
        __builtin_hcu_s_set_vgpr_size(32);
        run_mixed_publisher<kPublisherRegionBytes>(
            dout_source, lds, q_sums, dout_sums, stats, wave);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Barrier::kAllDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Barrier::kRawFilled);
        __builtin_hcu_s_abarrier_inv(Barrier::kRawUsed);
        __builtin_hcu_s_abarrier_inv(Barrier::kAllDone);
    }
#else
    (void)q_source;
    (void)dout_source;
    (void)q_sums;
    (void)dout_sums;
    (void)stats;
#endif
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     hipGetErrorString(status));
        std::exit(2);
    }
}

}  // namespace

int main() {
    const int source_values = kRawGenerations * kProducerWaves *
                              kMatrixElements;
    std::vector<__half> q_source(source_values);
    std::vector<__half> dout_source(source_values);
    for (int generation = 0; generation < kRawGenerations; ++generation) {
        for (int wave = 0; wave < kProducerWaves; ++wave) {
            const float q_value = static_cast<float>(1 + generation);
            const float dout_value = static_cast<float>(2 + generation);
            const int base = (generation * kProducerWaves + wave) *
                             kMatrixElements;
            for (int i = 0; i < kMatrixElements; ++i) {
                q_source[base + i] = __float2half(q_value);
                dout_source[base + i] = __float2half(dout_value);
            }
        }
    }

    std::vector<float> q_sums(kRawGenerations * kConsumerWaves * kWaveSize);
    std::vector<float> dout_sums(q_sums.size());
    int host_stats[3] = {};
    __half* q_dev = nullptr;
    __half* dout_dev = nullptr;
    float* q_sums_dev = nullptr;
    float* dout_sums_dev = nullptr;
    int* stats_dev = nullptr;
    check_hip(hipMalloc(&q_dev, q_source.size() * sizeof(__half)),
              "hipMalloc q");
    check_hip(hipMalloc(&dout_dev, dout_source.size() * sizeof(__half)),
              "hipMalloc dout");
    check_hip(hipMalloc(&q_sums_dev, q_sums.size() * sizeof(float)),
              "hipMalloc q sums");
    check_hip(hipMalloc(&dout_sums_dev, dout_sums.size() * sizeof(float)),
              "hipMalloc dout sums");
    check_hip(hipMalloc(&stats_dev, sizeof(host_stats)),
              "hipMalloc stats");
    check_hip(hipMemcpy(q_dev, q_source.data(),
                        q_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(dout_dev, dout_source.data(),
                        dout_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy dout");
    check_hip(hipMemset(q_sums_dev, 0, q_sums.size() * sizeof(float)),
              "clear q sums");
    check_hip(hipMemset(dout_sums_dev, 0,
                        dout_sums.size() * sizeof(float)),
              "clear dout sums");
    check_hip(hipMemset(stats_dev, 0, sizeof(host_stats)), "clear stats");

    hipLaunchKernelGGL(fused5_mixed_raw_publisher_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, q_dev, dout_dev,
                       q_sums_dev, dout_sums_dev, stats_dev);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(q_sums.data(), q_sums_dev,
                        q_sums.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy q sums");
    check_hip(hipMemcpy(dout_sums.data(), dout_sums_dev,
                        dout_sums.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy dout sums");
    check_hip(hipMemcpy(host_stats, stats_dev, sizeof(host_stats),
                        hipMemcpyDeviceToHost),
              "copy stats");

    int q_errors = 0;
    int dout_errors = 0;
    for (int generation = 0; generation < kRawGenerations; ++generation) {
        const float expected_q = 32.0f * static_cast<float>(1 + generation);
        const float expected_dout = 32.0f * static_cast<float>(2 + generation);
        for (int consumer = 0; consumer < kConsumerWaves; ++consumer) {
            for (int lane = 0; lane < kWaveSize; ++lane) {
                const int index =
                    (generation * kConsumerWaves + consumer) * kWaveSize +
                    lane;
                q_errors += std::fabs(q_sums[index] - expected_q) > 1.0e-3f;
                dout_errors +=
                    std::fabs(dout_sums[index] - expected_dout) > 1.0e-3f;
            }
        }
    }
    const bool stats_pass = host_stats[0] == 8 && host_stats[1] == 8 &&
                            host_stats[2] == 16;
    const bool pass = q_errors == 0 && dout_errors == 0 && stats_pass;
    std::printf(
        "fused5_mixed_raw_publisher config waves=16 generations=2 "
        "publishers=4+4 consumers=8 lds_bytes=%d\n",
        kLdsBytes);
    std::printf(
        "fused5_mixed_raw_publisher q_errors=%d dout_errors=%d "
        "producer_done=%d dout_publisher_done=%d consumer_done=%d "
        "stats_pass=%d pass=%d\n",
        q_errors, dout_errors, host_stats[0], host_stats[1], host_stats[2],
        stats_pass ? 1 : 0, pass ? 1 : 0);

    check_hip(hipFree(stats_dev), "free stats");
    check_hip(hipFree(dout_sums_dev), "free dout sums");
    check_hip(hipFree(q_sums_dev), "free q sums");
    check_hip(hipFree(dout_dev), "free dout");
    check_hip(hipFree(q_dev), "free q");
    return pass ? 0 : 3;
}
