#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kWaves = 16;
constexpr int kProducerEnd = 4;
constexpr int kConsumer0End = 8;
constexpr int kConsumer1End = 12;
constexpr int kPages = 2;
constexpr int kGenerations = 3;
constexpr int kMatrixElements = 32 * 32;
constexpr int kMatrixBytes = kMatrixElements * sizeof(__half);
constexpr int kRegionBytes = 4 * kMatrixBytes;
constexpr int kPageBytes = 2 * kRegionBytes;
constexpr int kLdsBytes = kPages * kPageBytes;
constexpr int kConsumers = kConsumer1End - kProducerEnd;

static_assert(kWaves * kWaveSize == 1024,
              "single producer probe must use one 16-wave CTA");
static_assert(kLdsBytes == 32768,
              "single producer probe must keep two 16KB pages");

struct Barrier {
    static constexpr int kQFilled0 = 0;
    static constexpr int kQFilled1 = 1;
    static constexpr int kDoutFilled0 = 2;
    static constexpr int kDoutFilled1 = 3;
    static constexpr int kQUsed0 = 4;
    static constexpr int kQUsed1 = 5;
    static constexpr int kDoutUsed0 = 6;
    static constexpr int kDoutUsed1 = 7;
    static constexpr int kAllDone = 8;
};

union Fragment {
    ins::Vec8F16 f16x8;
    _Float16 scalar[8];
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x % kWaveSize);
}

__device__ __forceinline__ int page_base(int page) {
    return page * kPageBytes;
}

__device__ __forceinline__ int source_base(int generation, int wave_local) {
    return (generation * 4 + wave_local) * kMatrixElements;
}

template <int BarrierId>
__device__ __forceinline__ void wait_token(int& phase) {
    ins::abarrier_try_wait<false>(BarrierId, phase);
}

template <int BarrierId>
__device__ __forceinline__ void seq_token() {
    ins::abarrier_seq<false>(BarrierId);
}

template <int BarrierId>
__device__ __forceinline__ void arrive_token() {
    ins::abarrier_arrive_cnt<false>(BarrierId, 1);
}

template <int RegionOffset>
__device__ __forceinline__ void load_region(const __half* source, __half* lds,
                                            int generation, int wave_local) {
    ins::matrix_load_32x32_b16_bps_lds(
        lds,
        ins::prepare_matrix_src(source + source_base(generation, wave_local),
                                32),
        page_base(generation & 1) + RegionOffset + wave_local * kMatrixBytes,
        true);
}

template <int RegionOffset>
__device__ __forceinline__ float read_region(const __half* lds, int page) {
    Fragment fragment[4];
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + page_base(page) + RegionOffset);
    ins::ds_read_matrix_32x16_trans_imm4<0, kMatrixBytes, 2 * kMatrixBytes,
                                         3 * kMatrixBytes>(
        base, fragment[0].f16x8, fragment[1].f16x8, fragment[2].f16x8,
        fragment[3].f16x8);
    ins::wait_lgkm(0);
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int word = 0; word < 8; ++word) {
            result += static_cast<float>(fragment[i].scalar[word]);
        }
    }
    return result;
}

__device__ __forceinline__ void run_single_producer(const __half* q_source,
                                                    const __half* dout_source,
                                                    __half* lds, int wave) {
    int q_used_phase[kPages] = {};
    int dout_used_phase[kPages] = {};
#pragma unroll
    for (int generation = 0; generation < kGenerations; ++generation) {
        const int page = generation & 1;
        // dO dies before Q. Refill only the dO half first, then wait for Q's
        // final dK read before refilling the Q half and publishing the pair.
        if (generation >= kPages) {
            if (page == 0) {
                wait_token<Barrier::kDoutUsed0>(dout_used_phase[page]);
            } else {
                wait_token<Barrier::kDoutUsed1>(dout_used_phase[page]);
            }
        }
        if (wave == 0) {
            if (page == 0) {
                seq_token<Barrier::kDoutFilled0>();
            } else {
                seq_token<Barrier::kDoutFilled1>();
            }
        }
        load_region<kRegionBytes>(dout_source, lds, generation, wave);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        if (page == 0) {
            arrive_token<Barrier::kDoutFilled0>();
        } else {
            arrive_token<Barrier::kDoutFilled1>();
        }

        if (generation >= kPages) {
            if (page == 0) {
                wait_token<Barrier::kQUsed0>(q_used_phase[page]);
            } else {
                wait_token<Barrier::kQUsed1>(q_used_phase[page]);
            }
        }
        if (wave == 0) {
            if (page == 0) {
                seq_token<Barrier::kQFilled0>();
            } else {
                seq_token<Barrier::kQFilled1>();
            }
        }
        load_region<0>(q_source, lds, generation, wave);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        if (page == 0) {
            arrive_token<Barrier::kQFilled0>();
        } else {
            arrive_token<Barrier::kQFilled1>();
        }
    }
    arrive_token<Barrier::kAllDone>();
}

__device__ __forceinline__ void run_consumer(const __half* lds, float* q_sum,
                                             float* dout_sum, int wave) {
    int q_filled_phase[kPages] = {};
    int dout_filled_phase[kPages] = {};
#pragma unroll
    for (int generation = 0; generation < kGenerations; ++generation) {
        const int page = generation & 1;
        if (page == 0) {
            wait_token<Barrier::kQFilled0>(q_filled_phase[page]);
            wait_token<Barrier::kDoutFilled0>(dout_filled_phase[page]);
        } else {
            wait_token<Barrier::kQFilled1>(q_filled_phase[page]);
            wait_token<Barrier::kDoutFilled1>(dout_filled_phase[page]);
        }

        const float dout = read_region<kRegionBytes>(lds, page);
        if (page == 0) {
            arrive_token<Barrier::kDoutUsed0>();
        } else {
            arrive_token<Barrier::kDoutUsed1>();
        }
        const float q = read_region<0>(lds, page);
        if (page == 0) {
            arrive_token<Barrier::kQUsed0>();
        } else {
            arrive_token<Barrier::kQUsed1>();
        }

        const int output =
            (generation * kConsumers + (wave - kProducerEnd)) * kWaveSize +
            lane_id();
        q_sum[output] = q;
        dout_sum[output] = dout;
    }
    arrive_token<Barrier::kAllDone>();
}

__device__ __forceinline__ void run_dq_writer_role() {
    arrive_token<Barrier::kAllDone>();
}

__global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_raw_q_dout_single_producer_probe_kernel(
    const __half* __restrict__ q_source,
    const __half* __restrict__ dout_source, float* __restrict__ q_sum,
    float* __restrict__ dout_sum) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(32, 64, 64, 32);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Barrier::kQFilled0, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kQFilled1, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kDoutFilled0, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kDoutFilled1, 4);
        __builtin_hcu_s_abarrier_init(Barrier::kQUsed0, 8);
        __builtin_hcu_s_abarrier_init(Barrier::kQUsed1, 8);
        __builtin_hcu_s_abarrier_init(Barrier::kDoutUsed0, 8);
        __builtin_hcu_s_abarrier_init(Barrier::kDoutUsed1, 8);
        __builtin_hcu_s_abarrier_init(Barrier::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < kProducerEnd) {
        __builtin_hcu_s_set_vgpr_size(32);
        run_single_producer(q_source, dout_source, lds, wave);
    } else if (wave < kConsumer0End) {
        __builtin_hcu_s_set_vgpr_size(64);
        run_consumer(lds, q_sum, dout_sum, wave);
    } else if (wave < kConsumer1End) {
        __builtin_hcu_s_set_vgpr_size(64);
        run_consumer(lds, q_sum, dout_sum, wave);
    } else {
        __builtin_hcu_s_set_vgpr_size(32);
        run_dq_writer_role();
    }

    int done_phase = 0;
    wait_token<Barrier::kAllDone>(done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Barrier::kQFilled0);
        __builtin_hcu_s_abarrier_inv(Barrier::kQFilled1);
        __builtin_hcu_s_abarrier_inv(Barrier::kDoutFilled0);
        __builtin_hcu_s_abarrier_inv(Barrier::kDoutFilled1);
        __builtin_hcu_s_abarrier_inv(Barrier::kQUsed0);
        __builtin_hcu_s_abarrier_inv(Barrier::kQUsed1);
        __builtin_hcu_s_abarrier_inv(Barrier::kDoutUsed0);
        __builtin_hcu_s_abarrier_inv(Barrier::kDoutUsed1);
        __builtin_hcu_s_abarrier_inv(Barrier::kAllDone);
    }
#else
    (void)q_source;
    (void)dout_source;
    (void)q_sum;
    (void)dout_sum;
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
    const int source_values = kGenerations * 4 * kMatrixElements;
    std::vector<__half> q_source(source_values);
    std::vector<__half> dout_source(source_values);
    for (int generation = 0; generation < kGenerations; ++generation) {
        for (int wave = 0; wave < 4; ++wave) {
            const int base = (generation * 4 + wave) * kMatrixElements;
            for (int i = 0; i < kMatrixElements; ++i) {
                q_source[base + i] = __float2half(1.0f + generation);
                dout_source[base + i] = __float2half(2.0f + generation);
            }
        }
    }
    const int output_values = kGenerations * kConsumers * kWaveSize;
    std::vector<float> q_sum(output_values);
    std::vector<float> dout_sum(output_values);
    __half* q_dev = nullptr;
    __half* dout_dev = nullptr;
    float* q_sum_dev = nullptr;
    float* dout_sum_dev = nullptr;
    check_hip(hipMalloc(&q_dev, q_source.size() * sizeof(__half)), "malloc q");
    check_hip(hipMalloc(&dout_dev, dout_source.size() * sizeof(__half)),
              "malloc dout");
    check_hip(hipMalloc(&q_sum_dev, q_sum.size() * sizeof(float)),
              "malloc q sum");
    check_hip(hipMalloc(&dout_sum_dev, dout_sum.size() * sizeof(float)),
              "malloc dout sum");
    check_hip(hipMemcpy(q_dev, q_source.data(),
                        q_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(dout_dev, dout_source.data(),
                        dout_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy dout");
    check_hip(hipMemset(q_sum_dev, 0, q_sum.size() * sizeof(float)),
              "clear q sum");
    check_hip(hipMemset(dout_sum_dev, 0, dout_sum.size() * sizeof(float)),
              "clear dout sum");
    hipLaunchKernelGGL(fused5_raw_q_dout_single_producer_probe_kernel,
                       dim3(1), dim3(kWaves * kWaveSize), 0, 0, q_dev,
                       dout_dev, q_sum_dev, dout_sum_dev);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(q_sum.data(), q_sum_dev,
                        q_sum.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy q sum");
    check_hip(hipMemcpy(dout_sum.data(), dout_sum_dev,
                        dout_sum.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy dout sum");

    int q_errors = 0;
    int dout_errors = 0;
    for (int generation = 0; generation < kGenerations; ++generation) {
        const float expected_q = 32.0f * (1.0f + generation);
        const float expected_dout = 32.0f * (2.0f + generation);
        for (int consumer = 0; consumer < kConsumers; ++consumer) {
            for (int lane = 0; lane < kWaveSize; ++lane) {
                const int index =
                    (generation * kConsumers + consumer) * kWaveSize + lane;
                q_errors += std::fabs(q_sum[index] - expected_q) > 1.0e-3f;
                dout_errors +=
                    std::fabs(dout_sum[index] - expected_dout) > 1.0e-3f;
            }
        }
    }
    const bool pass = q_errors == 0 && dout_errors == 0;
    std::printf(
        "fused5_raw_q_dout_single_producer config waves=16 pages=2 "
        "generations=3 producer=4 consumers=8 dq_writer=4 lds_bytes=%d\n",
        kLdsBytes);
    std::printf(
        "fused5_raw_q_dout_single_producer q_errors=%d dout_errors=%d "
        "pass=%d\n",
        q_errors, dout_errors, pass ? 1 : 0);
    check_hip(hipFree(dout_sum_dev), "free dout sum");
    check_hip(hipFree(q_sum_dev), "free q sum");
    check_hip(hipFree(dout_dev), "free dout");
    check_hip(hipFree(q_dev), "free q");
    return pass ? 0 : 3;
}
