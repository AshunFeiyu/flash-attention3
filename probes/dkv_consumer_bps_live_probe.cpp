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
constexpr int kThreads = kWaveSize * kWaves;
constexpr int kConsumerWaves = 8;
constexpr int kHeadDim = 128;
constexpr int kRows = 16;
constexpr int kDColsPerWave = 32;
constexpr int kRawBlockBytes = kRows * kDColsPerWave * sizeof(__half);
constexpr int kTileBytes = kRows * kHeadDim * sizeof(__half);
constexpr int kQBase = 0;
constexpr int kDoutBase = kQBase + kTileBytes;
constexpr int kLdsBytes = kDoutBase + kTileBytes;
constexpr int kAccVectors = 32;
constexpr int kAccScalars = kAccVectors * 4;

struct Bar {
    static constexpr int kFilled = 0;
    static constexpr int kUsed = 1;
    static constexpr int kAllDone = 2;
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x % kWaveSize);
}

__device__ __forceinline__ void initialize_live_accumulators(
    ins::F32x4 (&acc)[kAccVectors],
    int consumer_wave,
    int lane) {
#pragma unroll
    for (int i = 0; i < kAccVectors; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i].scalar[j] = static_cast<float>(
                consumer_wave * 100000 + lane * 1000 + i * 4 + j);
        }
    }
}

__device__ __forceinline__ void keep_live_accumulators(
    ins::F32x4 (&acc)[kAccVectors]) {
#pragma unroll
    for (int i = 0; i < kAccVectors; ++i) {
        ins::keep_accumulator_live(acc[i]);
    }
}

__device__ __forceinline__ void publish_matrix_block(
    __half* lds,
    int lds_base,
    const __half* src,
    int wave_local) {
    const __half* src_tile = src + wave_local * kDColsPerWave;
    ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, kHeadDim);
    ins::matrix_load_32x16_b16_bps_lds(
        lds, srsrc, lds_base + wave_local * kRawBlockBytes);
}

__device__ __forceinline__ int check_constant_fragment(
    const ins::F16x8& frag,
    float expected) {
    int errors = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        errors += std::fabs(static_cast<float>(frag.f16x8[i]) - expected) >
                  1.0e-3f;
    }
    return errors;
}

__global__ void __launch_bounds__(kThreads, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dkv_consumer_bps_live_probe_kernel(const __half* __restrict__ q,
                                   const __half* __restrict__ dout,
                                   float* __restrict__ acc_sink,
                                   int* __restrict__ status) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kFilled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kUsed, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(8);
        int used_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kUsed, used_phase);
        if (lane_id() == 0) {
            atomicAdd(status + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(248);
        const int lane = lane_id();
        const int wave_local = static_cast<int>(wave_id - 4);
        const int consumer_wave = wave_local;
        ins::F32x4 acc[kAccVectors];
        initialize_live_accumulators(acc, consumer_wave, lane);
        keep_live_accumulators(acc);

        ins::abarrier_seq<false>(Bar::kFilled);
        publish_matrix_block(lds, kQBase, q, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kFilled, 1);

        int filled_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kFilled, filled_phase);
        ins::F16x8 q_frag;
        ins::F16x8 dout_frag;
        ins::ds_read_matrix_32x16_normal(
            lds, kQBase + wave_local * kRawBlockBytes, q_frag.f16x8);
        ins::ds_read_matrix_32x16_normal(
            lds, kDoutBase + wave_local * kRawBlockBytes, dout_frag.f16x8);
        ins::wait_lgkm(0);
        keep_live_accumulators(acc);

#pragma unroll
        for (int i = 0; i < kAccVectors; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int sink_idx =
                    ((consumer_wave * kWaveSize + lane) * kAccScalars) +
                    i * 4 + j;
                acc_sink[sink_idx] = acc[i].scalar[j];
            }
        }
        const int errors = check_constant_fragment(q_frag, 1.0f) +
                           check_constant_fragment(dout_frag, 2.0f);
        if (errors != 0) {
            atomicAdd(status, errors);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kUsed, 1);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(248);
        const int lane = lane_id();
        const int wave_local = static_cast<int>(wave_id - 8);
        const int consumer_wave = 4 + wave_local;
        ins::F32x4 acc[kAccVectors];
        initialize_live_accumulators(acc, consumer_wave, lane);
        keep_live_accumulators(acc);

        ins::abarrier_seq<false>(Bar::kFilled);
        publish_matrix_block(lds, kDoutBase, dout, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kFilled, 1);

        int filled_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kFilled, filled_phase);
        ins::F16x8 q_frag;
        ins::F16x8 dout_frag;
        ins::ds_read_matrix_32x16_normal(
            lds, kQBase + wave_local * kRawBlockBytes, q_frag.f16x8);
        ins::ds_read_matrix_32x16_normal(
            lds, kDoutBase + wave_local * kRawBlockBytes, dout_frag.f16x8);
        ins::wait_lgkm(0);
        keep_live_accumulators(acc);

#pragma unroll
        for (int i = 0; i < kAccVectors; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int sink_idx =
                    ((consumer_wave * kWaveSize + lane) * kAccScalars) +
                    i * 4 + j;
                acc_sink[sink_idx] = acc[i].scalar[j];
            }
        }
        const int errors = check_constant_fragment(q_frag, 1.0f) +
                           check_constant_fragment(dout_frag, 2.0f);
        if (errors != 0) {
            atomicAdd(status, errors);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kUsed, 1);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(8);
        int used_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kUsed, used_phase);
        if (lane_id() == 0) {
            atomicAdd(status + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)dout;
    (void)acc_sink;
    (void)status;
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
    std::vector<__half> q(kRows * kHeadDim, __float2half(1.0f));
    std::vector<__half> dout(kRows * kHeadDim, __float2half(2.0f));
    std::vector<float> sink(kConsumerWaves * kWaveSize * kAccScalars, 0.0f);
    int status[2] = {0, 0};

    __half* d_q = nullptr;
    __half* d_dout = nullptr;
    float* d_sink = nullptr;
    int* d_status = nullptr;
    check_hip(hipMalloc(&d_q, q.size() * sizeof(__half)), "hipMalloc q");
    check_hip(hipMalloc(&d_dout, dout.size() * sizeof(__half)),
              "hipMalloc dout");
    check_hip(hipMalloc(&d_sink, sink.size() * sizeof(float)),
              "hipMalloc sink");
    check_hip(hipMalloc(&d_status, sizeof(status)), "hipMalloc status");
    check_hip(hipMemcpy(d_q, q.data(), q.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemcpy(d_dout, dout.data(), dout.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy dout");
    check_hip(hipMemset(d_sink, 0, sink.size() * sizeof(float)),
              "hipMemset sink");
    check_hip(hipMemset(d_status, 0, sizeof(status)), "hipMemset status");

    hipLaunchKernelGGL(dkv_consumer_bps_live_probe_kernel,
                       dim3(1), dim3(kThreads), 0, 0,
                       d_q, d_dout, d_sink, d_status);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(sink.data(), d_sink, sink.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy sink");
    check_hip(hipMemcpy(status, d_status, sizeof(status),
                        hipMemcpyDeviceToHost),
              "hipMemcpy status");

    int acc_errors = 0;
    int printed_errors = 0;
    for (int consumer_wave = 0; consumer_wave < kConsumerWaves;
         ++consumer_wave) {
        for (int lane = 0; lane < kWaveSize; ++lane) {
            for (int i = 0; i < kAccScalars; ++i) {
                const int idx =
                    (consumer_wave * kWaveSize + lane) * kAccScalars + i;
                const float expected = static_cast<float>(
                    consumer_wave * 100000 + lane * 1000 + i);
                if (sink[idx] != expected) {
                    ++acc_errors;
                    if (printed_errors < 16) {
                        std::printf(
                            "acc_mismatch wave=%d lane=%d scalar=%d got=%g "
                            "expected=%g\n",
                            consumer_wave, lane, i, sink[idx], expected);
                        ++printed_errors;
                    }
                }
            }
        }
    }

    hipFree(d_q);
    hipFree(d_dout);
    hipFree(d_sink);
    hipFree(d_status);
    const bool pass = status[0] == 0 && status[1] == 8 && acc_errors == 0;
    std::printf(
        "dkv_consumer_bps_live_probe fragment_errors=%d used_waiters=%d "
        "acc_errors=%d pass=%d\n",
        status[0], status[1], acc_errors, pass ? 1 : 0);
    return pass ? 0 : 2;
}
