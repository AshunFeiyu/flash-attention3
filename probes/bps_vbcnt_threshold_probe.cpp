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
constexpr int kWaves = 2;
constexpr int kThreads = kWaveSize * kWaves;
constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kPrefixBlocks = 24;
constexpr int kBlocksPerGroup = 4;
constexpr int kGroups = 2;
constexpr int kBlocks = kPrefixBlocks + kBlocksPerGroup * kGroups;
constexpr int kBlockElems = kRows * kCols;
constexpr int kBlockBytes = kBlockElems * sizeof(__half);
constexpr int kFragmentsPerLane = 8;

#ifndef BPS_VBCNT_PARTIAL_WAIT
#define BPS_VBCNT_PARTIAL_WAIT 1
#endif

struct Bar {
    static constexpr int kAReady = 0;
    static constexpr int kBReady = 1;
    static constexpr int kDone = 2;
};

__device__ __forceinline__ void issue_block(__half* lds,
                                            const __half* src,
                                            int block) {
    const ins::Vec4U32 srsrc =
        ins::prepare_matrix_src(src + block * kBlockElems, kCols);
    ins::matrix_load_32x16_b16_bps_lds(
        lds, srsrc, block * kBlockBytes);
}

__device__ __forceinline__ void read_group(const __half* lds,
                                           __half* sink,
                                           int group) {
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
#pragma unroll
    for (int local = 0; local < kBlocksPerGroup; ++local) {
        const int block =
            kPrefixBlocks + group * kBlocksPerGroup + local;
        ins::F16x8 frag;
        ins::ds_read_matrix_32x16_normal(
            lds, block * kBlockBytes, frag.f16x8);
        ins::wait_lgkm(0);
#pragma unroll
        for (int i = 0; i < kFragmentsPerLane; ++i) {
            const int output_block = group * kBlocksPerGroup + local;
            const int out = (output_block * kWaveSize + lane) *
                                kFragmentsPerLane +
                            i;
            sink[out] = static_cast<__half>(frag.f16x8[i]);
        }
    }
}

__global__ void __launch_bounds__(kThreads, 1)
bps_vbcnt_threshold_probe_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ sink) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kBlocks * kBlockElems];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kAReady, 1);
        __builtin_hcu_s_abarrier_init(Bar::kBReady, 1);
        __builtin_hcu_s_abarrier_init(Bar::kDone, kWaves);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id == 0) {
        ins::abarrier_seq<false>(Bar::kAReady);
        ins::abarrier_seq<false>(Bar::kBReady);
#pragma unroll
        for (int block = 0; block < kBlocks; ++block) {
            issue_block(lds, src, block);
        }

#if BPS_VBCNT_PARTIAL_WAIT
#if BPS_VBCNT_PARTIAL_WAIT == 1
        asm volatile("s_waitcnt_vbcnt 4\n" ::: "memory");
#else
        asm volatile("s_waitcnt_vbcnt 0\n" ::: "memory");
#endif
#endif
        ins::abarrier_arrive_cnt<false>(Bar::kAReady, 1);

        asm volatile("s_waitcnt_vbcnt 0\n" ::: "memory");
        ins::abarrier_arrive_cnt<false>(Bar::kBReady, 1);
    } else {
        int a_phase = 0;
        int b_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kAReady, a_phase);
        read_group(lds, sink, 0);
        ins::abarrier_try_wait<true>(Bar::kBReady, b_phase);
        read_group(lds, sink, 1);
    }

    ins::abarrier_arrive_cnt<false>(Bar::kDone, 1);
    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kAReady);
        __builtin_hcu_s_abarrier_inv(Bar::kBReady);
        __builtin_hcu_s_abarrier_inv(Bar::kDone);
    }
#else
    (void)src;
    (void)sink;
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
    const size_t src_count = static_cast<size_t>(kBlocks) * kBlockElems;
    const size_t sink_count =
        static_cast<size_t>(kGroups * kBlocksPerGroup) * kBlockElems;
    std::vector<__half> src(src_count);
    std::vector<__half> sink(sink_count, __float2half(0.0f));
    for (int block = 0; block < kBlocks; ++block) {
        const __half value = __float2half(static_cast<float>(block + 1));
        for (int i = 0; i < kBlockElems; ++i) {
            src[block * kBlockElems + i] = value;
        }
    }

    __half* d_src = nullptr;
    __half* d_sink = nullptr;
    check_hip(hipMalloc(&d_src, src_count * sizeof(__half)), "hipMalloc src");
    check_hip(
        hipMalloc(&d_sink, sink_count * sizeof(__half)), "hipMalloc sink");
    check_hip(hipMemcpy(
                  d_src, src.data(), src_count * sizeof(__half),
                  hipMemcpyHostToDevice),
              "hipMemcpy src");
    check_hip(
        hipMemset(d_sink, 0, sink_count * sizeof(__half)), "hipMemset sink");

    hipLaunchKernelGGL(
        bps_vbcnt_threshold_probe_kernel, dim3(1), dim3(kThreads), 0, 0,
        d_src, d_sink);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(
                  sink.data(), d_sink, sink_count * sizeof(__half),
                  hipMemcpyDeviceToHost),
              "hipMemcpy sink");

    int a_errors = 0;
    int b_errors = 0;
    for (int output_block = 0;
         output_block < kGroups * kBlocksPerGroup;
         ++output_block) {
        const int source_block = kPrefixBlocks + output_block;
        const float expected = static_cast<float>(source_block + 1);
        for (int i = 0; i < kBlockElems; ++i) {
            const float got = __half2float(
                sink[output_block * kBlockElems + i]);
            if (std::fabs(got - expected) > 1.0e-3f) {
                if (output_block < kBlocksPerGroup) {
                    ++a_errors;
                } else {
                    ++b_errors;
                }
            }
        }
    }

    check_hip(hipFree(d_src), "hipFree src");
    check_hip(hipFree(d_sink), "hipFree sink");
    const bool pass = a_errors == 0 && b_errors == 0;
    std::printf(
        "bps_vbcnt_threshold_probe a_errors=%d b_errors=%d values=%zu "
        "pass=%d\n",
        a_errors, b_errors, sink_count, pass ? 1 : 0);
    return pass ? 0 : 2;
}
