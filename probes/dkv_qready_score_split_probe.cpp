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
constexpr int kMRows = 16;
constexpr int kNRows = 32;
constexpr int kDBlock = 32;
constexpr int kDBlocks = kHeadDim / kDBlock;
constexpr int kKvBlockBytes = kNRows * kDBlock * sizeof(__half);
constexpr int kRawBlockBytes = kMRows * kDBlock * sizeof(__half);
constexpr int kKBase = 0;
constexpr int kVBase = kKBase + kNRows * kHeadDim * sizeof(__half);
constexpr int kQBase = kVBase + kNRows * kHeadDim * sizeof(__half);
constexpr int kDoutBase = kQBase + kMRows * kHeadDim * sizeof(__half);
constexpr int kLdsBytes = kDoutBase + kMRows * kHeadDim * sizeof(__half);
constexpr int kResultScalars = 16;

struct Bar {
    static constexpr int kResidentFilled = 0;
    static constexpr int kQFilled = 1;
    static constexpr int kDoutFilled = 2;
    static constexpr int kAllDone = 3;
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x % kWaveSize);
}

__device__ __forceinline__ void publish_resident(
    __half* lds,
    int lds_base,
    const __half* src,
    int wave_local) {
    const __half* src_tile = src + wave_local * kDBlock;
    const ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, kHeadDim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, srsrc, lds_base + wave_local * kKvBlockBytes, true);
}

__device__ __forceinline__ void publish_raw(
    __half* lds,
    int lds_base,
    const __half* src,
    int wave_local) {
    const __half* src_tile = src + wave_local * kDBlock;
    const ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, kHeadDim);
    ins::matrix_load_32x16_b16_bps_lds(
        lds, srsrc, lds_base + wave_local * kRawBlockBytes);
}

__device__ __forceinline__ void mmac_score_dblock(
    int dblock,
    const ins::F16x8 (&k0)[kDBlocks],
    const ins::F16x8 (&k1)[kDBlocks],
    const ins::F16x8 (&q)[kDBlocks],
    const ins::F16x8& zero,
    ins::F32x4 (&score)[2]) {
#pragma unroll
    for (int owner_half = 0; owner_half < 2; ++owner_half) {
        const ins::F16x8& k = owner_half == 0 ? k0[dblock] : k1[dblock];
#pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            const bool first = dblock == 0 && k_half == 0;
            score[owner_half].f32 = ins::mmac_f16_lit(
                k.f16x4[k_half], q[dblock].f16x4[k_half],
                first ? zero.f32 : score[owner_half].f32);
        }
    }
}

__device__ __forceinline__ void mmac_dp_dblock(
    int dblock,
    const ins::F16x8 (&v0)[kDBlocks],
    const ins::F16x8 (&v1)[kDBlocks],
    const ins::F16x8 (&dout)[kDBlocks],
    const ins::F16x8& zero,
    ins::F32x4 (&dp)[2]) {
#pragma unroll
    for (int owner_half = 0; owner_half < 2; ++owner_half) {
        const ins::F16x8& v = owner_half == 0 ? v0[dblock] : v1[dblock];
#pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            const bool first = dblock == 0 && k_half == 0;
            dp[owner_half].f32 = ins::mmac_f16_lit(
                v.f16x4[k_half], dout[dblock].f16x4[k_half],
                first ? zero.f32 : dp[owner_half].f32);
        }
    }
}

__device__ __forceinline__ void store_result(
    float* sink,
    int consumer_wave,
    int lane,
    const ins::F32x4 (&score)[2],
    const ins::F32x4 (&dp)[2]) {
    const int base =
        (consumer_wave * kWaveSize + lane) * kResultScalars;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sink[base + half * 4 + i] = score[half].scalar[i];
            sink[base + 8 + half * 4 + i] = dp[half].scalar[i];
        }
    }
}

__device__ __forceinline__ void run_consumer(
    const __half* lds,
    float* split_sink,
    float* fused_sink,
    int consumer_wave,
    int lane) {
    int resident_phase = 0;
    int q_phase = 0;
    int dout_phase = 0;
    ins::F16x8 k0[kDBlocks];
    ins::F16x8 k1[kDBlocks];
    ins::F16x8 v0[kDBlocks];
    ins::F16x8 v1[kDBlocks];
    ins::F16x8 q[kDBlocks];
    ins::F16x8 dout[kDBlocks];
    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    ins::abarrier_try_wait<true>(Bar::kResidentFilled, resident_phase);
#pragma unroll
    for (int dblock = 0; dblock < kDBlocks; ++dblock) {
        ins::ds_read_matrix_trans_pair(
            lds, kKBase + dblock * kKvBlockBytes,
            k0[dblock].f16x8, k1[dblock].f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, kVBase + dblock * kKvBlockBytes,
            v0[dblock].f16x8, v1[dblock].f16x8);
    }
    ins::wait_lgkm(0);

    ins::abarrier_try_wait<true>(Bar::kQFilled, q_phase);
#pragma unroll
    for (int dblock = 0; dblock < kDBlocks; ++dblock) {
        ins::ds_read_matrix_32x16_trans(
            lds, kQBase + dblock * kRawBlockBytes, q[dblock].f16x8);
    }
    ins::wait_lgkm(0);

    ins::F32x4 split_score[2];
    ins::F32x4 split_dp[2];
#pragma unroll
    for (int dblock = 0; dblock < kDBlocks; ++dblock) {
        mmac_score_dblock(dblock, k0, k1, q, zero, split_score);
    }

    ins::abarrier_try_wait<true>(Bar::kDoutFilled, dout_phase);
#pragma unroll
    for (int dblock = 0; dblock < kDBlocks; ++dblock) {
        ins::ds_read_matrix_32x16_trans(
            lds, kDoutBase + dblock * kRawBlockBytes,
            dout[dblock].f16x8);
    }
    ins::wait_lgkm(0);
#pragma unroll
    for (int dblock = 0; dblock < kDBlocks; ++dblock) {
        mmac_dp_dblock(dblock, v0, v1, dout, zero, split_dp);
    }
    store_result(
        split_sink, consumer_wave, lane, split_score, split_dp);

    ins::F32x4 fused_score[2];
    ins::F32x4 fused_dp[2];
#pragma unroll
    for (int dblock = 0; dblock < kDBlocks; ++dblock) {
        mmac_score_dblock(dblock, k0, k1, q, zero, fused_score);
        mmac_dp_dblock(dblock, v0, v1, dout, zero, fused_dp);
    }
    store_result(
        fused_sink, consumer_wave, lane, fused_score, fused_dp);
}

__global__ void __launch_bounds__(kThreads, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dkv_qready_score_split_probe_kernel(const __half* __restrict__ q,
                                    const __half* __restrict__ dout,
                                    const __half* __restrict__ k,
                                    const __half* __restrict__ v,
                                    float* __restrict__ split_sink,
                                    float* __restrict__ fused_sink) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDoutFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(8);
        ins::abarrier_seq<false>(Bar::kResidentFilled);
        publish_resident(lds, kKBase, k, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled, 1);
        ins::abarrier_seq<false>(Bar::kQFilled);
        publish_raw(lds, kQBase, q, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kQFilled, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(248);
        run_consumer(
            lds, split_sink, fused_sink, static_cast<int>(wave_id - 4),
            lane_id());
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(248);
        run_consumer(
            lds, split_sink, fused_sink, static_cast<int>(wave_id - 4),
            lane_id());
    } else {
        __builtin_hcu_s_set_vgpr_size(8);
        ins::abarrier_seq<false>(Bar::kResidentFilled);
        publish_resident(lds, kVBase, v, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled, 1);
        ins::abarrier_seq<false>(Bar::kDoutFilled);
        publish_raw(lds, kDoutBase, dout, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kDoutFilled, 1);
    }

    ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kQFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kDoutFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)dout;
    (void)k;
    (void)v;
    (void)split_sink;
    (void)fused_sink;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

std::vector<__half> make_input(int rows, int salt) {
    std::vector<__half> values(rows * kHeadDim);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < kHeadDim; ++d) {
            const int raw = ((r * 17 + d * 13 + salt) % 29) - 14;
            values[r * kHeadDim + d] = __float2half(raw * 0.03125f);
        }
    }
    return values;
}

}  // namespace

int main() {
    const std::vector<__half> q = make_input(kMRows, 1);
    const std::vector<__half> dout = make_input(kMRows, 5);
    const std::vector<__half> k = make_input(kNRows, 9);
    const std::vector<__half> v = make_input(kNRows, 13);
    const size_t result_count =
        static_cast<size_t>(kConsumerWaves) * kWaveSize * kResultScalars;
    std::vector<float> split(result_count, 0.0f);
    std::vector<float> fused(result_count, 0.0f);

    __half *d_q = nullptr, *d_dout = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_split = nullptr, *d_fused = nullptr;
    check_hip(hipMalloc(&d_q, q.size() * sizeof(__half)), "hipMalloc q");
    check_hip(hipMalloc(&d_dout, dout.size() * sizeof(__half)), "hipMalloc dO");
    check_hip(hipMalloc(&d_k, k.size() * sizeof(__half)), "hipMalloc k");
    check_hip(hipMalloc(&d_v, v.size() * sizeof(__half)), "hipMalloc v");
    check_hip(hipMalloc(&d_split, result_count * sizeof(float)), "hipMalloc split");
    check_hip(hipMalloc(&d_fused, result_count * sizeof(float)), "hipMalloc fused");
    check_hip(hipMemcpy(d_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice), "copy q");
    check_hip(hipMemcpy(d_dout, dout.data(), dout.size() * sizeof(__half), hipMemcpyHostToDevice), "copy dO");
    check_hip(hipMemcpy(d_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice), "copy k");
    check_hip(hipMemcpy(d_v, v.data(), v.size() * sizeof(__half), hipMemcpyHostToDevice), "copy v");
    check_hip(hipMemset(d_split, 0, result_count * sizeof(float)), "clear split");
    check_hip(hipMemset(d_fused, 0, result_count * sizeof(float)), "clear fused");

    hipLaunchKernelGGL(
        dkv_qready_score_split_probe_kernel, dim3(1), dim3(kThreads), 0, 0,
        d_q, d_dout, d_k, d_v, d_split, d_fused);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(split.data(), d_split, result_count * sizeof(float), hipMemcpyDeviceToHost), "copy split");
    check_hip(hipMemcpy(fused.data(), d_fused, result_count * sizeof(float), hipMemcpyDeviceToHost), "copy fused");

    int errors = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < result_count; ++i) {
        const float diff = std::fabs(split[i] - fused[i]);
        max_abs = diff > max_abs ? diff : max_abs;
        errors += !std::isfinite(split[i]) || !std::isfinite(fused[i]) ||
                  diff > 1.0e-6f;
    }
    const int pass = errors == 0;
    std::printf(
        "dkv_qready_score_split_probe errors=%d max_abs=%g pass=%d\n",
        errors, max_abs, pass);
    return pass ? 0 : 1;
}
