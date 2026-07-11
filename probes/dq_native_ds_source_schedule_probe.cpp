#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_contract.h"
#include "shaobo_instr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dq = shaobo::fa3::bwd::dq;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kThreads = 64;
constexpr int kWaveSize = 64;
constexpr int kFragWords = 8;
constexpr int kRows = 32;
constexpr int kCols = 64;
constexpr int kProbeK = 7;
constexpr int kKTagBase = 1024;
constexpr int kMmacFloatsPerLane = 4;
constexpr int kOutModes = 3;
constexpr int kDsPageWords = 0;
constexpr int kKPageBytes = 4096;
constexpr int kLdsWords = 4096;

union ProbeF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[kFragWords];
    uint16_t u16[kFragWords];
};

union ProbeF32x4 {
    ins::Vec4F32 f32;
    float scalar[kMmacFloatsPerLane];
};

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(2);
    }
}

__device__ __forceinline__ bool source_slot_to_dst(int src_lane,
                                                   int src_word,
                                                   int& dst_group,
                                                   int& dst_q,
                                                   int& dst_word) {
    const int low = src_word & 1;
#pragma unroll
    for (int carry = 0; carry < 2; ++carry) {
        const int q_hi_word = src_word - 2 * carry;
        if (q_hi_word < 0) {
            continue;
        }
        const int q_hi = q_hi_word >> 1;
        if (q_hi > 3) {
            continue;
        }
        const int raw_lane = src_lane + carry * dq::NativeDsSlotMap::kWaveSize;
        const int base = raw_lane - 4;
        if (base < 0 || base >= dq::NativeDsSlotMap::kWaveSize) {
            continue;
        }
        const int q_lo = base >> 4;
        const int rem = base & 15;
        const int word_hi = rem >> 3;
        const int rem2 = rem & 7;
        const int group = rem2 >> 1;
        const int word_mid = rem2 & 1;
        const int word = 4 * word_hi + 2 * word_mid + low;
        const int q = 4 * q_hi + q_lo;
        if (group < 4 && q < 16 && word < dq::NativeDsSlotMap::kWordsPerFrag &&
            dq::NativeDsSlotMap::dst_source_lane(group, q, word) ==
                src_lane &&
            dq::NativeDsSlotMap::dst_source_word(
                q, word,
                dq::NativeDsSlotMap::dst_raw_source_lane(group, q, word)) ==
                src_word) {
            dst_group = group;
            dst_q = q;
            dst_word = word;
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ ProbeF32x4 mmac_pair_lit(const ProbeF16x8& lhs,
                                                    const ProbeF16x8& rhs,
                                                    const ins::F16x8& zero) {
    ProbeF32x4 out{};
#if defined(__gfx946__) || defined(__gfx938__)
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], zero.f32);
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[1], rhs.f16x4[1], out.f32);
#else
    (void)lhs;
    (void)rhs;
    (void)zero;
#endif
    return out;
}

__device__ __forceinline__ ProbeF32x4 mmac_one_lit(ins::Vec4F16 lhs,
                                                   ins::Vec4F16 rhs,
                                                   const ins::F16x8& zero) {
    ProbeF32x4 out{};
#if defined(__gfx946__) || defined(__gfx938__)
    out.f32 = ins::mmac_f16_lit(lhs, rhs, zero.f32);
#else
    (void)lhs;
    (void)rhs;
    (void)zero;
#endif
    return out;
}

__device__ __forceinline__ void store_acc(float* out,
                                          int mode,
                                          int lane,
                                          const ProbeF32x4& acc) {
#pragma unroll
    for (int i = 0; i < kMmacFloatsPerLane; ++i) {
        out[(mode * kWaveSize + lane) * kMmacFloatsPerLane + i] =
            acc.scalar[i];
    }
}

__global__ void __launch_bounds__(kThreads, 1)
    source_schedule_kernel(const __half* __restrict__ k_input,
                           float* __restrict__ out,
                           int* __restrict__ stats) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) uint16_t lds[kLdsWords];
    const int lane = static_cast<int>(threadIdx.x & 63);
    constexpr int kMlsBarrier = 0;

    for (int i = threadIdx.x; i < kLdsWords; i += blockDim.x) {
        lds[i] = 0;
    }
    __syncthreads();

    if (lane == 0) {
        __builtin_hcu_s_abarrier_init(kMlsBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    ins::Vec4U32 k_src = ins::prepare_matrix_src(k_input, kCols);
    ins::abarrier_seq<false>(kMlsBarrier);
    ins::matrix_load_32x16_b16_bps_lds(reinterpret_cast<__half*>(lds), k_src,
                                       kKPageBytes);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(kMlsBarrier, 1);
    int phase = 0;
    ins::abarrier_try_wait<false>(kMlsBarrier, phase);

    ProbeF16x8 producer{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        int dst_group = -1;
        int dst_q = -1;
        int dst_word = -1;
        const bool mapped =
            source_slot_to_dst(lane, word, dst_group, dst_q, dst_word);
        const int krow = mapped
                             ? dq::NativeDsSlotMap::slot_krow(dst_group,
                                                              dst_word)
                             : -1;
        producer.scalar[word] =
            static_cast<_Float16>(krow == kProbeK ? 1.0f : 0.0f);
        if (mapped && lane == 0) {
            (void)dst_q;
        }
    }

    ins::ds_write_matrix_32x16_f16(
        producer.f16x8, reinterpret_cast<__half*>(lds),
        kDsPageWords * static_cast<int>(sizeof(uint16_t)));
    ins::wait_lgkm(0);

    ProbeF16x8 ds_frag{};
    ds_frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        reinterpret_cast<_Float16*>(lds + kDsPageWords), 16, 2, 1, 0);
    ProbeF16x8 k_frag{};
    ins::ds_read_matrix_32x16_normal(reinterpret_cast<__half*>(lds),
                                    kKPageBytes, k_frag.f16x8);
    ins::wait_lgkm(0);

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    store_acc(out, 0, lane, mmac_pair_lit(ds_frag, k_frag, zero));
    store_acc(out, 1, lane,
              mmac_one_lit(ds_frag.f16x4[0], k_frag.f16x4[0], zero));
    store_acc(out, 2, lane,
              mmac_one_lit(ds_frag.f16x4[1], k_frag.f16x4[1], zero));

    int local_mapped = 0;
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        int g = -1;
        int q = -1;
        int w = -1;
        local_mapped += source_slot_to_dst(lane, word, g, q, w) ? 1 : 0;
    }
    if (local_mapped != 0) {
        atomicAdd(stats, local_mapped);
    }

    __syncthreads();
    if (lane == 0) {
        __builtin_hcu_s_abarrier_inv(kMlsBarrier);
    }
#else
    (void)k_input;
    (void)out;
    (void)stats;
#endif
}

bool decode_k_tag_value(float value, int& krow, int& d) {
    if (!std::isfinite(value) || std::fabs(value) < 0.5f) {
        return false;
    }
    const float raw_tag = value - static_cast<float>(kKTagBase);
    const int tag_candidate = static_cast<int>(std::lround(raw_tag));
    if (tag_candidate < 0 || tag_candidate >= 16 * kCols) {
        return false;
    }
    const int krow_candidate = tag_candidate / kCols;
    const int d_candidate = tag_candidate % kCols;
    if (d_candidate >= 32) {
        return false;
    }
    const float expected = static_cast<float>(kKTagBase + tag_candidate);
    if (std::fabs(value - expected) > 0.75f) {
        return false;
    }
    krow = krow_candidate;
    d = d_candidate;
    return true;
}

float output_value(const std::vector<float>& out,
                   int mode,
                   int lane,
                   int vec) {
    return out[(mode * kWaveSize + lane) * kMmacFloatsPerLane + vec];
}

bool summarize_output_modes(const std::vector<float>& out,
                            int mode_begin,
                            int mode_end,
                            const char* label,
                            int expected_unique_output,
                            int expected_unique_d) {
    std::vector<uint8_t> seen_output(
        (mode_end - mode_begin) * kWaveSize * kMmacFloatsPerLane, 0);
    bool seen_q[16] = {};
    bool seen_krow[16] = {};
    bool seen_d[32] = {};
    int decoded = 0;
    int raw_nonzero = 0;
    for (int mode = mode_begin; mode < mode_end; ++mode) {
        for (int lane = 0; lane < kWaveSize; ++lane) {
            for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
                const float value = output_value(out, mode, lane, vec);
                if (std::isfinite(value) && std::fabs(value) > 0.5f) {
                    ++raw_nonzero;
                }
                int krow = -1;
                int d = -1;
                if (!decode_k_tag_value(value, krow, d)) {
                    continue;
                }
                const int q = lane & 15;
                const int output_idx =
                    ((mode - mode_begin) * kWaveSize + lane) *
                        kMmacFloatsPerLane +
                    vec;
                seen_output[output_idx] = 1;
                seen_q[q] = true;
                seen_krow[krow] = true;
                seen_d[d] = true;
                ++decoded;
            }
        }
    }
    int unique_output = 0;
    int unique_q = 0;
    int unique_krow = 0;
    int unique_d = 0;
    for (uint8_t v : seen_output) unique_output += v ? 1 : 0;
    for (bool v : seen_q) unique_q += v ? 1 : 0;
    for (bool v : seen_krow) unique_krow += v ? 1 : 0;
    for (bool v : seen_d) unique_d += v ? 1 : 0;
    const bool pass =
        unique_output == expected_unique_output && unique_q == 16 &&
        unique_krow == 1 && seen_krow[kProbeK] &&
        unique_d >= expected_unique_d;
    std::printf(
        "source_schedule_summary label=%s raw_nonzero=%d decoded=%d "
        "unique_output=%d unique_q=%d unique_krow=%d probe_k_seen=%d "
        "unique_d=%d pass=%d\n",
        label, raw_nonzero, decoded, unique_output, unique_q, unique_krow,
        seen_krow[kProbeK] ? 1 : 0, unique_d, pass ? 1 : 0);
    return pass;
}

}  // namespace

int main() {
    __half* k_dev = nullptr;
    float* out_dev = nullptr;
    int* stats_dev = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&k_dev),
                        kRows * kCols * sizeof(__half)),
              "hipMalloc k");
    check_hip(hipMalloc(reinterpret_cast<void**>(&out_dev),
                        kOutModes * kWaveSize * kMmacFloatsPerLane *
                            sizeof(float)),
              "hipMalloc out");
    check_hip(hipMalloc(reinterpret_cast<void**>(&stats_dev), sizeof(int)),
              "hipMalloc stats");

    std::vector<__half> k(kRows * kCols, __float2half(0.0f));
    for (int row = 0; row < 16; ++row) {
        for (int d = 0; d < 32; ++d) {
            k[row * kCols + d] = __float2half(
                static_cast<float>(kKTagBase + row * kCols + d));
        }
    }
    check_hip(hipMemcpy(k_dev, k.data(), k.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemset(out_dev, 0,
                        kOutModes * kWaveSize * kMmacFloatsPerLane *
                            sizeof(float)),
              "hipMemset out");
    check_hip(hipMemset(stats_dev, 0, sizeof(int)), "hipMemset stats");

    hipLaunchKernelGGL(source_schedule_kernel, dim3(1), dim3(kThreads), 0, 0,
                       k_dev, out_dev, stats_dev);
    check_hip(hipDeviceSynchronize(), "source_schedule_kernel");

    std::vector<float> out(kOutModes * kWaveSize * kMmacFloatsPerLane);
    int mapped = 0;
    check_hip(hipMemcpy(out.data(), out_dev,
                        out.size() * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy out");
    check_hip(hipMemcpy(&mapped, stats_dev, sizeof(int),
                        hipMemcpyDeviceToHost),
              "hipMemcpy stats");

    const bool pair_pass =
        summarize_output_modes(out, 0, 1, "pair_acc", 256, 16);
    const bool low_pass =
        summarize_output_modes(out, 1, 2, "split_low", 256, 16);
    const bool high_pass =
        summarize_output_modes(out, 2, 3, "split_high", 256, 16);
    const bool split_pass =
        summarize_output_modes(out, 1, 3, "split_combined", 512, 32);
    const bool pass = mapped == 504 && !pair_pass && low_pass && high_pass &&
                      split_pass;
    std::printf(
        "source_schedule_result mapped=%d pair_pass=%d low_pass=%d "
        "high_pass=%d split_pass=%d pass=%d\n",
        mapped, pair_pass ? 1 : 0, low_pass ? 1 : 0, high_pass ? 1 : 0,
        split_pass ? 1 : 0, pass ? 1 : 0);

    (void)hipFree(stats_dev);
    (void)hipFree(out_dev);
    (void)hipFree(k_dev);
    return pass ? 0 : 1;
}
