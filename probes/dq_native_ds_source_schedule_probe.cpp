#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_contract.h"
#include "shaobo_instr.h"

#include <algorithm>
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
constexpr int kMmacFloatsPerLane = 4;
constexpr int kOutModes = 5;
constexpr int kStatsWords = 2;
constexpr int kDsPageWords = 0;
constexpr int kKPageBytes = 4096;
constexpr int kLdsWords = 4096;
constexpr float kCompareTolerance = 0.15f;

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
    const int carry = src_lane < 4 ? 1 : 0;
    const int q_hi_word = src_word - (carry << 1);
    if (q_hi_word < 0) {
        return false;
    }
    const int base = src_lane + carry * dq::NativeDsSlotMap::kWaveSize - 4;
    const int q_lo = base >> 4;
    const int rem = base & 15;
    const int rem2 = rem & 7;
    dst_group = rem2 >> 1;
    dst_q = 4 * (q_hi_word >> 1) + q_lo;
    dst_word = 4 * (rem >> 3) + 2 * (rem2 & 1) + low;
    return dst_group < 4 && dst_q < 16 &&
           dst_word < dq::NativeDsSlotMap::kWordsPerFrag;
}

__host__ __device__ __forceinline__ uint16_t real_ds_bits(int q, int krow) {
    // Keep this layout probe independent from PMD's currently noisy half
    // arithmetic path.  Canonical dQ already validates the arithmetic; this
    // probe validates source-slot publication and native matrix consumption.
    return static_cast<uint16_t>(0x3000 + ((q & 15) << 4) + (krow & 15));
}

__device__ __forceinline__ _Float16 half_from_bits(uint16_t bits) {
    union {
        uint16_t u;
        _Float16 h;
    } v{bits};
    return v.h;
}

__device__ __forceinline__ uint16_t half_to_bits(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } v{value};
    return v.u;
}

__device__ __forceinline__ _Float16 real_ds_half(int q, int krow) {
    return half_from_bits(real_ds_bits(q, krow));
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

__device__ __forceinline__ void store_frag_half(float* out,
                                                int mode,
                                                int lane,
                                                const ProbeF16x8& frag,
                                                int word_base) {
#pragma unroll
    for (int i = 0; i < kMmacFloatsPerLane; ++i) {
        out[(mode * kWaveSize + lane) * kMmacFloatsPerLane + i] =
            static_cast<float>(frag.scalar[word_base + i]);
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
            mapped ? real_ds_half(dst_q, krow) : static_cast<_Float16>(0.0f);
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

    int local_read_errors = 0;
    const int dst_group = lane >> 4;
    const int dst_q = lane & 15;
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        const bool mapped =
            dq::NativeDsSlotMap::is_mapped(dst_group, dst_q, word);
        const int krow = dq::NativeDsSlotMap::slot_krow(dst_group, word);
        const uint16_t expected_bits =
            mapped ? real_ds_bits(dst_q, krow) : static_cast<uint16_t>(0);
        const uint16_t actual_bits = half_to_bits(ds_frag.scalar[word]);
        local_read_errors += actual_bits == expected_bits ? 0 : 1;
    }

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    store_acc(out, 0, lane, mmac_pair_lit(ds_frag, k_frag, zero));
    store_acc(out, 1, lane,
              mmac_one_lit(ds_frag.f16x4[0], k_frag.f16x4[0], zero));
    store_acc(out, 2, lane,
              mmac_one_lit(ds_frag.f16x4[1], k_frag.f16x4[1], zero));
    store_frag_half(out, 3, lane, ds_frag, 0);
    store_frag_half(out, 4, lane, ds_frag, 4);

    int local_mapped = 0;
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        int g = -1;
        int q = -1;
        int w = -1;
        local_mapped += source_slot_to_dst(lane, word, g, q, w) ? 1 : 0;
    }
    if (local_read_errors != 0) {
        atomicAdd(stats + 0, local_read_errors);
    }
    if (local_mapped != 0) {
        atomicAdd(stats + 1, local_mapped);
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

float output_value(const std::vector<float>& out,
                   int mode,
                   int lane,
                   int vec) {
    return out[(mode * kWaveSize + lane) * kMmacFloatsPerLane + vec];
}

float half_bits_to_float(uint16_t bits) {
    const int sign = (bits >> 15) & 1;
    const int exp = (bits >> 10) & 0x1f;
    const int mant = bits & 0x3ff;
    float value = 0.0f;
    if (exp == 0) {
        value = std::ldexp(static_cast<float>(mant), -24);
    } else if (exp == 31) {
        value = mant == 0 ? INFINITY : NAN;
    } else {
        value = std::ldexp(static_cast<float>(1024 + mant), exp - 25);
    }
    return sign ? -value : value;
}

float expected_split_sum(int q, bool high_half) {
    float sum = 0.0f;
    const int word_begin = high_half ? 4 : 0;
    const int word_end = high_half ? 8 : 4;
    for (int group = 0; group < 4; ++group) {
        for (int word = word_begin; word < word_end; ++word) {
            if (!dq::NativeDsSlotMap::is_mapped(group, q, word)) {
                continue;
            }
            sum += half_bits_to_float(
                real_ds_bits(q, dq::NativeDsSlotMap::slot_krow(group, word)));
        }
    }
    return sum;
}

bool summarize_split_output(const std::vector<float>& out,
                            int mode,
                            const char* label,
                            bool high_half) {
    int errors = 0;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float max_actual = -INFINITY;
    float min_actual = INFINITY;
    int samples = 0;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int q = lane & 15;
        const float expected = expected_split_sum(q, high_half);
        for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
            const float actual = output_value(out, mode, lane, vec);
            const float diff = std::fabs(actual - expected);
            max_actual = std::max(max_actual, actual);
            min_actual = std::min(min_actual, actual);
            max_abs = std::max(max_abs, diff);
            mean_abs += diff;
            ++samples;
            errors += diff > kCompareTolerance ? 1 : 0;
        }
    }
    mean_abs = samples != 0 ? mean_abs / static_cast<float>(samples) : 0.0f;
    std::printf(
        "real_ds_source_schedule_summary label=%s errors=%d max_abs=%g "
        "mean_abs=%g min_actual=%g max_actual=%g pass=%d\n",
        label, errors, max_abs, mean_abs, min_actual, max_actual,
        errors == 0 ? 1 : 0);
    return errors == 0;
}

bool summarize_frag_output(const std::vector<float>& out,
                           int mode,
                           const char* label,
                           bool high_half) {
    int errors = 0;
    float max_abs = 0.0f;
    float max_actual = -INFINITY;
    float min_actual = INFINITY;
    const int word_base = high_half ? 4 : 0;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int group = lane >> 4;
        const int q = lane & 15;
        for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
            const int word = word_base + vec;
            const bool mapped = dq::NativeDsSlotMap::is_mapped(group, q, word);
            const int krow = dq::NativeDsSlotMap::slot_krow(group, word);
            const float expected =
                mapped ? half_bits_to_float(real_ds_bits(q, krow)) : 0.0f;
            const float actual = output_value(out, mode, lane, vec);
            const float diff = std::fabs(actual - expected);
            max_actual = std::max(max_actual, actual);
            min_actual = std::min(min_actual, actual);
            max_abs = std::max(max_abs, diff);
            errors += diff > kCompareTolerance ? 1 : 0;
        }
    }
    std::printf(
        "real_ds_frag_summary label=%s errors=%d max_abs=%g min_actual=%g "
        "max_actual=%g pass=%d\n",
        label, errors, max_abs, min_actual, max_actual,
        errors == 0 ? 1 : 0);
    return errors == 0;
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
    check_hip(hipMalloc(reinterpret_cast<void**>(&stats_dev),
                        kStatsWords * sizeof(int)),
              "hipMalloc stats");

    std::vector<__half> k(kRows * kCols, __float2half(0.0f));
    for (int row = 0; row < 16; ++row) {
        for (int d = 0; d < 32; ++d) {
            k[row * kCols + d] = __float2half(1.0f);
        }
    }
    check_hip(hipMemcpy(k_dev, k.data(), k.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemset(out_dev, 0,
                        kOutModes * kWaveSize * kMmacFloatsPerLane *
                            sizeof(float)),
              "hipMemset out");
    check_hip(hipMemset(stats_dev, 0, kStatsWords * sizeof(int)),
              "hipMemset stats");

    hipLaunchKernelGGL(source_schedule_kernel, dim3(1), dim3(kThreads), 0, 0,
                       k_dev, out_dev, stats_dev);
    check_hip(hipDeviceSynchronize(), "source_schedule_kernel");

    std::vector<float> out(kOutModes * kWaveSize * kMmacFloatsPerLane);
    int stats[kStatsWords] = {};
    check_hip(hipMemcpy(out.data(), out_dev,
                        out.size() * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy out");
    check_hip(hipMemcpy(stats, stats_dev, sizeof(stats),
                        hipMemcpyDeviceToHost),
              "hipMemcpy stats");

    const bool frag_low_pass =
        summarize_frag_output(out, 3, "frag_low", false);
    const bool frag_high_pass =
        summarize_frag_output(out, 4, "frag_high", true);
    const bool low_pass = summarize_split_output(out, 1, "split_low", false);
    const bool high_pass = summarize_split_output(out, 2, "split_high", true);
    const bool pass = stats[0] == 0 && stats[1] == 504 && frag_low_pass &&
                      frag_high_pass && low_pass && high_pass;
    std::printf(
        "real_ds_source_schedule_result read_errors=%d mapped=%d "
        "frag_low_pass=%d frag_high_pass=%d low_pass=%d high_pass=%d "
        "pass=%d\n",
        stats[0], stats[1], frag_low_pass ? 1 : 0,
        frag_high_pass ? 1 : 0, low_pass ? 1 : 0, high_pass ? 1 : 0,
        pass ? 1 : 0);

    (void)hipFree(stats_dev);
    (void)hipFree(out_dev);
    (void)hipFree(k_dev);
    return pass ? 0 : 1;
}
