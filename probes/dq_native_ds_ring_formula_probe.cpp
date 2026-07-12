#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_probe_contract.h"
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

constexpr int kWaveSize = 64;
constexpr int kWaves = 12;
constexpr int kThreads = kWaveSize * kWaves;
constexpr int kFragWords = 8;
constexpr int kOutModes = 4;
constexpr int kMmacFloatsPerLane = 4;
constexpr int kKRows = 32;
constexpr int kKCols = 64;
constexpr int kReaderPageElems = 64 * 16;
constexpr int kDsSlot0Words = 0;
constexpr int kDsSlot1Words = kReaderPageElems;
constexpr int kKPageWords = 2 * kReaderPageElems;
constexpr int kLdsWords = kKPageWords + kReaderPageElems;
constexpr float kTolerance = 0.15f;

union FragF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[kFragWords];
};

union AccF32x4 {
    ins::Vec4F32 f32;
    float scalar[kMmacFloatsPerLane];
};

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
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

__host__ __device__ __forceinline__ float ds_formula_value(int q, int krow) {
    constexpr float kScale = 0.25f;
    constexpr float kScaleLog2 = 0.36067376022224085f;
    const float qf = static_cast<float>(q + 1);
    const float kf = static_cast<float>(krow + 1);
    const float qk = 0.0625f * qf + 0.03125f * kf;
    const float dp = 0.125f * qf - 0.015625f * kf;
    const float row_max = 0.0f;
    const float row_sum = 2.0f + 0.03125f * qf;
    const float row_delta = 0.05f * qf;
    const float p = exp2f((qk - row_max) * kScaleLog2) / row_sum;
    return p * (dp - row_delta) * kScale;
}

uint16_t half_bits_from_float(float value) {
    union {
        __half h;
        uint16_t u;
    } out{__float2half(value)};
    return out.u;
}

__device__ __forceinline__ _Float16 ds_formula_half(int q, int krow) {
    return static_cast<_Float16>(ds_formula_value(q, krow));
}

__device__ __forceinline__ _Float16 zero_half() {
    return static_cast<_Float16>(0.0f);
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

__device__ __forceinline__ FragF16x8 make_source_slot_frag(int lane,
                                                           int slot_base_k) {
    FragF16x8 frag{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        int group = -1;
        int q = -1;
        int dst_word = -1;
        if (source_slot_to_dst(lane, word, group, q, dst_word)) {
            const int krow =
                slot_base_k + dq::NativeDsSlotMap::slot_krow(group, dst_word);
            frag.scalar[word] = ds_formula_half(q, krow);
        } else {
            frag.scalar[word] = zero_half();
        }
    }
    return frag;
}

__device__ __forceinline__ AccF32x4 mmac_one_lit(ins::Vec4F16 lhs,
                                                 ins::Vec4F16 rhs) {
    ins::F16x8 zero{};
    ins::zero_f16x8(zero);
    AccF32x4 out{};
    out.f32 = ins::mmac_f16_lit(lhs, rhs, zero.f32);
    return out;
}

__device__ __forceinline__ void store_acc(float* out,
                                          int mode,
                                          int lane,
                                          const AccF32x4& acc) {
#pragma unroll
    for (int i = 0; i < kMmacFloatsPerLane; ++i) {
        out[(mode * kWaveSize + lane) * kMmacFloatsPerLane + i] =
            acc.scalar[i];
    }
}

template <int SlotId>
__device__ __forceinline__ void publish_ds_slot(uint16_t* lds, int lane) {
    constexpr int slot_words = SlotId == 0 ? kDsSlot0Words : kDsSlot1Words;
    constexpr int slot_base_k = SlotId == 0 ? 0 : 16;
    const FragF16x8 frag = make_source_slot_frag(lane, slot_base_k);
    ins::ds_write_matrix_32x16_f16(
        frag.f16x8, reinterpret_cast<__half*>(lds),
        slot_words * static_cast<int>(sizeof(uint16_t)));
    ins::wait_lgkm(0);
}

template <int SlotId>
__device__ __forceinline__ void consume_ds_slot(const uint16_t* lds,
                                                int lane,
                                                float* out) {
    constexpr int slot_words = SlotId == 0 ? kDsSlot0Words : kDsSlot1Words;
    FragF16x8 ds{};
    ds.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        reinterpret_cast<const _Float16*>(lds + slot_words), 16, 2, 1, 0);
    FragF16x8 k{};
    ins::ds_read_matrix_32x16_normal(reinterpret_cast<const __half*>(lds),
                                     kKPageWords *
                                         static_cast<int>(sizeof(uint16_t)),
                                     k.f16x8);
    ins::wait_lgkm(0);
    store_acc(out, SlotId * 2, lane, mmac_one_lit(ds.f16x4[0], k.f16x4[0]));
    store_acc(out, SlotId * 2 + 1, lane,
              mmac_one_lit(ds.f16x4[1], k.f16x4[1]));
}

__global__ void __launch_bounds__(kThreads, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dq_native_ds_ring_formula_probe_kernel(const __half* __restrict__ k_input,
                                          float* __restrict__ out,
                                          int* __restrict__ stats) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) uint16_t lds[kLdsWords];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    using Bar = dq::DqNativeDsRingBarrierLedger;

    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kKvFilled, 1);
        __builtin_hcu_s_abarrier_init(Bar::kDsSlot0Filled, 1);
        __builtin_hcu_s_abarrier_init(Bar::kDsSlot0Used, 1);
        __builtin_hcu_s_abarrier_init(Bar::kDsSlot1Filled, 1);
        __builtin_hcu_s_abarrier_init(Bar::kDsSlot1Used, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave == 0) {
        __builtin_hcu_s_set_vgpr_size(40);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        ins::abarrier_seq<false>(Bar::kKvFilled);
        ins::Vec4U32 k_src = ins::prepare_matrix_src(k_input, kKCols);
        ins::matrix_load_32x16_b16_bps_lds(
            reinterpret_cast<__half*>(lds), k_src,
            kKPageWords * static_cast<int>(sizeof(uint16_t)));
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kKvFilled, 1);
        if (lane == 0) {
            atomicAdd(stats + 0, 1);
        }
    } else if (wave == 4) {
        __builtin_hcu_s_set_vgpr_size(64);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        int kv_phase = 0;
        ins::abarrier_try_wait<false>(Bar::kKvFilled, kv_phase);
        ins::abarrier_seq<false>(Bar::kDsSlot0Filled);
        publish_ds_slot<0>(lds, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kDsSlot0Filled, 1);
        if (lane == 0) {
            atomicAdd(stats + 1, 1);
        }
    } else if (wave == 5) {
        __builtin_hcu_s_set_vgpr_size(64);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        int kv_phase = 0;
        ins::abarrier_try_wait<false>(Bar::kKvFilled, kv_phase);
        ins::abarrier_seq<false>(Bar::kDsSlot1Filled);
        publish_ds_slot<1>(lds, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kDsSlot1Filled, 1);
        if (lane == 0) {
            atomicAdd(stats + 1, 1);
        }
    } else if (wave == 8) {
        __builtin_hcu_s_set_vgpr_size(80);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        int kv_phase = 0;
        int ds_phase = 0;
        ins::abarrier_try_wait<false>(Bar::kKvFilled, kv_phase);
        ins::abarrier_try_wait<false>(Bar::kDsSlot0Filled, ds_phase);
        consume_ds_slot<0>(lds, lane, out);
        ins::abarrier_arrive_cnt<false>(Bar::kDsSlot0Used, 1);
        if (lane == 0) {
            atomicAdd(stats + 2, 1);
        }
    } else if (wave == 9) {
        __builtin_hcu_s_set_vgpr_size(80);
        const int lane = static_cast<int>(threadIdx.x % kWaveSize);
        int kv_phase = 0;
        int ds_phase = 0;
        ins::abarrier_try_wait<false>(Bar::kKvFilled, kv_phase);
        ins::abarrier_try_wait<false>(Bar::kDsSlot1Filled, ds_phase);
        consume_ds_slot<1>(lds, lane, out);
        ins::abarrier_arrive_cnt<false>(Bar::kDsSlot1Used, 1);
        if (lane == 0) {
            atomicAdd(stats + 2, 1);
        }
    } else {
        __builtin_hcu_s_set_vgpr_size(32);
    }

    __syncthreads();
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kKvFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kDsSlot0Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kDsSlot0Used);
        __builtin_hcu_s_abarrier_inv(Bar::kDsSlot1Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kDsSlot1Used);
    }
    __syncthreads();
#else
    (void)k_input;
    (void)out;
    (void)stats;
#endif
}

float expected_half_sum(int q, int slot, bool high_half) {
    float sum = 0.0f;
    const int slot_base_k = slot == 0 ? 0 : 16;
    const int word_begin = high_half ? 4 : 0;
    const int word_end = high_half ? 8 : 4;
    for (int group = 0; group < 4; ++group) {
        for (int word = word_begin; word < word_end; ++word) {
            if (!dq::NativeDsSlotMap::is_mapped(group, q, word)) {
                continue;
            }
            const int krow =
                slot_base_k + dq::NativeDsSlotMap::slot_krow(group, word);
            sum += half_bits_to_float(half_bits_from_float(
                ds_formula_value(q, krow)));
        }
    }
    return sum;
}

bool check_mode(const std::vector<float>& out,
                int mode,
                int slot,
                bool high_half,
                const char* label) {
    int errors = 0;
    float max_abs = 0.0f;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int q = lane & 15;
        const float expected = expected_half_sum(q, slot, high_half);
        for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
            const float actual =
                out[(mode * kWaveSize + lane) * kMmacFloatsPerLane + vec];
            const float diff = std::fabs(actual - expected);
            max_abs = std::max(max_abs, diff);
            errors += diff > kTolerance ? 1 : 0;
        }
    }
    std::printf("native_ds_ring_formula_%s errors=%d max_abs=%g pass=%d\n",
                label, errors, max_abs, errors == 0 ? 1 : 0);
    return errors == 0;
}

}  // namespace

int main() {
    __half* k_dev = nullptr;
    float* out_dev = nullptr;
    int* stats_dev = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&k_dev),
                        kKRows * kKCols * sizeof(__half)),
              "hipMalloc k");
    check_hip(hipMalloc(reinterpret_cast<void**>(&out_dev),
                        kOutModes * kWaveSize * kMmacFloatsPerLane *
                            sizeof(float)),
              "hipMalloc out");
    check_hip(hipMalloc(reinterpret_cast<void**>(&stats_dev),
                        3 * sizeof(int)),
              "hipMalloc stats");

    std::vector<__half> k(kKRows * kKCols, __float2half(0.0f));
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 32; ++col) {
            k[row * kKCols + col] = __float2half(1.0f);
        }
    }
    check_hip(hipMemcpy(k_dev, k.data(), k.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemset(out_dev, 0,
                        kOutModes * kWaveSize * kMmacFloatsPerLane *
                            sizeof(float)),
              "hipMemset out");
    check_hip(hipMemset(stats_dev, 0, 3 * sizeof(int)), "hipMemset stats");

    hipLaunchKernelGGL(dq_native_ds_ring_formula_probe_kernel, dim3(1),
                       dim3(kThreads), 0, 0, k_dev, out_dev, stats_dev);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    std::vector<float> out(kOutModes * kWaveSize * kMmacFloatsPerLane);
    int stats[3] = {};
    check_hip(hipMemcpy(out.data(), out_dev, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy out");
    check_hip(hipMemcpy(stats, stats_dev, sizeof(stats),
                        hipMemcpyDeviceToHost),
              "hipMemcpy stats");

    const bool pass0 = check_mode(out, 0, 0, false, "slot0_low");
    const bool pass1 = check_mode(out, 1, 0, true, "slot0_high");
    const bool pass2 = check_mode(out, 2, 1, false, "slot1_low");
    const bool pass3 = check_mode(out, 3, 1, true, "slot1_high");
    const bool stats_pass = stats[0] == 1 && stats[1] == 2 && stats[2] == 2;
    const bool pass = stats_pass && pass0 && pass1 && pass2 && pass3;
    std::printf(
        "native_ds_ring_formula_result producer_done=%d publisher_done=%d "
        "consumer_done=%d stats_pass=%d pass=%d\n",
        stats[0], stats[1], stats[2], stats_pass ? 1 : 0, pass ? 1 : 0);

    (void)hipFree(stats_dev);
    (void)hipFree(out_dev);
    (void)hipFree(k_dev);
    return pass ? 0 : 2;
}
