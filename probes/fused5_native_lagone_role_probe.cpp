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
constexpr int kProducerWaveBegin = 0;
constexpr int kConsumer0WaveBegin = 4;
constexpr int kConsumer1WaveBegin = 8;
constexpr int kDqWriterWaveBegin = 12;
constexpr int kWavesPerRole = 4;
constexpr int kPanels = 4;
constexpr int kGroups = 2;
constexpr int kWritersPerGroup = 4;
constexpr int kPageBytes = 2048;
constexpr int kGenerationBytes =
    kGroups * kWritersPerGroup * kPageBytes;
constexpr int kProducerVgprs = 32;
constexpr int kConsumerVgprs = 176;
constexpr int kDqWriterVgprs = 96;
constexpr int kConsumerMmacPerPanel = 32;
constexpr int kConsumerMmacPerTile =
    kPanels * kConsumerMmacPerPanel;
constexpr int kDqMmacPerPanel = 16;
constexpr int kDqMmacPerTile = kPanels * kDqMmacPerPanel;
constexpr int kUsefulMmacPerTile =
    8 * kConsumerMmacPerTile + 4 * kDqMmacPerTile;

constexpr int kSourceValues =
    kPanels * kGroups * kWritersPerGroup * kWaveSize * 8;
constexpr int kDqValues =
    kPanels * kWavesPerRole * 16 * 32;
constexpr int kDkValues =
    kGroups * kWritersPerGroup * 16 * 128;
constexpr int kPressureVectors = 24;
constexpr int kPressureValues =
    kGroups * kWritersPerGroup * kWaveSize * kPressureVectors * 4;

static_assert(kGenerationBytes == 16 * 1024);
static_assert(kConsumerMmacPerTile == 128);
static_assert(kDqMmacPerTile == 64);
static_assert(kUsefulMmacPerTile == 1280);
static_assert(kProducerVgprs + 2 * kConsumerVgprs + kDqWriterVgprs ==
              480);

union Fragment {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union Accumulator {
    ins::Vec4F32 f32;
    float scalar[4];
};

template <int Group>
struct Barrier {
    static_assert(Group == 0 || Group == 1);
    static constexpr int kFilled = Group * 2;
    static constexpr int kUsed = kFilled + 1;
};

template <int Group>
__host__ __device__ constexpr int page_offset(int writer) {
    static_assert(Group == 0 || Group == 1);
    return (Group * kWritersPerGroup + writer) * kPageBytes;
}

__host__ __device__ __forceinline__ float dense_value(int panel,
                                                       int group,
                                                       int writer,
                                                       int qrow,
                                                       int krow) {
    const int code =
        (panel * 47 + group * 37 + writer * 29 + qrow * 17 +
         krow * 11 + qrow * krow * 3) %
        31;
    return static_cast<float>(code - 15) * 0.0625f;
}

__host__ __device__ __forceinline__ void source_slot_qk(int lane,
                                                        int word,
                                                        int& qrow,
                                                        int& krow) {
    const int dst_lane =
        ((lane >> 0) & 1) | (((lane >> 1) & 1) << 1) |
        (((lane >> 2) & 1) << 2) | (((lane >> 3) & 1) << 3) |
        (((lane >> 5) & 1) << 4) | (((word >> 1) & 1) << 5);
    const int dst_word =
        ((word >> 0) & 1) | (((lane >> 4) & 1) << 1) |
        (((word >> 2) & 1) << 2);
    qrow = dst_lane & 15;
    krow = (dst_word >= 4 ? 16 : 0) + (dst_lane >> 4) * 4 +
           (dst_word & 3);
}

__device__ __forceinline__ Fragment load_source_fragment(
    const __half* source,
    int panel,
    int group,
    int writer,
    int lane) {
    const int base =
        ((((panel * kGroups + group) * kWritersPerGroup + writer) *
              kWaveSize +
          lane) *
         8);
    Fragment fragment;
    fragment.f16x8 =
        *reinterpret_cast<const ins::Vec8F16*>(source + base);
    return fragment;
}

__device__ __forceinline__ ins::Vec4F16 splat_f16(float value) {
    ins::Vec4F16 result;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        result[i] = static_cast<_Float16>(value);
    }
    return result;
}

template <int Count>
__device__ __forceinline__ void zero_accumulators(
    Accumulator (&acc)[Count]) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
#pragma unroll
    for (int i = 0; i < Count; ++i) {
        acc[i].f32 = zero.f32;
    }
}

__device__ __forceinline__ void run_score_dp_dv_island(
    Accumulator (&score_dp)[16],
    Accumulator (&dv)[8],
    const ins::Vec4F16& ones) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        score_dp[i].f32 =
            ins::mmac_f16_lit(ones, ones, score_dp[i].f32);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dv[i].f32 = ins::mmac_f16_lit(ones, ones, dv[i].f32);
    }
}

template <int Group>
__device__ __forceinline__ void publish_fragment(
    __half* lds,
    const Fragment& fragment,
    int writer) {
    ins::abarrier_seq<false>(Barrier<Group>::kFilled);
    ins::ds_write_matrix_32x16_trans_f16(
        fragment.f16x8, lds, page_offset<Group>(writer));
    ins::wait_lgkm(0);
    ins::abarrier_arrive_cnt<false>(Barrier<Group>::kFilled, 1);
}

template <int Group>
__device__ __forceinline__ void read_dk_fragment(
    const __half* lds,
    int writer,
    Fragment& normal) {
    ins::ds_read_matrix_32x16_normal(
        lds, page_offset<Group>(writer), normal.f16x8);
    ins::wait_lgkm(0);
}

__device__ __forceinline__ void store_dk(
    float* output,
    int group,
    int writer,
    int lane,
    const Accumulator (&dk)[8]) {
    const int krow = lane & 15;
    const int d_lane = (lane >> 4) * 4;
#pragma unroll
    for (int d_block = 0; d_block < 8; ++d_block) {
        const int base =
            (((group * kWritersPerGroup + writer) * 16 + krow) *
                 128 +
             d_block * 16 + d_lane);
        *reinterpret_cast<ins::Vec4F32*>(output + base) =
            dk[d_block].f32;
    }
}

__device__ __forceinline__ void store_pressure(
    float* output,
    int group,
    int writer,
    int lane,
    const Accumulator (&score_dp)[16],
    const Accumulator (&dv)[8]) {
    const int base =
        ((group * kWritersPerGroup + writer) * kWaveSize + lane) *
        kPressureVectors * 4;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        *reinterpret_cast<ins::Vec4F32*>(output + base + i * 4) =
            score_dp[i].f32;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        *reinterpret_cast<ins::Vec4F32*>(
            output + base + (16 + i) * 4) = dv[i].f32;
    }
}

template <int Group>
__device__ __forceinline__ void run_consumer(
    const __half* source,
    __half* lds,
    float* dk_output,
    float* pressure_output,
    int writer,
    int lane) {
    Accumulator score_dp[16];
    Accumulator dv[8];
    Accumulator dk[8];
    zero_accumulators(score_dp);
    zero_accumulators(dv);
    zero_accumulators(dk);
    const ins::Vec4F16 ones = splat_f16(1.0f);

    Fragment current =
        load_source_fragment(source, 0, Group, writer, lane);
    int used_phase = 0;
#pragma clang loop unroll(disable)
    for (int panel = 0; panel < kPanels; ++panel) {
        // Useful next-generation work happens before the page-reuse wait.
        run_score_dp_dv_island(score_dp, dv, ones);
        Fragment next{};
        if (panel + 1 < kPanels) {
            next = load_source_fragment(
                source, panel + 1, Group, writer, lane);
        }
        asm volatile("" : "+v"(current.f16x8), "+v"(next.f16x8) : :
                         "memory");

        if (panel != 0) {
            ins::abarrier_try_wait<true>(
                Barrier<Group>::kUsed, used_phase);
        }
        publish_fragment<Group>(lds, current, writer);

        Fragment normal;
        read_dk_fragment<Group>(lds, writer, normal);
#pragma unroll
        for (int d_block = 0; d_block < 8; ++d_block) {
            const ins::Vec4F16 rhs =
                splat_f16(static_cast<float>(d_block + 1));
            dk[d_block].f32 = ins::mmac_f16_lit(
                normal.f16x4[0], rhs, dk[d_block].f32);
        }
        ins::abarrier_arrive_cnt<false>(
            Barrier<Group>::kUsed, 1);
        current = next;
    }

    int final_used_phase = used_phase;
    ins::abarrier_try_wait<true>(
        Barrier<Group>::kUsed, final_used_phase);
    store_dk(dk_output, Group, writer, lane, dk);
    store_pressure(
        pressure_output, Group, writer, lane, score_dp, dv);
}

template <int Group>
__device__ __forceinline__ void consume_dq_group(
    const __half* lds,
    int& filled_phase,
    int d_block,
    Accumulator (&dq)[2]) {
    ins::abarrier_try_wait<true>(
        Barrier<Group>::kFilled, filled_phase);
    Fragment ds[kWritersPerGroup];
    ins::ds_read_matrix_32x16_trans_imm4<
        page_offset<Group>(0), page_offset<Group>(1),
        page_offset<Group>(2), page_offset<Group>(3)>(
        lds, ds[0].f16x8, ds[1].f16x8, ds[2].f16x8,
        ds[3].f16x8);
    ins::wait_lgkm(0);
    const ins::Vec4F16 rhs0 =
        splat_f16(static_cast<float>(d_block * 2 + 1));
    const ins::Vec4F16 rhs1 =
        splat_f16(static_cast<float>(d_block * 2 + 2));
#pragma unroll
    for (int writer = 0; writer < kWritersPerGroup; ++writer) {
        dq[0].f32 = ins::mmac_f16_lit(
            ds[writer].f16x4[0], rhs0, dq[0].f32);
        dq[1].f32 = ins::mmac_f16_lit(
            ds[writer].f16x4[0], rhs1, dq[1].f32);
    }
    ins::abarrier_arrive_cnt<false>(
        Barrier<Group>::kUsed, 1);
}

__device__ __forceinline__ void store_dq(
    float* output,
    int panel,
    int d_block,
    int lane,
    const Accumulator (&dq)[2]) {
    const int qrow = lane & 15;
    const int d_lane = (lane >> 4) * 4;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int base =
            (((panel * kWavesPerRole + d_block) * 16 + qrow) *
                 32 +
             half * 16 + d_lane);
        *reinterpret_cast<ins::Vec4F32*>(output + base) =
            dq[half].f32;
    }
}

__device__ __forceinline__ void run_dq_writer(
    const __half* lds,
    float* dq_output,
    int d_block,
    int lane) {
    int filled_phase0 = 0;
    int filled_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int panel = 0; panel < kPanels; ++panel) {
        Accumulator dq[2];
        zero_accumulators(dq);
        consume_dq_group<0>(
            lds, filled_phase0, d_block, dq);
        consume_dq_group<1>(
            lds, filled_phase1, d_block, dq);
        store_dq(dq_output, panel, d_block, lane, dq);
    }
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_native_lagone_role_probe_kernel(
    float* __restrict__ dq_output,
    float* __restrict__ dk_output,
    float* __restrict__ pressure_output,
    const __half* __restrict__ source) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(kProducerVgprs, kConsumerVgprs,
                            kConsumerVgprs, kDqWriterVgprs);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half
        lds[kGenerationBytes / sizeof(__half)];
    const int wave =
        static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Barrier<0>::kFilled, 4);
        __builtin_hcu_s_abarrier_init(Barrier<0>::kUsed, 8);
        __builtin_hcu_s_abarrier_init(Barrier<1>::kFilled, 4);
        __builtin_hcu_s_abarrier_init(Barrier<1>::kUsed, 8);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < kConsumer0WaveBegin) {
        __builtin_hcu_s_set_vgpr_size(kProducerVgprs);
    } else if (wave < kConsumer1WaveBegin) {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        const int lane =
            static_cast<int>(threadIdx.x % kWaveSize);
        run_consumer<0>(
            source, lds, dk_output, pressure_output,
            wave - kConsumer0WaveBegin, lane);
    } else if (wave < kDqWriterWaveBegin) {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        const int lane =
            static_cast<int>(threadIdx.x % kWaveSize);
        run_consumer<1>(
            source, lds, dk_output, pressure_output,
            wave - kConsumer1WaveBegin, lane);
    } else {
        __builtin_hcu_s_set_vgpr_size(kDqWriterVgprs);
        const int lane =
            static_cast<int>(threadIdx.x % kWaveSize);
        run_dq_writer(
            lds, dq_output, wave - kDqWriterWaveBegin, lane);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Barrier<0>::kFilled);
        __builtin_hcu_s_abarrier_inv(Barrier<0>::kUsed);
        __builtin_hcu_s_abarrier_inv(Barrier<1>::kFilled);
        __builtin_hcu_s_abarrier_inv(Barrier<1>::kUsed);
    }
#else
    (void)dq_output;
    (void)dk_output;
    (void)pressure_output;
    (void)source;
#endif
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

template <typename T>
T* allocate_device(std::size_t count, const char* what) {
    T* pointer = nullptr;
    check_hip(hipMalloc(&pointer, count * sizeof(T)), what);
    return pointer;
}

}  // namespace

int main() {
    std::vector<__half> source(kSourceValues);
    for (int panel = 0; panel < kPanels; ++panel) {
        for (int group = 0; group < kGroups; ++group) {
            for (int writer = 0; writer < kWritersPerGroup; ++writer) {
                for (int lane = 0; lane < kWaveSize; ++lane) {
                    for (int word = 0; word < 8; ++word) {
                        int qrow = 0;
                        int krow = 0;
                        source_slot_qk(lane, word, qrow, krow);
                        const int index =
                            ((((panel * kGroups + group) *
                                   kWritersPerGroup +
                               writer) *
                                  kWaveSize +
                              lane) *
                                 8 +
                             word);
                        source[index] = static_cast<__half>(
                            krow < 16
                                ? dense_value(panel, group, writer,
                                              qrow, krow)
                                : 0.0f);
                    }
                }
            }
        }
    }

    std::vector<float> dq(kDqValues);
    std::vector<float> dk(kDkValues);
    std::vector<float> pressure(kPressureValues);
    __half* d_source =
        allocate_device<__half>(source.size(), "allocate source");
    float* d_dq = allocate_device<float>(dq.size(), "allocate dQ");
    float* d_dk = allocate_device<float>(dk.size(), "allocate dK");
    float* d_pressure =
        allocate_device<float>(pressure.size(), "allocate pressure");

    check_hip(hipMemcpy(d_source, source.data(),
                        source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy source");
    check_hip(hipMemset(d_dq, 0, dq.size() * sizeof(float)),
              "clear dQ");
    check_hip(hipMemset(d_dk, 0, dk.size() * sizeof(float)),
              "clear dK");
    check_hip(hipMemset(d_pressure, 0,
                        pressure.size() * sizeof(float)),
              "clear pressure");

    hipLaunchKernelGGL(
        fused5_native_lagone_role_probe_kernel, dim3(1),
        dim3(kWaves * kWaveSize), 0, 0, d_dq, d_dk,
        d_pressure, d_source);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(dq.data(), d_dq,
                        dq.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy dQ");
    check_hip(hipMemcpy(dk.data(), d_dk,
                        dk.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy dK");
    check_hip(hipMemcpy(pressure.data(), d_pressure,
                        pressure.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy pressure");

    check_hip(hipFree(d_source), "free source");
    check_hip(hipFree(d_dq), "free dQ");
    check_hip(hipFree(d_dk), "free dK");
    check_hip(hipFree(d_pressure), "free pressure");

    int dq_mismatches = 0;
    for (int panel = 0; panel < kPanels; ++panel) {
        for (int d_block = 0; d_block < kWavesPerRole; ++d_block) {
            for (int qrow = 0; qrow < 16; ++qrow) {
                float row_sum = 0.0f;
                for (int group = 0; group < kGroups; ++group) {
                    for (int writer = 0; writer < kWritersPerGroup;
                         ++writer) {
                        for (int krow = 0; krow < 16; ++krow) {
                            row_sum += dense_value(
                                panel, group, writer, qrow, krow);
                        }
                    }
                }
                for (int d = 0; d < 32; ++d) {
                    const float expected =
                        row_sum *
                        static_cast<float>(d_block * 2 + d / 16 + 1);
                    const int index =
                        (((panel * kWavesPerRole + d_block) * 16 +
                          qrow) *
                             32 +
                         d);
                    dq_mismatches +=
                        std::fabs(dq[index] - expected) > 1.0e-4f;
                }
            }
        }
    }

    int dk_mismatches = 0;
    for (int group = 0; group < kGroups; ++group) {
        for (int writer = 0; writer < kWritersPerGroup; ++writer) {
            for (int krow = 0; krow < 16; ++krow) {
                float column_sum = 0.0f;
                for (int panel = 0; panel < kPanels; ++panel) {
                    for (int qrow = 0; qrow < 16; ++qrow) {
                        column_sum += dense_value(
                            panel, group, writer, qrow, krow);
                    }
                }
                for (int d = 0; d < 128; ++d) {
                    const float expected =
                        column_sum * static_cast<float>(d / 16 + 1);
                    const int index =
                        (((group * kWritersPerGroup + writer) * 16 +
                          krow) *
                             128 +
                         d);
                    dk_mismatches +=
                        std::fabs(dk[index] - expected) > 1.0e-4f;
                }
            }
        }
    }

    int pressure_mismatches = 0;
    for (float value : pressure) {
        pressure_mismatches += std::fabs(value - 64.0f) > 1.0e-4f;
    }
    const bool pass = dq_mismatches == 0 && dk_mismatches == 0 &&
                      pressure_mismatches == 0;
    std::printf(
        "fused5_native_lagone config waves=16 roles=32/176/176/96 "
        "panels=%d lds_bytes=%d mmac_per_tile=%d\n",
        kPanels, kGenerationBytes, kUsefulMmacPerTile);
    std::printf(
        "fused5_native_lagone dq_mismatches=%d dk_mismatches=%d "
        "pressure_mismatches=%d pass=%d\n",
        dq_mismatches, dk_mismatches, pressure_mismatches,
        pass ? 1 : 0);
    return pass ? 0 : 3;
}
