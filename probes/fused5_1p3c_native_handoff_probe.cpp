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
constexpr int kProducerWaves = 4;
constexpr int kGroups = 3;
constexpr int kWavesPerGroup = 4;
constexpr int kWaves = kProducerWaves + kGroups * kWavesPerGroup;
constexpr int kProducerVgprs = 32;
constexpr int kConsumerVgprs = 160;
constexpr int kIterations = 8;
constexpr int kPagesPerGroup = 4;
constexpr int kPageBytes = 2048;
constexpr int kGenerationBytes = kGroups * kPagesPerGroup * kPageBytes;
constexpr int kLdsBytes = 2 * kGenerationBytes;
// 112 persistent VGPRs model dK+dV (64), K/V trans fragments (32), and the
// K normal view needed by dQ (16). The native dS/dQ path below adds its real
// transient fragments on top of this pressure rather than testing an empty
// role window.
constexpr int kPersistentVectors = 28;

constexpr int kSourceValues =
    kIterations * kGroups * kPagesPerGroup * kWaveSize * 8;
constexpr int kDqValues =
    kIterations * kGroups * kWavesPerGroup * 16 * 32;
constexpr int kDkValues =
    kIterations * kGroups * kWavesPerGroup * 16 * 16;
constexpr int kPressureValues =
    kGroups * kWavesPerGroup * kWaveSize * kPersistentVectors * 4;

static_assert(kWaves == 16 && kWaves * kWaveSize == 1024);
static_assert(kProducerVgprs + kGroups * kConsumerVgprs == 512);
static_assert(kLdsBytes == 48 * 1024);

union Fragment {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union Accumulator {
    ins::Vec4F32 f32;
    float scalar[4];
};

__host__ __device__ __forceinline__ float half_round(float value) {
    return static_cast<float>(static_cast<_Float16>(value));
}

__host__ __device__ __forceinline__ float dense_value(
    int iteration, int group, int writer, int qrow, int krow) {
    const int code =
        (iteration * 97 + group * 71 + writer * 53 + qrow * 29 +
         krow * 11 + qrow * krow * 3) % 61;
    return half_round(static_cast<float>(code - 30) * 0.0625f);
}

__host__ __device__ __forceinline__ float padded_value(
    int iteration, int group, int writer, int qrow, int krow) {
    return krow < 16 ? dense_value(iteration, group, writer, qrow, krow)
                     : 0.0f;
}

__host__ __device__ __forceinline__ void source_slot_qk(
    int lane, int word, int& qrow, int& krow) {
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

template <int Group, int Generation>
__host__ __device__ constexpr int page_offset(int writer) {
    static_assert(Group >= 0 && Group < kGroups);
    static_assert(Generation == 0 || Generation == 1);
    return Generation * kGenerationBytes +
           (Group * kPagesPerGroup + writer) * kPageBytes;
}

template <int Group, int Generation>
struct Barrier {
    static constexpr int kFilled = Group * 4 + Generation * 2;
    static constexpr int kUsed = kFilled + 1;
};

__device__ __forceinline__ Fragment load_writer_fragment(
    const __half* source, int iteration, int group, int writer, int lane) {
    Fragment fragment{};
    const int base =
        ((((iteration * kGroups + group) * kPagesPerGroup + writer) *
           kWaveSize + lane) * 8);
    fragment.f16x8 =
        *reinterpret_cast<const ins::Vec8F16*>(source + base);
    return fragment;
}

__device__ __forceinline__ ins::Vec4F16 make_rhs(float scale) {
    ins::Vec4F16 rhs{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rhs[i] = static_cast<_Float16>(scale);
    }
    return rhs;
}

template <int Group, int Generation>
__device__ __forceinline__ void process_generation(
    const __half* source, __half* lds, float* dq_output,
    float* dk_output, int reader, int iteration, int& used_phase,
    int& filled_phase, Accumulator (&persistent)[kPersistentVectors],
    const ins::Vec4F16& ones) {
    constexpr int kFilled = Barrier<Group, Generation>::kFilled;
    constexpr int kUsed = Barrier<Group, Generation>::kUsed;
    const int lane = threadIdx.x % kWaveSize;

    if (iteration >= 2) {
        ins::abarrier_try_wait<true>(kUsed, used_phase);
    }
    ins::abarrier_seq<false>(kFilled);
    const Fragment writer =
        load_writer_fragment(source, iteration, Group, reader, lane);
    ins::ds_write_matrix_32x16_trans_f16(
        writer.f16x8, lds, page_offset<Group, Generation>(reader));
    ins::wait_lgkm(0);
    ins::abarrier_arrive_cnt<false>(kFilled, 1);

    ins::abarrier_try_wait<true>(kFilled, filled_phase);
    Fragment trans[kPagesPerGroup];
    ins::ds_read_matrix_32x16_trans_imm4<
        page_offset<Group, Generation>(0),
        page_offset<Group, Generation>(1),
        page_offset<Group, Generation>(2),
        page_offset<Group, Generation>(3)>(
        lds, trans[0].f16x8, trans[1].f16x8, trans[2].f16x8,
        trans[3].f16x8);
    Fragment normal{};
    ins::ds_read_matrix_32x16_normal(
        lds, page_offset<Group, Generation>(reader), normal.f16x8);
    ins::wait_lgkm(0);
    ins::abarrier_arrive_cnt<false>(kUsed, 1);

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    Accumulator dq_acc[2]{};
    dq_acc[0].f32 = zero.f32;
    dq_acc[1].f32 = zero.f32;
    const ins::Vec4F16 rhs0 = make_rhs(static_cast<float>(reader * 2 + 1));
    const ins::Vec4F16 rhs1 = make_rhs(static_cast<float>(reader * 2 + 2));
#pragma unroll
    for (int writer_index = 0; writer_index < kPagesPerGroup;
         ++writer_index) {
        dq_acc[0].f32 = ins::mmac_f16_lit(
            trans[writer_index].f16x4[0], rhs0, dq_acc[0].f32);
        dq_acc[0].f32 = ins::mmac_f16_lit(
            trans[writer_index].f16x4[1], rhs0, dq_acc[0].f32);
        dq_acc[1].f32 = ins::mmac_f16_lit(
            trans[writer_index].f16x4[0], rhs1, dq_acc[1].f32);
        dq_acc[1].f32 = ins::mmac_f16_lit(
            trans[writer_index].f16x4[1], rhs1, dq_acc[1].f32);
    }

    const int qrow = lane & 15;
    const int d_group = (lane >> 4) * 4;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float* output =
            dq_output +
            ((((iteration * kGroups + Group) * kWavesPerGroup + reader) *
               16 + qrow) * 32 + half * 16 + d_group);
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
            (void)__builtin_hcu_global_atomic_fadd_f32(
                output + vec, dq_acc[half].scalar[vec]);
        }
    }

    Accumulator dk_acc{};
    dk_acc.f32 = ins::mmac_f16_lit(normal.f16x4[0], ones, zero.f32);
    const int krow = lane & 15;
    *reinterpret_cast<ins::Vec4F32*>(
        dk_output +
        ((((iteration * kGroups + Group) * kWavesPerGroup + reader) * 16 +
           krow) * 16 + d_group)) = dk_acc.f32;

#pragma unroll
    for (int i = 0; i < kPersistentVectors; ++i) {
        persistent[i].f32 =
            ins::mmac_f16_lit(ones, ones, persistent[i].f32);
    }
}

template <int Group>
__device__ __forceinline__ void run_consumer(
    const __half* source, __half* lds, float* dq_output,
    float* dk_output, float* pressure_output, int reader) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    Accumulator persistent[kPersistentVectors];
#pragma unroll
    for (int i = 0; i < kPersistentVectors; ++i) {
        persistent[i].f32 = zero.f32;
    }
    const ins::Vec4F16 ones = make_rhs(1.0f);
    int used_phase0 = 0;
    int used_phase1 = 0;
    int filled_phase0 = 0;
    int filled_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int pair = 0; pair < kIterations / 2; ++pair) {
        process_generation<Group, 0>(
            source, lds, dq_output, dk_output, reader, pair * 2,
            used_phase0, filled_phase0, persistent, ones);
        process_generation<Group, 1>(
            source, lds, dq_output, dk_output, reader, pair * 2 + 1,
            used_phase1, filled_phase1, persistent, ones);
    }

    const int lane = threadIdx.x % kWaveSize;
    const int base =
        ((Group * kWavesPerGroup + reader) * kWaveSize + lane) *
        kPersistentVectors * 4;
#pragma unroll
    for (int i = 0; i < kPersistentVectors; ++i) {
        *reinterpret_cast<ins::Vec4F32*>(
            pressure_output + base + i * 4) = persistent[i].f32;
    }
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_1p3c_native_handoff_probe_kernel(
    float* __restrict__ dq_output, float* __restrict__ dk_output,
    float* __restrict__ pressure_output,
    const __half* __restrict__ source) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(kProducerVgprs, kConsumerVgprs,
                            kConsumerVgprs, kConsumerVgprs);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
#pragma unroll
        for (int barrier = 0; barrier < 12; ++barrier) {
            __builtin_hcu_s_abarrier_init(barrier, kWavesPerGroup);
        }
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < 4) {
        __builtin_hcu_s_set_vgpr_size(kProducerVgprs);
    } else if (wave < 8) {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        run_consumer<0>(source, lds, dq_output, dk_output,
                        pressure_output, wave - 4);
    } else if (wave < 12) {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        run_consumer<1>(source, lds, dq_output, dk_output,
                        pressure_output, wave - 8);
    } else {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        run_consumer<2>(source, lds, dq_output, dk_output,
                        pressure_output, wave - 12);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
#pragma unroll
        for (int barrier = 0; barrier < 12; ++barrier) {
            __builtin_hcu_s_abarrier_inv(barrier);
        }
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
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int group = 0; group < kGroups; ++group) {
            for (int writer = 0; writer < kPagesPerGroup; ++writer) {
                for (int lane = 0; lane < kWaveSize; ++lane) {
                    for (int word = 0; word < 8; ++word) {
                        int qrow = 0;
                        int krow = 0;
                        source_slot_qk(lane, word, qrow, krow);
                        const int index =
                            ((((iteration * kGroups + group) *
                               kPagesPerGroup + writer) * kWaveSize + lane) *
                             8 + word);
                        source[index] = static_cast<__half>(padded_value(
                            iteration, group, writer, qrow, krow));
                    }
                }
            }
        }
    }

    std::vector<float> dq(kDqValues);
    std::vector<float> dk(kDkValues);
    std::vector<float> pressure(kPressureValues);
    __half* d_source = allocate_device<__half>(kSourceValues, "alloc source");
    float* d_dq = allocate_device<float>(kDqValues, "alloc dQ");
    float* d_dk = allocate_device<float>(kDkValues, "alloc dK");
    float* d_pressure =
        allocate_device<float>(kPressureValues, "alloc pressure");

    check_hip(hipMemcpy(d_source, source.data(), source.size() * sizeof(__half),
                        hipMemcpyHostToDevice), "copy source");
    check_hip(hipMemset(d_dq, 0, dq.size() * sizeof(float)), "clear dQ");
    check_hip(hipMemset(d_dk, 0, dk.size() * sizeof(float)), "clear dK");
    check_hip(hipMemset(d_pressure, 0, pressure.size() * sizeof(float)),
              "clear pressure");

    hipLaunchKernelGGL(fused5_1p3c_native_handoff_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, d_dq, d_dk,
                       d_pressure, d_source);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(dq.data(), d_dq, dq.size() * sizeof(float),
                        hipMemcpyDeviceToHost), "copy dQ");
    check_hip(hipMemcpy(dk.data(), d_dk, dk.size() * sizeof(float),
                        hipMemcpyDeviceToHost), "copy dK");
    check_hip(hipMemcpy(pressure.data(), d_pressure,
                        pressure.size() * sizeof(float),
                        hipMemcpyDeviceToHost), "copy pressure");

    int dq_mismatches = 0;
    int dk_mismatches = 0;
    int pressure_mismatches = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int group = 0; group < kGroups; ++group) {
            for (int reader = 0; reader < kWavesPerGroup; ++reader) {
                for (int qrow = 0; qrow < 16; ++qrow) {
                    float row_sum = 0.0f;
                    for (int writer = 0; writer < kPagesPerGroup; ++writer) {
                        for (int krow = 0; krow < 16; ++krow) {
                            row_sum += dense_value(iteration, group, writer,
                                                   qrow, krow);
                        }
                    }
                    for (int d = 0; d < 32; ++d) {
                        const int index =
                            ((((iteration * kGroups + group) *
                               kWavesPerGroup + reader) * 16 + qrow) * 32 + d);
                        const float scale =
                            static_cast<float>(reader * 2 + d / 16 + 1);
                        dq_mismatches +=
                            std::fabs(dq[index] - row_sum * scale) > 1.0e-4f;
                    }
                }
                for (int krow = 0; krow < 16; ++krow) {
                    float column_sum = 0.0f;
                    for (int qrow = 0; qrow < 16; ++qrow) {
                        column_sum += dense_value(iteration, group, reader,
                                                  qrow, krow);
                    }
                    for (int d = 0; d < 16; ++d) {
                        const int index =
                            ((((iteration * kGroups + group) *
                               kWavesPerGroup + reader) * 16 + krow) * 16 + d);
                        dk_mismatches +=
                            std::fabs(dk[index] - column_sum) > 1.0e-4f;
                    }
                }
            }
        }
    }
    for (float value : pressure) {
        pressure_mismatches += value != 128.0f;
    }

    const int pass =
        dq_mismatches == 0 && dk_mismatches == 0 && pressure_mismatches == 0;
    std::printf(
        "fused5_1p3c_native_handoff config waves=16 roles=32/160/160/160 "
        "groups=3 lds_bytes=%d dq_mismatches=%d dk_mismatches=%d "
        "pressure_mismatches=%d pass=%d\n",
        kLdsBytes, dq_mismatches, dk_mismatches, pressure_mismatches, pass);

    check_hip(hipFree(d_source), "free source");
    check_hip(hipFree(d_dq), "free dQ");
    check_hip(hipFree(d_dk), "free dK");
    check_hip(hipFree(d_pressure), "free pressure");
    return pass ? 0 : 1;
}
