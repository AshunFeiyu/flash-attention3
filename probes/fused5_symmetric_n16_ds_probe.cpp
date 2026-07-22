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
constexpr int kGroups = 2;
constexpr int kConsumersPerGroup = 4;
constexpr int kWaves = kProducerWaves + kGroups * kConsumersPerGroup;
constexpr int kProducerVgprs = 32;
constexpr int kConsumerVgprs = 176;
constexpr int kIterations = 8;
constexpr int kPagesPerGroup = 4;
constexpr int kPageBytes = 2048;
constexpr int kGenerationBytes =
    kGroups * kPagesPerGroup * kPageBytes;
constexpr int kLdsBytes = 2 * kGenerationBytes;

constexpr int kSourceValues =
    kIterations * kGroups * kPagesPerGroup * kWaveSize * 8;
constexpr int kTransViewValues =
    kIterations * kGroups * kConsumersPerGroup * kPagesPerGroup * 16 * 32;
constexpr int kNormalViewValues =
    kIterations * kGroups * kConsumersPerGroup * 16 * 32;
constexpr int kDqValues =
    kIterations * kGroups * kConsumersPerGroup * 16 * 32;
constexpr int kDkValues =
    kIterations * kGroups * kConsumersPerGroup * 16 * 16;
constexpr int kPersistentVectors = 16;
constexpr int kPressureValues =
    kGroups * kConsumersPerGroup * kWaveSize * kPersistentVectors * 4;

static_assert(kWaves * kWaveSize == 768, "probe must use 12 waves");
static_assert(kLdsBytes == 32 * 1024,
              "two generations of eight 2KiB pages must use 32KiB");

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

__host__ __device__ __forceinline__ float dense_value(int iteration,
                                                       int group,
                                                       int writer,
                                                       int qrow,
                                                       int krow) {
    const int code =
        (iteration * 97 + group * 71 + writer * 53 + qrow * 29 +
         krow * 11 + qrow * krow * 3) %
        61;
    return half_round(static_cast<float>(code - 30) * 0.0625f);
}

__host__ __device__ __forceinline__ float padded_value(int iteration,
                                                        int group,
                                                        int writer,
                                                        int qrow,
                                                        int krow) {
    return krow < 16 ? dense_value(iteration, group, writer, qrow, krow)
                     : 0.0f;
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

template <int Group, int Generation>
__host__ __device__ constexpr int page_offset(int writer) {
    static_assert(Group == 0 || Group == 1);
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
    const __half* __restrict__ source,
    int iteration,
    int group,
    int writer,
    int lane) {
    Fragment fragment{};
    const int base =
        ((((iteration * kGroups + group) * kPagesPerGroup + writer) *
           kWaveSize +
          lane) *
         8);
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

__device__ __forceinline__ void store_trans_view(
    float* __restrict__ out,
    const Fragment& fragment,
    int iteration,
    int group,
    int reader,
    int writer,
    int lane) {
    const int qrow = lane & 15;
    const int k_group = (lane >> 4) * 4;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const int krow = (word >= 4 ? 16 : 0) + k_group + (word & 3);
        const int index =
            (((((iteration * kGroups + group) * kConsumersPerGroup + reader) *
                    kPagesPerGroup +
                writer) *
                   16 +
               qrow) *
                  32 +
              krow);
        out[index] = static_cast<float>(fragment.scalar[word]);
    }
}

__device__ __forceinline__ void store_normal_view(
    float* __restrict__ out,
    const Fragment& fragment,
    int iteration,
    int group,
    int reader,
    int lane) {
    const int krow_in_half = lane & 15;
    const int q_base = (lane >> 4) * 4;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
            const int krow = half * 16 + krow_in_half;
            const int index =
                ((((iteration * kGroups + group) * kConsumersPerGroup +
                    reader) *
                       16 +
                   q_base + vec) *
                      32 +
                  krow);
            out[index] = static_cast<float>(fragment.f16x4[half][vec]);
        }
    }
}

__device__ __forceinline__ void store_dq(
    float* __restrict__ out,
    const Accumulator (&acc)[2],
    int iteration,
    int group,
    int reader,
    int lane) {
    const int qrow = lane & 15;
    const int d_group = (lane >> 4) * 4;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
            const int d = half * 16 + d_group + vec;
            const int index =
                ((((iteration * kGroups + group) * kConsumersPerGroup +
                    reader) *
                       16 +
                   qrow) *
                      32 +
                  d);
            out[index] = acc[half].scalar[vec];
        }
    }
}

__device__ __forceinline__ void store_dk(float* __restrict__ out,
                                         const Accumulator& acc,
                                         int iteration,
                                         int group,
                                         int writer,
                                         int lane) {
    const int krow = lane & 15;
    const int d_group = (lane >> 4) * 4;
#pragma unroll
    for (int vec = 0; vec < 4; ++vec) {
        const int index =
            ((((iteration * kGroups + group) * kConsumersPerGroup + writer) *
                   16 +
               krow) *
                  16 +
              d_group + vec);
        out[index] = acc.scalar[vec];
    }
}

template <int Group, int Generation>
__device__ __forceinline__ void process_generation(
    const __half* __restrict__ source,
    __half* __restrict__ lds,
    float* __restrict__ trans_view,
    float* __restrict__ normal_view,
    float* __restrict__ dq_output,
    float* __restrict__ dk_output,
    int reader,
    int iteration,
    int& used_phase,
    int& filled_phase,
    Accumulator (&persistent)[kPersistentVectors],
    const ins::Vec4F16& ones) {
    constexpr int kFilled = Barrier<Group, Generation>::kFilled;
    constexpr int kUsed = Barrier<Group, Generation>::kUsed;

    if (iteration >= 2) {
        ins::abarrier_try_wait<true>(kUsed, used_phase);
    }
    ins::abarrier_seq<false>(kFilled);
    const int lane = threadIdx.x % kWaveSize;
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

#pragma unroll
    for (int writer_index = 0; writer_index < kPagesPerGroup;
         ++writer_index) {
        store_trans_view(trans_view, trans[writer_index], iteration, Group,
                         reader, writer_index, lane);
    }
    store_normal_view(normal_view, normal, iteration, Group, reader, lane);

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    Accumulator dq_acc[2]{};
    dq_acc[0].f32 = zero.f32;
    dq_acc[1].f32 = zero.f32;
    const ins::Vec4F16 rhs0 =
        make_rhs(static_cast<float>(reader * 2 + 1));
    const ins::Vec4F16 rhs1 =
        make_rhs(static_cast<float>(reader * 2 + 2));
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
    store_dq(dq_output, dq_acc, iteration, Group, reader, lane);

    Accumulator dk_acc{};
    dk_acc.f32 = ins::mmac_f16_lit(normal.f16x4[0], ones, zero.f32);
    store_dk(dk_output, dk_acc, iteration, Group, reader, lane);

#pragma unroll
    for (int i = 0; i < kPersistentVectors; ++i) {
        persistent[i].f32 =
            ins::mmac_f16_lit(ones, ones, persistent[i].f32);
    }
}

template <int Group>
__device__ __forceinline__ void run_consumer(
    const __half* source,
    __half* lds,
    float* trans_view,
    float* normal_view,
    float* dq_output,
    float* dk_output,
    float* pressure_output,
    int reader) {
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
            source, lds, trans_view, normal_view, dq_output, dk_output,
            reader, pair * 2, used_phase0, filled_phase0, persistent, ones);
        process_generation<Group, 1>(
            source, lds, trans_view, normal_view, dq_output, dk_output,
            reader, pair * 2 + 1, used_phase1, filled_phase1, persistent,
            ones);
    }

    const int lane = threadIdx.x % kWaveSize;
    const int base =
        ((Group * kConsumersPerGroup + reader) * kWaveSize + lane) *
        kPersistentVectors * 4;
#pragma unroll
    for (int i = 0; i < kPersistentVectors; ++i) {
        *reinterpret_cast<ins::Vec4F32*>(pressure_output + base + i * 4) =
            persistent[i].f32;
    }
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_symmetric_n16_ds_probe_kernel(
    float* __restrict__ trans_view,
    float* __restrict__ normal_view,
    float* __restrict__ dq_output,
    float* __restrict__ dk_output,
    float* __restrict__ pressure_output,
    const __half* __restrict__ source) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(kProducerVgprs, kConsumerVgprs, kConsumerVgprs);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
#pragma unroll
        for (int barrier = 0; barrier < 8; barrier += 2) {
            __builtin_hcu_s_abarrier_init(barrier, kConsumersPerGroup);
            __builtin_hcu_s_abarrier_init(barrier + 1,
                                          kConsumersPerGroup);
        }
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < 4) {
        __builtin_hcu_s_set_vgpr_size(kProducerVgprs);
    } else if (wave < 8) {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        run_consumer<0>(source, lds, trans_view, normal_view, dq_output,
                        dk_output, pressure_output, wave - 4);
    } else {
        __builtin_hcu_s_set_vgpr_size(kConsumerVgprs);
        run_consumer<1>(source, lds, trans_view, normal_view, dq_output,
                        dk_output, pressure_output, wave - 8);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
#pragma unroll
        for (int barrier = 0; barrier < 8; ++barrier) {
            __builtin_hcu_s_abarrier_inv(barrier);
        }
    }
#else
    (void)trans_view;
    (void)normal_view;
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
                                   kPagesPerGroup +
                               writer) *
                                  kWaveSize +
                              lane) *
                                 8 +
                             word);
                        source[index] = static_cast<__half>(padded_value(
                            iteration, group, writer, qrow, krow));
                    }
                }
            }
        }
    }

    std::vector<float> trans_view(kTransViewValues);
    std::vector<float> normal_view(kNormalViewValues);
    std::vector<float> dq_output(kDqValues);
    std::vector<float> dk_output(kDkValues);
    std::vector<float> pressure(kPressureValues);

    __half* d_source = allocate_device<__half>(kSourceValues, "alloc source");
    float* d_trans =
        allocate_device<float>(kTransViewValues, "alloc trans view");
    float* d_normal =
        allocate_device<float>(kNormalViewValues, "alloc normal view");
    float* d_dq = allocate_device<float>(kDqValues, "alloc dQ");
    float* d_dk = allocate_device<float>(kDkValues, "alloc dK");
    float* d_pressure =
        allocate_device<float>(kPressureValues, "alloc pressure");

    check_hip(hipMemcpy(d_source, source.data(), source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy source");
    check_hip(hipMemset(d_trans, 0, trans_view.size() * sizeof(float)),
              "clear trans view");
    check_hip(hipMemset(d_normal, 0, normal_view.size() * sizeof(float)),
              "clear normal view");
    check_hip(hipMemset(d_dq, 0, dq_output.size() * sizeof(float)),
              "clear dQ");
    check_hip(hipMemset(d_dk, 0, dk_output.size() * sizeof(float)),
              "clear dK");
    check_hip(hipMemset(d_pressure, 0, pressure.size() * sizeof(float)),
              "clear pressure");

    hipLaunchKernelGGL(fused5_symmetric_n16_ds_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, d_trans, d_normal,
                       d_dq, d_dk, d_pressure, d_source);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(trans_view.data(), d_trans,
                        trans_view.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy trans view");
    check_hip(hipMemcpy(normal_view.data(), d_normal,
                        normal_view.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy normal view");
    check_hip(hipMemcpy(dq_output.data(), d_dq,
                        dq_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy dQ");
    check_hip(hipMemcpy(dk_output.data(), d_dk,
                        dk_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy dK");
    check_hip(hipMemcpy(pressure.data(), d_pressure,
                        pressure.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy pressure");

    check_hip(hipFree(d_source), "free source");
    check_hip(hipFree(d_trans), "free trans view");
    check_hip(hipFree(d_normal), "free normal view");
    check_hip(hipFree(d_dq), "free dQ");
    check_hip(hipFree(d_dk), "free dK");
    check_hip(hipFree(d_pressure), "free pressure");

    int trans_mismatches = 0;
    int normal_mismatches = 0;
    int dq_mismatches = 0;
    int dk_mismatches = 0;
    int pressure_mismatches = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int group = 0; group < kGroups; ++group) {
            for (int reader = 0; reader < kConsumersPerGroup; ++reader) {
                for (int writer = 0; writer < kPagesPerGroup; ++writer) {
                    for (int qrow = 0; qrow < 16; ++qrow) {
                        for (int krow = 0; krow < 32; ++krow) {
                            const int index =
                                (((((iteration * kGroups + group) *
                                        kConsumersPerGroup +
                                    reader) *
                                       kPagesPerGroup +
                                   writer) *
                                      16 +
                                  qrow) *
                                     32 +
                                 krow);
                            trans_mismatches +=
                                trans_view[index] != padded_value(
                                                         iteration, group,
                                                         writer, qrow, krow);
                        }
                    }
                }
                for (int qrow = 0; qrow < 16; ++qrow) {
                    for (int krow = 0; krow < 32; ++krow) {
                        const int index =
                            ((((iteration * kGroups + group) *
                                   kConsumersPerGroup +
                               reader) *
                                  16 +
                              qrow) *
                                 32 +
                             krow);
                        normal_mismatches +=
                            normal_view[index] != padded_value(
                                                      iteration, group,
                                                      reader, qrow, krow);
                    }
                }
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
                                   kConsumersPerGroup +
                               reader) *
                                  16 +
                              qrow) *
                                 32 +
                             d);
                        const float scale =
                            static_cast<float>(reader * 2 + d / 16 + 1);
                        dq_mismatches +=
                            std::fabs(dq_output[index] - row_sum * scale) >
                            1.0e-4f;
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
                                   kConsumersPerGroup +
                               reader) *
                                  16 +
                              krow) *
                                 16 +
                             d);
                        dk_mismatches +=
                            std::fabs(dk_output[index] - column_sum) >
                            1.0e-4f;
                    }
                }
            }
        }
    }
    for (float value : pressure) {
        pressure_mismatches += value != 128.0f;
    }

    const bool pass = trans_mismatches == 0 && normal_mismatches == 0 &&
                      dq_mismatches == 0 && dk_mismatches == 0 &&
                      pressure_mismatches == 0;
    std::printf(
        "fused5_symmetric_n16_ds config waves=12 roles=32/176/176 "
        "groups=2 owners=4 iterations=%d generations=2 page_bytes=%d "
        "lds_bytes=%d\n",
        kIterations, kPageBytes, kLdsBytes);
    std::printf(
        "fused5_symmetric_n16_ds trans_mismatches=%d "
        "normal_mismatches=%d dq_mismatches=%d dk_mismatches=%d "
        "pressure_mismatches=%d pass=%d\n",
        trans_mismatches, normal_mismatches, dq_mismatches, dk_mismatches,
        pressure_mismatches, pass ? 1 : 0);
    return pass ? 0 : 3;
}
