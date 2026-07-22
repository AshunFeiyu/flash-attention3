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
constexpr int kDkvWaves = 4;
constexpr int kDqWaves = 4;
constexpr int kWaves = kProducerWaves + kDkvWaves + kDqWaves;
constexpr int kProducerVgprs = 24;
constexpr int kDkvVgprs = 240;
constexpr int kDqVgprs = 96;
constexpr int kIterations = 8;
constexpr int kPages = 4;
constexpr int kPageBytes = 2048;
constexpr int kGenerationBytes = kPages * kPageBytes;
constexpr int kLdsBytes = 2 * kGenerationBytes;
constexpr int kOutputValues = kIterations * kDqWaves * kWaveSize * 8;
constexpr int kViewValues =
    kIterations * kDqWaves * kPages * 16 * 32;
constexpr int kSourceValues = kIterations * kPages * kWaveSize * 8;
constexpr int kPressureValues = kDkvWaves * kWaveSize * 128;

static_assert(kWaves * kWaveSize == 768, "probe must use 12 waves");
static_assert(kLdsBytes == 16 * 1024,
              "two generations of four 2KiB pages must use 16KiB");

struct Barrier {
    static constexpr int kFilled0 = 0;
    static constexpr int kUsed0 = 1;
    static constexpr int kFilled1 = 2;
    static constexpr int kUsed1 = 3;
};

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
                                                       int writer,
                                                       int qrow,
                                                       int krow) {
    const int code =
        (iteration * 97 + writer * 53 + qrow * 29 + krow * 11 +
         qrow * krow * 3) %
        61;
    return half_round(static_cast<float>(code - 30) * 0.0625f);
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

__device__ __forceinline__ int page_offset(int generation, int writer) {
    return generation * kGenerationBytes + writer * kPageBytes;
}

__device__ __forceinline__ Fragment load_writer_fragment(
    const __half* __restrict__ source,
    int iteration,
    int writer,
    int lane) {
    Fragment fragment{};
    const int base =
        ((iteration * kPages + writer) * kWaveSize + lane) * 8;
    fragment.f16x8 = *reinterpret_cast<const ins::Vec8F16*>(source + base);
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
    float* __restrict__ view,
    const Fragment& fragment,
    int iteration,
    int reader,
    int writer,
    int lane) {
    const int qrow = lane & 15;
    const int k_group = (lane >> 4) * 4;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const int krow = (word >= 4 ? 16 : 0) + k_group + (word & 3);
        const int index =
            ((((iteration * kDqWaves + reader) * kPages + writer) * 16 +
              qrow) *
                 32 +
             krow);
        view[index] = static_cast<float>(fragment.scalar[word]);
    }
}

template <int Generation>
__device__ __forceinline__ void write_generation(
    const __half* __restrict__ source,
    __half* __restrict__ lds,
    int writer,
    int iteration,
    int& used_phase,
    const ins::Vec4F16& ones,
    ins::F32x4 (&pressure)[32]) {
    static_assert(Generation == 0 || Generation == 1);
    constexpr int kFilled =
        Generation == 0 ? Barrier::kFilled0 : Barrier::kFilled1;
    constexpr int kUsed = Generation == 0 ? Barrier::kUsed0 : Barrier::kUsed1;

    if (iteration >= 2) {
        ins::abarrier_try_wait<true>(kUsed, used_phase);
    }
    ins::abarrier_seq<false>(kFilled);
    const Fragment fragment = load_writer_fragment(
        source, iteration, writer, threadIdx.x % kWaveSize);
    ins::ds_write_matrix_32x16_trans_f16(
        fragment.f16x8, lds, page_offset(Generation, writer));
    ins::wait_lgkm(0);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        pressure[i].f32 =
            ins::mmac_f16_lit(ones, ones, pressure[i].f32);
    }
    ins::abarrier_arrive_cnt<false>(kFilled, 1);
}

template <int Generation>
__device__ __forceinline__ void read_generation(
    float* __restrict__ output,
    float* __restrict__ view,
    const __half* __restrict__ lds,
    int reader,
    int iteration,
    int& filled_phase) {
    static_assert(Generation == 0 || Generation == 1);
    constexpr int kFilled =
        Generation == 0 ? Barrier::kFilled0 : Barrier::kFilled1;
    constexpr int kUsed = Generation == 0 ? Barrier::kUsed0 : Barrier::kUsed1;

    ins::abarrier_try_wait<true>(kFilled, filled_phase);
    Fragment ds[kPages];
    ins::ds_read_matrix_32x16_trans(
        lds, page_offset(Generation, 0), ds[0].f16x8);
    ins::ds_read_matrix_32x16_trans(
        lds, page_offset(Generation, 1), ds[1].f16x8);
    ins::ds_read_matrix_32x16_trans(
        lds, page_offset(Generation, 2), ds[2].f16x8);
    ins::ds_read_matrix_32x16_trans(
        lds, page_offset(Generation, 3), ds[3].f16x8);
    ins::wait_lgkm(0);

    // All four pages are now private VGPR state. Release the LDS generation
    // before the MMAC island so the writer can publish iteration+2 in parallel.
    ins::abarrier_arrive_cnt<false>(kUsed, 1);

#pragma unroll
    for (int writer = 0; writer < kPages; ++writer) {
        store_trans_view(view, ds[writer], iteration, reader, writer,
                         threadIdx.x % kWaveSize);
    }

    Accumulator acc[2]{};
    const ins::Vec4F16 rhs0 = make_rhs(static_cast<float>(reader * 2 + 1));
    const ins::Vec4F16 rhs1 = make_rhs(static_cast<float>(reader * 2 + 2));
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    acc[0].f32 = zero.f32;
    acc[1].f32 = zero.f32;
#pragma unroll
    for (int writer = 0; writer < kPages; ++writer) {
        acc[0].f32 = ins::mmac_f16_lit(ds[writer].f16x4[0], rhs0,
                                      acc[0].f32);
        acc[0].f32 = ins::mmac_f16_lit(ds[writer].f16x4[1], rhs0,
                                      acc[0].f32);
        acc[1].f32 = ins::mmac_f16_lit(ds[writer].f16x4[0], rhs1,
                                      acc[1].f32);
        acc[1].f32 = ins::mmac_f16_lit(ds[writer].f16x4[1], rhs1,
                                      acc[1].f32);
    }

    const int lane = threadIdx.x % kWaveSize;
    const int base =
        ((iteration * kDqWaves + reader) * kWaveSize + lane) * 8;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            output[base + half * 4 + i] = acc[half].scalar[i];
        }
    }
}

__device__ __forceinline__ void run_dkv(const __half* source,
                                        float* pressure_output,
                                        __half* lds,
    int writer) {
    ins::F32x4 pressure[32];
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    const ins::Vec4F16 ones = make_rhs(1.0f);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        pressure[i].f32 = zero.f32;
    }

    int used_phase0 = 0;
    int used_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int pair = 0; pair < kIterations / 2; ++pair) {
        write_generation<0>(source, lds, writer, pair * 2, used_phase0, ones,
                            pressure);
        write_generation<1>(source, lds, writer, pair * 2 + 1, used_phase1,
                            ones, pressure);
    }

    const int pressure_base =
        (writer * kWaveSize + threadIdx.x % kWaveSize) * 128;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        *reinterpret_cast<ins::Vec4F32*>(pressure_output + pressure_base +
                                        i * 4) = pressure[i].f32;
    }
}

__device__ __forceinline__ void run_dq(float* output,
                                       float* view,
                                       const __half* lds,
                                       int reader) {
    int filled_phase0 = 0;
    int filled_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int pair = 0; pair < kIterations / 2; ++pair) {
        read_generation<0>(output, view, lds, reader, pair * 2,
                           filled_phase0);
        read_generation<1>(output, view, lds, reader, pair * 2 + 1,
                           filled_phase1);
    }
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_ds_conveyor_probe_kernel(float* __restrict__ output,
                                float* __restrict__ view,
                                float* __restrict__ pressure_output,
                                const __half* __restrict__ source) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(kProducerVgprs, kDkvVgprs, kDqVgprs);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Barrier::kFilled0, kDkvWaves);
        __builtin_hcu_s_abarrier_init(Barrier::kUsed0, kDqWaves);
        __builtin_hcu_s_abarrier_init(Barrier::kFilled1, kDkvWaves);
        __builtin_hcu_s_abarrier_init(Barrier::kUsed1, kDqWaves);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < 4) {
        __builtin_hcu_s_set_vgpr_size(kProducerVgprs);
    } else if (wave < 8) {
        __builtin_hcu_s_set_vgpr_size(kDkvVgprs);
        run_dkv(source, pressure_output, lds, wave - 4);
    } else {
        __builtin_hcu_s_set_vgpr_size(kDqVgprs);
        run_dq(output, view, lds, wave - 8);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Barrier::kFilled0);
        __builtin_hcu_s_abarrier_inv(Barrier::kUsed0);
        __builtin_hcu_s_abarrier_inv(Barrier::kFilled1);
        __builtin_hcu_s_abarrier_inv(Barrier::kUsed1);
    }
#else
    (void)output;
    (void)view;
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

}  // namespace

int main() {
    float* output_device = nullptr;
    float* view_device = nullptr;
    float* pressure_device = nullptr;
    __half* source_device = nullptr;
    std::vector<float> output(kOutputValues);
    std::vector<float> view(kViewValues);
    std::vector<float> pressure(kPressureValues);
    std::vector<__half> source(kSourceValues);
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int writer = 0; writer < kPages; ++writer) {
            for (int lane = 0; lane < kWaveSize; ++lane) {
                for (int word = 0; word < 8; ++word) {
                    int qrow = 0;
                    int krow = 0;
                    source_slot_qk(lane, word, qrow, krow);
                    const int index =
                        ((iteration * kPages + writer) * kWaveSize + lane) *
                            8 +
                        word;
                    source[index] = static_cast<__half>(
                        dense_value(iteration, writer, qrow, krow));
                }
            }
        }
    }
    check_hip(hipMalloc(&output_device, output.size() * sizeof(float)),
              "hipMalloc output");
    check_hip(hipMalloc(&view_device, view.size() * sizeof(float)),
              "hipMalloc view");
    check_hip(hipMalloc(&pressure_device, pressure.size() * sizeof(float)),
              "hipMalloc pressure");
    check_hip(hipMalloc(&source_device, source.size() * sizeof(__half)),
              "hipMalloc source");
    check_hip(hipMemset(output_device, 0, output.size() * sizeof(float)),
              "hipMemset output");
    check_hip(hipMemset(view_device, 0, view.size() * sizeof(float)),
              "hipMemset view");
    check_hip(hipMemset(pressure_device, 0,
                        pressure.size() * sizeof(float)),
              "hipMemset pressure");
    check_hip(hipMemcpy(source_device, source.data(),
                        source.size() * sizeof(__half), hipMemcpyHostToDevice),
              "copy source");

    hipLaunchKernelGGL(fused5_ds_conveyor_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, output_device,
                       view_device, pressure_device, source_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(output.data(), output_device,
                        output.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipMemcpy(view.data(), view_device, view.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy view");
    check_hip(hipMemcpy(pressure.data(), pressure_device,
                        pressure.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy pressure");
    check_hip(hipFree(output_device), "hipFree output");
    check_hip(hipFree(view_device), "hipFree view");
    check_hip(hipFree(pressure_device), "hipFree pressure");
    check_hip(hipFree(source_device), "hipFree source");

    int view_mismatches = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int reader = 0; reader < kDqWaves; ++reader) {
            for (int writer = 0; writer < kPages; ++writer) {
                for (int qrow = 0; qrow < 16; ++qrow) {
                    for (int krow = 0; krow < 32; ++krow) {
                        const int index =
                            ((((iteration * kDqWaves + reader) * kPages +
                               writer) *
                                  16 +
                              qrow) *
                                 32 +
                             krow);
                        const float expected =
                            dense_value(iteration, writer, qrow, krow);
                        view_mismatches += view[index] != expected;
                    }
                }
            }
        }
    }

    int output_mismatches = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int reader = 0; reader < kDqWaves; ++reader) {
            for (int lane = 0; lane < kWaveSize; ++lane) {
                const int qrow = lane & 15;
                float row_sum = 0.0f;
                for (int writer = 0; writer < kPages; ++writer) {
                    for (int krow = 0; krow < 32; ++krow) {
                        row_sum += dense_value(iteration, writer, qrow, krow);
                    }
                }
                const int base =
                    ((iteration * kDqWaves + reader) * kWaveSize + lane) * 8;
                for (int half = 0; half < 2; ++half) {
                    const float expected =
                        row_sum * static_cast<float>(reader * 2 + half + 1);
                    for (int i = 0; i < 4; ++i) {
                        output_mismatches +=
                            std::fabs(output[base + half * 4 + i] - expected) >
                            1.0e-4f;
                    }
                }
            }
        }
    }

    int pressure_mismatches = 0;
    for (int index = 0; index < kPressureValues; ++index) {
        pressure_mismatches += pressure[index] != 128.0f;
    }

    std::printf(
        "fused5_ds_conveyor config dense=1 waves=12 roles=24/240/96 iterations=%d "
        "generations=2 pages=4 page_bytes=%d lds_bytes=%d\n",
        kIterations, kPageBytes, kLdsBytes);
    std::printf(
        "fused5_ds_conveyor view_mismatches=%d output_mismatches=%d "
        "pressure_mismatches=%d pass=%d\n",
        view_mismatches, output_mismatches, pressure_mismatches,
        view_mismatches == 0 && output_mismatches == 0 &&
                pressure_mismatches == 0
            ? 1
            : 0);
    return view_mismatches == 0 && output_mismatches == 0 &&
                   pressure_mismatches == 0
               ? 0
               : 3;
}
