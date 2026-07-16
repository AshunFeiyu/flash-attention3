#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

#ifndef PDS_PROBE_HIGH_PRESSURE
#define PDS_PROBE_HIGH_PRESSURE 0
#endif

#ifndef PDS_PROBE_SPLIT_OUTPUT_READERS
#define PDS_PROBE_SPLIT_OUTPUT_READERS 0
#endif

#ifndef PDS_PROBE_ACCUM_F32X4
#define PDS_PROBE_ACCUM_F32X4 32
#endif

#ifndef PDS_PROBE_READER_VGPRS
#define PDS_PROBE_READER_VGPRS 216
#endif

#ifndef PDS_PROBE_MODE
#define PDS_PROBE_MODE 0
#endif

#ifndef PDS_PROBE_MMAC_SOURCE
#define PDS_PROBE_MMAC_SOURCE 0
#endif

#ifndef PDS_PROBE_PACK_MODE
#define PDS_PROBE_PACK_MODE 0
#endif

#ifndef PDS_PROBE_WRITER_ROW
#define PDS_PROBE_WRITER_ROW 2
#endif

#ifndef PDS_PROBE_WRITER_COL
#define PDS_PROBE_WRITER_COL 1
#endif

#ifndef PDS_PROBE_WRITER_TRANS
#define PDS_PROBE_WRITER_TRANS 0
#endif

#ifndef PDS_PROBE_WRITER_ALT
#define PDS_PROBE_WRITER_ALT 0
#endif

#ifndef PDS_PROBE_READER_TRANS
#define PDS_PROBE_READER_TRANS 1
#endif

#ifndef PDS_PROBE_READER_ROW
#define PDS_PROBE_READER_ROW 1
#endif

#ifndef PDS_PROBE_READER_COL
#define PDS_PROBE_READER_COL 2
#endif

#ifndef PDS_PROBE_READER_ALT
#define PDS_PROBE_READER_ALT 0
#endif

#ifndef PDS_PROBE_ITERATIONS
#define PDS_PROBE_ITERATIONS 8
#endif

constexpr int kWaveSize = 64;
constexpr int kProducerWaves = 4;
constexpr int kReaderGroups = PDS_PROBE_SPLIT_OUTPUT_READERS ? 2 : 1;
constexpr int kWavesPerReaderGroup = 4;
constexpr int kConsumerWaves = kReaderGroups * kWavesPerReaderGroup;
constexpr int kProducerWaveBase = 4;
constexpr int kConsumerWaveBase = 8;
constexpr int kWaves = 16;
constexpr int kOwnersPerWave = 2;
constexpr int kPageBytes = 2048;
constexpr int kPairBytes = 2 * kPageBytes;
constexpr int kFilledBarrierId = 4;
constexpr int kUsedBarrierId = 8;
constexpr int kIterations = PDS_PROBE_ITERATIONS;
constexpr int kHighBaseBytes = 66 * 1024;
constexpr int kLdsBytes =
    kHighBaseBytes + kProducerWaves * kOwnersPerWave * kPairBytes;
constexpr int kValuesPerWavePerIteration =
    kOwnersPerWave * 2 * kWaveSize * 8;
constexpr int kValuesPerWave =
    kIterations * kValuesPerWavePerIteration;
static_assert(PDS_PROBE_ACCUM_F32X4 > 0,
              "pressure probe must keep at least one accumulator live");
static_assert(PDS_PROBE_ACCUM_F32X4 <= 32,
              "pressure probe supports at most 128 live FP32 VGPRs");
static_assert(!PDS_PROBE_SPLIT_OUTPUT_READERS ||
                  PDS_PROBE_ACCUM_F32X4 == 16,
              "split-output probe models one 64-FP32-accumulator output");
static_assert(PDS_PROBE_MODE >= 0 && PDS_PROBE_MODE <= 3,
              "probe mode must be readback(0), downstream MMAC(1), or "
              "role-source identity(2), or cross-role MMAC(3)");
static_assert(PDS_PROBE_PACK_MODE >= 0 && PDS_PROBE_PACK_MODE <= 3,
              "pack mode must be m-major(0), component-interleave(1), "
              "even-odd(2), or component-even-odd(3)");
static_assert(PDS_PROBE_WRITER_ROW * PDS_PROBE_WRITER_COL == 2,
              "writer row/col shape must cover 512 f16 values");
static_assert(PDS_PROBE_READER_ROW * PDS_PROBE_READER_COL == 2,
              "reader row/col shape must cover 512 f16 values");

union F16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

__device__ __forceinline__ void write_matrix_candidate(
    const F16x8& frag,
    __half* lds,
    int lds_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(lds) + lds_offset);
    __builtin_hcu_ds_write_matrix_format_f16(
        frag.f16x8, ptr, 16, PDS_PROBE_WRITER_ROW,
        PDS_PROBE_WRITER_COL, PDS_PROBE_WRITER_TRANS,
        PDS_PROBE_WRITER_ALT);
#else
    (void)frag;
    (void)lds;
    (void)lds_offset;
#endif
}

__device__ __forceinline__ F16x8 pack_writer_fragment(
    const F16x8& natural) {
    F16x8 packed;
#if PDS_PROBE_PACK_MODE == 0
    packed = natural;
#elif PDS_PROBE_PACK_MODE == 1
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed.scalar[2 * i] = natural.scalar[i];
        packed.scalar[2 * i + 1] = natural.scalar[4 + i];
    }
#elif PDS_PROBE_PACK_MODE == 2
    packed.scalar[0] = natural.scalar[0];
    packed.scalar[1] = natural.scalar[2];
    packed.scalar[2] = natural.scalar[4];
    packed.scalar[3] = natural.scalar[6];
    packed.scalar[4] = natural.scalar[1];
    packed.scalar[5] = natural.scalar[3];
    packed.scalar[6] = natural.scalar[5];
    packed.scalar[7] = natural.scalar[7];
#else
    packed.scalar[0] = natural.scalar[0];
    packed.scalar[1] = natural.scalar[4];
    packed.scalar[2] = natural.scalar[2];
    packed.scalar[3] = natural.scalar[6];
    packed.scalar[4] = natural.scalar[1];
    packed.scalar[5] = natural.scalar[5];
    packed.scalar[6] = natural.scalar[3];
    packed.scalar[7] = natural.scalar[7];
#endif
    return packed;
}

__device__ __forceinline__ void read_matrix_candidate(
    const __half* lds,
    int lds_offset,
    F16x8& frag) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(const_cast<__half*>(lds)) + lds_offset);
#if PDS_PROBE_READER_TRANS
    frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        ptr, 16, PDS_PROBE_READER_ROW, PDS_PROBE_READER_COL,
        PDS_PROBE_READER_ALT);
#else
    frag.f16x8 = __builtin_hcu_ds_read_matrix_format_f16(
        ptr, 16, PDS_PROBE_READER_ROW, PDS_PROBE_READER_COL,
        PDS_PROBE_READER_ALT);
#endif
#else
    (void)lds;
    (void)lds_offset;
    frag = {};
#endif
}

__device__ __forceinline__ uint16_t half_bits(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } bits{value};
    return bits.u;
}

__device__ __forceinline__ _Float16 half_from_bits(uint16_t value) {
    union {
        uint16_t u;
        _Float16 h;
    } bits{value};
    return bits.h;
}

__device__ __forceinline__ F16x8 make_fragment(int wave_local,
                                                int lane,
                                                int output,
                                                int iteration) {
#if PDS_PROBE_MMAC_SOURCE
    F16x8 value;
#pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
        F16x8 lhs;
        F16x8 rhs;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int lhs_code =
                (wave_local * 5 + lane * 3 + iteration + m_half + i) % 11 - 5;
            const int rhs_code =
                (wave_local * 7 + lane + iteration * 3 + output + i) % 13 - 6;
            lhs.scalar[i] = static_cast<_Float16>(lhs_code * 0.03125f);
            rhs.scalar[i] = static_cast<_Float16>(rhs_code * 0.025f);
        }
        ins::F16x8 zero;
        ins::zero_f16x8(zero);
        ins::F32x4 accum;
        accum.f32 = ins::mmac_f16_lit(
            lhs.f16x4[0], rhs.f16x4[0], zero.f32);
        accum.f32 = ins::mmac_f16_lit(
            lhs.f16x4[1], rhs.f16x4[1], accum.f32);
        ins::Vec4F16 packed;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = accum.scalar[i];
            const float transformed =
                output == 0 ? exp2f(x * 0.25f) : x * exp2f(-x * 0.125f);
            packed[i] = static_cast<_Float16>(transformed);
        }
        value.f16x4[m_half] = packed;
    }
    return value;
#else
    F16x8 value;
    const uint16_t base = static_cast<uint16_t>(
        (output == 0 ? 0x2800 : 0x3000) + iteration * 0x40 +
        wave_local * 512 + lane * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value.scalar[i] = half_from_bits(static_cast<uint16_t>(base + i));
    }
    return value;
#endif
}

__device__ __forceinline__ F16x8 make_rhs_fragment(int owner_local,
                                                    int lane,
                                                    int output,
                                                    int iteration) {
    F16x8 rhs;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int code =
            (owner_local * 11 + lane * 5 + output * 3 + iteration + i) % 17 -
            8;
        rhs.scalar[i] = static_cast<_Float16>(code * 0.015625f);
    }
    return rhs;
}

__device__ __forceinline__ ins::F32x4 run_fragment_mmac(
    const F16x8& lhs,
    const F16x8& rhs) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    ins::F32x4 out;
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], zero.f32);
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[1], rhs.f16x4[1], out.f32);
    return out;
}

__device__ __forceinline__ void store_fragment(uint16_t* dst,
                                                int iteration,
                                                int wave_local,
                                                int owner,
                                                int output,
                                                int lane,
                                                const F16x8& value) {
    const int base = iteration * kProducerWaves *
                         kValuesPerWavePerIteration +
                     (((wave_local * kOwnersPerWave + owner) * 2 + output) *
                          kWaveSize +
                      lane) *
                         8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[base + i] = half_bits(value.scalar[i]);
    }
}

__device__ __forceinline__ void store_mmac_output(uint16_t* dst,
                                                   int iteration,
                                                   int wave_local,
                                                   int owner,
                                                   int output,
                                                   int lane,
                                                   const ins::F32x4& value) {
    const int half_base = iteration * kProducerWaves *
                              kValuesPerWavePerIteration +
                          (((wave_local * kOwnersPerWave + owner) * 2 +
                            output) *
                               kWaveSize +
                           lane) *
                              8;
    auto* dst32 = reinterpret_cast<uint32_t*>(dst);
    const int word_base = half_base / 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        union {
            float f;
            uint32_t u;
        } bits{value.scalar[i]};
        dst32[word_base + i] = bits.u;
    }
}

__device__ __forceinline__ void run_writer(
    uint16_t* expected,
    __half* lds,
    int base_bytes,
    int use_abarrier,
    int producer_readback,
    uint32_t wave) {
    const int wave_local = static_cast<int>(wave & 3u);
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    int used_phase = 0;
#pragma clang loop unroll(disable)
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        if (use_abarrier) {
            if (iteration > 0) {
                ins::abarrier_try_wait<true>(kUsedBarrierId, used_phase);
            }
            ins::abarrier_seq<false>(kFilledBarrierId);
        }
#pragma unroll
        for (int owner = 0; owner < kOwnersPerWave; ++owner) {
            const int owner_local = wave_local * kOwnersPerWave + owner;
            const int p_offset = base_bytes + owner_local * kPairBytes;
            const int ds_offset = p_offset + kPageBytes;
            const F16x8 p = make_fragment(owner_local, lane, 0, iteration);
            const F16x8 ds = make_fragment(owner_local, lane, 1, iteration);
            write_matrix_candidate(pack_writer_fragment(p), lds, p_offset);
            write_matrix_candidate(pack_writer_fragment(ds), lds, ds_offset);
#if PDS_PROBE_MODE == 2
            store_fragment(expected, iteration, wave_local, owner, 0, lane, p);
            store_fragment(expected, iteration, wave_local, owner, 1, lane,
                           ds);
#endif
        }
        ins::wait_lgkm(0);

        if (producer_readback &&
            (PDS_PROBE_MODE == 0 || PDS_PROBE_MODE == 3)) {
#pragma unroll
            for (int owner = 0; owner < kOwnersPerWave; ++owner) {
                const int owner_local = wave_local * kOwnersPerWave + owner;
                const int p_offset = base_bytes + owner_local * kPairBytes;
                const int ds_offset = p_offset + kPageBytes;
                F16x8 p_reference;
                F16x8 ds_reference;
                read_matrix_candidate(lds, p_offset, p_reference);
                read_matrix_candidate(lds, ds_offset, ds_reference);
                ins::wait_lgkm(0);
#if PDS_PROBE_MODE == 3
                const F16x8 p_rhs =
                    make_rhs_fragment(owner_local, lane, 0, iteration);
                const F16x8 ds_rhs =
                    make_rhs_fragment(owner_local, lane, 1, iteration);
                store_mmac_output(
                    expected, iteration, wave_local, owner, 0, lane,
                    run_fragment_mmac(p_reference, p_rhs));
                store_mmac_output(
                    expected, iteration, wave_local, owner, 1, lane,
                    run_fragment_mmac(ds_reference, ds_rhs));
#else
                store_fragment(expected, iteration, wave_local, owner, 0,
                               lane, p_reference);
                store_fragment(expected, iteration, wave_local, owner, 1,
                               lane, ds_reference);
#endif
            }
        }
        if (use_abarrier) {
            ins::abarrier_arrive_cnt<false>(kFilledBarrierId, 1);
        }
    }
}

template <int ReaderGroup>
__device__ __forceinline__ void run_reader(
    uint16_t* actual,
    uint16_t* expected,
    float* pressure_sink,
    __half* lds,
    int base_bytes,
    int use_abarrier,
    uint32_t wave) {
    static_assert(ReaderGroup >= -1 && ReaderGroup < 2,
                  "reader group must be combined, dV, or dK");
    constexpr int kFirstOutput = ReaderGroup < 0 ? 0 : ReaderGroup;
    constexpr int kOutputCount = ReaderGroup < 0 ? 2 : 1;
    const int wave_local = static_cast<int>(wave & 3u);
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    int filled_phase = 0;
#if PDS_PROBE_HIGH_PRESSURE
    ins::F32x4 pressure[PDS_PROBE_ACCUM_F32X4];
#pragma unroll
    for (int i = 0; i < PDS_PROBE_ACCUM_F32X4; ++i) {
        const float value = static_cast<float>(
            wave * 1024 + lane * PDS_PROBE_ACCUM_F32X4 + i);
        pressure[i].f32 = ins::Vec4F32{value, value + 1.0f,
                                       value + 2.0f, value + 3.0f};
        asm volatile("" : "+v"(pressure[i].f32) : : "memory");
    }
#endif
#pragma clang loop unroll(disable)
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        if (use_abarrier) {
            ins::abarrier_try_wait<true>(kFilledBarrierId, filled_phase);
        }
#pragma unroll
        for (int owner = 0; owner < kOwnersPerWave; ++owner) {
            const int owner_local = wave_local * kOwnersPerWave + owner;
            const int p_offset = base_bytes + owner_local * kPairBytes;
            const int ds_offset = p_offset + kPageBytes;
#if PDS_PROBE_MODE == 2
#pragma unroll
            for (int output = kFirstOutput;
                 output < kFirstOutput + kOutputCount;
                 ++output) {
                const F16x8 direct =
                    make_fragment(owner_local, lane, output, iteration);
                store_fragment(actual, iteration, wave_local, owner, output,
                               lane, direct);
            }
#else
            F16x8 read_values[2];
#pragma unroll
            for (int output = kFirstOutput;
                 output < kFirstOutput + kOutputCount;
                 ++output) {
#if PDS_PROBE_MODE == 1
                const F16x8 direct =
                    make_fragment(owner_local, lane, output, iteration);
                const F16x8 direct_rhs =
                    make_rhs_fragment(owner_local, lane, output, iteration);
                store_mmac_output(
                    expected, iteration, wave_local, owner, output, lane,
                    run_fragment_mmac(direct, direct_rhs));
#endif
                const int offset = output == 0 ? p_offset : ds_offset;
                read_matrix_candidate(lds, offset, read_values[output]);
            }
            ins::wait_lgkm(0);
#if PDS_PROBE_HIGH_PRESSURE
#pragma unroll
            for (int i = 0; i < PDS_PROBE_ACCUM_F32X4; ++i) {
                asm volatile("" : "+v"(pressure[i].f32) : : "memory");
            }
#endif
#pragma unroll
            for (int output = kFirstOutput;
                 output < kFirstOutput + kOutputCount;
                 ++output) {
#if PDS_PROBE_MODE == 1 || PDS_PROBE_MODE == 3
                const F16x8 rhs =
                    make_rhs_fragment(owner_local, lane, output, iteration);
                store_mmac_output(
                    actual, iteration, wave_local, owner, output, lane,
                    run_fragment_mmac(read_values[output], rhs));
#else
                store_fragment(actual, iteration, wave_local, owner, output,
                               lane, read_values[output]);
#endif
            }
#endif
        }
        if (use_abarrier) {
            ins::abarrier_arrive_cnt<false>(kUsedBarrierId, 1);
        }
    }
#if PDS_PROBE_HIGH_PRESSURE
    float checksum = 0.0f;
#pragma unroll
    for (int i = 0; i < PDS_PROBE_ACCUM_F32X4; ++i) {
        checksum += pressure[i].scalar[0];
    }
    pressure_sink[threadIdx.x] = checksum;
#else
    (void)pressure_sink;
#endif
}

__global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dkv_pds_cross_wave_probe_kernel(uint16_t* actual,
                                uint16_t* expected,
                                float* pressure_sink,
                                int base_bytes,
                                int use_abarrier,
                                int producer_readback) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave = __builtin_hcu_get_wave_id();
    if (use_abarrier && wave == 0) {
        __builtin_hcu_s_abarrier_init(kFilledBarrierId, kProducerWaves);
        __builtin_hcu_s_abarrier_init(kUsedBarrierId, kConsumerWaves);
    }
    if (use_abarrier) {
        __builtin_hcu_s_ebarrier_sync(0);
    } else {
        __syncthreads();
    }

#if PDS_PROBE_SPLIT_OUTPUT_READERS
    if (wave < kProducerWaveBase) {
        __builtin_hcu_s_set_vgpr_size(16);
    } else if (wave < kConsumerWaveBase) {
        __builtin_hcu_s_set_vgpr_size(176);
        run_writer(expected, lds, base_bytes, use_abarrier,
                   producer_readback, wave);
    } else if (wave < kConsumerWaveBase + kWavesPerReaderGroup) {
        __builtin_hcu_s_set_vgpr_size(PDS_PROBE_READER_VGPRS);
        run_reader<0>(actual, expected, pressure_sink, lds, base_bytes,
                      use_abarrier, wave);
    } else {
        __builtin_hcu_s_set_vgpr_size(PDS_PROBE_READER_VGPRS);
        run_reader<1>(actual, expected, pressure_sink, lds, base_bytes,
                      use_abarrier, wave);
    }
#else
    if (wave < kProducerWaveBase) {
        __builtin_hcu_s_set_vgpr_size(16);
    } else if (wave >= kConsumerWaveBase + kConsumerWaves) {
        __builtin_hcu_s_set_vgpr_size(8);
    }
    if (wave >= kProducerWaveBase &&
        wave < kProducerWaveBase + kProducerWaves) {
        __builtin_hcu_s_set_vgpr_size(176);
        run_writer(expected, lds, base_bytes, use_abarrier, producer_readback,
                   wave);
    }
    if (!use_abarrier) {
        __syncthreads();
    }
    if (wave >= kConsumerWaveBase &&
        wave < kConsumerWaveBase + kConsumerWaves) {
        __builtin_hcu_s_set_vgpr_size(PDS_PROBE_READER_VGPRS);
        run_reader<-1>(actual, expected, pressure_sink, lds, base_bytes,
                       use_abarrier, wave);
    }
#endif

    __syncthreads();
    if (use_abarrier && wave == 0) {
        __builtin_hcu_s_abarrier_inv(kFilledBarrierId);
        __builtin_hcu_s_abarrier_inv(kUsedBarrierId);
    }
#else
    (void)actual;
    (void)expected;
    (void)pressure_sink;
    (void)base_bytes;
    (void)use_abarrier;
    (void)producer_readback;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
        std::exit(2);
    }
}

}  // namespace

int main() {
    std::printf(
        "dkv_pds_cross_wave config split_readers=%d reader_groups=%d "
        "accum_fp32=%d reader_vgprs=%d mode=%d mmac_source=%d pack=%d "
        "writer=%dx%d_t%d_a%d reader=%s_%dx%d_a%d\n",
        PDS_PROBE_SPLIT_OUTPUT_READERS, kReaderGroups,
        PDS_PROBE_ACCUM_F32X4 * 4, PDS_PROBE_READER_VGPRS,
        PDS_PROBE_MODE, PDS_PROBE_MMAC_SOURCE, PDS_PROBE_PACK_MODE,
        PDS_PROBE_WRITER_ROW * 16, PDS_PROBE_WRITER_COL * 16,
        PDS_PROBE_WRITER_TRANS, PDS_PROBE_WRITER_ALT,
        PDS_PROBE_READER_TRANS ? "trans" : "normal",
        PDS_PROBE_READER_ROW * 16, PDS_PROBE_READER_COL * 16,
        PDS_PROBE_READER_ALT);
    const size_t values = kProducerWaves * kValuesPerWave;
    const size_t bytes = values * sizeof(uint16_t);
    uint16_t* actual_device = nullptr;
    uint16_t* expected_device = nullptr;
    float* pressure_sink_device = nullptr;
    check_hip(hipMalloc(&actual_device, bytes), "hipMalloc actual");
    check_hip(hipMalloc(&expected_device, bytes), "hipMalloc expected");
    check_hip(hipMalloc(&pressure_sink_device,
                        kWaves * kWaveSize * sizeof(float)),
              "hipMalloc pressure sink");

#if PDS_PROBE_SPLIT_OUTPUT_READERS
    const int sync_modes[] = {1};
#else
    const int sync_modes[] = {0, 1};
#endif
    size_t total_bad = 0;
    for (int use_abarrier : sync_modes) {
        for (int base_bytes : {0, kHighBaseBytes}) {
            check_hip(hipMemset(actual_device, 0xff, bytes), "memset actual");
            check_hip(hipMemset(expected_device, 0xff, bytes),
                      "memset expected");
            dkv_pds_cross_wave_probe_kernel<<<1, kWaves * kWaveSize>>>(
                actual_device, expected_device, pressure_sink_device,
                base_bytes, use_abarrier, 1);
            check_hip(hipDeviceSynchronize(), "reference sync");
            check_hip(hipMemset(actual_device, 0xff, bytes),
                      "memset actual cross-only");
            dkv_pds_cross_wave_probe_kernel<<<1, kWaves * kWaveSize>>>(
                actual_device, expected_device, pressure_sink_device,
                base_bytes, use_abarrier, 0);
            check_hip(hipDeviceSynchronize(), "cross-only sync");

            std::vector<uint16_t> actual(values);
            std::vector<uint16_t> expected(values);
            check_hip(hipMemcpy(actual.data(), actual_device, bytes,
                                hipMemcpyDeviceToHost),
                      "copy actual");
            check_hip(hipMemcpy(expected.data(), expected_device, bytes,
                                hipMemcpyDeviceToHost),
                      "copy expected");

            size_t bad = 0;
            size_t bad_by_iteration[kIterations] = {};
            constexpr size_t kValuesPerIteration =
                kProducerWaves * kValuesPerWavePerIteration;
            for (size_t i = 0; i < values; ++i) {
                if (actual[i] != expected[i]) {
                    ++bad;
                    ++bad_by_iteration[i / kValuesPerIteration];
                }
            }
            total_bad += bad;
            std::printf(
                "dkv_pds_cross_wave sync=%s base_bytes=%d mismatches=%zu "
                "expected=%04x,%04x actual=%04x,%04x pass=%d\n",
                use_abarrier ? "abarrier" : "cta", base_bytes, bad,
                expected[0], expected[1], actual[0], actual[1],
                bad == 0 ? 1 : 0);
            std::printf("dkv_pds_cross_wave mismatches_by_iteration=");
            for (int iteration = 0; iteration < kIterations; ++iteration) {
                std::printf("%s%zu", iteration == 0 ? "" : ",",
                            bad_by_iteration[iteration]);
            }
            std::printf("\n");
        }
    }

    check_hip(hipFree(expected_device), "hipFree expected");
    check_hip(hipFree(actual_device), "hipFree actual");
    check_hip(hipFree(pressure_sink_device), "hipFree pressure sink");
    return total_bad == 0 ? 0 : 3;
}
