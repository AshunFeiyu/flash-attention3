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
constexpr int kIterations = 8;
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

union F16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

__device__ __forceinline__ void read_matrix_16x32_trans(
    const __half* lds,
    int lds_offset,
    F16x8& frag) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(const_cast<__half*>(lds)) + lds_offset);
    frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        ptr, 16, 1, 2, 0);
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
    F16x8 value;
    const uint16_t base = static_cast<uint16_t>(
        (output == 0 ? 0x2800 : 0x3000) + iteration * 0x40 +
        wave_local * 512 + lane * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value.scalar[i] = half_from_bits(static_cast<uint16_t>(base + i));
    }
    return value;
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
            ins::ds_write_matrix_32x16_f16(p.f16x8, lds, p_offset);
            ins::ds_write_matrix_32x16_f16(ds.f16x8, lds, ds_offset);
        }
        ins::wait_lgkm(0);

        if (producer_readback) {
#pragma unroll
            for (int owner = 0; owner < kOwnersPerWave; ++owner) {
                const int owner_local = wave_local * kOwnersPerWave + owner;
                const int p_offset = base_bytes + owner_local * kPairBytes;
                const int ds_offset = p_offset + kPageBytes;
                F16x8 p_reference;
                F16x8 ds_reference;
                read_matrix_16x32_trans(lds, p_offset, p_reference);
                read_matrix_16x32_trans(lds, ds_offset, ds_reference);
                ins::wait_lgkm(0);
                store_fragment(expected, iteration, wave_local, owner, 0,
                               lane, p_reference);
                store_fragment(expected, iteration, wave_local, owner, 1,
                               lane, ds_reference);
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
            F16x8 read_values[2];
#pragma unroll
            for (int output = kFirstOutput;
                 output < kFirstOutput + kOutputCount;
                 ++output) {
                const int offset = output == 0 ? p_offset : ds_offset;
                read_matrix_16x32_trans(lds, offset, read_values[output]);
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
                store_fragment(actual, iteration, wave_local, owner, output,
                               lane, read_values[output]);
            }
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
        run_writer(expected, lds, base_bytes, use_abarrier, producer_readback,
                   wave);
    } else if (wave < kConsumerWaveBase + kWavesPerReaderGroup) {
        __builtin_hcu_s_set_vgpr_size(PDS_PROBE_READER_VGPRS);
        run_reader<0>(actual, pressure_sink, lds, base_bytes, use_abarrier,
                      wave);
    } else {
        __builtin_hcu_s_set_vgpr_size(PDS_PROBE_READER_VGPRS);
        run_reader<1>(actual, pressure_sink, lds, base_bytes, use_abarrier,
                      wave);
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
        run_reader<-1>(actual, pressure_sink, lds, base_bytes, use_abarrier,
                       wave);
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
        "accum_fp32=%d reader_vgprs=%d\n",
        PDS_PROBE_SPLIT_OUTPUT_READERS, kReaderGroups,
        PDS_PROBE_ACCUM_F32X4 * 4, PDS_PROBE_READER_VGPRS);
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
            for (size_t i = 0; i < values; ++i) {
                bad += actual[i] != expected[i] ? 1u : 0u;
            }
            total_bad += bad;
            std::printf(
                "dkv_pds_cross_wave sync=%s base_bytes=%d mismatches=%zu "
                "expected=%04x,%04x actual=%04x,%04x pass=%d\n",
                use_abarrier ? "abarrier" : "cta", base_bytes, bad,
                expected[0], expected[1], actual[0], actual[1],
                bad == 0 ? 1 : 0);
        }
    }

    check_hip(hipFree(expected_device), "hipFree expected");
    check_hip(hipFree(actual_device), "hipFree actual");
    check_hip(hipFree(pressure_sink_device), "hipFree pressure sink");
    return total_bad == 0 ? 0 : 3;
}
