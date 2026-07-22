#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;

constexpr int kWaveSize = 64;
constexpr int kWordsPerLane = 8;
constexpr int kWordsPerFragment = kWaveSize * kWordsPerLane;
constexpr int kLdsPageElems = 64 * 16;
constexpr int kWriterCount = 4;
constexpr int kReaderCount = 3;
constexpr int kPairCount = kWriterCount * kReaderCount;
constexpr uint16_t kCalibrationBase = 0x3000;
constexpr uint16_t kReplayBase = 0x3400;

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

__device__ __forceinline__ _Float16 half_from_bits(uint16_t bits) {
    HalfBits value{};
    value.bits = bits;
    return value.value;
}

__device__ __forceinline__ uint16_t half_bits(_Float16 value) {
    HalfBits result{};
    result.value = value;
    return result.bits;
}

__device__ __forceinline__ void write_fragment(
    int writer, const Vec8F16& fragment, _Float16* lds) {
    switch (writer) {
        case 0:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 0, 2, 1, 0, 0);
            break;
        case 1:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 0, 2, 1, 1, 0);
            break;
        case 2:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 0, 2, 1, 0, 1);
            break;
        default:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 0, 2, 1, 1, 1);
            break;
    }
}

__device__ __forceinline__ Vec8F16 read_fragment(
    int reader, _Float16* lds) {
    switch (reader) {
        case 0:
            return __builtin_hcu_ds_read_matrix_format_f16(
                lds, 0, 2, 1, 0);
        case 1:
            return __builtin_hcu_ds_read_matrix_format_f16(
                lds, 0, 2, 1, 1);
        default:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 0, 2, 1, 0);
    }
}

// One block is one matching m32 writer/reader pair. The first launch measures
// the register-slot permutation; the second launch replays an inverse-packed
// source prepared by the CPU. No MMAC or layout workaround is present.
__global__ void __launch_bounds__(kWaveSize, 1)
    ds_matrix_reg_roundtrip_probe_kernel(
        const uint16_t* __restrict__ input,
        uint16_t* __restrict__ output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 lds[kLdsPageElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int pair = static_cast<int>(blockIdx.x);
    const int writer = pair / kReaderCount;
    const int reader = pair % kReaderCount;
    const int base = pair * kWordsPerFragment + lane * kWordsPerLane;

    Vec8F16 source{};
#pragma unroll
    for (int word = 0; word < kWordsPerLane; ++word) {
        source[word] = half_from_bits(input[base + word]);
    }
    write_fragment(writer, source, lds);
    ins::wait_lgkm(0);
    const Vec8F16 result = read_fragment(reader, lds);
    ins::wait_lgkm(0);
#pragma unroll
    for (int word = 0; word < kWordsPerLane; ++word) {
        output[base + word] = half_bits(result[word]);
    }
#else
    (void)input;
    (void)output;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     hipGetErrorString(error));
        std::exit(1);
    }
}

void run_probe(
    uint16_t* device_input,
    uint16_t* device_output,
    const std::vector<uint16_t>& input,
    std::vector<uint16_t>& output,
    const char* phase) {
    const size_t bytes = input.size() * sizeof(uint16_t);
    check_hip(hipMemcpy(device_input, input.data(), bytes,
                        hipMemcpyHostToDevice),
              "hipMemcpy input");
    check_hip(hipMemset(device_output, 0, bytes), "hipMemset output");
    hipLaunchKernelGGL(ds_matrix_reg_roundtrip_probe_kernel,
                       dim3(kPairCount), dim3(kWaveSize), 0, 0,
                       device_input, device_output);
    check_hip(hipGetLastError(), phase);
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(output.data(), device_output, bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy output");
}

const char* kWriterNames[kWriterCount] = {
    "normal_alt0", "normal_alt1", "trans_alt0", "trans_alt1"};
const char* kReaderNames[kReaderCount] = {
    "normal_m32_alt0", "normal_m32_alt1", "trans_m32_alt0"};

}  // namespace

int main() {
    const size_t total_words =
        static_cast<size_t>(kPairCount) * kWordsPerFragment;
    const size_t bytes = total_words * sizeof(uint16_t);
    uint16_t* device_input = nullptr;
    uint16_t* device_output = nullptr;
    std::vector<uint16_t> calibration_input(total_words);
    std::vector<uint16_t> calibration_output(total_words);
    std::vector<uint16_t> replay_input(total_words);
    std::vector<uint16_t> replay_output(total_words);
    std::vector<int> dst_to_source(total_words, -1);

    for (int pair = 0; pair < kPairCount; ++pair) {
        const size_t base = static_cast<size_t>(pair) * kWordsPerFragment;
        for (int source = 0; source < kWordsPerFragment; ++source) {
            calibration_input[base + source] =
                static_cast<uint16_t>(kCalibrationBase + source);
        }
    }

    check_hip(hipMalloc(&device_input, bytes), "hipMalloc input");
    check_hip(hipMalloc(&device_output, bytes), "hipMalloc output");
    run_probe(device_input, device_output, calibration_input,
              calibration_output, "launch calibration");

    std::FILE* slot_map = std::fopen("ds_matrix_slot_map.csv", "w");
    if (slot_map == nullptr) {
        std::fprintf(stderr, "failed to create ds_matrix_slot_map.csv\n");
        return 1;
    }
    std::fprintf(
        slot_map,
        "writer,reader,dst_lane,dst_word,source_lane,source_word\n");
    int identity_pairs = 0;
    int permutation_pairs = 0;
    for (int pair = 0; pair < kPairCount; ++pair) {
        const int writer = pair / kReaderCount;
        const int reader = pair % kReaderCount;
        const size_t base = static_cast<size_t>(pair) * kWordsPerFragment;
        std::vector<int> counts(kWordsPerFragment, 0);
        int identity_mismatch = 0;
        int unmapped = 0;
        int duplicate = 0;
        for (int dst = 0; dst < kWordsPerFragment; ++dst) {
            const uint16_t actual = calibration_output[base + dst];
            identity_mismatch +=
                actual != static_cast<uint16_t>(kCalibrationBase + dst);
            if (actual >= kCalibrationBase &&
                actual < kCalibrationBase + kWordsPerFragment) {
                const int source = actual - kCalibrationBase;
                dst_to_source[base + dst] = source;
                duplicate += ++counts[source] > 1;
                std::fprintf(
                    slot_map, "%s,%s,%d,%d,%d,%d\n",
                    kWriterNames[writer], kReaderNames[reader],
                    dst / kWordsPerLane, dst % kWordsPerLane,
                    source / kWordsPerLane, source % kWordsPerLane);
            } else {
                ++unmapped;
            }
        }
        int missing = 0;
        for (int count : counts) {
            missing += count == 0;
        }
        const bool permutation =
            unmapped == 0 && duplicate == 0 && missing == 0;
        identity_pairs += identity_mismatch == 0;
        permutation_pairs += permutation ? 1 : 0;
        if (permutation) {
            for (int dst = 0; dst < kWordsPerFragment; ++dst) {
                const int source = dst_to_source[base + dst];
                replay_input[base + source] =
                    static_cast<uint16_t>(kReplayBase + dst);
            }
        }
        std::printf(
            "calibration_summary writer=%s reader=%s identity_mismatch=%d "
            "unmapped=%d duplicate=%d missing=%d permutation_pass=%d\n",
            kWriterNames[writer], kReaderNames[reader], identity_mismatch,
            unmapped, duplicate, missing, permutation ? 1 : 0);
    }
    std::fclose(slot_map);

    run_probe(device_input, device_output, replay_input, replay_output,
              "launch inverse replay");
    check_hip(hipFree(device_input), "hipFree input");
    check_hip(hipFree(device_output), "hipFree output");

    int replay_identity_pairs = 0;
    for (int pair = 0; pair < kPairCount; ++pair) {
        const int writer = pair / kReaderCount;
        const int reader = pair % kReaderCount;
        const size_t base = static_cast<size_t>(pair) * kWordsPerFragment;
        int mismatch = 0;
        for (int dst = 0; dst < kWordsPerFragment; ++dst) {
            mismatch += replay_output[base + dst] !=
                        static_cast<uint16_t>(kReplayBase + dst);
        }
        replay_identity_pairs += mismatch == 0;
        std::printf(
            "inverse_replay_summary writer=%s reader=%s mismatch=%d "
            "identity_pass=%d\n",
            kWriterNames[writer], kReaderNames[reader], mismatch,
            mismatch == 0 ? 1 : 0);
    }

    std::printf(
        "roundtrip_probe_complete=1 identity_pairs=%d permutation_pairs=%d "
        "replay_identity_pairs=%d total_pairs=%d\n",
        identity_pairs, permutation_pairs, replay_identity_pairs, kPairCount);
    return replay_identity_pairs == kPairCount ? 0 : 1;
}
