#include <hip/hip_fp16.h>
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
constexpr int kReaderCount = 5;
constexpr int kPairCount = kWriterCount * kReaderCount;
constexpr uint16_t kPatternBase = 0x3000;

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

__device__ __forceinline__ Vec8F16 read_fragment(int reader, _Float16* lds) {
    switch (reader) {
        case 0:
            return __builtin_hcu_ds_read_matrix_format_f16(
                lds, 0, 2, 1, 0);
        case 1:
            return __builtin_hcu_ds_read_matrix_format_f16(
                lds, 0, 2, 1, 1);
        case 2:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 0, 2, 1, 0);
        case 3:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 0, 1, 2, 0);
        default:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 0, 1, 2, 1);
    }
}

// One block is one writer/reader pair.  The same wave creates A in registers,
// writes A to LDS, reads A1 back, then exports both raw FP16 bit patterns.
// There is no MMAC, cross-wave handoff, ABarrier, or layout workaround here.
__global__ void __launch_bounds__(kWaveSize, 1)
    ds_matrix_reg_roundtrip_probe_kernel(
        uint16_t* __restrict__ a_bits,
        uint16_t* __restrict__ a1_bits) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 lds[kLdsPageElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int pair = static_cast<int>(blockIdx.x);
    const int writer = pair / kReaderCount;
    const int reader = pair % kReaderCount;

    Vec8F16 a{};
#pragma unroll
    for (int word = 0; word < kWordsPerLane; ++word) {
        const uint16_t bits = static_cast<uint16_t>(
            kPatternBase + lane * kWordsPerLane + word);
        a[word] = half_from_bits(bits);
    }

    write_fragment(writer, a, lds);
    ins::wait_lgkm(0);
    const Vec8F16 a1 = read_fragment(reader, lds);
    ins::wait_lgkm(0);

    const int base = pair * kWordsPerFragment + lane * kWordsPerLane;
#pragma unroll
    for (int word = 0; word < kWordsPerLane; ++word) {
        a_bits[base + word] = half_bits(a[word]);
        a1_bits[base + word] = half_bits(a1[word]);
    }
#else
    (void)a_bits;
    (void)a1_bits;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

const char* kWriterNames[kWriterCount] = {
    "normal_alt0", "normal_alt1", "trans_alt0", "trans_alt1"};
const char* kReaderNames[kReaderCount] = {
    "normal_m32_alt0",
    "normal_m32_alt1",
    "trans_m32_alt0",
    "trans_m16_alt0",
    "trans_m16_alt1",
};

}  // namespace

int main() {
    const size_t total_words =
        static_cast<size_t>(kPairCount) * kWordsPerFragment;
    uint16_t* d_a = nullptr;
    uint16_t* d_a1 = nullptr;
    std::vector<uint16_t> h_a(total_words);
    std::vector<uint16_t> h_a1(total_words);

    check_hip(hipMalloc(&d_a, total_words * sizeof(uint16_t)), "hipMalloc A");
    check_hip(hipMalloc(&d_a1, total_words * sizeof(uint16_t)), "hipMalloc A1");
    check_hip(hipMemset(d_a, 0, total_words * sizeof(uint16_t)), "hipMemset A");
    check_hip(hipMemset(d_a1, 0, total_words * sizeof(uint16_t)),
              "hipMemset A1");

    hipLaunchKernelGGL(ds_matrix_reg_roundtrip_probe_kernel,
                       dim3(kPairCount), dim3(kWaveSize), 0, 0, d_a, d_a1);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_a.data(), d_a, total_words * sizeof(uint16_t),
                       hipMemcpyDeviceToHost),
              "hipMemcpy A");
    check_hip(hipMemcpy(h_a1.data(), d_a1, total_words * sizeof(uint16_t),
                       hipMemcpyDeviceToHost),
              "hipMemcpy A1");
    check_hip(hipFree(d_a), "hipFree A");
    check_hip(hipFree(d_a1), "hipFree A1");

    int identity_pair_count = 0;
    int permutation_pair_count = 0;
    for (int writer = 0; writer < kWriterCount; ++writer) {
        for (int reader = 0; reader < kReaderCount; ++reader) {
            const int pair = writer * kReaderCount + reader;
            const size_t base = static_cast<size_t>(pair) * kWordsPerFragment;
            int identity_mismatch = 0;
            int unmapped = 0;
            int duplicate = 0;
            int shown = 0;
            int unmapped_shown = 0;
            std::vector<int> source_counts(kWordsPerFragment, 0);

            for (int dst = 0; dst < kWordsPerFragment; ++dst) {
                const uint16_t expected = h_a[base + dst];
                const uint16_t actual = h_a1[base + dst];
                if (actual != expected) {
                    ++identity_mismatch;
                }

                int source = -1;
                if (actual >= kPatternBase &&
                    actual < kPatternBase + kWordsPerFragment) {
                    source = static_cast<int>(actual - kPatternBase);
                    if (++source_counts[source] > 1) {
                        ++duplicate;
                    }
                } else {
                    ++unmapped;
                    if (unmapped_shown < 4) {
                        std::printf(
                            "roundtrip_unmapped writer=%s reader=%s "
                            "dst_lane=%d dst_word=%d actual=0x%04x\n",
                            kWriterNames[writer], kReaderNames[reader],
                            dst / kWordsPerLane, dst % kWordsPerLane,
                            static_cast<unsigned>(actual));
                        ++unmapped_shown;
                    }
                }

                if (actual != expected && shown < 4) {
                    const int dst_lane = dst / kWordsPerLane;
                    const int dst_word = dst % kWordsPerLane;
                    const int source_lane =
                        source >= 0 ? source / kWordsPerLane : -1;
                    const int source_word =
                        source >= 0 ? source % kWordsPerLane : -1;
                    std::printf(
                        "roundtrip_mismatch writer=%s reader=%s "
                        "dst_lane=%d dst_word=%d expected=0x%04x actual=0x%04x "
                        "source_lane=%d source_word=%d\n",
                        kWriterNames[writer], kReaderNames[reader], dst_lane,
                        dst_word, static_cast<unsigned>(expected),
                        static_cast<unsigned>(actual), source_lane,
                        source_word);
                    ++shown;
                }
            }

            int missing = 0;
            for (int count : source_counts) {
                if (count == 0) {
                    ++missing;
                }
            }
            const bool identity_pass = identity_mismatch == 0;
            const bool permutation_pass =
                unmapped == 0 && duplicate == 0 && missing == 0;
            identity_pair_count += identity_pass ? 1 : 0;
            permutation_pair_count += permutation_pass ? 1 : 0;
            std::printf(
                "roundtrip_summary writer=%s reader=%s "
                "identity_mismatch=%d identity_pass=%d unmapped=%d "
                "duplicate=%d missing=%d permutation_pass=%d\n",
                kWriterNames[writer], kReaderNames[reader], identity_mismatch,
                identity_pass ? 1 : 0, unmapped, duplicate, missing,
                permutation_pass ? 1 : 0);
        }
    }

    std::printf(
        "roundtrip_probe_complete=1 identity_pairs=%d permutation_pairs=%d "
        "total_pairs=%d\n",
        identity_pair_count, permutation_pair_count, kPairCount);
    return 0;
}
