#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;

constexpr int kWaveSize = 64;
constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kMatrixElems = kRows * kCols;
constexpr int kPageElems = 64 * 16;
constexpr int kLoadModeCount = 4;
constexpr int kReaderCount = 3;
constexpr int kWriterCount = 4;
constexpr int kStoreModeCount = 4;
constexpr int kLocalChainCount =
    kReaderCount * kWriterCount * kStoreModeCount;
constexpr int kChainCount = kLoadModeCount * kLocalChainCount;
constexpr int kDirectCount = kLoadModeCount * kStoreModeCount;
constexpr int kCandidateCount = kChainCount + kDirectCount;
constexpr int kOutputStrideElems = 4 * kMatrixElems;
constexpr int kStoreBarrier = 0;
constexpr uint16_t kPatternBase = 0x3000;

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

__host__ _Float16 half_from_bits(uint16_t bits) {
    HalfBits value{};
    value.bits = bits;
    return value.value;
}

__device__ __forceinline__ void matrix_load_mode(
    int mode, ins::Vec4U32 src, _Float16* lds) {
    switch (mode) {
        case 0:
            __builtin_hcu_matrix_load_32x16_b16(
                src, reinterpret_cast<short*>(lds), 0, false, false, false,
                false, true);
            break;
        case 1:
            __builtin_hcu_matrix_load_32x16_b16(
                src, reinterpret_cast<short*>(lds), 0, true, false, false,
                false, true);
            break;
        case 2:
            __builtin_hcu_matrix_load_32x16_b16(
                src, reinterpret_cast<short*>(lds), 0, false, true, false,
                false, true);
            break;
        default:
            __builtin_hcu_matrix_load_32x16_b16(
                src, reinterpret_cast<short*>(lds), 0, true, true, false,
                false, true);
            break;
    }
}

__device__ __forceinline__ Vec8F16 read_fragment(int mode, _Float16* lds) {
    switch (mode) {
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

__device__ __forceinline__ void write_fragment(
    int mode, const Vec8F16& fragment, _Float16* lds) {
    switch (mode) {
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

__device__ __forceinline__ void matrix_store_mode(
    int mode, ins::Vec4U32 dst, _Float16* lds) {
    switch (mode) {
        case 0:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0, false, false, false,
                false);
            break;
        case 1:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0, true, false, false,
                false);
            break;
        case 2:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0, false, true, false,
                false);
            break;
        default:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0, true, true, false,
                false);
            break;
    }
}

__device__ __forceinline__ void store_transaction(
    int mode,
    _Float16* lds,
    _Float16* output,
    int& store_phase) {
    const ins::Vec4U32 dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output), kCols);
    ins::abarrier_seq<false>(kStoreBarrier);
    matrix_store_mode(mode, dst, lds);
    ins::abarrier_arrive_cnt<false>(kStoreBarrier, 1);
    ins::abarrier_try_wait<false>(kStoreBarrier, store_phase);
    ins::wait_vmem_lgkm();
}

// Four blocks independently sweep one MLS t/r mode. Each chain candidate has
// a private 2 KiB LDS destination page, so an incomplete ds_write cannot be
// hidden by data from a previous candidate.
__global__ void __launch_bounds__(kWaveSize, 1)
    matrix_global_roundtrip_probe_kernel(
        const _Float16* __restrict__ input,
        _Float16* __restrict__ output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256)
        _Float16 lds[(1 + kLocalChainCount) * kPageElems];
    const int load_mode = static_cast<int>(blockIdx.x);
    _Float16* const load_page = lds;

    __builtin_hcu_s_abarrier_init(kStoreBarrier, 1);
    __builtin_hcu_s_ebarrier_sync(0);

    const ins::Vec4U32 src = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(input), kCols);
    matrix_load_mode(load_mode, src, load_page);
    ins::wait_vbcnt0();
    ins::wait_lgkm(0);

    int store_phase = 0;
    for (int local = 0; local < kLocalChainCount; ++local) {
        const int reader = local / (kWriterCount * kStoreModeCount);
        const int writer_store = local % (kWriterCount * kStoreModeCount);
        const int writer = writer_store / kStoreModeCount;
        const int store = writer_store % kStoreModeCount;
        const int candidate = load_mode * kLocalChainCount + local;
        _Float16* const store_page =
            lds + (1 + local) * kPageElems;

        const Vec8F16 fragment = read_fragment(reader, load_page);
        ins::wait_lgkm(0);
        write_fragment(writer, fragment, store_page);
        ins::wait_lgkm(0);
        store_transaction(
            store, store_page,
            output + candidate * kOutputStrideElems, store_phase);
    }

    // Direct MLS-page controls isolate matrix_store from ds_read/ds_write.
    for (int store = 0; store < kStoreModeCount; ++store) {
        const int candidate =
            kChainCount + load_mode * kStoreModeCount + store;
        store_transaction(
            store, load_page,
            output + candidate * kOutputStrideElems, store_phase);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    __builtin_hcu_s_abarrier_inv(kStoreBarrier);
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

const char* kLoadNames[kLoadModeCount] = {
    "t0r0", "t1r0", "t0r1", "t1r1"};
const char* kReaderNames[kReaderCount] = {
    "normal_alt0", "normal_alt1", "trans_alt0"};
const char* kWriterNames[kWriterCount] = {
    "t0_alt0", "t0_alt1", "t1_alt0", "t1_alt1"};
const char* kStoreNames[kStoreModeCount] = {
    "t0r0", "t1r0", "t0r1", "t1r1"};

struct Comparison {
    int row_mismatch = 0;
    int trans_mismatch = 0;
    int unmapped = 0;
    int duplicate = 0;
    int missing = 0;
    int first_index = -1;
    uint16_t first_expected = 0;
    uint16_t first_actual = 0;
};

Comparison compare_candidate(
    const uint16_t* actual, const std::vector<uint16_t>& expected) {
    Comparison result{};
    std::vector<int> source_counts(kMatrixElems, 0);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const int index = row * kCols + col;
            const uint16_t value = actual[index];
            if (value != expected[index]) {
                ++result.row_mismatch;
                if (result.first_index < 0) {
                    result.first_index = index;
                    result.first_expected = expected[index];
                    result.first_actual = value;
                }
            }
            result.trans_mismatch +=
                actual[col * kRows + row] != expected[index];
            if (value >= kPatternBase &&
                value < kPatternBase + kMatrixElems) {
                const int source = value - kPatternBase;
                if (++source_counts[source] > 1) {
                    ++result.duplicate;
                }
            } else {
                ++result.unmapped;
            }
        }
    }
    for (int count : source_counts) {
        result.missing += count == 0;
    }
    return result;
}

}  // namespace

int main() {
    std::vector<uint16_t> input_bits(kMatrixElems);
    std::vector<_Float16> input(kMatrixElems);
    for (int i = 0; i < kMatrixElems; ++i) {
        input_bits[i] = static_cast<uint16_t>(kPatternBase + i);
        input[i] = half_from_bits(input_bits[i]);
    }

    const size_t output_elems =
        static_cast<size_t>(kCandidateCount) * kOutputStrideElems;
    _Float16* d_input = nullptr;
    _Float16* d_output = nullptr;
    std::vector<uint16_t> output(output_elems);
    check_hip(hipMalloc(&d_input, kMatrixElems * sizeof(_Float16)),
              "hipMalloc input");
    check_hip(hipMalloc(&d_output, output_elems * sizeof(_Float16)),
              "hipMalloc output");
    check_hip(hipMemcpy(d_input, input.data(),
                        kMatrixElems * sizeof(_Float16),
                        hipMemcpyHostToDevice),
              "hipMemcpy input");
    check_hip(hipMemset(d_output, 0xfe, output_elems * sizeof(_Float16)),
              "hipMemset output");

    hipLaunchKernelGGL(matrix_global_roundtrip_probe_kernel,
                       dim3(kLoadModeCount), dim3(kWaveSize), 0, 0,
                       d_input, d_output);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(output.data(), d_output,
                        output_elems * sizeof(_Float16),
                        hipMemcpyDeviceToHost),
              "hipMemcpy output");
    check_hip(hipFree(d_output), "hipFree output");
    check_hip(hipFree(d_input), "hipFree input");

    int exact_chain_pairs = 0;
    int permutation_chain_pairs = 0;
    int transpose_chain_pairs = 0;
    int best_chain_mismatch = kMatrixElems + 1;
    int best_chain = -1;
    for (int candidate = 0; candidate < kChainCount; ++candidate) {
        const int load = candidate / kLocalChainCount;
        const int local = candidate % kLocalChainCount;
        const int reader = local / (kWriterCount * kStoreModeCount);
        const int writer_store = local % (kWriterCount * kStoreModeCount);
        const int writer = writer_store / kStoreModeCount;
        const int store = writer_store % kStoreModeCount;
        const uint16_t* actual =
            output.data() +
            static_cast<size_t>(candidate) * kOutputStrideElems;
        const Comparison result = compare_candidate(actual, input_bits);
        const bool exact = result.row_mismatch == 0;
        const bool trans_exact = result.trans_mismatch == 0;
        const bool permutation = result.unmapped == 0 &&
                                 result.duplicate == 0 && result.missing == 0;
        exact_chain_pairs += exact ? 1 : 0;
        transpose_chain_pairs += trans_exact ? 1 : 0;
        permutation_chain_pairs += permutation ? 1 : 0;
        if (result.row_mismatch < best_chain_mismatch) {
            best_chain_mismatch = result.row_mismatch;
            best_chain = candidate;
        }
        std::printf(
            "global_roundtrip_chain load=%s reader=%s writer=%s store=%s "
            "row_mismatch=%d trans_mismatch=%d unmapped=%d duplicate=%d "
            "missing=%d first_row=%d first_col=%d first_expected=0x%04x "
            "first_actual=0x%04x exact=%d trans_exact=%d permutation=%d\n",
            kLoadNames[load], kReaderNames[reader], kWriterNames[writer],
            kStoreNames[store], result.row_mismatch, result.trans_mismatch,
            result.unmapped, result.duplicate, result.missing,
            result.first_index < 0 ? -1 : result.first_index / kCols,
            result.first_index < 0 ? -1 : result.first_index % kCols,
            static_cast<unsigned>(result.first_expected),
            static_cast<unsigned>(result.first_actual),
            exact ? 1 : 0, trans_exact ? 1 : 0, permutation ? 1 : 0);
    }

    int exact_direct_pairs = 0;
    int permutation_direct_pairs = 0;
    int best_direct_mismatch = kMatrixElems + 1;
    int best_direct = -1;
    for (int local = 0; local < kDirectCount; ++local) {
        const int load = local / kStoreModeCount;
        const int store = local % kStoreModeCount;
        const int candidate = kChainCount + local;
        const uint16_t* actual =
            output.data() +
            static_cast<size_t>(candidate) * kOutputStrideElems;
        const Comparison result = compare_candidate(actual, input_bits);
        const bool exact = result.row_mismatch == 0;
        const bool permutation = result.unmapped == 0 &&
                                 result.duplicate == 0 && result.missing == 0;
        exact_direct_pairs += exact ? 1 : 0;
        permutation_direct_pairs += permutation ? 1 : 0;
        if (result.row_mismatch < best_direct_mismatch) {
            best_direct_mismatch = result.row_mismatch;
            best_direct = local;
        }
        std::printf(
            "global_roundtrip_direct load=%s store=%s row_mismatch=%d "
            "trans_mismatch=%d unmapped=%d duplicate=%d missing=%d exact=%d "
            "permutation=%d first_row=%d first_col=%d first_expected=0x%04x "
            "first_actual=0x%04x\n",
            kLoadNames[load], kStoreNames[store], result.row_mismatch,
            result.trans_mismatch, result.unmapped, result.duplicate,
            result.missing, exact ? 1 : 0, permutation ? 1 : 0,
            result.first_index < 0 ? -1 : result.first_index / kCols,
            result.first_index < 0 ? -1 : result.first_index % kCols,
            static_cast<unsigned>(result.first_expected),
            static_cast<unsigned>(result.first_actual));
    }

    std::printf(
        "global_roundtrip_complete=1 exact_chain_pairs=%d "
        "permutation_chain_pairs=%d transpose_chain_pairs=%d "
        "exact_direct_pairs=%d permutation_direct_pairs=%d "
        "best_chain=%d best_chain_mismatch=%d best_direct=%d "
        "best_direct_mismatch=%d total_chain=%d total_direct=%d\n",
        exact_chain_pairs, permutation_chain_pairs, transpose_chain_pairs,
        exact_direct_pairs, permutation_direct_pairs, best_chain,
        best_chain_mismatch, best_direct, best_direct_mismatch,
        kChainCount, kDirectCount);
    return 0;
}
