#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

using Vec8F32 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kWaveSize = 64;
constexpr int kThreads = 16 * kWaveSize;
constexpr int kRows = 32;
constexpr int kCols = 16;
constexpr int kElems = kRows * kCols;
constexpr int kStoreModes = 1;
constexpr int kCandidates = 2 * kStoreModes;
constexpr int kCandidateStrideElems = 4 * kElems;
constexpr int kLdsBytes = 2 * kElems * sizeof(__half);
constexpr int kStoreBarrier = 0;
constexpr float kTolerance = 0.0f;

union F16x8 {
    ins::Vec8F16 vec;
    _Float16 scalar[8];
    uint16_t bits[8];
};

union F32x8 {
    Vec8F32 vec;
    float scalar[8];
};

__device__ __forceinline__ F16x8 load_control_fragment(
    const __half* seed,
    __half* control_lds) {
    F16x8 fragment{};
#if defined(__gfx946__) || defined(__gfx92a__)
    const ins::Vec4U32 src = ins::prepare_matrix_src(seed, kCols);
    ins::matrix_load_32x16_b16_bps_lds(control_lds, src, 0);
    ins::wait_vbcnt0();
    ins::wait_lgkm(0);
    fragment.vec = __builtin_hcu_ds_read_matrix_format_f16(
        control_lds, kCols, 2, 1, 0);
    ins::wait_lgkm(0);
#else
    (void)seed;
    (void)control_lds;
#endif
    return fragment;
}

__device__ __forceinline__ F16x8 repack_from_fp32(F16x8 source) {
    F32x8 fp32{};
    F16x8 packed{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        fp32.scalar[i] = static_cast<float>(source.scalar[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        packed.scalar[i] = static_cast<_Float16>(fp32.scalar[i]);
    }
    return packed;
}

template <int Transpose, int Reverse>
__device__ __forceinline__ void matrix_store_tile(
    __half* lds,
    __half* output,
    int& store_phase) {
#if defined(__gfx946__) || defined(__gfx92a__)
    // HCU matrix_store bools are (t, r, glc, slc). This minimal row-major
    // control uses t=1/r=0 with glc/slc false. HCU matrix_store is the
    // LDS-to-global direction. The resource
    // descriptor describes the row-major half output tile; the second
    // operand is the swizzled LDS page produced by ds_write_matrix.
    const ins::Vec4U32 dst = ins::prepare_matrix_src(output, kCols);
    static_assert(__has_builtin(__builtin_hcu_matrix_store_32x16_b16),
                  "latest HCU matrix-store builtin is required");
    ins::abarrier_seq<false>(kStoreBarrier);
    __builtin_hcu_matrix_store_32x16_b16(
        dst, reinterpret_cast<short*>(lds), 0, Transpose != 0, Reverse != 0,
        false, false);
    ins::abarrier_arrive_cnt<false>(kStoreBarrier, 1);
    ins::abarrier_try_wait<false>(kStoreBarrier, store_phase);
    ins::wait_vmem_lgkm();
#else
    (void)lds;
    (void)output;
    (void)store_phase;
#endif
}

template <int Transpose, int Reverse>
__device__ __forceinline__ void write_and_store_matrix_tile(
    const F16x8& packed,
    __half* lds,
    __half* output,
    int& store_phase) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_hcu_ds_write_matrix_format_f16(
        packed.vec, lds, kCols, 2, 1, 0, 0);
    ins::wait_lgkm(0);
    matrix_store_tile<Transpose, Reverse>(lds, output, store_phase);
#else
    (void)packed;
    (void)lds;
    (void)output;
    (void)store_phase;
#endif
}

__global__ void __launch_bounds__(kThreads, 1)
    dkv_b16_matrix_store_probe_kernel(const __half* seed,
                                       __half* output,
                                       uint16_t* fragment_dump) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(32, 160, 160, 160);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(kStoreBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    if (wave_id == 0) {
        int store_phase = 0;
        const F16x8 control = load_control_fragment(seed, lds);
        const F16x8 packed = repack_from_fp32(control);
        matrix_store_tile<1, 0>(
            lds, output + 0 * kCandidateStrideElems, store_phase);

        write_and_store_matrix_tile<1, 0>(
            packed, lds + kElems, output + 1 * kCandidateStrideElems,
            store_phase);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            fragment_dump[lane * 8 + i] = packed.bits[i];
        }
    }
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(kStoreBarrier);
    }
#else
    (void)seed;
    (void)output;
    (void)fragment_dump;
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
        "comparison_contract=C2_b16_dswrite_matrix_store; "
        "C0_fp32_oracle=pending; C1_packed_b16_direct_global=pending\n");
    std::vector<_Float16> expected(kElems);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            expected[row * kCols + col] =
                static_cast<_Float16>(1.0f + row * kCols + col);
        }
    }

    __half* seed_device = nullptr;
    __half* output_device = nullptr;
    uint16_t* fragment_device = nullptr;
    check_hip(hipMalloc(&seed_device, kElems * sizeof(__half)),
              "hipMalloc seed");
    check_hip(hipMalloc(&output_device,
                        kCandidates * kCandidateStrideElems * sizeof(__half)),
              "hipMalloc output");
    check_hip(hipMalloc(&fragment_device, kWaveSize * 8 * sizeof(uint16_t)),
              "hipMalloc fragment");
    check_hip(hipMemcpy(seed_device, expected.data(),
                        kElems * sizeof(__half), hipMemcpyHostToDevice),
              "copy seed");
    check_hip(hipMemset(output_device, 0,
                        kCandidates * kCandidateStrideElems * sizeof(__half)),
              "clear output");
    check_hip(hipMemset(fragment_device, 0,
                        kWaveSize * 8 * sizeof(uint16_t)),
              "clear fragment");

    hipLaunchKernelGGL(dkv_b16_matrix_store_probe_kernel, dim3(1),
                       dim3(kThreads), 0, 0, seed_device, output_device,
                       fragment_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");

    std::vector<_Float16> actual(kCandidates * kCandidateStrideElems);
    std::vector<uint16_t> fragment(kWaveSize * 8);
    check_hip(hipMemcpy(actual.data(), output_device,
                        actual.size() * sizeof(__half),
                        hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipMemcpy(fragment.data(), fragment_device,
                        fragment.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "copy fragment");
    check_hip(hipFree(fragment_device), "free fragment");
    check_hip(hipFree(output_device), "free output");
    check_hip(hipFree(seed_device), "free seed");

    const int transpose_modes[kStoreModes] = {1};
    const int reverse_modes[kStoreModes] = {0};
    bool matrix_control_pass = false;
    bool writer_chain_pass = false;
    for (int candidate = 0; candidate < kCandidates; ++candidate) {
        const int path = candidate / kStoreModes;
        const int mode = candidate % kStoreModes;
        int row_major_mismatches = 0;
        int transpose_mismatches = 0;
        float row_major_max_abs = 0.0f;
        float transpose_max_abs = 0.0f;
        int first_row = -1;
        int first_col = -1;
        float first_got = 0.0f;
        float first_want = 0.0f;
        for (int row = 0; row < kRows; ++row) {
            for (int col = 0; col < kCols; ++col) {
                const int candidate_base = candidate * kCandidateStrideElems;
                const float got_row_major = static_cast<float>(
                    actual[candidate_base + row * kCols + col]);
                const float got_transpose = static_cast<float>(
                    actual[candidate_base + col * kRows + row]);
                const float want = static_cast<float>(expected[row * kCols + col]);
                const float row_major_diff = std::fabs(got_row_major - want);
                const float transpose_diff = std::fabs(got_transpose - want);
                row_major_max_abs = std::max(row_major_max_abs, row_major_diff);
                transpose_max_abs = std::max(transpose_max_abs, transpose_diff);
                if (row_major_diff > kTolerance) {
                    ++row_major_mismatches;
                    if (first_row < 0) {
                        first_row = row;
                        first_col = col;
                        first_got = got_row_major;
                        first_want = want;
                    }
                }
                transpose_mismatches += transpose_diff > kTolerance;
            }
        }
        const bool pass = row_major_mismatches == 0;
        matrix_control_pass |= path == 0 && pass;
        writer_chain_pass |= path == 1 && pass;
        std::printf(
            "b16_matrix_store path=%s candidate=t%d_r%d mismatches=%d "
            "transpose_mismatches=%d max_abs=%g transpose_max_abs=%g "
            "first_row=%d first_col=%d first_got=%g first_want=%g pass=%d\n",
            path == 0 ? "mls_direct" : "dswrite_chain",
            transpose_modes[mode], reverse_modes[mode], row_major_mismatches,
            transpose_mismatches, row_major_max_abs, transpose_max_abs,
            first_row, first_col, first_got, first_want, pass ? 1 : 0);
    }
    std::printf("b16_matrix_store_probe fragment_words=%zu\n",
                fragment.size());
    std::printf(
        "b16_matrix_store_probe matrix_control_pass=%d writer_chain_pass=%d\n",
        matrix_control_pass ? 1 : 0, writer_chain_pass ? 1 : 0);
    return matrix_control_pass ? (writer_chain_pass ? 0 : 4) : 3;
}
