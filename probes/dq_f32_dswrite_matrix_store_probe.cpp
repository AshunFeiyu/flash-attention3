#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kInputLd = 32;
constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kOutputLd = 128;
constexpr int kOutputElements = kRows * kOutputLd;
constexpr int kInputPageBytes = 32 * 32 * sizeof(__half);
constexpr int kOutputPageBytes = 2048;
constexpr int kReplayPageOffset = 2 * kInputPageBytes;
constexpr int kOutputPageOffset = kReplayPageOffset + kOutputPageBytes;
constexpr int kLdsBytes = kOutputPageOffset + kOutputPageBytes;
constexpr int kReplayModes = 4;
constexpr int kMmacModes = 8;
constexpr int kCandidates = 1 + kReplayModes + kMmacModes;
constexpr float kGuard = -777.0f;
constexpr float kTolerance = 5.0e-3f;

__device__ __forceinline__ ins::Vec4U32 matrix_desc(
    const float* ptr, int row_stride_elements, int mfmt) {
    ins::Vec4U32 desc{};
    *reinterpret_cast<unsigned long long*>(&desc) =
        reinterpret_cast<unsigned long long>(ptr) & 0xffffffffffffULL;
    desc[2] = static_cast<uint32_t>(row_stride_elements);
    desc[3] = static_cast<uint32_t>(mfmt) << 17;
    return desc;
}

__device__ __forceinline__ void write_f32_fragment(
    bool transpose, ins::Vec4F32 value, float* lds) {
    if (transpose) {
        __builtin_hcu_ds_write_matrix_format_f32(
            value, lds, 0, 1, 1, 0, 1);
    } else {
        __builtin_hcu_ds_write_matrix_format_f32(
            value, lds, 0, 1, 1, 0, 0);
    }
}

__device__ __forceinline__ ins::Vec4F32 read_f32_fragment(
    bool transpose, float* lds) {
    if (transpose) {
        return __builtin_hcu_ds_read_matrix_trans_format_f32(
            lds, 0, 1, 1, 0);
    }
    return __builtin_hcu_ds_read_matrix_format_f32(lds, 0, 1, 1, 0);
}

__device__ __forceinline__ ins::Vec4F32 mmac_mode(
    int mode, ins::Vec4F16 lhs, ins::Vec4F16 rhs, ins::Vec4F32 zero) {
    switch (mode) {
        case 0:
            return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
                lhs, rhs, zero, 0, 0);
        case 1:
            return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
                lhs, rhs, zero, 1, 0);
        case 2:
            return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
                lhs, rhs, zero, 0, 1);
        default:
            return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
                lhs, rhs, zero, 1, 1);
    }
}

__device__ __forceinline__ void store_f32_tile(float* output, int* lds) {
    __builtin_hcu_matrix_store_16x16_b32(
        matrix_desc(output, kOutputLd, 0), lds, 0,
        false, false, false, false);
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dq_f32_dswrite_matrix_store_probe_kernel(
        const __half* __restrict__ a,
        const __half* __restrict__ b,
        const float* __restrict__ golden,
        float* __restrict__ output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) unsigned char lds[kLdsBytes];
    const int candidate = static_cast<int>(blockIdx.x);
    auto* matrix_lds = reinterpret_cast<__half*>(lds);
    auto* replay_lds = reinterpret_cast<float*>(lds + kReplayPageOffset);
    auto* output_lds = reinterpret_cast<float*>(lds + kOutputPageOffset);

    if (candidate == 0) {
        __builtin_hcu_matrix_load_16x16_b32(
            matrix_desc(golden, kCols, 0),
            reinterpret_cast<int*>(output_lds), 0,
            false, false, false, false, false);
        ins::wait_vmem_lgkm();
    } else if (candidate <= kReplayModes) {
        const int replay = candidate - 1;
        __builtin_hcu_matrix_load_16x16_b32(
            matrix_desc(golden, kCols, 0),
            reinterpret_cast<int*>(replay_lds), 0,
            false, false, false, false, false);
        ins::wait_vmem_lgkm();
        const ins::Vec4F32 fragment =
            read_f32_fragment((replay & 2) != 0, replay_lds);
        ins::wait_lgkm(0);
        write_f32_fragment((replay & 1) != 0, fragment, output_lds);
        ins::wait_lgkm(0);
    } else {
        const int mmac_writer = candidate - 1 - kReplayModes;
        const int mmac = mmac_writer >> 1;
        const bool writer_transpose = (mmac_writer & 1) != 0;
        ins::matrix_load_32x32_b16_bps_lds(
            matrix_lds, ins::prepare_matrix_src(a, kInputLd), 0, true);
        ins::matrix_load_32x32_b16_bps_lds(
            matrix_lds, ins::prepare_matrix_src(b, kInputLd),
            kInputPageBytes, false);
        ins::wait_vmem_lgkm();
        ins::wait_vbcnt0();
        __syncthreads();

        ins::F16x8 lhs{};
        ins::F16x8 rhs{};
        ins::ds_read_matrix_32x16_trans(matrix_lds, 0, lhs.f16x8);
        ins::ds_read_matrix_32x16_normal(
            matrix_lds + kInputPageBytes / sizeof(__half), 0, rhs.f16x8);
        ins::wait_lgkm(0);

        ins::F32x4 acc{};
        ins::zero_vgpr2(acc.u64[0]);
        ins::zero_vgpr2(acc.u64[1]);
        acc.f32 = mmac_mode(
            mmac, lhs.f16x4[0], rhs.f16x4[0], acc.f32);

        write_f32_fragment(writer_transpose, acc.f32, output_lds);
        ins::wait_lgkm(0);
    }
    store_f32_tile(output + candidate * kOutputElements,
                   reinterpret_cast<int*>(output_lds));
    ins::wait_vmem_lgkm();
#else
    (void)a;
    (void)b;
    (void)golden;
    (void)output;
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
    std::vector<__half> a(kInputLd * kInputLd, __half(0.0f));
    std::vector<__half> b(kInputLd * kInputLd, __half(0.0f));
    float golden[kRows][kCols]{};
    std::vector<float> golden_storage(kRows * kCols, 0.0f);
    for (int row = 0; row < kRows; ++row) {
        for (int k = 0; k < kCols; ++k) {
            a[row * kInputLd + k] =
                __half(((row * 7 + k * 3) % 13 - 6) * 0.25f);
        }
    }
    for (int k = 0; k < kRows; ++k) {
        for (int col = 0; col < kCols; ++col) {
            b[k * kInputLd + col] =
                __half(((k * 5 + col * 11) % 9 - 4) * 0.5f);
        }
    }
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            for (int k = 0; k < kCols; ++k) {
                golden[row][col] +=
                    static_cast<float>(a[row * kInputLd + k]) *
                    static_cast<float>(b[k * kInputLd + col]);
            }
            golden_storage[row * kCols + col] = golden[row][col];
        }
    }

    const size_t input_bytes = a.size() * sizeof(__half);
    std::vector<float> output(kCandidates * kOutputElements, kGuard);
    __half* a_device = nullptr;
    __half* b_device = nullptr;
    float* golden_device = nullptr;
    float* output_device = nullptr;
    check_hip(hipMalloc(&a_device, input_bytes), "hipMalloc a");
    check_hip(hipMalloc(&b_device, input_bytes), "hipMalloc b");
    check_hip(hipMalloc(&golden_device,
                        golden_storage.size() * sizeof(float)),
              "hipMalloc golden");
    check_hip(hipMalloc(&output_device, output.size() * sizeof(float)),
              "hipMalloc output");
    check_hip(hipMemcpy(a_device, a.data(), input_bytes,
                        hipMemcpyHostToDevice),
              "copy a");
    check_hip(hipMemcpy(b_device, b.data(), input_bytes,
                        hipMemcpyHostToDevice),
              "copy b");
    check_hip(hipMemcpy(golden_device, golden_storage.data(),
                        golden_storage.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "copy golden");
    check_hip(hipMemcpy(output_device, output.data(),
                        output.size() * sizeof(float), hipMemcpyHostToDevice),
              "initialize output");

    hipLaunchKernelGGL(dq_f32_dswrite_matrix_store_probe_kernel,
                       dim3(kCandidates), dim3(kWaveSize), 0, 0,
                       a_device, b_device, golden_device, output_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(output.data(), output_device,
                        output.size() * sizeof(float), hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipFree(output_device), "free output");
    check_hip(hipFree(golden_device), "free golden");
    check_hip(hipFree(b_device), "free b");
    check_hip(hipFree(a_device), "free a");

    int passing = 0;
    for (int candidate = 0; candidate < kCandidates; ++candidate) {
        int mismatches = 0;
        int nonfinite = 0;
        int guard_mismatches = 0;
        int first = -1;
        float max_abs = 0.0f;
        for (int row = 0; row < kRows; ++row) {
            for (int col = 0; col < kOutputLd; ++col) {
                const int offset =
                    candidate * kOutputElements + row * kOutputLd + col;
                if (col < kCols) {
                    const float diff =
                        std::fabs(output[offset] - golden[row][col]);
                    max_abs = std::max(max_abs, diff);
                    if (!std::isfinite(output[offset])) {
                        ++nonfinite;
                    }
                    if (!std::isfinite(output[offset]) || diff > kTolerance) {
                        ++mismatches;
                        if (first < 0) {
                            first = row * kCols + col;
                        }
                    }
                } else if (output[offset] != kGuard) {
                    ++guard_mismatches;
                }
            }
        }
        const bool pass = mismatches == 0 && guard_mismatches == 0;
        passing += pass ? 1 : 0;
        char source_name[64]{};
        if (candidate == 0) {
            std::snprintf(source_name, sizeof(source_name), "mls_control");
        } else if (candidate <= kReplayModes) {
            const int replay = candidate - 1;
            std::snprintf(source_name, sizeof(source_name),
                          "mls_read_t%d_write_t%d", (replay >> 1) & 1,
                          replay & 1);
        } else {
            const int mmac_writer = candidate - 1 - kReplayModes;
            const int mmac = mmac_writer >> 1;
            std::snprintf(source_name, sizeof(source_name),
                          "mmac_lit%d_lts%d_write_t%d", mmac & 1,
                          (mmac >> 1) & 1, mmac_writer & 1);
        }
        std::printf(
            "dq_f32_dswrite_store source=%s mismatches=%d nonfinite=%d "
            "guard=%d first=%d "
            "max_abs=%g pass=%d\n",
            source_name, mismatches, nonfinite, guard_mismatches, first,
            max_abs, pass ? 1 : 0);
    }
    std::printf("dq_f32_dswrite_store passing=%d/%d\n",
                passing, kCandidates);
    return passing > 0 ? 0 : 1;
}
