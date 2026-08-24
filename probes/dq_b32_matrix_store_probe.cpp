#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kDstStride = 128;
constexpr int kStorageElements = kRows * kDstStride;
constexpr int kMmacModes = 4;
constexpr int kStoreModes = 4;
constexpr int kMfmtModes = 3;
constexpr int kCandidates = kMmacModes * kStoreModes * kMfmtModes;

union F32Bits {
    ins::Vec4F32 f32;
    ins::Vec4U32 u32;
};

uint32_t float_bits(float value) {
    union {
        float f32;
        uint32_t u32;
    } bits{value};
    return bits.u32;
}

__device__ __forceinline__ ins::Vec4U32 matrix_desc(
    const float* ptr, int row_stride_elements, int mfmt) {
    ins::Vec4U32 desc{};
    *reinterpret_cast<unsigned long long*>(&desc) =
        reinterpret_cast<unsigned long long>(ptr) & 0xffffffffffffULL;
    desc[2] = static_cast<uint32_t>(row_stride_elements);
    desc[3] = static_cast<uint32_t>(mfmt) << 17;
    return desc;
}

template <int Lit, int Lts>
__device__ __forceinline__ ins::Vec4F32 mmac_mode(
    ins::Vec4F16 lhs, ins::Vec4F16 rhs) {
    ins::Vec4F32 zero = {0.0f, 0.0f, 0.0f, 0.0f};
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs, rhs, zero, Lit, Lts);
}

__device__ __forceinline__ ins::Vec4F32 select_mmac(
    int mode, ins::Vec4F16 lhs, ins::Vec4F16 rhs) {
    switch (mode) {
        case 0:
            return mmac_mode<0, 0>(lhs, rhs);
        case 1:
            return mmac_mode<1, 0>(lhs, rhs);
        case 2:
            return mmac_mode<0, 1>(lhs, rhs);
        default:
            return mmac_mode<1, 1>(lhs, rhs);
    }
}

__device__ __forceinline__ void matrix_store_b32(
    int mode, ins::Vec4U32 value, ins::Vec4U32 dst) {
    const uint32_t offset = 0;
    switch (mode) {
        case 0:
            asm volatile("matrix_store_16x16_b32 %0 %1 %2\n"
                         :
                         : "v"(value), "s"(dst), "s"(offset)
                         : "memory");
            break;
        case 1:
            asm volatile("matrix_store_16x16_b32 %0 %1 %2 t\n"
                         :
                         : "v"(value), "s"(dst), "s"(offset)
                         : "memory");
            break;
        case 2:
            asm volatile("matrix_store_16x16_b32 %0 %1 %2 r\n"
                         :
                         : "v"(value), "s"(dst), "s"(offset)
                         : "memory");
            break;
        default:
            asm volatile("matrix_store_16x16_b32 %0 %1 %2 t r\n"
                         :
                         : "v"(value), "s"(dst), "s"(offset)
                         : "memory");
            break;
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dq_b32_matrix_store_probe_kernel(
        float* control, float* candidate_output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int candidate = static_cast<int>(blockIdx.x);
    const int mmac = candidate / (kStoreModes * kMfmtModes);
    const int store = (candidate / kMfmtModes) & 3;
    const int mfmt = candidate % kMfmtModes;
    const int row = lane & 15;
    const int lane_col = (lane >> 4) * 4;

    ins::Vec4F16 lhs{};
    ins::Vec4F16 rhs{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lhs[i] = static_cast<_Float16>(
            static_cast<float>((lane * 4 + i) % 29 + 1) / 32.0f);
        rhs[i] = static_cast<_Float16>(
            static_cast<float>((lane * 7 + i * 3) % 31 + 1) / 64.0f);
    }

    F32Bits result{};
    result.f32 = select_mmac(mmac, lhs, rhs);
    auto* control_vec = reinterpret_cast<ins::Vec4F32*>(
        control + candidate * kStorageElements + row * kDstStride + lane_col);
    *control_vec = result.f32;

    matrix_store_b32(
        store,
        result.u32,
        matrix_desc(candidate_output + candidate * kStorageElements,
                    kDstStride, mfmt));
    ins::wait_vmem_lgkm();
#else
    (void)control;
    (void)candidate_output;
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
    std::vector<float> control(kCandidates * kStorageElements, -1.0f);
    std::vector<float> candidate(kCandidates * kStorageElements, -2.0f);
    float* control_device = nullptr;
    float* candidate_device = nullptr;
    const size_t bytes = control.size() * sizeof(float);
    check_hip(hipMalloc(&control_device, bytes), "hipMalloc control");
    check_hip(hipMalloc(&candidate_device, bytes), "hipMalloc candidate");
    check_hip(hipMemcpy(control_device, control.data(), bytes,
                        hipMemcpyHostToDevice),
              "initialize control");
    check_hip(hipMemcpy(candidate_device, candidate.data(), bytes,
                        hipMemcpyHostToDevice),
              "initialize candidate");

    hipLaunchKernelGGL(dq_b32_matrix_store_probe_kernel,
                       dim3(kCandidates), dim3(kWaveSize), 0, 0,
                       control_device, candidate_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(control.data(), control_device, bytes,
                        hipMemcpyDeviceToHost),
              "copy control");
    check_hip(hipMemcpy(candidate.data(), candidate_device, bytes,
                        hipMemcpyDeviceToHost),
              "copy candidate");
    check_hip(hipFree(control_device), "hipFree control");
    check_hip(hipFree(candidate_device), "hipFree candidate");

    int passing = 0;
    for (int slot = 0; slot < kCandidates; ++slot) {
        int mismatches = 0;
        int guard_mismatches = 0;
        int first = -1;
        for (int row = 0; row < kRows; ++row) {
            for (int col = 0; col < kDstStride; ++col) {
                const int offset =
                    slot * kStorageElements + row * kDstStride + col;
                if (col < kCols) {
                    if (float_bits(control[offset]) !=
                        float_bits(candidate[offset])) {
                        ++mismatches;
                        if (first < 0) {
                            first = row * kCols + col;
                        }
                    }
                } else if (candidate[offset] != -2.0f) {
                    ++guard_mismatches;
                }
            }
        }
        passing += mismatches == 0 && guard_mismatches == 0;
        const int mmac = slot / (kStoreModes * kMfmtModes);
        const int store = (slot / kMfmtModes) & 3;
        const int mfmt = slot % kMfmtModes;
        std::printf(
            "dq_b32_vgpr_matrix_store mmac_lit=%d mmac_lts=%d "
            "store_t=%d store_r=%d mfmt=%d stride=%d mismatches=%d "
            "guard=%d first=%d pass=%d\n",
            mmac & 1,
            (mmac >> 1) & 1,
            store & 1,
            (store >> 1) & 1,
            mfmt,
            kDstStride,
            mismatches,
            guard_mismatches,
            first,
            mismatches == 0 && guard_mismatches == 0 ? 1 : 0);
    }
    std::printf("dq_b32_vgpr_matrix_store passing=%d/%d\n",
                passing, kCandidates);
    return passing > 0 ? 0 : 1;
}
