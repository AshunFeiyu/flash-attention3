#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kValuesPerLane = 4;
constexpr int kValues = kWaveSize * kValuesPerLane;
constexpr int kCandidates = 4;
constexpr int kPageBytes = 2048;
constexpr int kLdsBytes = 4 * kPageBytes;
constexpr float kTolerance = 1.0e-5f;

union F32x4 {
    ins::Vec4F32 vec;
    float scalar[4];
};

union F16x4 {
    ins::Vec4F16 vec;
    _Float16 scalar[4];
};

template <bool Transpose>
__device__ __forceinline__ void write_f32(ins::Vec4F32 value,
                                          unsigned char* lds,
                                          int byte_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<float*>(lds + byte_offset);
    __builtin_hcu_ds_write_matrix_format_f32(
        value, ptr, 0, 1, 1, 0, Transpose ? 1 : 0);
#else
    (void)value;
    (void)lds;
    (void)byte_offset;
#endif
}

template <bool Transpose>
__device__ __forceinline__ ins::Vec4F32 read_f32(
    unsigned char* lds,
    int byte_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<float*>(lds + byte_offset);
    if constexpr (Transpose) {
        return __builtin_hcu_ds_read_matrix_trans_format_f32(
            ptr, 0, 1, 1, 0);
    }
    return __builtin_hcu_ds_read_matrix_format_f32(ptr, 0, 1, 1, 0);
#else
    (void)lds;
    (void)byte_offset;
    return {};
#endif
}

__device__ __forceinline__ ins::Vec4F32 run_mmac(ins::Vec4F16 lhs,
                                                 ins::Vec4F16 rhs) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    return ins::mmac_f16_lit(lhs, rhs, zero.f32);
}

__device__ __forceinline__ ins::Vec4F16 to_f16(ins::Vec4F32 value) {
    F32x4 src{.vec = value};
    F16x4 dst{};
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
        dst.scalar[i] = static_cast<_Float16>(src.scalar[i]);
    }
    return dst.vec;
}

__device__ __forceinline__ void store_vec(float* out,
                                          int candidate,
                                          int lane,
                                          ins::Vec4F32 value) {
    F32x4 data{.vec = value};
    const int base = (candidate * kWaveSize + lane) * kValuesPerLane;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
        out[base + i] = data.scalar[i];
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dkv_pds_f32_roundtrip_probe_kernel(float* tag_reads,
                                       float* semantic_reads,
                                       float* direct_mmac,
                                       float* candidate_mmac) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) unsigned char lds[kLdsBytes];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    F32x4 tag{};
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
        tag.scalar[i] = static_cast<float>(lane * kValuesPerLane + i + 1);
    }

    F16x4 upstream_lhs{};
    F16x4 upstream_rhs{};
    F16x4 downstream_rhs{};
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
        const int lhs_code = (lane * 3 + i * 5) % 17 - 8;
        const int rhs_code = (lane * 7 + i * 3 + 1) % 19 - 9;
        const int down_code = (lane * 11 + i * 7 + 2) % 23 - 11;
        upstream_lhs.scalar[i] =
            static_cast<_Float16>(lhs_code * 0.03125f);
        upstream_rhs.scalar[i] =
            static_cast<_Float16>(rhs_code * 0.025f);
        downstream_rhs.scalar[i] =
            static_cast<_Float16>(down_code * 0.015625f);
    }
    const ins::Vec4F32 natural =
        run_mmac(upstream_lhs.vec, upstream_rhs.vec);

    constexpr int kTagT0 = 0 * kPageBytes;
    constexpr int kTagT1 = 1 * kPageBytes;
    constexpr int kSemanticT0 = 2 * kPageBytes;
    constexpr int kSemanticT1 = 3 * kPageBytes;
    write_f32<false>(tag.vec, lds, kTagT0);
    write_f32<true>(tag.vec, lds, kTagT1);
    write_f32<false>(natural, lds, kSemanticT0);
    write_f32<true>(natural, lds, kSemanticT1);
    ins::wait_lgkm(0);

    ins::Vec4F32 tag_candidate[kCandidates];
    ins::Vec4F32 semantic_candidate[kCandidates];
    tag_candidate[0] = read_f32<false>(lds, kTagT0);
    tag_candidate[1] = read_f32<true>(lds, kTagT0);
    tag_candidate[2] = read_f32<false>(lds, kTagT1);
    tag_candidate[3] = read_f32<true>(lds, kTagT1);
    semantic_candidate[0] = read_f32<false>(lds, kSemanticT0);
    semantic_candidate[1] = read_f32<true>(lds, kSemanticT0);
    semantic_candidate[2] = read_f32<false>(lds, kSemanticT1);
    semantic_candidate[3] = read_f32<true>(lds, kSemanticT1);
    ins::wait_lgkm(0);

    const ins::Vec4F32 direct =
        run_mmac(to_f16(natural), downstream_rhs.vec);
    store_vec(direct_mmac, 0, lane, direct);
#pragma unroll
    for (int candidate = 0; candidate < kCandidates; ++candidate) {
        store_vec(tag_reads, candidate, lane, tag_candidate[candidate]);
        store_vec(semantic_reads, candidate, lane,
                  semantic_candidate[candidate]);
        const ins::Vec4F32 actual = run_mmac(
            to_f16(semantic_candidate[candidate]), downstream_rhs.vec);
        store_vec(candidate_mmac, candidate, lane, actual);
    }
#else
    (void)tag_reads;
    (void)semantic_reads;
    (void)direct_mmac;
    (void)candidate_mmac;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
        std::exit(2);
    }
}

const char* candidate_name(int candidate) {
    static const char* names[kCandidates] = {
        "write_t0_read_normal",
        "write_t0_read_trans",
        "write_t1_read_normal",
        "write_t1_read_trans",
    };
    return names[candidate];
}

}  // namespace

int main() {
    constexpr size_t kCandidateFloats = kCandidates * kValues;
    float* tag_reads_device = nullptr;
    float* semantic_reads_device = nullptr;
    float* direct_mmac_device = nullptr;
    float* candidate_mmac_device = nullptr;
    check_hip(hipMalloc(&tag_reads_device,
                        kCandidateFloats * sizeof(float)),
              "hipMalloc tag reads");
    check_hip(hipMalloc(&semantic_reads_device,
                        kCandidateFloats * sizeof(float)),
              "hipMalloc semantic reads");
    check_hip(hipMalloc(&direct_mmac_device, kValues * sizeof(float)),
              "hipMalloc direct mmac");
    check_hip(hipMalloc(&candidate_mmac_device,
                        kCandidateFloats * sizeof(float)),
              "hipMalloc candidate mmac");

    hipLaunchKernelGGL(dkv_pds_f32_roundtrip_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, tag_reads_device,
                       semantic_reads_device, direct_mmac_device,
                       candidate_mmac_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");

    std::vector<float> tag_reads(kCandidateFloats);
    std::vector<float> semantic_reads(kCandidateFloats);
    std::vector<float> direct_mmac(kValues);
    std::vector<float> candidate_mmac(kCandidateFloats);
    check_hip(hipMemcpy(tag_reads.data(), tag_reads_device,
                        tag_reads.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy tag reads");
    check_hip(hipMemcpy(semantic_reads.data(), semantic_reads_device,
                        semantic_reads.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy semantic reads");
    check_hip(hipMemcpy(direct_mmac.data(), direct_mmac_device,
                        direct_mmac.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy direct mmac");
    check_hip(hipMemcpy(candidate_mmac.data(), candidate_mmac_device,
                        candidate_mmac.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy candidate mmac");
    check_hip(hipFree(candidate_mmac_device), "free candidate mmac");
    check_hip(hipFree(direct_mmac_device), "free direct mmac");
    check_hip(hipFree(semantic_reads_device), "free semantic reads");
    check_hip(hipFree(tag_reads_device), "free tag reads");

    bool any_semantic_pair = false;
    for (int candidate = 0; candidate < kCandidates; ++candidate) {
        std::vector<int> seen(kValues + 1, 0);
        int tag_nonfinite = 0;
        int tag_invalid = 0;
        int identity_errors = 0;
        int semantic_nonfinite = 0;
        int semantic_errors = 0;
        float max_abs = 0.0f;
        for (int i = 0; i < kValues; ++i) {
            const float tag = tag_reads[candidate * kValues + i];
            if (!std::isfinite(tag)) {
                ++tag_nonfinite;
            } else {
                const int code = static_cast<int>(std::lround(tag));
                if (std::fabs(tag - static_cast<float>(code)) > 0.0f ||
                    code < 1 || code > kValues) {
                    ++tag_invalid;
                } else {
                    ++seen[code];
                    identity_errors += code != i + 1 ? 1 : 0;
                }
            }
            const float read_value =
                semantic_reads[candidate * kValues + i];
            const float actual = candidate_mmac[candidate * kValues + i];
            const float expected = direct_mmac[i];
            if (!std::isfinite(read_value) || !std::isfinite(actual) ||
                !std::isfinite(expected)) {
                ++semantic_nonfinite;
                ++semantic_errors;
                continue;
            }
            const float diff = std::fabs(actual - expected);
            max_abs = std::max(max_abs, diff);
            semantic_errors += diff > kTolerance ? 1 : 0;
        }
        int permutation_errors = tag_nonfinite + tag_invalid;
        for (int code = 1; code <= kValues; ++code) {
            permutation_errors += seen[code] == 1 ? 0 : 1;
        }
        const bool semantic_pass = semantic_errors == 0;
        any_semantic_pair |= semantic_pass;
        std::printf(
            "f32_pds_roundtrip candidate=%s tag_nonfinite=%d "
            "permutation_errors=%d identity_errors=%d "
            "semantic_nonfinite=%d semantic_errors=%d max_abs=%g pass=%d\n",
            candidate_name(candidate), tag_nonfinite, permutation_errors,
            identity_errors, semantic_nonfinite, semantic_errors, max_abs,
            semantic_pass ? 1 : 0);
    }
    std::printf("f32_pds_roundtrip any_semantic_pair=%d\n",
                any_semantic_pair ? 1 : 0);
    return 0;
}
