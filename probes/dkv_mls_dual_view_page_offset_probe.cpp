#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

bool hip_ok(hipError_t status, const char* operation) {
    if (status == hipSuccess) {
        return true;
    }
    std::fprintf(
        stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
    return false;
}

constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kPage1Offset = 64 * 1024;
constexpr int kDoutInPageOffset = 32 * 1024;
constexpr int kPage0DoutOffset = kDoutInPageOffset;
constexpr int kPage1DoutOffset = 96 * 1024;
constexpr int kLdsBytes = 128 * 1024;

struct ProbeResult {
    uint32_t trans_mismatch;
    uint32_t normal_mismatch;
    uint32_t sidecar_mismatch;
    uint32_t page_base_trans_mismatch;
    uint32_t page_base_normal_mismatch;
};

__global__ void dual_view_page_offset_probe(
    const __half* src,
    ProbeResult* result) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int lane = static_cast<int>(threadIdx.x % 64);

    if (wave_id == 0) {
        const ins::Vec4U32 srsrc = ins::prepare_matrix_src(src, kCols);
        ins::matrix_load_32x16_b16_bps_lds(lds, srsrc, 0);
        ins::matrix_load_32x16_b16_bps_lds(
            lds, srsrc, kPage0DoutOffset);
        ins::matrix_load_32x16_b16_bps_lds(lds, srsrc, kPage1Offset);
        ins::matrix_load_32x16_b16_bps_lds(lds, srsrc, kPage1DoutOffset);
        ins::wait_vbcnt0();
    }
    __syncthreads();

    if (wave_id == 0) {
        float* sidecar = reinterpret_cast<float*>(lds);
        for (int i = lane; i < 3 * 128; i += 64) {
            sidecar[i] = static_cast<float>(i + 1) * 0.125f;
        }
    }
    __syncthreads();

    if (wave_id == 1) {
        ins::F16x8 trans0;
        ins::F16x8 trans1;
        ins::F16x8 normal0;
        ins::F16x8 normal1;
        ins::ds_read_matrix_32x16_trans(lds, kPage1Offset, trans0.f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, kPage1DoutOffset, trans1.f16x8);
        ins::ds_read_matrix_32x16_normal(lds, kPage1Offset, normal0.f16x8);
        ins::ds_read_matrix_32x16_normal(
            lds, kPage1DoutOffset, normal1.f16x8);
        ins::wait_lgkm(0);

        const uint32_t trans_mismatch =
            static_cast<uint32_t>(trans0.u64[0] != trans1.u64[0]) +
            static_cast<uint32_t>(trans0.u64[1] != trans1.u64[1]);
        const uint32_t normal_mismatch =
            static_cast<uint32_t>(normal0.u64[0] != normal1.u64[0]) +
            static_cast<uint32_t>(normal0.u64[1] != normal1.u64[1]);
        if (trans_mismatch != 0) {
            atomicAdd(&result->trans_mismatch, trans_mismatch);
        }
        if (normal_mismatch != 0) {
            atomicAdd(&result->normal_mismatch, normal_mismatch);
        }

        __half* page0 = lds;
        __half* page1 = lds + kPage1Offset / sizeof(__half);
        ins::F16x8 page0_q_trans;
        ins::F16x8 page0_dout_trans;
        ins::F16x8 page1_q_trans;
        ins::F16x8 page1_dout_trans;
        ins::ds_read_matrix_32x16_trans_imm2<0, kDoutInPageOffset>(
            page0, page0_q_trans.f16x8, page0_dout_trans.f16x8);
        ins::ds_read_matrix_32x16_trans_imm2<0, kDoutInPageOffset>(
            page1, page1_q_trans.f16x8, page1_dout_trans.f16x8);

        ins::F16x8 page0_normal[4];
        ins::F16x8 page1_normal[4];
        ins::ds_read_matrix_32x16_normal_imm4<
            kDoutInPageOffset,
            kDoutInPageOffset,
            kDoutInPageOffset,
            kDoutInPageOffset>(
            page0, page0_normal[0].f16x8, page0_normal[1].f16x8,
            page0_normal[2].f16x8, page0_normal[3].f16x8);
        ins::ds_read_matrix_32x16_normal_imm4<
            kDoutInPageOffset,
            kDoutInPageOffset,
            kDoutInPageOffset,
            kDoutInPageOffset>(
            page1, page1_normal[0].f16x8, page1_normal[1].f16x8,
            page1_normal[2].f16x8, page1_normal[3].f16x8);
        ins::wait_lgkm(0);

        const uint32_t page_base_trans_mismatch =
            static_cast<uint32_t>(
                page0_dout_trans.u64[0] != page1_dout_trans.u64[0]) +
            static_cast<uint32_t>(
                page0_dout_trans.u64[1] != page1_dout_trans.u64[1]);
        const uint32_t page_base_normal_mismatch =
            static_cast<uint32_t>(
                page0_normal[0].u64[0] != page1_normal[0].u64[0]) +
            static_cast<uint32_t>(
                page0_normal[0].u64[1] != page1_normal[0].u64[1]);
        if (page_base_trans_mismatch != 0) {
            atomicAdd(
                &result->page_base_trans_mismatch,
                page_base_trans_mismatch);
        }
        if (page_base_normal_mismatch != 0) {
            atomicAdd(
                &result->page_base_normal_mismatch,
                page_base_normal_mismatch);
        }

        const float* sidecar = reinterpret_cast<const float*>(lds);
        uint32_t sidecar_mismatch = 0;
        for (int i = lane; i < 3 * 128; i += 64) {
            const float expected = static_cast<float>(i + 1) * 0.125f;
            sidecar_mismatch += static_cast<uint32_t>(sidecar[i] != expected);
        }
        if (sidecar_mismatch != 0) {
            atomicAdd(&result->sidecar_mismatch, sidecar_mismatch);
        }
    }
#else
    (void)src;
    (void)result;
#endif
}

}  // namespace

int main() {
    std::vector<__half> src(kRows * kCols);
    for (int i = 0; i < kRows * kCols; ++i) {
        src[i] = __float2half(static_cast<float>((i % 97) - 48) * 0.03125f);
    }

    __half* src_dev = nullptr;
    ProbeResult* result_dev = nullptr;
    ProbeResult result{};
    if (!hip_ok(
            hipMalloc(
                reinterpret_cast<void**>(&src_dev),
                src.size() * sizeof(__half)),
            "hipMalloc(src)") ||
        !hip_ok(
            hipMalloc(reinterpret_cast<void**>(&result_dev), sizeof(result)),
            "hipMalloc(result)") ||
        !hip_ok(
            hipMemcpy(
                src_dev, src.data(), src.size() * sizeof(__half),
                hipMemcpyHostToDevice),
            "hipMemcpy(src)") ||
        !hip_ok(hipMemset(result_dev, 0, sizeof(result)), "hipMemset(result)")) {
        (void)hip_ok(hipFree(result_dev), "hipFree(result after setup error)");
        (void)hip_ok(hipFree(src_dev), "hipFree(src after setup error)");
        return 1;
    }

    hipLaunchKernelGGL(
        dual_view_page_offset_probe, dim3(1), dim3(128), 0, 0, src_dev,
        result_dev);
    if (!hip_ok(hipGetLastError(), "dual_view_page_offset_probe launch") ||
        !hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") ||
        !hip_ok(
            hipMemcpy(
                &result, result_dev, sizeof(result), hipMemcpyDeviceToHost),
            "hipMemcpy(result)")) {
        (void)hip_ok(hipFree(result_dev), "hipFree(result after run error)");
        (void)hip_ok(hipFree(src_dev), "hipFree(src after run error)");
        return 1;
    }

    const bool free_result_ok = hip_ok(hipFree(result_dev), "hipFree(result)");
    const bool free_src_ok = hip_ok(hipFree(src_dev), "hipFree(src)");
    const bool free_ok = free_result_ok && free_src_ok;
    const bool pass = result.trans_mismatch == 0 &&
                      result.normal_mismatch == 0 &&
                      result.sidecar_mismatch == 0 &&
                      result.page_base_trans_mismatch == 0 &&
                      result.page_base_normal_mismatch == 0;
    std::printf(
        "dkv_mls_dual_view_page_offset trans_mismatch=%u "
        "normal_mismatch=%u sidecar_mismatch=%u "
        "page_base_trans_mismatch=%u page_base_normal_mismatch=%u pass=%d\n",
        result.trans_mismatch, result.normal_mismatch,
        result.sidecar_mismatch, result.page_base_trans_mismatch,
        result.page_base_normal_mismatch, pass ? 1 : 0);
    return pass && free_ok ? 0 : 1;
}
