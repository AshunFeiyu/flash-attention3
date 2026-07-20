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

constexpr int kHeadDim = 128;
constexpr int kLookaheadRows = 48;
constexpr int kMBlocks = kLookaheadRows / 16;
constexpr int kDBlocks = kHeadDim / 32;
constexpr int kRawBlockBytes = 16 * 32 * sizeof(__half);
constexpr int kMainQBase = 0;
constexpr int kMainDoutBase = 192 * kHeadDim * sizeof(__half);
constexpr int kLookaheadQBase = 2 * 192 * kHeadDim * sizeof(__half);
constexpr int kLookaheadDoutBase =
    kLookaheadQBase + kLookaheadRows * kHeadDim * sizeof(__half);
constexpr int kMainSidecarBase =
    kLookaheadDoutBase + kLookaheadRows * kHeadDim * sizeof(__half);
constexpr int kLookaheadSidecarBase =
    kMainSidecarBase + 3 * 192 * sizeof(float);
constexpr int kPlannedLdsBytes =
    kLookaheadSidecarBase + 3 * kLookaheadRows * sizeof(float);
constexpr int kLdsBytes = 128 * 1024;

static_assert(kMainDoutBase == 49152);
static_assert(kLookaheadQBase == 98304);
static_assert(kLookaheadDoutBase == 110592);
static_assert(kMainSidecarBase == 122880);
static_assert(kLookaheadSidecarBase == 125184);
static_assert(kPlannedLdsBytes == 125760);
static_assert(kPlannedLdsBytes <= kLdsBytes);

constexpr int raw_offset(int m_block, int d_block) {
    return (m_block * kDBlocks + d_block) * kRawBlockBytes;
}

struct ProbeResult {
    uint32_t trans_mismatch;
    uint32_t normal_mismatch;
    uint32_t dual_base_mismatch;
    uint32_t sidecar_mismatch;
};

__global__ void dual_view_page_offset_probe(
    const __half* q,
    const __half* dout,
    ProbeResult* result) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);
    const int lane = static_cast<int>(threadIdx.x % 64);

    if (wave_id < 4) {
#pragma unroll
        for (int m_block = 0; m_block < kMBlocks; ++m_block) {
            const int block_offset = raw_offset(m_block, wave_local);
            const __half* q_tile =
                q + static_cast<int64_t>(m_block * 16) * kHeadDim +
                wave_local * 32;
            const __half* dout_tile =
                dout + static_cast<int64_t>(m_block * 16) * kHeadDim +
                wave_local * 32;
            const ins::Vec4U32 q_src =
                ins::prepare_matrix_src(q_tile, kHeadDim);
            const ins::Vec4U32 dout_src =
                ins::prepare_matrix_src(dout_tile, kHeadDim);
            ins::matrix_load_32x16_b16_bps_lds(
                lds, q_src, kMainQBase + block_offset);
            ins::matrix_load_32x16_b16_bps_lds(
                lds, dout_src, kMainDoutBase + block_offset);
            ins::matrix_load_32x16_b16_bps_lds(
                lds, q_src, kLookaheadQBase + block_offset);
            ins::matrix_load_32x16_b16_bps_lds(
                lds, dout_src, kLookaheadDoutBase + block_offset);
        }
        ins::wait_vbcnt0();
    }
    __syncthreads();

    if (wave_id == 0 && lane < kLookaheadRows) {
        float* main_sidecar = reinterpret_cast<float*>(
            reinterpret_cast<char*>(lds) + kMainSidecarBase);
        float* lookahead_sidecar = reinterpret_cast<float*>(
            reinterpret_cast<char*>(lds) + kLookaheadSidecarBase);
#pragma unroll
        for (int field = 0; field < 3; ++field) {
            const float value =
                static_cast<float>(field * kLookaheadRows + lane + 1) *
                0.125f;
            main_sidecar[field * 192 + lane] = value;
            lookahead_sidecar[field * kLookaheadRows + lane] = value;
        }
    }
    __syncthreads();

    if (wave_id == 4) {
        uint32_t trans_mismatch = 0;
        uint32_t normal_mismatch = 0;
#pragma unroll 1
        for (int m_block = 0; m_block < kMBlocks; ++m_block) {
#pragma unroll 1
            for (int d_block = 0; d_block < kDBlocks; ++d_block) {
                const int block_offset = raw_offset(m_block, d_block);
                {
                    ins::F16x8 main_fragment;
                    ins::F16x8 lookahead_fragment;
                    ins::ds_read_matrix_32x16_trans(
                        lds, kMainQBase + block_offset,
                        main_fragment.f16x8);
                    ins::ds_read_matrix_32x16_trans(
                        lds, kLookaheadQBase + block_offset,
                        lookahead_fragment.f16x8);
                    ins::wait_lgkm(0);
#pragma unroll
                    for (int word = 0; word < 2; ++word) {
                        trans_mismatch += static_cast<uint32_t>(
                            main_fragment.u64[word] !=
                            lookahead_fragment.u64[word]);
                    }
                }
                {
                    ins::F16x8 main_fragment;
                    ins::F16x8 lookahead_fragment;
                    ins::ds_read_matrix_32x16_trans(
                        lds, kMainDoutBase + block_offset,
                        main_fragment.f16x8);
                    ins::ds_read_matrix_32x16_trans(
                        lds, kLookaheadDoutBase + block_offset,
                        lookahead_fragment.f16x8);
                    ins::wait_lgkm(0);
#pragma unroll
                    for (int word = 0; word < 2; ++word) {
                        trans_mismatch += static_cast<uint32_t>(
                            main_fragment.u64[word] !=
                            lookahead_fragment.u64[word]);
                    }
                }
                {
                    ins::F16x8 main_fragment;
                    ins::F16x8 lookahead_fragment;
                    ins::ds_read_matrix_32x16_normal(
                        lds, kMainQBase + block_offset,
                        main_fragment.f16x8);
                    ins::ds_read_matrix_32x16_normal(
                        lds, kLookaheadQBase + block_offset,
                        lookahead_fragment.f16x8);
                    ins::wait_lgkm(0);
#pragma unroll
                    for (int word = 0; word < 2; ++word) {
                        normal_mismatch += static_cast<uint32_t>(
                            main_fragment.u64[word] !=
                            lookahead_fragment.u64[word]);
                    }
                }
                {
                    ins::F16x8 main_fragment;
                    ins::F16x8 lookahead_fragment;
                    ins::ds_read_matrix_32x16_normal(
                        lds, kMainDoutBase + block_offset,
                        main_fragment.f16x8);
                    ins::ds_read_matrix_32x16_normal(
                        lds, kLookaheadDoutBase + block_offset,
                        lookahead_fragment.f16x8);
                    ins::wait_lgkm(0);
#pragma unroll
                    for (int word = 0; word < 2; ++word) {
                        normal_mismatch += static_cast<uint32_t>(
                            main_fragment.u64[word] !=
                            lookahead_fragment.u64[word]);
                    }
                }
            }
        }
        if (trans_mismatch != 0) {
            atomicAdd(&result->trans_mismatch, trans_mismatch);
        }
        if (normal_mismatch != 0) {
            atomicAdd(&result->normal_mismatch, normal_mismatch);
        }

        const __half* lookahead_q =
            lds + kLookaheadQBase / sizeof(__half);
        const __half* lookahead_dout =
            lds + kLookaheadDoutBase / sizeof(__half);
        ins::F16x8 look_tq0;
        ins::F16x8 look_td0;
        ins::F16x8 look_tq1;
        ins::F16x8 look_td1;
        ins::ds_read_matrix_32x16_trans_dual_base_imm2<0, kRawBlockBytes>(
            lookahead_q, lookahead_dout, look_tq0.f16x8,
            look_td0.f16x8, look_tq1.f16x8, look_td1.f16x8);
        ins::wait_lgkm(0);
        uint32_t dual_base_mismatch = 0;
        {
            ins::F16x8 expected;
            ins::ds_read_matrix_32x16_trans(
                lds, kLookaheadQBase, expected.f16x8);
            ins::wait_lgkm(0);
#pragma unroll
            for (int word = 0; word < 2; ++word) {
                dual_base_mismatch += static_cast<uint32_t>(
                    expected.u64[word] != look_tq0.u64[word]);
            }
        }
        {
            ins::F16x8 expected;
            ins::ds_read_matrix_32x16_trans(
                lds, kLookaheadDoutBase, expected.f16x8);
            ins::wait_lgkm(0);
#pragma unroll
            for (int word = 0; word < 2; ++word) {
                dual_base_mismatch += static_cast<uint32_t>(
                    expected.u64[word] != look_td0.u64[word]);
            }
        }
        {
            ins::F16x8 expected;
            ins::ds_read_matrix_32x16_trans(
                lds, kLookaheadQBase + kRawBlockBytes,
                expected.f16x8);
            ins::wait_lgkm(0);
#pragma unroll
            for (int word = 0; word < 2; ++word) {
                dual_base_mismatch += static_cast<uint32_t>(
                    expected.u64[word] != look_tq1.u64[word]);
            }
        }
        {
            ins::F16x8 expected;
            ins::ds_read_matrix_32x16_trans(
                lds, kLookaheadDoutBase + kRawBlockBytes,
                expected.f16x8);
            ins::wait_lgkm(0);
#pragma unroll
            for (int word = 0; word < 2; ++word) {
                dual_base_mismatch += static_cast<uint32_t>(
                    expected.u64[word] != look_td1.u64[word]);
            }
        }
        if (dual_base_mismatch != 0) {
            atomicAdd(&result->dual_base_mismatch, dual_base_mismatch);
        }

        const float* main_sidecar = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(lds) + kMainSidecarBase);
        const float* lookahead_sidecar = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(lds) + kLookaheadSidecarBase);
        uint32_t sidecar_mismatch = 0;
        if (lane < kLookaheadRows) {
#pragma unroll
            for (int field = 0; field < 3; ++field) {
                sidecar_mismatch += static_cast<uint32_t>(
                    main_sidecar[field * 192 + lane] !=
                    lookahead_sidecar[field * kLookaheadRows + lane]);
            }
        }
        if (sidecar_mismatch != 0) {
            atomicAdd(&result->sidecar_mismatch, sidecar_mismatch);
        }
    }
#else
    (void)q;
    (void)dout;
    (void)result;
#endif
}

}  // namespace

int main() {
    std::vector<__half> q(kLookaheadRows * kHeadDim);
    std::vector<__half> dout(kLookaheadRows * kHeadDim);
    for (int i = 0; i < kLookaheadRows * kHeadDim; ++i) {
        q[i] = __float2half(static_cast<float>((i % 97) - 48) * 0.03125f);
        dout[i] =
            __float2half(static_cast<float>((i % 89) - 44) * 0.046875f);
    }

    __half* q_dev = nullptr;
    __half* dout_dev = nullptr;
    ProbeResult* result_dev = nullptr;
    ProbeResult result{};
    const size_t bytes = q.size() * sizeof(__half);
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&q_dev), bytes),
                "hipMalloc(q)") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&dout_dev), bytes),
                "hipMalloc(dout)") ||
        !hip_ok(
            hipMalloc(reinterpret_cast<void**>(&result_dev), sizeof(result)),
            "hipMalloc(result)") ||
        !hip_ok(
            hipMemcpy(q_dev, q.data(), bytes, hipMemcpyHostToDevice),
            "hipMemcpy(q)") ||
        !hip_ok(
            hipMemcpy(
                dout_dev, dout.data(), bytes, hipMemcpyHostToDevice),
            "hipMemcpy(dout)") ||
        !hip_ok(hipMemset(result_dev, 0, sizeof(result)), "hipMemset(result)")) {
        (void)hipFree(result_dev);
        (void)hipFree(dout_dev);
        (void)hipFree(q_dev);
        return 1;
    }

    hipLaunchKernelGGL(
        dual_view_page_offset_probe, dim3(1), dim3(320), 0, 0, q_dev,
        dout_dev, result_dev);
    const bool run_ok =
        hip_ok(hipGetLastError(), "dual_view_page_offset_probe launch") &&
        hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") &&
        hip_ok(
            hipMemcpy(
                &result, result_dev, sizeof(result), hipMemcpyDeviceToHost),
            "hipMemcpy(result)");

    const bool free_ok =
        hip_ok(hipFree(result_dev), "hipFree(result)") &&
        hip_ok(hipFree(dout_dev), "hipFree(dout)") &&
        hip_ok(hipFree(q_dev), "hipFree(q)");
    const bool pass = run_ok && result.trans_mismatch == 0 &&
                      result.normal_mismatch == 0 &&
                      result.dual_base_mismatch == 0 &&
                      result.sidecar_mismatch == 0;
    std::printf(
        "dkv_mls_dual_view_page_offset trans_mismatch=%u "
        "normal_mismatch=%u dual_base_mismatch=%u sidecar_mismatch=%u "
        "lds_bytes=%d pass=%d\n",
        result.trans_mismatch, result.normal_mismatch,
        result.dual_base_mismatch, result.sidecar_mismatch,
        kPlannedLdsBytes, pass ? 1 : 0);
    return pass && free_ok ? 0 : 1;
}
