#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_fa3_components.h"

#include <cstdint>

namespace {

constexpr float kLog2E = 1.44269504088896340736f;
constexpr int kPackedSidecarFields = 3;
constexpr int kHeadDim = 128;
constexpr int kWaveSize = 64;
constexpr int kWavesPerBlock = 4;
constexpr int kThreads = kWaveSize * kWavesPerBlock;
using Vec2F16 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;

__device__ __forceinline__ float wave_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, kWaveSize);
    }
    return value;
}

template <bool LseInput>
__global__ void dot_do_o_kernel(const __half* __restrict__ dout,
                                const __half* __restrict__ out,
                                const float* __restrict__ softmax_aux0,
                                const float* __restrict__ softmax_aux1,
                                float* __restrict__ delta,
                                float* __restrict__ packed_sidecar,
                                int rows,
                                float scale_log2) {
    const int lane = static_cast<int>(threadIdx.x & (kWaveSize - 1));
    const int wave = static_cast<int>(threadIdx.x >> 6);
    const int row = static_cast<int>(blockIdx.x * kWavesPerBlock + wave);
    if (row >= rows) {
        return;
    }

    const int64_t base = static_cast<int64_t>(row) * kHeadDim;
    const auto* dout_pairs = reinterpret_cast<const Vec2F16*>(dout + base);
    const auto* out_pairs = reinterpret_cast<const Vec2F16*>(out + base);
    float dot = __builtin_amdgcn_fdot2(dout_pairs[lane], out_pairs[lane],
                                       0.0f, false);
    dot = wave_reduce_sum(dot);

    if (lane == 0) {
        delta[row] = dot;
        const int packed = row * kPackedSidecarFields;
        if constexpr (LseInput) {
            packed_sidecar[packed + 0] = softmax_aux0[row] * kLog2E;
            packed_sidecar[packed + 1] = 1.0f;
        } else {
            packed_sidecar[packed + 0] = softmax_aux0[row] * scale_log2;
            packed_sidecar[packed + 1] = softmax_aux1[row] != 0.0f
                                                  ? 1.0f / softmax_aux1[row]
                                                  : 0.0f;
        }
        packed_sidecar[packed + 2] = dot;
    }
}

bool valid_params(const ShaoboFa3Params* params) {
    return params != nullptr && params->batch > 0 &&
           params->seqlen_q > 0 && params->num_heads_q > 0 &&
           params->head_dim_qk == kHeadDim &&
           params->head_dim_v == kHeadDim &&
           params->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           params->layout == SHAOBO_FA3_LAYOUT_BHSD &&
           (params->softmax_aux_mode == SHAOBO_FA3_SOFTMAX_AUX_SMAX_SSUM ||
            params->softmax_aux_mode == SHAOBO_FA3_SOFTMAX_AUX_LSE);
}

}  // namespace

extern "C" int shaobo_fa3_bwd_dot_do_o(const void* dout,
                                        const void* out,
                                        const void* softmax_aux0,
                                        const void* softmax_aux1,
                                        void* delta,
                                        void* packed_sidecar,
                                        const ShaoboFa3Params* params) {
    if (!valid_params(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    const bool lse_input =
        params->softmax_aux_mode == SHAOBO_FA3_SOFTMAX_AUX_LSE;
    if (dout == nullptr || out == nullptr || softmax_aux0 == nullptr ||
        (!lse_input && softmax_aux1 == nullptr) || delta == nullptr ||
        packed_sidecar == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int rows =
        params->batch * params->num_heads_q * params->seqlen_q;
    const int blocks = (rows + kWavesPerBlock - 1) / kWavesPerBlock;
    if (lse_input) {
        dot_do_o_kernel<true><<<blocks, kThreads>>>(
            static_cast<const __half*>(dout), static_cast<const __half*>(out),
            static_cast<const float*>(softmax_aux0), nullptr,
            static_cast<float*>(delta), static_cast<float*>(packed_sidecar),
            rows, params->softmax_scale * kLog2E);
    } else {
        dot_do_o_kernel<false><<<blocks, kThreads>>>(
            static_cast<const __half*>(dout), static_cast<const __half*>(out),
            static_cast<const float*>(softmax_aux0),
            static_cast<const float*>(softmax_aux1),
            static_cast<float*>(delta), static_cast<float*>(packed_sidecar),
            rows, params->softmax_scale * kLog2E);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return SHAOBO_FA3_STATUS_HIP_ERROR;
    }
    if (params->sync_after_launch) {
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            return SHAOBO_FA3_STATUS_HIP_ERROR;
        }
    }
    return SHAOBO_FA3_STATUS_SUCCESS;
}
