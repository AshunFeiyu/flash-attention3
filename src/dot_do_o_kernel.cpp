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

__device__ __forceinline__ float wave_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, kWaveSize);
    }
    return value;
}

__global__ void dot_do_o_kernel(const __half* __restrict__ dout,
                                const __half* __restrict__ out,
                                const float* __restrict__ scores_max,
                                const float* __restrict__ scores_sum,
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
    float dot = __half2float(dout[base + lane]) *
                __half2float(out[base + lane]);
    dot += __half2float(dout[base + lane + kWaveSize]) *
           __half2float(out[base + lane + kWaveSize]);
    dot = wave_reduce_sum(dot);

    if (lane == 0) {
        delta[row] = dot;
        const int packed = row * kPackedSidecarFields;
        packed_sidecar[packed + 0] = scores_max[row] * scale_log2;
        packed_sidecar[packed + 1] =
            scores_sum[row] != 0.0f ? 1.0f / scores_sum[row] : 0.0f;
        packed_sidecar[packed + 2] = dot;
    }
}

bool valid_params(const ShaoboFa3Params* params) {
    return params != nullptr && params->batch > 0 &&
           params->seqlen_q > 0 && params->num_heads_q > 0 &&
           params->head_dim_qk == kHeadDim &&
           params->head_dim_v == kHeadDim &&
           params->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           params->layout == SHAOBO_FA3_LAYOUT_BHSD;
}

}  // namespace

extern "C" int shaobo_fa3_bwd_dot_do_o(const void* dout,
                                        const void* out,
                                        const void* scores_max,
                                        const void* scores_sum,
                                        void* delta,
                                        void* packed_sidecar,
                                        const ShaoboFa3Params* params) {
    if (!valid_params(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    if (dout == nullptr || out == nullptr || scores_max == nullptr ||
        scores_sum == nullptr || delta == nullptr ||
        packed_sidecar == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int rows =
        params->batch * params->num_heads_q * params->seqlen_q;
    const int blocks = (rows + kWavesPerBlock - 1) / kWavesPerBlock;
    dot_do_o_kernel<<<blocks, kThreads>>>(
        static_cast<const __half*>(dout), static_cast<const __half*>(out),
        static_cast<const float*>(scores_max),
        static_cast<const float*>(scores_sum), static_cast<float*>(delta),
        static_cast<float*>(packed_sidecar), rows,
        params->softmax_scale * kLog2E);

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
