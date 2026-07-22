#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_fa3_components.h"

#include <cstdint>

namespace {

constexpr float kLog2E = 1.44269504088896340736f;
constexpr int kPackedSidecarFields = 3;

__global__ void dot_do_o_kernel(const __half* dout,
                                const __half* out,
                                const float* scores_max,
                                const float* scores_sum,
                                float* delta,
                                float* packed_sidecar,
                                int rows,
                                int dim,
                                float scale_log2) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int64_t base = static_cast<int64_t>(row) * dim;
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dot += __half2float(dout[base + d]) * __half2float(out[base + d]);
    }

    delta[row] = dot;
    const int packed = row * kPackedSidecarFields;
    packed_sidecar[packed + 0] = scores_max[row] * scale_log2;
    packed_sidecar[packed + 1] =
        scores_sum[row] != 0.0f ? 1.0f / scores_sum[row] : 0.0f;
    packed_sidecar[packed + 2] = dot;
}

bool valid_params(const ShaoboFa3Params* params) {
    return params != nullptr && params->batch > 0 &&
           params->seqlen_q > 0 && params->num_heads_q > 0 &&
           params->head_dim_qk > 0 &&
           params->head_dim_qk == params->head_dim_v &&
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
    constexpr int kThreads = 128;
    const int blocks = (rows + kThreads - 1) / kThreads;
    dot_do_o_kernel<<<blocks, kThreads>>>(
        static_cast<const __half*>(dout), static_cast<const __half*>(out),
        static_cast<const float*>(scores_max),
        static_cast<const float*>(scores_sum), static_cast<float*>(delta),
        static_cast<float*>(packed_sidecar), rows, params->head_dim_qk,
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
