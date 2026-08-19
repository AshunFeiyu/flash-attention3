#include <hip/hip_runtime.h>

#include "fused_bwd_contract.h"
#include "fused_bwd_dq_reduce.h"
#include "shaobo_fa3_components.h"
#include "shaobo_instr.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace fused = shaobo::fa3::bwd::fused_bwd;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Tile = fused::ActiveFusedBwdContract;
constexpr int kThreads = 256;

__global__ void fa3_bwd_dq_reduce_kernel(const float* __restrict__ partial,
                                         __half* __restrict__ dq,
                                         int batch,
                                         int heads,
                                         int seqlen,
                                         int k_tiles,
                                         int causal) {
    const int64_t vec =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t vectors_per_tensor =
        static_cast<int64_t>(seqlen) * Tile::kHeadDim / 4;
    const int64_t total_vectors =
        static_cast<int64_t>(batch) * heads * vectors_per_tensor;
    if (vec >= total_vectors) {
        return;
    }

    const int64_t bh = vec / vectors_per_tensor;
    const int64_t local_vec = vec - bh * vectors_per_tensor;
    const int row = static_cast<int>(local_vec / (Tile::kHeadDim / 4));
    const int64_t slice_vectors = vectors_per_tensor;
    const int last_k_tile = causal ? row / Tile::kNk : k_tiles - 1;
    const auto* partial4 = reinterpret_cast<const float4*>(partial);
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma clang loop unroll(disable)
    for (int k_tile = 0; k_tile <= last_k_tile; ++k_tile) {
        const int64_t offset =
            (bh * k_tiles + k_tile) * slice_vectors + local_vec;
        const float4 value = partial4[offset];
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
    }
    const ins::Vec4F16 out = {
        static_cast<_Float16>(sum.x), static_cast<_Float16>(sum.y),
        static_cast<_Float16>(sum.z), static_cast<_Float16>(sum.w)};
    reinterpret_cast<ins::Vec4F16*>(dq)[vec] = out;
}

bool valid_params(const ShaoboFa3Params* params) {
    return params != nullptr && params->batch > 0 &&
           params->num_heads_q > 0 && params->seqlen_q > 0 &&
           params->seqlen_q == params->seqlen_k &&
           params->seqlen_k % Tile::kNk == 0 &&
           params->head_dim_qk == Tile::kHeadDim &&
           params->head_dim_v == Tile::kHeadDim;
}

}  // namespace

namespace shaobo::fa3::bwd::fused_bwd {

size_t dq_workspace_bytes(const ShaoboFa3Params* params) {
    if (!valid_params(params)) {
        return 0;
    }
    const size_t k_tiles = static_cast<size_t>(params->seqlen_k / Tile::kNk);
    const size_t elements = static_cast<size_t>(params->batch) *
                            params->num_heads_q * k_tiles *
                            params->seqlen_q * Tile::kHeadDim;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return 0;
    }
    return elements * sizeof(float);
}

int launch_dq_reduction(const float* partial,
                        __half* dq,
                        const ShaoboFa3Params* params) {
    if (!valid_params(params) || partial == nullptr || dq == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    const int k_tiles = params->seqlen_k / Tile::kNk;
    const int64_t vectors =
        static_cast<int64_t>(params->batch) * params->num_heads_q *
        params->seqlen_q * Tile::kHeadDim / 4;
    const int64_t blocks = (vectors + kThreads - 1) / kThreads;
    if (blocks <= 0 || blocks > std::numeric_limits<uint32_t>::max()) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    hipLaunchKernelGGL(fa3_bwd_dq_reduce_kernel,
                       dim3(static_cast<uint32_t>(blocks)),
                       dim3(kThreads), 0, 0, partial, dq,
                       params->batch, params->num_heads_q, params->seqlen_q,
                       k_tiles, params->causal);
    return hipGetLastError() == hipSuccess ? SHAOBO_FA3_STATUS_SUCCESS
                                           : SHAOBO_FA3_STATUS_HIP_ERROR;
}

}  // namespace shaobo::fa3::bwd::fused_bwd

extern "C" size_t shaobo_fa3_bwd_fused5_workspace_bytes(
    const ShaoboFa3Params* params) {
    return fused::dq_workspace_bytes(params);
}
