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
constexpr int kVectorsPerRow = Tile::kHeadDim / 4;
constexpr int kRowsPerBlock = kThreads / kVectorsPerRow;
static_assert(kThreads % kVectorsPerRow == 0 &&
                  Tile::kNk % kRowsPerBlock == 0,
              "dQ reduction blocks must cover aligned causal row groups");

__global__ void fa3_bwd_dq_reduce_kernel(const __half* __restrict__ partial,
                                         __half* __restrict__ dq,
                                         int seqlen,
                                         int k_tiles,
                                         int causal) {
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock +
                    static_cast<int>(threadIdx.x) / kVectorsPerRow;
    const int d_vec = static_cast<int>(threadIdx.x) % kVectorsPerRow;
    const int bh = static_cast<int>(blockIdx.y);
    const int64_t vectors_per_tensor =
        static_cast<int64_t>(seqlen) * kVectorsPerRow;
    const int64_t local_vec =
        static_cast<int64_t>(row) * kVectorsPerRow + d_vec;
    const int64_t slice_vectors = vectors_per_tensor;
    const int last_k_tile =
        causal ? static_cast<int>(blockIdx.x) /
                     (Tile::kNk / kRowsPerBlock)
               : k_tiles - 1;
    const auto* partial4 = reinterpret_cast<const ins::Vec4F16*>(partial);
    ins::Vec4F32 sum = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma clang loop unroll(disable)
    for (int k_tile = 0; k_tile + 1 <= last_k_tile; k_tile += 2) {
        const int64_t offset =
            (bh * k_tiles + k_tile) * slice_vectors + local_vec;
        const ins::Vec4F16 value0 = partial4[offset];
        const ins::Vec4F16 value1 = partial4[offset + slice_vectors];
        sum += __builtin_convertvector(value0, ins::Vec4F32);
        sum += __builtin_convertvector(value1, ins::Vec4F32);
    }
    if ((last_k_tile & 1) == 0) {
        const int64_t offset =
            (bh * k_tiles + last_k_tile) * slice_vectors + local_vec;
        sum += __builtin_convertvector(partial4[offset], ins::Vec4F32);
    }
    const ins::Vec4F16 out =
        __builtin_convertvector(sum, ins::Vec4F16);
    const int64_t output_vec =
        static_cast<int64_t>(bh) * vectors_per_tensor + local_vec;
    reinterpret_cast<ins::Vec4F16*>(dq)[output_vec] = out;
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
    if (elements > std::numeric_limits<size_t>::max() / sizeof(__half)) {
        return 0;
    }
    return elements * sizeof(__half);
}

int launch_dq_reduction(const __half* partial,
                        __half* dq,
                        const ShaoboFa3Params* params) {
    if (!valid_params(params) || partial == nullptr || dq == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    const int k_tiles = params->seqlen_k / Tile::kNk;
    const int64_t bh_count =
        static_cast<int64_t>(params->batch) * params->num_heads_q;
    const int64_t row_blocks = params->seqlen_q / kRowsPerBlock;
    if (row_blocks <= 0 ||
        row_blocks > std::numeric_limits<uint32_t>::max() ||
        bh_count <= 0 || bh_count > std::numeric_limits<uint32_t>::max()) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    hipLaunchKernelGGL(fa3_bwd_dq_reduce_kernel,
                       dim3(static_cast<uint32_t>(row_blocks),
                            static_cast<uint32_t>(bh_count)),
                       dim3(kThreads), 0, 0, partial, dq,
                       params->seqlen_q, k_tiles, params->causal);
    return hipGetLastError() == hipSuccess ? SHAOBO_FA3_STATUS_SUCCESS
                                           : SHAOBO_FA3_STATUS_HIP_ERROR;
}

}  // namespace shaobo::fa3::bwd::fused_bwd

extern "C" size_t shaobo_fa3_bwd_fused5_workspace_bytes(
    const ShaoboFa3Params* params) {
    return fused::dq_workspace_bytes(params);
}
