#include <hip/hip_runtime.h>

#include "fused_bwd_contract.h"
#include "fused_bwd_dq_reduce.h"
#include "shaobo_fa3_components.h"
#include "shaobo_instr.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace fused = shaobo::fa3::bwd::fused_bwd;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Tile = fused::ActiveFusedBwdContract;
constexpr int kThreads = 256;
constexpr int kVectorsPerRow = Tile::kHeadDim / 4;
constexpr int kRowsPerBlock = kThreads / kVectorsPerRow;
constexpr size_t kWorkspaceAlignment = 256;
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

__global__ void fa3_bwd_dkv_reduce_kernel(
    const __half* __restrict__ dk_partial,
    const __half* __restrict__ dv_partial,
    __half* __restrict__ dk,
    __half* __restrict__ dv,
    int seqlen,
    int heads_q,
    int heads_kv) {
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock +
                    static_cast<int>(threadIdx.x) / kVectorsPerRow;
    const int d_vec = static_cast<int>(threadIdx.x) % kVectorsPerRow;
    const int kv_bh = static_cast<int>(blockIdx.y);
    const int batch = kv_bh / heads_kv;
    const int kv_head = kv_bh - batch * heads_kv;
    const int q_heads_per_kv = heads_q / heads_kv;
    const int q_head_begin = kv_head * q_heads_per_kv;
    const int64_t vectors_per_tensor =
        static_cast<int64_t>(seqlen) * kVectorsPerRow;
    const int64_t local_vec =
        static_cast<int64_t>(row) * kVectorsPerRow + d_vec;
    ins::Vec4F32 dk_sum = {0.0f, 0.0f, 0.0f, 0.0f};
    ins::Vec4F32 dv_sum = {0.0f, 0.0f, 0.0f, 0.0f};
    const auto* dk4 = reinterpret_cast<const ins::Vec4F16*>(dk_partial);
    const auto* dv4 = reinterpret_cast<const ins::Vec4F16*>(dv_partial);

#pragma clang loop unroll(disable)
    for (int q_offset = 0; q_offset < q_heads_per_kv; ++q_offset) {
        const int q_bh = batch * heads_q + q_head_begin + q_offset;
        const int64_t partial_vec =
            static_cast<int64_t>(q_bh) * vectors_per_tensor + local_vec;
        dk_sum += __builtin_convertvector(dk4[partial_vec], ins::Vec4F32);
        dv_sum += __builtin_convertvector(dv4[partial_vec], ins::Vec4F32);
    }

    const int64_t output_vec =
        static_cast<int64_t>(kv_bh) * vectors_per_tensor + local_vec;
    reinterpret_cast<ins::Vec4F16*>(dk)[output_vec] =
        __builtin_convertvector(dk_sum, ins::Vec4F16);
    reinterpret_cast<ins::Vec4F16*>(dv)[output_vec] =
        __builtin_convertvector(dv_sum, ins::Vec4F16);
}

bool valid_params(const ShaoboFa3Params* params) {
    return params != nullptr && params->batch > 0 &&
           params->num_heads_q > 0 && params->num_heads_kv > 0 &&
           params->num_heads_q % params->num_heads_kv == 0 &&
           params->seqlen_q > 0 &&
           params->seqlen_q == params->seqlen_k &&
           params->seqlen_k % Tile::kNk == 0 &&
           params->head_dim_qk == Tile::kHeadDim &&
           params->head_dim_v == Tile::kHeadDim;
}

bool checked_mul(size_t lhs, size_t rhs, size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t& result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool align_workspace(size_t bytes, size_t& aligned) {
    size_t padded = 0;
    if (!checked_add(bytes, kWorkspaceAlignment - 1, padded)) {
        return false;
    }
    aligned = padded & ~(kWorkspaceAlignment - 1);
    return true;
}

struct WorkspaceSizes {
    size_t dq_bytes = 0;
    size_t dkv_bytes = 0;
    size_t total_bytes = 0;
    bool valid = false;
};

WorkspaceSizes workspace_sizes(const ShaoboFa3Params* params) {
    WorkspaceSizes sizes;
    if (!valid_params(params)) {
        return sizes;
    }

    size_t dq_elements = static_cast<size_t>(params->batch);
    const size_t k_tiles = static_cast<size_t>(params->seqlen_k / Tile::kNk);
    for (size_t factor : {static_cast<size_t>(params->num_heads_q), k_tiles,
                          static_cast<size_t>(params->seqlen_q),
                          static_cast<size_t>(Tile::kHeadDim)}) {
        if (!checked_mul(dq_elements, factor, dq_elements)) {
            return sizes;
        }
    }
    size_t dq_raw_bytes = 0;
    if (!checked_mul(dq_elements, sizeof(__half), dq_raw_bytes) ||
        !align_workspace(dq_raw_bytes, sizes.dq_bytes)) {
        return sizes;
    }

    if (params->num_heads_q != params->num_heads_kv) {
        size_t dkv_elements = static_cast<size_t>(params->batch);
        for (size_t factor : {static_cast<size_t>(params->num_heads_q),
                              static_cast<size_t>(params->seqlen_k),
                              static_cast<size_t>(Tile::kHeadDim)}) {
            if (!checked_mul(dkv_elements, factor, dkv_elements)) {
                return sizes;
            }
        }
        if (!checked_mul(dkv_elements, sizeof(__half), sizes.dkv_bytes)) {
            return sizes;
        }
    }

    size_t two_dkv_bytes = 0;
    if (!checked_mul(sizes.dkv_bytes, 2, two_dkv_bytes) ||
        !checked_add(sizes.dq_bytes, two_dkv_bytes, sizes.total_bytes)) {
        return sizes;
    }
    sizes.valid = true;
    return sizes;
}

}  // namespace

namespace shaobo::fa3::bwd::fused_bwd {

size_t dq_workspace_bytes(const ShaoboFa3Params* params) {
    const WorkspaceSizes sizes = workspace_sizes(params);
    return sizes.valid ? sizes.total_bytes : 0;
}

FusedWorkspaceView workspace_view(void* workspace,
                                  const ShaoboFa3Params* params) {
    const WorkspaceSizes sizes = workspace_sizes(params);
    if (workspace == nullptr || !sizes.valid) {
        return FusedWorkspaceView{nullptr, nullptr, nullptr};
    }
    auto* bytes = static_cast<std::byte*>(workspace);
    __half* dk_partial = nullptr;
    __half* dv_partial = nullptr;
    if (sizes.dkv_bytes != 0) {
        dk_partial = reinterpret_cast<__half*>(bytes + sizes.dq_bytes);
        dv_partial = reinterpret_cast<__half*>(
            bytes + sizes.dq_bytes + sizes.dkv_bytes);
    }
    return FusedWorkspaceView{
        reinterpret_cast<__half*>(bytes), dk_partial, dv_partial};
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

int launch_dkv_reduction(const __half* dk_partial,
                         const __half* dv_partial,
                         __half* dk,
                         __half* dv,
                         const ShaoboFa3Params* params) {
    if (!valid_params(params) || dk == nullptr || dv == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->num_heads_q == params->num_heads_kv) {
        return SHAOBO_FA3_STATUS_SUCCESS;
    }
    if (dk_partial == nullptr || dv_partial == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    const int64_t bh_count =
        static_cast<int64_t>(params->batch) * params->num_heads_kv;
    const int64_t row_blocks = params->seqlen_k / kRowsPerBlock;
    if (row_blocks <= 0 ||
        row_blocks > std::numeric_limits<uint32_t>::max() ||
        bh_count <= 0 || bh_count > std::numeric_limits<uint32_t>::max()) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    hipLaunchKernelGGL(fa3_bwd_dkv_reduce_kernel,
                       dim3(static_cast<uint32_t>(row_blocks),
                            static_cast<uint32_t>(bh_count)),
                       dim3(kThreads), 0, 0, dk_partial, dv_partial, dk, dv,
                       params->seqlen_k, params->num_heads_q,
                       params->num_heads_kv);
    return hipGetLastError() == hipSuccess ? SHAOBO_FA3_STATUS_SUCCESS
                                           : SHAOBO_FA3_STATUS_HIP_ERROR;
}

}  // namespace shaobo::fa3::bwd::fused_bwd

extern "C" size_t shaobo_fa3_bwd_fused5_workspace_bytes(
    const ShaoboFa3Params* params) {
    return fused::dq_workspace_bytes(params);
}
