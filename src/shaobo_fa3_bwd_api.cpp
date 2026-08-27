#include "shaobo_fa3_api.h"

#include <hip/hip_runtime.h>

#include "shaobo_fa3_components.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace {

constexpr size_t kWorkspaceAlignment = 256;
constexpr int kSupportedHeadDim = 128;
constexpr int kSequenceAlignment = 128;

struct PublicWorkspaceLayout {
    size_t fused_bytes = 0;
    size_t delta_offset = 0;
    size_t sidecar_offset = 0;
    size_t total_bytes = 0;
    bool valid = false;
};

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

bool align_up(size_t bytes, size_t& aligned) {
    size_t padded = 0;
    if (!checked_add(bytes, kWorkspaceAlignment - 1, padded)) {
        return false;
    }
    aligned = padded & ~(kWorkspaceAlignment - 1);
    return true;
}

bool full_attention_window(const ShaoboFa3Params* params) {
    int left = params->window_left;
    int right = params->window_right;
    if (left >= params->seqlen_k - 1) {
        left = -1;
    }
    if (right >= params->seqlen_q - 1) {
        right = -1;
    }
    if (params->causal != 0) {
        right = 0;
    }
    const bool normalized_causal = left < 0 && right == 0;
    const bool normalized_full = left < 0 && right < 0;
    return params->causal != 0 ? normalized_causal : normalized_full;
}

int validate_params(const ShaoboFa3Params* params) {
    if (params == nullptr || params->struct_size < sizeof(ShaoboFa3Params)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->batch <= 0 || params->seqlen_q <= 0 ||
        params->seqlen_k <= 0 || params->num_heads_q <= 0 ||
        params->num_heads_kv <= 0 ||
        params->num_heads_q % params->num_heads_kv != 0 ||
        (params->causal != 0 && params->causal != 1) ||
        (params->deterministic != 0 && params->deterministic != 1) ||
        (params->sync_after_launch != 0 && params->sync_after_launch != 1) ||
        (params->softmax_scale_is_set != 0 &&
         params->softmax_scale_is_set != 1) ||
        (params->softmax_scale_is_set != 0 &&
         (!std::isfinite(params->softmax_scale) ||
          params->softmax_scale < 0.0f)) ||
        !std::isfinite(params->softcap) || params->softcap < 0.0f ||
        !std::isfinite(params->dropout_p) || params->dropout_p < 0.0f ||
        params->dropout_p >= 1.0f) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int64_t fixed_total_q =
        static_cast<int64_t>(params->batch) * params->seqlen_q;
    const int64_t fixed_total_k =
        static_cast<int64_t>(params->batch) * params->seqlen_k;
    const bool fixed_lengths = params->cu_seqlens_q == nullptr &&
                               params->cu_seqlens_k == nullptr &&
                               params->seqused_q == nullptr &&
                               params->seqused_k == nullptr &&
                               (params->max_seqlen_q == 0 ||
                                params->max_seqlen_q == params->seqlen_q) &&
                               (params->max_seqlen_k == 0 ||
                                params->max_seqlen_k == params->seqlen_k) &&
                               (params->total_q == 0 ||
                                params->total_q == fixed_total_q) &&
                               (params->total_k == 0 ||
                                params->total_k == fixed_total_k);

    if (!fixed_lengths || params->dtype != SHAOBO_FA3_DTYPE_FP16 ||
        params->layout != SHAOBO_FA3_LAYOUT_BHSD ||
        params->head_dim_qk != kSupportedHeadDim ||
        params->head_dim_v != kSupportedHeadDim ||
        params->seqlen_q != params->seqlen_k ||
        params->seqlen_q % kSequenceAlignment != 0 ||
        params->dropout_p != 0.0f || params->softcap != 0.0f ||
        params->sm_margin != 0 || !full_attention_window(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    return SHAOBO_FA3_STATUS_SUCCESS;
}

PublicWorkspaceLayout workspace_layout(const ShaoboFa3Params* params) {
    PublicWorkspaceLayout layout;
    if (validate_params(params) != SHAOBO_FA3_STATUS_SUCCESS) {
        return layout;
    }

    ShaoboFa3Params fused_params = *params;
    fused_params.softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_LSE;
    layout.fused_bytes = shaobo_fa3_bwd_fused5_workspace_bytes(&fused_params);
    if (layout.fused_bytes == 0 ||
        !align_up(layout.fused_bytes, layout.delta_offset)) {
        return layout;
    }

    size_t rows = static_cast<size_t>(params->batch);
    for (size_t factor : {static_cast<size_t>(params->num_heads_q),
                          static_cast<size_t>(params->seqlen_q)}) {
        if (!checked_mul(rows, factor, rows)) {
            return layout;
        }
    }
    size_t delta_bytes = 0;
    size_t sidecar_bytes = 0;
    size_t after_delta = 0;
    if (!checked_mul(rows, sizeof(float), delta_bytes) ||
        !checked_mul(delta_bytes, 3, sidecar_bytes) ||
        !checked_add(layout.delta_offset, delta_bytes, after_delta) ||
        !align_up(after_delta, layout.sidecar_offset) ||
        !checked_add(layout.sidecar_offset, sidecar_bytes,
                     layout.total_bytes)) {
        return layout;
    }
    layout.valid = true;
    return layout;
}

}  // namespace

extern "C" void shaobo_fa3_params_init(ShaoboFa3Params* params) {
    if (params == nullptr) {
        return;
    }
    std::memset(params, 0, sizeof(*params));
    params->struct_size = sizeof(*params);
    params->window_left = -1;
    params->window_right = -1;
    params->dtype = SHAOBO_FA3_DTYPE_FP16;
    params->layout = SHAOBO_FA3_LAYOUT_BHSD;
    params->softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_LSE;
}

extern "C" int shaobo_fa3_bwd_get_capabilities(
    ShaoboFa3BwdCapabilities* capabilities) {
    if (capabilities == nullptr ||
        capabilities->struct_size < sizeof(ShaoboFa3BwdCapabilities)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    capabilities->api_version = SHAOBO_FA3_API_VERSION;
    capabilities->feature_mask =
        SHAOBO_FA3_BWD_FEATURE_FIXED_LENGTH |
        SHAOBO_FA3_BWD_FEATURE_FP16 | SHAOBO_FA3_BWD_FEATURE_CAUSAL |
        SHAOBO_FA3_BWD_FEATURE_GQA |
        SHAOBO_FA3_BWD_FEATURE_DETERMINISTIC |
        SHAOBO_FA3_BWD_FEATURE_SOFTMAX_LSE |
        SHAOBO_FA3_BWD_FEATURE_BHSD |
        SHAOBO_FA3_BWD_FEATURE_FULL_ATTENTION;
    capabilities->min_head_dim = kSupportedHeadDim;
    capabilities->max_head_dim = kSupportedHeadDim;
    capabilities->head_dim_alignment = kSupportedHeadDim;
    capabilities->sequence_alignment = kSequenceAlignment;
    return SHAOBO_FA3_STATUS_SUCCESS;
}

extern "C" int shaobo_fa3_bwd_v2_validate(
    const ShaoboFa3Params* params) {
    return validate_params(params);
}

extern "C" size_t shaobo_fa3_bwd_v2_workspace_bytes(
    const ShaoboFa3Params* params) {
    const PublicWorkspaceLayout layout = workspace_layout(params);
    return layout.valid ? layout.total_bytes : 0;
}

extern "C" int shaobo_fa3_bwd_v2(const void* dout,
                                   const void* q,
                                   const void* k,
                                   const void* v,
                                   const void* out,
                                   const void* softmax_lse,
                                   void* dq,
                                   void* dk,
                                   void* dv,
                                   void* softmax_d,
                                   const ShaoboFa3Params* params) {
    const int validation = validate_params(params);
    if (validation != SHAOBO_FA3_STATUS_SUCCESS) {
        return validation;
    }
    if (dout == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        out == nullptr || softmax_lse == nullptr || dq == nullptr ||
        dk == nullptr || dv == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const PublicWorkspaceLayout layout = workspace_layout(params);
    if (!layout.valid || params->workspace == nullptr ||
        params->workspace_bytes < layout.total_bytes) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    auto* workspace = static_cast<std::byte*>(params->workspace);
    void* const delta = softmax_d != nullptr
                            ? softmax_d
                            : workspace + layout.delta_offset;
    void* const packed_sidecar = workspace + layout.sidecar_offset;

    ShaoboFa3Params launch = *params;
    launch.softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_LSE;
    if (launch.softmax_scale_is_set == 0) {
        launch.softmax_scale = 1.0f / std::sqrt(float(launch.head_dim_qk));
    }
    launch.workspace = workspace;
    launch.workspace_bytes = layout.fused_bytes;
    launch.sync_after_launch = 0;

    int status = shaobo_fa3_bwd_dot_do_o(
        dout, out, softmax_lse, nullptr, delta, packed_sidecar, &launch);
    if (status == SHAOBO_FA3_STATUS_SUCCESS) {
        status = shaobo_fa3_bwd_fused5(dout, q, k, v, packed_sidecar, dq, dk,
                                       dv, &launch);
    }
    if (status != SHAOBO_FA3_STATUS_SUCCESS || params->sync_after_launch == 0) {
        return status;
    }
    return hipDeviceSynchronize() == hipSuccess
               ? SHAOBO_FA3_STATUS_SUCCESS
               : SHAOBO_FA3_STATUS_HIP_ERROR;
}
