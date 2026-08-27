#include "shaobo_fa3_api.h"

#include <cstdio>

namespace {

bool expect_status(const char* name, int actual, int expected) {
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "%s: expected=%d actual=%d\n", name, expected,
                 actual);
    return false;
}

ShaoboFa3Params supported_params() {
    ShaoboFa3Params params;
    shaobo_fa3_params_init(&params);
    params.batch = 1;
    params.seqlen_q = 1024;
    params.seqlen_k = 1024;
    params.num_heads_q = 4;
    params.num_heads_kv = 2;
    params.head_dim_qk = 128;
    params.head_dim_v = 128;
    params.causal = 1;
    params.softmax_scale = 0.125f;
    params.softmax_scale_is_set = 1;
    return params;
}

}  // namespace

int main() {
    bool pass = true;
    ShaoboFa3BwdCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    pass &= expect_status("capabilities",
                          shaobo_fa3_bwd_get_capabilities(&capabilities),
                          SHAOBO_FA3_STATUS_SUCCESS);
    const ShaoboFa3BwdFeatureMask required =
        SHAOBO_FA3_BWD_FEATURE_FIXED_LENGTH |
        SHAOBO_FA3_BWD_FEATURE_FP16 | SHAOBO_FA3_BWD_FEATURE_CAUSAL |
        SHAOBO_FA3_BWD_FEATURE_GQA |
        SHAOBO_FA3_BWD_FEATURE_DETERMINISTIC |
        SHAOBO_FA3_BWD_FEATURE_SOFTMAX_LSE |
        SHAOBO_FA3_BWD_FEATURE_BHSD |
        SHAOBO_FA3_BWD_FEATURE_FULL_ATTENTION;
    if (capabilities.api_version != SHAOBO_FA3_API_VERSION ||
        (capabilities.feature_mask & required) != required ||
        capabilities.max_head_dim != 128 ||
        capabilities.sequence_alignment != 128) {
        std::fprintf(stderr, "capability payload mismatch\n");
        pass = false;
    }

    ShaoboFa3Params params = supported_params();
    pass &= expect_status("fixed_gqa_causal", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_SUCCESS);
    if (shaobo_fa3_bwd_v2_workspace_bytes(&params) == 0) {
        std::fprintf(stderr, "supported workspace must be nonzero\n");
        pass = false;
    }

    params.deterministic = 1;
    pass &= expect_status("deterministic", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_SUCCESS);
    params = supported_params();
    params.causal = 0;
    pass &= expect_status("full_attention", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_SUCCESS);
    params = supported_params();
    params.softmax_scale_is_set = 0;
    params.softmax_scale = 0.0f;
    pass &= expect_status("default_scale", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_SUCCESS);

    params = supported_params();
    params.window_left = 128;
    pass &= expect_status("local_window", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    params.softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_LSE;
    params.softcap = 1.0f;
    pass &= expect_status("softcap", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    params.dropout_p = 0.1f;
    pass &= expect_status("dropout", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    params.dtype = SHAOBO_FA3_DTYPE_BF16;
    pass &= expect_status("bf16", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    params.layout = SHAOBO_FA3_LAYOUT_BSHD;
    pass &= expect_status("bshd", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    int32_t cu_seqlens[2] = {0, 1024};
    params.cu_seqlens_q = cu_seqlens;
    pass &= expect_status("varlen", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    params.sm_margin = 1;
    pass &= expect_status("sm_margin", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_UNSUPPORTED);
    params = supported_params();
    params.softmax_scale = -1.0f;
    pass &= expect_status("negative_scale", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_INVALID_VALUE);
    params = supported_params();
    params.struct_size = 0;
    pass &= expect_status("old_struct", shaobo_fa3_bwd_v2_validate(&params),
                          SHAOBO_FA3_STATUS_INVALID_VALUE);

    std::printf("fused_bwd_api_contract pass=%d\n", pass ? 1 : 0);
    return pass ? 0 : 1;
}
