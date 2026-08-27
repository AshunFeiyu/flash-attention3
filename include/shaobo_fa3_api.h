#ifndef SHAOBO_FA3_API_H_
#define SHAOBO_FA3_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHAOBO_FA3_API_VERSION 2u

typedef enum ShaoboFa3Status {
    SHAOBO_FA3_STATUS_SUCCESS = 0,
    SHAOBO_FA3_STATUS_INVALID_VALUE = 1,
    SHAOBO_FA3_STATUS_UNSUPPORTED = 2,
    SHAOBO_FA3_STATUS_NOT_IMPLEMENTED = 3,
    SHAOBO_FA3_STATUS_HIP_ERROR = 4
} ShaoboFa3Status;

typedef enum ShaoboFa3DType {
    SHAOBO_FA3_DTYPE_FP16 = 0,
    SHAOBO_FA3_DTYPE_BF16 = 1,
    SHAOBO_FA3_DTYPE_FP8 = 2
} ShaoboFa3DType;

typedef enum ShaoboFa3Layout {
    SHAOBO_FA3_LAYOUT_BHSD = 0,
    SHAOBO_FA3_LAYOUT_BSHD = 1
} ShaoboFa3Layout;

typedef enum ShaoboFa3SoftmaxAuxMode {
    SHAOBO_FA3_SOFTMAX_AUX_SMAX_SSUM = 0,
    SHAOBO_FA3_SOFTMAX_AUX_LSE = 1
} ShaoboFa3SoftmaxAuxMode;

typedef uint64_t ShaoboFa3BwdFeatureMask;

#define SHAOBO_FA3_BWD_FEATURE_FIXED_LENGTH (UINT64_C(1) << 0)
#define SHAOBO_FA3_BWD_FEATURE_VARLEN (UINT64_C(1) << 1)
#define SHAOBO_FA3_BWD_FEATURE_FP16 (UINT64_C(1) << 2)
#define SHAOBO_FA3_BWD_FEATURE_BF16 (UINT64_C(1) << 3)
#define SHAOBO_FA3_BWD_FEATURE_CAUSAL (UINT64_C(1) << 4)
#define SHAOBO_FA3_BWD_FEATURE_LOCAL_WINDOW (UINT64_C(1) << 5)
#define SHAOBO_FA3_BWD_FEATURE_GQA (UINT64_C(1) << 6)
#define SHAOBO_FA3_BWD_FEATURE_SOFTCAP (UINT64_C(1) << 7)
#define SHAOBO_FA3_BWD_FEATURE_DETERMINISTIC (UINT64_C(1) << 8)
#define SHAOBO_FA3_BWD_FEATURE_SOFTMAX_LSE (UINT64_C(1) << 9)
#define SHAOBO_FA3_BWD_FEATURE_BSHD (UINT64_C(1) << 10)
#define SHAOBO_FA3_BWD_FEATURE_BHSD (UINT64_C(1) << 11)
#define SHAOBO_FA3_BWD_FEATURE_FULL_ATTENTION (UINT64_C(1) << 12)

typedef struct ShaoboFa3Params {
    uint32_t struct_size;

    int32_t batch;
    int32_t seqlen_q;
    int32_t seqlen_k;
    int32_t num_heads_q;
    int32_t num_heads_kv;
    int32_t head_dim_qk;
    int32_t head_dim_v;

    int32_t causal;
    int32_t window_left;
    int32_t window_right;
    float softmax_scale;
    float dropout_p;

    int32_t dtype;
    int32_t layout;
    int32_t softmax_aux_mode;
    int32_t dkv_path;
    int32_t dq_path;
    int32_t block_threads;
    int32_t sync_after_launch;

    const int32_t* cu_seqlens_q;
    const int32_t* cu_seqlens_k;
    int32_t max_seqlen_q;
    int32_t max_seqlen_k;

    void* workspace;
    size_t workspace_bytes;

    int32_t reserved_i32[16];
    void* reserved_ptr[8];

    // Tri Dao FA3 backward-compatible parameter tail. These fields are
    // appended so callers can use struct_size to remain ABI-compatible.
    const int32_t* seqused_q;
    const int32_t* seqused_k;
    int64_t total_q;
    int64_t total_k;
    float softcap;
    int32_t deterministic;
    int32_t sm_margin;
    int32_t softmax_scale_is_set;
} ShaoboFa3Params;

typedef struct ShaoboFa3BwdCapabilities {
    uint32_t struct_size;
    uint32_t api_version;
    ShaoboFa3BwdFeatureMask feature_mask;
    int32_t min_head_dim;
    int32_t max_head_dim;
    int32_t head_dim_alignment;
    int32_t sequence_alignment;
} ShaoboFa3BwdCapabilities;

void shaobo_fa3_params_init(ShaoboFa3Params* params);

int shaobo_fa3_bwd_get_capabilities(ShaoboFa3BwdCapabilities* capabilities);

int shaobo_fa3_bwd_v2_validate(const ShaoboFa3Params* params);

size_t shaobo_fa3_bwd_v2_workspace_bytes(const ShaoboFa3Params* params);

// Tri Dao FA3-style backward entry. softmax_lse is the natural-log LSE from
// forward. softmax_d is optional and receives sum(dout * out) in FP32.
int shaobo_fa3_bwd_v2(const void* dout,
                      const void* q,
                      const void* k,
                      const void* v,
                      const void* out,
                      const void* softmax_lse,
                      void* dq,
                      void* dk,
                      void* dv,
                      void* softmax_d,
                      const ShaoboFa3Params* params);

const char* shaobo_fa3_status_string(int status);

size_t shaobo_fa3_bwd_workspace_bytes(const ShaoboFa3Params* params);

int shaobo_fa3_bwd(const void* dout,
                   const void* q,
                   const void* k,
                   const void* v,
                   const void* out,
                   const void* softmax_aux0,
                   const void* softmax_aux1,
                   void* dq,
                   void* dk,
                   void* dv,
                   const ShaoboFa3Params* params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SHAOBO_FA3_API_H_
