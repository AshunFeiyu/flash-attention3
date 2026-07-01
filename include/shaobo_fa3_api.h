#ifndef SHAOBO_FA3_API_H_
#define SHAOBO_FA3_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
} ShaoboFa3Params;

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
