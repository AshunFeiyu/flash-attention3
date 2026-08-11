#ifndef SHAOBO_FA3_COMPONENTS_H_
#define SHAOBO_FA3_COMPONENTS_H_

#include "shaobo_fa3_api.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t shaobo_fa3_bwd_dkv_workspace_bytes(const ShaoboFa3Params* params);

int shaobo_fa3_bwd_dkv(const void* dout,
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

size_t shaobo_fa3_bwd_dq_workspace_bytes(const ShaoboFa3Params* params);

int shaobo_fa3_bwd_dq(const void* dout,
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

int shaobo_fa3_bwd_dot_do_o(const void* dout,
                            const void* out,
                            const void* scores_max,
                            const void* scores_sum,
                            void* delta,
                            void* packed_sidecar,
                            const ShaoboFa3Params* params);

size_t shaobo_fa3_bwd_fused5_workspace_bytes(
    const ShaoboFa3Params* params);

int shaobo_fa3_bwd_fused5(const void* dout,
                          const void* q,
                          const void* k,
                          const void* v,
                          const void* packed_sidecar,
                          void* dq,
                          void* dk,
                          void* dv,
                          const ShaoboFa3Params* params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SHAOBO_FA3_COMPONENTS_H_
