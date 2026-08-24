#pragma once

#include <hip/hip_fp16.h>

#include "shaobo_fa3_api.h"

#include <cstddef>

namespace shaobo::fa3::bwd::fused_bwd {

struct FusedWorkspaceView {
    __half* dq_partial;
    __half* dk_partial;
    __half* dv_partial;
};

size_t dq_workspace_bytes(const ShaoboFa3Params* params);

FusedWorkspaceView workspace_view(void* workspace,
                                  const ShaoboFa3Params* params);

int launch_dq_reduction(const __half* partial,
                        __half* dq,
                        const ShaoboFa3Params* params);

int launch_dkv_reduction(const __half* dk_partial,
                         const __half* dv_partial,
                         __half* dk,
                         __half* dv,
                         const ShaoboFa3Params* params);

}  // namespace shaobo::fa3::bwd::fused_bwd
