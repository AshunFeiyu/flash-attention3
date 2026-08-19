#pragma once

#include <hip/hip_fp16.h>

#include "shaobo_fa3_api.h"

#include <cstddef>

namespace shaobo::fa3::bwd::fused_bwd {

size_t dq_workspace_bytes(const ShaoboFa3Params* params);

int launch_dq_reduction(const __half* partial,
                        __half* dq,
                        const ShaoboFa3Params* params);

}  // namespace shaobo::fa3::bwd::fused_bwd
