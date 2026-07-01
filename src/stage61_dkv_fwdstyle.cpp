#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_fa3_api.h"
#include "stage61_dkv_contract.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace s61 = shaobo::fa3::bwd::stage61;

namespace {

constexpr int kDefaultBatch = 1;
constexpr int kDefaultHeads = 1;
constexpr int kDefaultSeq = 1024;
constexpr int kDefaultDim = 128;

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

inline int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::atoi(value) : fallback;
}

inline int arg_int(int argc, char** argv, const char* name, int fallback) {
    const std::string prefix = std::string(name) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return std::atoi(arg.c_str() + prefix.size());
        }
    }
    return fallback;
}

inline bool valid_stage61_shape(const ShaoboFa3Params* p) {
    return p != nullptr && p->batch > 0 && p->seqlen_q > 0 &&
           p->seqlen_k > 0 && p->num_heads_q > 0 &&
           p->head_dim_qk == s61::DkvTileD128Mq32Nk128::kHeadDim &&
           p->head_dim_v == s61::DkvTileD128Mq32Nk128::kHeadDim &&
           p->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           p->layout == SHAOBO_FA3_LAYOUT_BHSD;
}

__global__ void __launch_bounds__(s61::DkvTileD128Mq32Nk128::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(16)))
bwd_dkv_stage61_fwdstyle_scaffold_kernel(float* __restrict__ diag,
                                         int diag_stride,
                                         int seqlen) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = s61::DkvTileD128Mq32Nk128;
    using Bar = s61::DkvBarrierLedger;
    using Vgpr = s61::WdraResourceWindows;

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    (void)diag;
    (void)diag_stride;
    (void)seqlen;

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed, 8);
        __builtin_hcu_s_abarrier_init(Bar::kTransFilled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kTransUsed, 8);
        __builtin_hcu_s_abarrier_init(Bar::kKv0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kKv0Used, 4);
        __builtin_hcu_s_abarrier_init(Bar::kKv1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kKv1Used, 4);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
        __builtin_hcu_s_abarrier_init(Bar::kValuExec0, 4);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        __builtin_hcu_s_abarrier_seq(Bar::kKv0Filled);
        __builtin_hcu_s_abarrier_arrive(Bar::kKv0Filled);
        __builtin_hcu_s_abarrier_seq(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_arrive(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_seq(Bar::kTransFilled);
        __builtin_hcu_s_abarrier_arrive(Bar::kTransFilled);
        __builtin_hcu_s_abarrier_try_wait(Bar::kRawUsed, 0);
        __builtin_hcu_s_abarrier_try_wait(Bar::kTransUsed, 0);
        __builtin_hcu_s_abarrier_arrive(Bar::kAllDone);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        __builtin_hcu_s_abarrier_try_wait(Bar::kKv0Filled, 0);
        __builtin_hcu_s_abarrier_try_wait(Bar::kRawFilled, 0);
        __builtin_hcu_s_abarrier_try_wait(Bar::kTransFilled, 0);
        __builtin_hcu_s_abarrier_arrive(Bar::kKv0Used);
        __builtin_hcu_s_abarrier_arrive(Bar::kRawUsed);
        __builtin_hcu_s_abarrier_arrive(Bar::kTransUsed);
        __builtin_hcu_s_abarrier_arrive(Bar::kAllDone);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        __builtin_hcu_s_abarrier_try_wait(Bar::kKv1Filled, 0);
        __builtin_hcu_s_abarrier_try_wait(Bar::kRawFilled, 0);
        __builtin_hcu_s_abarrier_try_wait(Bar::kTransFilled, 0);
        __builtin_hcu_s_abarrier_arrive(Bar::kKv1Used);
        __builtin_hcu_s_abarrier_arrive(Bar::kRawUsed);
        __builtin_hcu_s_abarrier_arrive(Bar::kTransUsed);
        __builtin_hcu_s_abarrier_arrive(Bar::kAllDone);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        __builtin_hcu_s_abarrier_seq(Bar::kKv1Filled);
        __builtin_hcu_s_abarrier_arrive(Bar::kKv1Filled);
        __builtin_hcu_s_abarrier_seq(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_arrive(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_seq(Bar::kTransFilled);
        __builtin_hcu_s_abarrier_arrive(Bar::kTransFilled);
        __builtin_hcu_s_abarrier_try_wait(Bar::kRawUsed, 0);
        __builtin_hcu_s_abarrier_try_wait(Bar::kTransUsed, 0);
        __builtin_hcu_s_abarrier_arrive(Bar::kAllDone);
    }

    __builtin_hcu_s_abarrier_try_wait(Bar::kAllDone, 0);
#else
    (void)diag;
    (void)diag_stride;
    (void)seqlen;
#endif
}

}  // namespace

extern "C" const char* shaobo_fa3_status_string(int status) {
    switch (status) {
        case SHAOBO_FA3_STATUS_SUCCESS:
            return "success";
        case SHAOBO_FA3_STATUS_INVALID_VALUE:
            return "invalid_value";
        case SHAOBO_FA3_STATUS_UNSUPPORTED:
            return "unsupported";
        case SHAOBO_FA3_STATUS_NOT_IMPLEMENTED:
            return "not_implemented";
        case SHAOBO_FA3_STATUS_HIP_ERROR:
            return "hip_error";
        default:
            return "unknown";
    }
}

extern "C" size_t shaobo_fa3_bwd_workspace_bytes(
    const ShaoboFa3Params* params) {
    if (!valid_stage61_shape(params)) {
        return 0;
    }
    const int grid_x =
        ceil_div(params->seqlen_k, s61::DkvTileD128Mq32Nk128::kResidentNk);
    const size_t blocks = static_cast<size_t>(grid_x) *
                          static_cast<size_t>(params->num_heads_q) *
                          static_cast<size_t>(params->batch);
    return blocks * sizeof(float);
}

extern "C" int shaobo_fa3_bwd(const void* dout,
                              const void* q,
                              const void* k,
                              const void* v,
                              const void* out,
                              const void* softmax_aux0,
                              const void* softmax_aux1,
                              void* dq,
                              void* dk,
                              void* dv,
                              const ShaoboFa3Params* params) {
    (void)dout;
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    (void)softmax_aux0;
    (void)softmax_aux1;
    (void)dq;
    (void)dk;
    (void)dv;

    if (!valid_stage61_shape(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    if (params->workspace != nullptr &&
        params->workspace_bytes < shaobo_fa3_bwd_workspace_bytes(params)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int grid_x =
        ceil_div(params->seqlen_k, s61::DkvTileD128Mq32Nk128::kResidentNk);
    const int blocks = grid_x * params->num_heads_q * params->batch;
    dim3 grid(grid_x, params->num_heads_q, params->batch);
    dim3 block(s61::DkvTileD128Mq32Nk128::kThreadsPerCta);

    hipLaunchKernelGGL(bwd_dkv_stage61_fwdstyle_scaffold_kernel, grid, block, 0,
                       0, static_cast<float*>(params->workspace), blocks,
                       params->seqlen_k);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return SHAOBO_FA3_STATUS_HIP_ERROR;
    }
    if (params->sync_after_launch) {
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            return SHAOBO_FA3_STATUS_HIP_ERROR;
        }
    }
    return SHAOBO_FA3_STATUS_SUCCESS;
}

#ifndef SHAOBO_FA3_NO_STANDALONE
int main(int argc, char** argv) {
    const int batch = arg_int(argc, argv, "--B", env_int("B", kDefaultBatch));
    const int heads = arg_int(argc, argv, "--H", env_int("H", kDefaultHeads));
    const int seqlen = arg_int(argc, argv, "--S", env_int("S", kDefaultSeq));
    const int dim = arg_int(argc, argv, "--D", env_int("D", kDefaultDim));

    ShaoboFa3Params params{};
    params.struct_size = sizeof(params);
    params.batch = batch;
    params.seqlen_q = seqlen;
    params.seqlen_k = seqlen;
    params.num_heads_q = heads;
    params.num_heads_kv = heads;
    params.head_dim_qk = dim;
    params.head_dim_v = dim;
    params.causal = 1;
    params.softmax_scale = 0.08838834764831845f;
    params.dtype = SHAOBO_FA3_DTYPE_FP16;
    params.layout = SHAOBO_FA3_LAYOUT_BHSD;
    params.softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_SMAX_SSUM;
    params.block_threads = s61::DkvTileD128Mq32Nk128::kThreadsPerCta;
    params.sync_after_launch = 1;

    const size_t workspace_bytes = shaobo_fa3_bwd_workspace_bytes(&params);
    float* workspace = nullptr;
    hipError_t err = hipMalloc(&workspace, std::max<size_t>(workspace_bytes, 4));
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMalloc workspace failed: %s\n",
                     hipGetErrorString(err));
        return 2;
    }
    params.workspace = workspace;
    params.workspace_bytes = workspace_bytes;
    err = hipMemset(workspace, 0, std::max<size_t>(workspace_bytes, 4));
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMemset workspace failed: %s\n",
                     hipGetErrorString(err));
        static_cast<void>(hipFree(workspace));
        return 2;
    }

    const int status = shaobo_fa3_bwd(nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, &params);
    float first_diag = 0.0f;
    if (workspace_bytes >= sizeof(float)) {
        err = hipMemcpy(&first_diag, workspace, sizeof(float),
                        hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            std::fprintf(stderr, "hipMemcpy workspace failed: %s\n",
                         hipGetErrorString(err));
            static_cast<void>(hipFree(workspace));
            return 2;
        }
    }
    err = hipFree(workspace);
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipFree workspace failed: %s\n",
                     hipGetErrorString(err));
        return 2;
    }

    std::printf(
        "stage61_fwdstyle_scaffold status=%s B=%d H=%d S=%d D=%d "
        "workspace_bytes=%zu first_diag=%g\n",
        shaobo_fa3_status_string(status), batch, heads, seqlen, dim,
        workspace_bytes, first_diag);
    return status == SHAOBO_FA3_STATUS_SUCCESS ? 0 : 1;
}
#endif
