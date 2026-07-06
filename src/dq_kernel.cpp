#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_contract.h"
#include "shaobo_fa3_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace dq = shaobo::fa3::bwd::dq;

namespace {

constexpr int kDefaultBatch = 1;
constexpr int kDefaultHeads = 1;
constexpr int kDefaultSeq = 128;
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

inline int64_t tensor_offset(
    int b, int h, int s, int d, int heads, int seqlen, int dim) {
    return ((static_cast<int64_t>(b) * heads + h) * seqlen + s) * dim + d;
}

inline bool valid_dq_shape(const ShaoboFa3Params* p) {
    return p != nullptr && p->batch > 0 && p->seqlen_q > 0 &&
           p->seqlen_k > 0 && p->num_heads_q > 0 &&
           p->head_dim_qk == dq::ActiveDqTile::kHeadDim &&
           p->head_dim_v == dq::ActiveDqTile::kHeadDim &&
           p->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           p->layout == SHAOBO_FA3_LAYOUT_BHSD;
}

inline bool valid_reference_shape(const ShaoboFa3Params* p) {
    return valid_dq_shape(p) &&
           p->dq_path == dq::kDqPathReferenceCorrectness &&
           p->seqlen_k == p->seqlen_q &&
           p->num_heads_kv == p->num_heads_q &&
           p->dropout_p == 0.0f;
}

inline bool valid_canonical_shape(const ShaoboFa3Params* p) {
    return valid_dq_shape(p) &&
           p->dq_path == dq::kDqPathCanonicalDq &&
           p->seqlen_q % dq::ActiveDqTile::kBlockMq == 0 &&
           p->seqlen_k == p->seqlen_q &&
           p->seqlen_k % dq::ActiveDqTile::kBlockNk == 0 &&
           p->num_heads_kv == p->num_heads_q &&
           p->causal == 1;
}

inline size_t reference_workspace_bytes(const ShaoboFa3Params* p) {
    if (!valid_reference_shape(p)) {
        return 0;
    }
    const size_t rows = static_cast<size_t>(p->batch) *
                        static_cast<size_t>(p->num_heads_q) *
                        static_cast<size_t>(p->seqlen_q);
    const size_t pairs = rows * static_cast<size_t>(p->seqlen_k);
    return (2 * pairs + 3 * rows) * sizeof(float);
}

__global__ void fa3_bwd_dq_ref_softmax_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    float* __restrict__ prob,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    int batch,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float scale) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = batch * heads * seqlen;
    if (row >= rows) {
        return;
    }

    const int q_idx = row % seqlen;
    const int h = (row / seqlen) % heads;
    const int b = row / (seqlen * heads);
    const int64_t head_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;

    float max_score = -3.4028234663852886e38f;
    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
        if (causal && k_idx > q_idx) {
            continue;
        }
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += __half2float(q[head_base + q_idx * dim + d]) *
                   __half2float(k[head_base + k_idx * dim + d]);
        }
        max_score = fmaxf(max_score, dot * scale);
    }

    float denom = 0.0f;
    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
        const int64_t pair = static_cast<int64_t>(row) * seqlen + k_idx;
        if (causal && k_idx > q_idx) {
            prob[pair] = 0.0f;
            continue;
        }
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += __half2float(q[head_base + q_idx * dim + d]) *
                   __half2float(k[head_base + k_idx * dim + d]);
        }
        const float p = expf(dot * scale - max_score);
        prob[pair] = p;
        denom += p;
    }

    const float inv_denom = denom != 0.0f ? 1.0f / denom : 0.0f;
    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
        const int64_t pair = static_cast<int64_t>(row) * seqlen + k_idx;
        prob[pair] *= inv_denom;
    }
    row_max[row] = max_score;
    row_sum[row] = denom;
}

__global__ void fa3_bwd_dq_ref_delta_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ v,
    const float* __restrict__ prob,
    float* __restrict__ delta,
    int batch,
    int heads,
    int seqlen,
    int dim) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = batch * heads * seqlen;
    if (row >= rows) {
        return;
    }

    const int q_idx = row % seqlen;
    const int h = (row / seqlen) % heads;
    const int b = row / (seqlen * heads);
    const int64_t head_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;

    float delta_acc = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float out_d = 0.0f;
        for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
            out_d += prob[static_cast<int64_t>(row) * seqlen + k_idx] *
                     __half2float(v[head_base + k_idx * dim + d]);
        }
        delta_acc +=
            __half2float(dout[head_base + q_idx * dim + d]) * out_d;
    }
    delta[row] = delta_acc;
}

__global__ void fa3_bwd_dq_ref_dp_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ v,
    float* __restrict__ dp,
    int batch,
    int heads,
    int seqlen,
    int dim,
    int causal) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const int rows = batch * heads * seqlen;
    const int64_t pairs = static_cast<int64_t>(rows) * seqlen;
    if (pair >= pairs) {
        return;
    }

    const int row = static_cast<int>(pair / seqlen);
    const int k_idx = static_cast<int>(pair % seqlen);
    const int q_idx = row % seqlen;
    if (causal && k_idx > q_idx) {
        dp[pair] = 0.0f;
        return;
    }

    const int h = (row / seqlen) % heads;
    const int b = row / (seqlen * heads);
    const int64_t head_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;

    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dot += __half2float(dout[head_base + q_idx * dim + d]) *
               __half2float(v[head_base + k_idx * dim + d]);
    }
    dp[pair] = dot;
}

__global__ void fa3_bwd_dq_ref_output_kernel(
    const __half* __restrict__ k,
    const float* __restrict__ prob,
    const float* __restrict__ dp,
    const float* __restrict__ delta,
    float* __restrict__ dq_out,
    int batch,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float scale) {
    const int64_t elem = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    const int64_t elems =
        static_cast<int64_t>(batch) * heads * seqlen * dim;
    if (elem >= elems) {
        return;
    }

    const int d = static_cast<int>(elem % dim);
    const int q_idx = static_cast<int>((elem / dim) % seqlen);
    const int h = static_cast<int>((elem / dim / seqlen) % heads);
    const int b = static_cast<int>(elem / dim / seqlen / heads);
    const int row = (b * heads + h) * seqlen + q_idx;
    const int64_t head_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;

    float accum = 0.0f;
    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
        if (causal && k_idx > q_idx) {
            continue;
        }
        const int64_t pair = static_cast<int64_t>(row) * seqlen + k_idx;
        const float ds =
            prob[pair] * (dp[pair] - delta[row]) * scale;
        accum += ds * __half2float(k[head_base + k_idx * dim + d]);
    }
    dq_out[elem] = accum;
}

struct DqCompareMetrics {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float rmse = 0.0f;
    float rel_l2 = 0.0f;
    int bad_count = 0;
};

inline float deterministic_value(int64_t index, int mul, int mod, float scale) {
    const int value = static_cast<int>((index * mul + 7) % mod) - mod / 2;
    return static_cast<float>(value) * scale;
}

void fill_dq_inputs(std::vector<__half>& q,
                    std::vector<__half>& k,
                    std::vector<__half>& v,
                    std::vector<__half>& dout) {
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = __float2half(deterministic_value(i, 3, 29, 0.009f));
        k[i] = __float2half(deterministic_value(i, 5, 31, 0.008f));
        v[i] = __float2half(deterministic_value(i, 7, 37, 0.007f));
        dout[i] = __float2half(deterministic_value(i, 11, 41, 0.006f));
    }
}

void cpu_reference_dq(const std::vector<__half>& q,
                      const std::vector<__half>& k,
                      const std::vector<__half>& v,
                      const std::vector<__half>& dout,
                      std::vector<float>& dq_out,
                      int batch,
                      int heads,
                      int seqlen,
                      int dim,
                      int causal,
                      float scale) {
    const int rows = batch * heads * seqlen;
    std::vector<float> prob(static_cast<size_t>(rows) * seqlen, 0.0f);
    std::vector<float> delta(rows, 0.0f);
    std::vector<float> dp(static_cast<size_t>(rows) * seqlen, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
                const int row = (b * heads + h) * seqlen + q_idx;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                    if (causal && k_idx > q_idx) {
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        dot += __half2float(
                                   q[tensor_offset(
                                       b, h, q_idx, d, heads, seqlen, dim)]) *
                               __half2float(
                                   k[tensor_offset(
                                       b, h, k_idx, d, heads, seqlen, dim)]);
                    }
                    max_score = std::max(max_score, dot * scale);
                }

                float denom = 0.0f;
                for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                    if (causal && k_idx > q_idx) {
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        dot += __half2float(
                                   q[tensor_offset(
                                       b, h, q_idx, d, heads, seqlen, dim)]) *
                               __half2float(
                                   k[tensor_offset(
                                       b, h, k_idx, d, heads, seqlen, dim)]);
                    }
                    const float p = std::exp(dot * scale - max_score);
                    prob[static_cast<size_t>(row) * seqlen + k_idx] = p;
                    denom += p;
                }
                const float inv_denom = denom != 0.0f ? 1.0f / denom : 0.0f;
                for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                    prob[static_cast<size_t>(row) * seqlen + k_idx] *=
                        inv_denom;
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
                const int row = (b * heads + h) * seqlen + q_idx;
                float accum = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float out_d = 0.0f;
                    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                        out_d += prob[static_cast<size_t>(row) * seqlen +
                                      k_idx] *
                                 __half2float(
                                     v[tensor_offset(
                                         b, h, k_idx, d, heads, seqlen, dim)]);
                    }
                    accum += __half2float(
                                 dout[tensor_offset(
                                     b, h, q_idx, d, heads, seqlen, dim)]) *
                             out_d;
                }
                delta[row] = accum;
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
                const int row = (b * heads + h) * seqlen + q_idx;
                for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                    if (causal && k_idx > q_idx) {
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        dot += __half2float(
                                   dout[tensor_offset(
                                       b, h, q_idx, d, heads, seqlen, dim)]) *
                               __half2float(
                                   v[tensor_offset(
                                       b, h, k_idx, d, heads, seqlen, dim)]);
                    }
                    dp[static_cast<size_t>(row) * seqlen + k_idx] = dot;
                }
            }
        }
    }

    std::fill(dq_out.begin(), dq_out.end(), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
                const int row = (b * heads + h) * seqlen + q_idx;
                for (int d = 0; d < dim; ++d) {
                    float dq_accum = 0.0f;
                    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                        if (causal && k_idx > q_idx) {
                            continue;
                        }
                        const size_t pair =
                            static_cast<size_t>(row) * seqlen + k_idx;
                        const float ds =
                            prob[pair] * (dp[pair] - delta[row]) * scale;
                        dq_accum += ds * __half2float(
                                              k[tensor_offset(
                                                  b, h, k_idx, d, heads,
                                                  seqlen, dim)]);
                    }
                    dq_out[tensor_offset(
                        b, h, q_idx, d, heads, seqlen, dim)] = dq_accum;
                }
            }
        }
    }
}

DqCompareMetrics compare_vectors(const std::vector<float>& actual,
                                 const std::vector<float>& expected) {
    DqCompareMetrics m;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_ref_sq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++m.bad_count;
            continue;
        }
        const float diff = std::fabs(a - e);
        m.max_abs = std::max(m.max_abs, diff);
        sum_abs += diff;
        sum_sq += static_cast<double>(diff) * diff;
        sum_ref_sq += static_cast<double>(e) * e;
    }
    const double n = static_cast<double>(actual.size());
    m.mean_abs = static_cast<float>(sum_abs / std::max(1.0, n));
    m.rmse = static_cast<float>(std::sqrt(sum_sq / std::max(1.0, n)));
    m.rel_l2 = static_cast<float>(
        std::sqrt(sum_sq / std::max(1.0e-30, sum_ref_sq)));
    return m;
}

inline void ignore_hip_status(hipError_t err) {
    (void)err;
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
    if (params == nullptr) {
        return 0;
    }
    if (params->dq_path == dq::kDqPathReferenceCorrectness) {
        return reference_workspace_bytes(params);
    }
    return 0;
}

extern "C" int shaobo_fa3_bwd(const void* dout,
                              const void* q,
                              const void* k,
                              const void* v,
                              const void* out,
                              const void* softmax_aux0,
                              const void* softmax_aux1,
                              void* dq_out,
                              void* dk,
                              void* dv,
                              const ShaoboFa3Params* params) {
    (void)out;
    (void)softmax_aux0;
    (void)softmax_aux1;
    (void)dk;
    (void)dv;

    if (params == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->dq_path == dq::kDqPathCanonicalDq) {
        if (!valid_canonical_shape(params)) {
            return SHAOBO_FA3_STATUS_UNSUPPORTED;
        }
        return SHAOBO_FA3_STATUS_NOT_IMPLEMENTED;
    }
    if (params->dq_path != dq::kDqPathReferenceCorrectness ||
        !valid_reference_shape(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    if (dout == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        dq_out == nullptr || params->workspace == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    const size_t required_workspace = reference_workspace_bytes(params);
    if (params->workspace_bytes < required_workspace) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    float* workspace = static_cast<float*>(params->workspace);
    const int rows =
        params->batch * params->num_heads_q * params->seqlen_q;
    const int64_t pairs = static_cast<int64_t>(rows) * params->seqlen_k;
    float* prob = workspace;
    float* dp = prob + pairs;
    float* row_max = dp + pairs;
    float* row_sum = row_max + rows;
    float* delta = row_sum + rows;

    const int threads = 128;
    const int row_blocks = ceil_div(rows, threads);
    const int pair_blocks =
        static_cast<int>(ceil_div(static_cast<int>(pairs), threads));
    const int64_t elems =
        static_cast<int64_t>(rows) * params->head_dim_qk;
    const int elem_blocks =
        static_cast<int>(ceil_div(static_cast<int>(elems), threads));

    fa3_bwd_dq_ref_softmax_kernel<<<row_blocks, threads>>>(
        static_cast<const __half*>(q), static_cast<const __half*>(k),
        prob, row_max, row_sum, params->batch, params->num_heads_q,
        params->seqlen_q, params->head_dim_qk, params->causal,
        params->softmax_scale);
    fa3_bwd_dq_ref_delta_kernel<<<row_blocks, threads>>>(
        static_cast<const __half*>(dout), static_cast<const __half*>(v),
        prob, delta, params->batch, params->num_heads_q, params->seqlen_q,
        params->head_dim_qk);
    fa3_bwd_dq_ref_dp_kernel<<<pair_blocks, threads>>>(
        static_cast<const __half*>(dout), static_cast<const __half*>(v),
        dp, params->batch, params->num_heads_q, params->seqlen_q,
        params->head_dim_qk, params->causal);
    fa3_bwd_dq_ref_output_kernel<<<elem_blocks, threads>>>(
        static_cast<const __half*>(k), prob, dp, delta,
        static_cast<float*>(dq_out), params->batch, params->num_heads_q,
        params->seqlen_q, params->head_dim_qk, params->causal,
        params->softmax_scale);

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
    const int causal = arg_int(argc, argv, "--causal", env_int("CAUSAL", 1));

    ShaoboFa3Params params{};
    params.struct_size = sizeof(params);
    params.batch = batch;
    params.seqlen_q = seqlen;
    params.seqlen_k = seqlen;
    params.num_heads_q = heads;
    params.num_heads_kv = heads;
    params.head_dim_qk = dim;
    params.head_dim_v = dim;
    params.causal = causal != 0 ? 1 : 0;
    params.softmax_scale = 0.08838834764831845f;
    params.dtype = SHAOBO_FA3_DTYPE_FP16;
    params.layout = SHAOBO_FA3_LAYOUT_BHSD;
    params.softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_SMAX_SSUM;
    params.dq_path = dq::kDqPathReferenceCorrectness;
    params.block_threads = dq::ActiveDqTile::kThreadsPerCta;
    params.sync_after_launch = 1;

    const size_t tensor_elems =
        static_cast<size_t>(batch) * static_cast<size_t>(heads) *
        static_cast<size_t>(seqlen) * static_cast<size_t>(dim);
    const size_t tensor_bytes = tensor_elems * sizeof(__half);
    const size_t output_bytes = tensor_elems * sizeof(float);
    const size_t workspace_bytes = shaobo_fa3_bwd_workspace_bytes(&params);

    __half* q_dev = nullptr;
    __half* k_dev = nullptr;
    __half* v_dev = nullptr;
    __half* dout_dev = nullptr;
    float* dq_dev = nullptr;
    float* workspace = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void**>(&q_dev), tensor_bytes);
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&k_dev), tensor_bytes);
    }
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&v_dev), tensor_bytes);
    }
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dout_dev), tensor_bytes);
    }
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dq_dev), output_bytes);
    }
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&workspace),
                        std::max<size_t>(workspace_bytes, sizeof(float)));
    }
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        return 2;
    }

    std::vector<__half> q_host(tensor_elems);
    std::vector<__half> k_host(tensor_elems);
    std::vector<__half> v_host(tensor_elems);
    std::vector<__half> dout_host(tensor_elems);
    fill_dq_inputs(q_host, k_host, v_host, dout_host);
    ignore_hip_status(
        hipMemcpy(q_dev, q_host.data(), tensor_bytes, hipMemcpyHostToDevice));
    ignore_hip_status(
        hipMemcpy(k_dev, k_host.data(), tensor_bytes, hipMemcpyHostToDevice));
    ignore_hip_status(
        hipMemcpy(v_dev, v_host.data(), tensor_bytes, hipMemcpyHostToDevice));
    ignore_hip_status(hipMemcpy(
        dout_dev, dout_host.data(), tensor_bytes, hipMemcpyHostToDevice));
    ignore_hip_status(hipMemset(dq_dev, 0, output_bytes));
    ignore_hip_status(
        hipMemset(workspace, 0, std::max<size_t>(workspace_bytes, sizeof(float))));
    params.workspace = workspace;
    params.workspace_bytes = workspace_bytes;

    const int status = shaobo_fa3_bwd(
        dout_dev, q_dev, k_dev, v_dev, nullptr, nullptr, nullptr, dq_dev,
        nullptr, nullptr, &params);

    std::vector<float> dq_actual(tensor_elems);
    std::vector<float> dq_expected(tensor_elems);
    if (status == SHAOBO_FA3_STATUS_SUCCESS) {
        ignore_hip_status(hipMemcpy(
            dq_actual.data(), dq_dev, output_bytes, hipMemcpyDeviceToHost));
        cpu_reference_dq(
            q_host, k_host, v_host, dout_host, dq_expected, batch, heads,
            seqlen, dim, params.causal, params.softmax_scale);
    }
    const DqCompareMetrics dq_metrics =
        compare_vectors(dq_actual, dq_expected);
    const bool pass = status == SHAOBO_FA3_STATUS_SUCCESS &&
                      dq_metrics.bad_count == 0 &&
                      dq_metrics.max_abs <= 5.0e-4f &&
                      dq_metrics.rel_l2 <= 1.0e-4f;

    ignore_hip_status(hipFree(workspace));
    ignore_hip_status(hipFree(dq_dev));
    ignore_hip_status(hipFree(dout_dev));
    ignore_hip_status(hipFree(v_dev));
    ignore_hip_status(hipFree(k_dev));
    ignore_hip_status(hipFree(q_dev));

    std::printf(
        "%s status=%s B=%d H=%d S=%d D=%d causal=%d workspace_bytes=%zu "
        "dq_max_abs=%g dq_mean_abs=%g dq_rmse=%g dq_rel_l2=%g bad=%d pass=%d\n",
        "fa3_bwd_dq_correctness", shaobo_fa3_status_string(status), batch,
        heads, seqlen, dim, params.causal, workspace_bytes,
        dq_metrics.max_abs, dq_metrics.mean_abs, dq_metrics.rmse,
        dq_metrics.rel_l2, dq_metrics.bad_count, pass ? 1 : 0);
    return pass ? 0 : 1;
}
#endif
