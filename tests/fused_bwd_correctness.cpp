#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_fa3_api.h"
#include "shaobo_fa3_components.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr float kScale = 0.08838834764831845f;
constexpr float kLog2E = 1.44269504088896340736f;

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
        std::exit(2);
    }
}

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::atoi(value);
}

void build_inputs(std::vector<__half>& q,
                  std::vector<__half>& k,
                  std::vector<__half>& v,
                  std::vector<__half>& dout,
                  int seqlen) {
    for (int row = 0; row < seqlen; ++row) {
        for (int d = 0; d < kDim; ++d) {
            const int index = row * kDim + d;
            q[index] = __float2half(
                static_cast<float>((row * 7 + d * 13 + 3) % 17 - 8) / 64.0f);
            k[index] = __float2half(
                static_cast<float>((row * 5 + d * 11 + 5) % 19 - 9) / 64.0f);
            v[index] = __float2half(
                static_cast<float>((row * 3 + d * 7 + 9) % 23 - 11) / 16.0f);
            dout[index] = __float2half(
                static_cast<float>((row * 11 + d * 5 + 1) % 29 - 14) / 16.0f);
        }
    }
}

void cpu_golden(const std::vector<__half>& q,
                const std::vector<__half>& k,
                const std::vector<__half>& v,
                const std::vector<__half>& dout,
                int seqlen,
                bool causal,
                std::vector<float>& sidecar,
                std::vector<float>& dq,
                std::vector<float>& dk,
                std::vector<float>& dv) {
    std::vector<float> p(static_cast<size_t>(seqlen) * seqlen, 0.0f);
    std::vector<float> ds(p.size(), 0.0f);
    std::vector<float> out(static_cast<size_t>(seqlen) * kDim, 0.0f);
    const float scale_log2 = kScale * kLog2E;

    for (int qi = 0; qi < seqlen; ++qi) {
        float row_max = -std::numeric_limits<float>::infinity();
        const int key_count = causal ? qi + 1 : seqlen;
        for (int ki = 0; ki < key_count; ++ki) {
            float score = 0.0f;
            for (int d = 0; d < kDim; ++d) {
                score += __half2float(q[qi * kDim + d]) *
                         __half2float(k[ki * kDim + d]);
            }
            p[static_cast<size_t>(qi) * seqlen + ki] = score;
            row_max = std::max(row_max, score);
        }
        float row_sum = 0.0f;
        for (int ki = 0; ki < key_count; ++ki) {
            float& value = p[static_cast<size_t>(qi) * seqlen + ki];
            value = std::exp2((value - row_max) * scale_log2);
            row_sum += value;
        }
        const float inv_sum = 1.0f / row_sum;
        for (int ki = 0; ki < key_count; ++ki) {
            const float probability =
                p[static_cast<size_t>(qi) * seqlen + ki] * inv_sum;
            p[static_cast<size_t>(qi) * seqlen + ki] = probability;
            for (int d = 0; d < kDim; ++d) {
                out[qi * kDim + d] +=
                    probability * __half2float(v[ki * kDim + d]);
            }
        }

        float delta = 0.0f;
        for (int d = 0; d < kDim; ++d) {
            delta += __half2float(dout[qi * kDim + d]) *
                     out[qi * kDim + d];
        }
        sidecar[qi * 3 + 0] = row_max * scale_log2;
        sidecar[qi * 3 + 1] = inv_sum;
        sidecar[qi * 3 + 2] = delta;

        for (int ki = 0; ki < key_count; ++ki) {
            float dp = 0.0f;
            for (int d = 0; d < kDim; ++d) {
                dp += __half2float(dout[qi * kDim + d]) *
                      __half2float(v[ki * kDim + d]);
            }
            ds[static_cast<size_t>(qi) * seqlen + ki] =
                p[static_cast<size_t>(qi) * seqlen + ki] * (dp - delta) *
                kScale;
        }
    }

    for (int qi = 0; qi < seqlen; ++qi) {
        const int key_count = causal ? qi + 1 : seqlen;
        for (int ki = 0; ki < key_count; ++ki) {
            const float probability = p[static_cast<size_t>(qi) * seqlen + ki];
            const float dscore = ds[static_cast<size_t>(qi) * seqlen + ki];
            for (int d = 0; d < kDim; ++d) {
                dq[qi * kDim + d] +=
                    dscore * __half2float(k[ki * kDim + d]);
                dk[ki * kDim + d] +=
                    dscore * __half2float(q[qi * kDim + d]);
                dv[ki * kDim + d] +=
                    probability * __half2float(dout[qi * kDim + d]);
            }
        }
    }
}

bool report(const char* name,
            const std::vector<float>& actual,
            const std::vector<float>& expected,
            float max_abs_limit,
            float rel_l2_limit) {
    double error_sq = 0.0;
    double expected_sq = 0.0;
    float max_abs = 0.0f;
    float max_actual = 0.0f;
    float max_expected = 0.0f;
    int nonfinite = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            ++nonfinite;
            continue;
        }
        max_actual = std::max(max_actual, std::fabs(actual[i]));
        max_expected = std::max(max_expected, std::fabs(expected[i]));
        const float error = actual[i] - expected[i];
        max_abs = std::max(max_abs, std::fabs(error));
        error_sq += static_cast<double>(error) * error;
        expected_sq += static_cast<double>(expected[i]) * expected[i];
    }
    const float rel_l2 = static_cast<float>(
        std::sqrt(error_sq / std::max(expected_sq, 1.0e-30)));
    const bool pass = nonfinite == 0 && max_abs <= max_abs_limit &&
                      rel_l2 <= rel_l2_limit;
    std::printf(
        "fused5_correctness %s actual_max=%g expected_max=%g max_abs=%g "
        "rel_l2=%g nonfinite=%d pass=%d\n",
        name, max_actual, max_expected, max_abs, rel_l2, nonfinite,
        pass ? 1 : 0);
    return pass;
}

}  // namespace

int main() {
    const int seqlen = env_int("S", 128);
    const bool causal = env_int("CAUSAL", 1) != 0;
    if (seqlen <= 0 || seqlen % 128 != 0) {
        std::fprintf(stderr, "S must be a positive multiple of 128\n");
        return 2;
    }
    const size_t elements = static_cast<size_t>(seqlen) * kDim;
    const size_t half_bytes = elements * sizeof(__half);
    const size_t float_bytes = elements * sizeof(float);
    std::vector<__half> q(elements);
    std::vector<__half> k(elements);
    std::vector<__half> v(elements);
    std::vector<__half> dout(elements);
    std::vector<float> sidecar(static_cast<size_t>(seqlen) * 3);
    std::vector<float> dq_expected(elements, 0.0f);
    std::vector<float> dk_expected(elements, 0.0f);
    std::vector<float> dv_expected(elements, 0.0f);
    build_inputs(q, k, v, dout, seqlen);
    cpu_golden(q, k, v, dout, seqlen, causal, sidecar, dq_expected, dk_expected,
               dv_expected);

    __half* d_q = nullptr;
    __half* d_k = nullptr;
    __half* d_v = nullptr;
    __half* d_dout = nullptr;
    float* d_sidecar = nullptr;
    float* d_dq = nullptr;
    float* d_dk = nullptr;
    float* d_dv = nullptr;
    void* d_workspace = nullptr;
    check_hip(hipMalloc(&d_q, half_bytes), "hipMalloc q");
    check_hip(hipMalloc(&d_k, half_bytes), "hipMalloc k");
    check_hip(hipMalloc(&d_v, half_bytes), "hipMalloc v");
    check_hip(hipMalloc(&d_dout, half_bytes), "hipMalloc dout");
    check_hip(hipMalloc(&d_sidecar, sidecar.size() * sizeof(float)),
              "hipMalloc sidecar");
    check_hip(hipMalloc(&d_dq, float_bytes), "hipMalloc dq");
    check_hip(hipMalloc(&d_dk, float_bytes), "hipMalloc dk");
    check_hip(hipMalloc(&d_dv, float_bytes), "hipMalloc dv");
    check_hip(hipMemcpy(d_q, q.data(), half_bytes, hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(d_k, k.data(), half_bytes, hipMemcpyHostToDevice),
              "copy k");
    check_hip(hipMemcpy(d_v, v.data(), half_bytes, hipMemcpyHostToDevice),
              "copy v");
    check_hip(hipMemcpy(d_dout, dout.data(), half_bytes, hipMemcpyHostToDevice),
              "copy dout");
    check_hip(hipMemcpy(d_sidecar, sidecar.data(),
                        sidecar.size() * sizeof(float), hipMemcpyHostToDevice),
              "copy sidecar");
    check_hip(hipMemset(d_dq, 0, float_bytes), "clear dq");
    check_hip(hipMemset(d_dk, 0, float_bytes), "clear dk");
    check_hip(hipMemset(d_dv, 0, float_bytes), "clear dv");

    ShaoboFa3Params params{};
    params.struct_size = sizeof(params);
    params.batch = 1;
    params.seqlen_q = seqlen;
    params.seqlen_k = seqlen;
    params.num_heads_q = 1;
    params.num_heads_kv = 1;
    params.head_dim_qk = kDim;
    params.head_dim_v = kDim;
    params.causal = causal ? 1 : 0;
    params.softmax_scale = kScale;
    params.dtype = SHAOBO_FA3_DTYPE_FP16;
    params.layout = SHAOBO_FA3_LAYOUT_BHSD;
    params.sync_after_launch = 1;
    const size_t workspace_bytes =
        shaobo_fa3_bwd_fused5_workspace_bytes(&params);
    if (workspace_bytes == 0) {
        std::fprintf(stderr, "invalid fused5 workspace size\n");
        return 2;
    }
    check_hip(hipMalloc(&d_workspace, workspace_bytes),
              "hipMalloc fused5 workspace");
    params.workspace = d_workspace;
    params.workspace_bytes = workspace_bytes;
    const int status = shaobo_fa3_bwd_fused5(
        d_dout, d_q, d_k, d_v, d_sidecar, d_dq, d_dk, d_dv, &params);
    if (status != SHAOBO_FA3_STATUS_SUCCESS) {
        std::fprintf(stderr, "shaobo_fa3_bwd_fused5 status=%d\n", status);
        return 2;
    }

    std::vector<float> dq(elements);
    std::vector<float> dk(elements);
    std::vector<float> dv(elements);
    check_hip(hipMemcpy(dq.data(), d_dq, float_bytes, hipMemcpyDeviceToHost),
              "copy dq");
    check_hip(hipMemcpy(dk.data(), d_dk, float_bytes, hipMemcpyDeviceToHost),
              "copy dk");
    check_hip(hipMemcpy(dv.data(), d_dv, float_bytes, hipMemcpyDeviceToHost),
              "copy dv");
    const bool dq_pass = report("dQ", dq, dq_expected, 3.0e-2f, 3.0e-2f);
    const bool dk_pass = report("dK", dk, dk_expected, 3.0e-2f, 3.0e-2f);
    const bool dv_pass = report("dV", dv, dv_expected, 3.0e-2f, 3.0e-2f);
    const bool pass = dq_pass && dk_pass && dv_pass;
    std::printf("fused5_correctness_final S=%d D=%d causal=%d pass=%d\n",
                seqlen, kDim, causal ? 1 : 0, pass ? 1 : 0);

    (void)hipFree(d_dv);
    (void)hipFree(d_dk);
    (void)hipFree(d_dq);
    (void)hipFree(d_workspace);
    (void)hipFree(d_sidecar);
    (void)hipFree(d_dout);
    (void)hipFree(d_v);
    (void)hipFree(d_k);
    (void)hipFree(d_q);
    return pass ? 0 : 2;
}
