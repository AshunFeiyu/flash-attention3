#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_fa3_components.h"

#if !defined(SHAOBO_FULL_BWD_FUSED5)
#include "dkv_contract.h"
#include "dq_contract.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if !defined(SHAOBO_FULL_BWD_FUSED5)
namespace dkv = shaobo::fa3::bwd::dkv;
namespace dq = shaobo::fa3::bwd::dq;
#endif

namespace {

struct Metrics {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float rmse = 0.0f;
    float rel_l2 = 0.0f;
    float cosine_error = 0.0f;
    int nonfinite = 0;
};

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::atoi(value) : fallback;
}

float env_float(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::strtof(value, nullptr) : fallback;
}

int arg_int(int argc, char** argv, const char* name, int fallback) {
    const std::string prefix = std::string(name) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return std::atoi(arg.c_str() + prefix.size());
        }
    }
    return fallback;
}

float arg_float(int argc, char** argv, const char* name, float fallback) {
    const std::string prefix = std::string(name) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return std::strtof(arg.c_str() + prefix.size(), nullptr);
        }
    }
    return fallback;
}

std::string arg_string(int argc,
                       char** argv,
                       const char* name,
                       const char* fallback) {
    const std::string prefix = std::string(name) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return fallback != nullptr ? fallback : "";
}

template <typename T>
bool read_raw(const std::string& path, size_t count, std::vector<T>& out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "cannot open golden file: %s\n", path.c_str());
        return false;
    }
    const std::streamsize expected =
        static_cast<std::streamsize>(count * sizeof(T));
    if (file.tellg() != expected) {
        std::fprintf(stderr,
                     "golden size mismatch: %s expected=%lld actual=%lld\n",
                     path.c_str(), static_cast<long long>(expected),
                     static_cast<long long>(file.tellg()));
        return false;
    }
    out.resize(count);
    file.seekg(0, std::ios::beg);
    return static_cast<bool>(
        file.read(reinterpret_cast<char*>(out.data()), expected));
}

Metrics compare(const std::vector<float>& actual,
                const std::vector<float>& expected) {
    Metrics m;
    double abs_sum = 0.0;
    double diff_sq = 0.0;
    double actual_sq = 0.0;
    double expected_sq = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++m.nonfinite;
            continue;
        }
        const float d = std::fabs(a - e);
        m.max_abs = std::max(m.max_abs, d);
        abs_sum += d;
        diff_sq += static_cast<double>(d) * d;
        actual_sq += static_cast<double>(a) * a;
        expected_sq += static_cast<double>(e) * e;
        dot += static_cast<double>(a) * e;
    }
    const double n = std::max(1.0, static_cast<double>(actual.size()));
    m.mean_abs = static_cast<float>(abs_sum / n);
    m.rmse = static_cast<float>(std::sqrt(diff_sq / n));
    m.rel_l2 = static_cast<float>(
        std::sqrt(diff_sq / std::max(1.0e-30, expected_sq)));
    const double norm_product = std::sqrt(actual_sq * expected_sq);
    m.cosine_error = norm_product > 1.0e-30
                         ? static_cast<float>(1.0 - dot / norm_product)
                         : 0.0f;
    return m;
}

const char* status_string(int status) {
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

void free_device(std::vector<void*>& allocations) {
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        if (*it != nullptr) {
            (void)hipFree(*it);
        }
    }
}

bool hip_ok(hipError_t err, const char* operation) {
    if (err == hipSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 hipGetErrorString(err));
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int batch = arg_int(argc, argv, "--B", env_int("B", 1));
    const int heads = arg_int(argc, argv, "--H", env_int("H", 1));
    const int seqlen = arg_int(argc, argv, "--S", env_int("S", 128));
    const int dim = arg_int(argc, argv, "--D", env_int("D", 128));
    const int causal = arg_int(argc, argv, "--causal", env_int("CAUSAL", 1));
    const float softmax_scale = arg_float(
        argc, argv, "--softmax-scale",
        env_float("SOFTMAX_SCALE", 0.08838834764831845f));
    const char* env_golden = std::getenv("GOLDEN_DIR");
    const std::string golden_dir = arg_string(
        argc, argv, "--golden-dir", env_golden != nullptr ? env_golden : "");
    if (golden_dir.empty()) {
        std::fprintf(stderr, "--golden-dir is required\n");
        return 2;
    }

    const size_t rows = static_cast<size_t>(batch) * heads * seqlen;
    const size_t elems = rows * dim;
    std::vector<__half> q, k, v, out, dout;
    std::vector<float> scores_max, scores_sum;
    std::vector<float> delta_expected, dq_expected, dk_expected, dv_expected;
    const auto path = [&](const char* name) {
        return golden_dir + "/" + name;
    };
    const bool loaded =
        read_raw(path("q.f16"), elems, q) &&
        read_raw(path("k.f16"), elems, k) &&
        read_raw(path("v.f16"), elems, v) &&
        read_raw(path("o.f16"), elems, out) &&
        read_raw(path("dout.f16"), elems, dout) &&
        read_raw(path("scores_max.f32"), rows, scores_max) &&
        read_raw(path("scores_sum.f32"), rows, scores_sum) &&
        read_raw(path("delta.f32"), rows, delta_expected) &&
        read_raw(path("dq.f32"), elems, dq_expected) &&
        read_raw(path("dk.f32"), elems, dk_expected) &&
        read_raw(path("dv.f32"), elems, dv_expected);
    if (!loaded) {
        return 2;
    }

    __half *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr;
    __half *out_dev = nullptr, *dout_dev = nullptr;
    float *scores_max_dev = nullptr, *scores_sum_dev = nullptr;
    float *delta_dev = nullptr, *packed_sidecar_dev = nullptr;
    float* dq_dev = nullptr;
#if defined(SHAOBO_FULL_BWD_FUSED5)
    __half *dk_dev = nullptr, *dv_dev = nullptr;
#else
    float *dk_dev = nullptr, *dv_dev = nullptr;
#endif
    void* fused5_workspace_dev = nullptr;
    std::vector<void*> allocations;
    auto allocate = [&](auto** ptr, size_t bytes) {
        hipError_t err = hipMalloc(reinterpret_cast<void**>(ptr), bytes);
        if (err == hipSuccess) {
            allocations.push_back(*ptr);
        }
        return hip_ok(err, "hipMalloc");
    };

    const size_t half_bytes = elems * sizeof(__half);
    const size_t output_bytes = elems * sizeof(float);
#if defined(SHAOBO_FULL_BWD_FUSED5)
    const size_t dkv_output_bytes = half_bytes;
#else
    const size_t dkv_output_bytes = output_bytes;
#endif
    const size_t row_bytes = rows * sizeof(float);
    const size_t packed_bytes = rows * 3 * sizeof(float);
    if (!allocate(&q_dev, half_bytes) || !allocate(&k_dev, half_bytes) ||
        !allocate(&v_dev, half_bytes) || !allocate(&out_dev, half_bytes) ||
        !allocate(&dout_dev, half_bytes) ||
        !allocate(&scores_max_dev, row_bytes) ||
        !allocate(&scores_sum_dev, row_bytes) ||
        !allocate(&delta_dev, row_bytes) ||
        !allocate(&packed_sidecar_dev, packed_bytes) ||
        !allocate(&dq_dev, output_bytes) ||
        !allocate(&dk_dev, dkv_output_bytes) ||
        !allocate(&dv_dev, dkv_output_bytes)) {
        free_device(allocations);
        return 2;
    }

    bool copied =
        hip_ok(hipMemcpy(q_dev, q.data(), half_bytes, hipMemcpyHostToDevice),
               "copy q") &&
        hip_ok(hipMemcpy(k_dev, k.data(), half_bytes, hipMemcpyHostToDevice),
               "copy k") &&
        hip_ok(hipMemcpy(v_dev, v.data(), half_bytes, hipMemcpyHostToDevice),
               "copy v") &&
        hip_ok(hipMemcpy(out_dev, out.data(), half_bytes,
                         hipMemcpyHostToDevice),
               "copy o") &&
        hip_ok(hipMemcpy(dout_dev, dout.data(), half_bytes,
                         hipMemcpyHostToDevice),
               "copy dout") &&
        hip_ok(hipMemcpy(scores_max_dev, scores_max.data(), row_bytes,
                         hipMemcpyHostToDevice),
               "copy scores_max") &&
        hip_ok(hipMemcpy(scores_sum_dev, scores_sum.data(), row_bytes,
                         hipMemcpyHostToDevice),
               "copy scores_sum") &&
        hip_ok(hipMemset(delta_dev, 0, row_bytes), "clear delta") &&
        hip_ok(hipMemset(packed_sidecar_dev, 0, packed_bytes),
               "clear packed sidecar") &&
        hip_ok(hipMemset(dq_dev, 0, output_bytes), "clear dq") &&
        hip_ok(hipMemset(dk_dev, 0, dkv_output_bytes), "clear dk") &&
        hip_ok(hipMemset(dv_dev, 0, dkv_output_bytes), "clear dv");
    if (!copied) {
        free_device(allocations);
        return 2;
    }

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
    params.softmax_scale = softmax_scale;
    params.dtype = SHAOBO_FA3_DTYPE_FP16;
    params.layout = SHAOBO_FA3_LAYOUT_BHSD;
    params.softmax_aux_mode = SHAOBO_FA3_SOFTMAX_AUX_SMAX_SSUM;
#if defined(SHAOBO_FULL_BWD_FUSED5)
    params.dkv_path = 0;
    params.dq_path = 0;
#else
    params.dkv_path = dkv::kDkvPathCanonicalDkv;
    params.dq_path = dq::kDqPathCanonicalDq;
#endif
    params.sync_after_launch = 0;

#if defined(SHAOBO_FULL_BWD_FUSED5)
    const size_t fused5_workspace_bytes =
        shaobo_fa3_bwd_fused5_workspace_bytes(&params);
    if (fused5_workspace_bytes == 0 ||
        !allocate(&fused5_workspace_dev, fused5_workspace_bytes)) {
        free_device(allocations);
        return 2;
    }
    params.workspace = fused5_workspace_dev;
    params.workspace_bytes = fused5_workspace_bytes;
#endif

    const int dot_status = shaobo_fa3_bwd_dot_do_o(
        dout_dev, out_dev, scores_max_dev, scores_sum_dev, delta_dev,
        packed_sidecar_dev, &params);
#if defined(SHAOBO_FULL_BWD_FUSED5)
    const int fused_status = dot_status == SHAOBO_FA3_STATUS_SUCCESS
                                 ? shaobo_fa3_bwd_fused5(
                                       dout_dev, q_dev, k_dev, v_dev,
                                       packed_sidecar_dev, dq_dev, dk_dev,
                                       dv_dev, &params)
                                 : dot_status;
    const int dkv_status = fused_status;
    const int dq_status = fused_status;
    constexpr const char* kPathName = "fused5";
#else
    params.reserved_ptr[3] = packed_sidecar_dev;
    const int dkv_status = dot_status == SHAOBO_FA3_STATUS_SUCCESS
                               ? shaobo_fa3_bwd_dkv(
                                     dout_dev, q_dev, k_dev, v_dev, out_dev,
                                     scores_max_dev, scores_sum_dev, nullptr,
                                     dk_dev, dv_dev, &params)
                               : dot_status;
    params.reserved_ptr[0] = delta_dev;
    const int dq_status = dkv_status == SHAOBO_FA3_STATUS_SUCCESS
                              ? shaobo_fa3_bwd_dq(
                                    dout_dev, q_dev, k_dev, v_dev, out_dev,
                                    scores_max_dev, scores_sum_dev, dq_dev,
                                    nullptr, nullptr, &params)
                              : dkv_status;
    constexpr const char* kPathName = "seven_gemm";
#endif
    const hipError_t sync_error = hipDeviceSynchronize();

    std::vector<float> delta_actual(rows), dq_actual(elems), dk_actual(elems),
        dv_actual(elems);
    bool outputs_copied =
        hip_ok(sync_error, "full backward synchronize") &&
        hip_ok(hipMemcpy(delta_actual.data(), delta_dev, row_bytes,
                         hipMemcpyDeviceToHost),
               "copy delta result");
#if defined(SHAOBO_FULL_BWD_FUSED5)
    std::vector<__half> dk_half(elems), dv_half(elems);
    outputs_copied = outputs_copied &&
        hip_ok(hipMemcpy(dk_half.data(), dk_dev, dkv_output_bytes,
                         hipMemcpyDeviceToHost),
               "copy dk result") &&
        hip_ok(hipMemcpy(dv_half.data(), dv_dev, dkv_output_bytes,
                         hipMemcpyDeviceToHost),
               "copy dv result");
    if (outputs_copied) {
        for (size_t i = 0; i < elems; ++i) {
            dk_actual[i] = static_cast<float>(dk_half[i]);
            dv_actual[i] = static_cast<float>(dv_half[i]);
        }
    }
#else
    outputs_copied = outputs_copied &&
        hip_ok(hipMemcpy(dk_actual.data(), dk_dev, dkv_output_bytes,
                         hipMemcpyDeviceToHost),
               "copy dk result") &&
        hip_ok(hipMemcpy(dv_actual.data(), dv_dev, dkv_output_bytes,
                         hipMemcpyDeviceToHost),
               "copy dv result");
#endif
    outputs_copied = outputs_copied &&
        hip_ok(hipMemcpy(dq_actual.data(), dq_dev, output_bytes,
                         hipMemcpyDeviceToHost),
               "copy dq result");

    const Metrics delta_metrics = compare(delta_actual, delta_expected);
    const Metrics dk_metrics = compare(dk_actual, dk_expected);
    const Metrics dv_metrics = compare(dv_actual, dv_expected);
    const Metrics dq_metrics = compare(dq_actual, dq_expected);
    const bool delta_pass = outputs_copied && delta_metrics.nonfinite == 0 &&
                            delta_metrics.max_abs <= 5.0e-5f &&
                            delta_metrics.rel_l2 <= 1.0e-4f;
#if defined(SHAOBO_FULL_BWD_FUSED5)
    constexpr float kGradientMaxAbsLimit = 3.0e-2f;
    constexpr float kGradientRelL2Limit = 3.0e-2f;
    const bool dkv_pass = outputs_copied && dk_metrics.nonfinite == 0 &&
                          dv_metrics.nonfinite == 0 &&
                          dk_metrics.max_abs <= kGradientMaxAbsLimit &&
                          dv_metrics.max_abs <= kGradientMaxAbsLimit &&
                          dk_metrics.rel_l2 <= kGradientRelL2Limit &&
                          dv_metrics.rel_l2 <= kGradientRelL2Limit;
    const bool dq_pass = outputs_copied && dq_metrics.nonfinite == 0 &&
                         dq_metrics.max_abs <= kGradientMaxAbsLimit &&
                         dq_metrics.rel_l2 <= kGradientRelL2Limit;
#else
    const bool dkv_pass = outputs_copied && dk_metrics.nonfinite == 0 &&
                          dv_metrics.nonfinite == 0 &&
                          dk_metrics.max_abs <= 5.0e-4f &&
                          dv_metrics.max_abs <= 5.0e-4f &&
                          (dk_metrics.rel_l2 <= 5.0e-3f ||
                           dk_metrics.rmse <= 5.0e-8f) &&
                          (dv_metrics.rel_l2 <= 5.0e-3f ||
                           dv_metrics.rmse <= 5.0e-8f);
    const bool dq_pass = outputs_copied && dq_metrics.nonfinite == 0 &&
                         dq_metrics.max_abs <= 5.0e-4f &&
                         dq_metrics.rmse <= 5.0e-5f;
#endif
    const bool pass = dot_status == SHAOBO_FA3_STATUS_SUCCESS &&
                      dkv_status == SHAOBO_FA3_STATUS_SUCCESS &&
                      dq_status == SHAOBO_FA3_STATUS_SUCCESS && delta_pass &&
                      dkv_pass && dq_pass;

    std::printf(
        "fa3_bwd_full_correctness path=%s B=%d H=%d S=%d D=%d causal=%d scale=%g "
        "dot_status=%s dkv_status=%s dq_status=%s "
        "delta_max_abs=%g delta_mean_abs=%g delta_rmse=%g "
        "delta_rel_l2=%g delta_cosine_error=%g delta_nonfinite=%d "
        "dk_max_abs=%g dk_mean_abs=%g dk_rmse=%g dk_rel_l2=%g "
        "dk_cosine_error=%g dk_nonfinite=%d "
        "dv_max_abs=%g dv_mean_abs=%g dv_rmse=%g dv_rel_l2=%g "
        "dv_cosine_error=%g dv_nonfinite=%d "
        "dq_max_abs=%g dq_mean_abs=%g dq_rmse=%g dq_rel_l2=%g "
        "dq_cosine_error=%g dq_nonfinite=%d "
        "dot_pass=%d dkv_pass=%d dq_pass=%d pass=%d\n",
        kPathName, batch, heads, seqlen, dim, params.causal,
        params.softmax_scale,
        status_string(dot_status),
        status_string(dkv_status), status_string(dq_status),
        delta_metrics.max_abs, delta_metrics.mean_abs, delta_metrics.rmse,
        delta_metrics.rel_l2, delta_metrics.cosine_error,
        delta_metrics.nonfinite, dk_metrics.max_abs, dk_metrics.mean_abs,
        dk_metrics.rmse, dk_metrics.rel_l2, dk_metrics.cosine_error,
        dk_metrics.nonfinite, dv_metrics.max_abs, dv_metrics.mean_abs,
        dv_metrics.rmse, dv_metrics.rel_l2, dv_metrics.cosine_error,
        dv_metrics.nonfinite, dq_metrics.max_abs, dq_metrics.mean_abs,
        dq_metrics.rmse, dq_metrics.rel_l2, dq_metrics.cosine_error,
        dq_metrics.nonfinite, delta_pass ? 1 : 0, dkv_pass ? 1 : 0,
        dq_pass ? 1 : 0, pass ? 1 : 0);

    free_device(allocations);
    return pass ? 0 : 1;
}
