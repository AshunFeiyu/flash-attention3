#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kThreads = 256;
constexpr int kWaveSize = 64;
constexpr int kMmacFloatsPerLane = 4;
constexpr int kRows = 32;
constexpr int kCols = 64;
constexpr int kProbeK = 7;
constexpr int kKTagBase = 1024;
constexpr int kPackVariants = 4;

constexpr int kQPageBytes = 0;
constexpr int kScoreKPageBytes = 2048;
constexpr int kDqKPageBytes = 6144;
constexpr int kDsPageBytes = 8192;
constexpr int kReaderPageElems = 64 * 16;
constexpr int kLdsHalfs = 16 * 1024;

union ProbeF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union ProbeF32x4 {
    ins::Vec4F32 f32;
    float scalar[4];
};

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(2);
    }
}

float q_scale_for_decode(int q) {
    return 1.0f + static_cast<float>(q) * 0.0625f;
}

__device__ __forceinline__ ProbeF32x4 mmac_pair(const ProbeF16x8& lhs,
                                                const ProbeF16x8& rhs,
                                                const ins::F16x8& zero) {
    ProbeF32x4 out{};
#if defined(__gfx946__) || defined(__gfx938__)
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], zero.f32);
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[1], rhs.f16x4[1], out.f32);
#else
    (void)lhs;
    (void)rhs;
    (void)zero;
#endif
    return out;
}

__device__ __forceinline__ void pack_qowned_scores(
    const ProbeF32x4& score0,
    const ProbeF32x4& score1,
    int variant,
    ProbeF16x8& packed) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        packed.scalar[i] = static_cast<_Float16>(0.0f);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const _Float16 a = static_cast<_Float16>(score0.scalar[i]);
        const _Float16 b = static_cast<_Float16>(score1.scalar[i]);
        if (variant == 0) {
            packed.scalar[i] = a;
            packed.scalar[i + 4] = b;
        } else if (variant == 1) {
            packed.scalar[i * 2] = a;
            packed.scalar[i * 2 + 1] = b;
        } else if (variant == 2) {
            packed.scalar[(i < 2 ? i : i + 2)] = a;
            packed.scalar[(i < 2 ? i + 2 : i + 4)] = b;
        } else {
            packed.scalar[i] = score0.scalar[3 - i];
            packed.scalar[i + 4] = score1.scalar[3 - i];
        }
    }
}

__device__ __forceinline__ void store_acc(float* out,
                                          int variant,
                                          int lane,
                                          const ProbeF32x4& acc) {
#pragma unroll
    for (int i = 0; i < kMmacFloatsPerLane; ++i) {
        out[(variant * kWaveSize + lane) * kMmacFloatsPerLane + i] =
            acc.scalar[i];
    }
}

__global__ void dq_dswrite_qowned_chain_probe_kernel(
    const __half* __restrict__ q_input,
    const __half* __restrict__ score_k_input,
    const __half* __restrict__ dq_k_input,
    float* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) __half lds[kLdsHalfs];

    const int lane = static_cast<int>(threadIdx.x & 63);
    const int wave = static_cast<int>(threadIdx.x >> 6);
    constexpr int kMlsBarrier = 0;

    for (int i = threadIdx.x; i < kLdsHalfs; i += blockDim.x) {
        lds[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        __builtin_hcu_s_abarrier_init(kMlsBarrier, 3);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave == 0) {
        ins::Vec4U32 q_src = ins::prepare_matrix_src(q_input, kCols);
        ins::Vec4U32 score_k_src =
            ins::prepare_matrix_src(score_k_input, kCols);
        ins::Vec4U32 dq_k_src = ins::prepare_matrix_src(dq_k_input, kCols);
        ins::abarrier_seq<false>(kMlsBarrier);
        ins::matrix_load_32x16_b16_bps_lds(lds, q_src, kQPageBytes);
        ins::matrix_load_32x32_b16_bps_lds(lds, score_k_src,
                                           kScoreKPageBytes, true);
        ins::matrix_load_32x16_b16_bps_lds(lds, dq_k_src, kDqKPageBytes);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(kMlsBarrier, 3);
    }

    if (wave == 1) {
        int phase = 0;
        ins::abarrier_try_wait<false>(kMlsBarrier, phase);

        ProbeF16x8 q_trans{};
        ProbeF16x8 score_k0{};
        ProbeF16x8 score_k1{};
        ins::ds_read_matrix_32x16_trans(lds, kQPageBytes, q_trans.f16x8);
        ins::ds_read_matrix_trans_pair(lds, kScoreKPageBytes,
                                       score_k0.f16x8, score_k1.f16x8);
        ins::wait_lgkm(0);

        ins::F16x8 zero;
        ins::zero_f16x8(zero);
        const ProbeF32x4 score0 = mmac_pair(q_trans, score_k0, zero);
        const ProbeF32x4 score1 = mmac_pair(q_trans, score_k1, zero);

#pragma unroll
        for (int variant = 0; variant < kPackVariants; ++variant) {
            ProbeF16x8 packed{};
            pack_qowned_scores(score0, score1, variant, packed);
            __builtin_hcu_ds_write_matrix_format_f16(
                packed.f16x8,
                lds + (kDsPageBytes / 2) + variant * kReaderPageElems,
                16, 2, 1, 0, 0);
        }
        ins::wait_lgkm(0);
    }

    __syncthreads();

    if (wave == 2) {
        ProbeF16x8 dq_k{};
        ins::ds_read_matrix_32x16_normal(lds, kDqKPageBytes, dq_k.f16x8);
        ins::wait_lgkm(0);

        ins::F16x8 zero;
        ins::zero_f16x8(zero);
#pragma unroll
        for (int variant = 0; variant < kPackVariants; ++variant) {
            const ins::Vec8F16 ds_vec =
                __builtin_hcu_ds_read_matrix_trans_format_f16(
                    lds + (kDsPageBytes / 2) + variant * kReaderPageElems,
                    16, 2, 1, 0);
            ProbeF16x8 ds_frag{};
            ds_frag.f16x8 = ds_vec;
            ins::wait_lgkm(0);
            store_acc(out, variant, lane, mmac_pair(ds_frag, dq_k, zero));
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_hcu_s_abarrier_inv(kMlsBarrier);
    }
#else
    (void)q_input;
    (void)score_k_input;
    (void)dq_k_input;
    (void)out;
#endif
}

bool decode_dq_value(float value, int& q, int& krow, int& d) {
    if (!std::isfinite(value) || std::fabs(value) < 0.5f) {
        return false;
    }
    for (int q_candidate = 0; q_candidate < 16; ++q_candidate) {
        const float q_scale = q_scale_for_decode(q_candidate);
        const float raw_tag =
            value / q_scale - static_cast<float>(kKTagBase);
        const int tag_candidate = static_cast<int>(std::lround(raw_tag));
        if (tag_candidate < 0 || tag_candidate >= 16 * kCols) {
            continue;
        }
        const int krow_candidate = tag_candidate / kCols;
        const int d_candidate = tag_candidate % kCols;
        if (d_candidate >= 32) {
            continue;
        }
        const float expected =
            q_scale * static_cast<float>(kKTagBase + tag_candidate);
        if (std::fabs(value - expected) <= 2.0f) {
            q = q_candidate;
            krow = krow_candidate;
            d = d_candidate;
            return true;
        }
    }
    return false;
}

void summarize_variant(const std::vector<float>& out, int variant) {
    bool seen_output[kWaveSize * kMmacFloatsPerLane] = {};
    bool seen_q[16] = {};
    bool seen_krow[16] = {};
    bool seen_d[32] = {};
    int raw_nonzero = 0;
    int decoded = 0;
    int nan = 0;
    int printed = 0;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
            const float value =
                out[(variant * kWaveSize + lane) * kMmacFloatsPerLane + vec];
            if (!std::isfinite(value)) {
                ++nan;
                continue;
            }
            if (std::fabs(value) > 0.5f) {
                ++raw_nonzero;
            }
            int q = -1;
            int krow = -1;
            int d = -1;
            if (!decode_dq_value(value, q, krow, d)) {
                continue;
            }
            seen_output[lane * kMmacFloatsPerLane + vec] = true;
            seen_q[q] = true;
            seen_krow[krow] = true;
            seen_d[d] = true;
            ++decoded;
            if (printed < 8) {
                std::printf(
                    "qowned_chain_sample variant=%d lane=%d vec=%d "
                    "q=%d krow=%d d=%d value=%g\n",
                    variant, lane, vec, q, krow, d, value);
                ++printed;
            }
        }
    }

    int unique_output = 0;
    int unique_q = 0;
    int unique_krow = 0;
    int unique_d = 0;
    for (bool v : seen_output) {
        unique_output += v ? 1 : 0;
    }
    for (bool v : seen_q) {
        unique_q += v ? 1 : 0;
    }
    for (bool v : seen_krow) {
        unique_krow += v ? 1 : 0;
    }
    for (bool v : seen_d) {
        unique_d += v ? 1 : 0;
    }
    const bool pass =
        unique_output == 256 && unique_q == 16 && unique_krow == 1 &&
        seen_krow[kProbeK] && unique_d >= 16;
    std::printf(
        "qowned_chain_summary variant=%d raw_nonzero=%d nan=%d decoded=%d "
        "unique_output=%d unique_q=%d unique_krow=%d probe_k_seen=%d "
        "unique_d=%d pass=%d\n",
        variant, raw_nonzero, nan, decoded, unique_output, unique_q,
        unique_krow, seen_krow[kProbeK] ? 1 : 0, unique_d, pass ? 1 : 0);
}

}  // namespace

int main() {
    __half* q_dev = nullptr;
    __half* score_k_dev = nullptr;
    __half* dq_k_dev = nullptr;
    float* out_dev = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&q_dev),
                        kRows * kCols * sizeof(__half)),
              "hipMalloc q");
    check_hip(hipMalloc(reinterpret_cast<void**>(&score_k_dev),
                        kRows * kCols * sizeof(__half)),
              "hipMalloc score_k");
    check_hip(hipMalloc(reinterpret_cast<void**>(&dq_k_dev),
                        kRows * kCols * sizeof(__half)),
              "hipMalloc dq_k");
    check_hip(hipMalloc(reinterpret_cast<void**>(&out_dev),
                        kPackVariants * kWaveSize * kMmacFloatsPerLane *
                            sizeof(float)),
              "hipMalloc out");

    std::vector<__half> q(kRows * kCols, __float2half(0.0f));
    std::vector<__half> score_k(kRows * kCols, __float2half(0.0f));
    std::vector<__half> dq_k(kRows * kCols, __float2half(0.0f));

    for (int row = 0; row < 16; ++row) {
        const __half qv = __float2half(q_scale_for_decode(row));
        for (int d = 0; d < 32; ++d) {
            q[row * kCols + d] = qv;
        }
    }
    for (int d = 0; d < 32; ++d) {
        score_k[kProbeK * kCols + d] = __float2half(1.0f / 32.0f);
        dq_k[kProbeK * kCols + d] =
            __float2half(static_cast<float>(kKTagBase + kProbeK * kCols + d));
    }

    std::vector<float> out(kPackVariants * kWaveSize * kMmacFloatsPerLane);
    check_hip(hipMemcpy(q_dev, q.data(), q.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemcpy(score_k_dev, score_k.data(),
                        score_k.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy score_k");
    check_hip(hipMemcpy(dq_k_dev, dq_k.data(), dq_k.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy dq_k");
    check_hip(hipMemset(out_dev, 0, out.size() * sizeof(float)),
              "hipMemset out");

    hipLaunchKernelGGL(dq_dswrite_qowned_chain_probe_kernel, dim3(1),
                       dim3(kThreads), 0, 0, q_dev, score_k_dev, dq_k_dev,
                       out_dev);
    check_hip(hipDeviceSynchronize(),
              "dq_dswrite_qowned_chain_probe_kernel");
    check_hip(hipMemcpy(out.data(), out_dev, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy out");

    bool any_pass = false;
    for (int variant = 0; variant < kPackVariants; ++variant) {
        summarize_variant(out, variant);
    }
    for (int variant = 0; variant < kPackVariants; ++variant) {
        bool seen_q[16] = {};
        bool seen_krow[16] = {};
        bool seen_d[32] = {};
        bool seen_output[kWaveSize * kMmacFloatsPerLane] = {};
        for (int lane = 0; lane < kWaveSize; ++lane) {
            for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
                int qv = -1;
                int krow = -1;
                int d = -1;
                const float value =
                    out[(variant * kWaveSize + lane) * kMmacFloatsPerLane +
                        vec];
                if (decode_dq_value(value, qv, krow, d)) {
                    seen_q[qv] = true;
                    seen_krow[krow] = true;
                    seen_d[d] = true;
                    seen_output[lane * kMmacFloatsPerLane + vec] = true;
                }
            }
        }
        int unique_output = 0;
        int unique_q = 0;
        int unique_krow = 0;
        int unique_d = 0;
        for (bool v : seen_output) unique_output += v ? 1 : 0;
        for (bool v : seen_q) unique_q += v ? 1 : 0;
        for (bool v : seen_krow) unique_krow += v ? 1 : 0;
        for (bool v : seen_d) unique_d += v ? 1 : 0;
        any_pass = any_pass ||
                   (unique_output == 256 && unique_q == 16 &&
                    unique_krow == 1 && seen_krow[kProbeK] &&
                    unique_d >= 16);
    }

    std::printf("qowned_chain_final any_pass=%d\n", any_pass ? 1 : 0);
    (void)hipFree(q_dev);
    (void)hipFree(score_k_dev);
    (void)hipFree(dq_k_dev);
    (void)hipFree(out_dev);
    return any_pass ? 0 : 1;
}
