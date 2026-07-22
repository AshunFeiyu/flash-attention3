#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

#ifndef SHAOBO_DENSE_NATIVE_F16_SCORE
#define SHAOBO_DENSE_NATIVE_F16_SCORE 0
#endif

#ifndef SHAOBO_DENSE_NATIVE_F16_DS
#define SHAOBO_DENSE_NATIVE_F16_DS 0
#endif

#ifndef SHAOBO_DENSE_NATIVE_F16_LTS
#define SHAOBO_DENSE_NATIVE_F16_LTS 0
#endif

#ifndef SHAOBO_DENSE_WRITER_ALT
#define SHAOBO_DENSE_WRITER_ALT 0
#endif

#ifndef SHAOBO_DENSE_WRITER_T
#define SHAOBO_DENSE_WRITER_T 1
#endif

#ifndef SHAOBO_DENSE_DQ_READER
#define SHAOBO_DENSE_DQ_READER 0
#endif

#ifndef SHAOBO_DENSE_DK_READER
#define SHAOBO_DENSE_DK_READER 0
#endif

static_assert(SHAOBO_DENSE_NATIVE_F16_LTS == 0 ||
                  SHAOBO_DENSE_NATIVE_F16_LTS == 1,
              "native FP16 MMAC LTS must be 0 or 1");
static_assert(!SHAOBO_DENSE_NATIVE_F16_DS || SHAOBO_DENSE_NATIVE_F16_SCORE,
              "native FP16 dS requires native FP16 score/dP");
static_assert(SHAOBO_DENSE_WRITER_ALT == 0 || SHAOBO_DENSE_WRITER_ALT == 1,
              "writer alt must be 0 or 1");
static_assert(SHAOBO_DENSE_WRITER_T == 0 || SHAOBO_DENSE_WRITER_T == 1,
              "writer t must be 0 or 1");
static_assert(SHAOBO_DENSE_DQ_READER >= 0 && SHAOBO_DENSE_DQ_READER <= 4,
              "dQ reader must be 0..4");
static_assert(SHAOBO_DENSE_DK_READER >= 0 && SHAOBO_DENSE_DK_READER <= 4,
              "dK reader must be 0..4");

constexpr int kWaveSize = 64;
constexpr int kWaves = 4;
constexpr int kThreads = kWaveSize * kWaves;
constexpr int kM = 16;
constexpr int kN = 32;
constexpr int kD = 32;
constexpr int kTile = 16;
constexpr int kMatrixElems = kN * kD;
constexpr int kMatrixBytes = kMatrixElems * sizeof(__half);
constexpr int kShortPageBytes = kN * kTile * sizeof(__half);
constexpr int kQBase = 0;
constexpr int kKBase = kQBase + kMatrixBytes;
constexpr int kVBase = kKBase + kMatrixBytes;
constexpr int kDoutBase = kVBase + kMatrixBytes;
constexpr int kDsBase = kDoutBase + kShortPageBytes;
// The t=1 m32x16 reader footprint is 64x16 even though the logical tile is
// 32x16. Keep the entire footprint in bounds and zero its writer holes.
constexpr int kDsPageBytes = 64 * 16 * sizeof(__half);
constexpr int kLdsHalfs = (kDsBase + kDsPageBytes) / sizeof(__half);
constexpr int kScoreElems = kM * kN;
constexpr int kGradientElems = kM * kD;
constexpr int kDkElems = kN * kD;

union FragF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
};

union AccF32x4 {
    ins::Vec4F32 f32;
    float scalar[4];
};

__device__ __forceinline__ ins::Vec8F16 read_candidate(
    _Float16* ptr, int mode) {
#if defined(__gfx946__) || defined(__gfx92a__)
    switch (mode) {
        case 0:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                ptr, 0, 2, 1, 0);
        case 1:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                ptr, 0, 1, 2, 0);
        case 2:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                ptr, 0, 1, 2, 1);
        case 3:
            return __builtin_hcu_ds_read_matrix_format_f16(
                ptr, 0, 2, 1, 0);
        default:
            return __builtin_hcu_ds_read_matrix_format_f16(
                ptr, 0, 2, 1, 1);
    }
#else
    (void)ptr;
    (void)mode;
    return {};
#endif
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__host__ __device__ __forceinline__ float half_round(float value) {
    return static_cast<float>(static_cast<_Float16>(value));
}

// This is deliberately not a softmax replacement. It is a dense, nonlinear,
// deterministic dS transform whose fp16 result has the same ownership as the
// score accumulator and can be reproduced by the host oracle exactly.
__host__ __device__ __forceinline__ float make_ds(float score,
                                                  float dp,
                                                  int qrow,
                                                  int krow) {
    const float score16 = half_round(score);
    const float dp16 = half_round(dp);
    const float row_bias = (static_cast<float>(qrow) - 7.5f) * 0.00390625f;
    const float col_bias = (static_cast<float>(krow) - 15.5f) * 0.001953125f;
    return half_round(score16 * 0.3125f + dp16 * 0.4375f + row_bias + col_bias);
}

__device__ __forceinline__ void t1_trans_source_qk(int src_lane,
                                                    int src_word,
                                                    int& qrow,
                                                    int& krow) {
    // Inverse of the measured t1-writer -> trans-m32-alt0 reader slot map.
    const int dst_lane =
        ((src_lane >> 0) & 1) | (((src_lane >> 1) & 1) << 1) |
        (((src_lane >> 2) & 1) << 2) | (((src_lane >> 3) & 1) << 3) |
        (((src_lane >> 5) & 1) << 4) | (((src_word >> 1) & 1) << 5);
    const int dst_word =
        ((src_word >> 0) & 1) | (((src_lane >> 4) & 1) << 1) |
        (((src_word >> 2) & 1) << 2);
    qrow = dst_lane & 15;
    krow = (dst_word >= 4 ? 16 : 0) + (dst_lane >> 4) * 4 +
           (dst_word & 3);
}

__device__ __forceinline__ AccF32x4 mmac_score(const FragF16x8& lhs,
                                                const FragF16x8& rhs) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    AccF32x4 out{};
#if defined(__gfx946__) || defined(__gfx92a__)
    // Fixed orientation: Q first, K second, LIT=1/LTS=1. This is the
    // mathematically validated Q-owned score orientation from the old probe.
    out.f32 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs.f16x4[0], zero.f32, 1, 1);
    out.f32 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs.f16x4[1], rhs.f16x4[1], out.f32, 1, 1);
#else
    (void)lhs;
    (void)rhs;
#endif
    return out;
}

__device__ __forceinline__ FragF16x8 mmac_score_f16(
    const FragF16x8& lhs,
    const FragF16x8& rhs0,
    const FragF16x8& rhs1) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    FragF16x8 out{};
#if defined(__gfx946__) || defined(__gfx92a__)
    out.f16x4[0] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs0.f16x4[0], zero.f16x4[0], 0,
        SHAOBO_DENSE_NATIVE_F16_LTS);
    out.f16x4[0] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[1], rhs0.f16x4[1], out.f16x4[0], 0,
        SHAOBO_DENSE_NATIVE_F16_LTS);
    out.f16x4[1] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs1.f16x4[0], zero.f16x4[0], 0,
        SHAOBO_DENSE_NATIVE_F16_LTS);
    out.f16x4[1] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[1], rhs1.f16x4[1], out.f16x4[1], 0,
        SHAOBO_DENSE_NATIVE_F16_LTS);
#else
    (void)lhs;
    (void)rhs0;
    (void)rhs1;
#endif
    return out;
}

template <int DHalf>
__device__ __forceinline__ AccF32x4 mmac_dq_half(
    const FragF16x8& ds,
    const FragF16x8& k_n0,
    const FragF16x8& k_n1) {
    static_assert(DHalf == 0 || DHalf == 1, "D half must be 0 or 1");
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    AccF32x4 out{};
#if defined(__gfx946__) || defined(__gfx92a__)
    out.f32 = ins::mmac_f16_lit(
        ds.f16x4[0], k_n0.f16x4[DHalf], zero.f32);
    out.f32 = ins::mmac_f16_lit(
        ds.f16x4[1], k_n1.f16x4[DHalf], out.f32);
#else
    (void)ds;
    (void)k_n0;
    (void)k_n1;
#endif
    return out;
}

__device__ __forceinline__ AccF32x4 mmac_one(ins::Vec4F16 lhs,
                                              ins::Vec4F16 rhs) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    AccF32x4 out{};
#if defined(__gfx946__) || defined(__gfx92a__)
    out.f32 = ins::mmac_f16_lit(lhs, rhs, zero.f32);
#else
    (void)lhs;
    (void)rhs;
#endif
    return out;
}

__device__ __forceinline__ void store_qk(float* out,
                                         int n_base,
                                         int lane,
                                         const AccF32x4& acc) {
    const int q_base = (lane >> 4) * 4;
    const int krow = n_base + (lane & 15);
#pragma unroll
    for (int vec = 0; vec < 4; ++vec) {
        atomicAdd(out + (q_base + vec) * kN + krow, acc.scalar[vec]);
    }
}

__device__ __forceinline__ void store_ds(float* out,
                                         int n_base,
                                         int lane,
                                         ins::Vec4F16 frag) {
    const int q_base = (lane >> 4) * 4;
    const int krow = n_base + (lane & 15);
#pragma unroll
    for (int vec = 0; vec < 4; ++vec) {
        atomicAdd(out + (q_base + vec) * kN + krow,
                  static_cast<float>(frag[vec]));
    }
}

__device__ __forceinline__ void store_trans_view(float* out,
                                                  int lane,
                                                  const FragF16x8& frag) {
    const int qrow = lane & 15;
    const int k_group = (lane >> 4) * 4;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const int krow = (word >= 4 ? 16 : 0) + k_group + (word & 3);
        out[qrow * kN + krow] = static_cast<float>(frag.f16x8[word]);
    }
}

__device__ __forceinline__ void store_normal_view(float* out,
                                                   int lane,
                                                   const FragF16x8& frag) {
    const int k_in_half = lane & 15;
    const int q_base = (lane >> 4) * 4;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
            const int krow = half * 16 + k_in_half;
            out[(q_base + vec) * kN + krow] =
                static_cast<float>(frag.f16x4[half][vec]);
        }
    }
}

__device__ __forceinline__ void store_dq(float* out,
                                         int d_base,
                                         int lane,
                                         const AccF32x4& acc) {
    const int qrow = lane & 15;
    const int d_group = (lane >> 4) * 4;
#pragma unroll
    for (int vec = 0; vec < 4; ++vec) {
        atomicAdd(out + qrow * kD + d_base + d_group + vec, acc.scalar[vec]);
    }
}

__device__ __forceinline__ void store_dk(float* out,
                                         int n_base,
                                         int d_base,
                                         int lane,
                                         const AccF32x4& acc) {
    const int krow = n_base + (lane & 15);
    const int d_group = (lane >> 4) * 4;
#pragma unroll
    for (int vec = 0; vec < 4; ++vec) {
        atomicAdd(out + krow * kD + d_base + d_group + vec, acc.scalar[vec]);
    }
}

__global__ void __launch_bounds__(kThreads, 1)
    dq_native_ds_write_dense_probe_kernel(
        const __half* __restrict__ q,
        const __half* __restrict__ k,
        const __half* __restrict__ v,
        const __half* __restrict__ dout,
        float* __restrict__ score_out,
        float* __restrict__ dp_out,
        float* __restrict__ ds_out,
        float* __restrict__ dq_direct_out,
        float* __restrict__ dk_direct_out,
        float* __restrict__ dq_out,
        float* __restrict__ dk_out,
        float* __restrict__ trans_view_out,
        float* __restrict__ normal_view_out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) __half lds[kLdsHalfs];
    constexpr int kMlsFilled = 0;
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    for (int i = static_cast<int>(threadIdx.x); i < kLdsHalfs;
         i += static_cast<int>(blockDim.x)) {
        lds[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(kMlsFilled, 3);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave == 0) {
        ins::abarrier_seq<false>(kMlsFilled);
        // Match the source-slot probe's full contract: MLS32x32 t plus the
        // m32 trans reader. MLS shape and transpose are part of the ABI.
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(q, kD), kQBase, true);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(k, kD), kKBase, true);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(v, kD), kVBase, true);
        ins::matrix_load_32x16_b16_bps_lds(
            lds, ins::prepare_matrix_src(dout, kD), kDoutBase);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(kMlsFilled, 3);
    } else if (wave == 1) {
        int phase = 0;
        ins::abarrier_try_wait<false>(kMlsFilled, phase);

        FragF16x8 q_trans{};
        FragF16x8 k_trans_n0{};
        FragF16x8 k_trans_n1{};
        FragF16x8 v_trans_n0{};
        FragF16x8 v_trans_n1{};
        FragF16x8 dout_trans{};
        FragF16x8 k_normal_d0{};
        FragF16x8 k_normal_d1{};
        FragF16x8 q_normal{};
        ins::ds_read_matrix_32x16_trans(lds, kQBase, q_trans.f16x8);
        ins::ds_read_matrix_trans_pair(lds, kKBase, k_trans_n0.f16x8,
                                       k_trans_n1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, kVBase, v_trans_n0.f16x8,
                                       v_trans_n1.f16x8);
        ins::ds_read_matrix_32x16_trans(lds, kDoutBase, dout_trans.f16x8);
        ins::ds_read_matrix_normal_pair(lds, kKBase, k_normal_d0.f16x8,
                                        k_normal_d1.f16x8);
        ins::ds_read_matrix_32x16_normal(lds, kQBase, q_normal.f16x8);
        ins::wait_lgkm(0);

        const AccF32x4 score_n0 = mmac_score(q_trans, k_trans_n0);
        const AccF32x4 score_n1 = mmac_score(q_trans, k_trans_n1);
        const AccF32x4 dp_n0 = mmac_score(dout_trans, v_trans_n0);
        const AccF32x4 dp_n1 = mmac_score(dout_trans, v_trans_n1);
        store_qk(score_out, 0, lane, score_n0);
        store_qk(score_out, 16, lane, score_n1);
        store_qk(dp_out, 0, lane, dp_n0);
        store_qk(dp_out, 16, lane, dp_n1);

        ins::Vec4F16 ds_n0{};
        ins::Vec4F16 ds_n1{};
        const int q_base = (lane >> 4) * 4;
        const int krow = lane & 15;
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
            const int qrow = q_base + vec;
            ds_n0[vec] = static_cast<_Float16>(
                make_ds(score_n0.scalar[vec], dp_n0.scalar[vec], qrow, krow));
            ds_n1[vec] = static_cast<_Float16>(make_ds(
                score_n1.scalar[vec], dp_n1.scalar[vec], qrow, krow + 16));
        }
        store_ds(ds_out, 0, lane, ds_n0);
        store_ds(ds_out, 16, lane, ds_n1);

        FragF16x8 writer{};
#if SHAOBO_DENSE_NATIVE_F16_SCORE
        writer = mmac_score_f16(q_trans, k_trans_n0, k_trans_n1);
#if SHAOBO_DENSE_NATIVE_F16_DS
        static_assert(SHAOBO_DENSE_NATIVE_F16_LTS == 0,
                      "validated dS source-slot path uses LTS0");
        static_assert(SHAOBO_DENSE_WRITER_T == 1 &&
                          SHAOBO_DENSE_DQ_READER == 0,
                      "dS source coordinates require t1/trans-m32 ABI");
        const FragF16x8 dp_writer =
            mmac_score_f16(dout_trans, v_trans_n0, v_trans_n1);
#pragma unroll
        for (int word = 0; word < 8; ++word) {
            int qrow = 0;
            int krow = 0;
            t1_trans_source_qk(lane, word, qrow, krow);
            writer.f16x8[word] = static_cast<_Float16>(make_ds(
                static_cast<float>(writer.f16x8[word]),
                static_cast<float>(dp_writer.f16x8[word]), qrow, krow));
        }
#endif
#else
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
            writer.f16x4[0][vec] = ds_n0[vec];
            writer.f16x4[1][vec] = ds_n1[vec];
        }
#endif
        store_dq(dq_direct_out, 0, lane,
                 mmac_dq_half<0>(writer, k_normal_d0, k_normal_d1));
        store_dq(dq_direct_out, 16, lane,
                 mmac_dq_half<1>(writer, k_normal_d0, k_normal_d1));
        store_dk(dk_direct_out, 0, 0, lane,
                 mmac_one(writer.f16x4[0], q_normal.f16x4[0]));
        store_dk(dk_direct_out, 16, 0, lane,
                 mmac_one(writer.f16x4[1], q_normal.f16x4[0]));
        store_dk(dk_direct_out, 0, 16, lane,
                 mmac_one(writer.f16x4[0], q_normal.f16x4[1]));
        store_dk(dk_direct_out, 16, 16, lane,
                 mmac_one(writer.f16x4[1], q_normal.f16x4[1]));
        // All format controls are compile-time constants in the emitted ISA.
        __builtin_hcu_ds_write_matrix_format_f16(
            writer.f16x8,
            reinterpret_cast<_Float16*>(reinterpret_cast<char*>(lds) + kDsBase),
            0, 2, 1, SHAOBO_DENSE_WRITER_ALT, SHAOBO_DENSE_WRITER_T);
        ins::wait_lgkm(0);
    }

    // wave2/3 consume the same writer page; the barrier is only publication
    // ordering for this focused experiment, not an algorithmic pipeline claim.
    __syncthreads();

    if (wave == 2) {
        FragF16x8 ds_trans{};
        FragF16x8 k_normal_d0{};
        FragF16x8 k_normal_d1{};
        ds_trans.f16x8 = read_candidate(
            reinterpret_cast<_Float16*>(reinterpret_cast<char*>(lds) + kDsBase),
            SHAOBO_DENSE_DQ_READER);
        ins::ds_read_matrix_normal_pair(lds, kKBase, k_normal_d0.f16x8,
                                        k_normal_d1.f16x8);
        ins::wait_lgkm(0);
        store_trans_view(trans_view_out, lane, ds_trans);
        store_dq(dq_out, 0, lane,
                 mmac_dq_half<0>(ds_trans, k_normal_d0, k_normal_d1));
        store_dq(dq_out, 16, lane,
                 mmac_dq_half<1>(ds_trans, k_normal_d0, k_normal_d1));
    } else if (wave == 3) {
        FragF16x8 ds_normal{};
        FragF16x8 q_normal{};
        ds_normal.f16x8 = read_candidate(
            reinterpret_cast<_Float16*>(reinterpret_cast<char*>(lds) + kDsBase),
            SHAOBO_DENSE_DK_READER);
        ins::ds_read_matrix_32x16_normal(lds, kQBase, q_normal.f16x8);
        ins::wait_lgkm(0);
        store_normal_view(normal_view_out, lane, ds_normal);
        store_dk(dk_out, 0, 0, lane,
                 mmac_one(ds_normal.f16x4[0], q_normal.f16x4[0]));
        store_dk(dk_out, 16, 0, lane,
                 mmac_one(ds_normal.f16x4[1], q_normal.f16x4[0]));
        store_dk(dk_out, 0, 16, lane,
                 mmac_one(ds_normal.f16x4[0], q_normal.f16x4[1]));
        store_dk(dk_out, 16, 16, lane,
                 mmac_one(ds_normal.f16x4[1], q_normal.f16x4[1]));
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(kMlsFilled);
    }
#else
    (void)q;
    (void)k;
    (void)v;
    (void)dout;
    (void)score_out;
    (void)dp_out;
    (void)ds_out;
    (void)dq_direct_out;
    (void)dk_direct_out;
    (void)dq_out;
    (void)dk_out;
    (void)trans_view_out;
    (void)normal_view_out;
#endif
}

float max_abs_diff(const std::vector<float>& actual,
                   const std::vector<float>& expected) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return INFINITY;
        }
        max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
    }
    return max_abs;
}

bool report(const char* name,
            const std::vector<float>& actual,
            const std::vector<float>& expected,
            float tolerance,
            int columns) {
    const float diff = max_abs_diff(actual, expected);
    const bool pass = std::isfinite(diff) && diff <= tolerance;
    int mismatch_count = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::fabs(actual[i] - expected[i]) > tolerance) {
            if (mismatch_count < 16) {
                std::printf(
                    "dense_native_ds_mismatch %s row=%zu col=%zu "
                    "actual=%g expected=%g abs=%g\n",
                    name, i / static_cast<size_t>(columns),
                    i % static_cast<size_t>(columns), actual[i], expected[i],
                    std::fabs(actual[i] - expected[i]));
            }
            ++mismatch_count;
        }
    }
    std::printf("dense_native_ds %s max_abs=%g mismatches=%d pass=%d\n",
                name, diff, mismatch_count, pass ? 1 : 0);
    return pass;
}

}  // namespace

int main() {
    std::vector<__half> q(kMatrixElems, __float2half(0.0f));
    std::vector<__half> k(kMatrixElems, __float2half(0.0f));
    std::vector<__half> v(kMatrixElems, __float2half(0.0f));
    std::vector<__half> dout(kMatrixElems, __float2half(0.0f));
    for (int row = 0; row < kN; ++row) {
        for (int d = 0; d < kD; ++d) {
            if (row < kM) {
                q[row * kD + d] = __float2half(
                    static_cast<float>((row * 73 + d * 29 + 17) % 17 - 8) /
                    64.0f);
                dout[row * kD + d] = __float2half(
                    static_cast<float>((row * 37 + d * 43 + 5) % 23 - 11) /
                    64.0f);
            }
            k[row * kD + d] = __float2half(
                static_cast<float>((row * 47 + d * 31 + 11) % 19 - 9) / 64.0f);
            v[row * kD + d] = __float2half(
                static_cast<float>((row * 53 + d * 41 + 13) % 29 - 14) /
                64.0f);
        }
    }

    std::vector<float> score_expected(kScoreElems, 0.0f);
    std::vector<float> dp_expected(kScoreElems, 0.0f);
    std::vector<float> ds_expected(kScoreElems, 0.0f);
    for (int qrow = 0; qrow < kM; ++qrow) {
        for (int krow = 0; krow < kN; ++krow) {
            float score = 0.0f;
            float dp = 0.0f;
            for (int d = 0; d < kD; ++d) {
                score += __half2float(q[qrow * kD + d]) *
                         __half2float(k[krow * kD + d]);
                dp += __half2float(dout[qrow * kD + d]) *
                      __half2float(v[krow * kD + d]);
            }
            const int index = qrow * kN + krow;
            score_expected[index] = score;
            dp_expected[index] = dp;
            ds_expected[index] = make_ds(score, dp, qrow, krow);
        }
    }

    const std::vector<float>& published_expected =
#if SHAOBO_DENSE_NATIVE_F16_SCORE
#if SHAOBO_DENSE_NATIVE_F16_DS
        ds_expected;
#else
        score_expected;
#endif
#else
        ds_expected;
#endif
    std::vector<float> dq_expected(kGradientElems, 0.0f);
    std::vector<float> dk_expected(kDkElems, 0.0f);
    for (int qrow = 0; qrow < kM; ++qrow) {
        for (int krow = 0; krow < kN; ++krow) {
            const float ds = half_round(
                published_expected[qrow * kN + krow]);
            for (int d = 0; d < kD; ++d) {
                dq_expected[qrow * kD + d] +=
                    ds * __half2float(k[krow * kD + d]);
                dk_expected[krow * kD + d] +=
                    ds * __half2float(q[qrow * kD + d]);
            }
        }
    }

    __half* d_q = nullptr;
    __half* d_k = nullptr;
    __half* d_v = nullptr;
    __half* d_dout = nullptr;
    float* d_score = nullptr;
    float* d_dp = nullptr;
    float* d_ds = nullptr;
    float* d_dq_direct = nullptr;
    float* d_dk_direct = nullptr;
    float* d_dq = nullptr;
    float* d_dk = nullptr;
    float* d_trans_view = nullptr;
    float* d_normal_view = nullptr;
    std::vector<float> score(kScoreElems);
    std::vector<float> dp(kScoreElems);
    std::vector<float> ds(kScoreElems);
    std::vector<float> dq_direct(kGradientElems);
    std::vector<float> dk_direct(kDkElems);
    std::vector<float> dq(kGradientElems);
    std::vector<float> dk(kDkElems);
    std::vector<float> trans_view(kScoreElems);
    std::vector<float> normal_view(kScoreElems);

    check_hip(hipMalloc(&d_q, kMatrixBytes), "hipMalloc Q");
    check_hip(hipMalloc(&d_k, kMatrixBytes), "hipMalloc K");
    check_hip(hipMalloc(&d_v, kMatrixBytes), "hipMalloc V");
    check_hip(hipMalloc(&d_dout, kMatrixBytes), "hipMalloc dO");
    check_hip(hipMalloc(&d_score, kScoreElems * sizeof(float)), "hipMalloc score");
    check_hip(hipMalloc(&d_dp, kScoreElems * sizeof(float)), "hipMalloc dP");
    check_hip(hipMalloc(&d_ds, kScoreElems * sizeof(float)), "hipMalloc dS");
    check_hip(hipMalloc(&d_dq_direct, kGradientElems * sizeof(float)),
              "hipMalloc direct dQ");
    check_hip(hipMalloc(&d_dk_direct, kDkElems * sizeof(float)),
              "hipMalloc direct dK");
    check_hip(hipMalloc(&d_dq, kGradientElems * sizeof(float)), "hipMalloc dQ");
    check_hip(hipMalloc(&d_dk, kDkElems * sizeof(float)), "hipMalloc dK");
    check_hip(hipMalloc(&d_trans_view, kScoreElems * sizeof(float)),
              "hipMalloc trans view");
    check_hip(hipMalloc(&d_normal_view, kScoreElems * sizeof(float)),
              "hipMalloc normal view");

    check_hip(hipMemcpy(d_q, q.data(), kMatrixBytes, hipMemcpyHostToDevice),
              "copy Q");
    check_hip(hipMemcpy(d_k, k.data(), kMatrixBytes, hipMemcpyHostToDevice),
              "copy K");
    check_hip(hipMemcpy(d_v, v.data(), kMatrixBytes, hipMemcpyHostToDevice),
              "copy V");
    check_hip(hipMemcpy(d_dout, dout.data(), kMatrixBytes, hipMemcpyHostToDevice),
              "copy dO");
    check_hip(hipMemset(d_score, 0, kScoreElems * sizeof(float)), "clear score");
    check_hip(hipMemset(d_dp, 0, kScoreElems * sizeof(float)), "clear dP");
    check_hip(hipMemset(d_ds, 0, kScoreElems * sizeof(float)), "clear dS");
    check_hip(hipMemset(d_dq_direct, 0, kGradientElems * sizeof(float)),
              "clear direct dQ");
    check_hip(hipMemset(d_dk_direct, 0, kDkElems * sizeof(float)),
              "clear direct dK");
    check_hip(hipMemset(d_dq, 0, kGradientElems * sizeof(float)), "clear dQ");
    check_hip(hipMemset(d_dk, 0, kDkElems * sizeof(float)), "clear dK");
    check_hip(hipMemset(d_trans_view, 0, kScoreElems * sizeof(float)),
              "clear trans view");
    check_hip(hipMemset(d_normal_view, 0, kScoreElems * sizeof(float)),
              "clear normal view");

    hipLaunchKernelGGL(dq_native_ds_write_dense_probe_kernel, dim3(1),
                       dim3(kThreads), 0, 0, d_q, d_k, d_v, d_dout, d_score,
                       d_dp, d_ds, d_dq_direct, d_dk_direct, d_dq, d_dk,
                       d_trans_view, d_normal_view);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    check_hip(hipMemcpy(score.data(), d_score, kScoreElems * sizeof(float),
                        hipMemcpyDeviceToHost), "copy score");
    check_hip(hipMemcpy(dp.data(), d_dp, kScoreElems * sizeof(float),
                        hipMemcpyDeviceToHost), "copy dP");
    check_hip(hipMemcpy(ds.data(), d_ds, kScoreElems * sizeof(float),
                        hipMemcpyDeviceToHost), "copy dS");
    check_hip(hipMemcpy(dq_direct.data(), d_dq_direct,
                        kGradientElems * sizeof(float), hipMemcpyDeviceToHost),
              "copy direct dQ");
    check_hip(hipMemcpy(dk_direct.data(), d_dk_direct,
                        kDkElems * sizeof(float), hipMemcpyDeviceToHost),
              "copy direct dK");
    check_hip(hipMemcpy(dq.data(), d_dq, kGradientElems * sizeof(float),
                        hipMemcpyDeviceToHost), "copy dQ");
    check_hip(hipMemcpy(dk.data(), d_dk, kDkElems * sizeof(float),
                        hipMemcpyDeviceToHost), "copy dK");
    check_hip(hipMemcpy(trans_view.data(), d_trans_view,
                        kScoreElems * sizeof(float), hipMemcpyDeviceToHost),
              "copy trans view");
    check_hip(hipMemcpy(normal_view.data(), d_normal_view,
                        kScoreElems * sizeof(float), hipMemcpyDeviceToHost),
              "copy normal view");

    constexpr float kTolerance = 2.0e-3f;
    const bool score_pass =
        report("score", score, score_expected, kTolerance, kN);
    const bool dp_pass = report("dP", dp, dp_expected, kTolerance, kN);
    const bool ds_pass = report("dS", ds, ds_expected, kTolerance, kN);
    const bool dq_direct_pass = report("dQ_direct", dq_direct, dq_expected,
                                       kTolerance, kD);
    const bool dk_direct_pass = report("dK_direct", dk_direct, dk_expected,
                                       kTolerance, kD);
    const bool dq_pass = report("dQ_trans_reader_d32", dq, dq_expected,
                                kTolerance, kD);
    const bool dk_pass = report("dK_normal_reader_d32", dk, dk_expected,
                                kTolerance, kD);
    std::vector<float> published_half(kScoreElems);
    for (int i = 0; i < kScoreElems; ++i) {
        published_half[i] = half_round(published_expected[i]);
    }
    const bool trans_view_pass = report("trans_reader_tensor", trans_view,
                                        published_half, kTolerance, kN);
    const bool normal_view_pass = report("normal_reader_tensor", normal_view,
                                         published_half, kTolerance, kN);
    const bool pass = score_pass && dp_pass && ds_pass &&
#if SHAOBO_DENSE_NATIVE_F16_SCORE
#if SHAOBO_DENSE_NATIVE_F16_DS
                      trans_view_pass && normal_view_pass &&
#endif
                      dq_pass && dk_pass;
#else
                      dq_direct_pass && dk_direct_pass && dq_pass && dk_pass;
#endif
    std::printf(
        "dense_native_ds_final M=16 N=32 D=32 writer=t%d_alt%d "
        "dq_reader=%d dk_reader=%d source=%s lts=%d pass=%d\n",
        SHAOBO_DENSE_WRITER_T, SHAOBO_DENSE_WRITER_ALT,
        SHAOBO_DENSE_DQ_READER, SHAOBO_DENSE_DK_READER,
        SHAOBO_DENSE_NATIVE_F16_DS
            ? "f16_mmac_ds"
            : (SHAOBO_DENSE_NATIVE_F16_SCORE ? "f16_mmac_score"
                                             : "f32_ds_downcast"),
        SHAOBO_DENSE_NATIVE_F16_LTS, pass ? 1 : 0);

    check_hip(hipFree(d_dk), "hipFree dK");
    check_hip(hipFree(d_dq), "hipFree dQ");
    check_hip(hipFree(d_normal_view), "hipFree normal view");
    check_hip(hipFree(d_trans_view), "hipFree trans view");
    check_hip(hipFree(d_dk_direct), "hipFree direct dK");
    check_hip(hipFree(d_dq_direct), "hipFree direct dQ");
    check_hip(hipFree(d_ds), "hipFree dS");
    check_hip(hipFree(d_dp), "hipFree dP");
    check_hip(hipFree(d_score), "hipFree score");
    check_hip(hipFree(d_dout), "hipFree dO");
    check_hip(hipFree(d_v), "hipFree V");
    check_hip(hipFree(d_k), "hipFree K");
    check_hip(hipFree(d_q), "hipFree Q");
    return pass ? 0 : 2;
}
