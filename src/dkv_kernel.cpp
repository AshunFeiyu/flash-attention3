#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_fa3_api.h"
#include "dkv_contract.h"
#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace dkv = shaobo::fa3::bwd::dkv;
namespace ins = shaobo::fa3::bwd::dkv::instr;

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

inline bool valid_dkv_shape(const ShaoboFa3Params* p) {
    return p != nullptr && p->batch > 0 && p->seqlen_q > 0 &&
           p->seqlen_k > 0 && p->num_heads_q > 0 &&
           p->head_dim_qk == dkv::DkvTileD128Mq32Nk128::kHeadDim &&
           p->head_dim_v == dkv::DkvTileD128Mq32Nk128::kHeadDim &&
           p->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           p->layout == SHAOBO_FA3_LAYOUT_BHSD;
}

inline bool valid_probe_shape(const ShaoboFa3Params* p) {
    return valid_dkv_shape(p) &&
           p->dkv_path == dkv::kDkvPathWaspProbe &&
           p->seqlen_q == dkv::DkvTileD128Mq32Nk128::kProbeSeqLen &&
           p->seqlen_k == p->seqlen_q &&
           p->seqlen_k % dkv::DkvTileD128Mq32Nk128::kResidentNk == 0 &&
           p->num_heads_kv == p->num_heads_q;
}

inline bool valid_reference_shape(const ShaoboFa3Params* p) {
    return valid_dkv_shape(p) &&
           p->dkv_path == dkv::kDkvPathReferenceCorrectness &&
           p->seqlen_k == p->seqlen_q &&
           p->num_heads_kv == p->num_heads_q &&
           p->dropout_p == 0.0f;
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

inline size_t probe_workspace_bytes(const ShaoboFa3Params* p) {
    if (!valid_probe_shape(p)) {
        return 0;
    }
    const int grid_x =
        ceil_div(p->seqlen_k, dkv::DkvTileD128Mq32Nk128::kResidentNk);
    const size_t blocks = static_cast<size_t>(grid_x) *
                          static_cast<size_t>(p->num_heads_q) *
                          static_cast<size_t>(p->batch);
    return blocks * 8 * sizeof(float);
}

inline int64_t tensor_offset(
    int b, int h, int s, int d, int heads, int seqlen, int dim) {
    return ((static_cast<int64_t>(b) * heads + h) * seqlen + s) * dim + d;
}

template <typename Tile>
struct DkvLdsLayout {
    static constexpr int kBlockBytes = 32 * 32 * Tile::kHalfBytes;
    static constexpr int kRawPages = 2;
    static constexpr int kBlocksPerMqTile = Tile::kHeadDim / 32;
    static constexpr int kBlocksPerKvTile =
        Tile::kResidentNk / 32 * Tile::kHeadDim / 32;
    static constexpr int kQBase = 0;
    static constexpr int kDoutBase =
        kQBase + kRawPages * kBlocksPerMqTile * kBlockBytes;
    static constexpr int kKBase =
        kDoutBase + kRawPages * kBlocksPerMqTile * kBlockBytes;
    static constexpr int kVBase = kKBase + kBlocksPerKvTile * kBlockBytes;
    static constexpr int kBytes = kVBase + kBlocksPerKvTile * kBlockBytes;
    static_assert(kBytes <= Tile::kLdsBudgetBytes,
                  "dKV bringup probe LDS plan must fit 128KB");
};

template <typename Tile>
__device__ __forceinline__ int raw_page_block_offset(int page, int d_block) {
    return (page * DkvLdsLayout<Tile>::kBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int kv_block_offset(int row_block, int d_block) {
    return (row_block * 4 + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Wdra>
__device__ __forceinline__ void seq_raw_filled_page(int page) {
    if (page == 0) {
        ins::abarrier_seq<false>(Wdra::kRaw0Filled);
    } else {
        ins::abarrier_seq<false>(Wdra::kRaw1Filled);
    }
}

template <typename Wdra>
__device__ __forceinline__ void arrive_raw_filled_page(int page) {
    if (page == 0) {
        ins::abarrier_arrive_cnt<false>(Wdra::kRaw0Filled, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Wdra::kRaw1Filled, 1);
    }
}

template <typename Wdra>
__device__ __forceinline__ void wait_raw_used_page(
    int page,
    int& phase0,
    int& phase1) {
    if (page == 0) {
        ins::abarrier_try_wait<true>(Wdra::kRaw0Used, phase0);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kRaw1Used, phase1);
    }
}

template <typename Wdra>
__device__ __forceinline__ void wait_raw_filled_page(
    int page,
    int& phase0,
    int& phase1) {
    if (page == 0) {
        ins::abarrier_try_wait<true>(Wdra::kRaw0Filled, phase0);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kRaw1Filled, phase1);
    }
}

template <typename Wdra>
__device__ __forceinline__ void arrive_raw_used_page(int page) {
    if (page == 0) {
        ins::abarrier_arrive_cnt<false>(Wdra::kRaw0Used, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Wdra::kRaw1Used, 1);
    }
}

template <typename Tile>
__device__ __forceinline__ void publish_mq32_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    int q_base,
    int page,
    int wave_local) {
    const __half* src_tile =
        src + static_cast<int64_t>(q_base) * row_stride + wave_local * 32;
    ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, row_stride);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, srsrc, lds_base + raw_page_block_offset<Tile>(page, wave_local),
        true);
}

template <typename Tile>
__device__ __forceinline__ void publish_nk128_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    int k_base,
    int wave_local) {
#pragma unroll
    for (int row_block = 0; row_block < 4; ++row_block) {
        const __half* src_tile =
            src + static_cast<int64_t>(k_base + row_block * 32) *
                      row_stride +
            wave_local * 32;
        ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, row_stride);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, srsrc,
            lds_base + kv_block_offset<Tile>(row_block, wave_local), true);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_qk_loop(
    const __half* q_base_ptr,
    const __half* k_base_ptr,
    __half* lds,
    int q_base,
    int k_base,
    int row_stride,
    int wave_local) {
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < Tile::kQTilesPerCta; ++q_tile) {
        const int page = q_tile & 1;
        if (q_tile >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }
        seq_raw_filled_page<Wdra>(page);
        publish_mq32_tile<Tile>(
            lds, Layout::kQBase, q_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);
        arrive_raw_filled_page<Wdra>(page);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_dout_v_loop(
    const __half* dout_base_ptr,
    const __half* v_base_ptr,
    __half* lds,
    int q_base,
    int k_base,
    int row_stride,
    int wave_local) {
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < Tile::kQTilesPerCta; ++q_tile) {
        const int page = q_tile & 1;
        if (q_tile >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }
        seq_raw_filled_page<Wdra>(page);
        publish_mq32_tile<Tile>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);
        arrive_raw_filled_page<Wdra>(page);
    }
}

template <typename Tile>
__device__ __forceinline__ void score_dp_mmac_probe(
    __half* lds,
    int consumer_group,
    int page,
    int wave_local,
    ins::F32x4& score,
    ins::F32x4& dp) {
    using Layout = DkvLdsLayout<Tile>;

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    score.f32 = zero.f32;
    dp.f32 = zero.f32;

    const int row_block = (consumer_group * 2 + (wave_local & 1)) & 3;
    ins::raise_priority_2();
#pragma unroll
    for (int d_block = 0; d_block < 4; ++d_block) {
        ins::F16x8 q0;
        ins::F16x8 q1;
        ins::F16x8 k0;
        ins::F16x8 k1;
        ins::F16x8 dout0;
        ins::F16x8 dout1;
        ins::F16x8 v0;
        ins::F16x8 v1;
        const int q_off =
            Layout::kQBase + raw_page_block_offset<Tile>(page, d_block);
        const int dout_off =
            Layout::kDoutBase + raw_page_block_offset<Tile>(page, d_block);
        const int k_off =
            Layout::kKBase + kv_block_offset<Tile>(row_block, d_block);
        const int v_off =
            Layout::kVBase + kv_block_offset<Tile>(row_block, d_block);

        ins::ds_read_matrix_trans_pair(lds, q_off, q0.f16x8, q1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, k_off, k0.f16x8, k1.f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, dout_off, dout0.f16x8, dout1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, v_off, v0.f16x8, v1.f16x8);
        ins::wait_lgkm(0);

        score.f32 = ins::mmac_f16_lit(q0.f16x4[0], k0.f16x4[0], score.f32);
        score.f32 = ins::mmac_f16_lit(q0.f16x4[1], k0.f16x4[1], score.f32);
        score.f32 = ins::mmac_f16_lit(q1.f16x4[0], k1.f16x4[0], score.f32);
        score.f32 = ins::mmac_f16_lit(q1.f16x4[1], k1.f16x4[1], score.f32);

        dp.f32 = ins::mmac_f16_lit(dout0.f16x4[0], v0.f16x4[0], dp.f32);
        dp.f32 = ins::mmac_f16_lit(dout0.f16x4[1], v0.f16x4[1], dp.f32);
        dp.f32 = ins::mmac_f16_lit(dout1.f16x4[0], v1.f16x4[0], dp.f32);
        dp.f32 = ins::mmac_f16_lit(dout1.f16x4[1], v1.f16x4[1], dp.f32);
    }
    ins::lower_priority();
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void consumer_score_dp_loop(
    __half* lds,
    float* diag,
    int diag_index,
    int consumer_group,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

    float diag_accum = 0.0f;
#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < Tile::kQTilesPerCta; ++q_tile) {
        const int page = q_tile & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score;
        ins::F32x4 dp;
        score_dp_mmac_probe<Tile>(
            lds, consumer_group, page, wave_local, score, dp);
        diag_accum += score.scalar[0] + dp.scalar[0];

        arrive_raw_used_page<Wdra>(page);
    }

    if (lane == 0 && diag != nullptr) {
        diag[diag_index * 8 + consumer_group * 4 + wave_local] =
            diag_accum;
    }
}

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(16)))
fa3_bwd_dkv_probe_kernel(const __half* __restrict__ dout,
                         const __half* __restrict__ q,
                         const __half* __restrict__ k,
                         const __half* __restrict__ v,
                         float* __restrict__ diag,
                         int diag_stride,
                         int batch,
                         int heads,
                         int seqlen,
                         int dim) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV bringup probe LDS budget overflow");

    (void)diag_stride;
    (void)batch;
    __shared__ __half lds[Tile::kLdsBudgetBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);
    const int lane = static_cast<int>(threadIdx.x % 64);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int64_t tensor_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
    const __half* q_head = q + tensor_base;
    const __half* k_head = k + tensor_base;
    const __half* v_head = v + tensor_base;
    const __half* dout_head = dout + tensor_base;
    const int diag_index = blockIdx.z * gridDim.y * gridDim.x +
                           blockIdx.y * gridDim.x + blockIdx.x;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        producer_qk_loop<Tile, Bar>(
            q_head, k_head, lds, q_base, k_base, dim, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        consumer_score_dp_loop<Tile, Bar>(
            lds, diag, diag_index, 0, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        consumer_score_dp_loop<Tile, Bar>(
            lds, diag, diag_index, 1, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        producer_dout_v_loop<Tile, Bar>(
            dout_head, v_head, lds, q_base, k_base, dim, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __syncthreads();
    __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
    __builtin_hcu_s_abarrier_inv(Bar::kRaw0Filled);
    __builtin_hcu_s_abarrier_inv(Bar::kRaw0Used);
    __builtin_hcu_s_abarrier_inv(Bar::kRaw1Filled);
    __builtin_hcu_s_abarrier_inv(Bar::kRaw1Used);
    __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    __syncthreads();
#else
    (void)dout;
    (void)q;
    (void)k;
    (void)v;
    (void)diag;
    (void)diag_stride;
    (void)batch;
    (void)heads;
    (void)seqlen;
    (void)dim;
#endif
}

__global__ void fa3_bwd_dkv_ref_softmax_kernel(
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
        const int64_t p_index =
            static_cast<int64_t>(row) * seqlen + k_idx;
        if (causal && k_idx > q_idx) {
            prob[p_index] = 0.0f;
            continue;
        }
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += __half2float(q[head_base + q_idx * dim + d]) *
                   __half2float(k[head_base + k_idx * dim + d]);
        }
        const float p = expf(dot * scale - max_score);
        prob[p_index] = p;
        denom += p;
    }
    const float inv_denom = 1.0f / denom;
    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
        const int64_t p_index =
            static_cast<int64_t>(row) * seqlen + k_idx;
        prob[p_index] *= inv_denom;
    }
    row_max[row] = max_score;
    row_sum[row] = denom;
}

__global__ void fa3_bwd_dkv_ref_delta_kernel(
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

    float accum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float out_d = 0.0f;
        for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
            out_d += prob[static_cast<int64_t>(row) * seqlen + k_idx] *
                     __half2float(v[head_base + k_idx * dim + d]);
        }
        accum += __half2float(dout[head_base + q_idx * dim + d]) * out_d;
    }
    delta[row] = accum;
}

__global__ void fa3_bwd_dkv_ref_dp_kernel(
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
    const int64_t pairs = static_cast<int64_t>(batch) * heads * seqlen *
                          seqlen;
    if (pair >= pairs) {
        return;
    }

    const int k_idx = static_cast<int>(pair % seqlen);
    const int q_idx = static_cast<int>((pair / seqlen) % seqlen);
    const int h = static_cast<int>((pair / (seqlen * seqlen)) % heads);
    const int b = static_cast<int>(pair / (static_cast<int64_t>(heads) *
                                           seqlen * seqlen));
    if (causal && k_idx > q_idx) {
        dp[pair] = 0.0f;
        return;
    }

    const int64_t head_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dot += __half2float(dout[head_base + q_idx * dim + d]) *
               __half2float(v[head_base + k_idx * dim + d]);
    }
    dp[pair] = dot;
}

__global__ void fa3_bwd_dkv_ref_output_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ dout,
    const float* __restrict__ prob,
    const float* __restrict__ dp,
    const float* __restrict__ delta,
    float* __restrict__ dk,
    float* __restrict__ dv,
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
    const int k_idx = static_cast<int>((elem / dim) % seqlen);
    const int h = static_cast<int>((elem / (dim * seqlen)) % heads);
    const int b = static_cast<int>(elem / (static_cast<int64_t>(heads) *
                                           seqlen * dim));
    const int64_t row_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen;
    const int64_t head_base = row_base * dim;

    float dk_accum = 0.0f;
    float dv_accum = 0.0f;
    for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
        if (causal && k_idx > q_idx) {
            continue;
        }
        const int64_t row = row_base + q_idx;
        const int64_t pair = row * seqlen + k_idx;
        const float p = prob[pair];
        dv_accum += p * __half2float(dout[head_base + q_idx * dim + d]);
        const float ds = p * (dp[pair] - delta[row]) * scale;
        dk_accum += ds * __half2float(q[head_base + q_idx * dim + d]);
    }
    dk[elem] = dk_accum;
    dv[elem] = dv_accum;
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
    if (params->dkv_path == dkv::kDkvPathReferenceCorrectness) {
        return reference_workspace_bytes(params);
    }
    return probe_workspace_bytes(params);
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
    (void)out;
    (void)softmax_aux0;
    (void)softmax_aux1;
    (void)dq;

    if (params == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    if (params->dkv_path == dkv::kDkvPathReferenceCorrectness) {
        if (!valid_reference_shape(params)) {
            return SHAOBO_FA3_STATUS_UNSUPPORTED;
        }
        if (dout == nullptr || q == nullptr || k == nullptr ||
            v == nullptr || dk == nullptr || dv == nullptr ||
            params->workspace == nullptr) {
            return SHAOBO_FA3_STATUS_INVALID_VALUE;
        }
        const size_t required_workspace =
            reference_workspace_bytes(params);
        if (params->workspace_bytes < required_workspace) {
            return SHAOBO_FA3_STATUS_INVALID_VALUE;
        }

        float* workspace = static_cast<float*>(params->workspace);
        const int rows =
            params->batch * params->num_heads_q * params->seqlen_q;
        const int64_t pairs =
            static_cast<int64_t>(rows) * params->seqlen_k;
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

        fa3_bwd_dkv_ref_softmax_kernel<<<row_blocks, threads>>>(
            static_cast<const __half*>(q), static_cast<const __half*>(k),
            prob, row_max, row_sum, params->batch, params->num_heads_q,
            params->seqlen_q, params->head_dim_qk, params->causal,
            params->softmax_scale);
        fa3_bwd_dkv_ref_delta_kernel<<<row_blocks, threads>>>(
            static_cast<const __half*>(dout), static_cast<const __half*>(v),
            prob, delta, params->batch, params->num_heads_q,
            params->seqlen_q, params->head_dim_qk);
        fa3_bwd_dkv_ref_dp_kernel<<<pair_blocks, threads>>>(
            static_cast<const __half*>(dout), static_cast<const __half*>(v),
            dp, params->batch, params->num_heads_q, params->seqlen_q,
            params->head_dim_qk, params->causal);
        fa3_bwd_dkv_ref_output_kernel<<<elem_blocks, threads>>>(
            static_cast<const __half*>(q), static_cast<const __half*>(dout),
            prob, dp, delta, static_cast<float*>(dk),
            static_cast<float*>(dv), params->batch, params->num_heads_q,
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

    if (!valid_probe_shape(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    if (dout == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        params->workspace == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->workspace_bytes < shaobo_fa3_bwd_workspace_bytes(params)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int grid_x =
        ceil_div(params->seqlen_k, dkv::DkvTileD128Mq32Nk128::kResidentNk);
    dim3 grid(grid_x, params->num_heads_q, params->batch);
    dim3 block(dkv::DkvTileD128Mq32Nk128::kThreadsPerCta);

    hipLaunchKernelGGL(
        fa3_bwd_dkv_probe_kernel, grid, block, 0, 0,
        static_cast<const __half*>(dout), static_cast<const __half*>(q),
        static_cast<const __half*>(k), static_cast<const __half*>(v),
        static_cast<float*>(params->workspace), grid_x, params->batch,
        params->num_heads_q, params->seqlen_k, params->head_dim_qk);
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

struct DkvCompareMetrics {
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

void fill_dkv_inputs(std::vector<__half>& q,
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

void cpu_reference_dkv(const std::vector<__half>& q,
                       const std::vector<__half>& k,
                       const std::vector<__half>& v,
                       const std::vector<__half>& dout,
                       std::vector<float>& dk,
                       std::vector<float>& dv,
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
                for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                    prob[static_cast<size_t>(row) * seqlen + k_idx] /= denom;
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

    std::fill(dk.begin(), dk.end(), 0.0f);
    std::fill(dv.begin(), dv.end(), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                for (int d = 0; d < dim; ++d) {
                    float dk_accum = 0.0f;
                    float dv_accum = 0.0f;
                    for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
                        if (causal && k_idx > q_idx) {
                            continue;
                        }
                        const int row = (b * heads + h) * seqlen + q_idx;
                        const size_t pair =
                            static_cast<size_t>(row) * seqlen + k_idx;
                        const float p = prob[pair];
                        dv_accum += p * __half2float(
                                             dout[tensor_offset(
                                                 b, h, q_idx, d, heads,
                                                 seqlen, dim)]);
                        const float ds = p * (dp[pair] - delta[row]) * scale;
                        dk_accum += ds * __half2float(
                                             q[tensor_offset(
                                                 b, h, q_idx, d, heads,
                                                 seqlen, dim)]);
                    }
                    const int64_t out_index =
                        tensor_offset(b, h, k_idx, d, heads, seqlen, dim);
                    dk[out_index] = dk_accum;
                    dv[out_index] = dv_accum;
                }
            }
        }
    }
}

DkvCompareMetrics compare_vectors(const std::vector<float>& actual,
                                  const std::vector<float>& expected) {
    DkvCompareMetrics m;
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

#ifndef SHAOBO_FA3_NO_STANDALONE
int main(int argc, char** argv) {
    const int check = arg_int(
        argc, argv, "--check", env_int("CHECK_DKV", 0));
    const int batch = arg_int(argc, argv, "--B", env_int("B", kDefaultBatch));
    const int heads = arg_int(argc, argv, "--H", env_int("H", kDefaultHeads));
    const int default_seq = check ? 128 : kDefaultSeq;
    const int seqlen = arg_int(argc, argv, "--S", env_int("S", default_seq));
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
    params.dkv_path =
        check ? dkv::kDkvPathReferenceCorrectness : dkv::kDkvPathWaspProbe;
    params.block_threads = dkv::DkvTileD128Mq32Nk128::kThreadsPerCta;
    params.sync_after_launch = 1;

    const size_t tensor_elems =
        static_cast<size_t>(batch) * static_cast<size_t>(heads) *
        static_cast<size_t>(seqlen) * static_cast<size_t>(dim);
    const size_t tensor_bytes = tensor_elems * sizeof(__half);
    const size_t workspace_bytes = shaobo_fa3_bwd_workspace_bytes(&params);

    __half* q_dev = nullptr;
    __half* k_dev = nullptr;
    __half* v_dev = nullptr;
    __half* dout_dev = nullptr;
    float* dk_dev = nullptr;
    float* dv_dev = nullptr;
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
    const size_t output_bytes = tensor_elems * sizeof(float);
    if (check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dk_dev), output_bytes);
    }
    if (check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dv_dev), output_bytes);
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
    if (check) {
        fill_dkv_inputs(q_host, k_host, v_host, dout_host);
        ignore_hip_status(
            hipMemcpy(q_dev, q_host.data(), tensor_bytes,
                      hipMemcpyHostToDevice));
        ignore_hip_status(
            hipMemcpy(k_dev, k_host.data(), tensor_bytes,
                      hipMemcpyHostToDevice));
        ignore_hip_status(
            hipMemcpy(v_dev, v_host.data(), tensor_bytes,
                      hipMemcpyHostToDevice));
        ignore_hip_status(
            hipMemcpy(dout_dev, dout_host.data(), tensor_bytes,
                      hipMemcpyHostToDevice));
        ignore_hip_status(hipMemset(dk_dev, 0, output_bytes));
        ignore_hip_status(hipMemset(dv_dev, 0, output_bytes));
    } else {
        ignore_hip_status(hipMemset(q_dev, 0, tensor_bytes));
        ignore_hip_status(hipMemset(k_dev, 0, tensor_bytes));
        ignore_hip_status(hipMemset(v_dev, 0, tensor_bytes));
        ignore_hip_status(hipMemset(dout_dev, 0, tensor_bytes));
    }
    ignore_hip_status(
        hipMemset(workspace, 0,
                  std::max<size_t>(workspace_bytes, sizeof(float))));
    params.workspace = workspace;
    params.workspace_bytes = workspace_bytes;

    const int status =
        shaobo_fa3_bwd(dout_dev, q_dev, k_dev, v_dev, nullptr, nullptr,
                       nullptr, nullptr, dk_dev, dv_dev, &params);
    if (check) {
        std::vector<float> dk_actual(tensor_elems);
        std::vector<float> dv_actual(tensor_elems);
        std::vector<float> dk_expected(tensor_elems);
        std::vector<float> dv_expected(tensor_elems);
        if (status == SHAOBO_FA3_STATUS_SUCCESS) {
            ignore_hip_status(
                hipMemcpy(dk_actual.data(), dk_dev, output_bytes,
                          hipMemcpyDeviceToHost));
            ignore_hip_status(
                hipMemcpy(dv_actual.data(), dv_dev, output_bytes,
                          hipMemcpyDeviceToHost));
            cpu_reference_dkv(
                q_host, k_host, v_host, dout_host, dk_expected, dv_expected,
                batch, heads, seqlen, dim, params.causal,
                params.softmax_scale);
        }
        const DkvCompareMetrics dk_metrics =
            compare_vectors(dk_actual, dk_expected);
        const DkvCompareMetrics dv_metrics =
            compare_vectors(dv_actual, dv_expected);
        const bool pass =
            status == SHAOBO_FA3_STATUS_SUCCESS &&
            dk_metrics.bad_count == 0 && dv_metrics.bad_count == 0 &&
            dk_metrics.max_abs <= 5.0e-4f &&
            dv_metrics.max_abs <= 5.0e-4f &&
            dk_metrics.rel_l2 <= 1.0e-4f &&
            dv_metrics.rel_l2 <= 1.0e-4f;

        ignore_hip_status(hipFree(workspace));
        ignore_hip_status(hipFree(dv_dev));
        ignore_hip_status(hipFree(dk_dev));
        ignore_hip_status(hipFree(dout_dev));
        ignore_hip_status(hipFree(v_dev));
        ignore_hip_status(hipFree(k_dev));
        ignore_hip_status(hipFree(q_dev));

        std::printf(
            "fa3_bwd_dkv_correctness status=%s B=%d H=%d S=%d D=%d "
            "workspace_bytes=%zu dk_max_abs=%g dk_mean_abs=%g dk_rmse=%g "
            "dk_rel_l2=%g dv_max_abs=%g dv_mean_abs=%g dv_rmse=%g "
            "dv_rel_l2=%g bad=%d pass=%d\n",
            shaobo_fa3_status_string(status), batch, heads, seqlen, dim,
            workspace_bytes, dk_metrics.max_abs, dk_metrics.mean_abs,
            dk_metrics.rmse, dk_metrics.rel_l2, dv_metrics.max_abs,
            dv_metrics.mean_abs, dv_metrics.rmse, dv_metrics.rel_l2,
            dk_metrics.bad_count + dv_metrics.bad_count, pass ? 1 : 0);
        return pass ? 0 : 1;
    }

    float first_diag = 0.0f;
    if (workspace_bytes >= sizeof(float)) {
        err = hipMemcpy(&first_diag, workspace, sizeof(float),
                        hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            std::fprintf(stderr, "hipMemcpy workspace failed: %s\n",
                         hipGetErrorString(err));
        }
    }

    ignore_hip_status(hipFree(workspace));
    ignore_hip_status(hipFree(dv_dev));
    ignore_hip_status(hipFree(dk_dev));
    ignore_hip_status(hipFree(dout_dev));
    ignore_hip_status(hipFree(v_dev));
    ignore_hip_status(hipFree(k_dev));
    ignore_hip_status(hipFree(q_dev));

    std::printf(
        "fa3_bwd_dkv_probe status=%s B=%d H=%d S=%d D=%d "
        "workspace_bytes=%zu first_diag=%g\n",
        shaobo_fa3_status_string(status), batch, heads, seqlen, dim,
        workspace_bytes, first_diag);
    return status == SHAOBO_FA3_STATUS_SUCCESS ? 0 : 1;
}
#endif
