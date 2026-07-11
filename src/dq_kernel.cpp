#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_contract.h"
#include "shaobo_fa3_api.h"
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

namespace dq = shaobo::fa3::bwd::dq;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kDefaultBatch = 1;
constexpr int kDefaultHeads = 1;
constexpr int kDefaultSeq = 128;
constexpr int kDefaultDim = 128;
constexpr float kLog2E = 1.44269504088896340736f;

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

__device__ __forceinline__ int workitem_x() {
#if defined(__gfx946__) || defined(__gfx92a__)
    return static_cast<int>(__builtin_amdgcn_workitem_id_x());
#else
    return static_cast<int>(threadIdx.x);
#endif
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

template <typename Tile>
struct DqLdsLayout {
    static constexpr int kBlockMq = Tile::kBlockMq;
    static constexpr int kBlockNk = Tile::kBlockNk;
    static constexpr int kHeadDim = Tile::kHeadDim;
    static constexpr int kHalfBytes = Tile::kHalfBytes;
    static constexpr int kMatrixBlockBytes = 32 * 32 * kHalfBytes;
    static constexpr int kQBase = 0;
    static constexpr int kQBytes = kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kDoutBase = kQBase + kQBytes;
    static constexpr int kDoutBytes = kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kQDoBytes = kDoutBase + kDoutBytes;
    static constexpr int kPageBase = kQDoBytes;
    static constexpr int kKPageBytes = kBlockNk * kHeadDim * kHalfBytes;
    static constexpr int kVPageBytes = kBlockNk * kHeadDim * kHalfBytes;
    static constexpr int kPageBytes = kKPageBytes + kVPageBytes;
    static constexpr int kPages = 2;
    static constexpr int kPage0Base = kPageBase;
    static constexpr int kPage1Base = kQBase;
    static constexpr int kSidecarBase = kPage0Base;
    static constexpr int kSidecarRows = kBlockMq;
    static constexpr int kSidecarBytes = 3 * kSidecarRows * sizeof(float);
    static constexpr int kQBlockOffset(int m32_block, int d32_block) {
        return (m32_block * (kHeadDim / 32) + d32_block) * kMatrixBlockBytes;
    }
    static constexpr int kKBase() {
        return kPageBase;
    }
    static constexpr int kVBase() {
        return kKBase() + kKPageBytes;
    }
    static constexpr int kPageBaseFor(int page) {
        return page == 0 ? kPage0Base : kPage1Base;
    }
    static constexpr int kPageKBase(int page) {
        return kPageBaseFor(page);
    }
    static constexpr int kPageVBase(int page) {
        return kPageBaseFor(page) + kKPageBytes;
    }
    static constexpr int kStartupBytes = kQDoBytes + kSidecarBytes;
    static constexpr int kPage0End = kPage0Base + kPageBytes;
    static constexpr int kPage1End = kPage1Base + kPageBytes;
    static constexpr int kSteadyBytes =
        kPage0End > kPage1End ? kPage0End : kPage1End;
    static constexpr int kBytes =
        kStartupBytes > kSteadyBytes ? kStartupBytes : kSteadyBytes;
    static_assert(kBytes <= Tile::kLdsBudgetBytes,
                  "canonical dQ LDS plan must fit 128KB");
};

__device__ __forceinline__ void dq_zero_f32x4(ins::F32x4& acc) {
    ins::zero_vgpr2(acc.u64[0]);
    ins::zero_vgpr2(acc.u64[1]);
}

template <typename Tile>
__device__ __forceinline__ void dq_load_q_dout_group(
    const __half* __restrict__ q,
    const __half* __restrict__ dout,
    __half* __restrict__ lds,
    int producer_group,
    int producer_wave,
    int q_base_tile,
    int64_t qkv_base) {
    constexpr int Dim = Tile::kHeadDim;
    constexpr int MatrixBlockBytes = DqLdsLayout<Tile>::kMatrixBlockBytes;
    const int d_base = producer_wave * 32;

    const int m32_base = producer_group * 2;
#pragma unroll
    for (int local_m32 = 0; local_m32 < 2; ++local_m32) {
        const int m32 = m32_base + local_m32;
        const int qrow = q_base_tile + m32 * 32;
        const int lds_offset =
            DqLdsLayout<Tile>::kQBlockOffset(m32, producer_wave);

        ins::Vec4U32 q_src = ins::prepare_matrix_src(
            q + qkv_base + static_cast<int64_t>(qrow) * Dim + d_base, Dim);
        ins::matrix_load_32x32_b16_bps_lds(
            lds + DqLdsLayout<Tile>::kQBase / sizeof(__half), q_src,
            lds_offset, true);

        ins::Vec4U32 dout_src = ins::prepare_matrix_src(
            dout + qkv_base + static_cast<int64_t>(qrow) * Dim + d_base,
            Dim);
        ins::matrix_load_32x32_b16_bps_lds(
            lds + DqLdsLayout<Tile>::kDoutBase / sizeof(__half), dout_src,
            lds_offset, true);
    }
}

template <typename Tile>
__device__ __forceinline__ float* dq_sidecar_lds(
    __half* __restrict__ lds) {
    return reinterpret_cast<float*>(
        lds + DqLdsLayout<Tile>::kSidecarBase / sizeof(__half));
}

template <typename Tile>
__device__ __forceinline__ const float* dq_sidecar_lds(
    const __half* __restrict__ lds) {
    return reinterpret_cast<const float*>(
        lds + DqLdsLayout<Tile>::kSidecarBase / sizeof(__half));
}

template <typename Tile>
__device__ __forceinline__ void dq_load_sidecar_group(
    const float* __restrict__ scores_max,
    const float* __restrict__ scores_sum,
    const float* __restrict__ delta,
    __half* __restrict__ lds,
    int producer_group,
    int producer_wave,
    int lane,
    int q_base_tile,
    int64_t row_base) {
    constexpr int Rows = Tile::kRowsPerConsumerGroup;
    constexpr int SidecarRows = DqLdsLayout<Tile>::kSidecarRows;
    float* sidecar = dq_sidecar_lds<Tile>(lds);
    const int producer_lane = producer_wave * 64 + lane;
    const int group_row_base = producer_group * Rows;
    for (int idx = producer_lane; idx < 3 * Rows; idx += 4 * 64) {
        const int field = idx / Rows;
        const int local_row = idx - field * Rows;
        const int sidecar_idx = field * SidecarRows + group_row_base + local_row;
        const int64_t row =
            row_base + q_base_tile + group_row_base + local_row;
        if (field == 0) {
            sidecar[sidecar_idx] = scores_max[row];
        } else if (field == 1) {
            sidecar[sidecar_idx] = scores_sum[row];
        } else {
            sidecar[sidecar_idx] = delta[row];
        }
    }
}

template <typename Tile>
__device__ __forceinline__ void dq_load_k_tile_page(
    const __half* __restrict__ k,
    __half* __restrict__ lds,
    int producer_wave,
    int page,
    int k_base_tile,
    int64_t qkv_base) {
    constexpr int Dim = Tile::kHeadDim;
    constexpr int MatrixBlockBytes = DqLdsLayout<Tile>::kMatrixBlockBytes;
    const int d_base = producer_wave * 32;
    const int k_page_base = DqLdsLayout<Tile>::kPageKBase(page);

#pragma unroll
    for (int n32 = 0; n32 < Tile::kBlockNk / 32; ++n32) {
        ins::Vec4U32 k_src = ins::prepare_matrix_src(
            k + qkv_base +
                static_cast<int64_t>(k_base_tile + n32 * 32) * Dim + d_base,
            Dim);
        ins::matrix_load_32x32_b16_bps_lds(
            lds + k_page_base / sizeof(__half), k_src,
            (n32 * (Dim / 32) + producer_wave) * MatrixBlockBytes, true);
    }
}

template <typename Tile>
__device__ __forceinline__ void dq_load_v_tile_page(
    const __half* __restrict__ v,
    __half* __restrict__ lds,
    int producer_wave,
    int page,
    int k_base_tile,
    int64_t qkv_base) {
    constexpr int Dim = Tile::kHeadDim;
    constexpr int MatrixBlockBytes = DqLdsLayout<Tile>::kMatrixBlockBytes;
    const int d_base = producer_wave * 32;
    const int v_page_base = DqLdsLayout<Tile>::kPageVBase(page);

#pragma unroll
    for (int n32 = 0; n32 < Tile::kBlockNk / 32; ++n32) {
        ins::Vec4U32 v_src = ins::prepare_matrix_src(
            v + qkv_base +
                static_cast<int64_t>(k_base_tile + n32 * 32) * Dim + d_base,
            Dim);
        ins::matrix_load_32x32_b16_bps_lds(
            lds + v_page_base / sizeof(__half), v_src,
            (n32 * (Dim / 32) + producer_wave) * MatrixBlockBytes, true);
    }
}

template <typename Tile>
__device__ __forceinline__ void dq_store_m16_full_d_to_global(
    float* __restrict__ dq_out,
    ins::F32x4 (&dq_reg)[8],
    int lane,
    int q_base_tile,
    int local_m16,
    int seqlen,
    int bh) {
    constexpr int Dim = Tile::kHeadDim;
    const int lane_mq = lane % 16;
    const int lane_d = lane / 16;
#pragma unroll
    for (int d_idx = 0; d_idx < Dim / 16; ++d_idx) {
        const int qrow = q_base_tile + local_m16 * 16 + lane_mq;
        const int d = d_idx * 16 + lane_d * 4;
        const int64_t out_base =
            (static_cast<int64_t>(bh) * seqlen + qrow) * Dim + d;
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            dq_out[out_base + vec_id] = dq_reg[d_idx].scalar[vec_id];
        }
    }
}

template <typename Tile>
__device__ __forceinline__ void dq_update_from_ds_pair(
    const __half* __restrict__ lds,
    int page,
    int n_tile,
    const ins::Vec4F16& ds_vec0,
    const ins::Vec4F16& ds_vec1,
    ins::F32x4 (&dq_reg)[8]) {
    constexpr int MatrixBlockBytes = DqLdsLayout<Tile>::kMatrixBlockBytes;
    const __half* k_lds =
        lds + DqLdsLayout<Tile>::kPageKBase(page) / sizeof(__half);

    ins::F16x8 k_norm0[Tile::kHeadDim / 32];
    ins::F16x8 k_norm1[Tile::kHeadDim / 32];
#pragma unroll
    for (int d_block = 0; d_block < Tile::kHeadDim / 32; ++d_block) {
        const int block = n_tile * (Tile::kHeadDim / 32) + d_block;
        ins::ds_read_matrix_normal_pair(
            k_lds, block * MatrixBlockBytes,
            k_norm0[d_block].f16x8, k_norm1[d_block].f16x8);
    }
    ins::wait_lgkm(4);

#pragma unroll
    for (int d_block = 0; d_block < Tile::kHeadDim / 64; ++d_block) {
        dq_reg[d_block * 2 + 0].f32 =
            ins::mmac_f16_lit(ds_vec0, k_norm0[d_block].f16x4[0],
                              dq_reg[d_block * 2 + 0].f32);
        dq_reg[d_block * 2 + 1].f32 =
            ins::mmac_f16_lit(ds_vec0, k_norm0[d_block].f16x4[1],
                              dq_reg[d_block * 2 + 1].f32);
        dq_reg[d_block * 2 + 0].f32 =
            ins::mmac_f16_lit(ds_vec1, k_norm1[d_block].f16x4[0],
                              dq_reg[d_block * 2 + 0].f32);
        dq_reg[d_block * 2 + 1].f32 =
            ins::mmac_f16_lit(ds_vec1, k_norm1[d_block].f16x4[1],
                              dq_reg[d_block * 2 + 1].f32);
    }
    ins::wait_lgkm(0);

#pragma unroll
    for (int d_block = Tile::kHeadDim / 64;
         d_block < Tile::kHeadDim / 32; ++d_block) {
        dq_reg[d_block * 2 + 0].f32 =
            ins::mmac_f16_lit(ds_vec0, k_norm0[d_block].f16x4[0],
                              dq_reg[d_block * 2 + 0].f32);
        dq_reg[d_block * 2 + 1].f32 =
            ins::mmac_f16_lit(ds_vec0, k_norm0[d_block].f16x4[1],
                              dq_reg[d_block * 2 + 1].f32);
        dq_reg[d_block * 2 + 0].f32 =
            ins::mmac_f16_lit(ds_vec1, k_norm1[d_block].f16x4[0],
                              dq_reg[d_block * 2 + 0].f32);
        dq_reg[d_block * 2 + 1].f32 =
            ins::mmac_f16_lit(ds_vec1, k_norm1[d_block].f16x4[1],
                              dq_reg[d_block * 2 + 1].f32);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_wait_page_filled(int page, int& phase0, int& phase1);

template <typename Bar>
__device__ __forceinline__ void dq_arrive_page_used(int page);

template <typename Bar>
__device__ __forceinline__ void dq_wait_qdo_filled(int& phase);

template <typename Bar>
__device__ __forceinline__ void dq_arrive_qdo_filled();

template <typename Bar>
__device__ __forceinline__ void dq_arrive_qdo_latched();

template <typename Tile, typename Bar, int ConsumerGroup>
__device__ __forceinline__ void dq_consumer_full3gemm_role(
    __half* __restrict__ lds,
    float* __restrict__ dq_out,
    int q_base_tile,
    int seqlen,
    int bh,
    int diag_store,
    float softmax_scale,
    float softmax_scale_log2,
    int wave_local) {
    constexpr int Dim = Tile::kHeadDim;
    constexpr int Nk = Tile::kBlockNk;
    constexpr int KBlocks = Dim / 32;
    constexpr int MatrixBlockBytes = DqLdsLayout<Tile>::kMatrixBlockBytes;
    const int lane = static_cast<int>(threadIdx.x % 64);
    const int lane_mq = lane % 16;
    const int lane_n = lane / 16;
    const int local_m16 = ConsumerGroup * 4 + wave_local;
    const int m32_block = local_m16 / 2;
    const int mhalf_offset = (local_m16 & 1) == 0 ? 0 : 1024;
    const int local_row = local_m16 * 16 + lane_mq;
    const int qrow = q_base_tile + local_row;
    const int q_tile_end =
        q_base_tile + Tile::kBlockMq < seqlen ? q_base_tile + Tile::kBlockMq
                                              : seqlen;
    const int active_k_tiles = (q_tile_end + Nk - 1) / Nk;

    const __half* q_lds = lds + DqLdsLayout<Tile>::kQBase / sizeof(__half);
    const __half* dout_lds =
        lds + DqLdsLayout<Tile>::kDoutBase / sizeof(__half);
    int filled_phase0 = 0;
    int filled_phase1 = 0;
    int qdo_filled_phase = 0;
    dq_wait_qdo_filled<Bar>(qdo_filled_phase);

    constexpr int SidecarRows = DqLdsLayout<Tile>::kSidecarRows;
    const int sidecar_vec_base = local_row & ~3;
    const int sidecar_vec_idx = local_row & 3;
    const volatile float* sidecar = dq_sidecar_lds<Tile>(lds);
    const float row_max = sidecar[sidecar_vec_base + sidecar_vec_idx];
    const float row_sum =
        sidecar[SidecarRows + sidecar_vec_base + sidecar_vec_idx];
    const float row_delta =
        sidecar[2 * SidecarRows + sidecar_vec_base + sidecar_vec_idx];

    ins::F16x8 q_reg[KBlocks];
    ins::F16x8 dout_reg[KBlocks];
#pragma unroll
    for (int d_block = 0; d_block < KBlocks; ++d_block) {
        const int lds_offset =
            DqLdsLayout<Tile>::kQBlockOffset(m32_block, d_block) +
            mhalf_offset;
        ins::ds_read_matrix_32x16_trans(q_lds, lds_offset,
                                        q_reg[d_block].f16x8);
        ins::ds_read_matrix_32x16_trans(dout_lds, lds_offset,
                                        dout_reg[d_block].f16x8);
    }
    ins::wait_lgkm(0);
    dq_arrive_qdo_latched<Bar>();

    ins::F32x4 dq_reg[8];
#pragma unroll
    for (int d_idx = 0; d_idx < 8; ++d_idx) {
        dq_zero_f32x4(dq_reg[d_idx]);
    }
    ins::F32x4 mmac_zero;
    dq_zero_f32x4(mmac_zero);

    for (int kt = 0; kt < active_k_tiles; ++kt) {
        const int page = kt & 1;
        dq_wait_page_filled<Bar>(page, filled_phase0, filled_phase1);
        const int k_base_tile = kt * Nk;
        const __half* k_lds =
            lds + DqLdsLayout<Tile>::kPageKBase(page) / sizeof(__half);
        const __half* v_lds =
            lds + DqLdsLayout<Tile>::kPageVBase(page) / sizeof(__half);
#pragma unroll
        for (int n_tile = 0; n_tile < Nk / 32; ++n_tile) {
            ins::F16x8 k_frag0[KBlocks];
            ins::F16x8 k_frag1[KBlocks];
            ins::F16x8 v_frag0[KBlocks];
            ins::F16x8 v_frag1[KBlocks];
#pragma unroll
            for (int d_block = 0; d_block < KBlocks; ++d_block) {
                const int block = n_tile * KBlocks + d_block;
                ins::ds_read_matrix_trans_pair(
                    k_lds, block * MatrixBlockBytes,
                    k_frag0[d_block].f16x8, k_frag1[d_block].f16x8);
                ins::ds_read_matrix_trans_pair(
                    v_lds, block * MatrixBlockBytes,
                    v_frag0[d_block].f16x8, v_frag1[d_block].f16x8);
            }

            ins::F32x4 qk_acc0;
            ins::F32x4 qk_acc1;
            ins::F32x4 dp_acc0;
            ins::F32x4 dp_acc1;
            ins::wait_lgkm(8);
            qk_acc0.f32 = ins::mmac_f16_lit(
                q_reg[0].f16x4[0], k_frag0[0].f16x4[0], mmac_zero.f32);
            qk_acc1.f32 = ins::mmac_f16_lit(
                q_reg[0].f16x4[0], k_frag1[0].f16x4[0], mmac_zero.f32);
            dp_acc0.f32 = ins::mmac_f16_lit(
                dout_reg[0].f16x4[0], v_frag0[0].f16x4[0], mmac_zero.f32);
            dp_acc1.f32 = ins::mmac_f16_lit(
                dout_reg[0].f16x4[0], v_frag1[0].f16x4[0], mmac_zero.f32);
#pragma unroll
            for (int k_half = 1; k_half < 2; ++k_half) {
                qk_acc0.f32 = ins::mmac_f16_lit(
                    q_reg[0].f16x4[k_half], k_frag0[0].f16x4[k_half],
                    qk_acc0.f32);
                qk_acc1.f32 = ins::mmac_f16_lit(
                    q_reg[0].f16x4[k_half], k_frag1[0].f16x4[k_half],
                    qk_acc1.f32);
                dp_acc0.f32 = ins::mmac_f16_lit(
                    dout_reg[0].f16x4[k_half], v_frag0[0].f16x4[k_half],
                    dp_acc0.f32);
                dp_acc1.f32 = ins::mmac_f16_lit(
                    dout_reg[0].f16x4[k_half], v_frag1[0].f16x4[k_half],
                    dp_acc1.f32);
            }
#pragma unroll
            for (int d_block = 1; d_block < KBlocks / 2; ++d_block) {
#pragma unroll
                for (int k_half = 0; k_half < 2; ++k_half) {
                    qk_acc0.f32 = ins::mmac_f16_lit(
                        q_reg[d_block].f16x4[k_half],
                        k_frag0[d_block].f16x4[k_half], qk_acc0.f32);
                    qk_acc1.f32 = ins::mmac_f16_lit(
                        q_reg[d_block].f16x4[k_half],
                        k_frag1[d_block].f16x4[k_half], qk_acc1.f32);
                    dp_acc0.f32 = ins::mmac_f16_lit(
                        dout_reg[d_block].f16x4[k_half],
                        v_frag0[d_block].f16x4[k_half], dp_acc0.f32);
                    dp_acc1.f32 = ins::mmac_f16_lit(
                        dout_reg[d_block].f16x4[k_half],
                        v_frag1[d_block].f16x4[k_half], dp_acc1.f32);
                }
            }
            ins::wait_lgkm(0);
#pragma unroll
            for (int d_block = KBlocks / 2; d_block < KBlocks; ++d_block) {
#pragma unroll
                for (int k_half = 0; k_half < 2; ++k_half) {
                    qk_acc0.f32 = ins::mmac_f16_lit(
                        q_reg[d_block].f16x4[k_half],
                        k_frag0[d_block].f16x4[k_half], qk_acc0.f32);
                    qk_acc1.f32 = ins::mmac_f16_lit(
                        q_reg[d_block].f16x4[k_half],
                        k_frag1[d_block].f16x4[k_half], qk_acc1.f32);
                    dp_acc0.f32 = ins::mmac_f16_lit(
                        dout_reg[d_block].f16x4[k_half],
                        v_frag0[d_block].f16x4[k_half], dp_acc0.f32);
                    dp_acc1.f32 = ins::mmac_f16_lit(
                        dout_reg[d_block].f16x4[k_half],
                        v_frag1[d_block].f16x4[k_half], dp_acc1.f32);
                }
            }

            ins::Vec4F16 ds_vec0;
            ins::Vec4F16 ds_vec1;
#pragma unroll
            for (int vec_id = 0; vec_id < 4; ++vec_id) {
                const int nk0 = n_tile * 32 + lane_n * 4 + vec_id;
                const int krow0 = k_base_tile + nk0;
                const int valid0 = krow0 <= qrow;
                const float p0 =
                    exp2f((qk_acc0.scalar[vec_id] - row_max) *
                          softmax_scale_log2) /
                    row_sum;
                const float ds_value0 =
                    p0 * (dp_acc0.scalar[vec_id] - row_delta) *
                    softmax_scale * static_cast<float>(valid0);
                ds_vec0[vec_id] = static_cast<_Float16>(ds_value0);

                const int nk1 = n_tile * 32 + 16 + lane_n * 4 + vec_id;
                const int krow1 = k_base_tile + nk1;
                const int valid1 = krow1 <= qrow;
                const float p1 =
                    exp2f((qk_acc1.scalar[vec_id] - row_max) *
                          softmax_scale_log2) /
                    row_sum;
                const float ds_value1 =
                    p1 * (dp_acc1.scalar[vec_id] - row_delta) *
                    softmax_scale * static_cast<float>(valid1);
                ds_vec1[vec_id] = static_cast<_Float16>(ds_value1);
            }
            dq_update_from_ds_pair<Tile>(
                lds, page, n_tile, ds_vec0, ds_vec1, dq_reg);
        }
        dq_arrive_page_used<Bar>(page);
    }

    if (diag_store == 0) {
        dq_store_m16_full_d_to_global<Tile>(
            dq_out, dq_reg, lane, q_base_tile, local_m16, seqlen, bh);
    }
#pragma unroll
    for (int d_idx = 0; d_idx < 8; ++d_idx) {
        ins::keep_accumulator_live(dq_reg[d_idx]);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_seq_page_filled(int page) {
    if (page == 0) {
        ins::abarrier_seq<false>(Bar::kPage0Filled);
    } else {
        ins::abarrier_seq<false>(Bar::kPage1Filled);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_arrive_page_filled(int page) {
    if (page == 0) {
        ins::abarrier_arrive_cnt<false>(Bar::kPage0Filled, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Bar::kPage1Filled, 1);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_wait_page_filled(int page,
                                                    int& phase0,
                                                    int& phase1) {
    if (page == 0) {
        ins::abarrier_try_wait<true>(Bar::kPage0Filled, phase0);
    } else {
        ins::abarrier_try_wait<true>(Bar::kPage1Filled, phase1);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_wait_page_used(int page,
                                                  int& phase0,
                                                  int& phase1) {
    if (page == 0) {
        ins::abarrier_try_wait<true>(Bar::kPage0Used, phase0);
    } else {
        ins::abarrier_try_wait<true>(Bar::kPage1Used, phase1);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_arrive_page_used(int page) {
    if (page == 0) {
        ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Bar::kPage1Used, 1);
    }
}

template <typename Bar>
__device__ __forceinline__ void dq_wait_qdo_latched(int& phase) {
    ins::abarrier_try_wait<true>(Bar::kQDoLatched, phase);
}

template <typename Bar>
__device__ __forceinline__ void dq_wait_qdo_filled(int& phase) {
    ins::abarrier_try_wait<true>(Bar::kQDoFilled, phase);
}

template <typename Bar>
__device__ __forceinline__ void dq_arrive_qdo_filled() {
    ins::abarrier_arrive_cnt<false>(Bar::kQDoFilled, 1);
}

template <typename Bar>
__device__ __forceinline__ void dq_arrive_qdo_latched() {
    ins::abarrier_arrive_cnt<false>(Bar::kQDoLatched, 1);
}

__global__ void __launch_bounds__(1024, 1)
    __attribute__((hcu_wdra_waves_per_tg(16)))
fa3_bwd_dq_kernel(const __half* __restrict__ q,
                  const __half* __restrict__ k,
                  const __half* __restrict__ v,
                  const __half* __restrict__ dout,
                  const float* __restrict__ scores_max,
                  const float* __restrict__ scores_sum,
                  const float* __restrict__ delta,
                  float* __restrict__ dq_out,
                  int batch,
                  int heads,
                  int seqlen,
                  float softmax_scale,
                  float softmax_scale_log2,
                  int diag_store,
                  int q_tile_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dq::ActiveDqTile;
    using Bar = dq::DqBarrierLedger;
    constexpr int BlockMq = Tile::kBlockMq;
    constexpr int Nk = Tile::kBlockNk;
    constexpr int Dim = Tile::kHeadDim;
    __shared__ __half lds[DqLdsLayout<Tile>::kBytes / sizeof(__half)];

    (void)batch;
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kPage0Filled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kPage0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kPage1Filled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kPage1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQDoFilled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQDoLatched, 8);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int q_tile = static_cast<int>(blockIdx.x) + q_tile_offset;
    const int h = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    const int q_base_tile = q_tile * BlockMq;
    const int bh = b * heads + h;
    const int64_t row_base = static_cast<int64_t>(bh) * seqlen;
    const int64_t qkv_base = row_base * Dim;
    const int q_tile_end =
        q_base_tile + BlockMq < seqlen ? q_base_tile + BlockMq : seqlen;
    const int active_k_tiles = (q_tile_end + Nk - 1) / Nk;
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(40);
        const int lane = static_cast<int>(threadIdx.x % 64);
        int used_phase0 = 0;
        int used_phase1 = 0;
        int qdo_latched_phase = 0;
        dq_load_q_dout_group<Tile>(
            q, dout, lds, 0, wave_local, q_base_tile, qkv_base);
        dq_load_sidecar_group<Tile>(
            scores_max, scores_sum, delta, lds, 0, wave_local, lane,
            q_base_tile, row_base);
        ins::wait_vmem_lgkm();
        dq_seq_page_filled<Bar>(0);
        if (active_k_tiles > 1) {
            dq_seq_page_filled<Bar>(1);
        }
        ins::maybe_wait_bps_vbcnt_before_arrive();
        dq_arrive_qdo_filled<Bar>();
        for (int kt = 0; kt < active_k_tiles; ++kt) {
            const int page = kt & 1;
            const int k_base_tile = kt * Nk;
            if (kt == 0) {
                dq_wait_qdo_latched<Bar>(qdo_latched_phase);
            } else if (kt > 1) {
                dq_wait_page_used<Bar>(page, used_phase0, used_phase1);
                dq_seq_page_filled<Bar>(page);
            }
            dq_load_k_tile_page<Tile>(
                k, lds, wave_local, page, k_base_tile, qkv_base);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            dq_arrive_page_filled<Bar>(page);
        }
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(dq::WdraResourceWindows::kConsumerTargetVgprs);
        dq_consumer_full3gemm_role<Tile, Bar, 0>(
            lds, dq_out, q_base_tile, seqlen, bh, diag_store,
            softmax_scale, softmax_scale_log2, wave_local);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(dq::WdraResourceWindows::kConsumerTargetVgprs);
        dq_consumer_full3gemm_role<Tile, Bar, 1>(
            lds, dq_out, q_base_tile, seqlen, bh, diag_store,
            softmax_scale, softmax_scale_log2, wave_local);
    } else {
        __builtin_hcu_s_set_vgpr_size(40);
        const int lane = static_cast<int>(threadIdx.x % 64);
        int used_phase0 = 0;
        int used_phase1 = 0;
        int qdo_latched_phase = 0;
        dq_load_q_dout_group<Tile>(
            q, dout, lds, 1, wave_local, q_base_tile, qkv_base);
        dq_load_sidecar_group<Tile>(
            scores_max, scores_sum, delta, lds, 1, wave_local, lane,
            q_base_tile, row_base);
        ins::wait_vmem_lgkm();
        ins::maybe_wait_bps_vbcnt_before_arrive();
        dq_arrive_qdo_filled<Bar>();
        for (int kt = 0; kt < active_k_tiles; ++kt) {
            const int page = kt & 1;
            const int k_base_tile = kt * Nk;
            if (kt == 0) {
                dq_wait_qdo_latched<Bar>(qdo_latched_phase);
            } else if (kt > 1) {
                dq_wait_page_used<Bar>(page, used_phase0, used_phase1);
            }
            dq_load_v_tile_page<Tile>(
                v, lds, wave_local, page, k_base_tile, qkv_base);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            dq_arrive_page_filled<Bar>(page);
        }
    }

    __syncthreads();
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kPage0Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage0Used);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1Used);
        __builtin_hcu_s_abarrier_inv(Bar::kQDoFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kQDoLatched);
    }
    if (diag_store != 0) {
        __syncthreads();
        const int tid = workitem_x();
        if (tid < 64) {
            dq_out[tid] = 1.0f;
        }
    }
#else
    (void)q;
    (void)k;
    (void)v;
    (void)dout;
    (void)scores_max;
    (void)scores_sum;
    (void)delta;
    (void)dq_out;
    (void)batch;
    (void)heads;
    (void)seqlen;
    (void)softmax_scale;
    (void)softmax_scale_log2;
    (void)diag_store;
#endif
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
    float actual_l2 = 0.0f;
    float expected_l2 = 0.0f;
    int actual_nonzero = 0;
    int expected_nonzero = 0;
    int actual_nonfinite = 0;
    int expected_nonfinite = 0;
    size_t first_bad_index = static_cast<size_t>(-1);
    size_t last_bad_index = static_cast<size_t>(-1);
    float first_bad_actual = 0.0f;
    float first_bad_expected = 0.0f;
    float last_bad_actual = 0.0f;
    float last_bad_expected = 0.0f;
    int bad_rows = 0;
    int first_bad_rows[8] = {};
    uint32_t bad_row_mod32_mask = 0;
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

void cpu_softmax_aux_delta(const std::vector<__half>& q,
                           const std::vector<__half>& k,
                           const std::vector<__half>& v,
                           const std::vector<__half>& dout,
                           std::vector<float>& scores_max,
                           std::vector<float>& scores_sum,
                           std::vector<float>& delta,
                           int batch,
                           int heads,
                           int seqlen,
                           int dim,
                           int causal,
                           float scale) {
    const float scale_log2 = scale * kLog2E;
    std::fill(scores_max.begin(), scores_max.end(), 0.0f);
    std::fill(scores_sum.begin(), scores_sum.end(), 1.0f);
    std::fill(delta.begin(), delta.end(), 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q_idx = 0; q_idx < seqlen; ++q_idx) {
                const int row = (b * heads + h) * seqlen + q_idx;
                float max_qk = -std::numeric_limits<float>::infinity();
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
                    max_qk = std::max(max_qk, dot);
                }

                double denom = 0.0;
                std::vector<float> prob_num(seqlen, 0.0f);
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
                    const float p_num =
                        std::exp2((dot - max_qk) * scale_log2);
                    prob_num[k_idx] = p_num;
                    denom += p_num;
                }

                const float inv_sum =
                    denom != 0.0 ? 1.0f / static_cast<float>(denom) : 0.0f;
                scores_max[row] = max_qk;
                scores_sum[row] = static_cast<float>(denom);

                float delta_row = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float out_d = 0.0f;
                    for (int k_idx = 0; k_idx < seqlen; ++k_idx) {
                        if (causal && k_idx > q_idx) {
                            continue;
                        }
                        out_d += prob_num[k_idx] * inv_sum *
                                 __half2float(
                                     v[tensor_offset(
                                         b, h, k_idx, d, heads, seqlen, dim)]);
                    }
                    delta_row +=
                        __half2float(
                            dout[tensor_offset(
                                b, h, q_idx, d, heads, seqlen, dim)]) *
                        out_d;
                }
                delta[row] = delta_row;
            }
        }
    }
}

DqCompareMetrics compare_vectors(const std::vector<float>& actual,
                                 const std::vector<float>& expected,
                                 int dim) {
    DqCompareMetrics m;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_actual_sq = 0.0;
    double sum_ref_sq = 0.0;
    int last_recorded_bad_row = -1;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            const int bad_row = static_cast<int>(i / static_cast<size_t>(dim));
            if (bad_row != last_recorded_bad_row) {
                if (m.bad_rows < 8) {
                    m.first_bad_rows[m.bad_rows] = bad_row;
                }
                ++m.bad_rows;
                m.bad_row_mod32_mask |= (1u << (bad_row & 31));
                last_recorded_bad_row = bad_row;
            }
            if (m.first_bad_index == static_cast<size_t>(-1)) {
                m.first_bad_index = i;
                m.first_bad_actual = a;
                m.first_bad_expected = e;
            }
            m.last_bad_index = i;
            m.last_bad_actual = a;
            m.last_bad_expected = e;
            if (!std::isfinite(a)) {
                ++m.actual_nonfinite;
            }
            if (!std::isfinite(e)) {
                ++m.expected_nonfinite;
            }
            ++m.bad_count;
            continue;
        }
        const float diff = std::fabs(a - e);
        m.max_abs = std::max(m.max_abs, diff);
        sum_abs += diff;
        sum_sq += static_cast<double>(diff) * diff;
        sum_actual_sq += static_cast<double>(a) * a;
        sum_ref_sq += static_cast<double>(e) * e;
        if (std::fabs(a) > 1.0e-12f) {
            ++m.actual_nonzero;
        }
        if (std::fabs(e) > 1.0e-12f) {
            ++m.expected_nonzero;
        }
    }
    const double n = static_cast<double>(actual.size());
    m.mean_abs = static_cast<float>(sum_abs / std::max(1.0, n));
    m.rmse = static_cast<float>(std::sqrt(sum_sq / std::max(1.0, n)));
    m.actual_l2 = static_cast<float>(std::sqrt(sum_actual_sq));
    m.expected_l2 = static_cast<float>(std::sqrt(sum_ref_sq));
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
    (void)dk;
    (void)dv;

    if (params == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->dq_path == dq::kDqPathCanonicalDq) {
        if (!valid_canonical_shape(params)) {
            return SHAOBO_FA3_STATUS_UNSUPPORTED;
        }
        if (dout == nullptr || q == nullptr || k == nullptr ||
            v == nullptr || softmax_aux0 == nullptr ||
            softmax_aux1 == nullptr || dq_out == nullptr ||
            params->reserved_ptr[0] == nullptr) {
            return SHAOBO_FA3_STATUS_INVALID_VALUE;
        }
        using Tile = dq::ActiveDqTile;
        const dim3 grid(params->seqlen_q / Tile::kBlockMq,
                        params->num_heads_q, params->batch);
        const dim3 block(Tile::kThreadsPerCta);
        const int q_tiles = static_cast<int>(grid.x);
        const int tiles_per_dispatch =
            params->reserved_i32[1] > 0 ? params->reserved_i32[1] : 16;
        for (int q_tile_offset = 0; q_tile_offset < q_tiles;
             q_tile_offset += tiles_per_dispatch) {
            const int chunk_tiles =
                std::min(tiles_per_dispatch, q_tiles - q_tile_offset);
            const dim3 chunk_grid(
                static_cast<unsigned int>(chunk_tiles), grid.y, grid.z);
            fa3_bwd_dq_kernel<<<chunk_grid, block>>>(
                static_cast<const __half*>(q), static_cast<const __half*>(k),
                static_cast<const __half*>(v),
                static_cast<const __half*>(dout),
                static_cast<const float*>(softmax_aux0),
                static_cast<const float*>(softmax_aux1),
                static_cast<const float*>(params->reserved_ptr[0]),
                static_cast<float*>(dq_out), params->batch,
                params->num_heads_q, params->seqlen_q, params->softmax_scale,
                params->softmax_scale * kLog2E, params->reserved_i32[0],
                q_tile_offset);
        }
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
    (void)softmax_aux0;
    (void)softmax_aux1;
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
    const int canonical =
        arg_int(argc, argv, "--canonical", env_int("CANONICAL_DQ", 0));
    const int diag_store =
        arg_int(argc, argv, "--diag-store", env_int("DQ_DIAG_STORE", 0));
    const int tiles_per_dispatch = arg_int(
        argc, argv, "--tiles-per-dispatch",
        env_int("DQ_TILES_PER_DISPATCH", 0));

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
    params.dq_path = canonical != 0 ? dq::kDqPathCanonicalDq
                                    : dq::kDqPathReferenceCorrectness;
    params.block_threads = dq::ActiveDqTile::kThreadsPerCta;
    params.sync_after_launch = 1;
    params.reserved_i32[0] = diag_store;
    params.reserved_i32[1] = tiles_per_dispatch;

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
    float* scores_max_dev = nullptr;
    float* scores_sum_dev = nullptr;
    float* delta_dev = nullptr;

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
    const size_t sidecar_rows =
        static_cast<size_t>(batch) * static_cast<size_t>(heads) *
        static_cast<size_t>(seqlen);
    const size_t sidecar_bytes = sidecar_rows * sizeof(float);
    if (canonical != 0 && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&scores_max_dev),
                        sidecar_bytes);
    }
    if (canonical != 0 && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&scores_sum_dev),
                        sidecar_bytes);
    }
    if (canonical != 0 && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&delta_dev), sidecar_bytes);
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
    if (canonical != 0) {
        std::vector<float> scores_max_host(sidecar_rows);
        std::vector<float> scores_sum_host(sidecar_rows);
        std::vector<float> delta_host(sidecar_rows);
        cpu_softmax_aux_delta(
            q_host, k_host, v_host, dout_host, scores_max_host,
            scores_sum_host, delta_host, batch, heads, seqlen, dim,
            params.causal, params.softmax_scale);
        ignore_hip_status(
            hipMemcpy(scores_max_dev, scores_max_host.data(), sidecar_bytes,
                      hipMemcpyHostToDevice));
        ignore_hip_status(
            hipMemcpy(scores_sum_dev, scores_sum_host.data(), sidecar_bytes,
                      hipMemcpyHostToDevice));
        ignore_hip_status(
            hipMemcpy(delta_dev, delta_host.data(), sidecar_bytes,
                      hipMemcpyHostToDevice));
        params.reserved_ptr[0] = delta_dev;
    }

    const int status = shaobo_fa3_bwd(
        dout_dev, q_dev, k_dev, v_dev, nullptr, scores_max_dev,
        scores_sum_dev, dq_dev, nullptr, nullptr, &params);

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
        compare_vectors(dq_actual, dq_expected, dim);
    const float l2_ratio =
        dq_metrics.expected_l2 > 1.0e-12f
            ? dq_metrics.actual_l2 / dq_metrics.expected_l2
            : 1.0f;
    const size_t first_bad = dq_metrics.first_bad_index;
    const int first_bad_d =
        first_bad != static_cast<size_t>(-1)
            ? static_cast<int>(first_bad % static_cast<size_t>(dim))
            : -1;
    const int first_bad_row =
        first_bad != static_cast<size_t>(-1)
            ? static_cast<int>((first_bad / static_cast<size_t>(dim)) %
                               static_cast<size_t>(seqlen))
            : -1;
    const size_t last_bad = dq_metrics.last_bad_index;
    const int last_bad_d =
        last_bad != static_cast<size_t>(-1)
            ? static_cast<int>(last_bad % static_cast<size_t>(dim))
            : -1;
    const int last_bad_row =
        last_bad != static_cast<size_t>(-1)
            ? static_cast<int>((last_bad / static_cast<size_t>(dim)) %
                               static_cast<size_t>(seqlen))
            : -1;
    const bool canonical_path = params.dq_path == dq::kDqPathCanonicalDq;
    const bool reference_pass =
        dq_metrics.max_abs <= 5.0e-4f && dq_metrics.rel_l2 <= 1.0e-4f;
    const bool canonical_pass =
        dq_metrics.max_abs <= 5.0e-4f && dq_metrics.rmse <= 5.0e-5f &&
        l2_ratio >= 0.80f && l2_ratio <= 1.20f;
    const bool pass = status == SHAOBO_FA3_STATUS_SUCCESS &&
                      dq_metrics.bad_count == 0 &&
                      (canonical_path ? canonical_pass : reference_pass);

    ignore_hip_status(hipFree(delta_dev));
    ignore_hip_status(hipFree(scores_sum_dev));
    ignore_hip_status(hipFree(scores_max_dev));
    ignore_hip_status(hipFree(workspace));
    ignore_hip_status(hipFree(dq_dev));
    ignore_hip_status(hipFree(dout_dev));
    ignore_hip_status(hipFree(v_dev));
    ignore_hip_status(hipFree(k_dev));
    ignore_hip_status(hipFree(q_dev));

    std::printf(
        "%s status=%s B=%d H=%d S=%d D=%d causal=%d workspace_bytes=%zu "
        "path=%s dq_max_abs=%g dq_mean_abs=%g dq_rmse=%g dq_rel_l2=%g "
        "actual_l2=%g expected_l2=%g l2_ratio=%g actual_nz=%d expected_nz=%d "
        "actual_nonfinite=%d expected_nonfinite=%d "
        "first_bad_index=%zu first_bad_row=%d first_bad_d=%d "
        "first_bad_actual=%g first_bad_expected=%g "
        "last_bad_index=%zu last_bad_row=%d last_bad_d=%d "
        "last_bad_actual=%g last_bad_expected=%g "
        "bad_rows=%d bad_row_mod32_mask=0x%08x "
        "first_bad_rows=%d,%d,%d,%d,%d,%d,%d,%d "
        "sample0=%g/%g sample1=%g/%g bad=%d pass=%d\n",
        "fa3_bwd_dq_correctness", shaobo_fa3_status_string(status), batch,
        heads, seqlen, dim, params.causal, workspace_bytes,
        canonical != 0 ? "canonical" : "reference",
        dq_metrics.max_abs, dq_metrics.mean_abs, dq_metrics.rmse,
        dq_metrics.rel_l2, dq_metrics.actual_l2, dq_metrics.expected_l2,
        l2_ratio, dq_metrics.actual_nonzero, dq_metrics.expected_nonzero,
        dq_metrics.actual_nonfinite, dq_metrics.expected_nonfinite,
        first_bad, first_bad_row, first_bad_d, dq_metrics.first_bad_actual,
        dq_metrics.first_bad_expected,
        last_bad, last_bad_row, last_bad_d, dq_metrics.last_bad_actual,
        dq_metrics.last_bad_expected,
        dq_metrics.bad_rows, dq_metrics.bad_row_mod32_mask,
        dq_metrics.first_bad_rows[0], dq_metrics.first_bad_rows[1],
        dq_metrics.first_bad_rows[2], dq_metrics.first_bad_rows[3],
        dq_metrics.first_bad_rows[4], dq_metrics.first_bad_rows[5],
        dq_metrics.first_bad_rows[6], dq_metrics.first_bad_rows[7],
        !dq_actual.empty() ? dq_actual[0] : 0.0f,
        !dq_expected.empty() ? dq_expected[0] : 0.0f,
        dq_actual.size() > 1 ? dq_actual[1] : 0.0f,
        dq_expected.size() > 1 ? dq_expected[1] : 0.0f,
        dq_metrics.bad_count, pass ? 1 : 0);
    return pass ? 0 : 1;
}
#endif
