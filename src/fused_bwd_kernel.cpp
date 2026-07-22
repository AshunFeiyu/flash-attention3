#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "fused_bwd_contract.h"
#include "shaobo_fa3_api.h"
#include "shaobo_instr.h"

#include <cstdint>

namespace fused = shaobo::fa3::bwd::fused_bwd;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Tile = fused::ActiveFusedBwdContract;
using Bar = fused::FusedBwdBarrierLedger;
using Wdra = fused::FusedBwdWdraResourceWindow;

constexpr float kLog2E = 1.44269504088896340736f;
constexpr int kMatrixBlockBytes = 32 * 32 * sizeof(__half);
constexpr int kMatrixBlocksD = Tile::kHeadDim / 32;
constexpr int kQRowBlocks = Tile::kMq / 32;
constexpr int kKRowBlocks = Tile::kNk / 32;

struct LdsLayout {
    static constexpr int kQBase = 0;
    static constexpr int kQBytes = Tile::kMq * Tile::kHeadDim * sizeof(__half);
    static constexpr int kDoutBase = kQBase + kQBytes;
    static constexpr int kDoutBytes = kQBytes;
    static constexpr int kKBase = kDoutBase + kDoutBytes;
    static constexpr int kKBytes = Tile::kNk * Tile::kHeadDim * sizeof(__half);
    static constexpr int kVBase = kKBase + kKBytes;
    static constexpr int kVBytes = kKBytes;
    static constexpr int kScratchBase = kVBase + kVBytes;
    static constexpr int kScratchBytes = Tile::kPdsPageBytes;
    static constexpr int kSidecarBase = kScratchBase + kScratchBytes;
    static constexpr int kBytes = kSidecarBase + Tile::kSidecarBytes;

    static_assert(kQBytes == 16 * 1024 && kKBytes == 32 * 1024,
                  "fused input tile sizes changed");
    static_assert(kBytes == Tile::kPlannedLdsBytes,
                  "implementation and contract LDS ledgers disagree");
};

union Fragment {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union Accumulator {
    ins::Vec4F32 f32;
    float scalar[4];
};

__device__ __forceinline__ int matrix_block_offset(int row_block,
                                                   int d_block) {
    return (row_block * kMatrixBlocksD + d_block) * kMatrixBlockBytes;
}

__device__ __forceinline__ int m16_matrix_offset(int m_block, int d_block) {
    return matrix_block_offset(m_block >> 1, d_block) +
           (m_block & 1) * Tile::kWriterPageBytes / 2;
}

__device__ __forceinline__ int scratch_page_offset(int generation,
                                                   int n_block) {
    return LdsLayout::kScratchBase +
           generation * Tile::kPdsGenerationBytes +
           n_block * Tile::kWriterPageBytes;
}

__device__ __forceinline__ const __half* scratch_generation(
    const __half* lds, int generation) {
    return reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + LdsLayout::kScratchBase +
        generation * Tile::kPdsGenerationBytes);
}

__device__ __forceinline__ float* sidecar_field(__half* lds, int field) {
    auto* base = reinterpret_cast<float*>(
        reinterpret_cast<char*>(lds) + LdsLayout::kSidecarBase);
    return base + field * Tile::kMq;
}

__device__ __forceinline__ void source_slot_qk(int lane,
                                               int word,
                                               int& qrow,
                                               int& krow) {
    const int dst_lane =
        ((lane >> 0) & 1) | (((lane >> 1) & 1) << 1) |
        (((lane >> 2) & 1) << 2) | (((lane >> 3) & 1) << 3) |
        (((lane >> 5) & 1) << 4) | (((word >> 1) & 1) << 5);
    const int dst_word =
        ((word >> 0) & 1) | (((lane >> 4) & 1) << 1) |
        (((word >> 2) & 1) << 2);
    qrow = dst_lane & 15;
    krow = (dst_word >= 4 ? 16 : 0) + (dst_lane >> 4) * 4 +
           (dst_word & 3);
}

__device__ __forceinline__ void producer_load_resident(
    const __half* k,
    const __half* v,
    __half* lds,
    int64_t tensor_base,
    int k_base,
    int wave_local) {
#pragma unroll
    for (int n_block = 0; n_block < kKRowBlocks; ++n_block) {
        const int d_base = wave_local * 32;
        const int64_t input =
            tensor_base + static_cast<int64_t>(k_base + n_block * 32) *
                              Tile::kHeadDim +
            d_base;
        const int offset = matrix_block_offset(n_block, wave_local);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(k + input, Tile::kHeadDim),
            LdsLayout::kKBase + offset, true);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(v + input, Tile::kHeadDim),
            LdsLayout::kVBase + offset, true);
    }
}

__device__ __forceinline__ void producer_load_raw(
    const __half* q,
    const __half* dout,
    const float* packed_sidecar,
    __half* lds,
    int64_t tensor_base,
    int64_t row_base,
    int q_base,
    int wave_local,
    int lane) {
#pragma unroll
    for (int m_block = 0; m_block < kQRowBlocks; ++m_block) {
        const int d_base = wave_local * 32;
        const int64_t input =
            tensor_base + static_cast<int64_t>(q_base + m_block * 32) *
                              Tile::kHeadDim +
            d_base;
        const int offset = matrix_block_offset(m_block, wave_local);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(q + input, Tile::kHeadDim),
            LdsLayout::kQBase + offset, true);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(dout + input, Tile::kHeadDim),
            LdsLayout::kDoutBase + offset, true);
    }

    if (lane < 16) {
        const int local_row = wave_local * 16 + lane;
        const float* row =
            packed_sidecar + (row_base + q_base + local_row) * 3;
        sidecar_field(lds, 0)[local_row] = row[0];
        sidecar_field(lds, 1)[local_row] = row[1];
        sidecar_field(lds, 2)[local_row] = row[2];
    }
}

__device__ __forceinline__ void f16_mmac_pair(const Fragment& lhs,
                                               const Fragment& rhs0,
                                               const Fragment& rhs1,
                                               Fragment& out,
                                               bool first_d_block) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    const ins::Vec4F16 acc0 =
        first_d_block ? zero.f16x4[0] : out.f16x4[0];
    const ins::Vec4F16 acc1 =
        first_d_block ? zero.f16x4[0] : out.f16x4[1];
    out.f16x4[0] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs0.f16x4[0], acc0, 0, 0);
    out.f16x4[0] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[1], rhs0.f16x4[1], out.f16x4[0], 0, 0);
    out.f16x4[1] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs1.f16x4[0], acc1, 0, 0);
    out.f16x4[1] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[1], rhs1.f16x4[1], out.f16x4[1], 0, 0);
}

// Logical GEMMs 1 and 2: one M16xN32 score/dP panel, computed once.
__device__ __forceinline__ void score_dp_stage(const __half* lds,
                                               int m_block,
                                               int n_block,
                                               Fragment& score,
                                               Fragment& dp) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
        Fragment q_frag{};
        Fragment dout_frag{};
        Fragment k0{};
        Fragment k1{};
        Fragment v0{};
        Fragment v1{};
        const int q_offset = m16_matrix_offset(m_block, d_block);
        const int kv_offset = matrix_block_offset(n_block, d_block);
        ins::ds_read_matrix_32x16_trans(
            lds, LdsLayout::kQBase + q_offset, q_frag.f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, LdsLayout::kDoutBase + q_offset, dout_frag.f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, LdsLayout::kKBase + kv_offset, k0.f16x8, k1.f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, LdsLayout::kVBase + kv_offset, v0.f16x8, v1.f16x8);
        ins::wait_lgkm(0);
        f16_mmac_pair(q_frag, k0, k1, score, d_block == 0);
        f16_mmac_pair(dout_frag, v0, v1, dp, d_block == 0);
    }
}

__device__ __forceinline__ void softmax_ds_stage(
    const Fragment& score,
    const Fragment& dp,
    int lane,
    int q_base,
    int m_block,
    int k_base,
    int n_block,
    int causal,
    float row_max_log2,
    float row_inv_sum,
    float row_delta,
    float softmax_scale,
    Fragment& p,
    Fragment& ds) {
    const float scale_log2 = softmax_scale * kLog2E;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        int local_q = 0;
        int local_k = 0;
        source_slot_qk(lane, word, local_q, local_k);
        const int qrow = q_base + m_block * Tile::kMqPerPanel + local_q;
        const int krow = k_base + n_block * Tile::kNkPerDkvWave + local_k;
        const float probability =
            (!causal || krow <= qrow)
                ? exp2f(static_cast<float>(score.scalar[word]) * scale_log2 -
                        row_max_log2) *
                      row_inv_sum
                : 0.0f;
        p.scalar[word] = static_cast<_Float16>(probability);
        ds.scalar[word] = static_cast<_Float16>(
            probability * (static_cast<float>(dp.scalar[word]) - row_delta) *
            softmax_scale);
    }
}

__device__ __forceinline__ void zero_dkv_accumulators(
    Accumulator (&dv_acc)[16], Accumulator (&dk_acc)[16]) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        dv_acc[i].f32 = zero.f32;
        dk_acc[i].f32 = zero.f32;
    }
}

// Logical GEMMs 3 and 4: persistent N32 dV/dK ownership across the q-loop.
__device__ __forceinline__ void update_dv_dk_stage(
    const Fragment& p_normal,
    const Fragment& ds_normal,
    const Fragment (&dout_normal)[kMatrixBlocksD],
    const Fragment (&q_normal)[kMatrixBlocksD],
    Accumulator (&dv_acc)[16],
    Accumulator (&dk_acc)[16]) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
        for (int n_half = 0; n_half < 2; ++n_half) {
#pragma unroll
            for (int d_half = 0; d_half < 2; ++d_half) {
                const int out = n_half * 8 + d_block * 2 + d_half;
                dv_acc[out].f32 = ins::mmac_f16_lit(
                    p_normal.f16x4[n_half],
                    dout_normal[d_block].f16x4[d_half], dv_acc[out].f32);
                dk_acc[out].f32 = ins::mmac_f16_lit(
                    ds_normal.f16x4[n_half],
                    q_normal[d_block].f16x4[d_half], dk_acc[out].f32);
            }
        }
    }
}

template <int Generation>
__device__ __forceinline__ void produce_dkv_panel(
    const __half* lds,
    __half* mutable_lds,
    int q_base,
    int m_block,
    int k_base,
    int n_block,
    int causal,
    float softmax_scale,
    int lane,
    Accumulator (&dv_acc)[16],
    Accumulator (&dk_acc)[16]) {
    static_assert(Generation == 0 || Generation == 1);
    constexpr int kFilled =
        Generation == 0 ? Bar::kDsFilled0 : Bar::kDsFilled1;
    const int sidecar_row = m_block * Tile::kMqPerPanel + (lane & 15);
    const float row_max_log2 = sidecar_field(mutable_lds, 0)[sidecar_row];
    const float row_inv_sum = sidecar_field(mutable_lds, 1)[sidecar_row];
    const float row_delta = sidecar_field(mutable_lds, 2)[sidecar_row];

    Fragment score{};
    Fragment dp{};
    Fragment p{};
    Fragment ds{};
    score_dp_stage(lds, m_block, n_block, score, dp);
    softmax_ds_stage(score, dp, lane, q_base, m_block, k_base, n_block,
                     causal, row_max_log2, row_inv_sum, row_delta,
                     softmax_scale, p, ds);

    const int page = scratch_page_offset(Generation, n_block);
    ins::abarrier_seq<false>(kFilled);
    ins::ds_write_matrix_32x16_trans_f16(p.f16x8, mutable_lds, page);
    ins::wait_lgkm(0);
    Fragment p_normal{};
    ins::ds_read_matrix_32x16_normal(lds, page, p_normal.f16x8);
    ins::wait_lgkm(0);

    ins::ds_write_matrix_32x16_trans_f16(ds.f16x8, mutable_lds, page);
    ins::wait_lgkm(0);
    Fragment ds_normal{};
    ins::ds_read_matrix_32x16_normal(lds, page, ds_normal.f16x8);

    Fragment q_normal[kMatrixBlocksD];
    Fragment dout_normal[kMatrixBlocksD];
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
        const int raw_offset = m16_matrix_offset(m_block, d_block);
        ins::ds_read_matrix_32x16_normal(
            lds, LdsLayout::kQBase + raw_offset,
            q_normal[d_block].f16x8);
        ins::ds_read_matrix_32x16_normal(
            lds, LdsLayout::kDoutBase + raw_offset,
            dout_normal[d_block].f16x8);
    }
    ins::wait_lgkm(0);
    update_dv_dk_stage(p_normal, ds_normal, dout_normal, q_normal, dv_acc,
                       dk_acc);
    ins::abarrier_arrive_cnt<false>(kFilled, 1);
}

__device__ __forceinline__ void store_dkv_outputs(
    float* dk,
    float* dv,
    int64_t tensor_base,
    int k_base,
    int n_block,
    int lane,
    const Accumulator (&dk_acc)[16],
    const Accumulator (&dv_acc)[16]) {
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
        const int row = k_base + n_block * Tile::kNkPerDkvWave +
                        n_half * 16 + (lane & 15);
#pragma unroll
        for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
            for (int d_half = 0; d_half < 2; ++d_half) {
                const int out = n_half * 8 + d_block * 2 + d_half;
                const int col = d_block * 32 + d_half * 16 +
                                (lane >> 4) * 4;
                auto* dk_ptr = reinterpret_cast<ins::Vec4F32*>(
                    dk + tensor_base + static_cast<int64_t>(row) *
                                               Tile::kHeadDim +
                    col);
                auto* dv_ptr = reinterpret_cast<ins::Vec4F32*>(
                    dv + tensor_base + static_cast<int64_t>(row) *
                                               Tile::kHeadDim +
                    col);
                *dk_ptr = dk_acc[out].f32;
                *dv_ptr = dv_acc[out].f32;
            }
        }
    }
}

__device__ __forceinline__ void run_dkv_owner(
    const __half* lds,
    __half* mutable_lds,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int k_base,
    int q_tile_begin,
    int q_tile_count,
    int causal,
    float softmax_scale,
    int n_block,
    int lane) {
    Accumulator dv_acc[16];
    Accumulator dk_acc[16];
    zero_dkv_accumulators(dv_acc, dk_acc);
    int raw_phase = 0;
    int used_phase0 = 0;
    int used_phase1 = 0;

#pragma clang loop unroll(disable)
    for (int qi = 0; qi < q_tile_count; ++qi) {
        ins::abarrier_try_wait<true>(Bar::kRawFilled, raw_phase);
        const int q_base = (q_tile_begin + qi) * Tile::kMq;
        if (qi != 0) {
            ins::abarrier_try_wait<true>(Bar::kDsUsed0, used_phase0);
        }
        produce_dkv_panel<0>(lds, mutable_lds, q_base, 0, k_base, n_block,
                             causal, softmax_scale, lane, dv_acc, dk_acc);
        if (qi != 0) {
            ins::abarrier_try_wait<true>(Bar::kDsUsed1, used_phase1);
        }
        produce_dkv_panel<1>(lds, mutable_lds, q_base, 1, k_base, n_block,
                             causal, softmax_scale, lane, dv_acc, dk_acc);
        ins::abarrier_try_wait<true>(Bar::kDsUsed0, used_phase0);
        produce_dkv_panel<0>(lds, mutable_lds, q_base, 2, k_base, n_block,
                             causal, softmax_scale, lane, dv_acc, dk_acc);
        ins::abarrier_try_wait<true>(Bar::kDsUsed1, used_phase1);
        produce_dkv_panel<1>(lds, mutable_lds, q_base, 3, k_base, n_block,
                             causal, softmax_scale, lane, dv_acc, dk_acc);
        ins::abarrier_arrive_cnt<false>(Bar::kRawUsed, 1);
    }

    store_dkv_outputs(dk, dv, tensor_base, k_base, n_block, lane, dk_acc,
                      dv_acc);
}

__device__ __forceinline__ void atomic_store_dq_d32(
    float* dq,
    int64_t tensor_base,
    int q_base,
    int m_block,
    int d_block,
    int lane,
    const Accumulator (&dq_acc)[2]) {
    const int row = q_base + m_block * Tile::kMqPerPanel + (lane & 15);
#pragma unroll
    for (int d_half = 0; d_half < 2; ++d_half) {
        const int col = d_block * Tile::kHeadDimPerDqWave + d_half * 16 +
                        (lane >> 4) * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            atomicAdd(dq + tensor_base +
                          static_cast<int64_t>(row) * Tile::kHeadDim + col + i,
                      dq_acc[d_half].scalar[i]);
        }
    }
}

// Logical GEMM 5: one D32 owner consumes all four N32 dS pages.
template <int Generation>
__device__ __forceinline__ void consume_dq_panel(
    const __half* lds,
    float* dq,
    int64_t tensor_base,
    int q_base,
    int m_block,
    int d_block,
    int lane,
    int& filled_phase) {
    static_assert(Generation == 0 || Generation == 1);
    constexpr int kFilled =
        Generation == 0 ? Bar::kDsFilled0 : Bar::kDsFilled1;
    constexpr int kUsed =
        Generation == 0 ? Bar::kDsUsed0 : Bar::kDsUsed1;
    ins::abarrier_try_wait<true>(kFilled, filled_phase);

    Fragment ds[4];
    const __half* generation = scratch_generation(lds, Generation);
    ins::ds_read_matrix_32x16_trans_imm4<
        0 * Tile::kWriterPageBytes, 1 * Tile::kWriterPageBytes,
        2 * Tile::kWriterPageBytes, 3 * Tile::kWriterPageBytes>(
        generation, ds[0].f16x8, ds[1].f16x8, ds[2].f16x8,
        ds[3].f16x8);

    Fragment k0[4];
    Fragment k1[4];
#pragma unroll
    for (int n_block = 0; n_block < 4; ++n_block) {
        ins::ds_read_matrix_normal_pair(
            lds, LdsLayout::kKBase + matrix_block_offset(n_block, d_block),
            k0[n_block].f16x8, k1[n_block].f16x8);
    }
    ins::wait_lgkm(0);
    ins::abarrier_arrive_cnt<false>(kUsed, 1);

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    Accumulator dq_acc[2];
    dq_acc[0].f32 = zero.f32;
    dq_acc[1].f32 = zero.f32;
#pragma unroll
    for (int n_block = 0; n_block < 4; ++n_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            dq_acc[d_half].f32 = ins::mmac_f16_lit(
                ds[n_block].f16x4[0], k0[n_block].f16x4[d_half],
                dq_acc[d_half].f32);
            dq_acc[d_half].f32 = ins::mmac_f16_lit(
                ds[n_block].f16x4[1], k1[n_block].f16x4[d_half],
                dq_acc[d_half].f32);
        }
    }
    atomic_store_dq_d32(dq, tensor_base, q_base, m_block, d_block, lane,
                        dq_acc);
}

__device__ __forceinline__ void run_dq_owner(const __half* lds,
                                             float* dq,
                                             int64_t tensor_base,
                                             int q_tile_begin,
                                             int q_tile_count,
                                             int d_block,
                                             int lane) {
    int filled_phase0 = 0;
    int filled_phase1 = 0;
#pragma clang loop unroll(disable)
    for (int qi = 0; qi < q_tile_count; ++qi) {
        const int q_base = (q_tile_begin + qi) * Tile::kMq;
        consume_dq_panel<0>(lds, dq, tensor_base, q_base, 0, d_block, lane,
                            filled_phase0);
        consume_dq_panel<1>(lds, dq, tensor_base, q_base, 1, d_block, lane,
                            filled_phase1);
        consume_dq_panel<0>(lds, dq, tensor_base, q_base, 2, d_block, lane,
                            filled_phase0);
        consume_dq_panel<1>(lds, dq, tensor_base, q_base, 3, d_block, lane,
                            filled_phase1);
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(Tile::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(Tile::kWavesPerCta)))
fa3_bwd_5gemm_kernel(const __half* __restrict__ dout,
                     const __half* __restrict__ q,
                     const __half* __restrict__ k,
                     const __half* __restrict__ v,
                     const float* __restrict__ packed_sidecar,
                     float* __restrict__ dq,
                     float* __restrict__ dk,
                     float* __restrict__ dv,
                     int heads,
                     int seqlen,
                     int causal,
                     float softmax_scale) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(Wdra::kProducerVgprs, Wdra::kDkvVgprs,
                            Wdra::kDqVgprs);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[LdsLayout::kBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDsFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDsUsed0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDsFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDsUsed1, 4);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < Tile::kDkvWaveBegin) {
        __builtin_hcu_s_set_vgpr_size(Wdra::kProducerVgprs);
        const int wave_local = wave & 3;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int bh = static_cast<int>(blockIdx.y) * heads +
                       static_cast<int>(blockIdx.x);
        const int k_base = static_cast<int>(blockIdx.z) * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t row_base = static_cast<int64_t>(bh) * seqlen;
        const int64_t tensor_base = row_base * Tile::kHeadDim;

        ins::abarrier_seq<false>(Bar::kResidentFilled);
        producer_load_resident(k, v, lds, tensor_base, k_base, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled, 1);

        int raw_used_phase = 0;
        for (int qi = 0; qi < q_tile_count; ++qi) {
            if (qi != 0) {
                ins::abarrier_try_wait<true>(Bar::kRawUsed, raw_used_phase);
            }
            ins::abarrier_seq<false>(Bar::kRawFilled);
            producer_load_raw(q, dout, packed_sidecar, lds, tensor_base,
                              row_base, (q_tile_begin + qi) * Tile::kMq,
                              wave_local, lane);
            ins::wait_vmem_lgkm();
            ins::maybe_wait_bps_vbcnt_before_arrive();
            ins::abarrier_arrive_cnt<false>(Bar::kRawFilled, 1);
        }
        if (q_tile_count != 0) {
            ins::abarrier_try_wait<true>(Bar::kRawUsed, raw_used_phase);
        }
    } else if (wave < Tile::kDqWaveBegin) {
        __builtin_hcu_s_set_vgpr_size(Wdra::kDkvVgprs);
        const int n_block = wave - Tile::kDkvWaveBegin;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int bh = static_cast<int>(blockIdx.y) * heads +
                       static_cast<int>(blockIdx.x);
        const int k_base = static_cast<int>(blockIdx.z) * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t tensor_base =
            static_cast<int64_t>(bh) * seqlen * Tile::kHeadDim;
        int resident_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled, resident_phase);
        run_dkv_owner(lds, lds, dk, dv, tensor_base, k_base, q_tile_begin,
                      q_tile_count, causal, softmax_scale, n_block, lane);
    } else {
        __builtin_hcu_s_set_vgpr_size(Wdra::kDqVgprs);
        const int d_block = wave - Tile::kDqWaveBegin;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int bh = static_cast<int>(blockIdx.y) * heads +
                       static_cast<int>(blockIdx.x);
        const int k_base = static_cast<int>(blockIdx.z) * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t tensor_base =
            static_cast<int64_t>(bh) * seqlen * Tile::kHeadDim;
        int resident_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled, resident_phase);
        run_dq_owner(lds, dq, tensor_base, q_tile_begin, q_tile_count, d_block,
                     lane);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kRawUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kDsFilled0);
        __builtin_hcu_s_abarrier_inv(Bar::kDsUsed0);
        __builtin_hcu_s_abarrier_inv(Bar::kDsFilled1);
        __builtin_hcu_s_abarrier_inv(Bar::kDsUsed1);
    }
#else
    (void)dout;
    (void)q;
    (void)k;
    (void)v;
    (void)packed_sidecar;
    (void)dq;
    (void)dk;
    (void)dv;
    (void)heads;
    (void)seqlen;
    (void)causal;
    (void)softmax_scale;
#endif
}

extern "C" int shaobo_fa3_bwd_fused5(const void* dout,
                                      const void* q,
                                      const void* k,
                                      const void* v,
                                      const void* packed_sidecar,
                                      void* dq,
                                      void* dk,
                                      void* dv,
                                      const ShaoboFa3Params* params) {
    if (params == nullptr || dout == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || packed_sidecar == nullptr || dq == nullptr ||
        dk == nullptr || dv == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->dtype != SHAOBO_FA3_DTYPE_FP16 ||
        params->layout != SHAOBO_FA3_LAYOUT_BHSD ||
        params->head_dim_qk != Tile::kHeadDim ||
        params->head_dim_v != Tile::kHeadDim ||
        (params->causal != 0 && params->causal != 1) ||
        params->seqlen_q != params->seqlen_k ||
        params->seqlen_q % Tile::kNk != 0 ||
        params->num_heads_q != params->num_heads_kv) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }

    const dim3 grid(params->num_heads_q, params->batch,
                    params->seqlen_k / Tile::kNk);
    hipLaunchKernelGGL(
        fa3_bwd_5gemm_kernel, grid, dim3(Tile::kThreadsPerCta), 0, 0,
        static_cast<const __half*>(dout), static_cast<const __half*>(q),
        static_cast<const __half*>(k), static_cast<const __half*>(v),
        static_cast<const float*>(packed_sidecar), static_cast<float*>(dq),
        static_cast<float*>(dk), static_cast<float*>(dv),
        params->num_heads_q, params->seqlen_q, params->causal,
        params->softmax_scale);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return SHAOBO_FA3_STATUS_HIP_ERROR;
    }
    if (params->sync_after_launch != 0) {
        error = hipDeviceSynchronize();
        if (error != hipSuccess) {
            return SHAOBO_FA3_STATUS_HIP_ERROR;
        }
    }
    return SHAOBO_FA3_STATUS_SUCCESS;
}
