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

struct ResidentFragments {
    Fragment k_trans[kMatrixBlocksD];
    Fragment v_trans[kMatrixBlocksD];
    ins::Vec4F16 k_normal[Tile::kConsumerGroups *
                          Tile::kWavesPerConsumerGroup];
};

__device__ __forceinline__ int matrix_block_offset(int row_block,
                                                   int d_block) {
    return (row_block * kMatrixBlocksD + d_block) * kMatrixBlockBytes;
}

__device__ __forceinline__ int m16_matrix_offset(int m_block, int d_block) {
    return matrix_block_offset(m_block >> 1, d_block) +
           (m_block & 1) * Tile::kWriterPageBytes / 2;
}

template <int Group>
__host__ __device__ constexpr int scratch_page_offset(int writer) {
    static_assert(Group == 0 || Group == 1);
    return LdsLayout::kScratchBase +
           (Group * Tile::kWavesPerConsumerGroup + writer) *
               Tile::kWriterPageBytes;
}

template <int MBlock, int Group>
__host__ __device__ constexpr int batch_ds_group_offset() {
    static_assert(MBlock >= 0 && MBlock < Tile::kMqPanels);
    static_assert(Group == 0 || Group == 1);
    return LdsLayout::kKBase + MBlock * Tile::kPdsGenerationBytes +
           Group * Tile::kWavesPerConsumerGroup * Tile::kWriterPageBytes;
}

template <int MBlock, int Group>
__device__ __forceinline__ const __half* batch_ds_group(const __half* lds) {
    return reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) +
        batch_ds_group_offset<MBlock, Group>());
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

template <int Group>
__device__ __forceinline__ void producer_load_resident_group(
    const __half* k,
    const __half* v,
    __half* lds,
    int64_t tensor_base,
    int k_base,
    int wave_local) {
#pragma unroll
    for (int local_block = 0; local_block < 2; ++local_block) {
        const int n_block = Group * 2 + local_block;
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

__device__ __forceinline__ void f16_mmac_single(const Fragment& lhs,
                                                 const Fragment& rhs,
                                                 Fragment& out,
                                                 bool first_d_block) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    const ins::Vec4F16 acc =
        first_d_block ? zero.f16x4[0] : out.f16x4[0];
    out.f16x4[0] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs.f16x4[0], acc, 0, 0);
    out.f16x4[0] = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs.f16x4[1], rhs.f16x4[1], out.f16x4[0], 0, 0);
    if (first_d_block) {
        out.f16x4[1] = zero.f16x4[0];
    }
}

template <int DHalf>
__device__ __forceinline__ void latch_resident_fragments(
    const __half* lds,
    int n_owner,
    int d_block,
    ResidentFragments& resident) {
    static_assert(DHalf == 0 || DHalf == 1);
#pragma unroll
    for (int source_d = 0; source_d < kMatrixBlocksD; ++source_d) {
        const int kv_offset =
            matrix_block_offset(n_owner >> 1, source_d) +
            (n_owner & 1) * Tile::kWriterPageBytes / 2;
        ins::ds_read_matrix_32x16_trans(
            lds, LdsLayout::kKBase + kv_offset,
            resident.k_trans[source_d].f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, LdsLayout::kVBase + kv_offset,
            resident.v_trans[source_d].f16x8);
    }

    Fragment k_full[Tile::kConsumerGroups * Tile::kWavesPerConsumerGroup];
    const auto* k_d_base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + LdsLayout::kKBase +
        d_block * kMatrixBlockBytes);
    ins::ds_read_matrix_32x16_normal_imm4<
        0, Tile::kWriterPageBytes / 2,
        kMatrixBlocksD * kMatrixBlockBytes,
        kMatrixBlocksD * kMatrixBlockBytes + Tile::kWriterPageBytes / 2>(
        k_d_base, k_full[0].f16x8, k_full[1].f16x8,
        k_full[2].f16x8, k_full[3].f16x8);
    ins::ds_read_matrix_32x16_normal_imm4<
        2 * kMatrixBlocksD * kMatrixBlockBytes,
        2 * kMatrixBlocksD * kMatrixBlockBytes +
            Tile::kWriterPageBytes / 2,
        3 * kMatrixBlocksD * kMatrixBlockBytes,
        3 * kMatrixBlocksD * kMatrixBlockBytes +
            Tile::kWriterPageBytes / 2>(
        k_d_base, k_full[4].f16x8, k_full[5].f16x8,
        k_full[6].f16x8, k_full[7].f16x8);
    ins::wait_lgkm(0);
#pragma unroll
    for (int writer = 0;
         writer < Tile::kConsumerGroups * Tile::kWavesPerConsumerGroup;
         ++writer) {
        resident.k_normal[writer] = k_full[writer].f16x4[DHalf];
    }
}

// Logical GEMMs 1 and 2: one M16xN16 score/dP panel, computed once.
__device__ __forceinline__ void score_dp_stage(const __half* lds,
                                               int m_block,
                                               const ResidentFragments& resident,
                                               Fragment& score,
                                               Fragment& dp) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
        Fragment q_frag{};
        Fragment dout_frag{};
        const int q_offset = m16_matrix_offset(m_block, d_block);
        ins::ds_read_matrix_32x16_trans(
            lds, LdsLayout::kQBase + q_offset, q_frag.f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, LdsLayout::kDoutBase + q_offset, dout_frag.f16x8);
        ins::wait_lgkm(0);
        f16_mmac_single(q_frag, resident.k_trans[d_block], score,
                        d_block == 0);
        f16_mmac_single(dout_frag, resident.v_trans[d_block], dp,
                        d_block == 0);
    }
}

template <int RegionBase>
__device__ __forceinline__ void read_raw_panel_normal(
    const __half* lds,
    int m_block,
    Fragment (&normal)[kMatrixBlocksD]) {
    const int panel_offset = m16_matrix_offset(m_block, 0);
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + RegionBase +
        panel_offset);
    ins::ds_read_matrix_32x16_normal_imm4<0, 2048, 4096, 6144>(
        base, normal[0].f16x8, normal[1].f16x8, normal[2].f16x8,
        normal[3].f16x8);
}

__device__ __forceinline__ void softmax_ds_stage(
    const Fragment& score,
    const Fragment& dp,
    int lane,
    int q_base,
    int m_block,
    int k_base,
    int n_owner,
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
        const int krow =
            k_base + n_owner * Tile::kNkPerConsumerWave + local_k;
        const float probability =
            (local_k < Tile::kNkPerConsumerWave &&
             (!causal || krow <= qrow))
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
    Accumulator (&dv_acc)[8], Accumulator (&dk_acc)[8]) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dv_acc[i].f32 = zero.f32;
        dk_acc[i].f32 = zero.f32;
    }
}

// Logical GEMM 3: consume dO while its four matrix fragments are hot.
__device__ __forceinline__ void update_dv_stage(
    const Fragment& p_normal,
    const Fragment (&dout_normal)[kMatrixBlocksD],
    Accumulator (&dv_acc)[8]) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            const int out = d_block * 2 + d_half;
            dv_acc[out].f32 = ins::mmac_f16_lit(
                p_normal.f16x4[0],
                dout_normal[d_block].f16x4[d_half], dv_acc[out].f32);
        }
    }
}

// Logical GEMM 4: consume Q in a separate island so dO is already dead.
__device__ __forceinline__ void update_dk_stage(
    const Fragment& ds_normal,
    const Fragment (&q_normal)[kMatrixBlocksD],
    Accumulator (&dk_acc)[8]) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            const int out = d_block * 2 + d_half;
            dk_acc[out].f32 = ins::mmac_f16_lit(
                ds_normal.f16x4[0], q_normal[d_block].f16x4[d_half],
                dk_acc[out].f32);
        }
    }
}

template <int MBlock>
__device__ __forceinline__ void update_dq_batch_panel(
    const __half* lds,
    const ResidentFragments& resident,
    Accumulator& dq_acc) {
    Fragment ds_trans[Tile::kConsumerGroups *
                      Tile::kWavesPerConsumerGroup];
    const __half* group0 = batch_ds_group<MBlock, 0>(lds);
    const __half* group1 = batch_ds_group<MBlock, 1>(lds);
    ins::ds_read_matrix_32x16_trans_dual_base_imm4<
        0, Tile::kWriterPageBytes, 2 * Tile::kWriterPageBytes,
        3 * Tile::kWriterPageBytes>(
        group0, group1, ds_trans[0].f16x8, ds_trans[4].f16x8,
        ds_trans[1].f16x8, ds_trans[5].f16x8,
        ds_trans[2].f16x8, ds_trans[6].f16x8,
        ds_trans[3].f16x8, ds_trans[7].f16x8);
    ins::wait_lgkm(0);
#pragma unroll
    for (int writer = 0;
         writer < Tile::kConsumerGroups * Tile::kWavesPerConsumerGroup;
         ++writer) {
        ins::mmac_f16_lit_inplace(dq_acc.f32,
                                  ds_trans[writer].f16x4[0],
                                  resident.k_normal[writer]);
    }
}

__device__ __forceinline__ void atomic_store_dq_d16(
    float* dq,
    int64_t tensor_base,
    int q_base,
    int m_block,
    int d_owner,
    int lane,
    const Accumulator& dq_acc) {
    const int row = q_base + m_block * Tile::kMqPerPanel + (lane & 15);
    const int col = d_owner * Tile::kHeadDimPerConsumerWave +
                    (lane >> 4) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        atomicAdd(dq + tensor_base +
                      static_cast<int64_t>(row) * Tile::kHeadDim + col + i,
                  dq_acc.scalar[i]);
    }
}

template <int Group>
__device__ __forceinline__ void compute_dkv_panel(
    const __half* lds,
    __half* mutable_lds,
    int q_base,
    int m_block,
    int k_base,
    int owner,
    int causal,
    float softmax_scale,
    int lane,
    const ResidentFragments& resident,
    Fragment& retained_ds,
    Accumulator (&dv_acc)[8],
    Accumulator (&dk_acc)[8]) {
    static_assert(Group == 0 || Group == 1);
    const int n_owner = Group * Tile::kWavesPerConsumerGroup + owner;

    Fragment score{};
    Fragment dp{};
    Fragment p{};
    score_dp_stage(lds, m_block, resident, score, dp);
    const int sidecar_row =
        m_block * Tile::kMqPerPanel + (lane & 15);
    softmax_ds_stage(
        score, dp, lane, q_base, m_block, k_base, n_owner, causal,
        sidecar_field(mutable_lds, 0)[sidecar_row],
        sidecar_field(mutable_lds, 1)[sidecar_row],
        sidecar_field(mutable_lds, 2)[sidecar_row], softmax_scale, p,
        retained_ds);

    const int page = scratch_page_offset<Group>(owner);
    ins::ds_write_matrix_32x16_trans_f16(p.f16x8, mutable_lds, page);
    ins::wait_lgkm(0);
    Fragment p_normal{};
    ins::ds_read_matrix_32x16_normal(lds, page, p_normal.f16x8);
    ins::wait_lgkm(0);

    {
        Fragment dout_normal[kMatrixBlocksD];
        read_raw_panel_normal<LdsLayout::kDoutBase>(lds, m_block,
                                                    dout_normal);
        ins::wait_lgkm(0);
        update_dv_stage(p_normal, dout_normal, dv_acc);
    }

    ins::ds_write_matrix_32x16_trans_f16(retained_ds.f16x8, mutable_lds,
                                         page);
    ins::wait_lgkm(0);

    {
        Fragment ds_normal{};
        Fragment q_normal[kMatrixBlocksD];
        ins::ds_read_matrix_32x16_normal(lds, page, ds_normal.f16x8);
        read_raw_panel_normal<LdsLayout::kQBase>(lds, m_block, q_normal);
        ins::wait_lgkm(0);
        update_dk_stage(ds_normal, q_normal, dk_acc);
    }
}

template <int MBlock, int Group>
__device__ __forceinline__ void publish_batch_ds(
    const Fragment& ds,
    __half* lds,
    int owner) {
    const int page = batch_ds_group_offset<MBlock, Group>() +
                     owner * Tile::kWriterPageBytes;
    ins::ds_write_matrix_32x16_trans_f16(ds.f16x8, lds, page);
}

__device__ __forceinline__ void zero_dq_accumulators(
    Accumulator (&dq_acc)[Tile::kMqPanels]) {
    ins::F16x8 dq_zero;
    ins::zero_f16x8(dq_zero);
#pragma unroll
    for (int m_block = 0; m_block < Tile::kMqPanels; ++m_block) {
        dq_acc[m_block].f32 = dq_zero.f32;
    }
}

__device__ __forceinline__ void store_dkv_outputs(
    float* dk,
    float* dv,
    int64_t tensor_base,
    int k_base,
    int n_owner,
    int lane,
    const Accumulator (&dk_acc)[8],
    const Accumulator (&dv_acc)[8]) {
    const int row = k_base + n_owner * Tile::kNkPerConsumerWave +
                    (lane & 15);
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            const int out = d_block * 2 + d_half;
            const int col =
                d_block * 32 + d_half * 16 + (lane >> 4) * 4;
            auto* dk_ptr = reinterpret_cast<ins::Vec4F32*>(
                dk + tensor_base +
                static_cast<int64_t>(row) * Tile::kHeadDim + col);
            auto* dv_ptr = reinterpret_cast<ins::Vec4F32*>(
                dv + tensor_base +
                static_cast<int64_t>(row) * Tile::kHeadDim + col);
            *dk_ptr = dk_acc[out].f32;
            *dv_ptr = dv_acc[out].f32;
        }
    }
}

template <int Group>
__device__ __forceinline__ void run_consumer_group(
    const __half* lds,
    __half* mutable_lds,
    float* dq,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int k_base,
    int q_tile_begin,
    int q_tile_count,
    int causal,
    float softmax_scale,
    int owner,
    int lane) {
    const int n_owner = Group * Tile::kWavesPerConsumerGroup + owner;
    const int d_block = n_owner >> 1;
    const int d_half = n_owner & 1;
    ResidentFragments resident;
    if (d_half == 0) {
        latch_resident_fragments<0>(lds, n_owner, d_block, resident);
    } else {
        latch_resident_fragments<1>(lds, n_owner, d_block, resident);
    }

    int kv_ds_used_phase = 0;
    ins::abarrier_arrive_cnt<false>(Bar::kKvDsUsed, 1);
    ins::abarrier_try_wait<true>(Bar::kKvDsUsed, kv_ds_used_phase);

    Accumulator dv_acc[8];
    Accumulator dk_acc[8];
    zero_dkv_accumulators(dv_acc, dk_acc);
    int raw_phase = 0;
    int batch_filled_phase = 0;
#pragma clang loop unroll(disable)
    for (int qi = 0; qi < q_tile_count; ++qi) {
        ins::abarrier_try_wait<true>(Bar::kRawFilled, raw_phase);
        const int q_base = (q_tile_begin + qi) * Tile::kMq;

        Fragment ds_panels[Tile::kMqPanels];
        compute_dkv_panel<Group>(
            lds, mutable_lds, q_base, 0, k_base, owner, causal,
            softmax_scale, lane, resident, ds_panels[0], dv_acc, dk_acc);
        compute_dkv_panel<Group>(
            lds, mutable_lds, q_base, 1, k_base, owner, causal,
            softmax_scale, lane, resident, ds_panels[1], dv_acc, dk_acc);
        compute_dkv_panel<Group>(
            lds, mutable_lds, q_base, 2, k_base, owner, causal,
            softmax_scale, lane, resident, ds_panels[2], dv_acc, dk_acc);
        compute_dkv_panel<Group>(
            lds, mutable_lds, q_base, 3, k_base, owner, causal,
            softmax_scale, lane, resident, ds_panels[3], dv_acc, dk_acc);

        ins::abarrier_arrive_cnt<false>(Bar::kRawUsed, 1);
        if (qi != 0) {
            ins::abarrier_try_wait<true>(Bar::kKvDsUsed, kv_ds_used_phase);
        }

        publish_batch_ds<0, Group>(ds_panels[0], mutable_lds, owner);
        publish_batch_ds<1, Group>(ds_panels[1], mutable_lds, owner);
        publish_batch_ds<2, Group>(ds_panels[2], mutable_lds, owner);
        publish_batch_ds<3, Group>(ds_panels[3], mutable_lds, owner);
        ins::wait_lgkm(0);
        ins::abarrier_arrive_cnt<false>(Bar::kBatchDsFilled, 1);
        ins::abarrier_try_wait<true>(Bar::kBatchDsFilled,
                                     batch_filled_phase);

        Accumulator dq_acc[Tile::kMqPanels];
        zero_dq_accumulators(dq_acc);
        update_dq_batch_panel<0>(lds, resident, dq_acc[0]);
        update_dq_batch_panel<1>(lds, resident, dq_acc[1]);
        update_dq_batch_panel<2>(lds, resident, dq_acc[2]);
        update_dq_batch_panel<3>(lds, resident, dq_acc[3]);
        ins::abarrier_arrive_cnt<false>(Bar::kKvDsUsed, 1);

        atomic_store_dq_d16(dq, tensor_base, q_base, 0, n_owner, lane,
                            dq_acc[0]);
        atomic_store_dq_d16(dq, tensor_base, q_base, 1, n_owner, lane,
                            dq_acc[1]);
        atomic_store_dq_d16(dq, tensor_base, q_base, 2, n_owner, lane,
                            dq_acc[2]);
        atomic_store_dq_d16(dq, tensor_base, q_base, 3, n_owner, lane,
                            dq_acc[3]);
    }

    store_dkv_outputs(dk, dv, tensor_base, k_base, n_owner, lane, dk_acc,
                      dv_acc);
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
    __builtin_hcu_wdra_init(Wdra::kProducerVgprs, Wdra::kConsumer0Vgprs,
                            Wdra::kConsumer1Vgprs);
#endif
#if defined(__gfx946__)
    __shared__ __align__(2048) __half lds[LdsLayout::kBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kKvDsUsed, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed, 8);
        __builtin_hcu_s_abarrier_init(Bar::kBatchDsFilled, 8);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < Tile::kConsumer0WaveBegin) {
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

        ins::abarrier_seq<false>(Bar::kResidentFilled0);
        producer_load_resident_group<0>(k, v, lds, tensor_base, k_base,
                                        wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled0, 1);
        ins::abarrier_seq<false>(Bar::kResidentFilled1);
        producer_load_resident_group<1>(k, v, lds, tensor_base, k_base,
                                        wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled1, 1);

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
    } else if (wave < Tile::kConsumer1WaveBegin) {
        __builtin_hcu_s_set_vgpr_size(Wdra::kConsumer0Vgprs);
        const int owner = wave - Tile::kConsumer0WaveBegin;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int bh = static_cast<int>(blockIdx.y) * heads +
                       static_cast<int>(blockIdx.x);
        const int k_base = static_cast<int>(blockIdx.z) * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t tensor_base =
            static_cast<int64_t>(bh) * seqlen * Tile::kHeadDim;
        int resident0_phase = 0;
        int resident1_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled0,
                                     resident0_phase);
        ins::abarrier_try_wait<true>(Bar::kResidentFilled1,
                                     resident1_phase);
        run_consumer_group<0>(lds, lds, dq, dk, dv, tensor_base, k_base,
                              q_tile_begin, q_tile_count, causal,
                              softmax_scale, owner, lane);
    } else {
        __builtin_hcu_s_set_vgpr_size(Wdra::kConsumer1Vgprs);
        const int owner = wave - Tile::kConsumer1WaveBegin;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int bh = static_cast<int>(blockIdx.y) * heads +
                       static_cast<int>(blockIdx.x);
        const int k_base = static_cast<int>(blockIdx.z) * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t tensor_base =
            static_cast<int64_t>(bh) * seqlen * Tile::kHeadDim;
        int resident0_phase = 0;
        int resident1_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled0,
                                     resident0_phase);
        ins::abarrier_try_wait<true>(Bar::kResidentFilled1,
                                     resident1_phase);
        run_consumer_group<1>(lds, lds, dq, dk, dv, tensor_base, k_base,
                              q_tile_begin, q_tile_count, causal,
                              softmax_scale, owner, lane);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled0);
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled1);
        __builtin_hcu_s_abarrier_inv(Bar::kKvDsUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kRawUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kBatchDsFilled);
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
