#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "fused_bwd_contract.h"
#include "fused_bwd_dq_reduce.h"
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
    static constexpr int kRaw0Base = 0;
    static constexpr int kQBytes = Tile::kMq * Tile::kHeadDim * sizeof(__half);
    static constexpr int kDoutOffset = kQBytes;
    static constexpr int kDoutBytes = kQBytes;
    static constexpr int kKBase = kRaw0Base + Tile::kRawQDoBytes;
    static constexpr int kKBytes = Tile::kNk * Tile::kHeadDim * sizeof(__half);
    static constexpr int kVBase = kKBase + kKBytes;
    static constexpr int kVBytes = kKBytes;
    static constexpr int kScratchBase = kVBase;
    static constexpr int kScratchBytes = Tile::kPdsPageBytes;
    static constexpr int kSidecarBase = kScratchBase + kScratchBytes;
    static constexpr int kSidecarBytes =
        Tile::kSidecarPages * Tile::kSidecarBytes;
    static constexpr int kRaw1Base = kVBase + kVBytes;
    static constexpr int kBytes = kRaw1Base + Tile::kRawQDoBytes;

    static_assert(kQBytes == 16 * 1024 && kKBytes == 32 * 1024,
                  "fused input tile sizes changed");
    static_assert(kScratchBase + kScratchBytes + kSidecarBytes <= kRaw1Base,
                  "steady P/dS side data must not overlap raw page1");
    static_assert(kBytes == Tile::kPlannedLdsBytes,
                  "implementation and contract LDS ledgers disagree");
};

__host__ __device__ constexpr int raw_page_base(int page) {
    return page == 0 ? LdsLayout::kRaw0Base : LdsLayout::kRaw1Base;
}

union Fragment {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union Accumulator {
    ins::Vec4F32 f32;
    float scalar[4];
    uint64_t u64[2];
};

struct ResidentFragments {
    Fragment k_trans[kMatrixBlocksD];
    Fragment v_trans[kMatrixBlocksD];
};

struct DqResidentFragments {
    Fragment k_normal[Tile::kConsumerGroups *
                      Tile::kWavesPerConsumerGroup];
};

struct ProbabilityPanel {
    Fragment f16;
    float f32[8];
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
               Tile::kWriterStrideBytes;
}

template <int MBlock, int Group>
__host__ __device__ constexpr int batch_ds_group_offset() {
    static_assert(MBlock >= 0 && MBlock < Tile::kMqPanels);
    static_assert(Group == 0 || Group == 1);
    return LdsLayout::kKBase + MBlock * Tile::kPdsGenerationBytes +
           Group * Tile::kWavesPerConsumerGroup * Tile::kWriterStrideBytes;
}

template <int MBlock, int Group>
__device__ __forceinline__ const __half* batch_ds_group(const __half* lds) {
    return reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) +
        batch_ds_group_offset<MBlock, Group>());
}

__device__ __forceinline__ float* sidecar_field(__half* lds,
                                                int page,
                                                int field) {
    auto* base = reinterpret_cast<float*>(
        reinterpret_cast<char*>(lds) + LdsLayout::kSidecarBase +
        page * Tile::kSidecarBytes);
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

__device__ __forceinline__ void producer_load_raw_matrices(
    const __half* q,
    const __half* dout,
    __half* lds,
    int64_t tensor_base,
    int q_base,
    int raw_base,
    int wave_local) {
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
            raw_base + offset, true);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, ins::prepare_matrix_src(dout + input, Tile::kHeadDim),
            raw_base + LdsLayout::kDoutOffset + offset, true);
    }
}

__device__ __forceinline__ void producer_load_raw_sidecar(
    const float* packed_sidecar,
    __half* lds,
    int64_t row_base,
    int q_base,
    int page,
    int wave_local,
    int lane) {
    if (lane < 16) {
        const int local_row = wave_local * 16 + lane;
        const float* row =
            packed_sidecar + (row_base + q_base + local_row) * 3;
        sidecar_field(lds, page, 0)[local_row] = row[0];
        sidecar_field(lds, page, 1)[local_row] = row[1];
        sidecar_field(lds, page, 2)[local_row] = row[2];
    }
}

__device__ __forceinline__ void f16_mmac_single(const Fragment& lhs,
                                                 const Fragment& rhs,
                                                 const ins::F16x8& zero,
                                                 Fragment& out,
                                                 bool first_d_block) {
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

__device__ __forceinline__ void latch_resident_fragments(
    const __half* lds,
    int n_owner,
    ResidentFragments& resident) {
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
    ins::wait_lgkm(0);
}

__device__ __forceinline__ void latch_dq_k_normal(
    const __half* lds,
    int d_block,
    DqResidentFragments& resident) {
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
        resident.k_normal[writer] = k_full[writer];
    }
}

// Logical GEMM 1 or 2: four matrix reads, one first-use wait, eight MMAC.
__device__ __forceinline__ void matrix_product_stage(
    const __half* lds,
    int region_base,
    int m_block,
    const Fragment (&rhs)[kMatrixBlocksD],
    const ins::F16x8& zero,
    Fragment& out) {
    Fragment lhs[kMatrixBlocksD];
    const int panel_offset = m16_matrix_offset(m_block, 0);
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + region_base + panel_offset);
    ins::ds_read_matrix_32x16_trans_imm4<0, 2048, 4096, 6144>(
        base, lhs[0].f16x8, lhs[1].f16x8, lhs[2].f16x8,
        lhs[3].f16x8);
    ins::wait_lgkm(0);
    ins::raise_priority_2();
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
        f16_mmac_single(lhs[d_block], rhs[d_block], zero, out,
                        d_block == 0);
    }
    ins::lower_priority();
}

__device__ __forceinline__ void score_stage(
    const __half* lds,
    int raw_base,
    int m_block,
    const ResidentFragments& resident,
    const ins::F16x8& zero,
    Fragment& score) {
    matrix_product_stage(
        lds, raw_base, m_block, resident.k_trans, zero, score);
}

__device__ __forceinline__ void dp_stage(
    const __half* lds,
    int raw_base,
    int m_block,
    const ResidentFragments& resident,
    const ins::F16x8& zero,
    Fragment& dp) {
    matrix_product_stage(
        lds, raw_base + LdsLayout::kDoutOffset, m_block,
        resident.v_trans, zero, dp);
}

__device__ __forceinline__ void read_raw_panel_normal(
    const __half* lds,
    int region_base,
    int m_block,
    Fragment (&normal)[kMatrixBlocksD]) {
    const int panel_offset = m16_matrix_offset(m_block, 0);
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + region_base +
        panel_offset);
    ins::ds_read_matrix_32x16_normal_imm4<0, 2048, 4096, 6144>(
        base, normal[0].f16x8, normal[1].f16x8, normal[2].f16x8,
        normal[3].f16x8);
}

__device__ __forceinline__ void probability_stage(
    const Fragment& score,
    int lane,
    int q_base,
    int m_block,
    int k_base,
    int n_owner,
    int causal,
    float row_max_log2,
    float row_inv_sum,
    float softmax_scale,
    ProbabilityPanel& p) {
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
        p.f32[word] = probability;
        p.f16.scalar[word] = static_cast<_Float16>(probability);
    }
}

__device__ __forceinline__ void ds_stage(
    const ProbabilityPanel& p,
    const Fragment& dp,
    float row_delta,
    float softmax_scale,
    Fragment& ds) {
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        ds.scalar[word] = static_cast<_Float16>(
            p.f32[word] *
            (static_cast<float>(dp.scalar[word]) - row_delta) *
            softmax_scale);
    }
}

__device__ __forceinline__ void zero_dkv_accumulators(
    Accumulator (&dv_acc)[8], Accumulator (&dk_acc)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ins::zero_vgpr2(dv_acc[i].u64[0]);
        ins::zero_vgpr2(dv_acc[i].u64[1]);
        ins::zero_vgpr2(dk_acc[i].u64[0]);
        ins::zero_vgpr2(dk_acc[i].u64[1]);
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

template <int MBlock, int SourceGroup>
__device__ __forceinline__ void update_dq_writer_panel_group(
    const __half* lds,
    const DqResidentFragments& resident,
    Accumulator (&dq_acc)[2]) {
    static_assert(SourceGroup == 0 || SourceGroup == 1);
    Fragment ds_trans[Tile::kWavesPerConsumerGroup];
    const __half* source = batch_ds_group<MBlock, SourceGroup>(lds);
    ins::ds_read_matrix_32x16_trans_imm4<
        0, Tile::kWriterStrideBytes, 2 * Tile::kWriterStrideBytes,
        3 * Tile::kWriterStrideBytes>(
        source, ds_trans[0].f16x8, ds_trans[1].f16x8,
        ds_trans[2].f16x8, ds_trans[3].f16x8);
    ins::wait_lgkm(0);
#pragma unroll
    for (int writer = 0; writer < Tile::kWavesPerConsumerGroup; ++writer) {
        constexpr int kWriterBase =
            SourceGroup * Tile::kWavesPerConsumerGroup;
        ins::mmac_f16_lit_inplace(dq_acc[0].f32,
                                  ds_trans[writer].f16x4[0],
                                  resident.k_normal[kWriterBase + writer]
                                      .f16x4[0]);
        ins::mmac_f16_lit_inplace(dq_acc[1].f32,
                                  ds_trans[writer].f16x4[0],
                                  resident.k_normal[kWriterBase + writer]
                                      .f16x4[1]);
    }
}

__device__ __forceinline__ void store_dq_partial_d32(
    float* dq_partial,
    int64_t partial_base,
    int q_base,
    int m_block,
    int d_owner,
    int lane,
    const Accumulator (&dq_acc)[2]) {
    const int row = q_base + m_block * Tile::kMqPerPanel + (lane & 15);
    const int lane_col = (lane >> 4) * 4;
#pragma unroll
    for (int d_half = 0; d_half < 2; ++d_half) {
        const int col = d_owner * Tile::kHeadDimPerDqWriter +
                        d_half * 16 + lane_col;
        auto* output = reinterpret_cast<ins::Vec4F32*>(
            dq_partial + partial_base +
            static_cast<int64_t>(row) * Tile::kHeadDim + col);
        *output = dq_acc[d_half].f32;
    }
}

template <int Group>
__device__ __forceinline__ void update_dv_from_probability(
    const __half* lds,
    __half* mutable_lds,
    int raw_base,
    int m_block,
    int owner,
    const ProbabilityPanel& p,
    Accumulator (&dv_acc)[8]) {
    static_assert(Group == 0 || Group == 1);

    const int page = scratch_page_offset<Group>(owner);
    Fragment dout_normal[kMatrixBlocksD];
    Fragment p_normal{};
    ins::ds_write_matrix_32x16_trans_f16(p.f16.f16x8, mutable_lds, page);
    ins::ds_read_matrix_32x16_normal(lds, page, p_normal.f16x8);
    read_raw_panel_normal(
        lds, raw_base + LdsLayout::kDoutOffset, m_block, dout_normal);
    ins::wait_lgkm(0);
    ins::raise_priority_2();
    update_dv_stage(p_normal, dout_normal, dv_acc);
    ins::lower_priority();
}

template <int MBlock, int Group>
__device__ __forceinline__ void publish_final_ds(
    const Fragment& ds,
    __half* lds,
    int owner) {
    const int page = batch_ds_group_offset<MBlock, Group>() +
                     owner * Tile::kWriterStrideBytes;
    ins::ds_write_matrix_32x16_trans_f16(ds.f16x8, lds, page);
}

template <int MBlock, int Group>
__device__ __forceinline__ void read_final_ds_normal(
    const __half* lds,
    int owner,
    Fragment& ds) {
    const int page = batch_ds_group_offset<MBlock, Group>() +
                     owner * Tile::kWriterStrideBytes;
    ins::ds_read_matrix_32x16_normal(lds, page, ds.f16x8);
}

template <int MBlock, int Group>
__device__ __forceinline__ void read_dk_panel_static(
    const __half* lds,
    int raw_base,
    int owner,
    Fragment (&q)[kMatrixBlocksD],
    Fragment& ds) {
    read_raw_panel_normal(lds, raw_base, MBlock, q);
    read_final_ds_normal<MBlock, Group>(lds, owner, ds);
}

template <int Group>
__device__ __forceinline__ void update_dk_from_final_panels_read_ahead(
    const __half* lds,
    int raw_base,
    int owner,
    Accumulator (&dk_acc)[8]) {
    // Keep one current and one next Q/dS panel. The next panel's five matrix
    // reads are issued while the current eight-MMAC island is active.
    Fragment q_buf[2][kMatrixBlocksD];
    Fragment ds_buf[2];
    read_dk_panel_static<0, Group>(lds, raw_base, owner, q_buf[0], ds_buf[0]);
    ins::wait_lgkm(0);
    read_dk_panel_static<1, Group>(lds, raw_base, owner, q_buf[1], ds_buf[1]);
    update_dk_stage(ds_buf[0], q_buf[0], dk_acc);
    ins::wait_lgkm(0);
    read_dk_panel_static<2, Group>(lds, raw_base, owner, q_buf[0], ds_buf[0]);
    update_dk_stage(ds_buf[1], q_buf[1], dk_acc);
    ins::wait_lgkm(0);
    read_dk_panel_static<3, Group>(lds, raw_base, owner, q_buf[1], ds_buf[1]);
    update_dk_stage(ds_buf[0], q_buf[0], dk_acc);
    ins::wait_lgkm(0);
    update_dk_stage(ds_buf[1], q_buf[1], dk_acc);
}

__device__ __forceinline__ void zero_dq_writer_accumulators(
    Accumulator (&dq_acc)[Tile::kMqPanels][2]) {
    ins::F16x8 dq_zero;
    ins::zero_f16x8(dq_zero);
#pragma unroll
    for (int m_block = 0; m_block < Tile::kMqPanels; ++m_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            dq_acc[m_block][d_half].f32 = dq_zero.f32;
        }
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
    ResidentFragments resident;
    latch_resident_fragments(lds, n_owner, resident);

    // The producer is the only consumer of this completion token.  Once this
    // wave has latched its resident K/V fragments, waiting for the other
    // waves adds no data dependency: the producer cannot reuse the LDS
    // region until all twelve arrivals are present.
    if constexpr (Group == 0) {
        ins::abarrier_arrive_cnt<false>(Bar::kVSidecarReady, 1);
    }

    Accumulator dv_acc[8];
    Accumulator dk_acc[8];
    zero_dkv_accumulators(dv_acc, dk_acc);
    ins::F16x8 mmac_zero;
    ins::zero_f16x8(mmac_zero);
    int raw0_phase = 0;
    int raw1_phase = 0;
    constexpr int kLocalBatchFilled =
        Group == 0 ? Bar::kBatchDsFilled0 : Bar::kBatchDsFilled1;
    constexpr int kLocalDqDone =
        Group == 0 ? Bar::kDqDone0 : Bar::kDqDone1;
    int local_dq_done_phase = 0;
#pragma clang loop unroll(disable)
    for (int qi = 0; qi < q_tile_count; ++qi) {
        const int page = qi & 1;
        if (page == 0) {
            ins::abarrier_try_wait<true>(Bar::kRawFilled0, raw0_phase);
        } else {
            ins::abarrier_try_wait<true>(Bar::kRawFilled1, raw1_phase);
        }
        const int raw_base = raw_page_base(page);
        const int q_base = (q_tile_begin + qi) * Tile::kMq;

        Fragment ds_panels[Tile::kMqPanels];
#pragma unroll
        for (int m_block = 0; m_block < Tile::kMqPanels; ++m_block) {
            Fragment score{};
            Fragment dp{};
            ProbabilityPanel p{};
            const int sidecar_row =
                m_block * Tile::kMqPerPanel + (lane & 15);
            if constexpr (Group == 0) {
                score_stage(
                    lds, raw_base, m_block, resident, mmac_zero, score);
                probability_stage(
                    score, lane, q_base, m_block, k_base, n_owner, causal,
                    sidecar_field(mutable_lds, page, 0)[sidecar_row],
                    sidecar_field(mutable_lds, page, 1)[sidecar_row],
                    softmax_scale, p);
                update_dv_from_probability<Group>(
                    lds, mutable_lds, raw_base, m_block, owner, p, dv_acc);
                dp_stage(
                    lds, raw_base, m_block, resident, mmac_zero, dp);
                ds_stage(
                    p, dp,
                    sidecar_field(mutable_lds, page, 2)[sidecar_row],
                    softmax_scale, ds_panels[m_block]);
            } else {
                dp_stage(
                    lds, raw_base, m_block, resident, mmac_zero, dp);
                score_stage(
                    lds, raw_base, m_block, resident, mmac_zero, score);
                probability_stage(
                    score, lane, q_base, m_block, k_base, n_owner, causal,
                    sidecar_field(mutable_lds, page, 0)[sidecar_row],
                    sidecar_field(mutable_lds, page, 1)[sidecar_row],
                    softmax_scale, p);
                ds_stage(
                    p, dp,
                    sidecar_field(mutable_lds, page, 2)[sidecar_row],
                    softmax_scale, ds_panels[m_block]);
                update_dv_from_probability<Group>(
                    lds, mutable_lds, raw_base, m_block, owner, p, dv_acc);
            }
        }

        if (qi != 0) {
            // Only the matching dQ writer group must have consumed this
            // group's dS page before the next q tile reuses it.
            ins::abarrier_try_wait<true>(kLocalDqDone,
                                         local_dq_done_phase);
        }
        publish_final_ds<0, Group>(ds_panels[0], mutable_lds, owner);
        publish_final_ds<1, Group>(ds_panels[1], mutable_lds, owner);
        publish_final_ds<2, Group>(ds_panels[2], mutable_lds, owner);
        publish_final_ds<3, Group>(ds_panels[3], mutable_lds, owner);
        ins::abarrier_arrive_cnt<false>(kLocalBatchFilled, 1);

        update_dk_from_final_panels_read_ahead<Group>(
            lds, raw_base, owner, dk_acc);
        if (page == 0) {
            ins::abarrier_arrive_cnt<false>(Bar::kRawUsed0, 1);
        } else {
            ins::abarrier_arrive_cnt<false>(Bar::kRawUsed1, 1);
        }
        ins::abarrier_arrive_cnt<false>(kLocalDqDone, 1);
    }

    store_dkv_outputs(dk, dv, tensor_base, k_base, n_owner, lane, dk_acc,
                      dv_acc);
}

__device__ __forceinline__ void run_dq_writer(
    const __half* lds,
    float* dq_partial,
    int64_t partial_base,
    int q_tile_begin,
    int q_tile_count,
    int d_owner,
    int lane) {
    DqResidentFragments resident;
    latch_dq_k_normal(lds, d_owner, resident);

    // The producer owns the reuse edge; this writer only needs its own K
    // fragment before publishing dQ work.
    int filled0_phase = 0;
    int filled1_phase = 0;
#pragma clang loop unroll(disable)
    for (int qi = 0; qi < q_tile_count; ++qi) {
        Accumulator dq_acc[Tile::kMqPanels][2];
        zero_dq_writer_accumulators(dq_acc);

        ins::abarrier_try_wait<true>(Bar::kBatchDsFilled0,
                                     filled0_phase);
        update_dq_writer_panel_group<0, 0>(lds, resident, dq_acc[0]);
        update_dq_writer_panel_group<1, 0>(lds, resident, dq_acc[1]);
        update_dq_writer_panel_group<2, 0>(lds, resident, dq_acc[2]);
        update_dq_writer_panel_group<3, 0>(lds, resident, dq_acc[3]);
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone0, 1);

        ins::abarrier_try_wait<true>(Bar::kBatchDsFilled1,
                                     filled1_phase);
        update_dq_writer_panel_group<0, 1>(lds, resident, dq_acc[0]);
        update_dq_writer_panel_group<1, 1>(lds, resident, dq_acc[1]);
        update_dq_writer_panel_group<2, 1>(lds, resident, dq_acc[2]);
        update_dq_writer_panel_group<3, 1>(lds, resident, dq_acc[3]);
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone1, 1);

        const int q_base = (q_tile_begin + qi) * Tile::kMq;
        store_dq_partial_d32(dq_partial, partial_base, q_base, 0, d_owner,
                             lane, dq_acc[0]);
        store_dq_partial_d32(dq_partial, partial_base, q_base, 1, d_owner,
                             lane, dq_acc[1]);
        store_dq_partial_d32(dq_partial, partial_base, q_base, 2, d_owner,
                             lane, dq_acc[2]);
        store_dq_partial_d32(dq_partial, partial_base, q_base, 3, d_owner,
                             lane, dq_acc[3]);
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
                     float* __restrict__ dq_partial,
                     float* __restrict__ dk,
                     float* __restrict__ dv,
                     int heads,
                     int seqlen,
                     int causal,
                     float softmax_scale) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(Wdra::kProducerVgprs, Wdra::kConsumer0Vgprs,
                            Wdra::kConsumer1Vgprs,
                            Wdra::kDqWriterVgprs);
#endif
#if defined(__gfx946__)
    __shared__ __align__(2048) __half lds[LdsLayout::kBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kVSidecarReady, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed0, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed1, 8);
        __builtin_hcu_s_abarrier_init(Bar::kBatchDsFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kBatchDsFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDqDone0, 8);
        __builtin_hcu_s_abarrier_init(Bar::kDqDone1, 8);
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

        ins::abarrier_seq<false>(Bar::kResidentFilled);
        producer_load_resident_group<0>(k, v, lds, tensor_base, k_base,
                                        wave_local);
        producer_load_resident_group<1>(k, v, lds, tensor_base, k_base,
                                        wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled, 1);

        int kv_latched_phase = 0;
        int raw0_used_phase = 0;
        int raw1_used_phase = 0;
        if (q_tile_count != 0) {
            ins::abarrier_seq<false>(Bar::kRawFilled0);
            producer_load_raw_matrices(
                q, dout, lds, tensor_base,
                q_tile_begin * Tile::kMq, LdsLayout::kRaw0Base,
                wave_local);
            ins::abarrier_try_wait<true>(Bar::kVSidecarReady,
                                         kv_latched_phase);
            producer_load_raw_sidecar(
                packed_sidecar, lds, row_base,
                q_tile_begin * Tile::kMq, 0, wave_local, lane);
            ins::wait_vmem_lgkm();
            ins::maybe_wait_bps_vbcnt_before_arrive();
            ins::abarrier_arrive_cnt<false>(Bar::kRawFilled0, 1);
        }
        for (int qi = 1; qi < q_tile_count; ++qi) {
            const int page = qi & 1;
            if (qi >= 2) {
                if (page == 0) {
                    ins::abarrier_try_wait<true>(Bar::kRawUsed0,
                                                 raw0_used_phase);
                } else {
                    ins::abarrier_try_wait<true>(Bar::kRawUsed1,
                                                 raw1_used_phase);
                }
            }
            if (page == 0) {
                ins::abarrier_seq<false>(Bar::kRawFilled0);
            } else {
                ins::abarrier_seq<false>(Bar::kRawFilled1);
            }
            const int q_base = (q_tile_begin + qi) * Tile::kMq;
            producer_load_raw_matrices(
                q, dout, lds, tensor_base, q_base,
                raw_page_base(page), wave_local);
            producer_load_raw_sidecar(
                packed_sidecar, lds, row_base, q_base, page,
                wave_local, lane);
            ins::wait_vmem_lgkm();
            ins::maybe_wait_bps_vbcnt_before_arrive();
            if (page == 0) {
                ins::abarrier_arrive_cnt<false>(Bar::kRawFilled0, 1);
            } else {
                ins::abarrier_arrive_cnt<false>(Bar::kRawFilled1, 1);
            }
        }
        if (q_tile_count != 0) {
            ins::abarrier_try_wait<true>(Bar::kRawUsed0,
                                         raw0_used_phase);
        }
        if (q_tile_count > 1) {
            ins::abarrier_try_wait<true>(Bar::kRawUsed1,
                                         raw1_used_phase);
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
        int resident_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled,
                                     resident_phase);
        run_consumer_group<0>(lds, lds, dk, dv, tensor_base, k_base,
                              q_tile_begin, q_tile_count, causal,
                              softmax_scale, owner, lane);
    } else if (wave < Tile::kDqWriterWaveBegin) {
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
        int resident_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled,
                                     resident_phase);
        run_consumer_group<1>(lds, lds, dk, dv, tensor_base, k_base,
                              q_tile_begin, q_tile_count, causal,
                              softmax_scale, owner, lane);
    } else {
        __builtin_hcu_s_set_vgpr_size(Wdra::kDqWriterVgprs);
        const int d_owner = wave - Tile::kDqWriterWaveBegin;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int bh = static_cast<int>(blockIdx.y) * heads +
                       static_cast<int>(blockIdx.x);
        const int k_base = static_cast<int>(blockIdx.z) * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t partial_base =
            (static_cast<int64_t>(bh) * gridDim.z + blockIdx.z) * seqlen *
            Tile::kHeadDim;
        int resident_phase = 0;
        ins::abarrier_try_wait<true>(Bar::kResidentFilled,
                                     resident_phase);
        run_dq_writer(lds, dq_partial, partial_base, q_tile_begin,
                      q_tile_count, d_owner, lane);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kVSidecarReady);
        __builtin_hcu_s_abarrier_inv(Bar::kRawFilled0);
        __builtin_hcu_s_abarrier_inv(Bar::kRawUsed0);
        __builtin_hcu_s_abarrier_inv(Bar::kRawFilled1);
        __builtin_hcu_s_abarrier_inv(Bar::kRawUsed1);
        __builtin_hcu_s_abarrier_inv(Bar::kBatchDsFilled0);
        __builtin_hcu_s_abarrier_inv(Bar::kBatchDsFilled1);
        __builtin_hcu_s_abarrier_inv(Bar::kDqDone0);
        __builtin_hcu_s_abarrier_inv(Bar::kDqDone1);
    }
#else
    (void)dout;
    (void)q;
    (void)k;
    (void)v;
    (void)packed_sidecar;
    (void)dq_partial;
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

    const size_t required_workspace = fused::dq_workspace_bytes(params);
    if (required_workspace == 0 || params->workspace == nullptr ||
        params->workspace_bytes < required_workspace) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const dim3 grid(params->num_heads_q, params->batch,
                    params->seqlen_k / Tile::kNk);
    hipLaunchKernelGGL(
        fa3_bwd_5gemm_kernel, grid, dim3(Tile::kThreadsPerCta), 0, 0,
        static_cast<const __half*>(dout), static_cast<const __half*>(q),
        static_cast<const __half*>(k), static_cast<const __half*>(v),
        static_cast<const float*>(packed_sidecar),
        static_cast<float*>(params->workspace), static_cast<float*>(dk),
        static_cast<float*>(dv),
        params->num_heads_q, params->seqlen_q, params->causal,
        params->softmax_scale);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return SHAOBO_FA3_STATUS_HIP_ERROR;
    }
    const int reduce_status = fused::launch_dq_reduction(
        static_cast<const float*>(params->workspace),
        static_cast<float*>(dq), params);
    if (reduce_status != SHAOBO_FA3_STATUS_SUCCESS) {
        return reduce_status;
    }
    if (params->sync_after_launch != 0) {
        error = hipDeviceSynchronize();
        if (error != hipSuccess) {
            return SHAOBO_FA3_STATUS_HIP_ERROR;
        }
    }
    return SHAOBO_FA3_STATUS_SUCCESS;
}
