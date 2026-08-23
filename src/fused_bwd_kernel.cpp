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
constexpr int kUsefulScoreWordsPerLane = 4;

static_assert(Tile::kNkPerConsumerWave == 16,
              "the score fragment valid-half proof assumes an N16 owner");

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
    static constexpr int kC0AlternateDsBase = kVBase + 16 * 1024;
    static constexpr int kC0AlternateDsBytes = Tile::kC0AlternateDsBytes;
    static constexpr int kRaw1Base = kVBase + kVBytes;
    static constexpr int kBytes = kRaw1Base + Tile::kRawQDoBytes;

    static_assert(kQBytes == 16 * 1024 && kKBytes == 32 * 1024,
                  "fused input tile sizes changed");
    static_assert(kScratchBase + kScratchBytes + kSidecarBytes <= kRaw1Base,
                  "steady P/dS side data must not overlap raw page1");
    static_assert(kScratchBase + kScratchBytes + kSidecarBytes <=
                      kC0AlternateDsBase &&
                      kC0AlternateDsBase + kC0AlternateDsBytes <= kRaw1Base,
                  "alternate C0 dS page must fit released V LDS");
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

template <int MBlock, int Group, int Generation = 0>
__host__ __device__ constexpr int batch_ds_group_offset() {
    static_assert(MBlock >= 0 && MBlock < Tile::kMqPanels);
    static_assert(Group == 0 || Group == 1);
    static_assert(Generation == 0 || (Group == 0 && Generation == 1));
    if constexpr (Generation == 1) {
        return LdsLayout::kC0AlternateDsBase +
               MBlock * Tile::kWavesPerConsumerGroup *
                   Tile::kWriterStrideBytes;
    }
    return LdsLayout::kKBase + MBlock * Tile::kPdsGenerationBytes +
           Group * Tile::kWavesPerConsumerGroup * Tile::kWriterStrideBytes;
}

template <int MBlock, int Group, int Generation = 0>
__device__ __forceinline__ const __half* batch_ds_group(const __half* lds) {
    return reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) +
        batch_ds_group_offset<MBlock, Group, Generation>());
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
__device__ __forceinline__ void read_raw_panel_trans(
    const __half* lds,
    int region_base,
    int m_block,
    Fragment (&lhs)[kMatrixBlocksD]) {
    const int panel_offset = m16_matrix_offset(m_block, 0);
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + region_base + panel_offset);
    ins::ds_read_matrix_32x16_trans_imm4<0, 2048, 4096, 6144>(
        base, lhs[0].f16x8, lhs[1].f16x8, lhs[2].f16x8,
        lhs[3].f16x8);
}

template <int Count>
__device__ __forceinline__ void wait_matrix_packet(
    Fragment (&packet)[kMatrixBlocksD]) {
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile(
        "s_waitcnt lgkmcnt(%4)\n"
        : "+v"(packet[0].f16x8), "+v"(packet[1].f16x8),
          "+v"(packet[2].f16x8), "+v"(packet[3].f16x8)
        : "n"(Count)
        : "memory");
#else
    (void)packet;
#endif
}

template <int Count>
__device__ __forceinline__ void wait_dv_packet(
    Fragment& p, Fragment (&dout)[kMatrixBlocksD]) {
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile(
        "s_waitcnt lgkmcnt(%5)\n"
        : "+v"(p.f16x8), "+v"(dout[0].f16x8),
          "+v"(dout[1].f16x8), "+v"(dout[2].f16x8),
          "+v"(dout[3].f16x8)
        : "n"(Count)
        : "memory");
#else
    (void)p;
    (void)dout;
#endif
}

__device__ __forceinline__ void wait_sidecar_value(float& value) {
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile(
        "; sidecar_ready_before_dv_packet\n"
        : "+v"(value)
        :
        : "memory");
#else
    (void)value;
#endif
}

__device__ __forceinline__ void mmac_product_island(
    const Fragment (&lhs)[kMatrixBlocksD],
    const Fragment (&rhs)[kMatrixBlocksD],
    const ins::F16x8& zero,
    Fragment& out) {
    ins::raise_priority_2();
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
        f16_mmac_single(lhs[d_block], rhs[d_block], zero, out,
                        d_block == 0);
    }
    ins::lower_priority();
}

__device__ __forceinline__ void matrix_product_stage(
    const __half* lds,
    int region_base,
    int m_block,
    const Fragment (&rhs)[kMatrixBlocksD],
    const ins::F16x8& zero,
    Fragment& out) {
    Fragment lhs[kMatrixBlocksD];
    read_raw_panel_trans(lds, region_base, m_block, lhs);
    ins::wait_lgkm(0);
    mmac_product_island(lhs, rhs, zero, out);
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
    const ins::F16x8& zero,
    ProbabilityPanel& p) {
    const float scale_log2 = softmax_scale * kLog2E;
#pragma unroll
    for (int word = 0; word < kUsefulScoreWordsPerLane; ++word) {
        int local_q = 0;
        int local_k = 0;
        source_slot_qk(lane, word, local_q, local_k);
        const int qrow = q_base + m_block * Tile::kMqPerPanel + local_q;
        const int krow =
            k_base + n_owner * Tile::kNkPerConsumerWave + local_k;
        const float probability =
            (!causal || krow <= qrow)
                ? exp2f(static_cast<float>(score.scalar[word]) * scale_log2 -
                        row_max_log2) *
                      row_inv_sum
                : 0.0f;
        p.f32[word] = probability;
        p.f16.scalar[word] = static_cast<_Float16>(probability);
    }
    p.f16.f16x4[1] = zero.f16x4[0];
}

__device__ __forceinline__ void ds_stage(
    const ProbabilityPanel& p,
    const Fragment& dp,
    float row_delta,
    float softmax_scale,
    const ins::F16x8& zero,
    Fragment& ds) {
#pragma unroll
    for (int word = 0; word < kUsefulScoreWordsPerLane; ++word) {
        ds.scalar[word] = static_cast<_Float16>(
            p.f32[word] *
            (static_cast<float>(dp.scalar[word]) - row_delta) *
            softmax_scale);
    }
    ds.f16x4[1] = zero.f16x4[0];
}

__device__ __forceinline__ void scale_probability_for_ds(
    ProbabilityPanel& p,
    float softmax_scale) {
#pragma unroll
    for (int word = 0; word < kUsefulScoreWordsPerLane; ++word) {
        p.f32[word] *= softmax_scale;
    }
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile(
        "; scaled_probability_ready_before_dp\n"
        : "+v"(p.f32[0]), "+v"(p.f32[1]), "+v"(p.f32[2]),
          "+v"(p.f32[3])
        :
        : "memory");
#endif
}

__device__ __forceinline__ void ds_stage_from_scaled_probability(
    const ProbabilityPanel& scaled_p,
    const Fragment& dp,
    float row_delta,
    const ins::F16x8& zero,
    Fragment& ds) {
#pragma unroll
    for (int word = 0; word < kUsefulScoreWordsPerLane; ++word) {
        ds.scalar[word] = static_cast<_Float16>(
            scaled_p.f32[word] *
            (static_cast<float>(dp.scalar[word]) - row_delta));
    }
    ds.f16x4[1] = zero.f16x4[0];
}

__device__ __forceinline__ void c0_dp_stage(
    const __half* lds,
    int raw_base,
    int m_block,
    const ResidentFragments& resident,
    float softmax_scale,
    const ins::F16x8& zero,
    ProbabilityPanel& p,
    Fragment& dp) {
    Fragment dp_lhs[kMatrixBlocksD];
    read_raw_panel_trans(
        lds, raw_base + LdsLayout::kDoutOffset, m_block, dp_lhs);
    scale_probability_for_ds(p, softmax_scale);
    wait_matrix_packet<0>(dp_lhs);
    mmac_product_island(dp_lhs, resident.v_trans, zero, dp);
}

__device__ __forceinline__ void zero_accumulators(
    Accumulator (&acc)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ins::zero_vgpr2(acc[i].u64[0]);
        ins::zero_vgpr2(acc[i].u64[1]);
    }
}

// Logical GEMM 3: consume dO while its four matrix fragments are hot.
template <bool SeedZero>
__device__ __forceinline__ void update_dv_stage(
    const Fragment& p_normal,
    const Fragment (&dout_normal)[kMatrixBlocksD],
    const ins::F16x8& zero,
    Accumulator (&dv_acc)[8]) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            const int out = d_block * 2 + d_half;
            if constexpr (SeedZero) {
                dv_acc[out].f32 = ins::mmac_f16_lit(
                    p_normal.f16x4[0],
                    dout_normal[d_block].f16x4[d_half], zero.f32);
            } else {
                dv_acc[out].f32 = ins::mmac_f16_lit(
                    p_normal.f16x4[0],
                    dout_normal[d_block].f16x4[d_half], dv_acc[out].f32);
            }
        }
    }
}

// Logical GEMM 4: consume Q in a separate island so dO is already dead.
template <bool SeedZero>
__device__ __forceinline__ void update_dk_stage(
    const Fragment& ds_normal,
    const Fragment (&q_normal)[kMatrixBlocksD],
    const ins::F16x8& zero,
    Accumulator (&dk_acc)[8]) {
#pragma unroll
    for (int d_block = 0; d_block < kMatrixBlocksD; ++d_block) {
#pragma unroll
        for (int d_half = 0; d_half < 2; ++d_half) {
            const int out = d_block * 2 + d_half;
            if constexpr (SeedZero) {
                dk_acc[out].f32 = ins::mmac_f16_lit(
                    ds_normal.f16x4[0], q_normal[d_block].f16x4[d_half],
                    zero.f32);
            } else {
                dk_acc[out].f32 = ins::mmac_f16_lit(
                    ds_normal.f16x4[0], q_normal[d_block].f16x4[d_half],
                    dk_acc[out].f32);
            }
        }
    }
}

template <int MBlock, int SourceGroup, int Generation = 0>
__device__ __forceinline__ void update_dq_writer_panel_group(
    const __half* lds,
    const DqResidentFragments& resident,
    Accumulator (&dq_acc)[2]) {
    static_assert(SourceGroup == 0 || SourceGroup == 1);
    Fragment ds_trans[Tile::kWavesPerConsumerGroup];
    const __half* source =
        batch_ds_group<MBlock, SourceGroup, Generation>(lds);
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
    __half* dq_partial,
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
        auto* output = reinterpret_cast<ins::Vec4F16*>(
            dq_partial + partial_base +
            static_cast<int64_t>(row) * Tile::kHeadDim + col);
        *output = __builtin_convertvector(dq_acc[d_half].f32, ins::Vec4F16);
    }
}

struct DvOperandPacket {
    Fragment p_normal{};
    Fragment dout_normal[kMatrixBlocksD];
};

__device__ __forceinline__ void order_dv_packet_before_ds(
    DvOperandPacket& packet,
    float& row_delta) {
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile(
        "; c1_dv_packet_before_ds\n"
        : "+v"(row_delta)
        : "v"(packet.p_normal.f16x8),
          "v"(packet.dout_normal[0].f16x8),
          "v"(packet.dout_normal[1].f16x8),
          "v"(packet.dout_normal[2].f16x8),
          "v"(packet.dout_normal[3].f16x8)
        : "memory");
#else
    (void)packet;
    (void)row_delta;
#endif
}

template <int Count>
__device__ __forceinline__ void wait_dv_packet_after_ds(
    const Fragment& ds,
    DvOperandPacket& packet) {
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile(
        "s_waitcnt lgkmcnt(%6)\n"
        : "+v"(packet.p_normal.f16x8),
          "+v"(packet.dout_normal[0].f16x8),
          "+v"(packet.dout_normal[1].f16x8),
          "+v"(packet.dout_normal[2].f16x8),
          "+v"(packet.dout_normal[3].f16x8)
        : "v"(ds.f16x8), "n"(Count)
        : "memory");
#else
    (void)ds;
    (void)packet;
#endif
}

template <int Group>
__device__ __forceinline__ void issue_dv_operand_packet(
    const __half* lds,
    __half* mutable_lds,
    int raw_base,
    int m_block,
    int owner,
    const Fragment& probability,
    DvOperandPacket& packet) {
    static_assert(Group == 0 || Group == 1);

    const int page = scratch_page_offset<Group>(owner);
    ins::ds_write_matrix_32x16_trans_f16(probability.f16x8, mutable_lds, page);
    ins::ds_read_matrix_32x16_normal(lds, page, packet.p_normal.f16x8);
    read_raw_panel_normal(
        lds, raw_base + LdsLayout::kDoutOffset, m_block,
        packet.dout_normal);
}

template <int Count, bool SeedZero = false>
__device__ __forceinline__ void consume_dv_operand_packet(
    DvOperandPacket& packet,
    const ins::F16x8& zero,
    Accumulator (&dv_acc)[8]) {
    wait_dv_packet<Count>(packet.p_normal, packet.dout_normal);
    ins::raise_priority_2();
    update_dv_stage<SeedZero>(packet.p_normal, packet.dout_normal, zero,
                              dv_acc);
    ins::lower_priority();
}

template <int Count, bool SeedZero = false>
__device__ __forceinline__ void consume_dv_operand_packet_after_ds(
    DvOperandPacket& packet,
    const Fragment& ds,
    const ins::F16x8& zero,
    Accumulator (&dv_acc)[8]) {
    wait_dv_packet_after_ds<Count>(ds, packet);
    ins::raise_priority_2();
    update_dv_stage<SeedZero>(packet.p_normal, packet.dout_normal, zero,
                              dv_acc);
    ins::lower_priority();
}

template <int Group>
__device__ __forceinline__ void issue_dv_operand_packet_and_prefetch_next_dout(
    const __half* lds,
    __half* mutable_lds,
    int raw_base,
    int m_block,
    int next_m_block,
    int owner,
    const Fragment& probability,
    DvOperandPacket& packet,
    Fragment (&next_dout_trans)[kMatrixBlocksD]) {
    static_assert(Group == 0 || Group == 1);

    issue_dv_operand_packet<Group>(
        lds, mutable_lds, raw_base, m_block, owner, probability, packet);
    read_raw_panel_trans(
        lds, raw_base + LdsLayout::kDoutOffset, next_m_block,
        next_dout_trans);
}

template <int Group, bool SeedZero = false>
__device__ __forceinline__ void update_dv_from_probability(
    const __half* lds,
    __half* mutable_lds,
    int raw_base,
    int m_block,
    int owner,
    const Fragment& probability,
    const ins::F16x8& zero,
    Accumulator (&dv_acc)[8]) {
    DvOperandPacket packet;
    issue_dv_operand_packet<Group>(
        lds, mutable_lds, raw_base, m_block, owner, probability, packet);
    consume_dv_operand_packet<0, SeedZero>(packet, zero, dv_acc);
}

template <int MBlock, int Group, int Generation = 0>
__device__ __forceinline__ void publish_final_ds(
    const Fragment& ds,
    __half* lds,
    int owner) {
    const int page = batch_ds_group_offset<MBlock, Group, Generation>() +
                     owner * Tile::kWriterStrideBytes;
    ins::ds_write_matrix_32x16_trans_f16(ds.f16x8, lds, page);
}

template <int Group, int Generation = 0>
__device__ __forceinline__ void publish_ds_batch(
    const Fragment (&ds_panels)[Tile::kMqPanels],
    __half* lds,
    int owner) {
    publish_final_ds<0, Group, Generation>(ds_panels[0], lds, owner);
    publish_final_ds<1, Group, Generation>(ds_panels[1], lds, owner);
    publish_final_ds<2, Group, Generation>(ds_panels[2], lds, owner);
    publish_final_ds<3, Group, Generation>(ds_panels[3], lds, owner);
}

__device__ __forceinline__ void run_c1_causal_zero_front(
    __half* lds,
    int owner,
    const ins::F16x8& zero,
    int& raw_phase) {
    ins::abarrier_try_wait<true>(Bar::kRawFilled0, raw_phase);
    Fragment zero_ds;
    zero_ds.f16x8 = zero.f16x8;
    publish_final_ds<0, 1>(zero_ds, lds, owner);
    publish_final_ds<1, 1>(zero_ds, lds, owner);
    publish_final_ds<2, 1>(zero_ds, lds, owner);
    publish_final_ds<3, 1>(zero_ds, lds, owner);
    ins::abarrier_arrive_cnt<false>(Bar::kBatchDsFilled1, 1);
    ins::abarrier_arrive_cnt<false>(Bar::kRawUsed0, 1);
    ins::abarrier_arrive_cnt<false>(Bar::kDqDone1, 1);
}

template <int MBlock, int Group, int Generation = 0>
__device__ __forceinline__ void read_final_ds_normal(
    const __half* lds,
    int owner,
    Fragment& ds) {
    const int page = batch_ds_group_offset<MBlock, Group, Generation>() +
                     owner * Tile::kWriterStrideBytes;
    ins::ds_read_matrix_32x16_normal(lds, page, ds.f16x8);
}

template <int MBlock, int Group, int Generation = 0>
__device__ __forceinline__ void read_dk_panel_static(
    const __half* lds,
    int raw_base,
    int owner,
    Fragment (&q)[kMatrixBlocksD],
    Fragment& ds) {
    read_raw_panel_normal(lds, raw_base, MBlock, q);
    read_final_ds_normal<MBlock, Group, Generation>(lds, owner, ds);
}

template <int Group, int Generation = 0, bool SeedZero = false>
__device__ __forceinline__ void update_dk_from_final_panels_read_ahead(
    const __half* lds,
    int raw_base,
    int owner,
    const ins::F16x8& zero,
    Accumulator (&dk_acc)[8]) {
    // Keep one current and one next Q/dS panel. The next panel's five matrix
    // reads are issued while the current eight-MMAC island is active.
    Fragment q_buf[2][kMatrixBlocksD];
    Fragment ds_buf[2];
    read_dk_panel_static<0, Group, Generation>(
        lds, raw_base, owner, q_buf[0], ds_buf[0]);
    ins::wait_lgkm(0);
    read_dk_panel_static<1, Group, Generation>(
        lds, raw_base, owner, q_buf[1], ds_buf[1]);
    update_dk_stage<SeedZero>(ds_buf[0], q_buf[0], zero, dk_acc);
    ins::wait_lgkm(0);
    read_dk_panel_static<2, Group, Generation>(
        lds, raw_base, owner, q_buf[0], ds_buf[0]);
    update_dk_stage<false>(ds_buf[1], q_buf[1], zero, dk_acc);
    ins::wait_lgkm(0);
    read_dk_panel_static<3, Group, Generation>(
        lds, raw_base, owner, q_buf[1], ds_buf[1]);
    update_dk_stage<false>(ds_buf[0], q_buf[0], zero, dk_acc);
    ins::wait_lgkm(0);
    update_dk_stage<false>(ds_buf[1], q_buf[1], zero, dk_acc);
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

template <int SourceGroup, int Generation = 0>
__device__ __forceinline__ void update_dq_writer_group(
    const __half* lds,
    const DqResidentFragments& resident,
    Accumulator (&dq_acc)[Tile::kMqPanels][2]) {
    update_dq_writer_panel_group<0, SourceGroup, Generation>(
        lds, resident, dq_acc[0]);
    update_dq_writer_panel_group<1, SourceGroup, Generation>(
        lds, resident, dq_acc[1]);
    update_dq_writer_panel_group<2, SourceGroup, Generation>(
        lds, resident, dq_acc[2]);
    update_dq_writer_panel_group<3, SourceGroup, Generation>(
        lds, resident, dq_acc[3]);
}

__device__ __forceinline__ void store_dkv_outputs_impl(
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
            float* dk_ptr = dk + tensor_base +
                            static_cast<int64_t>(row) * Tile::kHeadDim + col;
            float* dv_ptr = dv + tensor_base +
                            static_cast<int64_t>(row) * Tile::kHeadDim + col;
            *reinterpret_cast<ins::Vec4F32*>(dk_ptr) = dk_acc[out].f32;
            *reinterpret_cast<ins::Vec4F32*>(dv_ptr) = dv_acc[out].f32;
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
    store_dkv_outputs_impl(dk, dv, tensor_base, k_base, n_owner, lane,
                           dk_acc, dv_acc);
}

template <int Group, int Page, bool ReuseDs, bool FirstAccum>
__device__ __forceinline__ void run_consumer_q_tile(
    const __half* lds,
    __half* mutable_lds,
    int k_base,
    int q_tile_begin,
    int qi,
    int causal,
    float softmax_scale,
    int n_owner,
    int owner,
    int lane,
    const ResidentFragments& resident,
    const ins::F16x8& mmac_zero,
    Accumulator (&dv_acc)[8],
    Accumulator (&dk_acc)[8],
    int& raw_phase,
    int& dq_done_phase) {
    static_assert(Page == 0 || Page == 1);
    if constexpr (Page == 0) {
        ins::abarrier_try_wait<true>(Bar::kRawFilled0, raw_phase);
    } else {
        ins::abarrier_try_wait<true>(Bar::kRawFilled1, raw_phase);
    }
    constexpr int kDsGeneration = Group == 0 ? Page : 0;
    const int raw_base = raw_page_base(Page);
    const int q_base = (q_tile_begin + qi) * Tile::kMq;

    Fragment ds_panels[Tile::kMqPanels];
    Fragment p_for_dv[Tile::kMqPanels];
    Fragment q_buf[2][kMatrixBlocksD];
    Fragment dout_buf[2][kMatrixBlocksD];
    if constexpr (Group == 0) {
        read_raw_panel_trans(lds, raw_base, 0, q_buf[0]);
    } else {
        read_raw_panel_trans(
            lds, raw_base + LdsLayout::kDoutOffset, 0, dout_buf[0]);
    }
#pragma unroll
    for (int m_block = 0; m_block < Tile::kMqPanels; ++m_block) {
        Fragment score{};
        Fragment dp{};
        ProbabilityPanel p;
        const int sidecar_row =
            m_block * Tile::kMqPerPanel + (lane & 15);
        constexpr int kPrefetchAlive = 4;
        if constexpr (Group == 0) {
            if (m_block < Tile::kMqPanels - 1) {
                read_raw_panel_trans(lds, raw_base, m_block + 1,
                                     q_buf[(m_block + 1) & 1]);
            }
            ins::wait_lgkm(m_block < Tile::kMqPanels - 1
                               ? kPrefetchAlive
                               : 0);
            mmac_product_island(q_buf[m_block & 1], resident.k_trans,
                                mmac_zero, score);
            probability_stage(
                score, lane, q_base, m_block, k_base, n_owner, causal,
                sidecar_field(mutable_lds, Page, 0)[sidecar_row],
                sidecar_field(mutable_lds, Page, 1)[sidecar_row],
                softmax_scale, mmac_zero, p);
            p_for_dv[m_block] = p.f16;
            c0_dp_stage(
                lds, raw_base, m_block, resident, softmax_scale, mmac_zero,
                p, dp);
            ds_stage_from_scaled_probability(
                p, dp,
                sidecar_field(mutable_lds, Page, 2)[sidecar_row],
                mmac_zero, ds_panels[m_block]);
        } else {
            Fragment score_packet[kMatrixBlocksD];
            read_raw_panel_trans(lds, raw_base, m_block, score_packet);
            wait_matrix_packet<kMatrixBlocksD>(dout_buf[m_block & 1]);
            mmac_product_island(
                dout_buf[m_block & 1], resident.v_trans, mmac_zero, dp);
            wait_matrix_packet<0>(score_packet);
            mmac_product_island(
                score_packet, resident.k_trans, mmac_zero, score);
            probability_stage(
                score, lane, q_base, m_block, k_base, n_owner, causal,
                sidecar_field(mutable_lds, Page, 0)[sidecar_row],
                sidecar_field(mutable_lds, Page, 1)[sidecar_row],
                softmax_scale, mmac_zero, p);
            float row_delta =
                sidecar_field(mutable_lds, Page, 2)[sidecar_row];
            wait_sidecar_value(row_delta);
            DvOperandPacket dv_packet;
            if (m_block < Tile::kMqPanels - 1) {
                issue_dv_operand_packet_and_prefetch_next_dout<Group>(
                    lds, mutable_lds, raw_base, m_block, m_block + 1,
                    owner, p.f16, dv_packet,
                    dout_buf[(m_block + 1) & 1]);
            } else {
                issue_dv_operand_packet<Group>(
                    lds, mutable_lds, raw_base, m_block, owner, p.f16,
                    dv_packet);
            }
            order_dv_packet_before_ds(dv_packet, row_delta);
            ds_stage(p, dp, row_delta, softmax_scale, mmac_zero,
                     ds_panels[m_block]);
            if (m_block < Tile::kMqPanels - 1) {
                consume_dv_operand_packet_after_ds<kMatrixBlocksD>(
                    dv_packet, ds_panels[m_block], mmac_zero, dv_acc);
            } else {
                consume_dv_operand_packet_after_ds<0>(
                    dv_packet, ds_panels[m_block], mmac_zero, dv_acc);
            }
        }
    }

    if constexpr (ReuseDs) {
        if constexpr (Group == 0 && kDsGeneration == 1) {
            ins::abarrier_try_wait<true>(Bar::kDqDone0Alt, dq_done_phase);
        } else if constexpr (Group == 0) {
            ins::abarrier_try_wait<true>(Bar::kDqDone0, dq_done_phase);
        } else {
            ins::abarrier_try_wait<true>(Bar::kDqDone1, dq_done_phase);
        }
    }

    if constexpr (Group == 0 && kDsGeneration == 1) {
        publish_ds_batch<0, 1>(ds_panels, mutable_lds, owner);
        ins::abarrier_arrive_cnt<false>(Bar::kBatchDsFilled0Alt, 1);
    } else if constexpr (Group == 0) {
        publish_ds_batch<0, 0>(ds_panels, mutable_lds, owner);
        ins::abarrier_arrive_cnt<false>(Bar::kBatchDsFilled0, 1);
    } else {
        publish_ds_batch<1>(ds_panels, mutable_lds, owner);
        ins::abarrier_arrive_cnt<false>(Bar::kBatchDsFilled1, 1);
    }

    if constexpr (Group == 0) {
        update_dv_from_probability<Group, FirstAccum>(
            lds, mutable_lds, raw_base, 0, owner, p_for_dv[0], mmac_zero,
            dv_acc);
#pragma unroll
        for (int m_block = 1; m_block < Tile::kMqPanels; ++m_block) {
            update_dv_from_probability<Group, false>(
                lds, mutable_lds, raw_base, m_block, owner,
                p_for_dv[m_block], mmac_zero, dv_acc);
        }
    }

    update_dk_from_final_panels_read_ahead<
        Group, kDsGeneration, FirstAccum>(
        lds, raw_base, owner, mmac_zero, dk_acc);
    if constexpr (Page == 0) {
        ins::abarrier_arrive_cnt<false>(Bar::kRawUsed0, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Bar::kRawUsed1, 1);
    }
    if constexpr (Group == 0 && kDsGeneration == 1) {
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone0Alt, 1);
    } else if constexpr (Group == 0) {
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone0, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone1, 1);
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
    if constexpr (Group == 0) {
        ins::abarrier_arrive_cnt<false>(Bar::kVSidecarReady, 1);
    }

    Accumulator dv_acc[8];
    Accumulator dk_acc[8];
    if constexpr (Group == 1) {
        // C1's first causal Q tile is provably outside its K64 domain. Both
        // accumulator families therefore start from explicit zero; C0 keeps
        // compile-time first-MMAC seeding.
        zero_accumulators(dv_acc);
        zero_accumulators(dk_acc);
    }
    ins::F16x8 mmac_zero;
    ins::zero_f16x8(mmac_zero);
    int raw0_phase = 0;
    int raw1_phase = 0;
    int done0_phase = 0;
    int done1_phase = 0;

    int qi = 0;
    if (q_tile_count > 0) {
        if constexpr (Group == 0) {
            run_consumer_q_tile<0, 0, false, true>(
                lds, mutable_lds, k_base, q_tile_begin, qi, causal,
                softmax_scale, n_owner, owner, lane, resident, mmac_zero,
                dv_acc, dk_acc, raw0_phase, done0_phase);
        } else if (causal) {
            run_c1_causal_zero_front(
                mutable_lds, owner, mmac_zero, raw0_phase);
        } else {
            run_consumer_q_tile<1, 0, false, false>(
                lds, mutable_lds, k_base, q_tile_begin, qi, causal,
                softmax_scale, n_owner, owner, lane, resident, mmac_zero,
                dv_acc, dk_acc, raw0_phase, done0_phase);
        }
        ++qi;
    }
    if (q_tile_count > 1) {
        if constexpr (Group == 0) {
            run_consumer_q_tile<0, 1, false, false>(
                lds, mutable_lds, k_base, q_tile_begin, qi, causal,
                softmax_scale, n_owner, owner, lane, resident, mmac_zero,
                dv_acc, dk_acc, raw1_phase, done1_phase);
        } else {
            run_consumer_q_tile<1, 1, true, false>(
                lds, mutable_lds, k_base, q_tile_begin, qi, causal,
                softmax_scale, n_owner, owner, lane, resident, mmac_zero,
                dv_acc, dk_acc, raw1_phase, done0_phase);
        }
        ++qi;
    }

#pragma clang loop unroll(disable)
    for (; qi + 1 < q_tile_count; qi += 2) {
        run_consumer_q_tile<Group, 0, true, false>(
            lds, mutable_lds, k_base, q_tile_begin, qi, 0,
            softmax_scale, n_owner, owner, lane, resident, mmac_zero,
            dv_acc, dk_acc, raw0_phase, done0_phase);
        if constexpr (Group == 0) {
            run_consumer_q_tile<0, 1, true, false>(
                lds, mutable_lds, k_base, q_tile_begin, qi + 1, 0,
                softmax_scale, n_owner, owner, lane, resident, mmac_zero,
                dv_acc, dk_acc, raw1_phase, done1_phase);
        } else {
            run_consumer_q_tile<1, 1, true, false>(
                lds, mutable_lds, k_base, q_tile_begin, qi + 1, 0,
                softmax_scale, n_owner, owner, lane, resident, mmac_zero,
                dv_acc, dk_acc, raw1_phase, done0_phase);
        }
    }
    if (qi < q_tile_count) {
        run_consumer_q_tile<Group, 0, true, false>(
            lds, mutable_lds, k_base, q_tile_begin, qi, 0,
            softmax_scale, n_owner, owner, lane, resident, mmac_zero,
            dv_acc, dk_acc, raw0_phase, done0_phase);
    }

    store_dkv_outputs(dk, dv, tensor_base, k_base, n_owner, lane, dk_acc,
                      dv_acc);
}

template <int Generation>
__device__ __forceinline__ void run_dq_writer_tile(
    const __half* lds,
    __half* dq_partial,
    int64_t partial_base,
    int q_tile_begin,
    int qi,
    int d_owner,
    int lane,
    const DqResidentFragments& resident,
    int& filled0_phase,
    int& filled1_phase) {
    Accumulator dq_acc[Tile::kMqPanels][2];
    zero_dq_writer_accumulators(dq_acc);

    ins::abarrier_try_wait<true>(Bar::kBatchDsFilled1, filled1_phase);
    update_dq_writer_group<1>(lds, resident, dq_acc);
    ins::abarrier_arrive_cnt<false>(Bar::kDqDone1, 1);

    if constexpr (Generation == 0) {
        ins::abarrier_try_wait<true>(Bar::kBatchDsFilled0, filled0_phase);
        update_dq_writer_group<0, 0>(lds, resident, dq_acc);
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone0, 1);
    } else {
        ins::abarrier_try_wait<true>(Bar::kBatchDsFilled0Alt, filled0_phase);
        update_dq_writer_group<0, 1>(lds, resident, dq_acc);
        ins::abarrier_arrive_cnt<false>(Bar::kDqDone0Alt, 1);
    }

    const int q_base = (q_tile_begin + qi) * Tile::kMq;
#pragma unroll
    for (int m_block = 0; m_block < Tile::kMqPanels; ++m_block) {
        store_dq_partial_d32(dq_partial, partial_base, q_base, m_block,
                             d_owner, lane, dq_acc[m_block]);
    }
}

__device__ __forceinline__ void run_dq_writer(
    const __half* lds,
    __half* dq_partial,
    int64_t partial_base,
    int q_tile_begin,
    int q_tile_count,
    int d_owner,
    int lane) {
    DqResidentFragments resident;
    latch_dq_k_normal(lds, d_owner, resident);
    int filled0_phase = 0;
    int filled0_alt_phase = 0;
    int filled1_phase = 0;

    int qi = 0;
#pragma clang loop unroll(disable)
    for (; qi + 1 < q_tile_count; qi += 2) {
        run_dq_writer_tile<0>(
            lds, dq_partial, partial_base, q_tile_begin, qi, d_owner, lane,
            resident, filled0_phase, filled1_phase);
        run_dq_writer_tile<1>(
            lds, dq_partial, partial_base, q_tile_begin, qi + 1, d_owner,
            lane, resident, filled0_alt_phase, filled1_phase);
    }
    if (qi < q_tile_count) {
        run_dq_writer_tile<0>(
            lds, dq_partial, partial_base, q_tile_begin, qi, d_owner, lane,
            resident, filled0_phase, filled1_phase);
    }
}

enum CtaOrderMode : int {
    kCtaOrderIdentity = 0,
    kCtaOrderSerpentine = 1,
};

__host__ __device__ __forceinline__ int remap_k_tile(
    int physical_k_tile, int k_tile_count, int order_mode,
    int order_width) {
    if (order_mode == kCtaOrderIdentity) {
        return physical_k_tile;
    }

    const int round = physical_k_tile / order_width;
    const int slot = physical_k_tile - round * order_width;
    const int base = round * order_width;
    int round_width = k_tile_count - base;
    if (round_width > order_width) {
        round_width = order_width;
    }
    return base + ((round & 1) == 0 ? slot : round_width - 1 - slot);
}

struct CtaOrderPlan {
    int mode;
    int width;
};

int modeled_causal_max_work(int k_tile_count, int bh_count, int mode,
                            int width) {
    int loads[Tile::kSingleDieCuCount] = {};
    const int remainder = bh_count % Tile::kSingleDieCuCount;
    const int q_tile_count =
        k_tile_count * Tile::kNk / Tile::kMq;
    for (int physical = 0; physical < k_tile_count; ++physical) {
        const int logical = remap_k_tile(
            physical, k_tile_count, mode, width);
        const int work =
            q_tile_count - logical * Tile::kNk / Tile::kMq;
        const int start =
            (physical * bh_count) % Tile::kSingleDieCuCount;
        for (int offset = 0; offset < remainder; ++offset) {
            const int cu = (start + offset) % Tile::kSingleDieCuCount;
            loads[cu] += work;
        }
    }

    int maximum = 0;
    for (int cu = 0; cu < Tile::kSingleDieCuCount; ++cu) {
        if (loads[cu] > maximum) {
            maximum = loads[cu];
        }
    }
    return maximum;
}

CtaOrderPlan choose_causal_cta_order(int k_tile_count, int bh_count,
                                     int causal) {
    CtaOrderPlan best{kCtaOrderIdentity, 1};
    if (causal == 0 ||
        bh_count % Tile::kSingleDieCuCount == 0) {
        return best;
    }

    int best_max = modeled_causal_max_work(
        k_tile_count, bh_count, best.mode, best.width);
    int max_width = k_tile_count;
    if (max_width > Tile::kSingleDieCuCount) {
        max_width = Tile::kSingleDieCuCount;
    }
    for (int width = 1; width <= max_width; ++width) {
        const int candidate_max = modeled_causal_max_work(
            k_tile_count, bh_count, kCtaOrderSerpentine, width);
        if (candidate_max < best_max) {
            best = CtaOrderPlan{kCtaOrderSerpentine, width};
            best_max = candidate_max;
        }
    }
    return best;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(Tile::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(Tile::kWavesPerCta)))
fa3_bwd_5gemm_kernel(const __half* __restrict__ dout,
                     const __half* __restrict__ q,
                     const __half* __restrict__ k,
                     const __half* __restrict__ v,
                     const float* __restrict__ packed_sidecar,
                     __half* __restrict__ dq_partial,
                     float* __restrict__ dk,
                     float* __restrict__ dv,
                     int heads_q,
                     int heads_kv,
                     int q_heads_per_kv,
                     int seqlen,
                     int causal,
                     int cta_order_mode,
                     int cta_order_width,
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
        __builtin_hcu_s_abarrier_init(Bar::kBatchDsFilled0Alt, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDqDone0Alt, 8);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < Tile::kConsumer0WaveBegin) {
        __builtin_hcu_s_set_vgpr_size(Wdra::kProducerVgprs);
        const int wave_local = wave & 3;
        const int lane = static_cast<int>(threadIdx.x % Tile::kWaveSize);
        const int q_head = static_cast<int>(blockIdx.x);
        const int q_bh =
            static_cast<int>(blockIdx.y) * heads_q + q_head;
        const int kv_head = q_heads_per_kv == 1
                                ? q_head
                                : q_head / q_heads_per_kv;
        const int kv_bh =
            static_cast<int>(blockIdx.y) * heads_kv + kv_head;
        const int logical_k_tile = remap_k_tile(
            static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z),
            cta_order_mode, cta_order_width);
        const int k_base = logical_k_tile * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t row_base =
            static_cast<int64_t>(q_bh) * seqlen;
        const int64_t q_tensor_base = row_base * Tile::kHeadDim;
        const int64_t kv_tensor_base =
            static_cast<int64_t>(kv_bh) * seqlen * Tile::kHeadDim;

        ins::abarrier_seq<false>(Bar::kResidentFilled);
        producer_load_resident_group<0>(k, v, lds, kv_tensor_base,
                                        k_base, wave_local);
        producer_load_resident_group<1>(k, v, lds, kv_tensor_base,
                                        k_base, wave_local);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled, 1);

        int kv_latched_phase = 0;
        int raw0_used_phase = 0;
        int raw1_used_phase = 0;
        if (q_tile_count != 0) {
            ins::abarrier_seq<false>(Bar::kRawFilled0);
            producer_load_raw_matrices(
                q, dout, lds, q_tensor_base,
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
                q, dout, lds, q_tensor_base, q_base,
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
        const int q_head = static_cast<int>(blockIdx.x);
        const int q_bh =
            static_cast<int>(blockIdx.y) * heads_q + q_head;
        const int logical_k_tile = remap_k_tile(
            static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z),
            cta_order_mode, cta_order_width);
        const int k_base = logical_k_tile * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t tensor_base =
            static_cast<int64_t>(q_bh) * seqlen * Tile::kHeadDim;
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
        const int q_head = static_cast<int>(blockIdx.x);
        const int q_bh =
            static_cast<int>(blockIdx.y) * heads_q + q_head;
        const int logical_k_tile = remap_k_tile(
            static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z),
            cta_order_mode, cta_order_width);
        const int k_base = logical_k_tile * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        const int64_t tensor_base =
            static_cast<int64_t>(q_bh) * seqlen * Tile::kHeadDim;
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
        const int q_head = static_cast<int>(blockIdx.x);
        const int bh =
            static_cast<int>(blockIdx.y) * heads_q + q_head;
        const int logical_k_tile = remap_k_tile(
            static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z),
            cta_order_mode, cta_order_width);
        const int k_base = logical_k_tile * Tile::kNk;
        const int q_tile_begin = causal ? k_base / Tile::kMq : 0;
        const int q_tile_count = seqlen / Tile::kMq - q_tile_begin;
        // Physical dispatch slots remain unique reduction owners even when
        // their logical K tiles are reordered for causal load balance.
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
        __builtin_hcu_s_abarrier_inv(Bar::kBatchDsFilled0Alt);
        __builtin_hcu_s_abarrier_inv(Bar::kDqDone0Alt);
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
    (void)heads_q;
    (void)heads_kv;
    (void)q_heads_per_kv;
    (void)seqlen;
    (void)causal;
    (void)cta_order_mode;
    (void)cta_order_width;
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
        params->seqlen_q % Tile::kNk != 0 || params->num_heads_kv <= 0 ||
        params->num_heads_q % params->num_heads_kv != 0) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }

    const size_t required_workspace = fused::dq_workspace_bytes(params);
    if (required_workspace == 0 || params->workspace == nullptr ||
        params->workspace_bytes < required_workspace) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int k_tile_count = params->seqlen_k / Tile::kNk;
    const int bh_count = params->batch * params->num_heads_q;
    const int q_heads_per_kv =
        params->num_heads_q / params->num_heads_kv;
    const CtaOrderPlan cta_order = choose_causal_cta_order(
        k_tile_count, bh_count, params->causal);
    const fused::FusedWorkspaceView workspace =
        fused::workspace_view(params->workspace, params);
    if (workspace.dq_partial == nullptr ||
        (q_heads_per_kv != 1 &&
         (workspace.dk_partial == nullptr || workspace.dv_partial == nullptr))) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    float* const dk_store =
        q_heads_per_kv == 1 ? static_cast<float*>(dk) : workspace.dk_partial;
    float* const dv_store =
        q_heads_per_kv == 1 ? static_cast<float*>(dv) : workspace.dv_partial;
    const dim3 grid(params->num_heads_q, params->batch, k_tile_count);
    hipLaunchKernelGGL(
        fa3_bwd_5gemm_kernel, grid, dim3(Tile::kThreadsPerCta), 0, 0,
        static_cast<const __half*>(dout), static_cast<const __half*>(q),
        static_cast<const __half*>(k), static_cast<const __half*>(v),
        static_cast<const float*>(packed_sidecar), workspace.dq_partial,
        dk_store, dv_store, params->num_heads_q, params->num_heads_kv,
        q_heads_per_kv, params->seqlen_q, params->causal, cta_order.mode,
        cta_order.width, params->softmax_scale);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return SHAOBO_FA3_STATUS_HIP_ERROR;
    }
    const int reduce_status = fused::launch_dq_reduction(
        workspace.dq_partial, static_cast<__half*>(dq), params);
    if (reduce_status != SHAOBO_FA3_STATUS_SUCCESS) {
        return reduce_status;
    }
    const int dkv_reduce_status = fused::launch_dkv_reduction(
        workspace.dk_partial, workspace.dv_partial, static_cast<float*>(dk),
        static_cast<float*>(dv), params);
    if (dkv_reduce_status != SHAOBO_FA3_STATUS_SUCCESS) {
        return dkv_reduce_status;
    }
    if (params->sync_after_launch != 0) {
        error = hipDeviceSynchronize();
        if (error != hipSuccess) {
            return SHAOBO_FA3_STATUS_HIP_ERROR;
        }
    }
    return SHAOBO_FA3_STATUS_SUCCESS;
}
