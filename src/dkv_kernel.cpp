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

inline bool valid_dkv_shape(const ShaoboFa3Params* p) {
    return p != nullptr && p->batch > 0 && p->seqlen_q > 0 &&
           p->seqlen_k > 0 && p->num_heads_q > 0 &&
           p->head_dim_qk == dkv::ActiveDkvTile::kHeadDim &&
           p->head_dim_v == dkv::ActiveDkvTile::kHeadDim &&
           p->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           p->layout == SHAOBO_FA3_LAYOUT_BHSD;
}

inline bool valid_canonical_shape(const ShaoboFa3Params* p) {
    return valid_dkv_shape(p) &&
           p->dkv_path == dkv::kDkvPathCanonicalDkv &&
           p->seqlen_q % dkv::ActiveDkvTile::kBlockMq == 0 &&
           p->seqlen_k == p->seqlen_q &&
           p->seqlen_k % dkv::ActiveDkvTile::kResidentNk == 0 &&
           p->num_heads_kv == p->num_heads_q &&
           p->causal == 1;
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

inline int64_t tensor_offset(
    int b, int h, int s, int d, int heads, int seqlen, int dim) {
    return ((static_cast<int64_t>(b) * heads + h) * seqlen + s) * dim + d;
}

template <typename Tile>
struct DkvLdsLayout {
    static constexpr int kKvBlockBytes = 32 * 32 * Tile::kHalfBytes;
    static constexpr int kRawBlockBytes = 16 * 32 * Tile::kHalfBytes;
    static constexpr int kRawPages = 1;
    static constexpr int kRawMBlocksPerMqTile = Tile::kBlockMq / 16;
    static constexpr int kDBlocksPerMqTile = Tile::kHeadDim / 32;
    static constexpr int kRawBlocksPerMqTile =
        kRawMBlocksPerMqTile * kDBlocksPerMqTile;
    static constexpr int kBlocksPerKvTile =
        Tile::kResidentNk / 32 * Tile::kHeadDim / 32;
    static constexpr int kQBase = 0;
    static constexpr int kDoutBase =
        kQBase + kRawPages * kRawBlocksPerMqTile * kRawBlockBytes;
    static constexpr int kRawEnd =
        kDoutBase + kRawPages * kRawBlocksPerMqTile * kRawBlockBytes;
    static constexpr int kKBase =
        Tile::kOverlayRawOnResidentKv ? 0 : kRawEnd;
    static constexpr int kVBase = kKBase + kBlocksPerKvTile * kKvBlockBytes;
    static constexpr int kKvEnd = kVBase + kBlocksPerKvTile * kKvBlockBytes;
    static constexpr int kSidecarBase =
        Tile::kOverlayRawOnResidentKv ? kRawEnd : kKvEnd;
    static constexpr int kSidecarEnd = kSidecarBase + Tile::kSidecarBytes;
    static constexpr int kBytes =
        Tile::kOverlayRawOnResidentKv
            ? (kKvEnd > kSidecarEnd ? kKvEnd : kSidecarEnd)
            : kSidecarEnd;
    static_assert(kSidecarBase % alignof(float) == 0,
                  "sidecar LDS base must be float-aligned");
    static_assert(kBytes <= Tile::kLdsBudgetBytes,
                  "canonical dKV LDS plan must fit 128KB");
};

template <typename Tile>
__device__ __forceinline__ int raw_page_block_offset(int page, int d_block) {
    return (page * DkvLdsLayout<Tile>::kRawBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kRawBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int raw_page_block_offset_m(
    int page,
    int m_block,
    int d_block) {
    return (page * DkvLdsLayout<Tile>::kRawBlocksPerMqTile +
            m_block * DkvLdsLayout<Tile>::kDBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kRawBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int kv_block_offset(int row_block, int d_block) {
    return (row_block * 4 + d_block) *
           DkvLdsLayout<Tile>::kKvBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int raw_page_for_q_tile(int q_tile) {
    static_assert(Tile::kRawBuffers == 1,
                  "canonical dKV route uses one raw Q/dO page");
    (void)q_tile;
    return 0;
}

template <typename Tile>
__device__ __forceinline__ float* sidecar_page_ptr(__half* lds, int page) {
    float* sidecar_base = reinterpret_cast<float*>(
        lds + DkvLdsLayout<Tile>::kSidecarBase / sizeof(__half));
    return sidecar_base + page * Tile::kSidecarFloats;
}

template <typename Tile>
__device__ __forceinline__ const float* sidecar_page_ptr(
    const __half* lds,
    int page) {
    const float* sidecar_base = reinterpret_cast<const float*>(
        lds + DkvLdsLayout<Tile>::kSidecarBase / sizeof(__half));
    return sidecar_base + page * Tile::kSidecarFloats;
}

template <typename Wdra>
__device__ __forceinline__ void wait_resident_used(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kResidentUsed, phase);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_resident_used() {
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentUsed, 1);
}

template <typename Wdra>
__device__ __forceinline__ void seq_raw_filled() {
    ins::abarrier_seq<false>(Wdra::kRawFilled);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_raw_filled() {
    ins::abarrier_arrive_cnt<false>(Wdra::kRawFilled, 1);
}

template <typename Wdra>
__device__ __forceinline__ void wait_raw_filled(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kRawFilled, phase);
}

template <typename Wdra>
__device__ __forceinline__ void wait_raw_used(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kRawUsed, phase);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_raw_used() {
    ins::abarrier_arrive_cnt<false>(Wdra::kRawUsed, 1);
}

template <typename Tile, int Half>
__device__ __forceinline__ void publish_mq_half_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    int q_base,
    int page,
    int wave_local) {
    static_assert(Tile::kBlockMq == 128,
                  "half-page conveyor currently targets Mq128");
    static_assert(Half == 0 || Half == 1, "half must be 0 or 1");
    constexpr int kMBlockStart = Half * 4;
#pragma unroll
    for (int local_m_block = 0; local_m_block < 4; ++local_m_block) {
        const int m_block = kMBlockStart + local_m_block;
        const __half* src_tile =
            src + static_cast<int64_t>(q_base + m_block * 16) * row_stride +
            wave_local * 32;
        ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, row_stride);
        ins::matrix_load_32x16_b16_bps_lds(
            lds, srsrc,
            lds_base +
                raw_page_block_offset_m<Tile>(page, m_block, wave_local));
    }
}

template <typename Tile, int Half>
__device__ __forceinline__ void publish_sidecar_half_tile_to_lds(
    __half* lds,
    const float* packed_sidecar,
    int64_t row_base,
    int q_base,
    int seqlen,
    int page,
    int wave_local,
    int lane) {
    static_assert(Tile::kBlockMq == 128,
                  "half-page sidecar currently targets Mq128");
    static_assert(Half == 0 || Half == 1, "half must be 0 or 1");
    if (wave_local != Half) {
        return;
    }
    const int local_row = Half * 64 + lane;
    float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
    const int q_row = q_base + local_row;
    float row_max_log2 = 0.0f;
    float row_inv_sum = 0.0f;
    float row_delta = 0.0f;
    if (q_row < seqlen && packed_sidecar != nullptr) {
        const float* row =
            packed_sidecar +
            (row_base + q_row) * Tile::kPackedSidecarFields;
        row_max_log2 = row[0];
        row_inv_sum = row[1];
        row_delta = row[2];
    }
    sidecar_page[Tile::kSidecarMaxLog2Base + local_row] = row_max_log2;
    sidecar_page[Tile::kSidecarInvSumBase + local_row] = row_inv_sum;
    sidecar_page[Tile::kSidecarDeltaBase + local_row] = row_delta;
}

template <typename Tile>
__device__ __forceinline__ void publish_resident_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    int k_base,
    int wave_local) {
#pragma unroll
    for (int row_block = 0; row_block < Tile::kResidentNk / 32;
         ++row_block) {
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
__device__ __forceinline__ void producer_k_q_dout_loop(
    const __half* q_base_ptr,
    const __half* dout_base_ptr,
    const __half* k_base_ptr,
    __half* lds,
    int q_base,
    int k_base,
    int seqlen,
    int row_stride,
    int q_tiles,
    int wave_local,
    const float* packed_sidecar = nullptr,
    int64_t row_base = 0,
    int lane = 0) {
    using Layout = DkvLdsLayout<Tile>;
    int raw_used_phase = 0;
    int resident_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_resident_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);
    if constexpr (Tile::kOverlayRawOnResidentKv) {
        wait_resident_used<Wdra>(resident_used_phase);
    }

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = raw_page_for_q_tile<Tile>(q_tile);
        const int packet_q_base = q_base + q_tile * Tile::kBlockMq;
        if (q_tile >= Tile::kRawBuffers) {
            wait_raw_used<Wdra>(raw_used_phase);
        }
        seq_raw_filled<Wdra>();
        publish_mq_half_tile<Tile, 0>(
            lds, Layout::kQBase, q_base_ptr, row_stride, packet_q_base, page,
            wave_local);
        publish_mq_half_tile<Tile, 1>(
            lds, Layout::kQBase, q_base_ptr, row_stride, packet_q_base, page,
            wave_local);
        publish_mq_half_tile<Tile, 0>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride, packet_q_base,
            page, wave_local);
        publish_mq_half_tile<Tile, 1>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride, packet_q_base,
            page, wave_local);
        publish_sidecar_half_tile_to_lds<Tile, 0>(
            lds, packed_sidecar, row_base, packet_q_base, seqlen, page,
            wave_local, lane);
        publish_sidecar_half_tile_to_lds<Tile, 1>(
            lds, packed_sidecar, row_base, packet_q_base, seqlen, page,
            wave_local, lane);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        arrive_raw_filled<Wdra>();
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void publish_resident_v(
    const __half* v_base_ptr,
    __half* lds,
    int k_base,
    int row_stride,
    int wave_local) {
    using Layout = DkvLdsLayout<Tile>;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_resident_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);
}

struct Owner16KvRegs {
    ins::F16x8 k[4];
    ins::F16x8 v[4];
};

template <typename Tile>
__device__ __forceinline__ void latch_owner16_kv_dblock(
    __half* lds,
    int row_block32,
    int owner_half,
    int d_block,
    ins::F16x8& k_out,
    ins::F16x8& v_out) {
    using Layout = DkvLdsLayout<Tile>;
    ins::F16x8 k0;
    ins::F16x8 k1;
    ins::F16x8 v0;
    ins::F16x8 v1;
    const int k_off =
        Layout::kKBase + kv_block_offset<Tile>(row_block32, d_block);
    const int v_off =
        Layout::kVBase + kv_block_offset<Tile>(row_block32, d_block);
    ins::ds_read_matrix_trans_pair(lds, k_off, k0.f16x8, k1.f16x8);
    ins::ds_read_matrix_trans_pair(lds, v_off, v0.f16x8, v1.f16x8);
    ins::wait_lgkm(0);
    k_out.f16x8 = owner_half == 0 ? k0.f16x8 : k1.f16x8;
    v_out.f16x8 = owner_half == 0 ? v0.f16x8 : v1.f16x8;
}

template <typename Tile>
__device__ __forceinline__ void latch_owner16_kv_regs(
    __half* lds,
    int owner_nblock,
    Owner16KvRegs& regs) {
    const int row_block32 = owner_nblock >> 1;
    const int owner_half = owner_nblock & 1;
#pragma unroll
    for (int d_block = 0; d_block < 4; ++d_block) {
        latch_owner16_kv_dblock<Tile>(
            lds, row_block32, owner_half, d_block, regs.k[d_block],
            regs.v[d_block]);
    }
}

struct Owner16ScoreDpSources {
    ins::F16x8 q_d0;
    ins::F16x8 dout_d0;
    ins::F16x8 q_d1;
    ins::F16x8 dout_d1;
    ins::F16x8 q_d2;
    ins::F16x8 dout_d2;
    ins::F16x8 q_d3;
    ins::F16x8 dout_d3;
};

template <typename Tile, int MBlockBase>
__device__ __forceinline__ void read_score_dp_owner16_all_dblocks(
    __half* lds,
    int page,
    Owner16ScoreDpSources& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(MBlockBase < Layout::kRawMBlocksPerMqTile,
                  "score/dP reads one M16 block");
    static_assert(Layout::kRawPages == 1,
                  "immediate score/dP reads require one canonical raw page");
    (void)page;

    constexpr int kQD0Off =
        Layout::kQBase +
        (MBlockBase * Layout::kDBlocksPerMqTile + 0) *
            Layout::kRawBlockBytes;
    constexpr int kDoutD0Off =
        Layout::kDoutBase +
        (MBlockBase * Layout::kDBlocksPerMqTile + 0) *
            Layout::kRawBlockBytes;
    constexpr int kQD1Off = kQD0Off + Layout::kRawBlockBytes;
    constexpr int kDoutD1Off = kDoutD0Off + Layout::kRawBlockBytes;
    constexpr int kQD2Off = kQD1Off + Layout::kRawBlockBytes;
    constexpr int kDoutD2Off = kDoutD1Off + Layout::kRawBlockBytes;
    constexpr int kQD3Off = kQD2Off + Layout::kRawBlockBytes;
    constexpr int kDoutD3Off = kDoutD2Off + Layout::kRawBlockBytes;

    ins::ds_read_matrix_32x16_trans_imm4<
        kQD0Off,
        kDoutD0Off,
        kQD1Off,
        kDoutD1Off>(lds, src.q_d0.f16x8, src.dout_d0.f16x8,
                    src.q_d1.f16x8, src.dout_d1.f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<
        kQD2Off,
        kDoutD2Off,
        kQD3Off,
        kDoutD3Off>(lds, src.q_d2.f16x8, src.dout_d2.f16x8,
                    src.q_d3.f16x8, src.dout_d3.f16x8);
}

template <int DBlock>
__device__ __forceinline__ void score_dp_mmac_owner16_dblock(
    const Owner16KvRegs& kv_regs,
    const ins::F16x8& q_frag,
    const ins::F16x8& dout_frag,
    const ins::F16x8& zero,
    ins::F32x4& score,
    ins::F32x4& dp) {
    static_assert(DBlock >= 0 && DBlock < 4,
                  "score/dP DBlock must be in D128");
    const ins::F16x8& k_reg = kv_regs.k[DBlock];
    const ins::F16x8& v_reg = kv_regs.v[DBlock];
#pragma unroll
    for (int k_half = 0; k_half < 2; ++k_half) {
        const bool first = DBlock == 0 && k_half == 0;
        score.f32 = ins::mmac_f16_lit(
            k_reg.f16x4[k_half], q_frag.f16x4[k_half],
            first ? zero.f32 : score.f32);
        dp.f32 = ins::mmac_f16_lit(
            v_reg.f16x4[k_half], dout_frag.f16x4[k_half],
            first ? zero.f32 : dp.f32);
    }
}

template <typename Tile, int MBlockBase>
__device__ __forceinline__ void score_dp_mmac_owner16(
    __half* lds,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4& score,
    ins::F32x4& dp) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(MBlockBase < Layout::kRawMBlocksPerMqTile,
                  "score/dP reads one M16 block");

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    Owner16ScoreDpSources src;
    read_score_dp_owner16_all_dblocks<Tile, MBlockBase>(lds, page, src);

    // Eight reads are issued in D0..D3 order. Retire each Q/dO pair at its
    // first use while later D blocks remain in flight behind the MMAC island.
    ins::wait_lgkm(6);
    ins::raise_priority_2();
    score_dp_mmac_owner16_dblock<0>(
        kv_regs, src.q_d0, src.dout_d0, zero, score, dp);
    ins::wait_lgkm(4);
    score_dp_mmac_owner16_dblock<1>(
        kv_regs, src.q_d1, src.dout_d1, zero, score, dp);
    ins::wait_lgkm(2);
    score_dp_mmac_owner16_dblock<2>(
        kv_regs, src.q_d2, src.dout_d2, zero, score, dp);
    ins::wait_lgkm(0);
    score_dp_mmac_owner16_dblock<3>(
        kv_regs, src.q_d3, src.dout_d3, zero, score, dp);
    ins::lower_priority();
}

__device__ __forceinline__ void softmax_ds_owner16(
    const ins::F32x4& score,
    const ins::F32x4& dp,
    const ins::Vec4F32& row_max_log2,
    const ins::Vec4F32& row_inv_sum,
    const ins::Vec4F32& row_delta,
    int q_m_base,
    int owner_krow,
    int lane_col_group,
    float softmax_scale,
    float softmax_scale_log2,
    ins::Vec4F16& p_frag,
    ins::Vec4F16& ds_frag) {
    const int local_m_base = lane_col_group * 4;
#pragma unroll
    for (int vec_id = 0; vec_id < 4; ++vec_id) {
        const int qrow = q_m_base + local_m_base + vec_id;
        const bool valid_pair = owner_krow <= qrow;
        const float p_unmasked =
            exp2f(score.scalar[vec_id] * softmax_scale_log2 -
                  row_max_log2[vec_id]) *
            row_inv_sum[vec_id];
        const float p_val = valid_pair ? p_unmasked : 0.0f;
        const float ds_val =
            p_val * (dp.scalar[vec_id] - row_delta[vec_id]) *
            softmax_scale;
        p_frag[vec_id] = static_cast<_Float16>(p_val);
        ds_frag[vec_id] = static_cast<_Float16>(ds_val);
    }
}

template <typename Tile, int MBlockBase>
__device__ __forceinline__ void softmax_ds_owner16_causal_exact_tile(
    const ins::F32x4& score,
    const ins::F32x4& dp,
    const float* sidecar_page,
    int q_m_base,
    int owner_krow,
    int lane_col_group,
    float softmax_scale,
    float softmax_scale_log2,
    ins::Vec4F16& p_frag,
    ins::Vec4F16& ds_frag) {
    static_assert(MBlockBase < DkvLdsLayout<Tile>::kRawMBlocksPerMqTile,
                  "softmax/dS consumes one M16 block");
    const int local_m_base = MBlockBase * 16 + lane_col_group * 4;
    const ins::Vec4F32 row_max_log2 =
        *reinterpret_cast<const ins::Vec4F32*>(
            sidecar_page + Tile::kSidecarMaxLog2Base + local_m_base);
    const ins::Vec4F32 row_inv_sum =
        *reinterpret_cast<const ins::Vec4F32*>(
            sidecar_page + Tile::kSidecarInvSumBase + local_m_base);
    const ins::Vec4F32 row_delta =
        *reinterpret_cast<const ins::Vec4F32*>(
            sidecar_page + Tile::kSidecarDeltaBase + local_m_base);

    softmax_ds_owner16(
        score, dp, row_max_log2, row_inv_sum, row_delta, q_m_base,
        owner_krow, lane_col_group, softmax_scale, softmax_scale_log2,
        p_frag, ds_frag);
}

struct Owner16DvDkSources {
    ins::F16x8 dout_d0;
    ins::F16x8 dout_d1;
    ins::F16x8 q_d0;
    ins::F16x8 q_d1;
};

template <typename Tile, int DBlockBase, int MBlockBase>
__device__ __forceinline__ void read_owner16_dv_dk_sources(
    __half* lds,
    int page,
    Owner16DvDkSources& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK source group must cover two D blocks");
    static_assert(MBlockBase < Layout::kRawMBlocksPerMqTile,
                  "dV/dK reads one M16 block");

    constexpr int kDoutD0Off =
        Layout::kDoutBase +
        (MBlockBase * Layout::kDBlocksPerMqTile + DBlockBase) *
            Layout::kRawBlockBytes;
    constexpr int kDoutD1Off =
        Layout::kDoutBase +
        (MBlockBase * Layout::kDBlocksPerMqTile + DBlockBase + 1) *
            Layout::kRawBlockBytes;
    constexpr int kQD0Off =
        Layout::kQBase +
        (MBlockBase * Layout::kDBlocksPerMqTile + DBlockBase) *
            Layout::kRawBlockBytes;
    constexpr int kQD1Off =
        Layout::kQBase +
        (MBlockBase * Layout::kDBlocksPerMqTile + DBlockBase + 1) *
            Layout::kRawBlockBytes;
    static_assert(Layout::kRawPages == 1,
                  "owner16 immediate reads require one raw page");
    (void)page;

    ins::ds_read_matrix_32x16_normal_imm4<
        kDoutD0Off,
        kDoutD1Off,
        kQD0Off,
        kQD1Off>(lds, src.dout_d0.f16x8, src.dout_d1.f16x8,
                 src.q_d0.f16x8, src.q_d1.f16x8);
}

template <bool FirstAccum, int OutIdx, int DHalf>
__device__ __forceinline__ void owner16_dv_dk_mmac_one_out(
    const ins::Vec4F16& p_frag,
    const ins::Vec4F16& ds_frag,
    const ins::F16x8& dout,
    const ins::F16x8& q,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
    static_assert(DHalf == 0 || DHalf == 1,
                  "32x16 raw page has two D16 halves");
    if constexpr (FirstAccum) {
        dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
            p_frag, dout.f16x4[DHalf], zero_f16.f32);
        dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
            ds_frag, q.f16x4[DHalf], zero_f16.f32);
    } else {
        dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
            p_frag, dout.f16x4[DHalf], dv_acc[OutIdx].f32);
        dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
            ds_frag, q.f16x4[DHalf], dk_acc[OutIdx].f32);
    }
}

template <bool FirstAccum, int OutBase>
__device__ __forceinline__ void owner16_dv_dk_mmac_four_out(
    const ins::Vec4F16& p_frag,
    const ins::Vec4F16& ds_frag,
    const Owner16DvDkSources& src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
    owner16_dv_dk_mmac_one_out<FirstAccum, OutBase + 0, 0>(
        p_frag, ds_frag, src.dout_d0, src.q_d0, dv_acc, dk_acc, zero_f16);
    owner16_dv_dk_mmac_one_out<FirstAccum, OutBase + 1, 1>(
        p_frag, ds_frag, src.dout_d0, src.q_d0, dv_acc, dk_acc, zero_f16);
    owner16_dv_dk_mmac_one_out<FirstAccum, OutBase + 2, 0>(
        p_frag, ds_frag, src.dout_d1, src.q_d1, dv_acc, dk_acc, zero_f16);
    owner16_dv_dk_mmac_one_out<FirstAccum, OutBase + 3, 1>(
        p_frag, ds_frag, src.dout_d1, src.q_d1, dv_acc, dk_acc, zero_f16);
}

template <typename Tile,
          typename Wdra,
          int MBlockBase,
          bool FirstAccum,
          bool ReleaseRaw>
__device__ __forceinline__ void owner16_dv_dk_read_mmac(
    __half* lds,
    const ins::Vec4F16& p_frag,
    const ins::Vec4F16& ds_frag,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    int page) {
    ins::F16x8 zero_f16;
    if constexpr (FirstAccum) {
        ins::zero_f16x8(zero_f16);
    }

    Owner16DvDkSources src;
    read_owner16_dv_dk_sources<Tile, 0, MBlockBase>(lds, page, src);

    ins::wait_lgkm(0);
    ins::raise_priority_2();
    owner16_dv_dk_mmac_four_out<FirstAccum, 0>(
        p_frag, ds_frag, src, dv_acc, dk_acc, zero_f16);

    read_owner16_dv_dk_sources<Tile, 2, MBlockBase>(lds, page, src);
    ins::wait_lgkm(0);
    if constexpr (ReleaseRaw) {
        arrive_raw_used<Wdra>();
    }
    owner16_dv_dk_mmac_four_out<FirstAccum, 4>(
        p_frag, ds_frag, src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <typename Tile>
__device__ __forceinline__ void store_dkv_owner16(
    float* dk,
    float* dv,
    const ins::F32x4 (&dk_acc)[8],
    const ins::F32x4 (&dv_acc)[8],
    int owner_krow,
    int lane_col_group,
    int64_t tensor_base) {
    if (dk == nullptr || dv == nullptr) {
        return;
    }
    const int64_t out_row =
        tensor_base + static_cast<int64_t>(owner_krow) * Tile::kHeadDim;
#pragma unroll
    for (int d_idx = 0; d_idx < 8; ++d_idx) {
        const int d_base = d_idx * 16 + lane_col_group * 4;
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            dk[out_row + d_base + vec_id] = dk_acc[d_idx].scalar[vec_id];
            dv[out_row + d_base + vec_id] = dv_acc[d_idx].scalar[vec_id];
        }
    }
}

template <typename Tile,
          typename Wdra,
          int MBlockBase,
          bool FirstAccum,
          bool ReleaseRaw>
__device__ __forceinline__ void consume_m16_owner16_causal_exact_tile(
    __half* lds,
    int q_tile_base,
    int owner_krow,
    int lane_col_group,
    float softmax_scale,
    float softmax_scale_log2,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    ins::Vec4F16 p_frag;
    ins::Vec4F16 ds_frag;
    {
        ins::F32x4 score;
        ins::F32x4 dp;
        score_dp_mmac_owner16<Tile, MBlockBase>(
            lds, page, kv_regs, score, dp);
        const float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
        softmax_ds_owner16_causal_exact_tile<Tile, MBlockBase>(
            score, dp, sidecar_page, q_tile_base + MBlockBase * 16,
            owner_krow, lane_col_group, softmax_scale, softmax_scale_log2,
            p_frag, ds_frag);
    }

    owner16_dv_dk_read_mmac<
        Tile,
        Wdra,
        MBlockBase,
        FirstAccum,
        ReleaseRaw>(lds, p_frag, ds_frag, dv_acc, dk_acc, page);
}

template <typename Tile, typename Wdra, bool FirstQTile>
__device__ __forceinline__ void consume_q_tile_owner16(
    __half* lds,
    int q_tile_base,
    int owner_krow,
    int lane_col_group,
    float softmax_scale,
    float softmax_scale_log2,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    consume_m16_owner16_causal_exact_tile<
        Tile, Wdra, 0, FirstQTile, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 1, false, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 2, false, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 3, false, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 4, false, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 5, false, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 6, false, false>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    consume_m16_owner16_causal_exact_tile<Tile, Wdra, 7, false, true>(
        lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
        softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
}

template <typename Tile, typename Wdra, int ConsumerGroup>
__device__ __forceinline__ void consumer_dkv_mmac_loop(
    __half* lds,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int k_base,
    int q_tiles,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_filled_phase = 0;
    int raw_filled_phase = 0;
    static_assert(ConsumerGroup >= 0 && ConsumerGroup < 3,
                  "dKV owner16 consumer group must be 0, 1, or 2");
    (void)causal;
    const int owner_nblock = ConsumerGroup * 4 + wave_local;
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int owner_krow =
        k_base + owner_nblock * Tile::kNkPerConsumerWave + lane_n;
    const float softmax_scale_log2 = softmax_scale * kLog2E;

    Owner16KvRegs kv_regs;
    ins::abarrier_try_wait<true>(
        Wdra::kResidentFilled, resident_filled_phase);
    latch_owner16_kv_regs<Tile>(lds, owner_nblock, kv_regs);
    if constexpr (Tile::kOverlayRawOnResidentKv) {
        arrive_resident_used<Wdra>();
    }

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];
    if (q_tiles > 0) {
        wait_raw_filled<Wdra>(raw_filled_phase);
        consume_q_tile_owner16<Tile, Wdra, true>(
            lds, 0, owner_krow, lane_col_group, softmax_scale,
            softmax_scale_log2, raw_page_for_q_tile<Tile>(0), kv_regs, dv_acc,
            dk_acc);
    }

#pragma clang loop unroll(disable)
    for (int q_tile = 1; q_tile < q_tiles; ++q_tile) {
        const int page = raw_page_for_q_tile<Tile>(q_tile);
        const int q_tile_base = q_tile * Tile::kBlockMq;
        wait_raw_filled<Wdra>(raw_filled_phase);
        consume_q_tile_owner16<Tile, Wdra, false>(
            lds, q_tile_base, owner_krow, lane_col_group, softmax_scale,
            softmax_scale_log2, page, kv_regs, dv_acc, dk_acc);
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, owner_krow, lane_col_group, tensor_base);
}
__global__ void __launch_bounds__(dkv::ActiveDkvTile::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(16)))
fa3_bwd_dkv_kernel(const __half* __restrict__ dout,
                   const __half* __restrict__ q,
                   const __half* __restrict__ k,
                   const __half* __restrict__ v,
                   const float* __restrict__ packed_sidecar,
                   float* __restrict__ dk,
                   float* __restrict__ dv,
                   int heads,
                   int seqlen,
                   int dim,
                   int causal,
                   float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::ActiveDkvTile;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV 16-wave LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 8);
        if constexpr (Tile::kOverlayRawOnResidentKv) {
            __builtin_hcu_s_abarrier_init(Bar::kResidentUsed, 12);
        }
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed, 12);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        producer_k_q_dout_loop<Tile, Bar>(
            q + tensor_base, dout + tensor_base, k + tensor_base, lds, q_base,
            k_base, seqlen, dim, q_tiles, wave_local, packed_sidecar, row_base,
            lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        consumer_dkv_mmac_loop<Tile, Bar, 0>(
            lds, dk, dv, tensor_base, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        consumer_dkv_mmac_loop<Tile, Bar, 1>(
            lds, dk, dv, tensor_base, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        publish_resident_v<Tile, Bar>(
            v + tensor_base, lds, k_base, dim, wave_local);
        consumer_dkv_mmac_loop<Tile, Bar, 2>(
            lds, dk, dv, tensor_base, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __syncthreads();
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
        if constexpr (Tile::kOverlayRawOnResidentKv) {
            __builtin_hcu_s_abarrier_inv(Bar::kResidentUsed);
        }
        __builtin_hcu_s_abarrier_inv(Bar::kRawFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kRawUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)dout;
    (void)q;
    (void)k;
    (void)v;
    (void)packed_sidecar;
    (void)dk;
    (void)dv;
    (void)heads;
    (void)seqlen;
    (void)dim;
    (void)causal;
    (void)softmax_scale;
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
    return 0;
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

    if (!valid_canonical_shape(params)) {
        return SHAOBO_FA3_STATUS_UNSUPPORTED;
    }
    if (dout == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        dk == nullptr || dv == nullptr ||
        params->reserved_ptr[3] == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int grid_x =
        ceil_div(params->seqlen_k, dkv::ActiveDkvTile::kResidentNk);
    dim3 grid(grid_x, params->num_heads_q, params->batch);
    dim3 block(dkv::ActiveDkvTile::kThreadsPerCta);

    hipLaunchKernelGGL(
        fa3_bwd_dkv_kernel, grid, block, 0, 0,
        static_cast<const __half*>(dout), static_cast<const __half*>(q),
        static_cast<const __half*>(k), static_cast<const __half*>(v),
        static_cast<const float*>(params->reserved_ptr[3]),
        static_cast<float*>(dk), static_cast<float*>(dv),
        params->num_heads_q, params->seqlen_k, params->head_dim_qk,
        params->causal, params->softmax_scale);
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

void cpu_pair_prob_ds(const std::vector<__half>& q,
                      const std::vector<__half>& k,
                      const std::vector<__half>& v,
                      const std::vector<__half>& dout,
                      int batch,
                      int heads,
                      int seqlen,
                      int dim,
                      int causal,
                      float scale,
                      int b,
                      int h,
                      int q_idx,
                      int k_idx,
                      float& prob_out,
                      float& ds_out) {
    if (causal && k_idx > q_idx) {
        prob_out = 0.0f;
        ds_out = 0.0f;
        return;
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (int kk = 0; kk < seqlen; ++kk) {
        if (causal && kk > q_idx) {
            continue;
        }
        float score = 0.0f;
        for (int d = 0; d < dim; ++d) {
            score += __half2float(
                         q[tensor_offset(b, h, q_idx, d, heads, seqlen,
                                         dim)]) *
                     __half2float(
                         k[tensor_offset(b, h, kk, d, heads, seqlen, dim)]);
        }
        max_score = std::max(max_score, score * scale);
    }

    float denom = 0.0f;
    float target_unscaled_prob = 0.0f;
    float target_dp = 0.0f;
    float delta_numer = 0.0f;
    for (int kk = 0; kk < seqlen; ++kk) {
        if (causal && kk > q_idx) {
            continue;
        }
        float score = 0.0f;
        float dp = 0.0f;
        for (int d = 0; d < dim; ++d) {
            score += __half2float(
                         q[tensor_offset(b, h, q_idx, d, heads, seqlen,
                                         dim)]) *
                     __half2float(
                         k[tensor_offset(b, h, kk, d, heads, seqlen, dim)]);
            dp += __half2float(
                      dout[tensor_offset(b, h, q_idx, d, heads, seqlen,
                                         dim)]) *
                  __half2float(
                      v[tensor_offset(b, h, kk, d, heads, seqlen, dim)]);
        }
        const float unscaled_prob = std::exp(score * scale - max_score);
        denom += unscaled_prob;
        delta_numer += unscaled_prob * dp;
        if (kk == k_idx) {
            target_unscaled_prob = unscaled_prob;
            target_dp = dp;
        }
    }
    const float inv_denom = 1.0f / denom;
    const float prob = target_unscaled_prob * inv_denom;
    const float delta = delta_numer * inv_denom;
    prob_out = prob;
    ds_out = prob * (target_dp - delta) * scale;
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
                std::vector<float> probs(seqlen, 0.0f);
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
                    const float p_num = std::exp2((dot - max_qk) * scale_log2);
                    probs[k_idx] = p_num;
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
                        out_d += probs[k_idx] * inv_sum *
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
    const bool canonical_check = check == 0;
    const int batch = arg_int(argc, argv, "--B", env_int("B", kDefaultBatch));
    const int heads = arg_int(argc, argv, "--H", env_int("H", kDefaultHeads));
    const int default_seq = 128;
    const int seqlen = arg_int(argc, argv, "--S", env_int("S", default_seq));
    const int dim = arg_int(argc, argv, "--D", env_int("D", kDefaultDim));
    const int causal = arg_int(argc, argv, "--causal", env_int("CAUSAL", 1));

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
    params.dkv_path = dkv::kDkvPathCanonicalDkv;
    if (check) {
        params.dkv_path = dkv::kDkvPathReferenceCorrectness;
    }
    params.block_threads = dkv::ActiveDkvTile::kThreadsPerCta;
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
    float* scores_max_dev = nullptr;
    float* scores_sum_dev = nullptr;
    float* delta_dev = nullptr;
    float* packed_sidecar_dev = nullptr;
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
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dk_dev), output_bytes);
    }
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dv_dev), output_bytes);
    }
    if (err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&workspace),
                        std::max<size_t>(workspace_bytes, sizeof(float)));
    }
    const size_t sidecar_rows =
        static_cast<size_t>(batch) * static_cast<size_t>(heads) *
        static_cast<size_t>(seqlen);
    const size_t sidecar_bytes = sidecar_rows * sizeof(float);
    if (canonical_check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&scores_max_dev), sidecar_bytes);
    }
    if (canonical_check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&scores_sum_dev), sidecar_bytes);
    }
    if (canonical_check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&delta_dev), sidecar_bytes);
    }
    const size_t packed_sidecar_bytes =
        sidecar_rows * dkv::ActiveDkvTile::kPackedSidecarFields *
        sizeof(float);
    if (canonical_check && err == hipSuccess) {
        err = hipMalloc(
            reinterpret_cast<void**>(&packed_sidecar_dev),
            packed_sidecar_bytes);
    }
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        return 2;
    }

    std::vector<__half> q_host(tensor_elems);
    std::vector<__half> k_host(tensor_elems);
    std::vector<__half> v_host(tensor_elems);
    std::vector<__half> dout_host(tensor_elems);
    {
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
        if (canonical_check) {
            std::vector<float> scores_max_host(sidecar_rows);
            std::vector<float> scores_sum_host(sidecar_rows);
            std::vector<float> delta_host(sidecar_rows);
            cpu_softmax_aux_delta(
                q_host, k_host, v_host, dout_host, scores_max_host,
                scores_sum_host, delta_host, batch, heads, seqlen, dim,
                params.causal, params.softmax_scale);
            ignore_hip_status(
                hipMemcpy(scores_max_dev, scores_max_host.data(),
                          sidecar_bytes, hipMemcpyHostToDevice));
            ignore_hip_status(
                hipMemcpy(scores_sum_dev, scores_sum_host.data(),
                          sidecar_bytes, hipMemcpyHostToDevice));
            ignore_hip_status(
                hipMemcpy(delta_dev, delta_host.data(), sidecar_bytes,
                          hipMemcpyHostToDevice));
            params.reserved_ptr[0] = delta_dev;
            std::vector<float> packed_sidecar_host(
                sidecar_rows *
                dkv::ActiveDkvTile::kPackedSidecarFields);
            for (size_t row = 0; row < sidecar_rows; ++row) {
                const size_t base =
                    row *
                    dkv::ActiveDkvTile::kPackedSidecarFields;
                packed_sidecar_host[base + 0] =
                    scores_max_host[row] * params.softmax_scale * kLog2E;
                packed_sidecar_host[base + 1] =
                    scores_sum_host[row] != 0.0f
                        ? 1.0f / scores_sum_host[row]
                        : 0.0f;
                packed_sidecar_host[base + 2] = delta_host[row];
            }
            ignore_hip_status(
                hipMemcpy(packed_sidecar_dev, packed_sidecar_host.data(),
                          packed_sidecar_bytes, hipMemcpyHostToDevice));
            params.reserved_ptr[3] = packed_sidecar_dev;
        }
    }
    ignore_hip_status(
        hipMemset(workspace, 0,
                  std::max<size_t>(workspace_bytes, sizeof(float))));
    params.workspace = workspace;
    params.workspace_bytes = workspace_bytes;

    const int status =
        shaobo_fa3_bwd(dout_dev, q_dev, k_dev, v_dev, nullptr,
                       scores_max_dev, scores_sum_dev, nullptr, dk_dev,
                       dv_dev, &params);
    {
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
        const float rel_l2_limit = canonical_check ? 5.0e-3f : 1.0e-4f;
        const float rmse_limit = canonical_check ? 5.0e-8f : 1.0e-9f;
        const bool dk_error_ok =
            dk_metrics.rel_l2 <= rel_l2_limit || dk_metrics.rmse <= rmse_limit;
        const bool dv_error_ok =
            dv_metrics.rel_l2 <= rel_l2_limit || dv_metrics.rmse <= rmse_limit;
        const bool pass =
            status == SHAOBO_FA3_STATUS_SUCCESS &&
            dk_metrics.bad_count == 0 && dv_metrics.bad_count == 0 &&
            dk_metrics.max_abs <= 5.0e-4f &&
            dv_metrics.max_abs <= 5.0e-4f &&
            dk_error_ok && dv_error_ok;

        ignore_hip_status(hipFree(delta_dev));
        ignore_hip_status(hipFree(scores_sum_dev));
        ignore_hip_status(hipFree(scores_max_dev));
        ignore_hip_status(hipFree(packed_sidecar_dev));
        ignore_hip_status(hipFree(workspace));
        ignore_hip_status(hipFree(dv_dev));
        ignore_hip_status(hipFree(dk_dev));
        ignore_hip_status(hipFree(dout_dev));
        ignore_hip_status(hipFree(v_dev));
        ignore_hip_status(hipFree(k_dev));
        ignore_hip_status(hipFree(q_dev));

        std::printf(
            "%s status=%s B=%d H=%d S=%d D=%d causal=%d "
            "workspace_bytes=%zu dk_max_abs=%g dk_mean_abs=%g dk_rmse=%g "
            "dk_rel_l2=%g dv_max_abs=%g dv_mean_abs=%g dv_rmse=%g "
            "dv_rel_l2=%g bad=%d pass=%d\n",
            "fa3_bwd_dkv_correctness",
            shaobo_fa3_status_string(status), batch, heads, seqlen, dim,
            params.causal, workspace_bytes, dk_metrics.max_abs, dk_metrics.mean_abs,
            dk_metrics.rmse, dk_metrics.rel_l2, dv_metrics.max_abs,
            dv_metrics.mean_abs, dv_metrics.rmse, dv_metrics.rel_l2,
            dk_metrics.bad_count + dv_metrics.bad_count, pass ? 1 : 0);
        return pass ? 0 : 1;
    }
}
#endif
