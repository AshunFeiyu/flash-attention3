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
    static constexpr int kRawPages = Tile::kRawBuffers;
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
        Tile::kOverlayRawOnResidentKv
            ? (kRawEnd > kKvEnd ? kRawEnd : kKvEnd)
            : kKvEnd;
    static constexpr int kBytes = kSidecarBase + Tile::kSidecarBytes;
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
    static_assert(Tile::kRawBuffers == 1 || Tile::kRawBuffers == 2,
                  "canonical dKV route supports one or two raw Q/dO pages");
    if constexpr (Tile::kRawBuffers == 1) {
        (void)q_tile;
        return 0;
    } else {
        return q_tile & 1;
    }
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
    int& raw0_phase,
    int& raw1_phase) {
    if (page == 0) {
        ins::abarrier_try_wait<true>(Wdra::kRaw0Used, raw0_phase);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kRaw1Used, raw1_phase);
    }
}

template <typename Wdra>
__device__ __forceinline__ void wait_raw_filled_page(
    int page,
    int& raw0_phase,
    int& raw1_phase) {
    if (page == 0) {
        ins::abarrier_try_wait<true>(Wdra::kRaw0Filled, raw0_phase);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kRaw1Filled, raw1_phase);
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

template <typename Wdra>
__device__ __forceinline__ void seq_q_filled() {
    ins::abarrier_seq<false>(Wdra::kQFilled);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_q_filled() {
    ins::abarrier_arrive_cnt<false>(Wdra::kQFilled, 1);
}

template <typename Wdra>
__device__ __forceinline__ void wait_q_used(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kQUsed, phase);
}

template <typename Wdra>
__device__ __forceinline__ void wait_q_filled(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kQFilled, phase);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_q_used() {
    ins::abarrier_arrive_cnt<false>(Wdra::kQUsed, 1);
}

template <typename Wdra>
__device__ __forceinline__ void seq_dout_filled() {
    ins::abarrier_seq<false>(Wdra::kDoutFilled);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_dout_filled() {
    ins::abarrier_arrive_cnt<false>(Wdra::kDoutFilled, 1);
}

template <typename Wdra>
__device__ __forceinline__ void wait_dout_used(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kDoutUsed, phase);
}

template <typename Wdra>
__device__ __forceinline__ void wait_dout_filled(int& phase) {
    ins::abarrier_try_wait<true>(Wdra::kDoutFilled, phase);
}

template <typename Wdra>
__device__ __forceinline__ void arrive_dout_used() {
    ins::abarrier_arrive_cnt<false>(Wdra::kDoutUsed, 1);
}

template <typename Wdra, int Half>
__device__ __forceinline__ void seq_q_half_filled() {
    static_assert(Half == 0 || Half == 1, "Q half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_seq<false>(Wdra::kQ0Filled);
    } else {
        ins::abarrier_seq<false>(Wdra::kQ1Filled);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void arrive_q_half_filled() {
    static_assert(Half == 0 || Half == 1, "Q half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_arrive_cnt<false>(Wdra::kQ0Filled, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Wdra::kQ1Filled, 1);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void wait_q_half_filled(int& phase) {
    static_assert(Half == 0 || Half == 1, "Q half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_try_wait<true>(Wdra::kQ0Filled, phase);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kQ1Filled, phase);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void wait_q_half_used(int& phase) {
    static_assert(Half == 0 || Half == 1, "Q half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_try_wait<true>(Wdra::kQ0Used, phase);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kQ1Used, phase);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void arrive_q_half_used() {
    static_assert(Half == 0 || Half == 1, "Q half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_arrive_cnt<false>(Wdra::kQ0Used, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Wdra::kQ1Used, 1);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void seq_dout_half_filled() {
    static_assert(Half == 0 || Half == 1, "dO half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_seq<false>(Wdra::kDout0Filled);
    } else {
        ins::abarrier_seq<false>(Wdra::kDout1Filled);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void arrive_dout_half_filled() {
    static_assert(Half == 0 || Half == 1, "dO half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_arrive_cnt<false>(Wdra::kDout0Filled, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Wdra::kDout1Filled, 1);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void wait_dout_half_filled(int& phase) {
    static_assert(Half == 0 || Half == 1, "dO half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_try_wait<true>(Wdra::kDout0Filled, phase);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kDout1Filled, phase);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void wait_dout_half_used(int& phase) {
    static_assert(Half == 0 || Half == 1, "dO half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_try_wait<true>(Wdra::kDout0Used, phase);
    } else {
        ins::abarrier_try_wait<true>(Wdra::kDout1Used, phase);
    }
}

template <typename Wdra, int Half>
__device__ __forceinline__ void arrive_dout_half_used() {
    static_assert(Half == 0 || Half == 1, "dO half must be 0 or 1");
    if constexpr (Half == 0) {
        ins::abarrier_arrive_cnt<false>(Wdra::kDout0Used, 1);
    } else {
        ins::abarrier_arrive_cnt<false>(Wdra::kDout1Used, 1);
    }
}

template <typename Tile>
__device__ __forceinline__ void publish_mq_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    int q_base,
    int page,
    int wave_local) {
#pragma unroll
    for (int m_block = 0; m_block < DkvLdsLayout<Tile>::kRawMBlocksPerMqTile;
         ++m_block) {
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

template <typename Tile>
__device__ __forceinline__ void publish_sidecar_tile_to_lds(
    __half* lds,
    const float* packed_sidecar,
    int64_t row_base,
    int q_base,
    int seqlen,
    int page,
    int wave_local,
    int lane) {
    const int local_row = wave_local * Tile::kWaveSize + lane;
    if (local_row >= Tile::kBlockMq) {
        return;
    }
    float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
    const int q_row = q_base + local_row;
    float row_max_log2 = 0.0f;
    float row_inv_sum = 0.0f;
    float row_delta = 0.0f;
    if (q_row >= seqlen) {
        sidecar_page[Tile::kSidecarMaxLog2Base + local_row] = row_max_log2;
        sidecar_page[Tile::kSidecarInvSumBase + local_row] = row_inv_sum;
        sidecar_page[Tile::kSidecarDeltaBase + local_row] = row_delta;
        return;
    }
    if (packed_sidecar != nullptr) {
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
__device__ __forceinline__ void producer_kq_loop(
    const __half* q_base_ptr,
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
    int q_used_phase = 0;
    int q0_used_phase = 0;
    int q1_used_phase = 0;
    int resident_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
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
        if constexpr (Tile::kBlockMq == 128) {
            if (q_tile >= Tile::kRawBuffers) {
                wait_q_half_used<Wdra, 0>(q0_used_phase);
            }
            seq_q_half_filled<Wdra, 0>();
            publish_mq_half_tile<Tile, 0>(
                lds, Layout::kQBase, q_base_ptr, row_stride, packet_q_base,
                page, wave_local);
            publish_sidecar_half_tile_to_lds<Tile, 0>(
                lds, packed_sidecar, row_base, packet_q_base, seqlen, page,
                wave_local, lane);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            arrive_q_half_filled<Wdra, 0>();

            if (q_tile >= Tile::kRawBuffers) {
                wait_q_half_used<Wdra, 1>(q1_used_phase);
            }
            seq_q_half_filled<Wdra, 1>();
            publish_mq_half_tile<Tile, 1>(
                lds, Layout::kQBase, q_base_ptr, row_stride, packet_q_base,
                page, wave_local);
            publish_sidecar_half_tile_to_lds<Tile, 1>(
                lds, packed_sidecar, row_base, packet_q_base, seqlen, page,
                wave_local, lane);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            arrive_q_half_filled<Wdra, 1>();
        } else {
            if (q_tile >= Tile::kRawBuffers) {
                wait_q_used<Wdra>(q_used_phase);
            }
            seq_q_filled<Wdra>();
            publish_mq_tile<Tile>(
                lds, Layout::kQBase, q_base_ptr, row_stride, packet_q_base,
                page, wave_local);
            publish_sidecar_tile_to_lds<Tile>(
                lds, packed_sidecar, row_base, packet_q_base, seqlen, page,
                wave_local, lane);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            arrive_q_filled<Wdra>();
        }
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_vdout_loop(
    const __half* v_base_ptr,
    const __half* dout_base_ptr,
    __half* lds,
    int q_base,
    int k_base,
    int seqlen,
    int row_stride,
    int q_tiles,
    int wave_local) {
    using Layout = DkvLdsLayout<Tile>;
    int dout_used_phase = 0;
    int dout0_used_phase = 0;
    int dout1_used_phase = 0;
    int resident_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);
    if constexpr (Tile::kOverlayRawOnResidentKv) {
        wait_resident_used<Wdra>(resident_used_phase);
    }

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = raw_page_for_q_tile<Tile>(q_tile);
        const int packet_q_base = q_base + q_tile * Tile::kBlockMq;
        if constexpr (Tile::kBlockMq == 128) {
            if (q_tile >= Tile::kRawBuffers) {
                wait_dout_half_used<Wdra, 0>(dout0_used_phase);
            }
            seq_q_half_filled<Wdra, 0>();
            publish_mq_half_tile<Tile, 0>(
                lds, Layout::kDoutBase, dout_base_ptr, row_stride,
                packet_q_base, page, wave_local);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            arrive_q_half_filled<Wdra, 0>();

            if (q_tile >= Tile::kRawBuffers) {
                wait_dout_half_used<Wdra, 1>(dout1_used_phase);
            }
            seq_q_half_filled<Wdra, 1>();
            publish_mq_half_tile<Tile, 1>(
                lds, Layout::kDoutBase, dout_base_ptr, row_stride,
                packet_q_base, page, wave_local);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            arrive_q_half_filled<Wdra, 1>();
        } else {
            if (q_tile >= Tile::kRawBuffers) {
                wait_dout_used<Wdra>(dout_used_phase);
            }
            seq_dout_filled<Wdra>();
            publish_mq_tile<Tile>(
                lds, Layout::kDoutBase, dout_base_ptr, row_stride,
                packet_q_base, page, wave_local);

            ins::maybe_wait_bps_vbcnt_before_arrive();
            arrive_dout_filled<Wdra>();
        }
    }
}

struct Owner16KvRegs {
    ins::F16x8 k[4];
    ins::F16x8 v[4];
};

template <typename Tile>
__device__ __forceinline__ void latch_owner16_kv_regs(
    __half* lds,
    int owner_nblock,
    Owner16KvRegs& regs) {
    using Layout = DkvLdsLayout<Tile>;
    const int row_block32 = owner_nblock >> 1;
    const int n_half = owner_nblock & 1;

    ins::F16x8 k0[4];
    ins::F16x8 k1[4];
    ins::F16x8 v0[4];
    ins::F16x8 v1[4];

#pragma unroll
    for (int d_block = 0; d_block < 4; ++d_block) {
        const int k_off =
            Layout::kKBase + kv_block_offset<Tile>(row_block32, d_block);
        const int v_off =
            Layout::kVBase + kv_block_offset<Tile>(row_block32, d_block);
        ins::ds_read_matrix_trans_pair(
            lds, k_off, k0[d_block].f16x8, k1[d_block].f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, v_off, v0[d_block].f16x8, v1[d_block].f16x8);
    }
    ins::wait_lgkm(0);

#pragma unroll
    for (int d_block = 0; d_block < 4; ++d_block) {
        if (n_half == 0) {
            regs.k[d_block].f16x8 = k0[d_block].f16x8;
            regs.v[d_block].f16x8 = v0[d_block].f16x8;
        } else {
            regs.k[d_block].f16x8 = k1[d_block].f16x8;
            regs.v[d_block].f16x8 = v1[d_block].f16x8;
        }
    }
}

template <typename Tile, int MBlockBase, int DBlock>
__device__ __forceinline__ void read_score_dp_owner16_dblock(
    __half* lds,
    int page,
    ins::F16x8& q0,
    ins::F16x8& q1,
    ins::F16x8& dout0,
    ins::F16x8& dout1) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(MBlockBase + 1 < Layout::kRawMBlocksPerMqTile,
                  "score/dP reads two adjacent M16 blocks");
    static_assert(DBlock >= 0 && DBlock < 4,
                  "score/dP DBlock must be in D128");

    const int q_off0 =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlock);
    const int q_off1 =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlock);
    const int dout_off0 =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlock);
    const int dout_off1 =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlock);

    ins::ds_read_matrix_32x16_trans(lds, q_off0, q0.f16x8);
    ins::ds_read_matrix_32x16_trans(lds, q_off1, q1.f16x8);
    ins::ds_read_matrix_32x16_trans(lds, dout_off0, dout0.f16x8);
    ins::ds_read_matrix_32x16_trans(lds, dout_off1, dout1.f16x8);
}

template <int DBlock>
__device__ __forceinline__ void score_dp_mmac_owner16_dblock(
    const Owner16KvRegs& kv_regs,
    const ins::F16x8& q0,
    const ins::F16x8& q1,
    const ins::F16x8& dout0,
    const ins::F16x8& dout1,
    const ins::F16x8& zero,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    static_assert(DBlock >= 0 && DBlock < 4,
                  "score/dP DBlock must be in D128");
    const ins::F16x8& k_reg = kv_regs.k[DBlock];
    const ins::F16x8& v_reg = kv_regs.v[DBlock];
#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const ins::Vec4F16 q_frag[2] = {
            m_idx == 0 ? q0.f16x4[0] : q1.f16x4[0],
            m_idx == 0 ? q0.f16x4[1] : q1.f16x4[1],
        };
        const ins::Vec4F16 dout_frag[2] = {
            m_idx == 0 ? dout0.f16x4[0] : dout1.f16x4[0],
            m_idx == 0 ? dout0.f16x4[1] : dout1.f16x4[1],
        };
#pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            const bool first = DBlock == 0 && k_half == 0;
            score[m_idx].f32 = ins::mmac_f16_lit(
                k_reg.f16x4[k_half], q_frag[k_half],
                first ? zero.f32 : score[m_idx].f32);
            dp[m_idx].f32 = ins::mmac_f16_lit(
                v_reg.f16x4[k_half], dout_frag[k_half],
                first ? zero.f32 : dp[m_idx].f32);
        }
    }
}

template <typename Tile, int MBlockBase>
__device__ __forceinline__ void score_dp_mmac_owner16(
    __half* lds,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(MBlockBase + 1 < Layout::kRawMBlocksPerMqTile,
                  "score/dP reads two adjacent M16 blocks");

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    ins::raise_priority_2();
    ins::F16x8 q0_d0;
    ins::F16x8 q1_d0;
    ins::F16x8 dout0_d0;
    ins::F16x8 dout1_d0;
    ins::F16x8 q0_d1;
    ins::F16x8 q1_d1;
    ins::F16x8 dout0_d1;
    ins::F16x8 dout1_d1;

    read_score_dp_owner16_dblock<Tile, MBlockBase, 0>(
        lds, page, q0_d0, q1_d0, dout0_d0, dout1_d0);
    read_score_dp_owner16_dblock<Tile, MBlockBase, 1>(
        lds, page, q0_d1, q1_d1, dout0_d1, dout1_d1);
    ins::wait_lgkm(0);
    score_dp_mmac_owner16_dblock<0>(
        kv_regs, q0_d0, q1_d0, dout0_d0, dout1_d0, zero, score, dp);
    score_dp_mmac_owner16_dblock<1>(
        kv_regs, q0_d1, q1_d1, dout0_d1, dout1_d1, zero, score, dp);

    read_score_dp_owner16_dblock<Tile, MBlockBase, 2>(
        lds, page, q0_d0, q1_d0, dout0_d0, dout1_d0);
    read_score_dp_owner16_dblock<Tile, MBlockBase, 3>(
        lds, page, q0_d1, q1_d1, dout0_d1, dout1_d1);
    ins::wait_lgkm(0);
    score_dp_mmac_owner16_dblock<2>(
        kv_regs, q0_d0, q1_d0, dout0_d0, dout1_d0, zero, score, dp);
    score_dp_mmac_owner16_dblock<3>(
        kv_regs, q0_d1, q1_d1, dout0_d1, dout1_d1, zero, score, dp);
    ins::lower_priority();
}

template <typename Tile>
__device__ __forceinline__ void score_dp_mmac_owner16_dyn(
    __half* lds,
    int page,
    int m_block_base,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    using Layout = DkvLdsLayout<Tile>;

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    ins::raise_priority_2();
#pragma unroll
    for (int d_block = 0; d_block < 4; ++d_block) {
        ins::F16x8 q0;
        ins::F16x8 q1;
        ins::F16x8 dout0;
        ins::F16x8 dout1;
        const int q_off0 =
            Layout::kQBase +
            raw_page_block_offset_m<Tile>(page, m_block_base + 0, d_block);
        const int q_off1 =
            Layout::kQBase +
            raw_page_block_offset_m<Tile>(page, m_block_base + 1, d_block);
        const int dout_off0 =
            Layout::kDoutBase +
            raw_page_block_offset_m<Tile>(page, m_block_base + 0, d_block);
        const int dout_off1 =
            Layout::kDoutBase +
            raw_page_block_offset_m<Tile>(page, m_block_base + 1, d_block);

        ins::ds_read_matrix_32x16_trans(lds, q_off0, q0.f16x8);
        ins::ds_read_matrix_32x16_trans(lds, q_off1, q1.f16x8);
        ins::ds_read_matrix_32x16_trans(lds, dout_off0, dout0.f16x8);
        ins::ds_read_matrix_32x16_trans(lds, dout_off1, dout1.f16x8);
        ins::wait_lgkm(0);

        const ins::F16x8& k_reg = kv_regs.k[d_block];
        const ins::F16x8& v_reg = kv_regs.v[d_block];
#pragma unroll
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            const ins::Vec4F16 q_frag[2] = {
                m_idx == 0 ? q0.f16x4[0] : q1.f16x4[0],
                m_idx == 0 ? q0.f16x4[1] : q1.f16x4[1],
            };
            const ins::Vec4F16 dout_frag[2] = {
                m_idx == 0 ? dout0.f16x4[0] : dout1.f16x4[0],
                m_idx == 0 ? dout0.f16x4[1] : dout1.f16x4[1],
            };
#pragma unroll
            for (int k_half = 0; k_half < 2; ++k_half) {
                const bool first = d_block == 0 && k_half == 0;
                score[m_idx].f32 = ins::mmac_f16_lit(
                    k_reg.f16x4[k_half], q_frag[k_half],
                    first ? zero.f32 : score[m_idx].f32);
                dp[m_idx].f32 = ins::mmac_f16_lit(
                    v_reg.f16x4[k_half], dout_frag[k_half],
                    first ? zero.f32 : dp[m_idx].f32);
            }
        }
    }
    ins::lower_priority();
}

template <typename Tile, int MBlockBase>
__device__ __forceinline__ void softmax_ds_owner16_from_lds_sidecar(
    const ins::F32x4 (&score)[2],
    const ins::F32x4 (&dp)[2],
    const float* sidecar_page,
    int q_tile,
    int k_base,
    int owner_nblock,
    int lane,
    int seqlen,
    int causal,
    float softmax_scale,
    ins::Vec4F16 (&p_frag)[2],
    ins::Vec4F16 (&ds_frag)[2]) {
    static_assert(MBlockBase + 1 < DkvLdsLayout<Tile>::kRawMBlocksPerMqTile,
                  "softmax/dS consumes two adjacent M16 blocks");
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int local_q_base = MBlockBase * 16;
    const int q_base = q_tile * Tile::kBlockMq + local_q_base;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    const bool full_valid_tile =
        q_base + 32 <= seqlen &&
        k_base + owner_nblock * 16 + 15 < seqlen &&
        (!causal || (k_base + owner_nblock * 16 + 15 <= q_base));
    const float* sidecar_pair = sidecar_page + local_q_base;

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        const ins::Vec4F32 row_max_log2_vec =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarMaxLog2Base + local_m_base);
        const ins::Vec4F32 row_inv_sum_vec =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarInvSumBase + local_m_base);
        const ins::Vec4F32 row_delta_vec =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarDeltaBase + local_m_base);
        ins::Vec4F16 p_vec;
        ins::Vec4F16 ds_vec;
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_base + local_m;
            const bool valid_pair =
                full_valid_tile ||
                (krow < seqlen && qrow < seqlen &&
                 (!causal || krow <= qrow));
            float p_val;
            float ds_val;
            if (valid_pair) {
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale * kLog2E -
                          row_max_log2_vec[vec_id]) *
                    row_inv_sum_vec[vec_id];
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta_vec[vec_id]) *
                    softmax_scale;
            } else {
                p_val = 0.0f;
                ds_val = 0.0f;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

template <typename Tile>
__device__ __forceinline__ void softmax_ds_owner16_from_lds_sidecar_dyn(
    const ins::F32x4 (&score)[2],
    const ins::F32x4 (&dp)[2],
    const float* sidecar_page,
    int q_tile,
    int m_block_base,
    int k_base,
    int owner_nblock,
    int lane,
    int seqlen,
    int causal,
    float softmax_scale,
    ins::Vec4F16 (&p_frag)[2],
    ins::Vec4F16 (&ds_frag)[2]) {
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int local_q_base = m_block_base * 16;
    const int q_base = q_tile * Tile::kBlockMq + local_q_base;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    const bool full_valid_tile =
        q_base + 32 <= seqlen &&
        k_base + owner_nblock * 16 + 15 < seqlen &&
        (!causal || (k_base + owner_nblock * 16 + 15 <= q_base));
    const float* sidecar_pair = sidecar_page + local_q_base;

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        const ins::Vec4F32 row_max_log2_vec =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarMaxLog2Base + local_m_base);
        const ins::Vec4F32 row_inv_sum_vec =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarInvSumBase + local_m_base);
        const ins::Vec4F32 row_delta_vec =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarDeltaBase + local_m_base);
        ins::Vec4F16 p_vec;
        ins::Vec4F16 ds_vec;
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_base + local_m;
            const bool valid_pair =
                full_valid_tile ||
                (krow < seqlen && qrow < seqlen &&
                 (!causal || krow <= qrow));
            float p_val;
            float ds_val;
            if (valid_pair) {
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale * kLog2E -
                          row_max_log2_vec[vec_id]) *
                    row_inv_sum_vec[vec_id];
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta_vec[vec_id]) *
                    softmax_scale;
            } else {
                p_val = 0.0f;
                ds_val = 0.0f;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

template <typename Tile, int MBlockBase>
__device__ __forceinline__ void softmax_ds_owner16_causal_exact_tile_ctx(
    const ins::F32x4 (&score)[2],
    const ins::F32x4 (&dp)[2],
    const float* sidecar_page,
    int q_pair_base,
    int owner_krow,
    int lane_col_group,
    float softmax_scale,
    float softmax_scale_log2,
    ins::Vec4F16 (&p_frag)[2],
    ins::Vec4F16 (&ds_frag)[2]) {
    static_assert(MBlockBase + 1 < DkvLdsLayout<Tile>::kRawMBlocksPerMqTile,
                  "softmax/dS consumes two adjacent M16 blocks");
    const float* sidecar_pair = sidecar_page + MBlockBase * 16;

    ins::Vec4F32 row_max_log2_vecs[2];
    ins::Vec4F32 row_inv_sum_vecs[2];
    ins::Vec4F32 row_delta_vecs[2];

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        row_max_log2_vecs[m_idx] =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarMaxLog2Base + local_m_base);
        row_inv_sum_vecs[m_idx] =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarInvSumBase + local_m_base);
        row_delta_vecs[m_idx] =
            *reinterpret_cast<const ins::Vec4F32*>(
                sidecar_pair + Tile::kSidecarDeltaBase + local_m_base);
    }

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        const ins::Vec4F32 row_max_log2_vec = row_max_log2_vecs[m_idx];
        const ins::Vec4F32 row_inv_sum_vec = row_inv_sum_vecs[m_idx];
        const ins::Vec4F32 row_delta_vec = row_delta_vecs[m_idx];
        ins::Vec4F16 p_vec;
        ins::Vec4F16 ds_vec;
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_pair_base + local_m;
            const bool valid_pair = owner_krow <= qrow;
            float p_val;
            float ds_val;
            if (valid_pair) {
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale_log2 -
                          row_max_log2_vec[vec_id]) *
                    row_inv_sum_vec[vec_id];
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta_vec[vec_id]) *
                    softmax_scale;
            } else {
                p_val = 0.0f;
                ds_val = 0.0f;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

struct DvDkSourceRegs4 {
    ins::F16x8 dout_m0_d0;
    ins::F16x8 dout_m1_d0;
    ins::F16x8 q_m0_d0;
    ins::F16x8 q_m1_d0;
    ins::F16x8 dout_m0_d1;
    ins::F16x8 dout_m1_d1;
    ins::F16x8 q_m0_d1;
    ins::F16x8 q_m1_d1;
};

template <typename Tile, int DBlockBase, int MBlockBase>
__device__ __forceinline__ void dv_dk_read_owner16_sources4(
    __half* lds,
    int page,
    DvDkSourceRegs4& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK source group must cover two D blocks");
    static_assert(MBlockBase + 1 < Layout::kRawMBlocksPerMqTile,
                  "dV/dK reads two adjacent M16 blocks");

    const int dout_m0_d0_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase);
    const int dout_m1_d0_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase);
    const int q_m0_d0_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase);
    const int q_m1_d0_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase);

    const int dout_m0_d1_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase + 1);
    const int dout_m1_d1_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase + 1);
    const int q_m0_d1_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase + 1);
    const int q_m1_d1_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase + 1);

    ins::ds_read_matrix_32x16_normal(
        lds, dout_m0_d0_off, src.dout_m0_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m1_d0_off, src.dout_m1_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m0_d0_off, src.q_m0_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m1_d0_off, src.q_m1_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m0_d1_off, src.dout_m0_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m1_d1_off, src.dout_m1_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m0_d1_off, src.q_m0_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m1_d1_off, src.q_m1_d1.f16x8);
}

template <typename Tile, int DBlockBase, int MBlockBase>
__device__ __forceinline__ void dv_dk_read_owner16_dout_sources4(
    __half* lds,
    int page,
    DvDkSourceRegs4& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK dO source group must cover two D blocks");
    static_assert(MBlockBase + 1 < Layout::kRawMBlocksPerMqTile,
                  "dV/dK dO reads two adjacent M16 blocks");

    const int dout_m0_d0_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase);
    const int dout_m1_d0_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase);
    const int dout_m0_d1_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase + 1);
    const int dout_m1_d1_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase + 1);

    ins::ds_read_matrix_32x16_normal(
        lds, dout_m0_d0_off, src.dout_m0_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m1_d0_off, src.dout_m1_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m0_d1_off, src.dout_m0_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m1_d1_off, src.dout_m1_d1.f16x8);
}

template <typename Tile, int DBlockBase, int MBlockBase>
__device__ __forceinline__ void dv_dk_read_owner16_q_sources4(
    __half* lds,
    int page,
    DvDkSourceRegs4& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK Q source group must cover two D blocks");
    static_assert(MBlockBase + 1 < Layout::kRawMBlocksPerMqTile,
                  "dV/dK Q reads two adjacent M16 blocks");

    const int q_m0_d0_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase);
    const int q_m1_d0_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase);
    const int q_m0_d1_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 0, DBlockBase + 1);
    const int q_m1_d1_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, MBlockBase + 1, DBlockBase + 1);

    ins::ds_read_matrix_32x16_normal(lds, q_m0_d0_off, src.q_m0_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m1_d0_off, src.q_m1_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m0_d1_off, src.q_m0_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m1_d1_off, src.q_m1_d1.f16x8);
}

template <typename Tile>
__device__ __forceinline__ void dv_dk_read_owner16_sources4_dyn(
    __half* lds,
    int page,
    int d_block_base,
    int m_block_base,
    DvDkSourceRegs4& src) {
    using Layout = DkvLdsLayout<Tile>;
    const int dout_m0_d0_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 0, d_block_base);
    const int dout_m1_d0_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 1, d_block_base);
    const int q_m0_d0_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 0, d_block_base);
    const int q_m1_d0_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 1, d_block_base);

    const int dout_m0_d1_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 0,
                                      d_block_base + 1);
    const int dout_m1_d1_off =
        Layout::kDoutBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 1,
                                      d_block_base + 1);
    const int q_m0_d1_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 0,
                                      d_block_base + 1);
    const int q_m1_d1_off =
        Layout::kQBase +
        raw_page_block_offset_m<Tile>(page, m_block_base + 1,
                                      d_block_base + 1);

    ins::ds_read_matrix_32x16_normal(
        lds, dout_m0_d0_off, src.dout_m0_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m1_d0_off, src.dout_m1_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m0_d0_off, src.q_m0_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m1_d0_off, src.q_m1_d0.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m0_d1_off, src.dout_m0_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, dout_m1_d1_off, src.dout_m1_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m0_d1_off, src.q_m0_d1.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, q_m1_d1_off, src.q_m1_d1.f16x8);
}

template <bool FirstQTile, int OutIdx, int DHalf>
__device__ __forceinline__ void dv_dk_mmac_one_out_raw32x16(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const ins::F16x8& dout_m0,
    const ins::F16x8& dout_m1,
    const ins::F16x8& q_m0,
    const ins::F16x8& q_m1,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
    static_assert(DHalf == 0 || DHalf == 1,
                  "32x16 raw page has two D16 halves");
#pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
        const ins::Vec4F16 dout_frag =
            m_half == 0 ? dout_m0.f16x4[DHalf] : dout_m1.f16x4[DHalf];
        const ins::Vec4F16 q_frag =
            m_half == 0 ? q_m0.f16x4[DHalf] : q_m1.f16x4[DHalf];
        if constexpr (FirstQTile) {
            dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
                p_frag[m_half], dout_frag,
                m_half == 0 ? zero_f16.f32 : dv_acc[OutIdx].f32);
            dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
                ds_frag[m_half], q_frag,
                m_half == 0 ? zero_f16.f32 : dk_acc[OutIdx].f32);
        } else {
            dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
                p_frag[m_half], dout_frag, dv_acc[OutIdx].f32);
            dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
                ds_frag[m_half], q_frag, dk_acc[OutIdx].f32);
        }
    }
}

template <bool FirstQTile, int OutBase>
__device__ __forceinline__ void dv_dk_mmac_four_out(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
    dv_dk_mmac_one_out_raw32x16<FirstQTile, OutBase + 0, 0>(
        p_frag, ds_frag, src.dout_m0_d0, src.dout_m1_d0, src.q_m0_d0,
        src.q_m1_d0, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out_raw32x16<FirstQTile, OutBase + 1, 1>(
        p_frag, ds_frag, src.dout_m0_d0, src.dout_m1_d0, src.q_m0_d0,
        src.q_m1_d0, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out_raw32x16<FirstQTile, OutBase + 2, 0>(
        p_frag, ds_frag, src.dout_m0_d1, src.dout_m1_d1, src.q_m0_d1,
        src.q_m1_d1, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out_raw32x16<FirstQTile, OutBase + 3, 1>(
        p_frag, ds_frag, src.dout_m0_d1, src.dout_m1_d1, src.q_m0_d1,
        src.q_m1_d1, dv_acc, dk_acc, zero_f16);
}

template <typename Tile, int MBlockBase, bool FirstQTile>
__device__ __forceinline__ void dv_dk_mmac_owner16_read4x2(
    __half* lds,
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& low_src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    int page) {
    ins::F16x8 zero_f16;
    if constexpr (FirstQTile) {
        ins::zero_f16x8(zero_f16);
    }

    DvDkSourceRegs4 high_src;
    dv_dk_read_owner16_sources4<Tile, 2, MBlockBase>(lds, page, high_src);

    ins::raise_priority_2();
    // Low sources were issued before softmax; keep the newly issued high
    // sources in flight while the low dV/dK MMAC island runs.
    ins::wait_lgkm(8);
    dv_dk_mmac_four_out<FirstQTile, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    ins::wait_lgkm(0);
    dv_dk_mmac_four_out<FirstQTile, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <typename Tile, typename Wdra, int MBlockBase, bool FirstQTile>
__device__ __forceinline__ void dv_dk_mmac_owner16_read4x2_early_release(
    __half* lds,
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& low_src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    int page) {
    ins::F16x8 zero_f16;
    if constexpr (FirstQTile) {
        ins::zero_f16x8(zero_f16);
    }

    ins::wait_lgkm(0);
    DvDkSourceRegs4 high_src;
    dv_dk_read_owner16_sources4<Tile, 2, MBlockBase>(lds, page, high_src);
    arrive_raw_used_page<Wdra>(page);

    ins::raise_priority_2();
    dv_dk_mmac_four_out<FirstQTile, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    ins::wait_lgkm(0);
    dv_dk_mmac_four_out<FirstQTile, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <bool FirstQTile>
__device__ __forceinline__ void dv_dk_mmac_owner16_qready(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& low_src,
    const DvDkSourceRegs4& high_src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    ins::F16x8 zero_f16;
    if constexpr (FirstQTile) {
        ins::zero_f16x8(zero_f16);
    }

    ins::raise_priority_2();
    dv_dk_mmac_four_out<FirstQTile, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_four_out<FirstQTile, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <int OutIdx, int DHalf>
__device__ __forceinline__ void dv_dk_mmac_one_out_raw32x16_dyn(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const ins::F16x8& dout_m0,
    const ins::F16x8& dout_m1,
    const ins::F16x8& q_m0,
    const ins::F16x8& q_m1,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16,
    bool first_accum) {
#pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
        const ins::Vec4F16 dout_frag =
            m_half == 0 ? dout_m0.f16x4[DHalf] : dout_m1.f16x4[DHalf];
        const ins::Vec4F16 q_frag =
            m_half == 0 ? q_m0.f16x4[DHalf] : q_m1.f16x4[DHalf];
        dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
            p_frag[m_half], dout_frag,
            first_accum && m_half == 0 ? zero_f16.f32
                                       : dv_acc[OutIdx].f32);
        dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
            ds_frag[m_half], q_frag,
            first_accum && m_half == 0 ? zero_f16.f32
                                       : dk_acc[OutIdx].f32);
    }
}

template <int OutBase>
__device__ __forceinline__ void dv_dk_mmac_four_out_dyn(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16,
    bool first_accum) {
    dv_dk_mmac_one_out_raw32x16_dyn<OutBase + 0, 0>(
        p_frag, ds_frag, src.dout_m0_d0, src.dout_m1_d0, src.q_m0_d0,
        src.q_m1_d0, dv_acc, dk_acc, zero_f16, first_accum);
    dv_dk_mmac_one_out_raw32x16_dyn<OutBase + 1, 1>(
        p_frag, ds_frag, src.dout_m0_d0, src.dout_m1_d0, src.q_m0_d0,
        src.q_m1_d0, dv_acc, dk_acc, zero_f16, first_accum);
    dv_dk_mmac_one_out_raw32x16_dyn<OutBase + 2, 0>(
        p_frag, ds_frag, src.dout_m0_d1, src.dout_m1_d1, src.q_m0_d1,
        src.q_m1_d1, dv_acc, dk_acc, zero_f16, first_accum);
    dv_dk_mmac_one_out_raw32x16_dyn<OutBase + 3, 1>(
        p_frag, ds_frag, src.dout_m0_d1, src.dout_m1_d1, src.q_m0_d1,
        src.q_m1_d1, dv_acc, dk_acc, zero_f16, first_accum);
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void dv_dk_mmac_owner16_read4x2_dyn(
    __half* lds,
    int m_block_base,
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& low_src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    int page,
    bool first_accum,
    bool release_page) {
    ins::F16x8 zero_f16;
    ins::zero_f16x8(zero_f16);

    ins::wait_lgkm(0);
    DvDkSourceRegs4 high_src;
    dv_dk_read_owner16_sources4_dyn<Tile>(
        lds, page, 2, m_block_base, high_src);
    if (release_page) {
        arrive_raw_used_page<Wdra>(page);
    }

    ins::raise_priority_2();
    dv_dk_mmac_four_out_dyn<0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16, first_accum);
    ins::wait_lgkm(0);
    dv_dk_mmac_four_out_dyn<4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16, first_accum);
    ins::lower_priority();
}

template <typename Tile>
__device__ __forceinline__ void store_dkv_owner16(
    float* dk,
    float* dv,
    const ins::F32x4 (&dk_acc)[8],
    const ins::F32x4 (&dv_acc)[8],
    int lane,
    int k_base,
    int owner_nblock,
    int seqlen,
    int64_t tensor_base) {
    if (dk == nullptr || dv == nullptr) {
        return;
    }
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    if (krow >= seqlen) {
        return;
    }
    const int64_t out_row =
        tensor_base + static_cast<int64_t>(krow) * Tile::kHeadDim;
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
          bool ReleasePage>
__device__ __forceinline__ void consume_mq_mpair_owner16(
    __half* lds,
    int seqlen,
    int k_base,
    int q_tile,
    int causal,
    float softmax_scale,
    int owner_nblock,
    int lane,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    ins::F32x4 score[2];
    ins::F32x4 dp[2];
    score_dp_mmac_owner16<Tile, MBlockBase>(
        lds, page, kv_regs, score, dp);

    DvDkSourceRegs4 dvdk_low;
    dv_dk_read_owner16_sources4<Tile, 0, MBlockBase>(
        lds, page, dvdk_low);

    ins::Vec4F16 p_frag[2];
    ins::Vec4F16 ds_frag[2];
    const float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
    softmax_ds_owner16_from_lds_sidecar<Tile, MBlockBase>(
        score, dp, sidecar_page, q_tile, k_base, owner_nblock, lane, seqlen,
        causal, softmax_scale, p_frag, ds_frag);

    if constexpr (ReleasePage) {
        dv_dk_mmac_owner16_read4x2_early_release<Tile,
                                                  Wdra,
                                                  MBlockBase,
                                                  FirstAccum>(
            lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
    } else {
        dv_dk_mmac_owner16_read4x2<Tile, MBlockBase, FirstAccum>(
            lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
    }
}

template <typename Tile,
          typename Wdra,
          int MBlockBase,
          bool FirstAccum,
          bool ReleasePage,
          int ReleaseHalf = 0>
__device__ __forceinline__ void consume_mq_mpair_owner16_causal_exact_tile(
    __half* lds,
    int q_pair_base,
    int owner_krow,
    int lane_col_group,
    float softmax_scale,
    float softmax_scale_log2,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    ins::F32x4 score[2];
    ins::F32x4 dp[2];
    score_dp_mmac_owner16<Tile, MBlockBase>(
        lds, page, kv_regs, score, dp);

    if constexpr (ReleasePage) {
        DvDkSourceRegs4 dvdk_low;
        DvDkSourceRegs4 dvdk_high;
        dv_dk_read_owner16_dout_sources4<Tile, 0, MBlockBase>(
            lds, page, dvdk_low);
        dv_dk_read_owner16_dout_sources4<Tile, 2, MBlockBase>(
            lds, page, dvdk_high);
        ins::wait_lgkm(0);
        arrive_dout_half_used<Wdra, ReleaseHalf>();

        dv_dk_read_owner16_q_sources4<Tile, 0, MBlockBase>(
            lds, page, dvdk_low);
        dv_dk_read_owner16_q_sources4<Tile, 2, MBlockBase>(
            lds, page, dvdk_high);
        ins::wait_lgkm(0);
        arrive_q_half_used<Wdra, ReleaseHalf>();

        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        const float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
        softmax_ds_owner16_causal_exact_tile_ctx<Tile, MBlockBase>(
            score, dp, sidecar_page, q_pair_base, owner_krow, lane_col_group,
            softmax_scale, softmax_scale_log2, p_frag, ds_frag);

        dv_dk_mmac_owner16_qready<FirstAccum>(
            p_frag, ds_frag, dvdk_low, dvdk_high, dv_acc, dk_acc);
    } else {
        DvDkSourceRegs4 dvdk_low;
        dv_dk_read_owner16_sources4<Tile, 0, MBlockBase>(
            lds, page, dvdk_low);

        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        const float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
        softmax_ds_owner16_causal_exact_tile_ctx<Tile, MBlockBase>(
            score, dp, sidecar_page, q_pair_base, owner_krow, lane_col_group,
            softmax_scale, softmax_scale_log2, p_frag, ds_frag);

        dv_dk_mmac_owner16_read4x2<Tile, MBlockBase, FirstAccum>(
            lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
    }
}

template <typename Tile,
          typename Wdra,
          int MBlockBase,
          bool FirstAccum>
__device__ __forceinline__ void consume_mq_tile_owner16(
    __half* lds,
    int seqlen,
    int k_base,
    int q_tile,
    int causal,
    float softmax_scale,
    int owner_nblock,
    int lane,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    static_assert(MBlockBase + 1 < DkvLdsLayout<Tile>::kRawMBlocksPerMqTile,
                  "Mq tile consumer must start on a valid M16 pair");
    constexpr bool kLastPair =
        MBlockBase + 2 >= DkvLdsLayout<Tile>::kRawMBlocksPerMqTile;

    consume_mq_mpair_owner16<Tile, Wdra, MBlockBase, FirstAccum, kLastPair>(
        lds, seqlen, k_base, q_tile, causal, softmax_scale, owner_nblock,
        lane, page, kv_regs, dv_acc, dk_acc);

    if constexpr (!kLastPair) {
        consume_mq_tile_owner16<Tile, Wdra, MBlockBase + 2, false>(
            lds, seqlen, k_base, q_tile, causal, softmax_scale,
            owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void consume_mq_tile_owner16_dyn(
    __half* lds,
    int seqlen,
    int k_base,
    int q_tile,
    int causal,
    float softmax_scale,
    int owner_nblock,
    int lane,
    int page,
    const Owner16KvRegs& kv_regs,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
#pragma clang loop unroll(disable)
    for (int m_block = 0;
         m_block < DkvLdsLayout<Tile>::kRawMBlocksPerMqTile;
         m_block += 2) {
        const bool first_accum = q_tile == 0 && m_block == 0;
        const bool release_page =
            m_block + 2 >= DkvLdsLayout<Tile>::kRawMBlocksPerMqTile;

        ins::F32x4 score[2];
        ins::F32x4 dp[2];
        score_dp_mmac_owner16_dyn<Tile>(
            lds, page, m_block, kv_regs, score, dp);

        DvDkSourceRegs4 dvdk_low;
        dv_dk_read_owner16_sources4_dyn<Tile>(
            lds, page, 0, m_block, dvdk_low);

        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        const float* sidecar_page = sidecar_page_ptr<Tile>(lds, page);
        softmax_ds_owner16_from_lds_sidecar_dyn<Tile>(
            score, dp, sidecar_page, q_tile, m_block, k_base, owner_nblock,
            lane, seqlen, causal, softmax_scale, p_frag, ds_frag);

        dv_dk_mmac_owner16_read4x2_dyn<Tile, Wdra>(
            lds, m_block, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page,
            first_accum, release_page);
    }
}

template <typename Tile,
          typename Wdra,
          int ConsumerGroup,
          bool EarlyReleasePage = false>
__device__ __forceinline__ void consumer_dkv_mmac_loop(
    __half* lds,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int seqlen,
    int k_base,
    int q_tiles,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int q_filled_phase = 0;
    int dout_filled_phase = 0;
    int q0_filled_phase = 0;
    int q1_filled_phase = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int owner_k_base = k_base + owner_nblock * 16;
    const int owner_krow = owner_k_base + lane_n;
    const float softmax_scale_log2 = softmax_scale * kLog2E;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];
    Owner16KvRegs kv_regs;

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);
    latch_owner16_kv_regs<Tile>(lds, owner_nblock, kv_regs);
    if constexpr (Tile::kOverlayRawOnResidentKv) {
        arrive_resident_used<Wdra>();
    }

    if (q_tiles > 0) {
        constexpr int q_tile = 0;
        constexpr int page = 0;
        if constexpr (Tile::kBlockMq == 64) {
            wait_q_filled<Wdra>(q_filled_phase);
            wait_dout_filled<Wdra>(dout_filled_phase);
            consume_mq_mpair_owner16<Tile, Wdra, 0, true, false>(
                lds, seqlen, k_base, q_tile, causal, softmax_scale,
                owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
            consume_mq_mpair_owner16<Tile, Wdra, 2, false, true>(
                lds, seqlen, k_base, q_tile, causal, softmax_scale,
                owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
        } else if constexpr (Tile::kBlockMq == 128) {
            constexpr int q_tile_base = 0;
            wait_q_half_filled<Wdra, 0>(q0_filled_phase);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        0,
                                                        true,
                                                        false>(
                lds, q_tile_base + 0, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        2,
                                                        false,
                                                        true,
                                                        0>(
                lds, q_tile_base + 32, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
            wait_q_half_filled<Wdra, 1>(q1_filled_phase);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        4,
                                                        false,
                                                        false>(
                lds, q_tile_base + 64, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        6,
                                                        false,
                                                        true,
                                                        1>(
                lds, q_tile_base + 96, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
        } else {
            wait_q_filled<Wdra>(q_filled_phase);
            wait_dout_filled<Wdra>(dout_filled_phase);
            consume_mq_tile_owner16_dyn<Tile, Wdra>(
                lds, seqlen, k_base, q_tile, causal, softmax_scale,
                owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
        }
    }

#pragma clang loop unroll(disable)
    for (int q_tile = 1; q_tile < q_tiles; ++q_tile) {
        const int page = raw_page_for_q_tile<Tile>(q_tile);

        if constexpr (Tile::kBlockMq == 64) {
            wait_q_filled<Wdra>(q_filled_phase);
            wait_dout_filled<Wdra>(dout_filled_phase);
            consume_mq_mpair_owner16<Tile, Wdra, 0, false, false>(
                lds, seqlen, k_base, q_tile, causal, softmax_scale,
                owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
            consume_mq_mpair_owner16<Tile, Wdra, 2, false, true>(
                lds, seqlen, k_base, q_tile, causal, softmax_scale,
                owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
        } else if constexpr (Tile::kBlockMq == 128) {
            const int q_tile_base = q_tile * Tile::kBlockMq;
            wait_q_half_filled<Wdra, 0>(q0_filled_phase);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        0,
                                                        false,
                                                        false>(
                lds, q_tile_base + 0, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        2,
                                                        false,
                                                        true,
                                                        0>(
                lds, q_tile_base + 32, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
            wait_q_half_filled<Wdra, 1>(q1_filled_phase);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        4,
                                                        false,
                                                        false>(
                lds, q_tile_base + 64, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
            consume_mq_mpair_owner16_causal_exact_tile<Tile,
                                                        Wdra,
                                                        6,
                                                        false,
                                                        true,
                                                        1>(
                lds, q_tile_base + 96, owner_krow, lane_col_group,
                softmax_scale, softmax_scale_log2, page, kv_regs, dv_acc,
                dk_acc);
        } else {
            wait_q_filled<Wdra>(q_filled_phase);
            wait_dout_filled<Wdra>(dout_filled_phase);
            consume_mq_tile_owner16_dyn<Tile, Wdra>(
                lds, seqlen, k_base, q_tile, causal, softmax_scale,
                owner_nblock, lane, page, kv_regs, dv_acc, dk_acc);
        }
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
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
            __builtin_hcu_s_abarrier_init(Bar::kResidentUsed, 8);
        }
        __builtin_hcu_s_abarrier_init(Bar::kQ0Filled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQ0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kDout0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQ1Filled, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQ1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kDout1Used, 8);
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
        producer_kq_loop<Tile, Bar>(
            q + tensor_base, k + tensor_base, lds, q_base, k_base, seqlen, dim,
            q_tiles, wave_local, packed_sidecar, row_base, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        consumer_dkv_mmac_loop<Tile, Bar, 0, true>(
            lds, dk, dv, tensor_base, seqlen, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        consumer_dkv_mmac_loop<Tile, Bar, 1, true>(
            lds, dk, dv, tensor_base, seqlen, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        producer_vdout_loop<Tile, Bar>(
            v + tensor_base, dout + tensor_base, lds, q_base, k_base, seqlen,
            dim, q_tiles, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __syncthreads();
    __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
    if constexpr (Tile::kOverlayRawOnResidentKv) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentUsed);
    }
    __builtin_hcu_s_abarrier_inv(Bar::kQ0Filled);
    __builtin_hcu_s_abarrier_inv(Bar::kQ0Used);
    __builtin_hcu_s_abarrier_inv(Bar::kDout0Used);
    __builtin_hcu_s_abarrier_inv(Bar::kQ1Filled);
    __builtin_hcu_s_abarrier_inv(Bar::kQ1Used);
    __builtin_hcu_s_abarrier_inv(Bar::kDout1Used);
    __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    __syncthreads();
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
        const bool pass =
            status == SHAOBO_FA3_STATUS_SUCCESS &&
            dk_metrics.bad_count == 0 && dv_metrics.bad_count == 0 &&
            dk_metrics.max_abs <= 5.0e-4f &&
            dv_metrics.max_abs <= 5.0e-4f &&
            dk_metrics.rel_l2 <= rel_l2_limit &&
            dv_metrics.rel_l2 <= rel_l2_limit;

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
