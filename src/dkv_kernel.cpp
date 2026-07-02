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
           p->head_dim_qk == dkv::DkvTileD128Mq32Nk128::kHeadDim &&
           p->head_dim_v == dkv::DkvTileD128Mq32Nk128::kHeadDim &&
           p->dtype == SHAOBO_FA3_DTYPE_FP16 &&
           p->layout == SHAOBO_FA3_LAYOUT_BHSD;
}

inline bool valid_probe_shape(const ShaoboFa3Params* p) {
    return valid_dkv_shape(p) &&
           (p->dkv_path == dkv::kDkvPathWaspProbe ||
            p->dkv_path == dkv::kDkvPathWaspSoftmaxDsSidecar ||
            p->dkv_path == dkv::kDkvPathWaspFragmentSidecar ||
            p->dkv_path == dkv::kDkvPathWaspDkvMmac ||
            p->dkv_path == dkv::kDkvPathWaspDkvMmac12Wave ||
            p->dkv_path == dkv::kDkvPathWaspDkvMmac12WaveMq64 ||
            p->dkv_path ==
                dkv::kDkvPathWaspDkvMmac12WaveSidecarOverlay ||
            p->dkv_path ==
                dkv::kDkvPathWaspDkvMmac12WaveScoreDpBrick ||
            p->dkv_path ==
                dkv::kDkvPathWaspDkvMmac12WaveMq64Semantic ||
            p->dkv_path ==
                dkv::kDkvPathWaspDkvMmac12WaveCausalSkip ||
            p->dkv_path ==
                dkv::kDkvPathWaspDkvMmac12WaveMixedScoreBrick ||
            p->dkv_path ==
                dkv::kDkvPathWaspDkvMmac12WaveEarlyRelease) &&
           p->seqlen_q % dkv::DkvTileD128Mq32Nk128::kResidentNk == 0 &&
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
    return blocks * dkv::DkvTileD128Mq32Nk128::kProbeDiagFloatsPerBlock *
           sizeof(float);
}

inline int64_t tensor_offset(
    int b, int h, int s, int d, int heads, int seqlen, int dim) {
    return ((static_cast<int64_t>(b) * heads + h) * seqlen + s) * dim + d;
}

template <typename Tile>
struct DkvLdsLayout {
    static constexpr int kBlockBytes = 32 * 32 * Tile::kHalfBytes;
    static constexpr int kRawPages = Tile::kRawBuffers;
    static constexpr int kMBlocksPerMqTile = Tile::kBlockMq / 32;
    static constexpr int kDBlocksPerMqTile = Tile::kHeadDim / 32;
    static constexpr int kBlocksPerMqTile =
        kMBlocksPerMqTile * kDBlocksPerMqTile;
    static constexpr int kBlocksPerKvTile =
        Tile::kResidentNk / 32 * Tile::kHeadDim / 32;
    static constexpr int kQBase = 0;
    static constexpr int kDoutBase =
        kQBase + kRawPages * kBlocksPerMqTile * kBlockBytes;
    static constexpr int kKBase =
        kDoutBase + kRawPages * kBlocksPerMqTile * kBlockBytes;
    static constexpr int kVBase = kKBase + kBlocksPerKvTile * kBlockBytes;
    static constexpr int kQtBase = kVBase + kBlocksPerKvTile * kBlockBytes;
    static constexpr int kDoutTBase =
        kQtBase + kRawPages * kBlocksPerMqTile * kBlockBytes;
    static constexpr int kBytes =
        kDoutTBase + kRawPages * kBlocksPerMqTile * kBlockBytes;
    static_assert(kBytes <= Tile::kLdsBudgetBytes,
                  "dKV bringup probe LDS plan must fit 128KB");
};

template <typename Tile>
struct DkvSemanticMq64LdsLayout {
    static constexpr int kBlockBytes = 32 * 32 * Tile::kHalfBytes;
    static constexpr int kMBlocksPerMqTile = Tile::kBlockMq / 32;
    static constexpr int kDBlocksPerMqTile = Tile::kHeadDim / 32;
    static constexpr int kBlocksPerMqTile =
        kMBlocksPerMqTile * kDBlocksPerMqTile;
    static constexpr int kBlocksPerKvTile =
        Tile::kResidentNk / 32 * Tile::kHeadDim / 32;
    static constexpr int kKBase = 0;
    static constexpr int kVBase = kKBase + kBlocksPerKvTile * kBlockBytes;
    static constexpr int kPage0Base = kVBase + kBlocksPerKvTile * kBlockBytes;
    static constexpr int kPageMatrixBytes =
        kBlocksPerMqTile * kBlockBytes;
    static constexpr int kPageBytes = 2 * kPageMatrixBytes;
    static constexpr int kBytes =
        kPage0Base + Tile::kSemanticPages * kPageBytes;

    static_assert(Tile::kBlockMq == 64,
                  "semantic-page layout is only defined for Mq64");
    static_assert(kPageMatrixBytes == Tile::kSemanticMatrixBytes,
                  "semantic matrix byte budget mismatch");
    static_assert(kBytes == Tile::kSemanticLdsBytes,
                  "semantic-page LDS budget mismatch");
};

template <typename Tile>
__device__ __forceinline__ int raw_page_block_offset(int page, int d_block) {
    return (page * DkvLdsLayout<Tile>::kBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int raw_page_block_offset_m(
    int page,
    int m_block,
    int d_block) {
    return (page * DkvLdsLayout<Tile>::kBlocksPerMqTile +
            m_block * DkvLdsLayout<Tile>::kDBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int source_page_block_offset(
    int page,
    int d_block) {
    return (page * DkvLdsLayout<Tile>::kBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int source_page_block_offset_m(
    int page,
    int m_block,
    int d_block) {
    return (page * DkvLdsLayout<Tile>::kBlocksPerMqTile +
            m_block * DkvLdsLayout<Tile>::kDBlocksPerMqTile + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int kv_block_offset(int row_block, int d_block) {
    return (row_block * 4 + d_block) *
           DkvLdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int semantic_kv_block_offset(
    int row_block,
    int d_block) {
    return (row_block * DkvSemanticMq64LdsLayout<Tile>::kDBlocksPerMqTile +
            d_block) *
           DkvSemanticMq64LdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ int semantic_page_block_offset_m(
    int page,
    int matrix,
    int m_block,
    int d_block) {
    return DkvSemanticMq64LdsLayout<Tile>::kPage0Base +
           page * DkvSemanticMq64LdsLayout<Tile>::kPageBytes +
           matrix * DkvSemanticMq64LdsLayout<Tile>::kPageMatrixBytes +
           (m_block * DkvSemanticMq64LdsLayout<Tile>::kDBlocksPerMqTile +
            d_block) *
               DkvSemanticMq64LdsLayout<Tile>::kBlockBytes;
}

template <typename Tile>
__device__ __forceinline__ float* raw_q_sidecar_overlay_page(
    __half* lds,
    int page) {
    char* base = reinterpret_cast<char*>(lds);
    const int byte_offset =
        DkvLdsLayout<Tile>::kQBase +
        page * DkvLdsLayout<Tile>::kBlocksPerMqTile *
            DkvLdsLayout<Tile>::kBlockBytes;
    return reinterpret_cast<float*>(base + byte_offset);
}

template <typename Tile>
__device__ __forceinline__ bool causal_q_tile_before_k_tile(
    int q_tile,
    int k_base,
    int causal) {
    return causal != 0 &&
           q_tile * Tile::kBlockMq + Tile::kBlockMq - 1 < k_base;
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
__device__ __forceinline__ void publish_mq32_source_layout_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    size_t tile_base,
    int page,
    int wave_local) {
    if (src == nullptr) {
        return;
    }
    constexpr int kBlockElems = 32 * 32;
    const __half* src_tile =
        src + tile_base + static_cast<size_t>(wave_local) * kBlockElems;
    ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, row_stride);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, srsrc,
        lds_base + source_page_block_offset<Tile>(page, wave_local), true);
}

template <typename Tile>
__device__ __forceinline__ void publish_mq64_tile(
    __half* lds,
    int lds_base,
    const __half* src,
    int row_stride,
    int q_base,
    int page,
    int wave_local) {
    static_assert(Tile::kBlockMq == 64,
                  "publish_mq64_tile expects a 64-row q tile");
#pragma unroll
    for (int m_block = 0; m_block < 2; ++m_block) {
        const __half* src_tile =
            src + static_cast<int64_t>(q_base + m_block * 32) * row_stride +
            wave_local * 32;
        ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, row_stride);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, srsrc,
            lds_base +
                raw_page_block_offset_m<Tile>(page, m_block, wave_local),
            true);
    }
}

template <typename Tile>
__device__ __forceinline__ void publish_mq64_source_from_mq32_tiles(
    __half* lds,
    int lds_base,
    const __half* src,
    int source_tiles_mq32,
    int bh,
    int mq64_q_tile,
    int page,
    int wave_local) {
    static_assert(Tile::kBlockMq == 64,
                  "Mq64 source publisher expects a 64-row q tile");
    if (src == nullptr) {
        return;
    }
    constexpr int kMq32 = 32;
    constexpr int kBlockElems = 32 * 32;
#pragma unroll
    for (int m_block = 0; m_block < 2; ++m_block) {
        const int source_q_tile = mq64_q_tile * 2 + m_block;
        const size_t tile_base =
            (static_cast<size_t>(bh) * static_cast<size_t>(source_tiles_mq32) +
             static_cast<size_t>(source_q_tile)) *
            static_cast<size_t>(Tile::kHeadDim) * kMq32;
        const __half* src_tile =
            src + tile_base + static_cast<size_t>(wave_local) * kBlockElems;
        ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, kMq32);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, srsrc,
            lds_base +
                source_page_block_offset_m<Tile>(page, m_block, wave_local),
            true);
    }
}

template <typename Tile>
__device__ __forceinline__ void publish_mq64_semantic_matrix(
    __half* lds,
    int page,
    int matrix,
    const __half* src,
    int row_stride,
    int q_base,
    int wave_local) {
    static_assert(Tile::kBlockMq == 64,
                  "semantic matrix publisher expects a 64-row q tile");
#pragma unroll
    for (int m_block = 0; m_block < 2; ++m_block) {
        const __half* src_tile =
            src + static_cast<int64_t>(q_base + m_block * 32) * row_stride +
            wave_local * 32;
        ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, row_stride);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, srsrc,
            semantic_page_block_offset_m<Tile>(
                page, matrix, m_block, wave_local),
            true);
    }
}

template <typename Tile>
__device__ __forceinline__ void publish_mq64_semantic_source_from_mq32_tiles(
    __half* lds,
    int page,
    int matrix,
    const __half* src,
    int source_tiles_mq32,
    int bh,
    int mq64_q_tile,
    int wave_local) {
    static_assert(Tile::kBlockMq == 64,
                  "semantic source publisher expects a 64-row q tile");
    if (src == nullptr) {
        return;
    }
    constexpr int kMq32 = 32;
    constexpr int kBlockElems = 32 * 32;
#pragma unroll
    for (int m_block = 0; m_block < 2; ++m_block) {
        const int source_q_tile = mq64_q_tile * 2 + m_block;
        const size_t tile_base =
            (static_cast<size_t>(bh) * static_cast<size_t>(source_tiles_mq32) +
             static_cast<size_t>(source_q_tile)) *
            static_cast<size_t>(Tile::kHeadDim) * kMq32;
        const __half* src_tile =
            src + tile_base + static_cast<size_t>(wave_local) * kBlockElems;
        ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, kMq32);
        ins::matrix_load_32x32_b16_bps_lds(
            lds, srsrc,
            semantic_page_block_offset_m<Tile>(
                page, matrix, m_block, wave_local),
            true);
    }
}

template <typename Tile>
__device__ __forceinline__ void publish_sidecar_page(
    float* sidecar_lds,
    const float* scores_max,
    const float* scores_sum,
    const float* delta,
    int64_t row_base,
    int q_base,
    int seqlen,
    int page,
    int wave_local,
    int lane,
    float softmax_scale_log2) {
    if (sidecar_lds == nullptr || wave_local != 0 || lane >= Tile::kBlockMq) {
        return;
    }
    const int q_row = q_base + lane;
    const bool valid = q_row < seqlen;
    const int64_t row = row_base + q_row;
    const float row_max = valid && scores_max != nullptr ? scores_max[row] : 0.0f;
    const float row_sum = valid && scores_sum != nullptr ? scores_sum[row] : 1.0f;
    const float row_delta = valid && delta != nullptr ? delta[row] : 0.0f;
    float* page_base = sidecar_lds + page * Tile::kSidecarFloats;
    page_base[Tile::kSidecarMaxLog2Base + lane] =
        row_max * softmax_scale_log2;
    page_base[Tile::kSidecarInvSumBase + lane] =
        row_sum != 0.0f ? 1.0f / row_sum : 0.0f;
    page_base[Tile::kSidecarDeltaBase + lane] = row_delta;
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
    const __half* q_t_source,
    __half* lds,
    float* sidecar_lds,
    const float* scores_max,
    const float* scores_sum,
    const float* delta,
    int64_t row_base,
    int bh,
    int q_base,
    int k_base,
    int seqlen,
    int row_stride,
    int q_tiles,
    int wave_local,
    int lane,
    float softmax_scale_log2,
    int fragment_sidecar_enabled,
    int source_layout_enabled) {
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
        if (fragment_sidecar_enabled) {
            publish_sidecar_page<Tile>(
                sidecar_lds, scores_max, scores_sum, delta, row_base,
                q_base + q_tile * Tile::kBlockMq, seqlen, page, wave_local,
                lane, softmax_scale_log2);
        }
        if (source_layout_enabled) {
            const int source_q_tile = q_base / Tile::kBlockMq + q_tile;
            const size_t tile_base =
                (static_cast<size_t>(bh) * static_cast<size_t>(q_tiles) +
                 static_cast<size_t>(source_q_tile)) *
                static_cast<size_t>(Tile::kHeadDim) * Tile::kBlockMq;
            publish_mq32_source_layout_tile<Tile>(
                lds, Layout::kQtBase, q_t_source, Tile::kBlockMq,
                tile_base, page, wave_local);
        }
        arrive_raw_filled_page<Wdra>(page);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_dout_v_loop(
    const __half* dout_base_ptr,
    const __half* v_base_ptr,
    const __half* dout_t_source,
    __half* lds,
    int bh,
    int q_base,
    int k_base,
    int row_stride,
    int q_tiles,
    int wave_local,
    int source_layout_enabled) {
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        if (q_tile >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }
        seq_raw_filled_page<Wdra>(page);
        publish_mq32_tile<Tile>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);
        if (source_layout_enabled) {
            const int source_q_tile = q_base / Tile::kBlockMq + q_tile;
            const size_t tile_base =
                (static_cast<size_t>(bh) * static_cast<size_t>(q_tiles) +
                 static_cast<size_t>(source_q_tile)) *
                static_cast<size_t>(Tile::kHeadDim) * Tile::kBlockMq;
            publish_mq32_source_layout_tile<Tile>(
                lds, Layout::kDoutTBase, dout_t_source, Tile::kBlockMq,
                tile_base, page, wave_local);
        }
        arrive_raw_filled_page<Wdra>(page);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_all_loop(
    const __half* q_base_ptr,
    const __half* k_base_ptr,
    const __half* v_base_ptr,
    const __half* dout_base_ptr,
    const __half* q_t_source,
    const __half* dout_t_source,
    __half* lds,
    int bh,
    int q_base,
    int k_base,
    int seqlen,
    int row_stride,
    int q_tiles,
    int wave_local) {
    (void)seqlen;
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        if (q_tile >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }
        seq_raw_filled_page<Wdra>(page);
        publish_mq32_tile<Tile>(
            lds, Layout::kQBase, q_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);
        publish_mq32_tile<Tile>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);

        const int source_q_tile = q_base / Tile::kBlockMq + q_tile;
        const size_t tile_base =
            (static_cast<size_t>(bh) * static_cast<size_t>(q_tiles) +
             static_cast<size_t>(source_q_tile)) *
            static_cast<size_t>(Tile::kHeadDim) * Tile::kBlockMq;
        publish_mq32_source_layout_tile<Tile>(
            lds, Layout::kQtBase, q_t_source, Tile::kBlockMq,
            tile_base, page, wave_local);
        publish_mq32_source_layout_tile<Tile>(
            lds, Layout::kDoutTBase, dout_t_source, Tile::kBlockMq,
            tile_base, page, wave_local);
        arrive_raw_filled_page<Wdra>(page);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_all_loop_causal_skip(
    const __half* q_base_ptr,
    const __half* k_base_ptr,
    const __half* v_base_ptr,
    const __half* dout_base_ptr,
    const __half* q_t_source,
    const __half* dout_t_source,
    __half* lds,
    int bh,
    int q_base,
    int k_base,
    int seqlen,
    int row_stride,
    int q_tiles,
    int wave_local,
    int causal) {
    (void)seqlen;
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;
    int packet_idx = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        if (causal_q_tile_before_k_tile<Tile>(q_tile, k_base, causal)) {
            continue;
        }

        const int page = packet_idx & 1;
        if (packet_idx >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }
        seq_raw_filled_page<Wdra>(page);
        publish_mq32_tile<Tile>(
            lds, Layout::kQBase, q_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);
        publish_mq32_tile<Tile>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);

        const int source_q_tile = q_base / Tile::kBlockMq + q_tile;
        const size_t tile_base =
            (static_cast<size_t>(bh) * static_cast<size_t>(q_tiles) +
             static_cast<size_t>(source_q_tile)) *
            static_cast<size_t>(Tile::kHeadDim) * Tile::kBlockMq;
        publish_mq32_source_layout_tile<Tile>(
            lds, Layout::kQtBase, q_t_source, Tile::kBlockMq,
            tile_base, page, wave_local);
        publish_mq32_source_layout_tile<Tile>(
            lds, Layout::kDoutTBase, dout_t_source, Tile::kBlockMq,
            tile_base, page, wave_local);
        arrive_raw_filled_page<Wdra>(page);
        ++packet_idx;
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_all_loop_sidecar_overlay(
    const __half* q_base_ptr,
    const __half* k_base_ptr,
    const __half* v_base_ptr,
    const __half* dout_base_ptr,
    const __half* q_t_source,
    const __half* dout_t_source,
    const float* scores_max,
    const float* scores_sum,
    const float* delta,
    __half* lds,
    int64_t row_base,
    int bh,
    int q_base,
    int k_base,
    int seqlen,
    int row_stride,
    int q_tiles,
    int wave_local,
    int lane,
    float softmax_scale_log2) {
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        if (q_tile >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }
        seq_raw_filled_page<Wdra>(page);
        publish_mq32_tile<Tile>(
            lds, Layout::kQBase, q_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);
        publish_mq32_tile<Tile>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride,
            q_base + q_tile * Tile::kBlockMq, page, wave_local);

        const int source_q_tile = q_base / Tile::kBlockMq + q_tile;
        const size_t tile_base =
            (static_cast<size_t>(bh) * static_cast<size_t>(q_tiles) +
             static_cast<size_t>(source_q_tile)) *
            static_cast<size_t>(Tile::kHeadDim) * Tile::kBlockMq;
        publish_mq32_source_layout_tile<Tile>(
            lds, Layout::kQtBase, q_t_source, Tile::kBlockMq,
            tile_base, page, wave_local);
        publish_mq32_source_layout_tile<Tile>(
            lds, Layout::kDoutTBase, dout_t_source, Tile::kBlockMq,
            tile_base, page, wave_local);
        arrive_raw_filled_page<Wdra>(page);

        if (q_tile >= 1) {
            const int sidecar_q_tile = q_tile - 1;
            const int sidecar_page_id = sidecar_q_tile & 1;
            wait_raw_used_page<Wdra>(
                sidecar_page_id, raw0_used_phase, raw1_used_phase);
            seq_raw_filled_page<Wdra>(sidecar_page_id);
            float* sidecar_page =
                raw_q_sidecar_overlay_page<Tile>(lds, sidecar_page_id);
            publish_sidecar_page<Tile>(
                sidecar_page, scores_max, scores_sum, delta, row_base,
                q_base + sidecar_q_tile * Tile::kBlockMq, seqlen, 0,
                wave_local, lane, softmax_scale_log2);
            arrive_raw_filled_page<Wdra>(sidecar_page_id);
        }
    }

    if (q_tiles > 0) {
        const int sidecar_q_tile = q_tiles - 1;
        const int sidecar_page_id = sidecar_q_tile & 1;
        wait_raw_used_page<Wdra>(
            sidecar_page_id, raw0_used_phase, raw1_used_phase);
        seq_raw_filled_page<Wdra>(sidecar_page_id);
        float* sidecar_page =
            raw_q_sidecar_overlay_page<Tile>(lds, sidecar_page_id);
        publish_sidecar_page<Tile>(
            sidecar_page, scores_max, scores_sum, delta, row_base,
            q_base + sidecar_q_tile * Tile::kBlockMq, seqlen, 0, wave_local,
            lane, softmax_scale_log2);
        arrive_raw_filled_page<Wdra>(sidecar_page_id);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_all_loop_mq64_single(
    const __half* q_base_ptr,
    const __half* k_base_ptr,
    const __half* v_base_ptr,
    const __half* dout_base_ptr,
    const __half* q_t_source,
    const __half* dout_t_source,
    __half* lds,
    int bh,
    int k_base,
    int row_stride,
    int wave_local) {
    using Layout = DkvLdsLayout<Tile>;
    int raw0_used_phase = 0;
    constexpr int kSourceTilesMq32 =
        Tile::kProbeSeqLen / dkv::DkvTileD128Mq32Nk128::kBlockMq;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < Tile::kQTilesPerCta; ++q_tile) {
        if (q_tile >= 1) {
            ins::abarrier_try_wait<true>(Wdra::kRaw0Used, raw0_used_phase);
        }
        ins::abarrier_seq<false>(Wdra::kRaw0Filled);
        publish_mq64_tile<Tile>(
            lds, Layout::kQBase, q_base_ptr, row_stride,
            q_tile * Tile::kBlockMq, 0, wave_local);
        publish_mq64_tile<Tile>(
            lds, Layout::kDoutBase, dout_base_ptr, row_stride,
            q_tile * Tile::kBlockMq, 0, wave_local);
        publish_mq64_source_from_mq32_tiles<Tile>(
            lds, Layout::kQtBase, q_t_source, kSourceTilesMq32, bh, q_tile,
            0, wave_local);
        publish_mq64_source_from_mq32_tiles<Tile>(
            lds, Layout::kDoutTBase, dout_t_source, kSourceTilesMq32, bh,
            q_tile, 0, wave_local);
        ins::abarrier_arrive_cnt<false>(Wdra::kRaw0Filled, 1);
    }
}

template <typename Tile, typename Wdra>
__device__ __forceinline__ void producer_all_loop_mq64_semantic(
    const __half* q_base_ptr,
    const __half* k_base_ptr,
    const __half* v_base_ptr,
    const __half* dout_base_ptr,
    const __half* q_t_source,
    const __half* dout_t_source,
    __half* lds,
    int bh,
    int k_base,
    int seqlen,
    int row_stride,
    int wave_local) {
    using Layout = DkvSemanticMq64LdsLayout<Tile>;
    int raw0_used_phase = 0;
    int raw1_used_phase = 0;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int source_tiles_mq32 =
        seqlen / dkv::DkvTileD128Mq32Nk128::kBlockMq;

    ins::abarrier_seq<false>(Wdra::kResidentFilled);
    publish_nk128_tile<Tile>(
        lds, Layout::kKBase, k_base_ptr, row_stride, k_base, wave_local);
    publish_nk128_tile<Tile>(
        lds, Layout::kVBase, v_base_ptr, row_stride, k_base, wave_local);
    ins::abarrier_arrive_cnt<false>(Wdra::kResidentFilled, 1);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        if (q_tile >= 2) {
            wait_raw_used_page<Wdra>(
                page, raw0_used_phase, raw1_used_phase);
        }

        seq_raw_filled_page<Wdra>(page);
        publish_mq64_semantic_matrix<Tile>(
            lds, page, 0, q_base_ptr, row_stride, q_tile * Tile::kBlockMq,
            wave_local);
        publish_mq64_semantic_matrix<Tile>(
            lds, page, 1, dout_base_ptr, row_stride,
            q_tile * Tile::kBlockMq, wave_local);
        arrive_raw_filled_page<Wdra>(page);

        if (q_tile >= 1) {
            const int prev_tile = q_tile - 1;
            const int prev_page = prev_tile & 1;
            wait_raw_used_page<Wdra>(
                prev_page, raw0_used_phase, raw1_used_phase);
            seq_raw_filled_page<Wdra>(prev_page);
            publish_mq64_semantic_source_from_mq32_tiles<Tile>(
                lds, prev_page, 0, q_t_source, source_tiles_mq32, bh,
                prev_tile, wave_local);
            publish_mq64_semantic_source_from_mq32_tiles<Tile>(
                lds, prev_page, 1, dout_t_source, source_tiles_mq32, bh,
                prev_tile, wave_local);
            arrive_raw_filled_page<Wdra>(prev_page);
        }
    }

    if (q_tiles > 0) {
        const int last_tile = q_tiles - 1;
        const int last_page = last_tile & 1;
        wait_raw_used_page<Wdra>(
            last_page, raw0_used_phase, raw1_used_phase);
        seq_raw_filled_page<Wdra>(last_page);
        publish_mq64_semantic_source_from_mq32_tiles<Tile>(
            lds, last_page, 0, q_t_source, source_tiles_mq32, bh,
            last_tile, wave_local);
        publish_mq64_semantic_source_from_mq32_tiles<Tile>(
            lds, last_page, 1, dout_t_source, source_tiles_mq32, bh,
            last_tile, wave_local);
        arrive_raw_filled_page<Wdra>(last_page);
    }
}

template <typename Tile>
__device__ __forceinline__ void score_dp_mmac_fragments(
    __half* lds,
    int consumer_group,
    int page,
    int wave_local,
    ins::F32x4 (&score)[4],
    ins::F32x4 (&dp)[4]) {
    using Layout = DkvLdsLayout<Tile>;

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

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

#pragma unroll
        for (int n_idx = 0; n_idx < 2; ++n_idx) {
#pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                const int frag = n_idx * 2 + m_idx;
                const ins::Vec4F16 k_frag[2] = {
                    n_idx == 0 ? k0.f16x4[0] : k1.f16x4[0],
                    n_idx == 0 ? k0.f16x4[1] : k1.f16x4[1],
                };
                const ins::Vec4F16 q_frag[2] = {
                    m_idx == 0 ? q0.f16x4[0] : q1.f16x4[0],
                    m_idx == 0 ? q0.f16x4[1] : q1.f16x4[1],
                };
                const ins::Vec4F16 v_frag[2] = {
                    n_idx == 0 ? v0.f16x4[0] : v1.f16x4[0],
                    n_idx == 0 ? v0.f16x4[1] : v1.f16x4[1],
                };
                const ins::Vec4F16 dout_frag[2] = {
                    m_idx == 0 ? dout0.f16x4[0] : dout1.f16x4[0],
                    m_idx == 0 ? dout0.f16x4[1] : dout1.f16x4[1],
                };
#pragma unroll
                for (int k_half = 0; k_half < 2; ++k_half) {
                    const bool first = d_block == 0 && k_half == 0;
                    if (first) {
                        score[frag].f32 = ins::mmac_f16_lit(
                            k_frag[k_half], q_frag[k_half], zero.f32);
                        dp[frag].f32 = ins::mmac_f16_lit(
                            v_frag[k_half], dout_frag[k_half], zero.f32);
                    } else {
                        score[frag].f32 = ins::mmac_f16_lit(
                            k_frag[k_half], q_frag[k_half],
                            score[frag].f32);
                        dp[frag].f32 = ins::mmac_f16_lit(
                            v_frag[k_half], dout_frag[k_half], dp[frag].f32);
                    }
                }
            }
        }
    }
    ins::lower_priority();
}

template <typename Tile>
__device__ __forceinline__ void softmax_ds_fragment_from_sidecar(
    const ins::F32x4 (&score)[4],
    const ins::F32x4 (&dp)[4],
    const float* sidecar_page,
    int q_tile,
    int k_base,
    int row_block,
    int lane,
    int seqlen,
    int causal,
    float softmax_scale,
    ins::Vec4F16 (&p_frag)[4],
    ins::Vec4F16 (&ds_frag)[4]) {
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int q_base = q_tile * Tile::kBlockMq;
    const bool full_valid_tile =
        q_base + Tile::kBlockMq <= seqlen &&
        k_base + row_block * 32 + 31 < seqlen &&
        (!causal || (k_base + row_block * 32 + 31 < q_base));

#pragma unroll
    for (int n_idx = 0; n_idx < 2; ++n_idx) {
#pragma unroll
        for (int m_idx = 0; m_idx < 2; ++m_idx) {
            const int frag = n_idx * 2 + m_idx;
            const int krow = k_base + row_block * 32 + n_idx * 16 + lane_n;
            const int local_m_base = m_idx * 16 + lane_col_group * 4;
            ins::Vec4F16 p_vec{};
            ins::Vec4F16 ds_vec{};
#pragma unroll
            for (int vec_id = 0; vec_id < 4; ++vec_id) {
                const int local_m = local_m_base + vec_id;
                const int qrow = q_base + local_m;
                const bool valid_pair =
                    full_valid_tile ||
                    (krow < seqlen && qrow < seqlen &&
                     (!causal || krow <= qrow));
                float p_val = 0.0f;
                float ds_val = 0.0f;
                if (valid_pair) {
                    const float row_max_log2 =
                        sidecar_page[Tile::kSidecarMaxLog2Base + local_m];
                    const float row_inv_sum =
                        sidecar_page[Tile::kSidecarInvSumBase + local_m];
                    const float row_delta =
                        sidecar_page[Tile::kSidecarDeltaBase + local_m];
                    p_val =
                        exp2f(score[frag].scalar[vec_id] *
                                  softmax_scale * kLog2E -
                              row_max_log2) *
                        row_inv_sum;
                    ds_val =
                        p_val * (dp[frag].scalar[vec_id] - row_delta) *
                        softmax_scale;
                }
                p_vec[vec_id] = static_cast<_Float16>(p_val);
                ds_vec[vec_id] = static_cast<_Float16>(ds_val);
            }
            p_frag[frag] = p_vec;
            ds_frag[frag] = ds_vec;
        }
    }
}

template <typename Tile>
__device__ __forceinline__ void score_dp_mmac_owner16(
    __half* lds,
    int owner_nblock,
    int page,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    using Layout = DkvLdsLayout<Tile>;

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    const int row_block32 = owner_nblock >> 1;
    const int n_half = owner_nblock & 1;
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
            Layout::kKBase + kv_block_offset<Tile>(row_block32, d_block);
        const int v_off =
            Layout::kVBase + kv_block_offset<Tile>(row_block32, d_block);

        ins::ds_read_matrix_trans_pair(lds, q_off, q0.f16x8, q1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, k_off, k0.f16x8, k1.f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, dout_off, dout0.f16x8, dout1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, v_off, v0.f16x8, v1.f16x8);
        ins::wait_lgkm(0);

        const ins::F16x8& k_reg = n_half == 0 ? k0 : k1;
        const ins::F16x8& v_reg = n_half == 0 ? v0 : v1;
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

template <typename Tile>
__device__ __forceinline__ void score_dp_mmac_owner16_read2x(
    __half* lds,
    int owner_nblock,
    int page,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    using Layout = DkvLdsLayout<Tile>;

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    const int row_block32 = owner_nblock >> 1;
    const int n_half = owner_nblock & 1;
    ins::raise_priority_2();
#pragma unroll
    for (int d_pair = 0; d_pair < 2; ++d_pair) {
        ins::F16x8 q0[2];
        ins::F16x8 q1[2];
        ins::F16x8 k0[2];
        ins::F16x8 k1[2];
        ins::F16x8 dout0[2];
        ins::F16x8 dout1[2];
        ins::F16x8 v0[2];
        ins::F16x8 v1[2];

#pragma unroll
        for (int local_d = 0; local_d < 2; ++local_d) {
            const int d_block = d_pair * 2 + local_d;
            const int q_off =
                Layout::kQBase + raw_page_block_offset<Tile>(page, d_block);
            const int dout_off =
                Layout::kDoutBase +
                raw_page_block_offset<Tile>(page, d_block);
            const int k_off =
                Layout::kKBase +
                kv_block_offset<Tile>(row_block32, d_block);
            const int v_off =
                Layout::kVBase +
                kv_block_offset<Tile>(row_block32, d_block);

            ins::ds_read_matrix_trans_pair(
                lds, q_off, q0[local_d].f16x8, q1[local_d].f16x8);
            ins::ds_read_matrix_trans_pair(
                lds, k_off, k0[local_d].f16x8, k1[local_d].f16x8);
            ins::ds_read_matrix_trans_pair(
                lds, dout_off, dout0[local_d].f16x8,
                dout1[local_d].f16x8);
            ins::ds_read_matrix_trans_pair(
                lds, v_off, v0[local_d].f16x8, v1[local_d].f16x8);
        }
        ins::wait_lgkm(0);

#pragma unroll
        for (int local_d = 0; local_d < 2; ++local_d) {
            const int d_block = d_pair * 2 + local_d;
            const ins::F16x8& k_reg =
                n_half == 0 ? k0[local_d] : k1[local_d];
            const ins::F16x8& v_reg =
                n_half == 0 ? v0[local_d] : v1[local_d];
#pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                const ins::Vec4F16 q_frag[2] = {
                    m_idx == 0 ? q0[local_d].f16x4[0]
                               : q1[local_d].f16x4[0],
                    m_idx == 0 ? q0[local_d].f16x4[1]
                               : q1[local_d].f16x4[1],
                };
                const ins::Vec4F16 dout_frag[2] = {
                    m_idx == 0 ? dout0[local_d].f16x4[0]
                               : dout1[local_d].f16x4[0],
                    m_idx == 0 ? dout0[local_d].f16x4[1]
                               : dout1[local_d].f16x4[1],
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
    }
    ins::lower_priority();
}

template <typename Tile>
__device__ __forceinline__ void softmax_ds_owner16_from_sidecar(
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
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int q_base = q_tile * Tile::kBlockMq;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    const bool full_valid_tile =
        q_base + Tile::kBlockMq <= seqlen &&
        k_base + owner_nblock * 16 + 15 < seqlen &&
        (!causal || (k_base + owner_nblock * 16 + 15 <= q_base));

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        ins::Vec4F16 p_vec{};
        ins::Vec4F16 ds_vec{};
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_base + local_m;
            const bool valid_pair =
                full_valid_tile ||
                (krow < seqlen && qrow < seqlen &&
                 (!causal || krow <= qrow));
            float p_val = 0.0f;
            float ds_val = 0.0f;
            if (valid_pair) {
                const float row_max_log2 =
                    sidecar_page[Tile::kSidecarMaxLog2Base + local_m];
                const float row_inv_sum =
                    sidecar_page[Tile::kSidecarInvSumBase + local_m];
                const float row_delta =
                    sidecar_page[Tile::kSidecarDeltaBase + local_m];
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale * kLog2E -
                          row_max_log2) *
                    row_inv_sum;
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta) *
                    softmax_scale;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

template <typename Tile>
__device__ __forceinline__ void softmax_ds_owner16_from_global_sidecar(
    const ins::F32x4 (&score)[2],
    const ins::F32x4 (&dp)[2],
    const float* packed_sidecar,
    int64_t row_base,
    int q_tile,
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
    const int q_base = q_tile * Tile::kBlockMq;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    const float* sidecar_tile =
        packed_sidecar +
        (row_base + q_base) * Tile::kPackedSidecarFields;
    const bool full_valid_tile =
        q_base + Tile::kBlockMq <= seqlen &&
        k_base + owner_nblock * 16 + 15 < seqlen &&
        (!causal || (k_base + owner_nblock * 16 + 15 <= q_base));

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        ins::Vec4F16 p_vec{};
        ins::Vec4F16 ds_vec{};
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_base + local_m;
            const bool valid_pair =
                full_valid_tile ||
                (krow < seqlen && qrow < seqlen &&
                 (!causal || krow <= qrow));
            float p_val = 0.0f;
            float ds_val = 0.0f;
            if (valid_pair) {
                const int sidecar_base =
                    local_m * Tile::kPackedSidecarFields;
                const float row_max_log2 = sidecar_tile[sidecar_base + 0];
                const float row_inv_sum = sidecar_tile[sidecar_base + 1];
                const float row_delta = sidecar_tile[sidecar_base + 2];
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale * kLog2E -
                          row_max_log2) *
                    row_inv_sum;
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta) *
                    softmax_scale;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

template <typename Tile>
__device__ __forceinline__ void score_dp_mmac_owner16_mq64(
    __half* lds,
    int owner_nblock,
    ins::F32x4 (&score)[4],
    ins::F32x4 (&dp)[4]) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Tile::kBlockMq == 64,
                  "score_dp_mmac_owner16_mq64 expects Mq64");

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    const int row_block32 = owner_nblock >> 1;
    const int n_half = owner_nblock & 1;
    ins::raise_priority_2();
#pragma unroll
    for (int d_block = 0; d_block < 4; ++d_block) {
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

        const ins::F16x8& k_reg = n_half == 0 ? k0 : k1;
        const ins::F16x8& v_reg = n_half == 0 ? v0 : v1;
#pragma unroll
        for (int m_block = 0; m_block < 2; ++m_block) {
            ins::F16x8 q0;
            ins::F16x8 q1;
            ins::F16x8 dout0;
            ins::F16x8 dout1;
            const int q_off =
                Layout::kQBase + raw_page_block_offset_m<Tile>(
                                     0, m_block, d_block);
            const int dout_off =
                Layout::kDoutBase + raw_page_block_offset_m<Tile>(
                                        0, m_block, d_block);
            ins::ds_read_matrix_trans_pair(lds, q_off, q0.f16x8, q1.f16x8);
            ins::ds_read_matrix_trans_pair(
                lds, dout_off, dout0.f16x8, dout1.f16x8);
            ins::wait_lgkm(0);

#pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                const int frag = m_block * 2 + m_idx;
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
                    score[frag].f32 = ins::mmac_f16_lit(
                        k_reg.f16x4[k_half], q_frag[k_half],
                        first ? zero.f32 : score[frag].f32);
                    dp[frag].f32 = ins::mmac_f16_lit(
                        v_reg.f16x4[k_half], dout_frag[k_half],
                        first ? zero.f32 : dp[frag].f32);
                }
            }
        }
    }
    ins::lower_priority();
}

template <typename Tile, int MBlock>
__device__ __forceinline__ void score_dp_mmac_owner16_mq64_half(
    __half* lds,
    int owner_nblock,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Tile::kBlockMq == 64,
                  "score_dp_mmac_owner16_mq64_half expects Mq64");
    static_assert(MBlock == 0 || MBlock == 1,
                  "Mq64 half must select one 32-row half");

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    const int row_block32 = owner_nblock >> 1;
    const int n_half = owner_nblock & 1;
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
            Layout::kQBase + raw_page_block_offset_m<Tile>(
                                 0, MBlock, d_block);
        const int dout_off =
            Layout::kDoutBase + raw_page_block_offset_m<Tile>(
                                    0, MBlock, d_block);
        const int k_off =
            Layout::kKBase + kv_block_offset<Tile>(row_block32, d_block);
        const int v_off =
            Layout::kVBase + kv_block_offset<Tile>(row_block32, d_block);

        ins::ds_read_matrix_trans_pair(lds, q_off, q0.f16x8, q1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, k_off, k0.f16x8, k1.f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, dout_off, dout0.f16x8, dout1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, v_off, v0.f16x8, v1.f16x8);
        ins::wait_lgkm(0);

        const ins::F16x8& k_reg = n_half == 0 ? k0 : k1;
        const ins::F16x8& v_reg = n_half == 0 ? v0 : v1;
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

template <typename Tile, int MBlock>
__device__ __forceinline__ void score_dp_mmac_owner16_mq64_semantic_half(
    __half* lds,
    int page,
    int owner_nblock,
    ins::F32x4 (&score)[2],
    ins::F32x4 (&dp)[2]) {
    using Layout = DkvSemanticMq64LdsLayout<Tile>;
    static_assert(Tile::kBlockMq == 64,
                  "semantic score/dP helper expects Mq64");
    static_assert(MBlock == 0 || MBlock == 1,
                  "Mq64 half must select one 32-row half");

    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    const int row_block32 = owner_nblock >> 1;
    const int n_half = owner_nblock & 1;
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
            semantic_page_block_offset_m<Tile>(page, 0, MBlock, d_block);
        const int dout_off =
            semantic_page_block_offset_m<Tile>(page, 1, MBlock, d_block);
        const int k_off =
            Layout::kKBase +
            semantic_kv_block_offset<Tile>(row_block32, d_block);
        const int v_off =
            Layout::kVBase +
            semantic_kv_block_offset<Tile>(row_block32, d_block);

        ins::ds_read_matrix_trans_pair(lds, q_off, q0.f16x8, q1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, k_off, k0.f16x8, k1.f16x8);
        ins::ds_read_matrix_trans_pair(
            lds, dout_off, dout0.f16x8, dout1.f16x8);
        ins::ds_read_matrix_trans_pair(lds, v_off, v0.f16x8, v1.f16x8);
        ins::wait_lgkm(0);

        const ins::F16x8& k_reg = n_half == 0 ? k0 : k1;
        const ins::F16x8& v_reg = n_half == 0 ? v0 : v1;
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

template <typename Tile>
__device__ __forceinline__ void softmax_ds_owner16_from_global_sidecar_mq64(
    const ins::F32x4 (&score)[4],
    const ins::F32x4 (&dp)[4],
    const float* packed_sidecar,
    int64_t row_base,
    int q_tile,
    int k_base,
    int owner_nblock,
    int lane,
    int seqlen,
    int causal,
    float softmax_scale,
    ins::Vec4F16 (&p_frag)[4],
    ins::Vec4F16 (&ds_frag)[4]) {
    static_assert(Tile::kBlockMq == 64,
                  "softmax_ds_owner16_from_global_sidecar_mq64 expects Mq64");
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int q_base = q_tile * Tile::kBlockMq;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    const float* sidecar_tile =
        packed_sidecar +
        (row_base + q_base) * Tile::kPackedSidecarFields;
    const bool full_valid_tile =
        q_base + Tile::kBlockMq <= seqlen &&
        k_base + owner_nblock * 16 + 15 < seqlen &&
        (!causal || (k_base + owner_nblock * 16 + 15 <= q_base));

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        const int local_m_base = m_idx * 16 + lane_col_group * 4;
        ins::Vec4F16 p_vec{};
        ins::Vec4F16 ds_vec{};
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_base + local_m;
            const bool valid_pair =
                full_valid_tile ||
                (krow < seqlen && qrow < seqlen &&
                 (!causal || krow <= qrow));
            float p_val = 0.0f;
            float ds_val = 0.0f;
            if (valid_pair) {
                const int sidecar_base =
                    local_m * Tile::kPackedSidecarFields;
                const float row_max_log2 = sidecar_tile[sidecar_base + 0];
                const float row_inv_sum = sidecar_tile[sidecar_base + 1];
                const float row_delta = sidecar_tile[sidecar_base + 2];
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale * kLog2E -
                          row_max_log2) *
                    row_inv_sum;
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta) *
                    softmax_scale;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

template <typename Tile, int MBlock>
__device__ __forceinline__ void softmax_ds_owner16_from_global_sidecar_mq64_half(
    const ins::F32x4 (&score)[2],
    const ins::F32x4 (&dp)[2],
    const float* packed_sidecar,
    int64_t row_base,
    int q_tile,
    int k_base,
    int owner_nblock,
    int lane,
    int seqlen,
    int causal,
    float softmax_scale,
    ins::Vec4F16 (&p_frag)[2],
    ins::Vec4F16 (&ds_frag)[2]) {
    static_assert(Tile::kBlockMq == 64,
                  "softmax_ds_owner16_from_global_sidecar_mq64_half expects Mq64");
    static_assert(MBlock == 0 || MBlock == 1,
                  "Mq64 half must select one 32-row half");
    const int lane_n = lane & 15;
    const int lane_col_group = lane >> 4;
    const int q_base = q_tile * Tile::kBlockMq;
    const int krow = k_base + owner_nblock * 16 + lane_n;
    const float* sidecar_tile =
        packed_sidecar +
        (row_base + q_base) * Tile::kPackedSidecarFields;
    const bool full_valid_tile =
        q_base + Tile::kBlockMq <= seqlen &&
        k_base + owner_nblock * 16 + 15 < seqlen &&
        (!causal || (k_base + owner_nblock * 16 + 15 <= q_base));

#pragma unroll
    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        const int local_m_base =
            MBlock * 32 + m_idx * 16 + lane_col_group * 4;
        ins::Vec4F16 p_vec{};
        ins::Vec4F16 ds_vec{};
#pragma unroll
        for (int vec_id = 0; vec_id < 4; ++vec_id) {
            const int local_m = local_m_base + vec_id;
            const int qrow = q_base + local_m;
            const bool valid_pair =
                full_valid_tile ||
                (krow < seqlen && qrow < seqlen &&
                 (!causal || krow <= qrow));
            float p_val = 0.0f;
            float ds_val = 0.0f;
            if (valid_pair) {
                const int sidecar_base =
                    local_m * Tile::kPackedSidecarFields;
                const float row_max_log2 = sidecar_tile[sidecar_base + 0];
                const float row_inv_sum = sidecar_tile[sidecar_base + 1];
                const float row_delta = sidecar_tile[sidecar_base + 2];
                p_val =
                    exp2f(score[m_idx].scalar[vec_id] *
                              softmax_scale * kLog2E -
                          row_max_log2) *
                    row_inv_sum;
                ds_val =
                    p_val * (dp[m_idx].scalar[vec_id] - row_delta) *
                    softmax_scale;
            }
            p_vec[vec_id] = static_cast<_Float16>(p_val);
            ds_vec[vec_id] = static_cast<_Float16>(ds_val);
        }
        p_frag[m_idx] = p_vec;
        ds_frag[m_idx] = ds_vec;
    }
}

template <bool FirstQTile, int OutIdx>
__device__ __forceinline__ void dv_dk_mmac_one_out(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const ins::F16x8& dout_t,
    const ins::F16x8& q_t,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
#pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
        if constexpr (FirstQTile) {
            dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
                p_frag[m_half], dout_t.f16x4[m_half],
                m_half == 0 ? zero_f16.f32 : dv_acc[OutIdx].f32);
            dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
                ds_frag[m_half], q_t.f16x4[m_half],
                m_half == 0 ? zero_f16.f32 : dk_acc[OutIdx].f32);
        } else {
            dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
                p_frag[m_half], dout_t.f16x4[m_half],
                dv_acc[OutIdx].f32);
            dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
                ds_frag[m_half], q_t.f16x4[m_half], dk_acc[OutIdx].f32);
        }
    }
}

struct DvDkSourceRegs4 {
    ins::F16x8 dout_t0;
    ins::F16x8 dout_t1;
    ins::F16x8 dout_t2;
    ins::F16x8 dout_t3;
    ins::F16x8 q_t0;
    ins::F16x8 q_t1;
    ins::F16x8 q_t2;
    ins::F16x8 q_t3;
};

template <typename Tile, int DBlockBase>
__device__ __forceinline__ void dv_dk_read_owner16_sources4(
    __half* lds,
    int page,
    DvDkSourceRegs4& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK source group must cover two D blocks");

    const int dout_t_off0 =
        Layout::kDoutTBase +
        source_page_block_offset<Tile>(page, DBlockBase);
    const int q_t_off0 =
        Layout::kQtBase + source_page_block_offset<Tile>(page, DBlockBase);
    const int dout_t_off1 =
        Layout::kDoutTBase +
        source_page_block_offset<Tile>(page, DBlockBase + 1);
    const int q_t_off1 =
        Layout::kQtBase +
        source_page_block_offset<Tile>(page, DBlockBase + 1);

    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off0, src.dout_t0.f16x8, src.dout_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off0, src.q_t0.f16x8, src.q_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off1, src.dout_t2.f16x8, src.dout_t3.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off1, src.q_t2.f16x8, src.q_t3.f16x8);
}

template <bool FirstQTile, int OutBase>
__device__ __forceinline__ void dv_dk_mmac_four_out(
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    const DvDkSourceRegs4& src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
    dv_dk_mmac_one_out<FirstQTile, OutBase + 0>(
        p_frag, ds_frag, src.dout_t0, src.q_t0, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, OutBase + 1>(
        p_frag, ds_frag, src.dout_t1, src.q_t1, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, OutBase + 2>(
        p_frag, ds_frag, src.dout_t2, src.q_t2, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, OutBase + 3>(
        p_frag, ds_frag, src.dout_t3, src.q_t3, dv_acc, dk_acc, zero_f16);
}

template <typename Tile, int DBlockBase>
__device__ __forceinline__ void dv_dk_read_owner16_sources4_mq64(
    __half* lds,
    int m_block,
    DvDkSourceRegs4& src) {
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Tile::kBlockMq == 64,
                  "Mq64 source read expects a 64-row q tile");
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK source group must cover two D blocks");

    const int dout_t_off0 =
        Layout::kDoutTBase +
        source_page_block_offset_m<Tile>(0, m_block, DBlockBase);
    const int q_t_off0 =
        Layout::kQtBase +
        source_page_block_offset_m<Tile>(0, m_block, DBlockBase);
    const int dout_t_off1 =
        Layout::kDoutTBase +
        source_page_block_offset_m<Tile>(0, m_block, DBlockBase + 1);
    const int q_t_off1 =
        Layout::kQtBase +
        source_page_block_offset_m<Tile>(0, m_block, DBlockBase + 1);

    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off0, src.dout_t0.f16x8, src.dout_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off0, src.q_t0.f16x8, src.q_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off1, src.dout_t2.f16x8, src.dout_t3.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off1, src.q_t2.f16x8, src.q_t3.f16x8);
}

template <typename Tile, int DBlockBase>
__device__ __forceinline__ void dv_dk_read_owner16_sources4_mq64_semantic(
    __half* lds,
    int page,
    int m_block,
    DvDkSourceRegs4& src) {
    static_assert(Tile::kBlockMq == 64,
                  "semantic source read expects a 64-row q tile");
    static_assert(DBlockBase == 0 || DBlockBase == 2,
                  "dV/dK source group must cover two D blocks");

    const int dout_t_off0 =
        semantic_page_block_offset_m<Tile>(page, 1, m_block, DBlockBase);
    const int q_t_off0 =
        semantic_page_block_offset_m<Tile>(page, 0, m_block, DBlockBase);
    const int dout_t_off1 =
        semantic_page_block_offset_m<Tile>(page, 1, m_block, DBlockBase + 1);
    const int q_t_off1 =
        semantic_page_block_offset_m<Tile>(page, 0, m_block, DBlockBase + 1);

    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off0, src.dout_t0.f16x8, src.dout_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off0, src.q_t0.f16x8, src.q_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off1, src.dout_t2.f16x8, src.dout_t3.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off1, src.q_t2.f16x8, src.q_t3.f16x8);
}

template <bool SeedAccumulator, int OutIdx, int MFragBase>
__device__ __forceinline__ void dv_dk_mmac_one_out_mq64_part(
    const ins::Vec4F16 (&p_frag)[4],
    const ins::Vec4F16 (&ds_frag)[4],
    const ins::F16x8& dout_t,
    const ins::F16x8& q_t,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
#pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
        if constexpr (SeedAccumulator) {
            dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
                p_frag[MFragBase + m_half], dout_t.f16x4[m_half],
                m_half == 0 ? zero_f16.f32 : dv_acc[OutIdx].f32);
            dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
                ds_frag[MFragBase + m_half], q_t.f16x4[m_half],
                m_half == 0 ? zero_f16.f32 : dk_acc[OutIdx].f32);
        } else {
            dv_acc[OutIdx].f32 = ins::mmac_f16_lit(
                p_frag[MFragBase + m_half], dout_t.f16x4[m_half],
                dv_acc[OutIdx].f32);
            dk_acc[OutIdx].f32 = ins::mmac_f16_lit(
                ds_frag[MFragBase + m_half], q_t.f16x4[m_half],
                dk_acc[OutIdx].f32);
        }
    }
}

template <bool SeedAccumulator, int OutBase, int MFragBase>
__device__ __forceinline__ void dv_dk_mmac_four_out_mq64_part(
    const ins::Vec4F16 (&p_frag)[4],
    const ins::Vec4F16 (&ds_frag)[4],
    const DvDkSourceRegs4& src,
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    const ins::F16x8& zero_f16) {
    dv_dk_mmac_one_out_mq64_part<SeedAccumulator, OutBase + 0, MFragBase>(
        p_frag, ds_frag, src.dout_t0, src.q_t0, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out_mq64_part<SeedAccumulator, OutBase + 1, MFragBase>(
        p_frag, ds_frag, src.dout_t1, src.q_t1, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out_mq64_part<SeedAccumulator, OutBase + 2, MFragBase>(
        p_frag, ds_frag, src.dout_t2, src.q_t2, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out_mq64_part<SeedAccumulator, OutBase + 3, MFragBase>(
        p_frag, ds_frag, src.dout_t3, src.q_t3, dv_acc, dk_acc, zero_f16);
}

template <typename Tile, bool FirstQTile>
__device__ __forceinline__ void dv_dk_mmac_owner16_read4x2_mq64(
    __half* lds,
    const ins::Vec4F16 (&p_frag)[4],
    const ins::Vec4F16 (&ds_frag)[4],
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    ins::F16x8 zero_f16;
    if constexpr (FirstQTile) {
        ins::zero_f16x8(zero_f16);
    }

#pragma unroll
    for (int m_block = 0; m_block < 2; ++m_block) {
        DvDkSourceRegs4 low_src;
        dv_dk_read_owner16_sources4_mq64<Tile, 0>(lds, m_block, low_src);
        ins::wait_lgkm(0);
        DvDkSourceRegs4 high_src;
        dv_dk_read_owner16_sources4_mq64<Tile, 2>(lds, m_block, high_src);

        ins::raise_priority_2();
        if constexpr (FirstQTile) {
            if (m_block == 0) {
                dv_dk_mmac_four_out_mq64_part<true, 0, 0>(
                    p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
            } else {
                dv_dk_mmac_four_out_mq64_part<false, 0, 2>(
                    p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
            }
        } else {
            if (m_block == 0) {
                dv_dk_mmac_four_out_mq64_part<false, 0, 0>(
                    p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
            } else {
                dv_dk_mmac_four_out_mq64_part<false, 0, 2>(
                    p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
            }
        }
        ins::wait_lgkm(0);
        if constexpr (FirstQTile) {
            if (m_block == 0) {
                dv_dk_mmac_four_out_mq64_part<true, 4, 0>(
                    p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
            } else {
                dv_dk_mmac_four_out_mq64_part<false, 4, 2>(
                    p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
            }
        } else {
            if (m_block == 0) {
                dv_dk_mmac_four_out_mq64_part<false, 4, 0>(
                    p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
            } else {
                dv_dk_mmac_four_out_mq64_part<false, 4, 2>(
                    p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
            }
        }
        ins::lower_priority();
    }
}

template <typename Tile, bool SeedAccumulator, int MBlock>
__device__ __forceinline__ void dv_dk_mmac_owner16_read4x2_mq64_half(
    __half* lds,
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    static_assert(MBlock == 0 || MBlock == 1,
                  "Mq64 half must select one 32-row half");
    ins::F16x8 zero_f16;
    if constexpr (SeedAccumulator) {
        ins::zero_f16x8(zero_f16);
    }

    DvDkSourceRegs4 low_src;
    dv_dk_read_owner16_sources4_mq64<Tile, 0>(lds, MBlock, low_src);
    ins::wait_lgkm(0);
    DvDkSourceRegs4 high_src;
    dv_dk_read_owner16_sources4_mq64<Tile, 2>(lds, MBlock, high_src);

    ins::raise_priority_2();
    dv_dk_mmac_four_out<SeedAccumulator, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    ins::wait_lgkm(0);
    dv_dk_mmac_four_out<SeedAccumulator, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <typename Tile, bool SeedAccumulator, int MBlock>
__device__ __forceinline__ void dv_dk_mmac_owner16_read4x2_mq64_semantic_half(
    __half* lds,
    int page,
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8]) {
    static_assert(MBlock == 0 || MBlock == 1,
                  "Mq64 half must select one 32-row half");
    ins::F16x8 zero_f16;
    if constexpr (SeedAccumulator) {
        ins::zero_f16x8(zero_f16);
    }

    DvDkSourceRegs4 low_src;
    dv_dk_read_owner16_sources4_mq64_semantic<Tile, 0>(
        lds, page, MBlock, low_src);
    ins::wait_lgkm(0);
    DvDkSourceRegs4 high_src;
    dv_dk_read_owner16_sources4_mq64_semantic<Tile, 2>(
        lds, page, MBlock, high_src);

    ins::raise_priority_2();
    dv_dk_mmac_four_out<SeedAccumulator, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    ins::wait_lgkm(0);
    dv_dk_mmac_four_out<SeedAccumulator, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <typename Tile, bool FirstQTile>
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

    ins::wait_lgkm(0);
    DvDkSourceRegs4 high_src;
    dv_dk_read_owner16_sources4<Tile, 2>(lds, page, high_src);

    ins::raise_priority_2();
    dv_dk_mmac_four_out<FirstQTile, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    ins::wait_lgkm(0);
    dv_dk_mmac_four_out<FirstQTile, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <typename Tile, typename Wdra, bool FirstQTile>
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
    dv_dk_read_owner16_sources4<Tile, 2>(lds, page, high_src);

    ins::raise_priority_2();
    dv_dk_mmac_four_out<FirstQTile, 0>(
        p_frag, ds_frag, low_src, dv_acc, dk_acc, zero_f16);
    ins::wait_lgkm(0);
    arrive_raw_used_page<Wdra>(page);
    dv_dk_mmac_four_out<FirstQTile, 4>(
        p_frag, ds_frag, high_src, dv_acc, dk_acc, zero_f16);
    ins::lower_priority();
}

template <typename Tile, bool FirstQTile>
__device__ __forceinline__ void dv_dk_mmac_owner16(
    __half* lds,
    const ins::Vec4F16 (&p_frag)[2],
    const ins::Vec4F16 (&ds_frag)[2],
    ins::F32x4 (&dv_acc)[8],
    ins::F32x4 (&dk_acc)[8],
    int page) {
    using Layout = DkvLdsLayout<Tile>;

    ins::F16x8 zero_f16;
    if constexpr (FirstQTile) {
        ins::zero_f16x8(zero_f16);
    }

    ins::raise_priority_2();
    ins::F16x8 dout_t0;
    ins::F16x8 dout_t1;
    ins::F16x8 dout_t2;
    ins::F16x8 dout_t3;
    ins::F16x8 dout_t4;
    ins::F16x8 dout_t5;
    ins::F16x8 dout_t6;
    ins::F16x8 dout_t7;
    ins::F16x8 q_t0;
    ins::F16x8 q_t1;
    ins::F16x8 q_t2;
    ins::F16x8 q_t3;
    ins::F16x8 q_t4;
    ins::F16x8 q_t5;
    ins::F16x8 q_t6;
    ins::F16x8 q_t7;

    const int dout_t_off0 =
        Layout::kDoutTBase + source_page_block_offset<Tile>(page, 0);
    const int q_t_off0 =
        Layout::kQtBase + source_page_block_offset<Tile>(page, 0);
    const int dout_t_off1 =
        Layout::kDoutTBase + source_page_block_offset<Tile>(page, 1);
    const int q_t_off1 =
        Layout::kQtBase + source_page_block_offset<Tile>(page, 1);
    const int dout_t_off2 =
        Layout::kDoutTBase + source_page_block_offset<Tile>(page, 2);
    const int q_t_off2 =
        Layout::kQtBase + source_page_block_offset<Tile>(page, 2);
    const int dout_t_off3 =
        Layout::kDoutTBase + source_page_block_offset<Tile>(page, 3);
    const int q_t_off3 =
        Layout::kQtBase + source_page_block_offset<Tile>(page, 3);

    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off0, dout_t0.f16x8, dout_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off0, q_t0.f16x8, q_t1.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off1, dout_t2.f16x8, dout_t3.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off1, q_t2.f16x8, q_t3.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off2, dout_t4.f16x8, dout_t5.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off2, q_t4.f16x8, q_t5.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, dout_t_off3, dout_t6.f16x8, dout_t7.f16x8);
    ins::ds_read_matrix_trans_pair(
        lds, q_t_off3, q_t6.f16x8, q_t7.f16x8);
    ins::wait_lgkm(0);

    dv_dk_mmac_one_out<FirstQTile, 0>(
        p_frag, ds_frag, dout_t0, q_t0, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 1>(
        p_frag, ds_frag, dout_t1, q_t1, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 2>(
        p_frag, ds_frag, dout_t2, q_t2, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 3>(
        p_frag, ds_frag, dout_t3, q_t3, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 4>(
        p_frag, ds_frag, dout_t4, q_t4, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 5>(
        p_frag, ds_frag, dout_t5, q_t5, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 6>(
        p_frag, ds_frag, dout_t6, q_t6, dv_acc, dk_acc, zero_f16);
    dv_dk_mmac_one_out<FirstQTile, 7>(
        p_frag, ds_frag, dout_t7, q_t7, dv_acc, dk_acc, zero_f16);
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

__device__ __forceinline__ void softmax_ds_pair_sidecar(
    const __half* q_head,
    const __half* k_head,
    const __half* v_head,
    const __half* dout_head,
    int q_idx,
    int k_idx,
    int seqlen,
    int dim,
    int causal,
    float scale,
    float& prob_out,
    float& ds_out) {
    if (causal && k_idx > q_idx) {
        prob_out = 0.0f;
        ds_out = 0.0f;
        return;
    }

    float max_score = -3.4028234663852886e38f;
    for (int kk = 0; kk < seqlen; ++kk) {
        if (causal && kk > q_idx) {
            continue;
        }
        float score = 0.0f;
        for (int d = 0; d < dim; ++d) {
            score += __half2float(q_head[q_idx * dim + d]) *
                     __half2float(k_head[kk * dim + d]);
        }
        max_score = fmaxf(max_score, score * scale);
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
            score += __half2float(q_head[q_idx * dim + d]) *
                     __half2float(k_head[kk * dim + d]);
            dp += __half2float(dout_head[q_idx * dim + d]) *
                  __half2float(v_head[kk * dim + d]);
        }
        const float unscaled_prob = expf(score * scale - max_score);
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

template <typename Tile, typename Wdra>
__device__ __forceinline__ void consumer_score_dp_loop(
    __half* lds,
    const float* sidecar_lds,
    float* diag,
    int diag_index,
    const __half* q_head,
    const __half* k_head,
    const __half* v_head,
    const __half* dout_head,
    int seqlen,
    int dim,
    int k_base,
    int q_tiles,
    int sidecar_enabled,
    int fragment_sidecar_enabled,
    int causal,
    float softmax_scale,
    int consumer_group,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

    float diag_accum = 0.0f;
#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score[4];
        ins::F32x4 dp[4];
        score_dp_mmac_fragments<Tile>(
            lds, consumer_group, page, wave_local, score, dp);
        diag_accum += score[0].scalar[0] + dp[0].scalar[0];

        const int row_block = (consumer_group * 2 + (wave_local & 1)) & 3;
        if (fragment_sidecar_enabled && sidecar_lds != nullptr) {
            const float* sidecar_page =
                sidecar_lds + page * Tile::kSidecarFloats;
            ins::Vec4F16 p_frag[4];
            ins::Vec4F16 ds_frag[4];
            softmax_ds_fragment_from_sidecar<Tile>(
                score, dp, sidecar_page, q_tile, k_base, row_block, lane,
                seqlen, causal, softmax_scale, p_frag, ds_frag);
            if (lane == 0 && diag != nullptr && q_tile == row_block) {
                const int slot = consumer_group * 4 + wave_local;
                const int diag_base =
                    diag_index * Tile::kProbeDiagFloatsPerBlock;
                diag[diag_base + Tile::kProbeProbDiagBase + slot] =
                    static_cast<float>(p_frag[0][0]);
                diag[diag_base + Tile::kProbeDsDiagBase + slot] =
                    static_cast<float>(ds_frag[0][0]);
            }
        }

        arrive_raw_used_page<Wdra>(page);
    }

    if (lane == 0 && diag != nullptr) {
        const int slot = consumer_group * 4 + wave_local;
        const int diag_base =
            diag_index * Tile::kProbeDiagFloatsPerBlock;
        diag[diag_base + Tile::kProbeScoreDiagBase + slot] = diag_accum;
        if (sidecar_enabled && !fragment_sidecar_enabled) {
            const int k_idx = k_base + consumer_group * 64 + wave_local * 16;
            const int q_idx = k_idx < seqlen ? k_idx : seqlen - 1;
            float prob = 0.0f;
            float ds = 0.0f;
            softmax_ds_pair_sidecar(
                q_head, k_head, v_head, dout_head, q_idx, k_idx, seqlen,
                dim, causal, softmax_scale, prob, ds);
            diag[diag_base + Tile::kProbeProbDiagBase + slot] = prob;
            diag[diag_base + Tile::kProbeDsDiagBase + slot] = ds;
        }
    }
}

template <typename Tile,
          typename Wdra,
          int ConsumerGroup,
          bool EarlyReleasePage = false>
__device__ __forceinline__ void consumer_dkv_mmac_loop(
    __half* lds,
    const float* packed_sidecar,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int64_t row_base,
    int seqlen,
    int k_base,
    int q_tiles,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score[2];
        ins::F32x4 dp[2];
        score_dp_mmac_owner16<Tile>(lds, owner_nblock, page, score, dp);

        DvDkSourceRegs4 dvdk_low;
        dv_dk_read_owner16_sources4<Tile, 0>(lds, page, dvdk_low);

        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        softmax_ds_owner16_from_global_sidecar<Tile>(
            score, dp, packed_sidecar, row_base, q_tile, k_base,
            owner_nblock, lane, seqlen, causal, softmax_scale, p_frag,
            ds_frag);

        if constexpr (EarlyReleasePage) {
            if (q_tile == 0) {
                dv_dk_mmac_owner16_read4x2_early_release<Tile, Wdra, true>(
                    lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
            } else {
                dv_dk_mmac_owner16_read4x2_early_release<Tile, Wdra, false>(
                    lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
            }
        } else {
            if (q_tile == 0) {
                dv_dk_mmac_owner16_read4x2<Tile, true>(
                    lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
            } else {
                dv_dk_mmac_owner16_read4x2<Tile, false>(
                    lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
            }
            arrive_raw_used_page<Wdra>(page);
        }
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
}

template <typename Tile, typename Wdra, int ConsumerGroup>
__device__ __forceinline__ void consumer_dkv_mmac_loop_causal_skip(
    __half* lds,
    const float* packed_sidecar,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int64_t row_base,
    int seqlen,
    int k_base,
    int q_tiles,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;
    int packet_idx = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        if (causal_q_tile_before_k_tile<Tile>(q_tile, k_base, causal)) {
            continue;
        }

        const int page = packet_idx & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score[2];
        ins::F32x4 dp[2];
        score_dp_mmac_owner16<Tile>(lds, owner_nblock, page, score, dp);

        DvDkSourceRegs4 dvdk_low;
        dv_dk_read_owner16_sources4<Tile, 0>(lds, page, dvdk_low);

        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        softmax_ds_owner16_from_global_sidecar<Tile>(
            score, dp, packed_sidecar, row_base, q_tile, k_base,
            owner_nblock, lane, seqlen, causal, softmax_scale, p_frag,
            ds_frag);

        if (packet_idx == 0) {
            dv_dk_mmac_owner16_read4x2<Tile, true>(
                lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
        } else {
            dv_dk_mmac_owner16_read4x2<Tile, false>(
                lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
        }
        arrive_raw_used_page<Wdra>(page);
        ++packet_idx;
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
}

template <typename Tile, typename Wdra, int ConsumerGroup>
__device__ __forceinline__ void consumer_dkv_mmac_loop_score_dp_brick(
    __half* lds,
    const float* packed_sidecar,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int64_t row_base,
    int seqlen,
    int k_base,
    int q_tiles,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score[2];
        ins::F32x4 dp[2];
        score_dp_mmac_owner16_read2x<Tile>(
            lds, owner_nblock, page, score, dp);

        DvDkSourceRegs4 dvdk_low;
        dv_dk_read_owner16_sources4<Tile, 0>(lds, page, dvdk_low);

        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        softmax_ds_owner16_from_global_sidecar<Tile>(
            score, dp, packed_sidecar, row_base, q_tile, k_base,
            owner_nblock, lane, seqlen, causal, softmax_scale, p_frag,
            ds_frag);

        if (q_tile == 0) {
            dv_dk_mmac_owner16_read4x2<Tile, true>(
                lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
        } else {
            dv_dk_mmac_owner16_read4x2<Tile, false>(
                lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
        }
        arrive_raw_used_page<Wdra>(page);
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
}

template <typename Tile, typename Wdra, int ConsumerGroup>
__device__ __forceinline__ void consumer_dkv_mmac_loop_sidecar_overlay(
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
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score[2];
        ins::F32x4 dp[2];
        score_dp_mmac_owner16<Tile>(lds, owner_nblock, page, score, dp);
        arrive_raw_used_page<Wdra>(page);

        DvDkSourceRegs4 dvdk_low;
        dv_dk_read_owner16_sources4<Tile, 0>(lds, page, dvdk_low);

        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);
        const float* sidecar_page = raw_q_sidecar_overlay_page<Tile>(lds, page);
        ins::Vec4F16 p_frag[2];
        ins::Vec4F16 ds_frag[2];
        softmax_ds_owner16_from_sidecar<Tile>(
            score, dp, sidecar_page, q_tile, k_base, owner_nblock, lane,
            seqlen, causal, softmax_scale, p_frag, ds_frag);

        if (q_tile == 0) {
            dv_dk_mmac_owner16_read4x2<Tile, true>(
                lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
        } else {
            dv_dk_mmac_owner16_read4x2<Tile, false>(
                lds, p_frag, ds_frag, dvdk_low, dv_acc, dk_acc, page);
        }
        arrive_raw_used_page<Wdra>(page);
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
}

template <typename Tile, typename Wdra, int ConsumerGroup>
__device__ __forceinline__ void consumer_dkv_mmac_loop_mq64_single(
    __half* lds,
    const float* packed_sidecar,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int64_t row_base,
    int seqlen,
    int k_base,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    static_assert(Tile::kBlockMq == 64,
                  "consumer_dkv_mmac_loop_mq64_single expects Mq64");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < Tile::kQTilesPerCta; ++q_tile) {
        ins::abarrier_try_wait<true>(Wdra::kRaw0Filled, raw0_filled_phase);

        {
            ins::F32x4 score[2];
            ins::F32x4 dp[2];
            score_dp_mmac_owner16_mq64_half<Tile, 0>(
                lds, owner_nblock, score, dp);

            ins::Vec4F16 p_frag[2];
            ins::Vec4F16 ds_frag[2];
            softmax_ds_owner16_from_global_sidecar_mq64_half<Tile, 0>(
                score, dp, packed_sidecar, row_base, q_tile, k_base,
                owner_nblock, lane, seqlen, causal, softmax_scale, p_frag,
                ds_frag);

            if (q_tile == 0) {
                dv_dk_mmac_owner16_read4x2_mq64_half<Tile, true, 0>(
                    lds, p_frag, ds_frag, dv_acc, dk_acc);
            } else {
                dv_dk_mmac_owner16_read4x2_mq64_half<Tile, false, 0>(
                    lds, p_frag, ds_frag, dv_acc, dk_acc);
            }
        }
        {
            ins::F32x4 score[2];
            ins::F32x4 dp[2];
            score_dp_mmac_owner16_mq64_half<Tile, 1>(
                lds, owner_nblock, score, dp);

            ins::Vec4F16 p_frag[2];
            ins::Vec4F16 ds_frag[2];
            softmax_ds_owner16_from_global_sidecar_mq64_half<Tile, 1>(
                score, dp, packed_sidecar, row_base, q_tile, k_base,
                owner_nblock, lane, seqlen, causal, softmax_scale, p_frag,
                ds_frag);

            dv_dk_mmac_owner16_read4x2_mq64_half<Tile, false, 1>(
                lds, p_frag, ds_frag, dv_acc, dk_acc);
        }
        ins::abarrier_arrive_cnt<false>(Wdra::kRaw0Used, 1);
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
}

template <typename Tile, typename Wdra, int ConsumerGroup>
__device__ __forceinline__ void consumer_dkv_mmac_loop_mq64_semantic(
    __half* lds,
    const float* packed_sidecar,
    float* dk,
    float* dv,
    int64_t tensor_base,
    int64_t row_base,
    int seqlen,
    int k_base,
    int causal,
    float softmax_scale,
    int wave_local,
    int lane) {
    int resident_phase = 0;
    int raw0_filled_phase = 0;
    int raw1_filled_phase = 0;
    static_assert(ConsumerGroup == 0 || ConsumerGroup == 1,
                  "dKV consumer group must be 0 or 1");
    static_assert(Tile::kBlockMq == 64,
                  "semantic consumer expects Mq64");
    const int owner_nblock = ConsumerGroup * 4 + wave_local;
    const int q_tiles = seqlen / Tile::kBlockMq;

    ins::F32x4 dv_acc[8];
    ins::F32x4 dk_acc[8];

    ins::abarrier_try_wait<true>(Wdra::kResidentFilled, resident_phase);

#pragma clang loop unroll(disable)
    for (int q_tile = 0; q_tile < q_tiles; ++q_tile) {
        const int page = q_tile & 1;
        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);

        ins::F32x4 score0[2];
        ins::F32x4 dp0[2];
        ins::F32x4 score1[2];
        ins::F32x4 dp1[2];
        score_dp_mmac_owner16_mq64_semantic_half<Tile, 0>(
            lds, page, owner_nblock, score0, dp0);
        score_dp_mmac_owner16_mq64_semantic_half<Tile, 1>(
            lds, page, owner_nblock, score1, dp1);
        arrive_raw_used_page<Wdra>(page);

        ins::Vec4F16 p0[2];
        ins::Vec4F16 ds0[2];
        ins::Vec4F16 p1[2];
        ins::Vec4F16 ds1[2];
        softmax_ds_owner16_from_global_sidecar_mq64_half<Tile, 0>(
            score0, dp0, packed_sidecar, row_base, q_tile, k_base,
            owner_nblock, lane, seqlen, causal, softmax_scale, p0, ds0);
        softmax_ds_owner16_from_global_sidecar_mq64_half<Tile, 1>(
            score1, dp1, packed_sidecar, row_base, q_tile, k_base,
            owner_nblock, lane, seqlen, causal, softmax_scale, p1, ds1);

        wait_raw_filled_page<Wdra>(
            page, raw0_filled_phase, raw1_filled_phase);
        if (q_tile == 0) {
            dv_dk_mmac_owner16_read4x2_mq64_semantic_half<Tile, true, 0>(
                lds, page, p0, ds0, dv_acc, dk_acc);
        } else {
            dv_dk_mmac_owner16_read4x2_mq64_semantic_half<Tile, false, 0>(
                lds, page, p0, ds0, dv_acc, dk_acc);
        }
        dv_dk_mmac_owner16_read4x2_mq64_semantic_half<Tile, false, 1>(
            lds, page, p1, ds1, dv_acc, dk_acc);
        arrive_raw_used_page<Wdra>(page);
    }

    store_dkv_owner16<Tile>(
        dk, dv, dk_acc, dv_acc, lane, k_base, owner_nblock, seqlen,
        tensor_base);
}

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(16)))
fa3_bwd_dkv_probe_kernel(const __half* __restrict__ dout,
                         const __half* __restrict__ q,
                         const __half* __restrict__ k,
                         const __half* __restrict__ v,
                         const float* __restrict__ scores_max,
                         const float* __restrict__ scores_sum,
                         const float* __restrict__ delta,
                         float* __restrict__ diag,
                         int diag_stride,
                         int batch,
                         int heads,
                         int seqlen,
                         int dim,
                         int causal,
                         float softmax_scale,
                         int sidecar_enabled,
                         int fragment_sidecar_enabled) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV bringup probe LDS budget overflow");

    (void)diag_stride;
    (void)batch;
    __shared__ __half lds[Layout::kBytes / sizeof(__half)];
    float* sidecar_lds =
        reinterpret_cast<float*>(
            reinterpret_cast<char*>(lds) + Layout::kQtBase);

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

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
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int64_t tensor_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
    const int64_t row_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen;
    const __half* q_head = q + tensor_base;
    const __half* k_head = k + tensor_base;
    const __half* v_head = v + tensor_base;
    const __half* dout_head = dout + tensor_base;
    const int diag_index = blockIdx.z * gridDim.y * gridDim.x +
                           blockIdx.y * gridDim.x + blockIdx.x;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        producer_qk_loop<Tile, Bar>(
            q_head, k_head, nullptr, lds, sidecar_lds, scores_max,
            scores_sum, delta, row_base, 0, q_base, k_base, seqlen, dim, q_tiles,
            wave_local, lane, softmax_scale * kLog2E,
            fragment_sidecar_enabled, 0);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        consumer_score_dp_loop<Tile, Bar>(
            lds, sidecar_lds, diag, diag_index, q_head, k_head, v_head,
            dout_head, seqlen, dim, k_base, q_tiles, sidecar_enabled,
            fragment_sidecar_enabled, causal, softmax_scale, 0,
            wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        consumer_score_dp_loop<Tile, Bar>(
            lds, sidecar_lds, diag, diag_index, q_head, k_head, v_head,
            dout_head, seqlen, dim, k_base, q_tiles, sidecar_enabled,
            fragment_sidecar_enabled, causal, softmax_scale, 1,
            wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        producer_dout_v_loop<Tile, Bar>(
            dout_head, v_head, nullptr, lds, 0, q_base, k_base, dim,
            q_tiles, wave_local, 0);
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
    (void)scores_max;
    (void)scores_sum;
    (void)delta;
    (void)diag;
    (void)diag_stride;
    (void)batch;
    (void)heads;
    (void)seqlen;
    (void)dim;
    (void)causal;
    (void)softmax_scale;
    (void)sidecar_enabled;
    (void)fragment_sidecar_enabled;
#endif
}

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(16)))
fa3_bwd_dkv_mmac_kernel(const __half* __restrict__ dout,
                        const __half* __restrict__ q,
                        const __half* __restrict__ k,
                        const __half* __restrict__ v,
                        const __half* __restrict__ q_t_source,
                        const __half* __restrict__ dout_t_source,
                        const float* __restrict__ packed_sidecar,
                        float* __restrict__ dk,
                        float* __restrict__ dv,
                        int heads,
                        int seqlen,
                        int dim,
                        int causal,
                        float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV MMAC LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

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
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const __half* q_head = q + tensor_base;
        const __half* k_head = k + tensor_base;
        producer_qk_loop<Tile, Bar>(
            q_head, k_head, q_t_source, lds, nullptr, nullptr,
            nullptr, nullptr, 0, bh, q_base, k_base, seqlen, dim,
            q_tiles, wave_local, lane, 0.0f, 0, 1);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base,
            q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base,
            q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducerVgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const __half* v_head = v + tensor_base;
        const __half* dout_head = dout + tensor_base;
        producer_dout_v_loop<Tile, Bar>(
            dout_head, v_head, dout_t_source, lds, bh, q_base, k_base, dim,
            q_tiles, wave_local, 1);
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
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_kernel(const __half* __restrict__ dout,
                          const __half* __restrict__ q,
                          const __half* __restrict__ k,
                          const __half* __restrict__ v,
                          const __half* __restrict__ q_t_source,
                          const __half* __restrict__ dout_t_source,
                          const float* __restrict__ packed_sidecar,
                          float* __restrict__ dk,
                          float* __restrict__ dv,
                          int heads,
                          int seqlen,
                          int dim,
                          int causal,
                          float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV MMAC12 LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        producer_all_loop<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            q_base, k_base, seqlen, dim, q_tiles, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_early_release_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q_t_source,
    const __half* __restrict__ dout_t_source,
    const float* __restrict__ packed_sidecar,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV MMAC12 early release LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        producer_all_loop<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            q_base, k_base, seqlen, dim, q_tiles, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 0, true>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 1, true>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_score_dp_brick_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q_t_source,
    const __half* __restrict__ dout_t_source,
    const float* __restrict__ packed_sidecar,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV score/dP brick LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        producer_all_loop<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            q_base, k_base, seqlen, dim, q_tiles, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop_score_dp_brick<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop_score_dp_brick<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_mixed_score_brick_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q_t_source,
    const __half* __restrict__ dout_t_source,
    const float* __restrict__ packed_sidecar,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV mixed score brick LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        producer_all_loop<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            q_base, k_base, seqlen, dim, q_tiles, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop_score_dp_brick<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_causal_skip_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q_t_source,
    const __half* __restrict__ dout_t_source,
    const float* __restrict__ packed_sidecar,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV causal-skip LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        producer_all_loop_causal_skip<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            q_base, k_base, seqlen, dim, q_tiles, wave_local, causal);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerMq64Vgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop_causal_skip<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerMq64Vgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * seqlen;
        consumer_dkv_mmac_loop_causal_skip<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            seqlen, k_base, q_tiles, causal, softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_sidecar_overlay_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q_t_source,
    const __half* __restrict__ dout_t_source,
    const float* __restrict__ scores_max,
    const float* __restrict__ scores_sum,
    const float* __restrict__ delta,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq32Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes <= Tile::kLdsBudgetBytes,
                  "dKV overlay LDS budget overflow");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q_base = 0;
    const int k_base = k_tile * Tile::kResidentNk;
    const int q_tiles = seqlen / Tile::kBlockMq;
    const int bh = b * heads + h;
    const int64_t tensor_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        producer_all_loop_sidecar_overlay<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, scores_max,
            scores_sum, delta, lds,
            (static_cast<int64_t>(b) * heads + h) * seqlen, bh, q_base,
            k_base, seqlen, dim, q_tiles, wave_local, lane,
            softmax_scale * kLog2E);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        consumer_dkv_mmac_loop_sidecar_overlay<Tile, Bar, 0>(
            lds, dk, dv, tensor_base, seqlen, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerVgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        consumer_dkv_mmac_loop_sidecar_overlay<Tile, Bar, 1>(
            lds, dk, dv, tensor_base, seqlen, k_base, q_tiles, causal,
            softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
    (void)scores_max;
    (void)scores_sum;
    (void)delta;
    (void)dk;
    (void)dv;
    (void)heads;
    (void)seqlen;
    (void)dim;
    (void)causal;
    (void)softmax_scale;
#endif
}

__global__ void __launch_bounds__(dkv::DkvTileD128Mq64Nk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_mq64_kernel(const __half* __restrict__ dout,
                               const __half* __restrict__ q,
                               const __half* __restrict__ k,
                               const __half* __restrict__ v,
                               const __half* __restrict__ q_t_source,
                               const __half* __restrict__ dout_t_source,
                               const float* __restrict__ packed_sidecar,
                               float* __restrict__ dk,
                               float* __restrict__ dv,
                               int heads,
                               int seqlen,
                               int dim,
                               int causal,
                               float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq64Nk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvLdsLayout<Tile>;
    static_assert(Layout::kBytes == Tile::kLdsBudgetBytes,
                  "Mq64 single-buffer LDS plan should fill 128KB");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    constexpr int kSeqLen = Tile::kProbeSeqLen;
    constexpr int kCausal = 1;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        const int h = blockIdx.y;
        const int b = blockIdx.z;
        const int bh = b * heads + h;
        const int k_base = blockIdx.x * Tile::kResidentNk;
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * kSeqLen * dim;
        producer_all_loop_mq64_single<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            k_base, dim, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerMq64Vgprs);
        const int h = blockIdx.y;
        const int b = blockIdx.z;
        const int k_base = blockIdx.x * Tile::kResidentNk;
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * kSeqLen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * kSeqLen;
        consumer_dkv_mmac_loop_mq64_single<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            kSeqLen, k_base, kCausal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerMq64Vgprs);
        const int h = blockIdx.y;
        const int b = blockIdx.z;
        const int k_base = blockIdx.x * Tile::kResidentNk;
        const int lane = static_cast<int>(threadIdx.x % 64);
        const int64_t tensor_base =
            (static_cast<int64_t>(b) * heads + h) * kSeqLen * dim;
        const int64_t row_base =
            (static_cast<int64_t>(b) * heads + h) * kSeqLen;
        consumer_dkv_mmac_loop_mq64_single<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base,
            kSeqLen, k_base, kCausal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __syncthreads();
    __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
    __builtin_hcu_s_abarrier_inv(Bar::kRaw0Filled);
    __builtin_hcu_s_abarrier_inv(Bar::kRaw0Used);
    __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    __syncthreads();
#else
    (void)dout;
    (void)q;
    (void)k;
    (void)v;
    (void)q_t_source;
    (void)dout_t_source;
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

__global__ void __launch_bounds__(
    dkv::DkvTileD128Mq64SemanticNk128W12::kThreadsPerCta, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
fa3_bwd_dkv_mmac12_mq64_semantic_kernel(
    const __half* __restrict__ dout,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q_t_source,
    const __half* __restrict__ dout_t_source,
    const float* __restrict__ packed_sidecar,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int heads,
    int seqlen,
    int dim,
    int causal,
    float softmax_scale) {
#if defined(__gfx946__) || defined(__gfx92a__)
    using Tile = dkv::DkvTileD128Mq64SemanticNk128W12;
    using Bar = dkv::DkvBarrierLedger;
    using Vgpr = dkv::WdraResourceWindows;
    using Layout = DkvSemanticMq64LdsLayout<Tile>;
    static_assert(Layout::kBytes == Tile::kLdsBudgetBytes,
                  "Mq64 semantic-page LDS plan should fill 128KB");

    __shared__ __half lds[Layout::kBytes / sizeof(__half)];

    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRaw1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int k_tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int bh = b * heads + h;
    const int k_base = k_tile * Tile::kResidentNk;
    const int64_t tensor_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen * dim;
    const int64_t row_base =
        (static_cast<int64_t>(b) * heads + h) * seqlen;

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kProducer12Vgprs);
        producer_all_loop_mq64_semantic<Tile, Bar>(
            q + tensor_base, k + tensor_base, v + tensor_base,
            dout + tensor_base, q_t_source, dout_t_source, lds, bh,
            k_base, seqlen, dim, wave_local);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerMq64Vgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        consumer_dkv_mmac_loop_mq64_semantic<Tile, Bar, 0>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base, seqlen,
            k_base, causal, softmax_scale, wave_local, lane);
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(Vgpr::kConsumerMq64Vgprs);
        const int lane = static_cast<int>(threadIdx.x % 64);
        consumer_dkv_mmac_loop_mq64_semantic<Tile, Bar, 1>(
            lds, packed_sidecar, dk, dv, tensor_base, row_base, seqlen,
            k_base, causal, softmax_scale, wave_local, lane);
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
    (void)q_t_source;
    (void)dout_t_source;
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
    if (params->dkv_path == dkv::kDkvPathWaspFragmentSidecar &&
        (softmax_aux0 == nullptr || softmax_aux1 == nullptr ||
         params->reserved_ptr[0] == nullptr)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if ((params->dkv_path == dkv::kDkvPathWaspDkvMmac ||
         params->dkv_path == dkv::kDkvPathWaspDkvMmac12Wave ||
         params->dkv_path == dkv::kDkvPathWaspDkvMmac12WaveMq64 ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveSidecarOverlay ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveScoreDpBrick ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveMq64Semantic ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveCausalSkip ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveMixedScoreBrick ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveEarlyRelease) &&
        (dk == nullptr || dv == nullptr ||
         params->reserved_ptr[1] == nullptr ||
         params->reserved_ptr[2] == nullptr)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if ((params->dkv_path == dkv::kDkvPathWaspDkvMmac ||
         params->dkv_path == dkv::kDkvPathWaspDkvMmac12Wave ||
         params->dkv_path == dkv::kDkvPathWaspDkvMmac12WaveMq64 ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveScoreDpBrick ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveMq64Semantic ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveCausalSkip ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveMixedScoreBrick ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveEarlyRelease) &&
        params->reserved_ptr[3] == nullptr) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->dkv_path ==
            dkv::kDkvPathWaspDkvMmac12WaveSidecarOverlay &&
        (softmax_aux0 == nullptr || softmax_aux1 == nullptr ||
         params->reserved_ptr[0] == nullptr)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }
    if (params->workspace_bytes < shaobo_fa3_bwd_workspace_bytes(params)) {
        return SHAOBO_FA3_STATUS_INVALID_VALUE;
    }

    const int grid_x =
        ceil_div(params->seqlen_k, dkv::DkvTileD128Mq32Nk128::kResidentNk);
    dim3 grid(grid_x, params->num_heads_q, params->batch);
    dim3 block(
        (params->dkv_path == dkv::kDkvPathWaspDkvMmac12Wave ||
         params->dkv_path == dkv::kDkvPathWaspDkvMmac12WaveMq64 ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveSidecarOverlay ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveScoreDpBrick ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveMq64Semantic ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveCausalSkip ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveMixedScoreBrick ||
         params->dkv_path ==
             dkv::kDkvPathWaspDkvMmac12WaveEarlyRelease)
            ? dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta
            : dkv::DkvTileD128Mq32Nk128::kThreadsPerCta);

    if (params->dkv_path == dkv::kDkvPathWaspDkvMmac12WaveMq64Semantic) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_mq64_semantic_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (
        params->dkv_path ==
        dkv::kDkvPathWaspDkvMmac12WaveCausalSkip) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_causal_skip_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (
        params->dkv_path ==
        dkv::kDkvPathWaspDkvMmac12WaveMixedScoreBrick) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_mixed_score_brick_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (
        params->dkv_path ==
        dkv::kDkvPathWaspDkvMmac12WaveEarlyRelease) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_early_release_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (params->dkv_path == dkv::kDkvPathWaspDkvMmac12WaveMq64) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_mq64_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (
        params->dkv_path ==
        dkv::kDkvPathWaspDkvMmac12WaveSidecarOverlay) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_sidecar_overlay_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(softmax_aux0),
            static_cast<const float*>(softmax_aux1),
            static_cast<const float*>(params->reserved_ptr[0]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (params->dkv_path == dkv::kDkvPathWaspDkvMmac12Wave) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (
        params->dkv_path ==
        dkv::kDkvPathWaspDkvMmac12WaveScoreDpBrick) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac12_score_dp_brick_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else if (params->dkv_path == dkv::kDkvPathWaspDkvMmac) {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_mmac_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const __half*>(params->reserved_ptr[1]),
            static_cast<const __half*>(params->reserved_ptr[2]),
            static_cast<const float*>(params->reserved_ptr[3]),
            static_cast<float*>(dk), static_cast<float*>(dv),
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale);
    } else {
        hipLaunchKernelGGL(
            fa3_bwd_dkv_probe_kernel, grid, block, 0, 0,
            static_cast<const __half*>(dout), static_cast<const __half*>(q),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            static_cast<const float*>(softmax_aux0),
            static_cast<const float*>(softmax_aux1),
            static_cast<const float*>(params->reserved_ptr[0]),
            static_cast<float*>(params->workspace), grid_x, params->batch,
            params->num_heads_q, params->seqlen_k, params->head_dim_qk,
            params->causal, params->softmax_scale,
            (params->dkv_path == dkv::kDkvPathWaspSoftmaxDsSidecar ||
             params->dkv_path == dkv::kDkvPathWaspFragmentSidecar)
                ? 1
                : 0,
            params->dkv_path == dkv::kDkvPathWaspFragmentSidecar ? 1 : 0);
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

std::vector<__half> make_mq32_source_layout_host(
    const std::vector<__half>& src,
    int batch,
    int heads,
    int seqlen,
    int dim) {
    constexpr int kMq = dkv::DkvTileD128Mq32Nk128::kBlockMq;
    const int q_tiles = seqlen / kMq;
    std::vector<__half> source(
        static_cast<size_t>(batch) * static_cast<size_t>(heads) *
            static_cast<size_t>(q_tiles) * static_cast<size_t>(dim) * kMq,
        __float2half(0.0f));
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            const size_t bh_tile_base =
                (static_cast<size_t>(b) * heads + h) *
                static_cast<size_t>(q_tiles) * dim * kMq;
            for (int qt = 0; qt < q_tiles; ++qt) {
                const int q_base = qt * kMq;
                const size_t tile_base =
                    bh_tile_base + static_cast<size_t>(qt) * dim * kMq;
                for (int d = 0; d < dim; ++d) {
                    for (int mq = 0; mq < kMq; ++mq) {
                        const int qrow = q_base + mq;
                        source[tile_base + static_cast<size_t>(d) * kMq +
                               mq] =
                            src[tensor_offset(
                                b, h, qrow, d, heads, seqlen, dim)];
                    }
                }
            }
        }
    }
    return source;
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
    const int probe_check = arg_int(
        argc, argv, "--probe-check", env_int("CHECK_WASP_SIDECAR", 0));
    const int fragment_check = arg_int(
        argc, argv, "--fragment-check", env_int("CHECK_WASP_FRAGMENT", 0));
    const int mmac_check = arg_int(
        argc, argv, "--dkv-mmac-check", env_int("CHECK_WASP_DKV_MMAC", 0));
    const int mmac12_check = arg_int(
        argc, argv, "--dkv-mmac12-check",
        env_int("CHECK_WASP_DKV_MMAC12", 0));
    const int mmac12_overlay_check = arg_int(
        argc, argv, "--dkv-mmac12-overlay-check",
        env_int("CHECK_WASP_DKV_MMAC12_OVERLAY", 0));
    const int mmac12_score_brick_check = arg_int(
        argc, argv, "--dkv-mmac12-score-brick-check",
        env_int("CHECK_WASP_DKV_MMAC12_SCORE_BRICK", 0));
    const int mmac12_mq64_check = arg_int(
        argc, argv, "--dkv-mmac12-mq64-check",
        env_int("CHECK_WASP_DKV_MMAC12_MQ64", 0));
    const int mmac12_mq64_semantic_check = arg_int(
        argc, argv, "--dkv-mmac12-mq64-semantic-check",
        env_int("CHECK_WASP_DKV_MMAC12_MQ64_SEMANTIC", 0));
    const int mmac12_causal_skip_check = arg_int(
        argc, argv, "--dkv-mmac12-causal-skip-check",
        env_int("CHECK_WASP_DKV_MMAC12_CAUSAL_SKIP", 0));
    const int mmac12_mixed_score_brick_check = arg_int(
        argc, argv, "--dkv-mmac12-mixed-score-brick-check",
        env_int("CHECK_WASP_DKV_MMAC12_MIXED_SCORE_BRICK", 0));
    const int mmac12_early_release_check = arg_int(
        argc, argv, "--dkv-mmac12-early-release-check",
        env_int("CHECK_WASP_DKV_MMAC12_EARLY_RELEASE", 0));
    const int full_mmac_check =
        mmac_check || mmac12_check || mmac12_overlay_check ||
        mmac12_score_brick_check || mmac12_mq64_check ||
        mmac12_mq64_semantic_check || mmac12_causal_skip_check ||
        mmac12_mixed_score_brick_check || mmac12_early_release_check;
    const int batch = arg_int(argc, argv, "--B", env_int("B", kDefaultBatch));
    const int heads = arg_int(argc, argv, "--H", env_int("H", kDefaultHeads));
    const int default_seq =
        (check || probe_check || fragment_check || full_mmac_check)
            ? 128
            : kDefaultSeq;
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
    params.dkv_path = dkv::kDkvPathWaspProbe;
    if (probe_check) {
        params.dkv_path = dkv::kDkvPathWaspSoftmaxDsSidecar;
    }
    if (fragment_check) {
        params.dkv_path = dkv::kDkvPathWaspFragmentSidecar;
    }
    if (mmac_check) {
        params.dkv_path = dkv::kDkvPathWaspDkvMmac;
    }
    if (mmac12_check) {
        params.dkv_path = dkv::kDkvPathWaspDkvMmac12Wave;
    }
    if (mmac12_overlay_check) {
        params.dkv_path =
            dkv::kDkvPathWaspDkvMmac12WaveSidecarOverlay;
    }
    if (mmac12_score_brick_check) {
        params.dkv_path =
            dkv::kDkvPathWaspDkvMmac12WaveScoreDpBrick;
    }
    if (mmac12_mq64_check) {
        params.dkv_path = dkv::kDkvPathWaspDkvMmac12WaveMq64;
    }
    if (mmac12_mq64_semantic_check) {
        params.dkv_path =
            dkv::kDkvPathWaspDkvMmac12WaveMq64Semantic;
    }
    if (mmac12_causal_skip_check) {
        params.dkv_path =
            dkv::kDkvPathWaspDkvMmac12WaveCausalSkip;
    }
    if (mmac12_mixed_score_brick_check) {
        params.dkv_path =
            dkv::kDkvPathWaspDkvMmac12WaveMixedScoreBrick;
    }
    if (mmac12_early_release_check) {
        params.dkv_path =
            dkv::kDkvPathWaspDkvMmac12WaveEarlyRelease;
    }
    if (check) {
        params.dkv_path = dkv::kDkvPathReferenceCorrectness;
    }
    params.block_threads =
        (mmac12_check || mmac12_overlay_check ||
         mmac12_score_brick_check || mmac12_mq64_check ||
         mmac12_mq64_semantic_check || mmac12_causal_skip_check ||
         mmac12_mixed_score_brick_check || mmac12_early_release_check)
            ? dkv::DkvTileD128Mq32Nk128W12::kThreadsPerCta
            : dkv::DkvTileD128Mq32Nk128::kThreadsPerCta;
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
    __half* q_t_dev = nullptr;
    __half* dout_t_dev = nullptr;
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
    if ((check || full_mmac_check) && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dk_dev), output_bytes);
    }
    if ((check || full_mmac_check) && err == hipSuccess) {
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
    if ((fragment_check || full_mmac_check) && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&scores_max_dev), sidecar_bytes);
    }
    if ((fragment_check || full_mmac_check) && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&scores_sum_dev), sidecar_bytes);
    }
    if ((fragment_check || full_mmac_check) && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&delta_dev), sidecar_bytes);
    }
    const size_t packed_sidecar_bytes =
        sidecar_rows * dkv::DkvTileD128Mq32Nk128::kPackedSidecarFields *
        sizeof(float);
    if (full_mmac_check && err == hipSuccess) {
        err = hipMalloc(
            reinterpret_cast<void**>(&packed_sidecar_dev),
            packed_sidecar_bytes);
    }
    const int source_q_tiles =
        seqlen / dkv::DkvTileD128Mq32Nk128::kBlockMq;
    const size_t source_elems =
        static_cast<size_t>(batch) * static_cast<size_t>(heads) *
        static_cast<size_t>(source_q_tiles) * static_cast<size_t>(dim) *
        dkv::DkvTileD128Mq32Nk128::kBlockMq;
    const size_t source_bytes = source_elems * sizeof(__half);
    if (full_mmac_check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&q_t_dev), source_bytes);
    }
    if (full_mmac_check && err == hipSuccess) {
        err = hipMalloc(reinterpret_cast<void**>(&dout_t_dev), source_bytes);
    }
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        return 2;
    }

    std::vector<__half> q_host(tensor_elems);
    std::vector<__half> k_host(tensor_elems);
    std::vector<__half> v_host(tensor_elems);
    std::vector<__half> dout_host(tensor_elems);
    if (check || probe_check || fragment_check || full_mmac_check) {
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
        if (check || full_mmac_check) {
            ignore_hip_status(hipMemset(dk_dev, 0, output_bytes));
            ignore_hip_status(hipMemset(dv_dev, 0, output_bytes));
        }
        if (fragment_check || full_mmac_check) {
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
            if (full_mmac_check) {
                std::vector<float> packed_sidecar_host(
                    sidecar_rows *
                    dkv::DkvTileD128Mq32Nk128::kPackedSidecarFields);
                for (size_t row = 0; row < sidecar_rows; ++row) {
                    const size_t base =
                        row *
                        dkv::DkvTileD128Mq32Nk128::kPackedSidecarFields;
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
        if (full_mmac_check) {
            std::vector<__half> q_t_host =
                make_mq32_source_layout_host(
                    q_host, batch, heads, seqlen, dim);
            std::vector<__half> dout_t_host =
                make_mq32_source_layout_host(
                    dout_host, batch, heads, seqlen, dim);
            ignore_hip_status(
                hipMemcpy(q_t_dev, q_t_host.data(), source_bytes,
                          hipMemcpyHostToDevice));
            ignore_hip_status(
                hipMemcpy(dout_t_dev, dout_t_host.data(), source_bytes,
                          hipMemcpyHostToDevice));
            params.reserved_ptr[1] = q_t_dev;
            params.reserved_ptr[2] = dout_t_dev;
        }
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
        shaobo_fa3_bwd(dout_dev, q_dev, k_dev, v_dev, nullptr,
                       scores_max_dev, scores_sum_dev, nullptr, dk_dev,
                       dv_dev, &params);
    if (check || full_mmac_check) {
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
        const float rel_l2_limit = full_mmac_check ? 5.0e-3f : 1.0e-4f;
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
        ignore_hip_status(hipFree(dout_t_dev));
        ignore_hip_status(hipFree(q_t_dev));
        ignore_hip_status(hipFree(workspace));
        ignore_hip_status(hipFree(dv_dev));
        ignore_hip_status(hipFree(dk_dev));
        ignore_hip_status(hipFree(dout_dev));
        ignore_hip_status(hipFree(v_dev));
        ignore_hip_status(hipFree(k_dev));
        ignore_hip_status(hipFree(q_dev));

        const char* full_label = "fa3_bwd_dkv_mmac_correctness";
        if (mmac12_mq64_semantic_check) {
            full_label = "fa3_bwd_dkv_mmac12_mq64_semantic_correctness";
        } else if (mmac12_causal_skip_check) {
            full_label = "fa3_bwd_dkv_mmac12_causal_skip_correctness";
        } else if (mmac12_mixed_score_brick_check) {
            full_label = "fa3_bwd_dkv_mmac12_mixed_score_brick_correctness";
        } else if (mmac12_early_release_check) {
            full_label = "fa3_bwd_dkv_mmac12_early_release_correctness";
        } else if (mmac12_mq64_check) {
            full_label = "fa3_bwd_dkv_mmac12_mq64_correctness";
        } else if (mmac12_overlay_check) {
            full_label = "fa3_bwd_dkv_mmac12_overlay_correctness";
        } else if (mmac12_score_brick_check) {
            full_label = "fa3_bwd_dkv_mmac12_score_brick_correctness";
        } else if (mmac12_check) {
            full_label = "fa3_bwd_dkv_mmac12_correctness";
        }

        std::printf(
            "%s status=%s B=%d H=%d S=%d D=%d causal=%d "
            "workspace_bytes=%zu dk_max_abs=%g dk_mean_abs=%g dk_rmse=%g "
            "dk_rel_l2=%g dv_max_abs=%g dv_mean_abs=%g dv_rmse=%g "
            "dv_rel_l2=%g bad=%d pass=%d\n",
            full_mmac_check ? full_label : "fa3_bwd_dkv_correctness",
            shaobo_fa3_status_string(status), batch, heads, seqlen, dim,
            params.causal, workspace_bytes, dk_metrics.max_abs, dk_metrics.mean_abs,
            dk_metrics.rmse, dk_metrics.rel_l2, dv_metrics.max_abs,
            dv_metrics.mean_abs, dv_metrics.rmse, dv_metrics.rel_l2,
            dk_metrics.bad_count + dv_metrics.bad_count, pass ? 1 : 0);
        return pass ? 0 : 1;
    }

    if (probe_check || fragment_check) {
        const size_t workspace_floats = workspace_bytes / sizeof(float);
        std::vector<float> workspace_host(workspace_floats, 0.0f);
        ignore_hip_status(
            hipMemcpy(workspace_host.data(), workspace, workspace_bytes,
                      hipMemcpyDeviceToHost));

        const int grid_x =
            ceil_div(seqlen, dkv::DkvTileD128Mq32Nk128::kResidentNk);
        std::vector<float> prob_actual;
        std::vector<float> prob_expected;
        std::vector<float> ds_actual;
        std::vector<float> ds_expected;
        prob_actual.reserve(static_cast<size_t>(batch) * heads * grid_x * 8);
        prob_expected.reserve(prob_actual.capacity());
        ds_actual.reserve(prob_actual.capacity());
        ds_expected.reserve(prob_actual.capacity());

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < heads; ++h) {
                for (int k_tile = 0; k_tile < grid_x; ++k_tile) {
                    const int diag_index =
                        b * heads * grid_x + h * grid_x + k_tile;
                    const int diag_base =
                        diag_index *
                        dkv::DkvTileD128Mq32Nk128::kProbeDiagFloatsPerBlock;
                    const int k_base =
                        k_tile * dkv::DkvTileD128Mq32Nk128::kResidentNk;
                    for (int consumer_group = 0; consumer_group < 2;
                         ++consumer_group) {
                        for (int wave_local = 0; wave_local < 4;
                             ++wave_local) {
                            const int slot = consumer_group * 4 + wave_local;
                            const int row_block =
                                fragment_check
                                    ? ((consumer_group * 2 +
                                        (wave_local & 1)) &
                                       3)
                                    : 0;
                            const int k_idx =
                                fragment_check
                                    ? k_base + row_block * 32
                                    : k_base + consumer_group * 64 +
                                          wave_local * 16;
                            const int q_idx =
                                k_idx < seqlen ? k_idx : seqlen - 1;
                            float p_ref = 0.0f;
                            float ds_ref = 0.0f;
                            cpu_pair_prob_ds(
                                q_host, k_host, v_host, dout_host, batch,
                                heads, seqlen, dim, params.causal,
                                params.softmax_scale, b, h, q_idx, k_idx,
                                p_ref, ds_ref);
                            prob_actual.push_back(
                                workspace_host[
                                    diag_base +
                                    dkv::DkvTileD128Mq32Nk128::
                                        kProbeProbDiagBase +
                                    slot]);
                            prob_expected.push_back(p_ref);
                            ds_actual.push_back(
                                workspace_host[
                                    diag_base +
                                    dkv::DkvTileD128Mq32Nk128::
                                        kProbeDsDiagBase +
                                    slot]);
                            ds_expected.push_back(ds_ref);
                        }
                    }
                }
            }
        }

        const DkvCompareMetrics p_metrics =
            compare_vectors(prob_actual, prob_expected);
        const DkvCompareMetrics ds_metrics =
            compare_vectors(ds_actual, ds_expected);
        const float ds_rel_l2_limit = fragment_check ? 5.0e-4f : 1.0e-5f;
        const bool pass =
            status == SHAOBO_FA3_STATUS_SUCCESS &&
            p_metrics.bad_count == 0 && ds_metrics.bad_count == 0 &&
            p_metrics.max_abs <= 1.0e-5f &&
            ds_metrics.max_abs <= 1.0e-5f &&
            p_metrics.rel_l2 <= 1.0e-5f &&
            ds_metrics.rel_l2 <= ds_rel_l2_limit;

        ignore_hip_status(hipFree(delta_dev));
        ignore_hip_status(hipFree(scores_sum_dev));
        ignore_hip_status(hipFree(scores_max_dev));
        ignore_hip_status(hipFree(packed_sidecar_dev));
        ignore_hip_status(hipFree(dout_t_dev));
        ignore_hip_status(hipFree(q_t_dev));
        ignore_hip_status(hipFree(workspace));
        ignore_hip_status(hipFree(dv_dev));
        ignore_hip_status(hipFree(dk_dev));
        ignore_hip_status(hipFree(dout_dev));
        ignore_hip_status(hipFree(v_dev));
        ignore_hip_status(hipFree(k_dev));
        ignore_hip_status(hipFree(q_dev));

        std::printf(
            "%s status=%s B=%d H=%d S=%d D=%d "
            "workspace_bytes=%zu p_max_abs=%g p_mean_abs=%g p_rmse=%g "
            "p_rel_l2=%g ds_max_abs=%g ds_mean_abs=%g ds_rmse=%g "
            "ds_rel_l2=%g bad=%d pass=%d\n",
            fragment_check ? "fa3_bwd_dkv_fragment_sidecar"
                           : "fa3_bwd_dkv_sidecar",
            shaobo_fa3_status_string(status), batch, heads, seqlen, dim,
            workspace_bytes, p_metrics.max_abs, p_metrics.mean_abs,
            p_metrics.rmse, p_metrics.rel_l2, ds_metrics.max_abs,
            ds_metrics.mean_abs, ds_metrics.rmse, ds_metrics.rel_l2,
            p_metrics.bad_count + ds_metrics.bad_count, pass ? 1 : 0);
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

    ignore_hip_status(hipFree(delta_dev));
    ignore_hip_status(hipFree(scores_sum_dev));
    ignore_hip_status(hipFree(scores_max_dev));
    ignore_hip_status(hipFree(packed_sidecar_dev));
    ignore_hip_status(hipFree(dout_t_dev));
    ignore_hip_status(hipFree(q_t_dev));
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
