#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Vec4U32 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using Vec4F16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using Vec4F32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

union F16x8 {
    Vec8F16 f16x8;
    Vec4F16 f16x4[2];
    uint16_t u16[8];
};

union F32x4 {
    Vec4F32 f32;
    float scalar[4];
};

constexpr int kNk = 64;
constexpr int kDim = 128;
constexpr int kDTiles = 4;
constexpr int kNTiles = 2;
constexpr int kFragPerD = 4;
constexpr int kReaders = 5;
constexpr int kLoadKinds = 6;
constexpr int kDfmtKinds = 2;
constexpr int kLanes = 64;
constexpr int kVec = 8;
constexpr int kAcc = 4;
constexpr int kMatrixBlockBytes = 32 * 32 * 2;
constexpr int kKtDtileBytes = 32 * kNk * 2;
constexpr uint16_t kPayloadBase = 0x3c00;

__host__ __device__ __forceinline__ int frag_index(int d_tile,
                                                   int frag,
                                                   int lane,
                                                   int vec) {
    int idx = d_tile;
    idx = idx * kFragPerD + frag;
    idx = idx * kLanes + lane;
    idx = idx * kVec + vec;
    return idx;
}

__host__ __device__ __forceinline__ int cand_frag_index(int reader,
                                                        int d_tile,
                                                        int frag,
                                                        int lane,
                                                        int vec) {
    int idx = reader;
    idx = idx * kDTiles + d_tile;
    idx = idx * kFragPerD + frag;
    idx = idx * kLanes + lane;
    idx = idx * kVec + vec;
    return idx;
}

__host__ __device__ __forceinline__ int acc_index(int d_tile,
                                                  int frag,
                                                  int lane,
                                                  int elem) {
    int idx = d_tile;
    idx = idx * kFragPerD + frag;
    idx = idx * kLanes + lane;
    idx = idx * kAcc + elem;
    return idx;
}

__host__ __device__ __forceinline__ int cand_acc_index(int reader,
                                                       int d_tile,
                                                       int frag,
                                                       int lane,
                                                       int elem) {
    int idx = reader;
    idx = idx * kDTiles + d_tile;
    idx = idx * kFragPerD + frag;
    idx = idx * kLanes + lane;
    idx = idx * kAcc + elem;
    return idx;
}

__host__ __device__ __forceinline__ uint16_t pattern_value(int krow, int d) {
    return static_cast<uint16_t>(kPayloadBase + krow * kDim + d);
}

const char* load_kind_name(int kind) {
    switch (kind) {
        case 0:
            return "raw32x32_t_current";
        case 1:
            return "raw32x32_n";
        case 2:
            return "raw32x16_pair_n";
        case 3:
            return "raw32x16_pair_t";
        case 4:
            return "raw32x16_pair_r";
        case 5:
            return "raw32x16_pair_tr";
        default:
            return "unknown";
    }
}

const char* reader_name(int reader) {
    switch (reader) {
        case 0:
            return "ds32x32_trans_r2c1_alt0_ref_shape";
        case 1:
            return "ds32x32_normal_r2c1_alt0";
        case 2:
            return "ds32x32_trans_m16x32_alt0";
        case 3:
            return "ds32x32_trans_m16x32_alt1";
        case 4:
            return "ds32x32_normal_r2c1_alt1";
        default:
            return "unknown";
    }
}

__device__ __forceinline__ Vec4U32 prepare_matrix_src(const uint16_t* ptr,
                                                       int row_stride) {
    Vec4U32 srsrc{};
    *reinterpret_cast<unsigned long long*>(&srsrc) =
        reinterpret_cast<unsigned long long>(ptr) & 0xffffffffffffULL;
    srsrc[2] = static_cast<uint32_t>(row_stride);
    srsrc[3] = 0;
    return srsrc;
}

template <bool Transpose>
__device__ __forceinline__ void matrix_load_32x32_b16_bps_lds(
    uint16_t* lds,
    Vec4U32 srsrc,
    int lds_offset,
    int lds_dfmt) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr_v =
        (static_cast<uint32_t>(reinterpret_cast<size_t>(lds)) +
         static_cast<uint32_t>(lds_offset)) |
        (lds_dfmt ? 0x80000000u : 0u);
    const int lds_addr =
        __builtin_amdgcn_readfirstlane(static_cast<int>(lds_addr_v));
    if constexpr (Transpose) {
        asm volatile(
            "s_nop 0\n\t"
            "s_mov_b32 m0, %1\n\t"
            "matrix_load_32x32_b16 %0, m0 moffset:%2 t bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    } else {
        asm volatile(
            "s_nop 0\n\t"
            "s_mov_b32 m0, %1\n\t"
            "matrix_load_32x32_b16 %0, m0 moffset:%2 bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    }
#else
    (void)lds;
    (void)srsrc;
    (void)lds_offset;
    (void)lds_dfmt;
#endif
}

template <bool Transpose, bool Reverse>
__device__ __forceinline__ void matrix_load_32x16_b16_bps_lds(
    uint16_t* lds,
    Vec4U32 srsrc,
    int lds_offset,
    int lds_dfmt) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr_v =
        (static_cast<uint32_t>(reinterpret_cast<size_t>(lds)) +
         static_cast<uint32_t>(lds_offset)) |
        (lds_dfmt ? 0x80000000u : 0u);
    const int lds_addr =
        __builtin_amdgcn_readfirstlane(static_cast<int>(lds_addr_v));
    if constexpr (Transpose && Reverse) {
        asm volatile(
            "s_nop 0\n\t"
            "s_mov_b32 m0, %1\n\t"
            "matrix_load_32x16_b16 %0, m0 moffset:%2 t r bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    } else if constexpr (Transpose) {
        asm volatile(
            "s_nop 0\n\t"
            "s_mov_b32 m0, %1\n\t"
            "matrix_load_32x16_b16 %0, m0 moffset:%2 t bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    } else if constexpr (Reverse) {
        asm volatile(
            "s_nop 0\n\t"
            "s_mov_b32 m0, %1\n\t"
            "matrix_load_32x16_b16 %0, m0 moffset:%2 r bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    } else {
        asm volatile(
            "s_nop 0\n\t"
            "s_mov_b32 m0, %1\n\t"
            "matrix_load_32x16_b16 %0, m0 moffset:%2 bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    }
#else
    (void)lds;
    (void)srsrc;
    (void)lds_offset;
    (void)lds_dfmt;
#endif
}

__device__ __forceinline__ void wait_lgkm0() {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
}

template <int Row, int Col, int Alt, int Offset1>
__device__ __forceinline__ void ds_read_matrix_trans_pair(
    const uint16_t* lds,
    int lds_offset,
    Vec8F16& frag0,
    Vec8F16& frag1) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_offset;
    asm volatile(
        "s_nop 0\n\t"
        "ds_read_matrix_trans_format %0, %2 offset:0 element:0x2 row:%3 col:%4 alt:%5\n\t"
        "ds_read_matrix_trans_format %1, %2 offset:%6 element:0x2 row:%3 col:%4 alt:%5\n"
        : "=v"(frag0), "=v"(frag1)
        : "s"(lds_addr), "n"(Row), "n"(Col), "n"(Alt), "n"(Offset1)
        : "memory");
#else
    (void)lds;
    (void)lds_offset;
    frag0 = {};
    frag1 = {};
#endif
}

template <int Row, int Col, int Alt, int Offset1>
__device__ __forceinline__ void ds_read_matrix_normal_pair(
    const uint16_t* lds,
    int lds_offset,
    Vec8F16& frag0,
    Vec8F16& frag1) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_offset;
    asm volatile(
        "s_nop 0\n\t"
        "ds_read_matrix_format %0, %2 offset:0 element:0x2 row:%3 col:%4 alt:%5\n\t"
        "ds_read_matrix_format %1, %2 offset:%6 element:0x2 row:%3 col:%4 alt:%5\n"
        : "=v"(frag0), "=v"(frag1)
        : "s"(lds_addr), "n"(Row), "n"(Col), "n"(Alt), "n"(Offset1)
        : "memory");
#else
    (void)lds;
    (void)lds_offset;
    frag0 = {};
    frag1 = {};
#endif
}

__device__ __forceinline__ Vec4F32 mmac_f16_lit(Vec4F16 lhs,
                                                Vec4F16 rhs,
                                                Vec4F32 acc) {
#if defined(__gfx946__) || defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(lhs, rhs, acc, 1, 0);
#else
    (void)lhs;
    (void)rhs;
    return acc;
#endif
}

__device__ __forceinline__ F32x4 fingerprint(const F16x8& rhs, int lane) {
    F16x8 lhs{};
    F32x4 acc{};
#if defined(__gfx946__) || defined(__gfx938__)
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        lhs.u16[i] = static_cast<uint16_t>(0x3400 + lane * kVec + i);
    }
    acc.f32 = mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], acc.f32);
    acc.f32 = mmac_f16_lit(lhs.f16x4[1], rhs.f16x4[1], acc.f32);
#else
    (void)rhs;
    (void)lane;
#endif
    return acc;
}

__device__ __forceinline__ void read_candidate_pair(int reader,
                                                    const uint16_t* lds,
                                                    int base,
                                                    F16x8& a,
                                                    F16x8& b) {
    if (reader == 0) {
        ds_read_matrix_trans_pair<0x2, 0x1, 0x0, 1024>(
            lds, base, a.f16x8, b.f16x8);
    } else if (reader == 1) {
        ds_read_matrix_normal_pair<0x2, 0x1, 0x0, 1024>(
            lds, base, a.f16x8, b.f16x8);
    } else if (reader == 2) {
        ds_read_matrix_trans_pair<0x1, 0x2, 0x0, 32>(
            lds, base, a.f16x8, b.f16x8);
    } else if (reader == 3) {
        ds_read_matrix_trans_pair<0x1, 0x2, 0x1, 32>(
            lds, base, a.f16x8, b.f16x8);
    } else {
        ds_read_matrix_normal_pair<0x2, 0x1, 0x1, 32>(
            lds, base, a.f16x8, b.f16x8);
    }
}

__device__ __forceinline__ void load_raw_block(int load_kind,
                                               uint16_t* lds,
                                               const uint16_t* k,
                                               int n_base,
                                               int d_base,
                                               int lds_base,
                                               int lds_dfmt) {
    if (load_kind == 0) {
        matrix_load_32x32_b16_bps_lds<true>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base, kDim),
            lds_base, lds_dfmt);
    } else if (load_kind == 1) {
        matrix_load_32x32_b16_bps_lds<false>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base, kDim),
            lds_base, lds_dfmt);
    } else if (load_kind == 2) {
        matrix_load_32x16_b16_bps_lds<false, false>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base, kDim),
            lds_base, lds_dfmt);
        matrix_load_32x16_b16_bps_lds<false, false>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base + 16, kDim),
            lds_base + 1024, lds_dfmt);
    } else if (load_kind == 3) {
        matrix_load_32x16_b16_bps_lds<true, false>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base, kDim),
            lds_base, lds_dfmt);
        matrix_load_32x16_b16_bps_lds<true, false>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base + 16, kDim),
            lds_base + 1024, lds_dfmt);
    } else if (load_kind == 4) {
        matrix_load_32x16_b16_bps_lds<false, true>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base, kDim),
            lds_base, lds_dfmt);
        matrix_load_32x16_b16_bps_lds<false, true>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base + 16, kDim),
            lds_base + 1024, lds_dfmt);
    } else {
        matrix_load_32x16_b16_bps_lds<true, true>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base, kDim),
            lds_base, lds_dfmt);
        matrix_load_32x16_b16_bps_lds<true, true>(
            lds, prepare_matrix_src(k + n_base * kDim + d_base + 16, kDim),
            lds_base + 1024, lds_dfmt);
    }
}

__global__ void dq_k_same_lds_native_probe_kernel(
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ kt_source,
    int load_kind,
    int lds_dfmt,
    uint16_t* __restrict__ ref_frag,
    uint16_t* __restrict__ cand_frag,
    float* __restrict__ ref_acc,
    float* __restrict__ cand_acc) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ uint16_t raw_lds[(kNTiles * kDTiles * kMatrixBlockBytes) /
                                sizeof(uint16_t)];
    __shared__ uint16_t ref_lds[(kDTiles * kKtDtileBytes) / sizeof(uint16_t)];
    const int lane = static_cast<int>(threadIdx.x % 64);

#pragma unroll
    for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
        const int d_base = d_tile * 32;
        const int ref_base = d_tile * kKtDtileBytes;
        matrix_load_32x32_b16_bps_lds<true>(
            ref_lds, prepare_matrix_src(kt_source + d_base * kNk, kNk),
            ref_base, 0);
        matrix_load_32x32_b16_bps_lds<true>(
            ref_lds, prepare_matrix_src(kt_source + d_base * kNk + 32, kNk),
            ref_base + kMatrixBlockBytes, 0);

#pragma unroll
        for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
            const int n_base = n_tile * 32;
            const int raw_base = (n_tile * kDTiles + d_tile) * kMatrixBlockBytes;
            load_raw_block(
                load_kind, raw_lds, k, n_base, d_base, raw_base, lds_dfmt);
        }
    }
    wait_lgkm0();
    __syncthreads();

#pragma unroll
    for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
        F16x8 ref[4];
        const int ref_base = d_tile * kKtDtileBytes;
        ds_read_matrix_trans_pair<0x2, 0x1, 0x0, 1024>(
            ref_lds, ref_base, ref[0].f16x8, ref[2].f16x8);
        ds_read_matrix_trans_pair<0x2, 0x1, 0x0, 1024>(
            ref_lds, ref_base + kMatrixBlockBytes,
            ref[1].f16x8, ref[3].f16x8);
        wait_lgkm0();

#pragma unroll
        for (int frag = 0; frag < kFragPerD; ++frag) {
#pragma unroll
            for (int vec = 0; vec < kVec; ++vec) {
                ref_frag[frag_index(d_tile, frag, lane, vec)] =
                    ref[frag].u16[vec];
            }
            const F32x4 acc = fingerprint(ref[frag], lane);
#pragma unroll
            for (int elem = 0; elem < kAcc; ++elem) {
                ref_acc[acc_index(d_tile, frag, lane, elem)] = acc.scalar[elem];
            }
        }

        for (int reader = 0; reader < kReaders; ++reader) {
            F16x8 cand_pair[2][2];
#pragma unroll
            for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
                const int raw_base =
                    (n_tile * kDTiles + d_tile) * kMatrixBlockBytes;
                read_candidate_pair(
                    reader, raw_lds, raw_base,
                    cand_pair[n_tile][0], cand_pair[n_tile][1]);
            }
            wait_lgkm0();

            F16x8 cand[4];
            cand[0] = cand_pair[0][0];
            cand[2] = cand_pair[0][1];
            cand[1] = cand_pair[1][0];
            cand[3] = cand_pair[1][1];

#pragma unroll
            for (int frag = 0; frag < kFragPerD; ++frag) {
#pragma unroll
                for (int vec = 0; vec < kVec; ++vec) {
                    cand_frag[cand_frag_index(reader, d_tile, frag, lane, vec)] =
                        cand[frag].u16[vec];
                }
                const F32x4 acc = fingerprint(cand[frag], lane);
#pragma unroll
                for (int elem = 0; elem < kAcc; ++elem) {
                    cand_acc[cand_acc_index(reader, d_tile, frag, lane, elem)] =
                        acc.scalar[elem];
                }
            }
        }
    }
#else
    (void)k;
    (void)kt_source;
    (void)load_kind;
    (void)lds_dfmt;
    (void)ref_frag;
    (void)cand_frag;
    (void)ref_acc;
    (void)cand_acc;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::cerr << what << ": " << hipGetErrorString(err) << "\n";
        std::exit(1);
    }
}

std::string hex16(uint16_t value) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(4) << std::setfill('0')
       << static_cast<int>(value);
    return os.str();
}

struct Metrics {
    int frag_mismatch = 0;
    int frag_total = 0;
    int halfblock_match = 0;
    int halfblock_total = 0;
    int first_unmatched_d_tile = -1;
    int first_unmatched_frag = -1;
    int first_unmatched_half = -1;
    bool halfblock_map_uniform = true;
    std::string halfblock_map_d0;
    int first_d_tile = -1;
    int first_frag = -1;
    int first_lane = -1;
    int first_vec = -1;
    uint16_t first_ref = 0;
    uint16_t first_cand = 0;
    double acc_max_abs = 0.0;
    double acc_rmse = 0.0;
    double acc_rel_l2 = 0.0;
};

bool halfblock_equal(const std::vector<uint16_t>& ref_frag,
                     const std::vector<uint16_t>& cand_frag,
                     int reader,
                     int d_tile,
                     int ref_frag_id,
                     int ref_half,
                     int cand_frag_id,
                     int cand_half) {
    for (int lane = 0; lane < kLanes; ++lane) {
        for (int v = 0; v < 4; ++v) {
            const int ref_vec = ref_half * 4 + v;
            const int cand_vec = cand_half * 4 + v;
            const uint16_t ref =
                ref_frag[frag_index(d_tile, ref_frag_id, lane, ref_vec)];
            const uint16_t cand = cand_frag[cand_frag_index(
                reader, d_tile, cand_frag_id, lane, cand_vec)];
            if (ref != cand) {
                return false;
            }
        }
    }
    return true;
}

Metrics compare_reader(const std::vector<uint16_t>& ref_frag,
                       const std::vector<uint16_t>& cand_frag,
                       const std::vector<float>& ref_acc,
                       const std::vector<float>& cand_acc,
                       int reader) {
    Metrics m;
    m.frag_total = kDTiles * kFragPerD * kLanes * kVec;
    for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
        for (int frag = 0; frag < kFragPerD; ++frag) {
            for (int lane = 0; lane < kLanes; ++lane) {
                for (int vec = 0; vec < kVec; ++vec) {
                    const uint16_t ref =
                        ref_frag[frag_index(d_tile, frag, lane, vec)];
                    const uint16_t cand = cand_frag[cand_frag_index(
                        reader, d_tile, frag, lane, vec)];
                    if (ref != cand) {
                        if (m.frag_mismatch == 0) {
                            m.first_d_tile = d_tile;
                            m.first_frag = frag;
                            m.first_lane = lane;
                            m.first_vec = vec;
                            m.first_ref = ref;
                            m.first_cand = cand;
                        }
                        ++m.frag_mismatch;
                    }
                }
            }
        }
    }

    m.halfblock_total = kDTiles * kFragPerD * 2;
    int match_frag[kDTiles][kFragPerD][2];
    int match_half[kDTiles][kFragPerD][2];
    for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
        for (int ref_frag_id = 0; ref_frag_id < kFragPerD; ++ref_frag_id) {
            for (int ref_half = 0; ref_half < 2; ++ref_half) {
                match_frag[d_tile][ref_frag_id][ref_half] = -1;
                match_half[d_tile][ref_frag_id][ref_half] = -1;
            }
        }
    }
    for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
        for (int ref_frag_id = 0; ref_frag_id < kFragPerD; ++ref_frag_id) {
            for (int ref_half = 0; ref_half < 2; ++ref_half) {
                bool matched = false;
                for (int cand_frag_id = 0; cand_frag_id < kFragPerD && !matched;
                     ++cand_frag_id) {
                    for (int cand_half = 0; cand_half < 2; ++cand_half) {
                        if (halfblock_equal(ref_frag, cand_frag, reader, d_tile,
                                            ref_frag_id, ref_half,
                                            cand_frag_id, cand_half)) {
                            matched = true;
                            match_frag[d_tile][ref_frag_id][ref_half] =
                                cand_frag_id;
                            match_half[d_tile][ref_frag_id][ref_half] =
                                cand_half;
                            break;
                        }
                    }
                }
                if (matched) {
                    ++m.halfblock_match;
                } else if (m.first_unmatched_d_tile < 0) {
                    m.first_unmatched_d_tile = d_tile;
                    m.first_unmatched_frag = ref_frag_id;
                    m.first_unmatched_half = ref_half;
                }
            }
        }
    }
    std::ostringstream map;
    for (int ref_frag_id = 0; ref_frag_id < kFragPerD; ++ref_frag_id) {
        for (int ref_half = 0; ref_half < 2; ++ref_half) {
            if (map.tellp() > 0) {
                map << "|";
            }
            map << "r" << ref_frag_id << "h" << ref_half << "=";
            if (match_frag[0][ref_frag_id][ref_half] >= 0) {
                map << "c" << match_frag[0][ref_frag_id][ref_half]
                    << "h" << match_half[0][ref_frag_id][ref_half];
            } else {
                map << "NA";
            }
        }
    }
    m.halfblock_map_d0 = map.str();
    for (int d_tile = 1; d_tile < kDTiles; ++d_tile) {
        for (int ref_frag_id = 0; ref_frag_id < kFragPerD; ++ref_frag_id) {
            for (int ref_half = 0; ref_half < 2; ++ref_half) {
                if (match_frag[d_tile][ref_frag_id][ref_half] !=
                        match_frag[0][ref_frag_id][ref_half] ||
                    match_half[d_tile][ref_frag_id][ref_half] !=
                        match_half[0][ref_frag_id][ref_half]) {
                    m.halfblock_map_uniform = false;
                }
            }
        }
    }

    double sum_sq = 0.0;
    double ref_sq = 0.0;
    int acc_total = 0;
    for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
        for (int frag = 0; frag < kFragPerD; ++frag) {
            for (int lane = 0; lane < kLanes; ++lane) {
                for (int elem = 0; elem < kAcc; ++elem) {
                    const float ref =
                        ref_acc[acc_index(d_tile, frag, lane, elem)];
                    const float cand = cand_acc[cand_acc_index(
                        reader, d_tile, frag, lane, elem)];
                    const double diff = static_cast<double>(cand) - ref;
                    m.acc_max_abs =
                        std::max(m.acc_max_abs, std::fabs(diff));
                    sum_sq += diff * diff;
                    ref_sq += static_cast<double>(ref) * ref;
                    ++acc_total;
                }
            }
        }
    }
    m.acc_rmse = std::sqrt(sum_sq / std::max(1, acc_total));
    m.acc_rel_l2 = std::sqrt(sum_sq / std::max(1.0e-30, ref_sq));
    return m;
}

void write_header(std::ostream& os) {
    os << "load_kind,lds_dfmt,reader,frag_mismatch,frag_total,"
          "frag_exact,halfblock_match,halfblock_total,halfblock_all_match,"
          "halfblock_map_uniform,halfblock_map_d0,"
          "acc_max_abs,acc_rmse,acc_rel_l2,"
          "first_d_tile,first_frag,first_lane,first_vec,first_ref,first_cand,"
          "first_unmatched_d_tile,first_unmatched_frag,first_unmatched_half,"
          "decision\n";
}

void write_metric_row(std::ostream& os,
                      int load_kind,
                      int lds_dfmt,
                      int reader,
                      const Metrics& m) {
    const bool exact = m.frag_mismatch == 0 && m.acc_max_abs == 0.0;
    const bool half_remap =
        m.halfblock_match == m.halfblock_total && m.halfblock_map_uniform;
    const char* decision = exact ? "ACCEPT_NATIVE_EQUIV"
                         : half_remap ? "ACCEPT_NATIVE_HALF_REMAP"
                                      : "REJECT_MISMATCH";
    os << load_kind_name(load_kind) << "," << lds_dfmt << ","
       << reader_name(reader) << "," << m.frag_mismatch << ","
       << m.frag_total << "," << (m.frag_mismatch == 0 ? 1 : 0) << ","
       << m.halfblock_match << "," << m.halfblock_total << ","
       << (m.halfblock_match == m.halfblock_total ? 1 : 0) << ","
       << (m.halfblock_map_uniform ? 1 : 0) << ","
       << m.halfblock_map_d0 << ","
       << std::setprecision(9) << m.acc_max_abs << "," << m.acc_rmse << ","
       << m.acc_rel_l2 << "," << m.first_d_tile << "," << m.first_frag
       << "," << m.first_lane << "," << m.first_vec << ","
       << hex16(m.first_ref) << "," << hex16(m.first_cand) << ","
       << m.first_unmatched_d_tile << "," << m.first_unmatched_frag << ","
       << m.first_unmatched_half << ","
       << decision << "\n";
}

std::string parse_csv_path(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--csv") {
            return argv[i + 1];
        }
    }
    return "";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string csv_path = parse_csv_path(argc, argv);
    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!csv_path.empty()) {
        file.open(csv_path);
        if (!file) {
            std::cerr << "failed to open csv: " << csv_path << "\n";
            return 1;
        }
        out = &file;
    }

    std::vector<uint16_t> h_k(kNk * kDim);
    for (int nk = 0; nk < kNk; ++nk) {
        for (int d = 0; d < kDim; ++d) {
            h_k[nk * kDim + d] = pattern_value(nk, d);
        }
    }

    std::vector<uint16_t> h_kt(kDim * kNk);
    for (int d = 0; d < kDim; ++d) {
        for (int nk = 0; nk < kNk; ++nk) {
            h_kt[d * kNk + nk] = h_k[nk * kDim + d];
        }
    }

    uint16_t* d_k = nullptr;
    uint16_t* d_kt = nullptr;
    uint16_t* d_ref_frag = nullptr;
    uint16_t* d_cand_frag = nullptr;
    float* d_ref_acc = nullptr;
    float* d_cand_acc = nullptr;

    const size_t ref_frag_elems = kDTiles * kFragPerD * kLanes * kVec;
    const size_t cand_frag_elems =
        kReaders * kDTiles * kFragPerD * kLanes * kVec;
    const size_t ref_acc_elems = kDTiles * kFragPerD * kLanes * kAcc;
    const size_t cand_acc_elems =
        kReaders * kDTiles * kFragPerD * kLanes * kAcc;

    std::vector<uint16_t> h_ref_frag(ref_frag_elems);
    std::vector<uint16_t> h_cand_frag(cand_frag_elems);
    std::vector<float> h_ref_acc(ref_acc_elems);
    std::vector<float> h_cand_acc(cand_acc_elems);

    check_hip(hipMalloc(&d_k, h_k.size() * sizeof(uint16_t)), "hipMalloc k");
    check_hip(hipMalloc(&d_kt, h_kt.size() * sizeof(uint16_t)), "hipMalloc kt");
    check_hip(hipMalloc(&d_ref_frag, ref_frag_elems * sizeof(uint16_t)),
              "hipMalloc ref_frag");
    check_hip(hipMalloc(&d_cand_frag, cand_frag_elems * sizeof(uint16_t)),
              "hipMalloc cand_frag");
    check_hip(hipMalloc(&d_ref_acc, ref_acc_elems * sizeof(float)),
              "hipMalloc ref_acc");
    check_hip(hipMalloc(&d_cand_acc, cand_acc_elems * sizeof(float)),
              "hipMalloc cand_acc");
    check_hip(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemcpy(d_kt, h_kt.data(), h_kt.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy kt");

    write_header(*out);
    for (int load_kind = 0; load_kind < kLoadKinds; ++load_kind) {
        for (int dfmt = 0; dfmt < kDfmtKinds; ++dfmt) {
            check_hip(hipMemset(d_ref_frag, 0,
                                ref_frag_elems * sizeof(uint16_t)),
                      "hipMemset ref_frag");
            check_hip(hipMemset(d_cand_frag, 0,
                                cand_frag_elems * sizeof(uint16_t)),
                      "hipMemset cand_frag");
            check_hip(hipMemset(d_ref_acc, 0, ref_acc_elems * sizeof(float)),
                      "hipMemset ref_acc");
            check_hip(hipMemset(d_cand_acc, 0, cand_acc_elems * sizeof(float)),
                      "hipMemset cand_acc");

            hipLaunchKernelGGL(dq_k_same_lds_native_probe_kernel, dim3(1),
                               dim3(64), 0, 0, d_k, d_kt, load_kind, dfmt,
                               d_ref_frag, d_cand_frag, d_ref_acc, d_cand_acc);
            check_hip(hipGetLastError(), "launch");
            check_hip(hipDeviceSynchronize(), "sync");
            check_hip(hipMemcpy(h_ref_frag.data(), d_ref_frag,
                                ref_frag_elems * sizeof(uint16_t),
                                hipMemcpyDeviceToHost),
                      "hipMemcpy ref_frag");
            check_hip(hipMemcpy(h_cand_frag.data(), d_cand_frag,
                                cand_frag_elems * sizeof(uint16_t),
                                hipMemcpyDeviceToHost),
                      "hipMemcpy cand_frag");
            check_hip(hipMemcpy(h_ref_acc.data(), d_ref_acc,
                                ref_acc_elems * sizeof(float),
                                hipMemcpyDeviceToHost),
                      "hipMemcpy ref_acc");
            check_hip(hipMemcpy(h_cand_acc.data(), d_cand_acc,
                                cand_acc_elems * sizeof(float),
                                hipMemcpyDeviceToHost),
                      "hipMemcpy cand_acc");

            for (int reader = 0; reader < kReaders; ++reader) {
                const Metrics m = compare_reader(
                    h_ref_frag, h_cand_frag, h_ref_acc, h_cand_acc, reader);
                write_metric_row(*out, load_kind, dfmt, reader, m);
            }
        }
    }

    (void)hipFree(d_cand_acc);
    (void)hipFree(d_ref_acc);
    (void)hipFree(d_cand_frag);
    (void)hipFree(d_ref_frag);
    (void)hipFree(d_kt);
    (void)hipFree(d_k);
    return 0;
}
