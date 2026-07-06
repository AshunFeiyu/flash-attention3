#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kSubMq = 32;
constexpr int kBlockMq = 64;
constexpr int kBlockNk = 64;
constexpr int kHeadDim = 128;
constexpr int kMatrixBlockBytes = 32 * 32 * 2;
constexpr int kHalfBytes = 2;

struct Tile {
    static constexpr int kSubMq = ::kSubMq;
    static constexpr int kBlockMq = ::kBlockMq;
    static constexpr int kBlockNk = ::kBlockNk;
    static constexpr int kHeadDim = ::kHeadDim;
    static constexpr int kHalfBytes = ::kHalfBytes;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
};

template <typename T>
struct LdsLayout {
    static constexpr int kSubMq = T::kSubMq;
    static constexpr int kBlockNk = T::kBlockNk;
    static constexpr int kHeadDim = T::kHeadDim;
    static constexpr int kHalfBytes = T::kHalfBytes;
    static constexpr int kMatrixBlockBytes = 32 * 32 * kHalfBytes;
    static constexpr int kQBase = 0;
    static constexpr int kDoutBase = kQBase + kSubMq * kHeadDim * kHalfBytes;
    static constexpr int kPageBase =
        kDoutBase + kSubMq * kHeadDim * kHalfBytes;
    static constexpr int kKPageBytes = kBlockNk * kHeadDim * kHalfBytes;
    static constexpr int kVPageBytes = kBlockNk * kHeadDim * kHalfBytes;
    static constexpr int kKtPageBytes = kHeadDim * kBlockNk * kHalfBytes;
    static constexpr int kDsPageBytes = kSubMq * kBlockNk * kHalfBytes;
    static constexpr int kPageBytes =
        kKPageBytes + kVPageBytes + kKtPageBytes + kDsPageBytes;
    static constexpr int kPages = 2;
    static constexpr int kSidecarBase = kPageBase + kPages * kPageBytes;
    static constexpr int kSidecarRows = kSubMq;
    static constexpr int kSidecarBytes = 3 * kSidecarRows * sizeof(float);
    static constexpr int kKBase() {
        return kPageBase;
    }
    static constexpr int kVBase() {
        return kKBase() + kKPageBytes;
    }
    static constexpr int kKtBase() {
        return kVBase() + kVPageBytes;
    }
    static constexpr int kDsBase() {
        return kKtBase() + kKtPageBytes;
    }
    static constexpr int kBytes = kSidecarBase + kSidecarBytes;
    static_assert(kBytes <= T::kLdsBudgetBytes,
                  "probe LDS plan must fit 128KB");
};

struct Bar {
    static constexpr int kPage0Filled = 0;
    static constexpr int kPage0DsFilled = 1;
    static constexpr int kPage0Used = 2;
    static constexpr int kPage1Filled = 3;
    static constexpr int kPage1DsFilled = 4;
    static constexpr int kPage1Used = 5;
    static constexpr int kQDoUsed = 6;
    static constexpr int kAllDone = 7;
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x & 63);
}

__device__ __forceinline__ void zero_f32x4(ins::F32x4& acc) {
    ins::zero_vgpr2(acc.u64[0]);
    ins::zero_vgpr2(acc.u64[1]);
}

__device__ __forceinline__ int pds_lds_dst(int mq, int nk) {
    const int nk_bit5 = (nk >> 5) & 1;
    const int n = nk & 31;
    const int y0_2 = n & 7;
    const int y3 = ((n >> 3) ^ (mq >> 2)) & 1;
    const int y4 = ((n >> 4) ^ (mq >> 3)) & 1;
    const int y5 = ((mq >> 0) ^ (mq >> 4)) & 1;
    const int y6 = (nk_bit5 ^ (mq >> 1)) & 1;
    const int y7 = (mq >> 2) & 1;
    const int y8 = (mq >> 3) & 1;
    const int y9 = (mq >> 4) & 1;
    const int y10 = nk_bit5;
    return y0_2 | (y3 << 3) | (y4 << 4) | (y5 << 5) | (y6 << 6) |
           (y7 << 7) | (y8 << 8) | (y9 << 9) | (y10 << 10);
}

template <typename T>
__device__ __forceinline__ float* sidecar_lds(__half* __restrict__ lds) {
    return reinterpret_cast<float*>(
        lds + LdsLayout<T>::kSidecarBase / sizeof(__half));
}

template <typename T>
__device__ __forceinline__ const float* sidecar_lds(
    const __half* __restrict__ lds) {
    return reinterpret_cast<const float*>(
        lds + LdsLayout<T>::kSidecarBase / sizeof(__half));
}

template <typename T>
__device__ __forceinline__ void load_q_dout_tile(
    const __half* __restrict__ q,
    const __half* __restrict__ dout,
    __half* __restrict__ lds,
    int producer_wave,
    int q_base_sub) {
    constexpr int Dim = T::kHeadDim;
    const int d_base = producer_wave * 32;
    ins::Vec4U32 q_src =
        ins::prepare_matrix_src(q + q_base_sub * Dim + d_base, Dim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + LdsLayout<T>::kQBase / sizeof(__half), q_src,
        producer_wave * kMatrixBlockBytes, true);

    ins::Vec4U32 dout_src =
        ins::prepare_matrix_src(dout + q_base_sub * Dim + d_base, Dim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + LdsLayout<T>::kDoutBase / sizeof(__half), dout_src,
        producer_wave * kMatrixBlockBytes, true);
}

template <typename T>
__device__ __forceinline__ void load_sidecar_tile(
    const float* __restrict__ scores_max,
    const float* __restrict__ scores_sum,
    const float* __restrict__ delta,
    __half* __restrict__ lds,
    int producer_wave,
    int lane,
    int q_base_sub) {
    constexpr int Rows = LdsLayout<T>::kSidecarRows;
    float* sidecar = sidecar_lds<T>(lds);
    const int producer_lane = producer_wave * 64 + lane;
    for (int idx = producer_lane; idx < 3 * Rows; idx += 4 * 64) {
        const int field = idx / Rows;
        const int local_row = idx - field * Rows;
        const int row = q_base_sub + local_row;
        if (field == 0) {
            sidecar[idx] = scores_max[row];
        } else if (field == 1) {
            sidecar[idx] = scores_sum[row];
        } else {
            sidecar[idx] = delta[row];
        }
    }
}

template <typename T>
__device__ __forceinline__ void load_kv_tile_page(
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ lds,
    int producer_wave) {
    constexpr int Dim = T::kHeadDim;
    const int d_base = producer_wave * 32;
    ins::Vec4U32 k_src0 = ins::prepare_matrix_src(k + d_base, Dim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + LdsLayout<T>::kKBase() / sizeof(__half), k_src0,
        producer_wave * kMatrixBlockBytes, true);
    ins::Vec4U32 k_src1 = ins::prepare_matrix_src(k + 32 * Dim + d_base, Dim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + LdsLayout<T>::kKBase() / sizeof(__half), k_src1,
        (4 + producer_wave) * kMatrixBlockBytes, true);

    ins::Vec4U32 v_src0 = ins::prepare_matrix_src(v + d_base, Dim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + LdsLayout<T>::kVBase() / sizeof(__half), v_src0,
        producer_wave * kMatrixBlockBytes, true);
    ins::Vec4U32 v_src1 = ins::prepare_matrix_src(v + 32 * Dim + d_base, Dim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + LdsLayout<T>::kVBase() / sizeof(__half), v_src1,
        (4 + producer_wave) * kMatrixBlockBytes, true);
}

template <typename T>
__device__ __forceinline__ void load_kt_tile(
    const __half* __restrict__ k_t_source,
    __half* __restrict__ lds,
    int d_tile) {
    constexpr int Nk = T::kBlockNk;
    constexpr int DTile = 32;
    const int d_base = d_tile * DTile;
    const int k_lds_base = d_tile * DTile * Nk * sizeof(__half);
    __half* kt_lds = lds + LdsLayout<T>::kKtBase() / sizeof(__half);
    ins::Vec4U32 src0 =
        ins::prepare_matrix_src(k_t_source + d_base * Nk, Nk);
    ins::matrix_load_32x32_b16_bps_lds(kt_lds, src0, k_lds_base, true);
    ins::Vec4U32 src1 =
        ins::prepare_matrix_src(k_t_source + d_base * Nk + 32, Nk);
    ins::matrix_load_32x32_b16_bps_lds(
        kt_lds, src1, k_lds_base + kMatrixBlockBytes, true);
}

template <typename T, int MHalf, int NChunk>
__device__ __forceinline__ void publish_ds_chunk(
    const __half* __restrict__ lds,
    int lane,
    int q_base_sub,
    float softmax_scale,
    float softmax_scale_log2) {
    constexpr int Nk = T::kBlockNk;
    constexpr int Dim = T::kHeadDim;
    constexpr int KBlocks = Dim / 32;
    constexpr int SidecarRows = LdsLayout<T>::kSidecarRows;
    const int lane_mq = lane % 16;
    const int lane_n = lane / 16;
    const int qrow = q_base_sub + MHalf * 16 + lane_mq;
    const int local_mq = MHalf * 16 + lane_mq;
    const int n_tile = NChunk / 2;
    const int n_half = NChunk & 1;

    const __half* q_lds = lds + LdsLayout<T>::kQBase / sizeof(__half);
    const __half* dout_lds = lds + LdsLayout<T>::kDoutBase / sizeof(__half);
    const __half* k_lds = lds + LdsLayout<T>::kKBase() / sizeof(__half);
    const __half* v_lds = lds + LdsLayout<T>::kVBase() / sizeof(__half);
    __half* ds_lds =
        const_cast<__half*>(lds) + LdsLayout<T>::kDsBase() / sizeof(__half);
    const float* sidecar = sidecar_lds<T>(lds);
    const float row_max = sidecar[local_mq];
    const float row_sum = sidecar[SidecarRows + local_mq];
    const float row_delta = sidecar[2 * SidecarRows + local_mq];

    ins::F16x8 q_reg[KBlocks];
#pragma unroll
    for (int k_block = 0; k_block < KBlocks; ++k_block) {
        constexpr int MHalfOffset = MHalf == 0 ? 0 : 1024;
        ins::ds_read_matrix_32x16_trans(
            q_lds, k_block * kMatrixBlockBytes + MHalfOffset,
            q_reg[k_block].f16x8);
    }
    ins::wait_lgkm(0);

    ins::F32x4 qk_acc;
    ins::F32x4 dp_acc;
    zero_f32x4(qk_acc);
    zero_f32x4(dp_acc);

#pragma unroll
    for (int k_block = 0; k_block < KBlocks; ++k_block) {
        ins::F16x8 dout_frag;
        ins::F16x8 k_frag;
        ins::F16x8 v_frag;
        const int b_block = n_tile * KBlocks + k_block;
        constexpr int MHalfOffset = MHalf == 0 ? 0 : 1024;
        const int n_half_offset = n_half == 0 ? 0 : 1024;
        ins::ds_read_matrix_32x16_trans(
            dout_lds, k_block * kMatrixBlockBytes + MHalfOffset,
            dout_frag.f16x8);
        ins::ds_read_matrix_32x16_trans(
            k_lds, b_block * kMatrixBlockBytes + n_half_offset,
            k_frag.f16x8);
        ins::ds_read_matrix_32x16_trans(
            v_lds, b_block * kMatrixBlockBytes + n_half_offset,
            v_frag.f16x8);
        ins::wait_lgkm(0);

#pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            const ins::Vec4F16 k_rhs = k_frag.f16x4[k_half];
            const ins::Vec4F16 v_rhs = v_frag.f16x4[k_half];
            const ins::Vec4F16 dout_lhs = dout_frag.f16x4[k_half];
            qk_acc.f32 =
                ins::mmac_f16_lit(q_reg[k_block].f16x4[k_half], k_rhs,
                                  qk_acc.f32);
            dp_acc.f32 =
                ins::mmac_f16_lit(dout_lhs, v_rhs, dp_acc.f32);
        }
    }

#pragma unroll
    for (int vec_id = 0; vec_id < 4; ++vec_id) {
        const int nk = NChunk * 16 + lane_n * 4 + vec_id;
        const int krow = nk;
        float ds_value = 0.0f;
        if (krow <= qrow) {
            const float p =
                exp2f((qk_acc.scalar[vec_id] - row_max) *
                      softmax_scale_log2) /
                row_sum;
            ds_value =
                p * (dp_acc.scalar[vec_id] - row_delta) * softmax_scale;
        }
        ds_lds[pds_lds_dst(local_mq, nk)] = __float2half(ds_value);
    }
    ins::wait_lgkm(0);
}

template <typename T, int DTileId>
__device__ __forceinline__ void consume_ds_kt_full_dtile(
    const __half* __restrict__ lds,
    ins::F32x4 (&dq_reg)[4]) {
    constexpr int Nk = T::kBlockNk;
    constexpr int DTile = 32;
    const __half* ds_lds = lds + LdsLayout<T>::kDsBase() / sizeof(__half);
    const __half* kt_lds = lds + LdsLayout<T>::kKtBase() / sizeof(__half);
    const int kt_lds_base = DTileId * DTile * Nk * sizeof(__half);

    ins::F16x8 ds_reg[4];
    ins::F16x8 kt_reg[4];
    ins::ds_read_matrix_trans_pair(
        ds_lds, 0, ds_reg[0].f16x8, ds_reg[2].f16x8);
    ins::ds_read_matrix_trans_pair(
        ds_lds, kMatrixBlockBytes, ds_reg[1].f16x8, ds_reg[3].f16x8);
    ins::ds_read_matrix_trans_pair(
        kt_lds, kt_lds_base, kt_reg[0].f16x8, kt_reg[2].f16x8);
    ins::ds_read_matrix_trans_pair(
        kt_lds, kt_lds_base + kMatrixBlockBytes,
        kt_reg[1].f16x8, kt_reg[3].f16x8);
    ins::wait_lgkm(0);

#pragma unroll
    for (int mq_idx = 0; mq_idx < 2; ++mq_idx) {
#pragma unroll
        for (int d_idx = 0; d_idx < DTile / 16; ++d_idx) {
#pragma unroll
            for (int nk_idx = 0; nk_idx < Nk / 16; ++nk_idx) {
                const int acc_idx = mq_idx * (DTile / 16) + d_idx;
                dq_reg[acc_idx].f32 =
                    ins::mmac_f16_lit(
                        ds_reg[mq_idx * (Nk / 32) + nk_idx / 2]
                            .f16x4[nk_idx % 2],
                        kt_reg[d_idx * (Nk / 32) + nk_idx / 2]
                            .f16x4[nk_idx % 2],
                        dq_reg[acc_idx].f32);
            }
        }
    }
    ins::keep_accumulator_live(dq_reg[0]);
    ins::keep_accumulator_live(dq_reg[1]);
    ins::keep_accumulator_live(dq_reg[2]);
    ins::keep_accumulator_live(dq_reg[3]);
}

template <typename T, int MHalf>
__device__ __forceinline__ void publish_worker_chunk(
    int worker_slot,
    const __half* __restrict__ lds,
    int lane,
    int q_base_sub,
    float softmax_scale,
    float softmax_scale_log2) {
    if (worker_slot == 0) {
        publish_ds_chunk<T, MHalf, 0>(
            lds, lane, q_base_sub, softmax_scale, softmax_scale_log2);
    } else if (worker_slot == 1) {
        publish_ds_chunk<T, MHalf, 1>(
            lds, lane, q_base_sub, softmax_scale, softmax_scale_log2);
    } else if (worker_slot == 2) {
        publish_ds_chunk<T, MHalf, 2>(
            lds, lane, q_base_sub, softmax_scale, softmax_scale_log2);
    } else {
        publish_ds_chunk<T, MHalf, 3>(
            lds, lane, q_base_sub, softmax_scale, softmax_scale_log2);
    }
}

__device__ __forceinline__ void skew_worker_progress(int worker_slot) {
    for (int i = 0; i < worker_slot * 96; ++i) {
        asm volatile("s_nop 0" ::: "memory");
    }
}

template <int ConsumerSlot>
__device__ __forceinline__ void consume_role_body(
    const __half* __restrict__ lds,
    int* __restrict__ out) {
    ins::F32x4 dq_reg[4];
    zero_f32x4(dq_reg[0]);
    zero_f32x4(dq_reg[1]);
    zero_f32x4(dq_reg[2]);
    zero_f32x4(dq_reg[3]);
    consume_ds_kt_full_dtile<Tile, ConsumerSlot>(lds, dq_reg);
    if (lane_id() == 0) {
        atomicAdd(out + 2, 1);
    }
}

__global__ void __launch_bounds__(768, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
dq_qsubtile_real_helper_probe_kernel(const __half* __restrict__ q,
                                     const __half* __restrict__ k,
                                     const __half* __restrict__ v,
                                     const __half* __restrict__ dout,
                                     const float* __restrict__ scores_max,
                                     const float* __restrict__ scores_sum,
                                     const float* __restrict__ delta,
                                     const __half* __restrict__ k_t_source,
                                     int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[LdsLayout<Tile>::kBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kPage0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0DsFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kPage1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage1DsFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage1Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQDoUsed, 4);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(56);
        const int producer_wave = static_cast<int>(wave_id);
        const int lane = lane_id();
        int qdo_phase = 0;
        int page_used_phase = 0;
        int page0_seen = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            const int q_base_sub = q_sub * kSubMq;
            if (q_sub > 0) {
                ins::abarrier_try_wait<false>(Bar::kQDoUsed, qdo_phase);
            }
            load_q_dout_tile<Tile>(q, dout, lds, producer_wave, q_base_sub);
            load_sidecar_tile<Tile>(
                scores_max, scores_sum, delta, lds, producer_wave, lane,
                q_base_sub);
            if (page0_seen != 0) {
                ins::abarrier_try_wait<false>(
                    Bar::kPage0Used, page_used_phase);
            }
            page0_seen = 1;
            ins::abarrier_seq<false>(Bar::kPage0Filled);
            load_kv_tile_page<Tile>(k, v, lds, producer_wave);
            load_kt_tile<Tile>(k_t_source, lds, producer_wave);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Filled, 1);
        }
        if (lane == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 8 && wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(144);
        const int worker_slot = static_cast<int>(wave_id - 8);
        const int lane = lane_id();
        int filled_phase = 0;
        int qdo_worker_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            const int q_base_sub = q_sub * kSubMq;
            ins::abarrier_try_wait<false>(Bar::kPage0Filled, filled_phase);
            skew_worker_progress(worker_slot);
            ins::abarrier_seq<false>(Bar::kPage0DsFilled);
            publish_worker_chunk<Tile, 0>(
                worker_slot, lds, lane, q_base_sub, 1.0f, 1.0f);
            publish_worker_chunk<Tile, 1>(
                worker_slot, lds, lane, q_base_sub, 1.0f, 1.0f);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0DsFilled, 1);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
            if (q_sub == 0) {
                ins::abarrier_arrive_cnt<false>(Bar::kQDoUsed, 1);
                ins::abarrier_try_wait<false>(
                    Bar::kQDoUsed, qdo_worker_phase);
            }
        }
        if (lane == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 4 && wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(88);
        int ds_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            (void)q_sub;
            ins::abarrier_try_wait<false>(Bar::kPage0DsFilled, ds_phase);
            if (wave_id == 4) {
                consume_role_body<0>(lds, out);
            } else if (wave_id == 5) {
                consume_role_body<1>(lds, out);
            } else if (wave_id == 6) {
                consume_role_body<2>(lds, out);
            } else {
                consume_role_body<3>(lds, out);
            }
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
        }
        if (lane_id() == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __syncthreads();
    if (static_cast<int>(threadIdx.x) == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kPage0Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage0DsFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage0Used);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1DsFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1Used);
        __builtin_hcu_s_abarrier_inv(Bar::kQDoUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)k;
    (void)v;
    (void)dout;
    (void)scores_max;
    (void)scores_sum;
    (void)delta;
    (void)k_t_source;
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

void fill_zero_half(std::vector<uint16_t>& dst) {
    for (uint16_t& v : dst) {
        v = 0;
    }
}

}  // namespace

int main() {
    std::vector<uint16_t> h_q(kBlockMq * kHeadDim);
    std::vector<uint16_t> h_k(kBlockNk * kHeadDim);
    std::vector<uint16_t> h_v(kBlockNk * kHeadDim);
    std::vector<uint16_t> h_dout(kBlockMq * kHeadDim);
    std::vector<uint16_t> h_kt(kHeadDim * kBlockNk);
    std::vector<float> h_max(kBlockMq, 0.0f);
    std::vector<float> h_sum(kBlockMq, 1.0f);
    std::vector<float> h_delta(kBlockMq, 0.0f);
    fill_zero_half(h_q);
    fill_zero_half(h_k);
    fill_zero_half(h_v);
    fill_zero_half(h_dout);
    fill_zero_half(h_kt);

    __half* d_q = nullptr;
    __half* d_k = nullptr;
    __half* d_v = nullptr;
    __half* d_dout = nullptr;
    __half* d_kt = nullptr;
    float* d_max = nullptr;
    float* d_sum = nullptr;
    float* d_delta = nullptr;
    int* d_out = nullptr;
    int h_out[3] = {0, 0, 0};

    check_hip(hipMalloc(&d_q, h_q.size() * sizeof(uint16_t)), "hipMalloc q");
    check_hip(hipMalloc(&d_k, h_k.size() * sizeof(uint16_t)), "hipMalloc k");
    check_hip(hipMalloc(&d_v, h_v.size() * sizeof(uint16_t)), "hipMalloc v");
    check_hip(hipMalloc(&d_dout, h_dout.size() * sizeof(uint16_t)),
              "hipMalloc dout");
    check_hip(hipMalloc(&d_kt, h_kt.size() * sizeof(uint16_t)),
              "hipMalloc kt");
    check_hip(hipMalloc(&d_max, h_max.size() * sizeof(float)),
              "hipMalloc max");
    check_hip(hipMalloc(&d_sum, h_sum.size() * sizeof(float)),
              "hipMalloc sum");
    check_hip(hipMalloc(&d_delta, h_delta.size() * sizeof(float)),
              "hipMalloc delta");
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc out");

    check_hip(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemcpy(d_v, h_v.data(), h_v.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy v");
    check_hip(hipMemcpy(d_dout, h_dout.data(),
                        h_dout.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy dout");
    check_hip(hipMemcpy(d_kt, h_kt.data(), h_kt.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy kt");
    check_hip(hipMemcpy(d_max, h_max.data(), h_max.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy max");
    check_hip(hipMemcpy(d_sum, h_sum.data(), h_sum.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy sum");
    check_hip(hipMemcpy(d_delta, h_delta.data(),
                        h_delta.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "hipMemcpy delta");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset out");

    hipLaunchKernelGGL(dq_qsubtile_real_helper_probe_kernel,
                       dim3(1), dim3(768), 0, 0,
                       d_q, d_k, d_v, d_dout,
                       d_max, d_sum, d_delta, d_kt, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy out");

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_dout);
    hipFree(d_kt);
    hipFree(d_max);
    hipFree(d_sum);
    hipFree(d_delta);
    hipFree(d_out);

    const bool pass = (h_out[1] == 12 && h_out[2] == 8);
    std::printf("real_helper_probe role_done=%d consumer_epochs=%d pass=%d\n",
                h_out[1], h_out[2], pass ? 1 : 0);
    return pass ? 0 : 2;
}
