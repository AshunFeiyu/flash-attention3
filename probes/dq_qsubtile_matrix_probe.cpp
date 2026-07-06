#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kSubMq = 32;
constexpr int kHeadDim = 128;
constexpr int kBlockNk = 64;
constexpr int kMatrixBlockBytes = 32 * 32 * 2;
constexpr uint16_t kQBaseBits = 0x1000;
constexpr uint16_t kDoutBaseBits = 0x5000;
constexpr uint16_t kKBaseBits = 0x9000;

struct Bar {
    static constexpr int kPage0Filled = 0;
    static constexpr int kPage0DsFilled = 1;
    static constexpr int kPage0Used = 2;
    static constexpr int kQDoUsed = 3;
    static constexpr int kAllDone = 4;
};

struct Lds {
    static constexpr int kQBase = 0;
    static constexpr int kDoutBase = kQBase + kSubMq * kHeadDim * 2;
    static constexpr int kKBase = kDoutBase + kSubMq * kHeadDim * 2;
    static constexpr int kBytes = kKBase + kBlockNk * kHeadDim * 2;
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x & 63);
}

__host__ __device__ __forceinline__ int decode_row(uint16_t bits,
                                                   uint16_t base_bits) {
    if (bits < base_bits) {
        return -1;
    }
    const uint16_t rel = static_cast<uint16_t>(bits - base_bits);
    if (rel >= kBlockNk * kHeadDim) {
        return -1;
    }
    return static_cast<int>(rel) / kHeadDim;
}

__device__ __forceinline__ void check_frag_rows(ins::F16x8 frag,
                                                uint16_t base_bits,
                                                int row_lo,
                                                int row_hi,
                                                int* out) {
    const uint16_t* bits = reinterpret_cast<const uint16_t*>(&frag.f16x8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = decode_row(bits[i], base_bits);
        if (row < row_lo || row >= row_hi) {
            atomicAdd(out, 1);
        }
    }
}

__device__ __forceinline__ void load_qdo_qsub(const __half* __restrict__ q,
                                              const __half* __restrict__ dout,
                                              __half* __restrict__ lds,
                                              int producer_wave,
                                              int q_sub) {
    const int d_base = producer_wave * 32;
    const int row_base = q_sub * kSubMq;
    ins::Vec4U32 q_src =
        ins::prepare_matrix_src(q + row_base * kHeadDim + d_base, kHeadDim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + Lds::kQBase / sizeof(__half), q_src,
        producer_wave * kMatrixBlockBytes, true);
    ins::Vec4U32 dout_src = ins::prepare_matrix_src(
        dout + row_base * kHeadDim + d_base, kHeadDim);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + Lds::kDoutBase / sizeof(__half), dout_src,
        producer_wave * kMatrixBlockBytes, true);
}

__device__ __forceinline__ void load_k_page(const __half* __restrict__ k,
                                            __half* __restrict__ lds,
                                            int producer_wave) {
    const int d_base = producer_wave * 32;
    __half* k_lds = lds + Lds::kKBase / sizeof(__half);
    ins::Vec4U32 k0 =
        ins::prepare_matrix_src(k + d_base, kHeadDim);
    ins::matrix_load_32x32_b16_bps_lds(
        k_lds, k0, producer_wave * kMatrixBlockBytes, true);
    ins::Vec4U32 k1 =
        ins::prepare_matrix_src(k + 32 * kHeadDim + d_base, kHeadDim);
    ins::matrix_load_32x32_b16_bps_lds(
        k_lds, k1, (4 + producer_wave) * kMatrixBlockBytes, true);
}

__device__ __forceinline__ void check_qdo_fragments(
    const __half* __restrict__ lds,
    int worker_slot,
    int q_sub,
    int* out) {
    const int row_lo = q_sub * kSubMq;
    const int row_hi = row_lo + kSubMq;
    const int d_block = worker_slot;
    const __half* q_lds = lds + Lds::kQBase / sizeof(__half);
    const __half* dout_lds = lds + Lds::kDoutBase / sizeof(__half);
    const int base_off = d_block * kMatrixBlockBytes;
    ins::F16x8 q_m0;
    ins::F16x8 q_m1;
    ins::F16x8 dout_m0;
    ins::F16x8 dout_m1;
    ins::ds_read_matrix_32x16_trans(q_lds, base_off, q_m0.f16x8);
    ins::ds_read_matrix_32x16_trans(q_lds, base_off + 1024, q_m1.f16x8);
    ins::ds_read_matrix_32x16_trans(dout_lds, base_off, dout_m0.f16x8);
    ins::ds_read_matrix_32x16_trans(dout_lds, base_off + 1024,
                                    dout_m1.f16x8);
    ins::wait_lgkm(0);
    check_frag_rows(q_m0, kQBaseBits, row_lo, row_hi, out);
    check_frag_rows(q_m1, kQBaseBits, row_lo, row_hi, out);
    check_frag_rows(dout_m0, kDoutBaseBits, row_lo, row_hi, out);
    check_frag_rows(dout_m1, kDoutBaseBits, row_lo, row_hi, out);
}

__global__ void __launch_bounds__(768, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
dq_qsubtile_matrix_probe_kernel(const __half* __restrict__ q,
                                const __half* __restrict__ dout,
                                const __half* __restrict__ k,
                                int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[Lds::kBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kPage0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0DsFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQDoUsed, 4);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(48);
        const int producer_wave = static_cast<int>(wave_id);
        const int lane = lane_id();
        (void)lane;
        int qdo_phase = 0;
        int page_used_phase = 0;
        int page0_seen = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            if (q_sub > 0) {
                ins::abarrier_try_wait<true>(Bar::kQDoUsed, qdo_phase);
            }
            load_qdo_qsub(q, dout, lds, producer_wave, q_sub);
            if (page0_seen != 0) {
                ins::abarrier_try_wait<true>(
                    Bar::kPage0Used, page_used_phase);
            }
            page0_seen = 1;
            ins::abarrier_seq<false>(Bar::kPage0Filled);
            load_k_page(k, lds, producer_wave);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Filled, 1);
        }
        if (lane_id() == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 8 && wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(48);
        const int worker_slot = static_cast<int>(wave_id - 8);
        int filled_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            ins::abarrier_try_wait<true>(Bar::kPage0Filled, filled_phase);
            check_qdo_fragments(lds, worker_slot, q_sub, out);
            ins::abarrier_seq<false>(Bar::kPage0DsFilled);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0DsFilled, 1);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
            ins::abarrier_arrive_cnt<false>(Bar::kQDoUsed, 1);
        }
        if (lane_id() == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 4 && wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(48);
        int ds_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            (void)q_sub;
            ins::abarrier_try_wait<true>(Bar::kPage0DsFilled, ds_phase);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
        }
        if (lane_id() == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(48);
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
        __builtin_hcu_s_abarrier_inv(Bar::kQDoUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)dout;
    (void)k;
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

void fill_matrix(std::vector<uint16_t>& dst, uint16_t base_bits, int rows) {
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < kHeadDim; ++d) {
            dst[r * kHeadDim + d] =
                static_cast<uint16_t>(base_bits + r * kHeadDim + d);
        }
    }
}

}  // namespace

int main() {
    std::vector<uint16_t> h_q(2 * kSubMq * kHeadDim);
    std::vector<uint16_t> h_dout(2 * kSubMq * kHeadDim);
    std::vector<uint16_t> h_k(kBlockNk * kHeadDim);
    fill_matrix(h_q, kQBaseBits, 2 * kSubMq);
    fill_matrix(h_dout, kDoutBaseBits, 2 * kSubMq);
    fill_matrix(h_k, kKBaseBits, kBlockNk);

    __half* d_q = nullptr;
    __half* d_dout = nullptr;
    __half* d_k = nullptr;
    int* d_out = nullptr;
    int h_out[2] = {0, 0};
    check_hip(hipMalloc(&d_q, h_q.size() * sizeof(uint16_t)), "hipMalloc q");
    check_hip(hipMalloc(&d_dout, h_dout.size() * sizeof(uint16_t)),
              "hipMalloc dout");
    check_hip(hipMalloc(&d_k, h_k.size() * sizeof(uint16_t)), "hipMalloc k");
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc out");
    check_hip(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemcpy(d_dout, h_dout.data(),
                        h_dout.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy dout");
    check_hip(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset out");

    hipLaunchKernelGGL(dq_qsubtile_matrix_probe_kernel,
                       dim3(1), dim3(768), 0, 0, d_q, d_dout, d_k, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy out");
    hipFree(d_out);
    hipFree(d_k);
    hipFree(d_dout);
    hipFree(d_q);
    std::printf("dq_qsubtile_matrix_probe errors=%d done_waves=%d pass=%d\n",
                h_out[0], h_out[1], h_out[0] == 0 && h_out[1] == 12);
    return h_out[0] == 0 && h_out[1] == 12 ? 0 : 2;
}
