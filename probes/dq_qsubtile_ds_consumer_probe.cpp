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
    static constexpr int kDsBase = kDoutBase + kSubMq * kHeadDim * 2;
    static constexpr int kBytes = kDsBase + kSubMq * kBlockNk * 2;
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x & 63);
}

__device__ __forceinline__ int ds_probe_dst(int mq, int nk) {
    return mq * kBlockNk + nk;
}

__device__ __forceinline__ void skew_worker_progress(int worker_slot) {
    for (int i = 0; i < worker_slot * 96; ++i) {
        asm volatile("s_nop 0" ::: "memory");
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

__device__ __forceinline__ void publish_ds_like_tile(__half* __restrict__ lds,
                                                     int worker_slot,
                                                     int lane,
                                                     int q_sub) {
    __half* ds_lds = lds + Lds::kDsBase / sizeof(__half);
    for (int idx = worker_slot * 64 + lane; idx < kSubMq * kBlockNk;
         idx += 4 * 64) {
        const int mq = idx / kBlockNk;
        const int nk = idx - mq * kBlockNk;
        const uint16_t bits =
            static_cast<uint16_t>(0x3000 + q_sub * 0x0400 + mq * 64 + nk);
        ds_lds[ds_probe_dst(mq, nk)] =
            *reinterpret_cast<const __half*>(&bits);
    }
    ins::wait_lgkm(0);
}

__device__ __forceinline__ void consume_ds_like_tile(
    const __half* __restrict__ lds,
    int consumer_slot,
    int* __restrict__ out) {
    const __half* ds_lds = lds + Lds::kDsBase / sizeof(__half);
    const __half* q_lds = lds + Lds::kQBase / sizeof(__half);
    ins::F16x8 ds0;
    ins::F16x8 ds1;
    ins::F16x8 q0;
    ins::F16x8 q1;
    ins::ds_read_matrix_32x16_trans(ds_lds, 0, ds0.f16x8);
    ins::ds_read_matrix_32x16_trans(ds_lds, kMatrixBlockBytes, ds1.f16x8);
    ins::ds_read_matrix_32x16_trans(
        q_lds, consumer_slot * kMatrixBlockBytes, q0.f16x8);
    ins::ds_read_matrix_32x16_trans(
        q_lds, consumer_slot * kMatrixBlockBytes + 1024, q1.f16x8);
    ins::wait_lgkm(0);

    ins::F32x4 acc;
    ins::zero_vgpr2(acc.u64[0]);
    ins::zero_vgpr2(acc.u64[1]);
    acc.f32 = ins::mmac_f16_lit(ds0.f16x4[0], q0.f16x4[0], acc.f32);
    acc.f32 = ins::mmac_f16_lit(ds1.f16x4[1], q1.f16x4[1], acc.f32);
    ins::keep_accumulator_live(acc);
    if (lane_id() == 0) {
        atomicAdd(out + 2, 1);
    }
}

__global__ void __launch_bounds__(768, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
dq_qsubtile_ds_consumer_probe_kernel(const __half* __restrict__ q,
                                     const __half* __restrict__ dout,
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
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Filled, 1);
        }
        if (lane_id() == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 8 && wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(48);
        const int worker_slot = static_cast<int>(wave_id - 8);
        const int lane = lane_id();
        int filled_phase = 0;
        int qdo_worker_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            ins::abarrier_try_wait<true>(Bar::kPage0Filled, filled_phase);
            skew_worker_progress(worker_slot);
            publish_ds_like_tile(lds, worker_slot, lane, q_sub);
            ins::abarrier_seq<false>(Bar::kPage0DsFilled);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0DsFilled, 1);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
            if (q_sub == 0) {
                ins::abarrier_arrive_cnt<false>(Bar::kQDoUsed, 1);
                ins::abarrier_try_wait<true>(
                    Bar::kQDoUsed, qdo_worker_phase);
            }
        }
        if (lane_id() == 0) {
            atomicAdd(out + 1, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 4 && wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(48);
        const int consumer_slot = static_cast<int>(wave_id - 4);
        int ds_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            (void)q_sub;
            ins::abarrier_try_wait<true>(Bar::kPage0DsFilled, ds_phase);
            consume_ds_like_tile(lds, consumer_slot, out);
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
        __builtin_hcu_s_abarrier_inv(Bar::kQDoUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)dout;
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

void fill_matrix(std::vector<uint16_t>& dst, uint16_t base_bits) {
    for (int r = 0; r < 2 * kSubMq; ++r) {
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
    fill_matrix(h_q, kQBaseBits);
    fill_matrix(h_dout, kDoutBaseBits);

    __half* d_q = nullptr;
    __half* d_dout = nullptr;
    int* d_out = nullptr;
    int h_out[3] = {0, 0, 0};
    check_hip(hipMalloc(&d_q, h_q.size() * sizeof(uint16_t)), "hipMalloc q");
    check_hip(hipMalloc(&d_dout, h_dout.size() * sizeof(uint16_t)),
              "hipMalloc dout");
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc out");
    check_hip(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemcpy(d_dout, h_dout.data(),
                        h_dout.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy dout");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset out");

    hipLaunchKernelGGL(dq_qsubtile_ds_consumer_probe_kernel,
                       dim3(1), dim3(768), 0, 0, d_q, d_dout, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy out");
    hipFree(d_out);
    hipFree(d_dout);
    hipFree(d_q);
    const int pass = h_out[0] == 0 && h_out[1] == 12 && h_out[2] == 8;
    std::printf("dq_qsubtile_ds_consumer_probe errors=%d done_waves=%d "
                "consumer_epochs=%d pass=%d\n",
                h_out[0], h_out[1], h_out[2], pass);
    return pass ? 0 : 2;
}
