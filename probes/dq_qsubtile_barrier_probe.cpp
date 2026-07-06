#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

struct Bar {
    static constexpr int kPage0Filled = 0;
    static constexpr int kPage0DsFilled = 1;
    static constexpr int kPage0Used = 2;
    static constexpr int kQDoUsed = 3;
    static constexpr int kAllDone = 4;
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x & 63);
}

__global__ void __launch_bounds__(768, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
dq_qsubtile_barrier_probe_kernel(int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ int qdo_value;
    __shared__ int page_value;
    __shared__ int ds_value;
    const uint32_t wave_id = __builtin_hcu_get_wave_id();

    if (wave_id == 0) {
        const int lane = lane_id();
        __builtin_hcu_s_abarrier_init(Bar::kPage0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0DsFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0Used, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQDoUsed, 4);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 12);
        if (lane == 0) {
            qdo_value = -1;
            page_value = -1;
            ds_value = -1;
            out[0] = 0;
            out[1] = 0;
        }
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(16);
        const int lane = lane_id();
        int qdo_phase = 0;
        int page_used_phase = 0;
        int page0_seen = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            if (q_sub > 0) {
                ins::abarrier_try_wait<true>(Bar::kQDoUsed, qdo_phase);
            }
            if (lane == 0 && wave_id == 0) {
                qdo_value = 100 + q_sub;
            }
            if (page0_seen != 0) {
                ins::abarrier_try_wait<true>(
                    Bar::kPage0Used, page_used_phase);
            }
            page0_seen = 1;
            ins::abarrier_seq<false>(Bar::kPage0Filled);
            if (lane == 0 && wave_id == 0) {
                page_value = 200 + q_sub;
            }
            ins::wait_lgkm(0);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Filled, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 8 && wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(32);
        const int lane = lane_id();
        int page_filled_phase = 0;
        const int worker = static_cast<int>(wave_id - 8);
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            ins::abarrier_try_wait<true>(
                Bar::kPage0Filled, page_filled_phase);
            if (lane == 0) {
                if (qdo_value != 100 + q_sub ||
                    page_value != 200 + q_sub) {
                    atomicAdd(out, 1);
                }
                if (worker == 0) {
                    ds_value = 300 + q_sub;
                }
            }
            ins::wait_lgkm(0);
            ins::abarrier_seq<false>(Bar::kPage0DsFilled);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0DsFilled, 1);
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
            ins::abarrier_arrive_cnt<false>(Bar::kQDoUsed, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else if (wave_id >= 4 && wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(32);
        const int lane = lane_id();
        int ds_phase = 0;
        for (int q_sub = 0; q_sub < 2; ++q_sub) {
            ins::abarrier_try_wait<true>(Bar::kPage0DsFilled, ds_phase);
            if (lane == 0) {
                if (page_value != 200 + q_sub ||
                    ds_value != 300 + q_sub) {
                    atomicAdd(out, 1);
                }
            }
            ins::abarrier_arrive_cnt<false>(Bar::kPage0Used, 1);
        }
        ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    } else {
        __builtin_hcu_s_set_vgpr_size(16);
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
        out[1] = 1;
    }
#else
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace

int main() {
    int* d_out = nullptr;
    int h_out[2] = {0, 0};
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc out");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset out");
    hipLaunchKernelGGL(dq_qsubtile_barrier_probe_kernel,
                       dim3(1), dim3(768), 0, 0, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy out");
    hipFree(d_out);
    std::printf("dq_qsubtile_barrier_probe errors=%d done=%d pass=%d\n",
                h_out[0], h_out[1], h_out[0] == 0 && h_out[1] == 1);
    return h_out[0] == 0 && h_out[1] == 1 ? 0 : 2;
}
