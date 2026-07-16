#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kBarrierId = 0;
constexpr int kWaveSize = 64;
constexpr int kResultWords = 7;

enum ResultIndex {
    kBeforeFirst = 0,
    kBeforeSecond,
    kAfterArrive,
    kAfterTryWait,
    kNextPhase,
    kTryWaitTargetPhase,
    kProbeReachedEnd,
};

template <int Phase>
__device__ __forceinline__ int abarrier_test_wait() {
    int state;
    asm volatile("s_abarrier_test_wait %0, %1, %2\n"
                 : "=s"(state)
                 : "n"(kBarrierId), "n"(Phase)
                 :);
    return state;
}

__global__ void __launch_bounds__(kWaveSize, 1)
abarrier_test_wait_semantics_kernel(int* result) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t wave = __builtin_hcu_get_wave_id();
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(kBarrierId, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const int before_first = abarrier_test_wait<0>();
    const int before_second = abarrier_test_wait<0>();

    ins::abarrier_arrive_cnt<false>(kBarrierId, 1);
    const int after_arrive = abarrier_test_wait<0>();

    int target_phase = 0;
    ins::abarrier_try_wait<true>(kBarrierId, target_phase);
    const int after_try_wait = abarrier_test_wait<0>();
    const int next_phase = abarrier_test_wait<1>();

    if (lane == 0) {
        result[kBeforeFirst] = before_first;
        result[kBeforeSecond] = before_second;
        result[kAfterArrive] = after_arrive;
        result[kAfterTryWait] = after_try_wait;
        result[kNextPhase] = next_phase;
        result[kTryWaitTargetPhase] = target_phase;
        result[kProbeReachedEnd] = 1;
    }

    __syncthreads();
    if (wave == 0) {
        __builtin_hcu_s_abarrier_inv(kBarrierId);
    }
#else
    (void)result;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
        std::exit(2);
    }
}

}  // namespace

int main() {
    int* device_result = nullptr;
    std::array<int, kResultWords> host_result{};
    check_hip(hipMalloc(&device_result, sizeof(host_result)),
              "hipMalloc result");
    check_hip(hipMemset(device_result, 0, sizeof(host_result)),
              "hipMemset result");

    hipLaunchKernelGGL(abarrier_test_wait_semantics_kernel,
                       dim3(1), dim3(kWaveSize), 0, 0, device_result);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");
    check_hip(hipMemcpy(host_result.data(), device_result,
                        sizeof(host_result), hipMemcpyDeviceToHost),
              "hipMemcpy result");
    check_hip(hipFree(device_result), "hipFree result");

    const bool repeat_is_stable =
        host_result[kBeforeFirst] == host_result[kBeforeSecond];
    const bool completed_is_stable =
        host_result[kAfterArrive] == host_result[kAfterTryWait];
    const bool phase_is_distinguishable =
        host_result[kAfterTryWait] != host_result[kNextPhase];
    const bool try_wait_toggled_phase =
        host_result[kTryWaitTargetPhase] == 1;
    const bool reached_end = host_result[kProbeReachedEnd] == 1;
    const bool pass = repeat_is_stable && completed_is_stable &&
                      phase_is_distinguishable && try_wait_toggled_phase &&
                      reached_end;

    std::printf(
        "before_first=%d before_second=%d after_arrive=%d "
        "after_try_wait=%d next_phase=%d try_wait_target_phase=%d "
        "reached_end=%d\n",
        host_result[kBeforeFirst], host_result[kBeforeSecond],
        host_result[kAfterArrive], host_result[kAfterTryWait],
        host_result[kNextPhase], host_result[kTryWaitTargetPhase],
        host_result[kProbeReachedEnd]);
    std::printf(
        "repeat_stable=%d completed_stable=%d phase_distinguishable=%d "
        "try_wait_toggled=%d pass=%d\n",
        repeat_is_stable, completed_is_stable, phase_is_distinguishable,
        try_wait_toggled_phase, pass);
    return pass ? 0 : 1;
}
