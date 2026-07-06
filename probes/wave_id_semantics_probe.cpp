#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

__global__ void __launch_bounds__(768, 1)
    __attribute__((hcu_wdra_waves_per_tg(12)))
wave_id_semantics_probe_kernel(int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t hw_wave_id = __builtin_hcu_get_wave_id();
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 63;
    const int cta_wave = tid >> 6;
    if (lane == 0 && cta_wave < 12) {
        out[cta_wave] = static_cast<int>(hw_wave_id);
        out[16 + cta_wave] = cta_wave;
    }
#else
    (void)out;
#endif
}

}  // namespace

int main() {
    int* d_out = nullptr;
    int h_out[32];
    for (int& v : h_out) {
        v = -1;
    }
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc out");
    check_hip(
        hipMemcpy(d_out, h_out, sizeof(h_out), hipMemcpyHostToDevice),
        "hipMemcpy init");
    hipLaunchKernelGGL(wave_id_semantics_probe_kernel,
                       dim3(1), dim3(768), 0, 0, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(
        hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
        "hipMemcpy out");
    hipFree(d_out);

    std::printf("wave_id_semantics hw_by_cta_wave=");
    for (int i = 0; i < 12; ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", h_out[i]);
    }
    std::printf(" cta_wave_echo=");
    for (int i = 0; i < 12; ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", h_out[16 + i]);
    }
    std::printf("\n");
    return 0;
}
