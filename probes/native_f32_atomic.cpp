#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kElements = 16;
constexpr int kThreads = 64;

__global__ void native_global_atomic_fadd_probe(float* output) {
#if defined(__gfx946__)
    const int lane = static_cast<int>(threadIdx.x);
    if (lane < kThreads) {
        (void)__builtin_hcu_global_atomic_fadd_f32(
            output + (lane & (kElements - 1)), 1.0f);
    }
#else
    (void)output;
#endif
}

bool hip_ok(hipError_t status, const char* what) {
    if (status == hipSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    float* device = nullptr;
    std::vector<float> host(kElements, 0.0f);
    if (!hip_ok(hipMalloc(&device, host.size() * sizeof(float)), "hipMalloc") ||
        !hip_ok(hipMemset(device, 0, host.size() * sizeof(float)), "hipMemset")) {
        return 1;
    }

    hipLaunchKernelGGL(native_global_atomic_fadd_probe, dim3(1),
                       dim3(kThreads), 0, 0, device);
    bool launch_ok = hip_ok(hipGetLastError(), "kernel launch") &&
                     hip_ok(hipDeviceSynchronize(), "kernel sync") &&
                     hip_ok(hipMemcpy(host.data(), device,
                                     host.size() * sizeof(float),
                                     hipMemcpyDeviceToHost),
                            "hipMemcpy");
    launch_ok = hip_ok(hipFree(device), "hipFree") && launch_ok;

    int mismatches = 0;
    for (int i = 0; i < kElements; ++i) {
        if (std::fabs(host[i] - 4.0f) > 1.0e-6f) {
            ++mismatches;
        }
    }
    std::printf("native_f32_atomic values=%d expected=4 mismatches=%d pass=%d\n",
                kElements, mismatches, launch_ok && mismatches == 0);
    return launch_ok && mismatches == 0 ? 0 : 1;
}
