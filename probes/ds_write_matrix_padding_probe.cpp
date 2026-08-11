#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kPageBytes = 2048;
constexpr int kPageHalfs = kPageBytes / sizeof(uint16_t);
constexpr uint16_t kSentinel = 0x7bff;

union Fragment {
    ins::Vec8F16 f16x8;
    _Float16 scalar[8];
};

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__global__ void ds_write_matrix_padding_probe_kernel(uint16_t* out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) uint16_t page[kPageHalfs];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    for (int i = lane; i < kPageHalfs; i += kWaveSize) {
        page[i] = kSentinel;
    }
    __syncthreads();

    Fragment fragment{};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        fragment.scalar[word] = static_cast<_Float16>(
            1.0f + static_cast<float>(lane) * 0.125f +
            static_cast<float>(word) * 0.0078125f);
    }
    ins::ds_write_matrix_32x16_trans_f16(
        fragment.f16x8, reinterpret_cast<__half*>(page), 0);
    ins::wait_lgkm(0);
    __syncthreads();

    for (int i = lane; i < kPageHalfs; i += kWaveSize) {
        out[i] = page[i];
    }
#else
    (void)out;
#endif
}

}  // namespace

int main() {
    std::vector<uint16_t> host(kPageHalfs, 0);
    uint16_t* device = nullptr;
    check_hip(hipMalloc(&device, kPageBytes), "hipMalloc");

    hipLaunchKernelGGL(ds_write_matrix_padding_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, device);
    check_hip(hipGetLastError(), "kernel launch");
    check_hip(hipDeviceSynchronize(), "kernel sync");
    check_hip(hipMemcpy(host.data(), device, kPageBytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy");

    int touched = 0;
    int first_touched = kPageHalfs;
    int last_touched = -1;
    int longest_untouched = 0;
    int longest_begin = -1;
    int run_begin = -1;
    for (int i = 0; i <= kPageHalfs; ++i) {
        const bool is_untouched = i < kPageHalfs && host[i] == kSentinel;
        if (is_untouched && run_begin < 0) {
            run_begin = i;
        }
        if (!is_untouched && run_begin >= 0) {
            const int length = i - run_begin;
            if (length > longest_untouched) {
                longest_untouched = length;
                longest_begin = run_begin;
            }
            run_begin = -1;
        }
        if (i < kPageHalfs && host[i] != kSentinel) {
            ++touched;
            first_touched = std::min(first_touched, i);
            last_touched = i;
        }
    }

    std::printf(
        "ds_write_padding touched_halfs=%d first_byte=%d last_byte=%d "
        "longest_untouched_begin_byte=%d longest_untouched_bytes=%d\n",
        touched, first_touched * 2, last_touched * 2,
        longest_begin * 2, longest_untouched * 2);

    int range_begin = -1;
    for (int i = 0; i <= kPageHalfs; ++i) {
        const bool is_touched = i < kPageHalfs && host[i] != kSentinel;
        if (is_touched && range_begin < 0) {
            range_begin = i;
        }
        if (!is_touched && range_begin >= 0) {
            std::printf("touched_range_bytes=%d:%d\n", range_begin * 2,
                        i * 2);
            range_begin = -1;
        }
    }

    const bool pass = touched > 0 && touched < kPageHalfs &&
                      longest_untouched * 2 >= 768 &&
                      (longest_begin * 2) % alignof(float) == 0;
    std::printf("ds_write_matrix_padding_probe_status=%s\n",
                pass ? "PASS" : "FAIL");
    check_hip(hipFree(device), "hipFree");
    return pass ? 0 : 1;
}
