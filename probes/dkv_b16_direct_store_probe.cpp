#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Vec4F16 =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using Vec2F16 =
    __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;

constexpr int kWaveSize = 64;
constexpr int kWaves = 16;
constexpr int kThreads = kWaveSize * kWaves;
constexpr int kRows = 16;
constexpr int kCols = 128;
constexpr int kElems = kRows * kCols;
constexpr float kTolerance = 0.0f;

union F16x4 {
    Vec4F16 vec;
    Vec2F16 pair[2];
    _Float16 scalar[4];
};

__global__ void __launch_bounds__(kThreads, 1)
    dkv_b16_direct_store_probe_kernel(__half* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    if (wave_id != 0) {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int owner_row = lane & 15;
    const int lane_col_group = lane >> 4;
    const int row_base = owner_row * kCols;

#pragma unroll
    for (int d_idx = 0; d_idx < 8; ++d_idx) {
        const int col_base = d_idx * 16 + lane_col_group * 4;
        F16x4 packed{};
        const float value0 = static_cast<float>(1 + row_base + col_base);
        const float value1 = value0 + 1.0f;
        const float value2 = value0 + 2.0f;
        const float value3 = value0 + 3.0f;
        packed.pair[0] = __builtin_hcu_cvt_pk_f16_f32(
            value0, value1, false, 0);
        packed.pair[1] = __builtin_hcu_cvt_pk_f16_f32(
            value2, value3, false, 0);
        *reinterpret_cast<Vec4F16*>(output + row_base + col_base) = packed.vec;
    }
#else
    (void)output;
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
    __half* output_device = nullptr;
    check_hip(hipMalloc(&output_device, kElems * sizeof(__half)),
              "hipMalloc output");
    check_hip(hipMemset(output_device, 0, kElems * sizeof(__half)),
              "clear output");

    hipLaunchKernelGGL(dkv_b16_direct_store_probe_kernel, dim3(1),
                       dim3(kThreads), 0, 0, output_device);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");

    std::vector<_Float16> actual(kElems);
    check_hip(hipMemcpy(actual.data(), output_device,
                        kElems * sizeof(__half), hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipFree(output_device), "free output");

    int mismatches = 0;
    float max_abs = 0.0f;
    int first_row = -1;
    int first_col = -1;
    float first_got = 0.0f;
    float first_want = 0.0f;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const int index = row * kCols + col;
            const float got = static_cast<float>(actual[index]);
            const float want = static_cast<float>(
                static_cast<_Float16>(1 + index));
            const float diff = std::fabs(got - want);
            max_abs = std::max(max_abs, diff);
            if (diff > kTolerance) {
                ++mismatches;
                if (first_row < 0) {
                    first_row = row;
                    first_col = col;
                    first_got = got;
                    first_want = want;
                }
            }
        }
    }

    std::printf(
        "dkv_b16_direct_store rows=%d cols=%d mismatches=%d max_abs=%g "
        "first_row=%d first_col=%d first_got=%g first_want=%g pass=%d\n",
        kRows, kCols, mismatches, max_abs, first_row, first_col, first_got,
        first_want, mismatches == 0 ? 1 : 0);
    return mismatches == 0 ? 0 : 1;
}
