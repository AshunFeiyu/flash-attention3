#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 16;
constexpr int kCols = 64;
constexpr int kElems = kRows * kCols;
constexpr int kModes = 1;
constexpr int kLdsHalfs = 4096;
constexpr uint16_t kPatternBase = 0x3000;
constexpr uint16_t kPoison = 0xfefe;

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

__host__ __device__ _Float16 half_from_bits(uint16_t bits) {
    HalfBits value{};
    value.bits = bits;
    return value.value;
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__device__ __forceinline__ void matrix_load(
    int, ins::Vec4U32 src, _Float16* lds) {
    __builtin_hcu_matrix_load_64x16_b16(
        src, reinterpret_cast<short*>(lds), 0,
        true, false, false, false, true);
}

__device__ __forceinline__ void matrix_store(
    int, ins::Vec4U32 dst, _Float16* lds) {
    __builtin_hcu_matrix_store_64x16_b16(
        dst, reinterpret_cast<short*>(lds), 0,
        true, false, false, false);
}

__global__ void __launch_bounds__(kWaveSize, 1)
matrix_store_64x16_direct_kernel(
    const _Float16* input, _Float16* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(4096) _Float16 lds[kLdsHalfs];
    const int mode = static_cast<int>(blockIdx.x);
    const auto src = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(input), kCols);
    matrix_load(mode, src, lds);
    ins::wait_vbcnt0();
    ins::wait_lgkm(0);

    const auto dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output + mode * kElems), kCols);
    matrix_store(mode, dst, lds);
    ins::wait_vmem_lgkm();
#else
    (void)input;
    (void)output;
#endif
}

}  // namespace

int main() {
    std::vector<_Float16> input(kElems);
    for (int i = 0; i < kElems; ++i) {
        input[i] = half_from_bits(static_cast<uint16_t>(kPatternBase + i));
    }

    _Float16* device_input = nullptr;
    _Float16* device_output = nullptr;
    check_hip(hipMalloc(&device_input, kElems * sizeof(_Float16)),
              "malloc input");
    check_hip(hipMalloc(&device_output, kModes * kElems * sizeof(_Float16)),
              "malloc output");
    check_hip(hipMemcpy(device_input, input.data(),
                        kElems * sizeof(_Float16), hipMemcpyHostToDevice),
              "copy input");
    std::vector<uint16_t> poison(kModes * kElems, kPoison);
    check_hip(hipMemcpy(device_output, poison.data(),
                        poison.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice), "poison output");

    hipLaunchKernelGGL(matrix_store_64x16_direct_kernel,
                       dim3(kModes), dim3(kWaveSize), 0, 0,
                       device_input, device_output);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    std::vector<uint16_t> output(kModes * kElems);
    check_hip(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), "copy output");
    check_hip(hipFree(device_output), "free output");
    check_hip(hipFree(device_input), "free input");

    int exact = 0;
    for (int mode = 0; mode < kModes; ++mode) {
        int mismatches = 0;
        int poison_count = 0;
        int first = -1;
        for (int i = 0; i < kElems; ++i) {
            const uint16_t got = output[mode * kElems + i];
            poison_count += got == kPoison;
            if (got != static_cast<uint16_t>(kPatternBase + i)) {
                ++mismatches;
                if (first < 0) first = i;
            }
        }
        exact += mismatches == 0;
        std::printf(
            "matrix_store_64x16_direct mode=t%dr%d mismatches=%d "
            "poison=%d first_row=%d first_col=%d exact=%d\n",
            1, 0, mismatches, poison_count,
            first < 0 ? -1 : first / kCols,
            first < 0 ? -1 : first % kCols,
            mismatches == 0 ? 1 : 0);
    }
    std::printf(
        "matrix_store_64x16_direct_status=%s exact=%d/%d\n",
        exact > 0 ? "PASS" : "FAIL", exact, kModes);
    return exact > 0 ? 0 : 1;
}
