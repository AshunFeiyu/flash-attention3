#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kMaxElems = 64 * 16;
constexpr int kLdsElems = 2 * kMaxElems;
constexpr uint16_t kPatternBase = 0x3000;
constexpr uint16_t kPoison = 0xfefe;

enum ShapeMode {
    k32x16 = 0,
    k64x16 = 1,
    k32x32 = 2,
};

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

__host__ _Float16 half_from_bits(uint16_t bits) {
    HalfBits value{};
    value.bits = bits;
    return value.value;
}

__device__ __forceinline__ void load_shape(
    int shape, ins::Vec4U32 src, _Float16* lds) {
#if defined(__gfx946__) || defined(__gfx92a__)
    if (shape == k32x16) {
        __builtin_hcu_matrix_load_32x16_b16(
            src, reinterpret_cast<short*>(lds), 0, false, false, false,
            false, true);
    } else if (shape == k64x16) {
        __builtin_hcu_matrix_load_64x16_b16(
            src, reinterpret_cast<short*>(lds), 0, false, false, false,
            false, true);
    } else {
        __builtin_hcu_matrix_load_32x32_b16(
            src, reinterpret_cast<short*>(lds), 0, false, false, false,
            false, true);
    }
#else
    (void)shape;
    (void)src;
    (void)lds;
#endif
}

__device__ __forceinline__ void store_shape(
    int shape, ins::Vec4U32 dst, _Float16* lds) {
#if defined(__gfx946__) || defined(__gfx92a__)
    if (shape == k32x16) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0, false, false, false,
            false);
    } else if (shape == k64x16) {
        __builtin_hcu_matrix_store_64x16_b16(
            dst, reinterpret_cast<short*>(lds), 0, false, false, false,
            false);
    } else {
        __builtin_hcu_matrix_store_32x32_b16(
            dst, reinterpret_cast<short*>(lds), 0, false, false, false,
            false);
    }
#else
    (void)shape;
    (void)dst;
    (void)lds;
#endif
}

__global__ void __launch_bounds__(kWaveSize, 1)
    matrix_store_shape_family_probe_kernel(
        const _Float16* __restrict__ input,
        _Float16* __restrict__ output,
        int shape,
        int stride) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 lds[kLdsElems];
    const ins::Vec4U32 src = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(input), stride);
    const ins::Vec4U32 dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output), stride);
    load_shape(shape, src, lds);
    ins::wait_vbcnt0();
    ins::wait_lgkm(0);
    store_shape(shape, dst, lds);
    ins::wait_vmem_lgkm();
#else
    (void)input;
    (void)output;
    (void)shape;
    (void)stride;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     hipGetErrorString(error));
        std::exit(2);
    }
}

}  // namespace

int main() {
    const char* shape_text = std::getenv("MATRIX_STORE_SHAPE_MODE");
    if (shape_text == nullptr) {
        std::fprintf(stderr, "set MATRIX_STORE_SHAPE_MODE=0,1,2\n");
        return 2;
    }
    const int shape = std::atoi(shape_text);
    if (shape < k32x16 || shape > k32x32) {
        std::fprintf(stderr, "invalid shape=%d\n", shape);
        return 2;
    }
    const int rows = shape == k64x16 ? 64 : 32;
    const int cols = shape == k32x32 ? 32 : 16;
    const int elems = rows * cols;

    std::vector<uint16_t> expected_bits(elems);
    std::vector<_Float16> input(elems);
    std::vector<uint16_t> poison(elems, kPoison);
    for (int i = 0; i < elems; ++i) {
        expected_bits[i] = static_cast<uint16_t>(kPatternBase + i);
        input[i] = half_from_bits(expected_bits[i]);
    }

    _Float16* input_device = nullptr;
    _Float16* output_device = nullptr;
    check_hip(hipMalloc(&input_device, elems * sizeof(_Float16)),
              "hipMalloc input");
    check_hip(hipMalloc(&output_device, elems * sizeof(_Float16)),
              "hipMalloc output");
    check_hip(hipMemcpy(input_device, input.data(), elems * sizeof(_Float16),
                        hipMemcpyHostToDevice),
              "copy input");
    check_hip(hipMemcpy(output_device, poison.data(),
                        elems * sizeof(uint16_t), hipMemcpyHostToDevice),
              "poison output");

    hipLaunchKernelGGL(matrix_store_shape_family_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, input_device, output_device,
                       shape, cols);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");

    std::vector<uint16_t> actual(elems);
    check_hip(hipMemcpy(actual.data(), output_device,
                        elems * sizeof(uint16_t), hipMemcpyDeviceToHost),
              "copy output");

    int mismatches = 0;
    int poison_count = 0;
    int first = -1;
    for (int i = 0; i < elems; ++i) {
        if (actual[i] != expected_bits[i]) {
            ++mismatches;
            if (first < 0) {
                first = i;
            }
        }
        poison_count += actual[i] == kPoison;
    }
    std::printf(
        "matrix_store_shape shape=%dx%d mismatches=%d poison=%d "
        "first_row=%d first_col=%d first_expected=0x%04x "
        "first_actual=0x%04x pass=%d\n",
        rows, cols, mismatches, poison_count, first < 0 ? -1 : first / cols,
        first < 0 ? -1 : first % cols,
        first < 0 ? 0 : expected_bits[first], first < 0 ? 0 : actual[first],
        mismatches == 0 ? 1 : 0);

    check_hip(hipFree(output_device), "free output");
    check_hip(hipFree(input_device), "free input");
    return 0;
}
