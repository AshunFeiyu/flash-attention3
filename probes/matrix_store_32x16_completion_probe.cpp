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
constexpr int kCols = 32;
constexpr int kElems = kRows * kCols;
constexpr int kModes = 7;
constexpr int kLdsHalfs = 1024;
constexpr int kBarrier = 0;
constexpr uint16_t kPatternBase = 0x3000;
constexpr uint16_t kPoison = 0xfefe;

enum Mode {
    kVmcnt,
    kAbarrier,
    kReturn,
    kGlc,
    kSlc,
    kGlcSlc,
    kWbinvl1,
};

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

_Float16 half_from_bits(uint16_t bits) {
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

__device__ __forceinline__ void store_matrix(
    int mode, ins::Vec4U32 dst, _Float16* lds) {
    if (mode == kReturn) {
        (void)__builtin_hcu_matrix_store_32x16_b16_rtn(
            dst, reinterpret_cast<short*>(lds), 0,
            true, false, false, false);
    } else if (mode == kGlc) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0,
            true, false, true, false);
    } else if (mode == kSlc) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0,
            true, false, false, true);
    } else if (mode == kGlcSlc) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0,
            true, false, true, true);
    } else {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0,
            true, false, false, false);
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
matrix_store_32x16_completion_kernel(
    const _Float16* input, _Float16* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(4096) _Float16 lds[kLdsHalfs];
    const int mode = static_cast<int>(blockIdx.x);
    if (mode == kAbarrier) {
        __builtin_hcu_s_abarrier_init(kBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const auto src = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(input), kCols);
    __builtin_hcu_matrix_load_32x16_b16(
        src, reinterpret_cast<short*>(lds), 0,
        true, false, false, false, true);
    ins::wait_vbcnt0();
    ins::wait_lgkm(0);

    const auto dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output + mode * kElems), kCols);
    if (mode == kAbarrier) {
        int phase = 0;
        ins::abarrier_seq<false>(kBarrier);
        store_matrix(mode, dst, lds);
        ins::abarrier_arrive_cnt<false>(kBarrier, 1);
        ins::abarrier_try_wait<false>(kBarrier, phase);
    } else {
        store_matrix(mode, dst, lds);
    }
    ins::wait_vmem_lgkm();
    if (mode == kWbinvl1) {
        __builtin_amdgcn_buffer_wbinvl1_vol();
        ins::wait_vmem_lgkm();
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (mode == kAbarrier) {
        __builtin_hcu_s_abarrier_inv(kBarrier);
    }
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
    std::vector<uint16_t> poison(kModes * kElems, kPoison);

    _Float16* device_input = nullptr;
    _Float16* device_output = nullptr;
    check_hip(hipMalloc(&device_input, input.size() * sizeof(_Float16)),
              "malloc input");
    check_hip(hipMalloc(&device_output, poison.size() * sizeof(uint16_t)),
              "malloc output");
    check_hip(hipMemcpy(device_input, input.data(),
                        input.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy input");
    check_hip(hipMemcpy(device_output, poison.data(),
                        poison.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice), "poison output");

    hipLaunchKernelGGL(matrix_store_32x16_completion_kernel,
                       dim3(kModes), dim3(kWaveSize), 0, 0,
                       device_input, device_output);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    std::vector<uint16_t> output(poison.size());
    check_hip(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), "copy output");
    check_hip(hipFree(device_output), "free output");
    check_hip(hipFree(device_input), "free input");

    int direct_exact = 0;
    int return_exact = 0;
    for (int mode = 0; mode < kModes; ++mode) {
        int mismatches = 0;
        int poison_count = 0;
        for (int i = 0; i < kElems; ++i) {
            const uint16_t got = output[mode * kElems + i];
            mismatches += got != static_cast<uint16_t>(kPatternBase + i);
            poison_count += got == kPoison;
        }
        if (mode == kReturn) {
            return_exact = mismatches == 0;
        } else {
            direct_exact += mismatches == 0;
        }
        std::printf(
            "matrix_store_32x16_completion mode=%d mismatches=%d "
            "poison=%d exact=%d\n",
            mode, mismatches, poison_count, mismatches == 0 ? 1 : 0);
    }
    std::printf(
        "matrix_store_32x16_completion_status=%s direct_exact=%d/6 "
        "rtn_exact=%d/1 rtn_status=%s\n",
        direct_exact == 6 ? "PASS" : "FAIL", direct_exact,
        return_exact, return_exact ? "PASS" : "OBSERVE");
    return direct_exact == 6 ? 0 : 1;
}
