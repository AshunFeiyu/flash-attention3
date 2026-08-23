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
constexpr int kRows = 32;
constexpr int kCols = 16;
constexpr int kElems = kRows * kCols;
constexpr int kLdsPageElems = 64 * 16;
constexpr uint16_t kPatternBase = 0x3000;
constexpr uint16_t kPoison = 0xfefe;
constexpr int kBarrier = 0;

#ifndef SHAOBO_PROBE_ENABLE_VWCNT
#define SHAOBO_PROBE_ENABLE_VWCNT 0
#endif

enum CompletionMode {
    kVmcnt = 0,
    kVwcntVmcnt = 1,
    kAbarrierVmcnt = 2,
    kAbarrierVwcntVmcnt = 3,
    kRtnVmcnt = 4,
    kRtnVwcntVmcnt = 5,
    kGlcVmcnt = 6,
    kSlcVmcnt = 7,
    kGlcSlcVmcnt = 8,
    kWbinvl1Vmcnt = 9,
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

__device__ __forceinline__ void wait_vwcnt0() {
#if SHAOBO_PROBE_ENABLE_VWCNT && \
    (defined(__gfx946__) || defined(__gfx92a__))
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt_vwcnt 0\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
}

__device__ __forceinline__ void store_void(ins::Vec4U32 dst,
                                            _Float16* lds) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_hcu_matrix_store_32x16_b16(
        dst, reinterpret_cast<short*>(lds), 0, false, false, false, false);
#else
    (void)dst;
    (void)lds;
#endif
}

__device__ __forceinline__ void store_policy(int mode,
                                              ins::Vec4U32 dst,
                                              _Float16* lds) {
#if defined(__gfx946__) || defined(__gfx92a__)
    if (mode == kGlcVmcnt) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0, false, false, true,
            false);
    } else if (mode == kSlcVmcnt) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0, false, false, false,
            true);
    } else if (mode == kGlcSlcVmcnt) {
        __builtin_hcu_matrix_store_32x16_b16(
            dst, reinterpret_cast<short*>(lds), 0, false, false, true, true);
    } else {
        store_void(dst, lds);
    }
#else
    (void)mode;
    (void)dst;
    (void)lds;
#endif
}

__device__ __forceinline__ ins::Vec4U32 store_rtn(ins::Vec4U32 dst,
                                                   _Float16* lds) {
#if defined(__gfx946__) || defined(__gfx92a__)
    return __builtin_hcu_matrix_store_32x16_b16_rtn(
        dst, reinterpret_cast<short*>(lds), 0, false, false, false, false);
#else
    (void)lds;
    return dst;
#endif
}

__global__ void __launch_bounds__(kWaveSize, 1)
    matrix_store_completion_probe_kernel(
        const _Float16* __restrict__ input,
        _Float16* __restrict__ output,
        uint32_t* __restrict__ rtn_dump,
        int mode) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 lds[kLdsPageElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    if (mode >= kAbarrierVmcnt && mode <= kAbarrierVwcntVmcnt) {
        __builtin_hcu_s_abarrier_init(kBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    const ins::Vec4U32 src = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(input), kCols);
    const ins::Vec4U32 dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output), kCols);
    __builtin_hcu_matrix_load_32x16_b16(
        src, reinterpret_cast<short*>(lds), 0, false, false, false, false,
        true);
    ins::wait_vbcnt0();
    ins::wait_lgkm(0);

    ins::Vec4U32 rtn{};
    if (mode == kVmcnt) {
        store_void(dst, lds);
        ins::wait_vmem_lgkm();
    } else if (mode == kVwcntVmcnt) {
        store_void(dst, lds);
        wait_vwcnt0();
        ins::wait_vmem_lgkm();
    } else if (mode == kAbarrierVmcnt ||
               mode == kAbarrierVwcntVmcnt) {
        int phase = 0;
        ins::abarrier_seq<false>(kBarrier);
        store_void(dst, lds);
        ins::abarrier_arrive_cnt<false>(kBarrier, 1);
        ins::abarrier_try_wait<false>(kBarrier, phase);
        if (mode == kAbarrierVwcntVmcnt) {
            wait_vwcnt0();
        }
        ins::wait_vmem_lgkm();
    } else if (mode == kRtnVmcnt || mode == kRtnVwcntVmcnt) {
        rtn = store_rtn(dst, lds);
        if (mode == kRtnVwcntVmcnt) {
            wait_vwcnt0();
        }
        ins::wait_vmem_lgkm();
    } else if (mode >= kGlcVmcnt && mode <= kWbinvl1Vmcnt) {
        store_policy(mode, dst, lds);
        ins::wait_vmem_lgkm();
        if (mode == kWbinvl1Vmcnt) {
            __builtin_amdgcn_buffer_wbinvl1_vol();
            ins::wait_vmem_lgkm();
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            rtn_dump[i] = rtn[i];
        }
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (mode >= kAbarrierVmcnt && mode <= kAbarrierVwcntVmcnt) {
        __builtin_hcu_s_abarrier_inv(kBarrier);
    }
#else
    (void)input;
    (void)output;
    (void)rtn_dump;
    (void)mode;
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

int main(int argc, char** argv) {
    const char* mode_text = argc == 2
                                ? argv[1]
                                : std::getenv("MATRIX_STORE_COMPLETION_MODE");
    if (mode_text == nullptr) {
        std::fprintf(
            stderr,
            "usage: %s <mode 0..9> or set MATRIX_STORE_COMPLETION_MODE\n",
            argv[0]);
        return 2;
    }
    const int mode = std::atoi(mode_text);
    if (mode < kVmcnt || mode > kWbinvl1Vmcnt) {
        std::fprintf(stderr, "invalid mode=%d\n", mode);
        return 2;
    }

    std::vector<uint16_t> expected_bits(kElems);
    std::vector<_Float16> input(kElems);
    for (int i = 0; i < kElems; ++i) {
        expected_bits[i] = static_cast<uint16_t>(kPatternBase + i);
        input[i] = half_from_bits(expected_bits[i]);
    }
    std::vector<uint16_t> poison(kElems, kPoison);

    _Float16* input_device = nullptr;
    _Float16* output_device = nullptr;
    uint32_t* rtn_device = nullptr;
    check_hip(hipMalloc(&input_device, kElems * sizeof(_Float16)),
              "hipMalloc input");
    check_hip(hipMalloc(&output_device, kElems * sizeof(_Float16)),
              "hipMalloc output");
    check_hip(hipMalloc(&rtn_device, 4 * sizeof(uint32_t)), "hipMalloc rtn");
    check_hip(hipMemcpy(input_device, input.data(), kElems * sizeof(_Float16),
                        hipMemcpyHostToDevice),
              "copy input");
    check_hip(hipMemcpy(output_device, poison.data(),
                        kElems * sizeof(uint16_t), hipMemcpyHostToDevice),
              "poison output");
    check_hip(hipMemset(rtn_device, 0, 4 * sizeof(uint32_t)), "clear rtn");

    hipLaunchKernelGGL(matrix_store_completion_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, input_device, output_device,
                       rtn_device, mode);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "synchronize");

    std::vector<uint16_t> actual(kElems);
    uint32_t rtn[4]{};
    check_hip(hipMemcpy(actual.data(), output_device,
                        kElems * sizeof(uint16_t), hipMemcpyDeviceToHost),
              "copy output");
    check_hip(hipMemcpy(rtn, rtn_device, sizeof(rtn), hipMemcpyDeviceToHost),
              "copy rtn");

    int mismatches = 0;
    int poison_count = 0;
    int first = -1;
    for (int i = 0; i < kElems; ++i) {
        if (actual[i] != expected_bits[i]) {
            ++mismatches;
            if (first < 0) {
                first = i;
            }
        }
        poison_count += actual[i] == kPoison;
    }
    std::printf(
        "matrix_store_completion mode=%d mismatches=%d poison=%d "
        "first_row=%d first_col=%d first_expected=0x%04x "
        "first_actual=0x%04x rtn=%08x,%08x,%08x,%08x pass=%d\n",
        mode, mismatches, poison_count, first < 0 ? -1 : first / kCols,
        first < 0 ? -1 : first % kCols,
        first < 0 ? 0 : expected_bits[first], first < 0 ? 0 : actual[first],
        rtn[0], rtn[1], rtn[2], rtn[3], mismatches == 0 ? 1 : 0);

    check_hip(hipFree(rtn_device), "free rtn");
    check_hip(hipFree(output_device), "free output");
    check_hip(hipFree(input_device), "free input");
    return 0;
}
