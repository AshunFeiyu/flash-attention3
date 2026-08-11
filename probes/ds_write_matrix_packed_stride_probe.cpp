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
constexpr int kWaves = 4;
constexpr int kPackedStride = 1024;
constexpr int kControlBase = 4096;
constexpr int kControlStride = 2048;
constexpr int kLdsBytes = 8192;
constexpr uint16_t kSentinel = 0x7bff;

union Fragment {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
    uint16_t bits[8];
};

union Accumulator {
    ins::Vec4F32 f32;
    uint32_t bits[4];
};

struct ProbeResult {
    int normal_mismatches;
    int trans_mismatches;
    int mmac_mismatches;
};

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__global__ void ds_write_matrix_packed_stride_probe_kernel(ProbeResult* out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) uint16_t storage[kLdsBytes / sizeof(uint16_t)];
    const int wave = static_cast<int>(threadIdx.x / kWaveSize);
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    auto* lds = reinterpret_cast<__half*>(storage);

    for (int index = static_cast<int>(threadIdx.x);
         index < kLdsBytes / static_cast<int>(sizeof(uint16_t));
         index += static_cast<int>(blockDim.x)) {
        storage[index] = kSentinel;
    }
    __syncthreads();

    Fragment source{};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        source.scalar[word] = static_cast<_Float16>(
            1.0f + static_cast<float>(wave) * 2.0f +
            static_cast<float>(lane) * 0.03125f +
            static_cast<float>(word) * 0.00390625f);
    }

    if (wave < 2) {
        ins::ds_write_matrix_32x16_trans_f16(
            source.f16x8, lds, wave * kPackedStride);
        ins::ds_write_matrix_32x16_trans_f16(
            source.f16x8, lds, kControlBase + wave * kControlStride);
        ins::wait_lgkm(0);
    }
    __syncthreads();

    if (wave >= 2) {
        const int writer = wave - 2;
        Fragment packed_normal{};
        Fragment packed_trans{};
        Fragment control_normal{};
        Fragment control_trans{};
        ins::ds_read_matrix_32x16_normal(
            lds, writer * kPackedStride, packed_normal.f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, writer * kPackedStride, packed_trans.f16x8);
        ins::ds_read_matrix_32x16_normal(
            lds, kControlBase + writer * kControlStride,
            control_normal.f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, kControlBase + writer * kControlStride,
            control_trans.f16x8);
        ins::wait_lgkm(0);

        int normal_mismatches = 0;
        int trans_mismatches = 0;
#pragma unroll
        for (int word = 0; word < 8; ++word) {
            normal_mismatches +=
                packed_normal.bits[word] != control_normal.bits[word];
            trans_mismatches +=
                packed_trans.bits[word] != control_trans.bits[word];
        }

        Accumulator zero{};
        Accumulator packed_acc{};
        Accumulator control_acc{};
        packed_acc.f32 = ins::mmac_f16_lit(
            packed_normal.f16x4[0], packed_trans.f16x4[0], zero.f32);
        control_acc.f32 = ins::mmac_f16_lit(
            control_normal.f16x4[0], control_trans.f16x4[0], zero.f32);
        int mmac_mismatches = 0;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            mmac_mismatches +=
                packed_acc.bits[word] != control_acc.bits[word];
        }
        if (lane == 0) {
            out[writer] =
                {normal_mismatches, trans_mismatches, mmac_mismatches};
        }
    }
#else
    (void)out;
#endif
}

}  // namespace

int main() {
    std::vector<ProbeResult> host(2);
    ProbeResult* device = nullptr;
    check_hip(hipMalloc(&device, host.size() * sizeof(ProbeResult)),
              "hipMalloc");
    check_hip(hipMemset(device, 0xff, host.size() * sizeof(ProbeResult)),
              "hipMemset");

    hipLaunchKernelGGL(ds_write_matrix_packed_stride_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, device);
    check_hip(hipGetLastError(), "kernel launch");
    check_hip(hipDeviceSynchronize(), "kernel sync");
    check_hip(hipMemcpy(host.data(), device, host.size() * sizeof(ProbeResult),
                        hipMemcpyDeviceToHost),
              "hipMemcpy");

    int total_mismatches = 0;
    for (int writer = 0; writer < 2; ++writer) {
        const ProbeResult& result = host[writer];
        total_mismatches += result.normal_mismatches +
                            result.trans_mismatches +
                            result.mmac_mismatches;
        std::printf(
            "packed_stride writer=%d normal_mismatches=%d "
            "trans_mismatches=%d mmac_mismatches=%d\n",
            writer, result.normal_mismatches, result.trans_mismatches,
            result.mmac_mismatches);
    }
    const bool pass = total_mismatches == 0;
    std::printf("ds_write_matrix_packed_stride_probe_status=%s\n",
                pass ? "PASS" : "FAIL");
    check_hip(hipFree(device), "hipFree");
    return pass ? 0 : 1;
}
