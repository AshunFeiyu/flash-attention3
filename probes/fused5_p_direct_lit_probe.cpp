// A2: can FP16 score MMAC lit=1 directly provide the P operand layout for dV?
//
// Control:   score(lit0) -> trans matrix writer -> normal matrix reader -> dV
// Candidate: score(lit1) ---------------------------------------------> dV
//
// Dense asymmetric D32 inputs prevent a sparse transport fingerprint from
// being mistaken for a fragment match. Logical score coordinates are gated by
// dq_source_slot_coordinate_probe; this probe is the downstream differential.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 32;
constexpr int kD = 32;
constexpr int kMatrixElems = kRows * kD;
constexpr int kMatrixBytes = kMatrixElems * sizeof(__half);
constexpr int kQBase = 0;
constexpr int kKBase = kQBase + kMatrixBytes;
constexpr int kDoutBase = kKBase + kMatrixBytes;
constexpr int kBridgeBase = kDoutBase + kMatrixBytes;
constexpr int kLdsBytes = kBridgeBase + kMatrixBytes;
constexpr int kDumpHalfs = kWaveSize * 4;
constexpr int kDumpFloats = kWaveSize * 4;
constexpr float kTolerance = 2.0e-3f;

union Frag {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union Acc {
    ins::Vec4F32 f32;
    float scalar[4];
    uint64_t u64[2];
};

__device__ __forceinline__ ins::Vec4F16 score_mmac(
    const Frag& q, const Frag& k, int lit) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    ins::Vec4F16 out = zero.f16x4[0];
#if defined(__gfx946__) || defined(__gfx92a__)
    if (lit == 0) {
        out = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
            q.f16x4[0], k.f16x4[0], out, 0, 0);
        out = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
            q.f16x4[1], k.f16x4[1], out, 0, 0);
    } else {
        out = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
            q.f16x4[0], k.f16x4[0], out, 1, 0);
        out = __builtin_hcu_mmac_16x16x16_f16_lit_lts(
            q.f16x4[1], k.f16x4[1], out, 1, 0);
    }
#else
    (void)q;
    (void)k;
    (void)lit;
#endif
    return out;
}

__device__ __forceinline__ Acc dv_mmac(ins::Vec4F16 p,
                                        ins::Vec4F16 dout) {
    Acc out{};
    ins::zero_vgpr2(out.u64[0]);
    ins::zero_vgpr2(out.u64[1]);
    out.f32 = ins::mmac_f16_lit(p, dout, out.f32);
    return out;
}

__global__ void __launch_bounds__(kWaveSize, 1) fused5_p_direct_lit_probe_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ dout,
    __half* __restrict__ score_lit0_dump,
    __half* __restrict__ score_lit1_dump,
    __half* __restrict__ bridge_dump,
    float* __restrict__ dv_control,
    float* __restrict__ dv_candidate) {
    const int lane = static_cast<int>(threadIdx.x);
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];

    ins::matrix_load_32x32_b16_bps_lds(
        lds, ins::prepare_matrix_src(q, kD), kQBase, true);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, ins::prepare_matrix_src(k, kD), kKBase, true);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, ins::prepare_matrix_src(dout, kD), kDoutBase, true);
    ins::wait_vmem_lgkm();
    ins::wait_vbcnt0();
    __syncthreads();

    Frag q_trans{};
    Frag k_trans{};
    Frag dout_normal{};
    ins::ds_read_matrix_32x16_trans(lds, kQBase, q_trans.f16x8);
    ins::ds_read_matrix_32x16_trans(lds, kKBase, k_trans.f16x8);
    ins::ds_read_matrix_32x16_normal(lds, kDoutBase, dout_normal.f16x8);
    ins::wait_lgkm(0);

    const ins::Vec4F16 score_lit0 = score_mmac(q_trans, k_trans, 0);
    const ins::Vec4F16 score_lit1 = score_mmac(q_trans, k_trans, 1);

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    Frag writer{};
    writer.f16x4[0] = score_lit0;
    writer.f16x4[1] = zero.f16x4[0];
    ins::ds_write_matrix_32x16_trans_f16(writer.f16x8, lds, kBridgeBase);
    ins::wait_lgkm(0);

    Frag bridged{};
    ins::ds_read_matrix_32x16_normal(lds, kBridgeBase, bridged.f16x8);
    ins::wait_lgkm(0);

    const Acc control = dv_mmac(bridged.f16x4[0], dout_normal.f16x4[0]);
    const Acc candidate = dv_mmac(score_lit1, dout_normal.f16x4[0]);

#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const int offset = lane * 4 + word;
        score_lit0_dump[offset] = score_lit0[word];
        score_lit1_dump[offset] = score_lit1[word];
        bridge_dump[offset] = bridged.f16x4[0][word];
        dv_control[offset] = control.scalar[word];
        dv_candidate[offset] = candidate.scalar[word];
    }
#else
    (void)q;
    (void)k;
    (void)dout;
    (void)score_lit0_dump;
    (void)score_lit1_dump;
    (void)bridge_dump;
    (void)dv_control;
    (void)dv_candidate;
#endif
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

float max_abs(const std::vector<float>& actual,
              const std::vector<float>& expected) {
    float out = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return INFINITY;
        }
        out = std::max(out, std::fabs(actual[i] - expected[i]));
    }
    return out;
}

std::vector<float> half_dump_to_float(const std::vector<__half>& values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = __half2float(values[i]);
    }
    return out;
}

}  // namespace

int main() {
    std::vector<__half> q(kMatrixElems, __float2half(0.0f));
    std::vector<__half> k(kMatrixElems, __float2half(0.0f));
    std::vector<__half> dout(kMatrixElems, __float2half(0.0f));
    for (int row = 0; row < kRows; ++row) {
        for (int d = 0; d < kD; ++d) {
            q[row * kD + d] = __float2half(
                static_cast<float>((row * 73 + d * 29 + row * d * 7 + 17) %
                                       251 -
                                   125) /
                127.0f);
            k[row * kD + d] = __float2half(
                static_cast<float>((row * 47 + d * 31 + row * d * 13 + 11) %
                                       239 -
                                   119) /
                131.0f);
            dout[row * kD + d] = __float2half(
                static_cast<float>((row * 37 + d * 43 + row * d * 11 + 5) %
                                       241 -
                                   120) /
                113.0f);
        }
    }

    __half* d_q = nullptr;
    __half* d_k = nullptr;
    __half* d_dout = nullptr;
    __half* d_score0 = nullptr;
    __half* d_score1 = nullptr;
    __half* d_bridge = nullptr;
    float* d_control = nullptr;
    float* d_candidate = nullptr;
    check_hip(hipMalloc(&d_q, kMatrixBytes), "hipMalloc q");
    check_hip(hipMalloc(&d_k, kMatrixBytes), "hipMalloc k");
    check_hip(hipMalloc(&d_dout, kMatrixBytes), "hipMalloc dout");
    check_hip(hipMalloc(&d_score0, kDumpHalfs * sizeof(__half)),
              "hipMalloc score0");
    check_hip(hipMalloc(&d_score1, kDumpHalfs * sizeof(__half)),
              "hipMalloc score1");
    check_hip(hipMalloc(&d_bridge, kDumpHalfs * sizeof(__half)),
              "hipMalloc bridge");
    check_hip(hipMalloc(&d_control, kDumpFloats * sizeof(float)),
              "hipMalloc control");
    check_hip(hipMalloc(&d_candidate, kDumpFloats * sizeof(float)),
              "hipMalloc candidate");

    check_hip(hipMemcpy(d_q, q.data(), kMatrixBytes, hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(d_k, k.data(), kMatrixBytes, hipMemcpyHostToDevice),
              "copy k");
    check_hip(
        hipMemcpy(d_dout, dout.data(), kMatrixBytes, hipMemcpyHostToDevice),
        "copy dout");

    hipLaunchKernelGGL(fused5_p_direct_lit_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, d_q, d_k, d_dout, d_score0,
                       d_score1, d_bridge, d_control, d_candidate);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    std::vector<__half> score0_h(kDumpHalfs);
    std::vector<__half> score1_h(kDumpHalfs);
    std::vector<__half> bridge_h(kDumpHalfs);
    std::vector<float> control(kDumpFloats);
    std::vector<float> candidate(kDumpFloats);
    check_hip(hipMemcpy(score0_h.data(), d_score0,
                        kDumpHalfs * sizeof(__half), hipMemcpyDeviceToHost),
              "copy score0");
    check_hip(hipMemcpy(score1_h.data(), d_score1,
                        kDumpHalfs * sizeof(__half), hipMemcpyDeviceToHost),
              "copy score1");
    check_hip(hipMemcpy(bridge_h.data(), d_bridge,
                        kDumpHalfs * sizeof(__half), hipMemcpyDeviceToHost),
              "copy bridge");
    check_hip(hipMemcpy(control.data(), d_control,
                        kDumpFloats * sizeof(float), hipMemcpyDeviceToHost),
              "copy control");
    check_hip(hipMemcpy(candidate.data(), d_candidate,
                        kDumpFloats * sizeof(float), hipMemcpyDeviceToHost),
              "copy candidate");

    const std::vector<float> score0 = half_dump_to_float(score0_h);
    const std::vector<float> score1 = half_dump_to_float(score1_h);
    const std::vector<float> bridge = half_dump_to_float(bridge_h);
    const float score1_vs_bridge = max_abs(score1, bridge);
    const float score0_vs_score1 = max_abs(score0, score1);
    const float candidate_vs_control = max_abs(candidate, control);
    const bool pass = candidate_vs_control <= kTolerance;

    std::printf(
        "p_direct_lit score1_vs_bridge=%g score0_vs_score1=%g "
        "candidate_vs_control=%g coordinate_oracle=source_slot_mode36 "
        "pass=%d\n",
        score1_vs_bridge, score0_vs_score1, candidate_vs_control,
        pass ? 1 : 0);

    check_hip(hipFree(d_candidate), "hipFree candidate");
    check_hip(hipFree(d_control), "hipFree control");
    check_hip(hipFree(d_bridge), "hipFree bridge");
    check_hip(hipFree(d_score1), "hipFree score1");
    check_hip(hipFree(d_score0), "hipFree score0");
    check_hip(hipFree(d_dout), "hipFree dout");
    check_hip(hipFree(d_k), "hipFree k");
    check_hip(hipFree(d_q), "hipFree q");
    return pass ? 0 : 1;
}
