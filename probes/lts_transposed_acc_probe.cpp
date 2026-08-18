// PR1: MMOP LTS bit semantics probe (gfx946).
//
// Question: __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(lhs, rhs, acc, lit,
// lts) always passes lts=0 in the canonical kernels. This probe runs the
// same operand contract twice, once with lts=0 and once with lts=1, dumps
// both accumulators, and classifies the lts=1 layout against a CPU golden
// 16x16 product: no-op, transposed accumulator, or unsupported.
//
// Evidence ladder: compile/ASM (encoding delta of the two v_mmac sites),
// dense asymmetric oracle, resource gate via the run script.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kAccFloats = 4;
constexpr int kPageBytes = 32 * 32 * sizeof(__half);
constexpr int kLdsBytes = 2 * kPageBytes;

__global__ void __launch_bounds__(kWaveSize, 1) lts_probe_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ out0,
    float* __restrict__ out1) {
    const int lane = static_cast<int>(threadIdx.x);
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    // A plays the score-K role: MLS-transposed page, trans reader.
    // B plays the score-Q role: MLS-normal page, normal reader.
    ins::matrix_load_32x32_b16_bps_lds(
        lds, ins::prepare_matrix_src(a, 32), 0, true);
    ins::matrix_load_32x32_b16_bps_lds(
        lds, ins::prepare_matrix_src(b, 32), kPageBytes, false);
    ins::wait_vmem_lgkm();
    __syncthreads();

    ins::F16x8 lhs;
    ins::F16x8 rhs;
    ins::ds_read_matrix_32x16_trans(lds, 0, lhs.f16x8);
    ins::ds_read_matrix_32x16_normal(lds + kPageBytes / sizeof(__half), 0,
                                     rhs.f16x8);
    ins::wait_lgkm(0);

    ins::F32x4 acc0;
    ins::F32x4 acc1;
    ins::zero_vgpr2(acc0.u64[0]);
    ins::zero_vgpr2(acc0.u64[1]);
    ins::zero_vgpr2(acc1.u64[0]);
    ins::zero_vgpr2(acc1.u64[1]);

    acc0.f32 = ins::mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], acc0.f32);
    acc1.f32 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs.f16x4[0], rhs.f16x4[0], acc1.f32, 1, 1);

#pragma unroll
    for (int w = 0; w < kAccFloats; ++w) {
        out0[lane * kAccFloats + w] = acc0.scalar[w];
        out1[lane * kAccFloats + w] = acc1.scalar[w];
    }
#else
    (void)a;
    (void)b;
    for (int w = 0; w < kAccFloats; ++w) {
        out0[lane * kAccFloats + w] = 0.0f;
        out1[lane * kAccFloats + w] = 0.0f;
    }
#endif
}

// Canonical C extraction for lts=0, taken from the kernel's proven dV/score
// store mapping: row = lane & 15, col = (lane >> 4) * 4 + word.
void extract_m1(const std::vector<float>& dump, float c[16][16]) {
    for (int lane = 0; lane < kWaveSize; ++lane) {
        for (int w = 0; w < kAccFloats; ++w) {
            c[lane & 15][(lane >> 4) * 4 + w] = dump[lane * 4 + w];
        }
    }
}

// Transposed interpretation: the same dump read as C^T.
void extract_transposed(const std::vector<float>& dump, float c[16][16]) {
    for (int lane = 0; lane < kWaveSize; ++lane) {
        for (int w = 0; w < kAccFloats; ++w) {
            c[(lane >> 4) * 4 + w][lane & 15] = dump[lane * 4 + w];
        }
    }
}

float max_abs_diff(const float a[16][16], const float b[16][16]) {
    float m = 0.0f;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            m = std::fmax(m, std::fabs(a[i][j] - b[i][j]));
        }
    }
    return m;
}

}  // namespace

int main() {
    std::vector<__half> ha(32 * 32, __half(0.0f));
    std::vector<__half> hb(32 * 32, __half(0.0f));
    float golden[16][16];
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            ha[i * 32 + j] = __half(((i * 7 + j * 3) % 13 - 6) * 0.25f);
            hb[i * 32 + j] = __half(((i * 5 + j * 11) % 9 - 4) * 0.5f);
        }
    }
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 16; ++k) {
                s += static_cast<float>(ha[i * 32 + k]) *
                     static_cast<float>(hb[k * 32 + j]);
            }
            golden[i][j] = s;
        }
    }

    __half* da = nullptr;
    __half* db = nullptr;
    float* out0 = nullptr;
    float* out1 = nullptr;
    hipMalloc(&da, ha.size() * sizeof(__half));
    hipMalloc(&db, hb.size() * sizeof(__half));
    hipMalloc(&out0, kWaveSize * kAccFloats * sizeof(float));
    hipMalloc(&out1, kWaveSize * kAccFloats * sizeof(float));
    hipMemcpy(da, ha.data(), ha.size() * sizeof(__half),
              hipMemcpyHostToDevice);
    hipMemcpy(db, hb.data(), hb.size() * sizeof(__half),
              hipMemcpyHostToDevice);

    lts_probe_kernel<<<dim3(1), dim3(kWaveSize), 0, 0>>>(da, db, out0,
                                                                out1);

    std::vector<float> v0(kWaveSize * kAccFloats);
    std::vector<float> v1(kWaveSize * kAccFloats);
    hipMemcpy(v0.data(), out0, v0.size() * sizeof(float),
              hipMemcpyDeviceToHost);
    hipMemcpy(v1.data(), out1, v1.size() * sizeof(float),
              hipMemcpyDeviceToHost);

    float c0[16][16];
    extract_m1(v0, c0);
    const float base_diff = max_abs_diff(c0, golden);

    float c1_m1[16][16];
    extract_m1(v1, c1_m1);
    float c1_t[16][16];
    extract_transposed(v1, c1_t);
    float golden_t[16][16];
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            golden_t[i][j] = golden[j][i];
        }
    }

    // Raw dump comparison disambiguates a true layout change from a no-op:
    // with v1 == v0 both extractions trivially match their hypotheses.
    float raw_diff = 0.0f;
    for (size_t i = 0; i < v0.size(); ++i) {
        raw_diff = std::fmax(raw_diff, std::fabs(v1[i] - v0[i]));
    }
    const float noop_diff = max_abs_diff(c1_m1, c0);
    const float trans_diff = max_abs_diff(c1_t, golden_t);
    const char* verdict = "UNKNOWN";
    if (base_diff >= 5.0e-3f) {
        verdict = "BASE_MAPPING_BROKEN";
    } else if (raw_diff == 0.0f) {
        verdict = "NOOP_SAME_DUMP";
    } else if (trans_diff < 5.0e-3f) {
        verdict = "TRANSPOSED_ACCUMULATOR";
    } else if (noop_diff < 5.0e-3f) {
        verdict = "SAME_LAYOUT_DIFFERENT_VALUES";
    }

    std::printf(
        "lts_probe base_m1_vs_golden=%.6g raw_v1_vs_v0=%.6g "
        "lts1_noop_vs_base=%.6g lts1_transposed_vs_goldenT=%.6g "
        "verdict=%s\n",
        base_diff, raw_diff, noop_diff, trans_diff, verdict);

    hipFree(da);
    hipFree(db);
    hipFree(out0);
    hipFree(out1);
    return 0;
}
