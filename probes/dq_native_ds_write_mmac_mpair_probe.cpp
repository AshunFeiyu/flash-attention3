#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Vec4F16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;

constexpr int kWaveSize = 64;
constexpr int kReaderPageElems = 64 * 16;
constexpr int kMmacModeCount = 4;
constexpr int kWriterCount = 4;
constexpr int kReaderCount = 5;
constexpr int kResultCount = kMmacModeCount * kWriterCount * kReaderCount;

__device__ __forceinline__ _Float16 f16_from_bits(uint16_t bits) {
    return *reinterpret_cast<const _Float16*>(&bits);
}

template <int Lit, int Lts>
__device__ __forceinline__ ins::Vec4F32 mmac_mode(
    ins::Vec4F16 lhs, ins::Vec4F16 rhs, ins::Vec4F32 acc) {
#if defined(__gfx946__) || defined(__gfx92a__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs, rhs, acc, Lit, Lts);
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(lhs, rhs, acc);
#endif
}

__device__ __forceinline__ Vec8F16 join_mpair(
    const ins::F32x4& m0, const ins::F32x4& m1) {
    Vec8F16 out{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<_Float16>(m0.scalar[i]);
        out[4 + i] = static_cast<_Float16>(m1.scalar[i]);
    }
    return out;
}

__device__ __forceinline__ Vec8F16 select_source(
    int mode,
    const Vec8F16& lit0_lts0,
    const Vec8F16& lit1_lts0,
    const Vec8F16& lit0_lts1,
    const Vec8F16& lit1_lts1) {
    switch (mode) {
        case 0:
            return lit0_lts0;
        case 1:
            return lit1_lts0;
        case 2:
            return lit0_lts1;
        default:
            return lit1_lts1;
    }
}

__device__ __forceinline__ void write_fragment(
    int writer, const Vec8F16& fragment, _Float16* lds) {
    switch (writer) {
        case 0:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 16, 2, 1, 0, 0);
            break;
        case 1:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 16, 2, 1, 1, 0);
            break;
        case 2:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 16, 2, 1, 0, 1);
            break;
        default:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, lds, 16, 2, 1, 1, 1);
            break;
    }
}

__device__ __forceinline__ Vec8F16 read_fragment(int reader, _Float16* lds) {
    switch (reader) {
        case 0:
            return __builtin_hcu_ds_read_matrix_format_f16(
                lds, 16, 2, 1, 0);
        case 1:
            return __builtin_hcu_ds_read_matrix_format_f16(
                lds, 16, 2, 1, 1);
        case 2:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 16, 2, 1, 0);
        case 3:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 16, 1, 2, 0);
        default:
            return __builtin_hcu_ds_read_matrix_trans_format_f16(
                lds, 16, 1, 2, 1);
    }
}

__device__ __forceinline__ void count_dq_mpair_mismatch(
    const Vec4F16& expected_m0,
    const Vec4F16& expected_m1,
    const Vec8F16& candidate,
    const Vec4F16& k,
    int* out,
    int slot) {
    Vec4F16 candidate_m0{};
    Vec4F16 candidate_m1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        candidate_m0[i] = candidate[i];
        candidate_m1[i] = candidate[4 + i];
    }

    ins::F32x4 expected0{};
    ins::F32x4 expected1{};
    ins::F32x4 actual0{};
    ins::F32x4 actual1{};
    expected0.f32 = ins::mmac_f16_lit(expected_m0, k, expected0.f32);
    expected1.f32 = ins::mmac_f16_lit(expected_m1, k, expected1.f32);
    actual0.f32 = ins::mmac_f16_lit(candidate_m0, k, actual0.f32);
    actual1.f32 = ins::mmac_f16_lit(candidate_m1, k, actual1.f32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (expected0.scalar[i] != actual0.scalar[i]) {
            atomicAdd(out + slot, 1);
        }
        if (expected1.scalar[i] != actual1.scalar[i]) {
            atomicAdd(out + slot, 1);
        }
    }
}

// The f16 writer accepts one m32x16 source.  Build that source from two real
// adjacent-M MMAC outputs sharing the same K fragment.  Only native MMAC C/D
// controls (LIT/LTS) are swept; no lane permutation or manual source packing
// is allowed.  LIT=1,LTS=0 is the proved FWD-style logical reference.
__global__ void __launch_bounds__(kWaveSize, 1)
    dq_native_ds_write_mmac_mpair_probe_kernel(int* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 lds[kReaderPageElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int mmac_mode_id = static_cast<int>(blockIdx.x) / kWriterCount;
    const int writer = static_cast<int>(blockIdx.x) % kWriterCount;

    Vec4F16 q_m0{};
    Vec4F16 q_m1{};
    Vec4F16 score_k{};
    Vec4F16 dq_k{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_m0[i] = f16_from_bits(static_cast<uint16_t>(0x3400 + lane * 4 + i));
        q_m1[i] = f16_from_bits(static_cast<uint16_t>(0x3800 + lane * 4 + i));
        score_k[i] = f16_from_bits(static_cast<uint16_t>(0x3a00 + lane * 4 + i));
        dq_k[i] = f16_from_bits(static_cast<uint16_t>(0x3600 + lane * 4 + i));
    }

    ins::F32x4 zero{};
    ins::F32x4 m0_lit0_lts0{};
    ins::F32x4 m1_lit0_lts0{};
    ins::F32x4 m0_lit1_lts0{};
    ins::F32x4 m1_lit1_lts0{};
    ins::F32x4 m0_lit0_lts1{};
    ins::F32x4 m1_lit0_lts1{};
    ins::F32x4 m0_lit1_lts1{};
    ins::F32x4 m1_lit1_lts1{};
    m0_lit0_lts0.f32 = mmac_mode<0, 0>(q_m0, score_k, zero.f32);
    m1_lit0_lts0.f32 = mmac_mode<0, 0>(q_m1, score_k, zero.f32);
    m0_lit1_lts0.f32 = mmac_mode<1, 0>(q_m0, score_k, zero.f32);
    m1_lit1_lts0.f32 = mmac_mode<1, 0>(q_m1, score_k, zero.f32);
    m0_lit0_lts1.f32 = mmac_mode<0, 1>(q_m0, score_k, zero.f32);
    m1_lit0_lts1.f32 = mmac_mode<0, 1>(q_m1, score_k, zero.f32);
    m0_lit1_lts1.f32 = mmac_mode<1, 1>(q_m0, score_k, zero.f32);
    m1_lit1_lts1.f32 = mmac_mode<1, 1>(q_m1, score_k, zero.f32);

    const Vec8F16 source = select_source(
        mmac_mode_id, join_mpair(m0_lit0_lts0, m1_lit0_lts0),
        join_mpair(m0_lit1_lts0, m1_lit1_lts0),
        join_mpair(m0_lit0_lts1, m1_lit0_lts1),
        join_mpair(m0_lit1_lts1, m1_lit1_lts1));
    const Vec8F16 expected = join_mpair(m0_lit1_lts0, m1_lit1_lts0);
    Vec4F16 expected_m0{};
    Vec4F16 expected_m1{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        expected_m0[i] = expected[i];
        expected_m1[i] = expected[4 + i];
    }

    write_fragment(writer, source, lds);
    ins::wait_lgkm(0);
#pragma unroll
    for (int reader = 0; reader < kReaderCount; ++reader) {
        const Vec8F16 candidate = read_fragment(reader, lds);
        ins::wait_lgkm(0);
        const int slot =
            (mmac_mode_id * kWriterCount + writer) * kReaderCount + reader;
        count_dq_mpair_mismatch(expected_m0, expected_m1, candidate, dq_k,
                                out, slot);
    }
#else
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace

int main() {
    int* d_out = nullptr;
    int h_out[kResultCount]{};
    check_hip(hipMalloc(&d_out, sizeof(h_out)), "hipMalloc");
    check_hip(hipMemset(d_out, 0, sizeof(h_out)), "hipMemset");
    hipLaunchKernelGGL(dq_native_ds_write_mmac_mpair_probe_kernel,
                       dim3(kMmacModeCount * kWriterCount), dim3(kWaveSize), 0,
                       0, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost),
              "hipMemcpy");
    check_hip(hipFree(d_out), "hipFree");

    const char* mmac_names[kMmacModeCount] = {
        "mmac_lit0_lts0", "mmac_lit1_lts0", "mmac_lit0_lts1",
        "mmac_lit1_lts1",
    };
    const char* writer_names[kWriterCount] = {
        "write_n_alt0", "write_n_alt1", "write_t_alt0", "write_t_alt1",
    };
    const char* reader_names[kReaderCount] = {
        "read_n_m32_alt0", "read_n_m32_alt1", "read_t_m32_alt0",
        "read_t_m16_alt0", "read_t_m16_alt1",
    };

    bool any_pass = false;
    int best_slot = 0;
    for (int mode = 0; mode < kMmacModeCount; ++mode) {
        for (int writer = 0; writer < kWriterCount; ++writer) {
            for (int reader = 0; reader < kReaderCount; ++reader) {
                const int slot =
                    (mode * kWriterCount + writer) * kReaderCount + reader;
                if (h_out[slot] < h_out[best_slot]) {
                    best_slot = slot;
                }
                if (h_out[slot] == 0) {
                    std::printf("PASS mmac=%s writer=%s reader=%s\n",
                                mmac_names[mode], writer_names[writer],
                                reader_names[reader]);
                    any_pass = true;
                }
            }
        }
    }
    if (!any_pass) {
        const int mode = best_slot / (kWriterCount * kReaderCount);
        const int writer =
            (best_slot / kReaderCount) % kWriterCount;
        const int reader = best_slot % kReaderCount;
        std::printf("BEST mismatch=%d mmac=%s writer=%s reader=%s\n",
                    h_out[best_slot], mmac_names[mode], writer_names[writer],
                    reader_names[reader]);
    }
    std::printf("native_ds_write_mmac_mpair_any_pass=%d\n",
                any_pass ? 1 : 0);
    std::printf("native_ds_write_mmac_mpair_no_pack_pass=%d\n",
                any_pass ? 1 : 0);
    return any_pass ? 0 : 2;
}
