#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 32;
constexpr int kCols = 32;
constexpr int kDim = 32;
constexpr int kElems = kRows * kCols;
constexpr int kMmacModes = 4;
constexpr int kWriters = 2;
constexpr int kPairings = 2;
constexpr int kPacks = 4;
constexpr int kOutputKinds = 2;
constexpr int kCandidates =
    kOutputKinds * kMmacModes * kWriters * kPairings * kPacks;
constexpr int kMatrixHalfs = kRows * kDim;
constexpr int kResultPageHalfs = 4096;
constexpr int kLdsHalfs = 2 * kMatrixHalfs + kResultPageHalfs;
constexpr uint16_t kPoison = 0xfefe;

union Frag8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    ins::Vec2F16 f16x2[4];
    _Float16 scalar[8];
};

union Acc4 {
    ins::Vec4F32 f32;
    float scalar[4];
};

struct AccF32Set {
    Acc4 c00;
    Acc4 c01;
    Acc4 c10;
    Acc4 c11;
};

struct AccF16Set {
    ins::Vec4F16 c00;
    ins::Vec4F16 c01;
    ins::Vec4F16 c10;
    ins::Vec4F16 c11;
};

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

template <int Lit, int Lts>
__device__ __forceinline__ ins::Vec4F32 mmac_f32(
    ins::Vec4F16 lhs, ins::Vec4F16 rhs, ins::Vec4F32 acc) {
#if defined(__gfx946__) || defined(__gfx92a__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs, rhs, acc, Lit, Lts);
#else
    (void)lhs;
    (void)rhs;
    return acc;
#endif
}

template <int Lit, int Lts>
__device__ __forceinline__ ins::Vec4F16 mmac_f16(
    ins::Vec4F16 lhs, ins::Vec4F16 rhs, ins::Vec4F16 acc) {
#if defined(__gfx946__) || defined(__gfx92a__)
    return __builtin_hcu_mmac_16x16x16_f16_lit_lts(
        lhs, rhs, acc, Lit, Lts);
#else
    (void)lhs;
    (void)rhs;
    return acc;
#endif
}

template <int Lit, int Lts>
__device__ __forceinline__ void compute_f32(
    const Frag8& a0, const Frag8& a1, const Frag8& b0, const Frag8& b1,
    AccF32Set& c) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        c.c00.f32 = mmac_f32<Lit, Lts>(
            a0.f16x4[k], b0.f16x4[k], c.c00.f32);
        c.c01.f32 = mmac_f32<Lit, Lts>(
            a0.f16x4[k], b1.f16x4[k], c.c01.f32);
        c.c10.f32 = mmac_f32<Lit, Lts>(
            a1.f16x4[k], b0.f16x4[k], c.c10.f32);
        c.c11.f32 = mmac_f32<Lit, Lts>(
            a1.f16x4[k], b1.f16x4[k], c.c11.f32);
    }
}

template <int Lit, int Lts>
__device__ __forceinline__ void compute_f16(
    const Frag8& a0, const Frag8& a1, const Frag8& b0, const Frag8& b1,
    AccF16Set& c) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        c.c00 = mmac_f16<Lit, Lts>(a0.f16x4[k], b0.f16x4[k], c.c00);
        c.c01 = mmac_f16<Lit, Lts>(a0.f16x4[k], b1.f16x4[k], c.c01);
        c.c10 = mmac_f16<Lit, Lts>(a1.f16x4[k], b0.f16x4[k], c.c10);
        c.c11 = mmac_f16<Lit, Lts>(a1.f16x4[k], b1.f16x4[k], c.c11);
    }
}

template <typename Fragment>
__device__ __forceinline__ _Float16 fragment_value(
    const Fragment& fragment, int index) {
    return static_cast<_Float16>(fragment[index]);
}

template <>
__device__ __forceinline__ _Float16 fragment_value<Acc4>(
    const Acc4& fragment, int index) {
    return static_cast<_Float16>(fragment.scalar[index]);
}

template <typename Fragment>
__device__ __forceinline__ Frag8 pack_pair(
    const Fragment& first, const Fragment& second, int pack) {
    Frag8 out{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const _Float16 a = fragment_value(first, i);
        const _Float16 b = fragment_value(second, i);
        if ((pack & 1) == 0) {
            out.scalar[i] = a;
            out.scalar[4 + i] = b;
        } else {
            out.scalar[2 * i] = a;
            out.scalar[2 * i + 1] = b;
        }
    }
    return out;
}

template <>
__device__ __forceinline__ Frag8 pack_pair<Acc4>(
    const Acc4& first, const Acc4& second, int pack) {
    if (pack < 2) {
        Frag8 out{};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const _Float16 a = static_cast<_Float16>(first.scalar[i]);
            const _Float16 b = static_cast<_Float16>(second.scalar[i]);
            if (pack == 0) {
                out.scalar[i] = a;
                out.scalar[4 + i] = b;
            } else {
                out.scalar[2 * i] = a;
                out.scalar[2 * i + 1] = b;
            }
        }
        return out;
    }

    Frag8 out{};
    if (pack == 2) {
        out.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(
            first.scalar[0], first.scalar[1], false, 0);
        out.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(
            first.scalar[2], first.scalar[3], false, 0);
        out.f16x2[2] = __builtin_hcu_cvt_pk_f16_f32(
            second.scalar[0], second.scalar[1], false, 0);
        out.f16x2[3] = __builtin_hcu_cvt_pk_f16_f32(
            second.scalar[2], second.scalar[3], false, 0);
    } else {
        out.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(
            first.scalar[0], second.scalar[0], false, 0);
        out.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(
            first.scalar[1], second.scalar[1], false, 0);
        out.f16x2[2] = __builtin_hcu_cvt_pk_f16_f32(
            first.scalar[2], second.scalar[2], false, 0);
        out.f16x2[3] = __builtin_hcu_cvt_pk_f16_f32(
            first.scalar[3], second.scalar[3], false, 0);
    }
    return out;
}

__device__ __forceinline__ void write_fragment(
    int writer, const Frag8& fragment, _Float16* page, int byte_offset) {
    _Float16* const ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(page) + byte_offset);
    if (writer == 0) {
        __builtin_hcu_ds_write_matrix_format_f16(
            fragment.f16x8, ptr, 0, 2, 1, 0, 0);
    } else {
        __builtin_hcu_ds_write_matrix_format_f16(
            fragment.f16x8, ptr, 0, 2, 1, 0, 1);
    }
}

template <typename Fragment>
__device__ __forceinline__ void write_result(
    const Fragment& c00, const Fragment& c01,
    const Fragment& c10, const Fragment& c11,
    int writer, int pairing, int pack, _Float16* page) {
    if (pairing == 0) {
        write_fragment(writer, pack_pair(c00, c01, pack), page, 0);
        write_fragment(writer, pack_pair(c10, c11, pack), page, 1024);
    } else {
        write_fragment(writer, pack_pair(c00, c10, pack), page, 0);
        write_fragment(writer, pack_pair(c01, c11, pack), page, 1024);
    }
}

__device__ __forceinline__ void compute_and_write(
    int output_kind, int mode, int writer, int pairing, int pack,
    const Frag8& a0, const Frag8& a1, const Frag8& b0, const Frag8& b1,
    _Float16* page) {
    if (output_kind == 0) {
        AccF32Set c{};
        switch (mode) {
            case 0: compute_f32<0, 0>(a0, a1, b0, b1, c); break;
            case 1: compute_f32<1, 0>(a0, a1, b0, b1, c); break;
            case 2: compute_f32<0, 1>(a0, a1, b0, b1, c); break;
            default: compute_f32<1, 1>(a0, a1, b0, b1, c); break;
        }
        write_result(c.c00, c.c01, c.c10, c.c11,
                     writer, pairing, pack, page);
        return;
    }

    AccF16Set c{};
    switch (mode) {
        case 0: compute_f16<0, 0>(a0, a1, b0, b1, c); break;
        case 1: compute_f16<1, 0>(a0, a1, b0, b1, c); break;
        case 2: compute_f16<0, 1>(a0, a1, b0, b1, c); break;
        default: compute_f16<1, 1>(a0, a1, b0, b1, c); break;
    }
    write_result(c.c00, c.c01, c.c10, c.c11,
                 writer, pairing, pack, page);
}

__global__ void __launch_bounds__(kWaveSize, 1)
matrix_store_32x32_mmac_source_kernel(
    const _Float16* a, const _Float16* b, _Float16* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) _Float16 lds[kLdsHalfs];
    _Float16* const a_lds = lds;
    _Float16* const b_lds = a_lds + kMatrixHalfs;
    _Float16* const result_lds = b_lds + kMatrixHalfs;
    const int path = static_cast<int>(blockIdx.x);

    ins::matrix_load_32x32_b16_bps_lds(
        reinterpret_cast<__half*>(a_lds),
        ins::prepare_matrix_src(reinterpret_cast<const __half*>(a), kDim),
        0, true);
    ins::matrix_load_32x32_b16_bps_lds(
        reinterpret_cast<__half*>(b_lds),
        ins::prepare_matrix_src(reinterpret_cast<const __half*>(b), kDim),
        0, true);
    ins::wait_vbcnt0();

    Frag8 a0{};
    Frag8 a1{};
    Frag8 b0{};
    Frag8 b1{};
    ins::ds_read_matrix_trans_pair(
        reinterpret_cast<const __half*>(a_lds), 0,
        a0.f16x8, a1.f16x8);
    ins::ds_read_matrix_trans_pair(
        reinterpret_cast<const __half*>(b_lds), 0,
        b0.f16x8, b1.f16x8);
    ins::wait_lgkm(0);

    int value = path;
    const int pack = value % kPacks;
    value /= kPacks;
    const int pairing = value % kPairings;
    value /= kPairings;
    const int writer = value % kWriters;
    value /= kWriters;
    const int mode = value % kMmacModes;
    const int output_kind = value / kMmacModes;
    compute_and_write(output_kind, mode, writer, pairing, pack,
                      a0, a1, b0, b1, result_lds);
    ins::wait_lgkm(0);

    const ins::Vec4U32 dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output + path * kElems), kCols);
    __builtin_hcu_matrix_store_32x32_b16(
        dst, reinterpret_cast<short*>(result_lds), 0,
        false, false, false, false);
    ins::wait_vmem_lgkm();
#else
    (void)a;
    (void)b;
    (void)output;
#endif
}

float input_a(int row, int dim) {
    return static_cast<float>(((row * 5 + dim * 3) % 11) - 5);
}

float input_b(int row, int dim) {
    return static_cast<float>(((row * 7 + dim * 2) % 13) - 6);
}

}  // namespace

int main() {
    const char* output_names[kOutputKinds] = {"fp32_pack", "fp16_mmac"};
    const char* mode_names[kMmacModes] = {
        "lit0_lts0", "lit1_lts0", "lit0_lts1", "lit1_lts1"};
    const char* writer_names[kWriters] = {
        "m32x16_normal", "m32x16_trans"};
    const char* pairing_names[kPairings] = {"npair", "mpair"};
    const char* pack_names[kPacks] = {
        "scalar_concat", "scalar_interleave",
        "cvtpk_concat", "cvtpk_interleave"};

    std::vector<_Float16> a(kRows * kDim);
    std::vector<_Float16> b(kRows * kDim);
    std::vector<_Float16> expected(kElems);
    for (int row = 0; row < kRows; ++row) {
        for (int dim = 0; dim < kDim; ++dim) {
            a[row * kDim + dim] = static_cast<_Float16>(input_a(row, dim));
            b[row * kDim + dim] = static_cast<_Float16>(input_b(row, dim));
        }
    }
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            float sum = 0.0f;
            for (int dim = 0; dim < kDim; ++dim) {
                sum += static_cast<float>(a[row * kDim + dim]) *
                       static_cast<float>(b[col * kDim + dim]);
            }
            expected[row * kCols + col] = static_cast<_Float16>(sum);
        }
    }

    _Float16* device_a = nullptr;
    _Float16* device_b = nullptr;
    _Float16* device_output = nullptr;
    check_hip(hipMalloc(&device_a, a.size() * sizeof(_Float16)), "malloc a");
    check_hip(hipMalloc(&device_b, b.size() * sizeof(_Float16)), "malloc b");
    check_hip(hipMalloc(&device_output,
                        kCandidates * kElems * sizeof(_Float16)),
              "malloc output");
    check_hip(hipMemcpy(device_a, a.data(), a.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy a");
    check_hip(hipMemcpy(device_b, b.data(), b.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy b");
    std::vector<uint16_t> poison(kCandidates * kElems, kPoison);
    check_hip(hipMemcpy(device_output, poison.data(),
                        poison.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice), "poison output");

    hipLaunchKernelGGL(matrix_store_32x32_mmac_source_kernel,
                       dim3(kCandidates), dim3(kWaveSize), 0, 0,
                       device_a, device_b, device_output);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    std::vector<_Float16> output(kCandidates * kElems);
    check_hip(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(_Float16),
                        hipMemcpyDeviceToHost), "copy output");
    check_hip(hipFree(device_output), "free output");
    check_hip(hipFree(device_b), "free b");
    check_hip(hipFree(device_a), "free a");

    int exact[kOutputKinds]{};
    int complete = 0;
    for (int path = 0; path < kCandidates; ++path) {
        int value = path;
        const int pack = value % kPacks;
        value /= kPacks;
        const int pairing = value % kPairings;
        value /= kPairings;
        const int writer = value % kWriters;
        value /= kWriters;
        const int mode = value % kMmacModes;
        const int output_kind = value / kMmacModes;
        int mismatches = 0;
        int poison_count = 0;
        float max_abs = 0.0f;
        int first = -1;
        for (int index = 0; index < kElems; ++index) {
            HalfBits bits{output[path * kElems + index]};
            poison_count += bits.bits == kPoison;
            const float got = static_cast<float>(output[path * kElems + index]);
            const float want = static_cast<float>(expected[index]);
            const float error = std::fabs(got - want);
            if (error != 0.0f) {
                ++mismatches;
                if (first < 0) first = index;
            }
            if (error > max_abs) max_abs = error;
        }
        exact[output_kind] += mismatches == 0;
        complete += poison_count == 0;
        std::printf(
            "matrix_store_32x32_mmac output=%s mode=%s writer=%s "
            "pairing=%s pack=%s mismatches=%d poison=%d max_abs=%g "
            "first_row=%d first_col=%d exact=%d\n",
            output_names[output_kind], mode_names[mode], writer_names[writer],
            pairing_names[pairing], pack_names[pack], mismatches,
            poison_count, max_abs, first < 0 ? -1 : first / kCols,
            first < 0 ? -1 : first % kCols, mismatches == 0 ? 1 : 0);
    }
    const bool ran = complete == kCandidates;
    std::printf(
        "matrix_store_32x32_mmac_source_status=%s candidates=%d "
        "complete=%d fp32_exact=%d fp16_exact=%d\n",
        ran ? "PASS" : "FAIL", kCandidates, complete, exact[0], exact[1]);
    return ran ? 0 : 1;
}
