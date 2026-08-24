#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 16;
constexpr int kCols = 32;
constexpr int kDim = 32;
constexpr int kElems = kRows * kCols;
constexpr int kWriters = 2;
constexpr int kStoreModes = 4;
constexpr int kPaths = kWriters * kStoreModes;
constexpr int kMatrixHalfs = 32 * 32;
constexpr int kLdsHalfs = 4 * kMatrixHalfs;
constexpr uint16_t kPoison = 0xfefe;

union Frag8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    ins::Vec2F16 f16x2[4];
    _Float16 scalar[8];
};

union Frag4 {
    ins::Vec4F16 f16x4;
    ins::Vec2F16 f16x2[2];
    _Float16 scalar[4];
};

union Acc4 {
    ins::Vec4F32 f32;
    float scalar[4];
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

__device__ __forceinline__ ins::Vec4F32 mmac(
    ins::Vec4F16 lhs, ins::Vec4F16 rhs, ins::Vec4F32 acc) {
#if defined(__gfx946__) || defined(__gfx92a__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(
        lhs, rhs, acc, 1, 0);
#else
    (void)lhs;
    (void)rhs;
    return acc;
#endif
}

__device__ __forceinline__ void read_v_pair(
    const _Float16* lds, Frag8& v0, Frag8& v1) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int address = static_cast<int>(reinterpret_cast<size_t>(lds));
    asm volatile(
        "ds_read_matrix_format %0, %2 offset:0 element:0x2 row:0x2 col:0x1 alt:0x1\n\t"
        "ds_read_matrix_format %1, %2 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x1\n"
        : "=v"(v0.f16x8), "=v"(v1.f16x8)
        : "s"(address)
        : "memory");
#else
    (void)lds;
    v0 = {};
    v1 = {};
#endif
}

__device__ __forceinline__ Frag4 pack_score(const Acc4& score) {
    Frag4 packed{};
#if defined(__gfx946__) || defined(__gfx92a__)
    packed.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(
        score.scalar[0], score.scalar[1], false, 0);
    packed.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(
        score.scalar[2], score.scalar[3], false, 0);
#endif
    return packed;
}

__device__ __forceinline__ Frag8 pack_output(
    const Acc4& out0, const Acc4& out1) {
    Frag8 packed{};
#if defined(__gfx946__) || defined(__gfx92a__)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed.f16x2[i] = __builtin_hcu_cvt_pk_f16_f32(
            out0.scalar[i], out1.scalar[i], false, 0);
    }
#endif
    return packed;
}

__device__ __forceinline__ void matrix_store(
    int mode, ins::Vec4U32 dst, _Float16* lds) {
    switch (mode) {
        case 0:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0,
                false, false, false, false);
            break;
        case 1:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0,
                true, false, false, false);
            break;
        case 2:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0,
                false, true, false, false);
            break;
        default:
            __builtin_hcu_matrix_store_32x16_b16(
                dst, reinterpret_cast<short*>(lds), 0,
                true, true, false, false);
            break;
    }
}

__global__ void __launch_bounds__(kWaveSize, 1)
fwd_native_mmac_writer_chain_kernel(
    const _Float16* q, const _Float16* k, const _Float16* v,
    _Float16* direct_output, _Float16* matrix_output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(4096) _Float16 lds[kLdsHalfs];
    _Float16* const q_lds = lds;
    _Float16* const k_lds = q_lds + kMatrixHalfs;
    _Float16* const v_lds = k_lds + kMatrixHalfs;
    _Float16* const out_lds = v_lds + kMatrixHalfs;
    const int path = static_cast<int>(blockIdx.x);
    const int writer = path / kStoreModes;
    const int store_mode = path % kStoreModes;

    ins::matrix_load_32x32_b16_bps_lds(
        reinterpret_cast<__half*>(q_lds),
        ins::prepare_matrix_src(reinterpret_cast<const __half*>(q), kDim),
        0, true);
    ins::matrix_load_32x32_b16_bps_lds(
        reinterpret_cast<__half*>(k_lds),
        ins::prepare_matrix_src(reinterpret_cast<const __half*>(k), kDim),
        0, false);
    ins::matrix_load_32x32_b16_bps_lds(
        reinterpret_cast<__half*>(v_lds),
        ins::prepare_matrix_src(reinterpret_cast<const __half*>(v), kCols),
        0, false);
    ins::wait_vbcnt0();

    Frag8 q_fragment{};
    Frag8 k0{};
    Frag8 k1{};
    Frag8 v0{};
    Frag8 v1{};
    ins::ds_read_matrix_32x16_trans(
        reinterpret_cast<const __half*>(q_lds), 0, q_fragment.f16x8);
    ins::ds_read_matrix_trans_pair(
        reinterpret_cast<const __half*>(k_lds), 0,
        k0.f16x8, k1.f16x8);
    read_v_pair(v_lds, v0, v1);
    ins::wait_lgkm(0);

    Acc4 score0{};
    Acc4 score1{};
    score0.f32 = mmac(q_fragment.f16x4[0], k0.f16x4[0], score0.f32);
    score1.f32 = mmac(q_fragment.f16x4[0], k1.f16x4[0], score1.f32);
    score0.f32 = mmac(q_fragment.f16x4[1], k0.f16x4[1], score0.f32);
    score1.f32 = mmac(q_fragment.f16x4[1], k1.f16x4[1], score1.f32);

    const Frag4 p0 = pack_score(score0);
    const Frag4 p1 = pack_score(score1);
    Acc4 out0{};
    Acc4 out1{};
    out0.f32 = mmac(p0.f16x4, v0.f16x4[0], out0.f32);
    out1.f32 = mmac(p0.f16x4, v0.f16x4[1], out1.f32);
    out0.f32 = mmac(p1.f16x4, v1.f16x4[0], out0.f32);
    out1.f32 = mmac(p1.f16x4, v1.f16x4[1], out1.f32);
    const Frag8 packed = pack_output(out0, out1);

    const int lane = static_cast<int>(threadIdx.x & 63);
    const int row = lane & 15;
    const int col = (lane >> 4) * 8;
    *reinterpret_cast<ins::Vec8F16*>(
        direct_output + path * kElems + row * kCols + col) = packed.f16x8;

    if (writer == 0) {
        ins::ds_write_matrix_32x16_f16(
            packed.f16x8, reinterpret_cast<__half*>(out_lds), 0);
    } else {
        ins::ds_write_matrix_32x16_trans_f16(
            packed.f16x8, reinterpret_cast<__half*>(out_lds), 0);
    }
    ins::wait_lgkm(0);
    const ins::Vec4U32 dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(matrix_output + path * kElems),
        kCols);
    matrix_store(store_mode, dst, out_lds);
    ins::wait_vmem_lgkm();
#else
    (void)q;
    (void)k;
    (void)v;
    (void)direct_output;
    (void)matrix_output;
#endif
}

float q_value(int row, int dim) {
    return static_cast<float>(((row * 3 + dim * 2) % 3) - 1);
}

float k_value(int row, int dim) {
    return static_cast<float>(((row * 2 + dim * 5 + 1) % 3) - 1);
}

float v_value(int row, int col) {
    return static_cast<float>(((row * 5 + col * 3 + 2) % 5) - 2);
}

struct Comparison {
    int mismatches = 0;
    int poison = 0;
    int first = -1;
    float max_abs = 0.0f;
};

Comparison compare(
    const std::vector<_Float16>& output, int path,
    const std::vector<_Float16>& expected) {
    Comparison result{};
    for (int i = 0; i < kElems; ++i) {
        HalfBits bits{output[path * kElems + i]};
        result.poison += bits.bits == kPoison;
        const float error = std::fabs(
            static_cast<float>(output[path * kElems + i]) -
            static_cast<float>(expected[i]));
        if (error != 0.0f) {
            ++result.mismatches;
            if (result.first < 0) result.first = i;
        }
        result.max_abs = std::max(result.max_abs, error);
    }
    return result;
}

}  // namespace

int main() {
    std::vector<_Float16> q(32 * kDim);
    std::vector<_Float16> k(32 * kDim);
    std::vector<_Float16> v(32 * kCols);
    for (int row = 0; row < 32; ++row) {
        for (int dim = 0; dim < kDim; ++dim) {
            q[row * kDim + dim] = static_cast<_Float16>(q_value(row, dim));
            k[row * kDim + dim] = static_cast<_Float16>(k_value(row, dim));
        }
        for (int col = 0; col < kCols; ++col) {
            v[row * kCols + col] = static_cast<_Float16>(v_value(row, col));
        }
    }

    std::vector<_Float16> p(kRows * 32);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < 32; ++col) {
            float score = 0.0f;
            for (int dim = 0; dim < kDim; ++dim) {
                score += static_cast<float>(q[row * kDim + dim]) *
                         static_cast<float>(k[col * kDim + dim]);
            }
            p[row * 32 + col] = static_cast<_Float16>(score);
        }
    }
    std::vector<_Float16> expected(kElems);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            float sum = 0.0f;
            for (int dim = 0; dim < 32; ++dim) {
                sum += static_cast<float>(p[row * 32 + dim]) *
                       static_cast<float>(v[dim * kCols + col]);
            }
            expected[row * kCols + col] = static_cast<_Float16>(sum);
        }
    }

    _Float16* device_q = nullptr;
    _Float16* device_k = nullptr;
    _Float16* device_v = nullptr;
    _Float16* device_direct = nullptr;
    _Float16* device_matrix = nullptr;
    check_hip(hipMalloc(&device_q, q.size() * sizeof(_Float16)), "malloc q");
    check_hip(hipMalloc(&device_k, k.size() * sizeof(_Float16)), "malloc k");
    check_hip(hipMalloc(&device_v, v.size() * sizeof(_Float16)), "malloc v");
    check_hip(hipMalloc(&device_direct, kPaths * kElems * sizeof(_Float16)),
              "malloc direct");
    check_hip(hipMalloc(&device_matrix, kPaths * kElems * sizeof(_Float16)),
              "malloc matrix");
    check_hip(hipMemcpy(device_q, q.data(), q.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy q");
    check_hip(hipMemcpy(device_k, k.data(), k.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy k");
    check_hip(hipMemcpy(device_v, v.data(), v.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice), "copy v");
    std::vector<uint16_t> poison(kPaths * kElems, kPoison);
    check_hip(hipMemcpy(device_direct, poison.data(),
                        poison.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              "poison direct");
    check_hip(hipMemcpy(device_matrix, poison.data(),
                        poison.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              "poison matrix");

    hipLaunchKernelGGL(fwd_native_mmac_writer_chain_kernel,
                       dim3(kPaths), dim3(kWaveSize), 0, 0,
                       device_q, device_k, device_v,
                       device_direct, device_matrix);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    std::vector<_Float16> direct(kPaths * kElems);
    std::vector<_Float16> matrix(kPaths * kElems);
    check_hip(hipMemcpy(direct.data(), device_direct,
                        direct.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
              "copy direct");
    check_hip(hipMemcpy(matrix.data(), device_matrix,
                        matrix.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
              "copy matrix");
    check_hip(hipFree(device_matrix), "free matrix");
    check_hip(hipFree(device_direct), "free direct");
    check_hip(hipFree(device_v), "free v");
    check_hip(hipFree(device_k), "free k");
    check_hip(hipFree(device_q), "free q");

    int direct_exact = 0;
    int matrix_exact = 0;
    const char* writer_names[kWriters] = {"normal", "trans"};
    const char* store_names[kStoreModes] = {
        "t0r0", "t1r0", "t0r1", "t1r1"};
    for (int path = 0; path < kPaths; ++path) {
        const int writer = path / kStoreModes;
        const int store_mode = path % kStoreModes;
        const Comparison direct_result = compare(direct, path, expected);
        const Comparison matrix_result = compare(matrix, path, expected);
        direct_exact += direct_result.mismatches == 0;
        matrix_exact += matrix_result.mismatches == 0;
        std::printf(
            "fwd_native_mmac_writer writer=%s store=%s direct_mismatch=%d "
            "direct_poison=%d direct_max_abs=%g direct_first=(%d,%d) "
            "matrix_mismatch=%d matrix_poison=%d matrix_max_abs=%g "
            "matrix_first=(%d,%d)\n",
            writer_names[writer], store_names[store_mode],
            direct_result.mismatches, direct_result.poison,
            direct_result.max_abs,
            direct_result.first < 0 ? -1 : direct_result.first / kCols,
            direct_result.first < 0 ? -1 : direct_result.first % kCols,
            matrix_result.mismatches, matrix_result.poison,
            matrix_result.max_abs,
            matrix_result.first < 0 ? -1 : matrix_result.first / kCols,
            matrix_result.first < 0 ? -1 : matrix_result.first % kCols);
    }
    const bool direct_pass = direct_exact == kPaths;
    std::printf(
        "fwd_native_mmac_writer_status=%s direct_exact=%d/%d "
        "matrix_exact=%d/%d writer_status=%s\n",
        direct_pass ? "PASS" : "FAIL", direct_exact, kPaths,
        matrix_exact, kPaths, matrix_exact > 0 ? "PASS" : "OPEN");
    return direct_pass ? 0 : 1;
}
