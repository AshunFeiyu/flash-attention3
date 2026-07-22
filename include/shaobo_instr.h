#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shaobo::fa3::bwd::dkv::instr {

#ifndef SHAOBO_BPS_VBCNT_BEFORE_ARRIVE
#define SHAOBO_BPS_VBCNT_BEFORE_ARRIVE 1
#endif

using Vec4U32 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using Vec2F16 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using Vec4F16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using Vec4F32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

union F16x8 {
    Vec8F16 f16x8;
    Vec4F16 f16x4[2];
    Vec4F32 f32;
    uint64_t u64[2];
};

union F32x4 {
    Vec4F32 f32;
    float scalar[4];
    uint64_t u64[2];
};

template <int ByteOffset1, int ByteOffset2>
__device__ __forceinline__ void ds_read_b32_lds_imm3(
    const float* lds,
    float& value0,
    float& value1,
    float& value2) {
    static_assert(ByteOffset1 >= 0 && ByteOffset1 < (1 << 20) &&
                      ByteOffset2 >= 0 && ByteOffset2 < (1 << 20),
                  "LDS sidecar offset must fit ds_read_b32");
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
    asm volatile(
        "ds_read_b32 %0, %3 offset:0\n\t"
        "ds_read_b32 %1, %3 offset:%4\n\t"
        "ds_read_b32 %2, %3 offset:%5\n"
        : "=v"(value0), "=v"(value1), "=v"(value2)
        : "v"(lds_addr), "n"(ByteOffset1), "n"(ByteOffset2)
        : "memory");
#else
    (void)lds;
    value0 = 0.0f;
    value1 = 0.0f;
    value2 = 0.0f;
#endif
}

template <int ByteOffset1, int ByteOffset2>
__device__ __forceinline__ void ds_read_b128_lds_imm3(
    const float* lds,
    Vec4F32& frag0,
    Vec4F32& frag1,
    Vec4F32& frag2) {
    static_assert(ByteOffset1 >= 0 && ByteOffset1 < (1 << 20) &&
                      ByteOffset2 >= 0 && ByteOffset2 < (1 << 20),
                  "LDS sidecar offset must fit ds_read_b128");
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
    asm volatile(
        "ds_read_b128 %0, %3 offset:0\n\t"
        "ds_read_b128 %1, %3 offset:%4\n\t"
        "ds_read_b128 %2, %3 offset:%5\n"
        : "=v"(frag0), "=v"(frag1), "=v"(frag2)
        : "v"(lds_addr), "n"(ByteOffset1), "n"(ByteOffset2)
        : "memory");
#else
    (void)lds;
    frag0 = {};
    frag1 = {};
    frag2 = {};
#endif
}

template <int ByteOffset1>
__device__ __forceinline__ void ds_read_b128_lds_imm2(
    const float* lds,
    Vec4F32& frag0,
    Vec4F32& frag1) {
    static_assert(ByteOffset1 >= 0 && ByteOffset1 < (1 << 20),
                  "LDS sidecar offset must fit ds_read_b128");
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
    asm volatile(
        "ds_read_b128 %0, %2 offset:0\n\t"
        "ds_read_b128 %1, %2 offset:%3\n"
        : "=v"(frag0), "=v"(frag1)
        : "v"(lds_addr), "n"(ByteOffset1)
        : "memory");
#else
    (void)lds;
    frag0 = {};
    frag1 = {};
#endif
}

template <int ByteOffset>
__device__ __forceinline__ void ds_read_b128_lds_imm1(
    const float* lds,
    Vec4F32& frag) {
    static_assert(ByteOffset >= 0 && ByteOffset < (1 << 20),
                  "LDS sidecar offset must fit ds_read_b128");
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
    asm volatile("ds_read_b128 %0, %1 offset:%2"
                 : "=v"(frag)
                 : "v"(lds_addr), "n"(ByteOffset)
                 : "memory");
#else
    (void)lds;
    frag = {};
#endif
}

template <typename Vec>
__device__ __forceinline__ void zero_vgpr2(Vec& dst) {
#if defined(__gfx946__) || defined(__gfx92a__)
    asm volatile("v_mov_b64 %0, 0x0" : "=v"(dst) :);
    __builtin_amdgcn_sched_barrier(0);
#else
    dst = Vec{};
#endif
}

__device__ __forceinline__ void zero_f16x8(F16x8& frag) {
    zero_vgpr2(frag.u64[0]);
    zero_vgpr2(frag.u64[1]);
    asm volatile("" : "+v"(frag.f16x8) : : "memory");
}

__device__ __forceinline__ Vec4U32 prepare_matrix_src(const __half* ptr,
                                                       int row_stride) {
    Vec4U32 srsrc{};
    *reinterpret_cast<unsigned long long*>(&srsrc) =
        reinterpret_cast<unsigned long long>(ptr) & 0xffffffffffffULL;
    srsrc[2] = static_cast<uint32_t>(row_stride);
    srsrc[3] = 0;
    return srsrc;
}

__device__ __forceinline__ void matrix_load_32x32_b16_bps_lds(
    __half* shared_addr,
    Vec4U32 srsrc,
    int lds_offset,
    bool transpose) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(shared_addr)) + lds_offset;
    if (transpose) {
        asm volatile(
            "s_nop 0\n\t"
            "matrix_load_32x32_b16 %0, %1 moffset:%2 t bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    } else {
        asm volatile(
            "s_nop 0\n\t"
            "matrix_load_32x32_b16 %0, %1 moffset:%2 bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            : "memory");
    }
#else
    (void)shared_addr;
    (void)srsrc;
    (void)lds_offset;
    (void)transpose;
#endif
}

__device__ __forceinline__ void matrix_load_32x16_b16_bps_lds(
    __half* shared_addr,
    Vec4U32 srsrc,
    int lds_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t lds_addr_v =
        static_cast<uint32_t>(reinterpret_cast<size_t>(shared_addr)) +
        static_cast<uint32_t>(lds_offset);
    const int lds_addr =
        __builtin_amdgcn_readfirstlane(static_cast<int>(lds_addr_v));
    asm volatile(
        "s_nop 0\n\t"
        "s_mov_b32 m0, %1\n\t"
        "matrix_load_32x16_b16 %0, m0 moffset:%2 bps lds\n"
        :
        : "s"(srsrc), "s"(lds_addr), "n"(0)
        : "memory");
#else
    (void)shared_addr;
    (void)srsrc;
    (void)lds_offset;
#endif
}

__device__ __forceinline__ void ds_read_matrix_32x16_normal(
    const __half* lds,
    int lds_offset,
    Vec8F16& frag) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_offset;
    asm volatile(
        "ds_read_matrix_format %0, %1 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag)
        : "s"(lds_addr)
        : "memory");
#else
    (void)lds;
    (void)lds_offset;
    frag = {};
#endif
}

template <int LdsOffset0, int LdsOffset1, int LdsOffset2, int LdsOffset3>
__device__ __forceinline__ void ds_read_matrix_32x16_normal_imm4(
    const __half* lds,
    Vec8F16& frag0,
    Vec8F16& frag1,
    Vec8F16& frag2,
    Vec8F16& frag3) {
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 16) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 16) &&
                      LdsOffset2 >= 0 && LdsOffset2 < (1 << 16) &&
                      LdsOffset3 >= 0 && LdsOffset3 < (1 << 16),
                  "DS matrix-read immediate must fit the byte offset field");
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr = static_cast<int>(reinterpret_cast<size_t>(lds));
    asm volatile(
        "ds_read_matrix_format %0, %4 offset:%5 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %1, %4 offset:%6 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %2, %4 offset:%7 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %3, %4 offset:%8 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag0), "=v"(frag1), "=v"(frag2), "=v"(frag3)
        : "s"(lds_addr), "n"(LdsOffset0), "n"(LdsOffset1),
          "n"(LdsOffset2), "n"(LdsOffset3)
        : "memory");
#else
    (void)lds;
    frag0 = {};
    frag1 = {};
    frag2 = {};
    frag3 = {};
#endif
}

template <int LdsOffset0, int LdsOffset1>
__device__ __forceinline__ void ds_read_matrix_32x16_normal_dual_base_imm2(
    const __half* lhs_lds,
    const __half* rhs_lds,
    Vec8F16& lhs0,
    Vec8F16& lhs1,
    Vec8F16& rhs0,
    Vec8F16& rhs1) {
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 16) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 16),
                  "DS matrix-read immediate must fit the byte offset field");
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lhs_addr = static_cast<int>(reinterpret_cast<size_t>(lhs_lds));
    const int rhs_addr = static_cast<int>(reinterpret_cast<size_t>(rhs_lds));
    asm volatile(
        "ds_read_matrix_format %0, %4 offset:%6 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %1, %4 offset:%7 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %2, %5 offset:%6 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %3, %5 offset:%7 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(lhs0), "=v"(lhs1), "=v"(rhs0), "=v"(rhs1)
        : "s"(lhs_addr), "s"(rhs_addr), "n"(LdsOffset0),
          "n"(LdsOffset1)
        : "memory");
#else
    (void)lhs_lds;
    (void)rhs_lds;
    lhs0 = {};
    lhs1 = {};
    rhs0 = {};
    rhs1 = {};
#endif
}

__device__ __forceinline__ void ds_read_matrix_32x16_trans(
    const __half* lds,
    int lds_offset,
    Vec8F16& frag) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_offset;
    asm volatile(
        "ds_read_matrix_trans_format %0, %1 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag)
        : "s"(lds_addr)
        : "memory");
#else
    (void)lds;
    (void)lds_offset;
    frag = {};
#endif
}

template <int LdsOffset0, int LdsOffset1>
__device__ __forceinline__ void ds_read_matrix_32x16_trans_imm2(
    const __half* lds,
    Vec8F16& frag0,
    Vec8F16& frag1) {
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 16) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 16),
                  "DS matrix-read immediate must fit the byte offset field");
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr = static_cast<int>(reinterpret_cast<size_t>(lds));
    asm volatile(
        "ds_read_matrix_trans_format %0, %2 offset:%3 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %1, %2 offset:%4 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag0), "=v"(frag1)
        : "s"(lds_addr), "n"(LdsOffset0), "n"(LdsOffset1)
        : "memory");
#else
    (void)lds;
    frag0 = {};
    frag1 = {};
#endif
}

template <int LdsOffset0, int LdsOffset1, int LdsOffset2, int LdsOffset3>
__device__ __forceinline__ void ds_read_matrix_32x16_trans_imm4(
    const __half* lds,
    Vec8F16& frag0,
    Vec8F16& frag1,
    Vec8F16& frag2,
    Vec8F16& frag3) {
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 16) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 16) &&
                      LdsOffset2 >= 0 && LdsOffset2 < (1 << 16) &&
                      LdsOffset3 >= 0 && LdsOffset3 < (1 << 16),
                  "DS matrix-read immediate must fit the byte offset field");
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr = static_cast<int>(reinterpret_cast<size_t>(lds));
    asm volatile(
        "ds_read_matrix_trans_format %0, %4 offset:%5 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %1, %4 offset:%6 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %2, %4 offset:%7 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %3, %4 offset:%8 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag0), "=v"(frag1), "=v"(frag2), "=v"(frag3)
        : "s"(lds_addr), "n"(LdsOffset0), "n"(LdsOffset1),
          "n"(LdsOffset2), "n"(LdsOffset3)
        : "memory");
#else
    (void)lds;
    frag0 = {};
    frag1 = {};
    frag2 = {};
    frag3 = {};
#endif
}

template <int LdsOffset0, int LdsOffset1, int LdsOffset2, int LdsOffset3>
__device__ __forceinline__ void ds_read_matrix_32x16_trans_dual_base_imm4(
    const __half* lhs_lds,
    const __half* rhs_lds,
    Vec8F16& lhs0,
    Vec8F16& rhs0,
    Vec8F16& lhs1,
    Vec8F16& rhs1,
    Vec8F16& lhs2,
    Vec8F16& rhs2,
    Vec8F16& lhs3,
    Vec8F16& rhs3) {
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 16) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 16) &&
                      LdsOffset2 >= 0 && LdsOffset2 < (1 << 16) &&
                      LdsOffset3 >= 0 && LdsOffset3 < (1 << 16),
                  "DS matrix-read immediate must fit the byte offset field");
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lhs_addr = static_cast<int>(reinterpret_cast<size_t>(lhs_lds));
    const int rhs_addr = static_cast<int>(reinterpret_cast<size_t>(rhs_lds));
    asm volatile(
        "ds_read_matrix_trans_format %0, %8 offset:%10 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %1, %9 offset:%10 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %2, %8 offset:%11 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %3, %9 offset:%11 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %4, %8 offset:%12 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %5, %9 offset:%12 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %6, %8 offset:%13 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %7, %9 offset:%13 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(lhs0), "=v"(rhs0), "=v"(lhs1), "=v"(rhs1),
          "=v"(lhs2), "=v"(rhs2), "=v"(lhs3), "=v"(rhs3)
        : "s"(lhs_addr), "s"(rhs_addr), "n"(LdsOffset0),
          "n"(LdsOffset1), "n"(LdsOffset2), "n"(LdsOffset3)
        : "memory");
#else
    (void)lhs_lds;
    (void)rhs_lds;
    lhs0 = {};
    rhs0 = {};
    lhs1 = {};
    rhs1 = {};
    lhs2 = {};
    rhs2 = {};
    lhs3 = {};
    rhs3 = {};
#endif
}

template <int LdsOffset0, int LdsOffset1>
__device__ __forceinline__ void ds_read_matrix_32x16_trans_dual_base_imm2(
    const __half* lhs_lds,
    const __half* rhs_lds,
    Vec8F16& lhs0,
    Vec8F16& rhs0,
    Vec8F16& lhs1,
    Vec8F16& rhs1) {
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 16) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 16),
                  "DS matrix-read immediate must fit the byte offset field");
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lhs_addr = static_cast<int>(reinterpret_cast<size_t>(lhs_lds));
    const int rhs_addr = static_cast<int>(reinterpret_cast<size_t>(rhs_lds));
    asm volatile(
        "ds_read_matrix_trans_format %0, %4 offset:%6 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %1, %5 offset:%6 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %2, %4 offset:%7 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %3, %5 offset:%7 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(lhs0), "=v"(rhs0), "=v"(lhs1), "=v"(rhs1)
        : "s"(lhs_addr), "s"(rhs_addr), "n"(LdsOffset0),
          "n"(LdsOffset1)
        : "memory");
#else
    (void)lhs_lds;
    (void)rhs_lds;
    lhs0 = {};
    rhs0 = {};
    lhs1 = {};
    rhs1 = {};
#endif
}

__device__ __forceinline__ void ds_write_matrix_32x16_f16(
    Vec8F16 frag,
    __half* lds,
    int lds_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(lds) + lds_offset);
    // ptr already names the target page. The third builtin operand is an LDS
    // byte offset, not the element width.
    __builtin_hcu_ds_write_matrix_format_f16(frag, ptr, 0, 2, 1, 0, 0);
#else
    (void)frag;
    (void)lds;
    (void)lds_offset;
#endif
}

// Native dS/P handoff ABI proved by the D128 dense probe.  The pointer names
// the page, so the builtin byte offset remains zero.
__device__ __forceinline__ void ds_write_matrix_32x16_trans_f16(
    Vec8F16 frag,
    __half* lds,
    int lds_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(lds) + lds_offset);
    __builtin_hcu_ds_write_matrix_format_f16(frag, ptr, 0, 2, 1, 0, 1);
#else
    (void)frag;
    (void)lds;
    (void)lds_offset;
#endif
}

__device__ __forceinline__ void ds_read_matrix_trans_pair(
    const __half* lds,
    int lds_offset,
    Vec8F16& frag0,
    Vec8F16& frag1) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_offset;
    asm volatile(
        "ds_read_matrix_trans_format %0, %2 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %1, %2 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag0), "=v"(frag1)
        : "s"(lds_addr)
        : "memory");
#else
    (void)lds;
    (void)lds_offset;
    frag0 = {};
    frag1 = {};
#endif
}

__device__ __forceinline__ void ds_read_matrix_normal_pair(
    const __half* lds,
    int lds_offset,
    Vec8F16& frag0,
    Vec8F16& frag1) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int lds_addr =
        static_cast<int>(reinterpret_cast<size_t>(lds)) + lds_offset;
    asm volatile(
        "ds_read_matrix_format %0, %2 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_format %1, %2 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag0), "=v"(frag1)
        : "s"(lds_addr)
        : "memory");
#else
    (void)lds;
    (void)lds_offset;
    frag0 = {};
    frag1 = {};
#endif
}

__device__ __forceinline__ void wait_lgkm(int count = 0) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(%0)\n" : : "n"(count) : "memory");
    __builtin_amdgcn_sched_barrier(0);
#else
    (void)count;
#endif
}

__device__ __forceinline__ void wait_vbcnt0() {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt_vbcnt 0\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
}

__device__ __forceinline__ void maybe_wait_bps_vbcnt_before_arrive() {
#if SHAOBO_BPS_VBCNT_BEFORE_ARRIVE
    wait_vbcnt0();
#endif
}

__device__ __forceinline__ void wait_vmem_lgkm() {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
}

__device__ __forceinline__ Vec4F32 mmac_f16_lit(Vec4F16 lhs,
                                                Vec4F16 rhs,
                                                Vec4F32 acc) {
#if defined(__gfx946__) || defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(lhs, rhs, acc, 1, 0);
#else
    (void)lhs;
    (void)rhs;
    return acc;
#endif
}

__device__ __forceinline__ void keep_accumulator_live(F32x4& acc) {
#if defined(__gfx946__) || defined(__gfx938__) || defined(__gfx92a__)
    asm volatile("" : "+v"(acc.f32) : : "memory");
    __builtin_amdgcn_sched_barrier(0);
#else
    (void)acc;
#endif
}

__device__ __forceinline__ void s_barrier_after_accumulators(F32x4& acc0,
                                                             F32x4& acc1) {
#if defined(__gfx946__) || defined(__gfx938__) || defined(__gfx92a__)
    asm volatile("s_barrier\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#else
    (void)acc0;
    (void)acc1;
    __syncthreads();
#endif
}

__device__ __forceinline__ void s_barrier_after_accumulators(F32x4& acc0,
                                                             F32x4& acc1,
                                                             F32x4& acc2,
                                                             F32x4& acc3) {
#if defined(__gfx946__) || defined(__gfx938__) || defined(__gfx92a__)
    asm volatile("s_barrier\n" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#else
    (void)acc0;
    (void)acc1;
    (void)acc2;
    (void)acc3;
    __syncthreads();
#endif
}

template <bool UseAsm>
__device__ __forceinline__ void abarrier_try_wait(int abarrier_id,
                                                  int& target_phase) {
    if constexpr (UseAsm) {
#if defined(__gfx946__) || defined(__gfx92a__)
        __builtin_amdgcn_sched_barrier(0);
        int state;
        asm volatile(
            "s_abarrier_try_wait %0, %2, %1\n\t"
            "s_xor_b32 %1, %1, 1\n"
            : "=s"(state), "+s"(target_phase)
            : "n"(abarrier_id)
            :);
        __builtin_amdgcn_sched_barrier(0);
#else
        (void)abarrier_id;
        target_phase ^= 1;
#endif
    } else {
        __builtin_hcu_s_abarrier_try_wait(abarrier_id, target_phase);
        target_phase ^= 1;
    }
}

template <bool UseAsm>
__device__ __forceinline__ void abarrier_arrive_cnt(int abarrier_id,
                                                    int arrive_cnt) {
    if constexpr (UseAsm) {
#if defined(__gfx946__) || defined(__gfx92a__)
        __builtin_amdgcn_sched_barrier(0);
        int state;
        asm volatile("s_abarrier_arrive %0, %1, %2\n"
                     : "=s"(state)
                     : "n"(abarrier_id), "n"(arrive_cnt)
                     :);
        __builtin_amdgcn_sched_barrier(0);
#else
        (void)abarrier_id;
        (void)arrive_cnt;
#endif
    } else {
        __builtin_hcu_s_abarrier_arrive_cnt(abarrier_id, arrive_cnt);
    }
}

template <bool UseAsm>
__device__ __forceinline__ void abarrier_seq(int abarrier_id) {
    if constexpr (UseAsm) {
#if defined(__gfx946__) || defined(__gfx92a__)
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_abarrier_seq %0\n" : : "n"(abarrier_id));
        __builtin_amdgcn_sched_barrier(0);
#else
        (void)abarrier_id;
#endif
    } else {
        __builtin_hcu_s_abarrier_seq(abarrier_id);
    }
}

__device__ __forceinline__ void raise_priority_2() {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio 2" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
}

__device__ __forceinline__ void lower_priority() {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio 0" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
}

}  // namespace shaobo::fa3::bwd::dkv::instr

namespace shaobo::fa3::bwd::instr {
using namespace shaobo::fa3::bwd::dkv::instr;
}  // namespace shaobo::fa3::bwd::instr
