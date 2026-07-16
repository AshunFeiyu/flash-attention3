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
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 18) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 18) &&
                      LdsOffset2 >= 0 && LdsOffset2 < (1 << 18) &&
                      LdsOffset3 >= 0 && LdsOffset3 < (1 << 18),
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
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 18) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 18),
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
    static_assert(LdsOffset0 >= 0 && LdsOffset0 < (1 << 18) &&
                      LdsOffset1 >= 0 && LdsOffset1 < (1 << 18) &&
                      LdsOffset2 >= 0 && LdsOffset2 < (1 << 18) &&
                      LdsOffset3 >= 0 && LdsOffset3 < (1 << 18),
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

__device__ __forceinline__ void ds_write_matrix_32x16_f16(
    Vec8F16 frag,
    __half* lds,
    int lds_offset) {
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(lds) + lds_offset);
    __builtin_hcu_ds_write_matrix_format_f16(frag, ptr, 16, 2, 1, 0, 0);
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
