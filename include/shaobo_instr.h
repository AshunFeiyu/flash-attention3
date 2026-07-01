#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shaobo::fa3::bwd::dkv::instr {

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
            :);
    } else {
        asm volatile(
            "s_nop 0\n\t"
            "matrix_load_32x32_b16 %0, %1 moffset:%2 bps lds\n"
            :
            : "s"(srsrc), "s"(lds_addr), "n"(0)
            :);
    }
#else
    (void)shared_addr;
    (void)srsrc;
    (void)lds_offset;
    (void)transpose;
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
        "s_nop 0\n\t"
        "ds_read_matrix_trans_format %0, %2 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t"
        "ds_read_matrix_trans_format %1, %2 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n"
        : "=v"(frag0), "=v"(frag1)
        : "s"(lds_addr)
        :);
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
