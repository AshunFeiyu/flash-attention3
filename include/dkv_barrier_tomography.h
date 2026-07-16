#pragma once

#include "shaobo_instr.h"

namespace shaobo::fa3::bwd::dkv::barrier_tomography {

#ifndef SHAOBO_ABARRIER_TOMOGRAPHY
#define SHAOBO_ABARRIER_TOMOGRAPHY 0
#endif

#if SHAOBO_ABARRIER_TOMOGRAPHY
#define SHAOBO_DKV_TOMO_TRY_WAIT(BarrierId, TargetPhase)                    \
    do {                                                                    \
        __builtin_amdgcn_sched_barrier(0);                                  \
        int state;                                                          \
        asm volatile(                                                       \
            "s_abarrier_try_wait %0, %2, %1\n\t"                           \
            "s_xor_b32 %1, %1, 1\n"                                       \
            : "=s"(state), "+s"(TargetPhase)                              \
            : "n"(BarrierId)                                               \
            :);                                                             \
        __builtin_amdgcn_sched_barrier(0);                                  \
    } while (false)
#else
#define SHAOBO_DKV_TOMO_TRY_WAIT(BarrierId, TargetPhase)                    \
    ::shaobo::fa3::bwd::dkv::instr::abarrier_try_wait<true>(                \
        BarrierId, TargetPhase)
#endif

// Each wrapper intentionally owns a distinct debug line. Diagnostic builds use
// those lines to split XCU issue gaps without adding runtime marker instructions.
template <typename Barrier>
__device__ __forceinline__ void wait_resident_filled_consumer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kResidentFilled, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_resident_used_kq_producer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kResidentUsed, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_resident_used_vdout_producer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kResidentUsed, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_q0_filled_consumer_first(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kQ0Filled, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_q1_filled_consumer_first(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kQ1Filled, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_q0_filled_consumer_loop(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kQ0Filled, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_q1_filled_consumer_loop(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kQ1Filled, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_q0_used_kq_producer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kQ0Used, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_q1_used_kq_producer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kQ1Used, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_dout0_used_vdout_producer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kDout0Used, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_dout1_used_vdout_producer(int& phase) {
    SHAOBO_DKV_TOMO_TRY_WAIT(Barrier::kDout1Used, phase);
}

template <typename Barrier>
__device__ __forceinline__ void wait_all_done(int& phase) {
    instr::abarrier_try_wait<false>(Barrier::kAllDone, phase);
}

#undef SHAOBO_DKV_TOMO_TRY_WAIT

}  // namespace shaobo::fa3::bwd::dkv::barrier_tomography
