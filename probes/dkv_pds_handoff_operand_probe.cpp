#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kWriterModes = 4;
constexpr int kReaderModes = 5;
constexpr int kCandidates = kWriterModes * kReaderModes;
constexpr int kOutputsPerCandidate = 2;
constexpr int kFloatsPerOutput = 4;
constexpr int kPageElems = 1024;
constexpr int kSourceScheduledPage = kWriterModes * 2 * kPageElems;
constexpr int kLdsElems = kSourceScheduledPage + 2 * kPageElems;
constexpr float kTolerance = 1.0e-5f;

union ProbeF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union ProbeF32x4 {
    ins::Vec4F32 f32;
    float scalar[4];
};

__device__ __forceinline__ _Float16 half_from_bits(uint16_t bits) {
    union {
        uint16_t u;
        _Float16 h;
    } value{bits};
    return value.h;
}

__device__ __forceinline__ uint16_t half_to_bits(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } bits{value};
    return bits.u;
}

__device__ __forceinline__ bool source_slot_to_normal_read(
    int src_lane,
    int src_word,
    int& read_lane,
    int& read_word) {
    const int carry = src_lane < 4 ? 1 : 0;
    const int adjusted_word = src_word - 2 * carry;
    if (adjusted_word < 0) {
        return false;
    }
    const int base = src_lane + 64 * carry - 4;
    const int column = base >> 4;
    const int rem = base & 15;
    const int lane_half = rem & 7;
    read_word = 4 * (rem >> 3) + column;
    read_lane = 16 * (adjusted_word >> 1) + 2 * lane_half +
                (adjusted_word & 1);
    return read_lane < kWaveSize && read_word < 8;
}

__device__ __forceinline__ ProbeF32x4 run_pair_mmac(
    const ProbeF16x8& lhs,
    const ProbeF16x8& rhs) {
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    ProbeF32x4 out{};
#if defined(__gfx946__) || defined(__gfx92a__)
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], zero.f32);
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[1], rhs.f16x4[1], out.f32);
#else
    (void)lhs;
    (void)rhs;
#endif
    return out;
}

__device__ __forceinline__ void store_output(float* out,
                                              int candidate,
                                              int output,
                                              int lane,
                                              const ProbeF32x4& value) {
    const int base =
        (((candidate * kOutputsPerCandidate + output) * kWaveSize + lane) *
         kFloatsPerOutput);
#pragma unroll
    for (int i = 0; i < kFloatsPerOutput; ++i) {
        out[base + i] = value.scalar[i];
    }
}

template <int Transpose, int Alt>
__device__ __forceinline__ void write_matrix(ProbeF16x8 frag,
                                              _Float16* lds) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __builtin_hcu_ds_write_matrix_format_f16(
        frag.f16x8, lds, 16, 2, 1, Transpose, Alt);
#else
    (void)frag;
    (void)lds;
#endif
}

template <int ReaderMode>
__device__ __forceinline__ ProbeF16x8 read_matrix(const _Float16* lds) {
    ProbeF16x8 frag{};
#if defined(__gfx946__) || defined(__gfx92a__)
    if constexpr (ReaderMode == 0) {
        frag.f16x8 = __builtin_hcu_ds_read_matrix_format_f16(
            const_cast<_Float16*>(lds), 16, 2, 1, 0);
    } else if constexpr (ReaderMode == 1) {
        frag.f16x8 = __builtin_hcu_ds_read_matrix_format_f16(
            const_cast<_Float16*>(lds), 16, 2, 1, 1);
    } else if constexpr (ReaderMode == 2) {
        frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
            const_cast<_Float16*>(lds), 16, 2, 1, 0);
    } else if constexpr (ReaderMode == 3) {
        frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
            const_cast<_Float16*>(lds), 16, 1, 2, 0);
    } else {
        static_assert(ReaderMode == 4, "unsupported reader mode");
        frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
            const_cast<_Float16*>(lds), 16, 1, 2, 1);
    }
#else
    (void)lds;
#endif
    return frag;
}

template <int WriterMode>
__device__ __forceinline__ void publish_pair(ProbeF16x8 p,
                                              ProbeF16x8 ds,
                                              _Float16* lds) {
    constexpr int kWriterTranspose = WriterMode / 2;
    constexpr int kWriterAlt = WriterMode % 2;
    _Float16* p_page = lds + WriterMode * 2 * kPageElems;
    _Float16* ds_page = p_page + kPageElems;
    write_matrix<kWriterTranspose, kWriterAlt>(p, p_page);
    write_matrix<kWriterTranspose, kWriterAlt>(ds, ds_page);
}

template <int WriterMode, int ReaderMode>
__device__ __forceinline__ void consume_pair(const _Float16* lds,
                                              const ProbeF16x8& dout,
                                              const ProbeF16x8& q,
                                              float* out,
                                              int lane) {
    static_assert(ReaderMode >= 0 && ReaderMode < kReaderModes,
                  "unsupported reader mode");
    constexpr int kCandidate = WriterMode * kReaderModes + ReaderMode;
    const _Float16* p_page = lds + WriterMode * 2 * kPageElems;
    const _Float16* ds_page = p_page + kPageElems;
    const ProbeF16x8 p =
        read_matrix<ReaderMode>(p_page);
    const ProbeF16x8 ds =
        read_matrix<ReaderMode>(ds_page);
    ins::wait_lgkm(0);
    store_output(out, kCandidate, 0, lane, run_pair_mmac(p, dout));
    store_output(out, kCandidate, 1, lane, run_pair_mmac(ds, q));
}

__global__ void __launch_bounds__(kWaveSize, 1)
    dkv_pds_handoff_operand_probe_kernel(float* __restrict__ direct,
                                         float* __restrict__ candidates,
                                         uint16_t* __restrict__ normal_map,
                                         float* __restrict__ source_scheduled) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 lds[kLdsElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);

    ProbeF16x8 p{};
    ProbeF16x8 ds{};
    ProbeF16x8 dout{};
    ProbeF16x8 q{};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const uint16_t slot = static_cast<uint16_t>(lane * 8 + word);
        p.scalar[word] = half_from_bits(static_cast<uint16_t>(0x2800 + slot));
        ds.scalar[word] =
            half_from_bits(static_cast<uint16_t>(0x2c00 + slot));
        dout.scalar[word] = half_from_bits(
            static_cast<uint16_t>(0x3000 + ((lane * 3 + word) & 0x3ff)));
        q.scalar[word] = half_from_bits(
            static_cast<uint16_t>(0x3400 + ((lane * 5 + word) & 0x3ff)));
    }

    const ProbeF32x4 direct_dv = run_pair_mmac(p, dout);
    const ProbeF32x4 direct_dk = run_pair_mmac(ds, q);
#pragma unroll
    for (int i = 0; i < kFloatsPerOutput; ++i) {
        direct[(0 * kWaveSize + lane) * kFloatsPerOutput + i] =
            direct_dv.scalar[i];
        direct[(1 * kWaveSize + lane) * kFloatsPerOutput + i] =
            direct_dk.scalar[i];
    }

    publish_pair<0>(p, ds, lds);
    publish_pair<1>(p, ds, lds);
    publish_pair<2>(p, ds, lds);
    publish_pair<3>(p, ds, lds);
    ins::wait_lgkm(0);

    const ProbeF16x8 mapped = read_matrix<0>(lds);
    ins::wait_lgkm(0);
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        normal_map[lane * 8 + word] = half_to_bits(mapped.scalar[word]);
    }

    consume_pair<0, 0>(lds, dout, q, candidates, lane);
    consume_pair<0, 1>(lds, dout, q, candidates, lane);
    consume_pair<0, 2>(lds, dout, q, candidates, lane);
    consume_pair<0, 3>(lds, dout, q, candidates, lane);
    consume_pair<0, 4>(lds, dout, q, candidates, lane);
    consume_pair<1, 0>(lds, dout, q, candidates, lane);
    consume_pair<1, 1>(lds, dout, q, candidates, lane);
    consume_pair<1, 2>(lds, dout, q, candidates, lane);
    consume_pair<1, 3>(lds, dout, q, candidates, lane);
    consume_pair<1, 4>(lds, dout, q, candidates, lane);
    consume_pair<2, 0>(lds, dout, q, candidates, lane);
    consume_pair<2, 1>(lds, dout, q, candidates, lane);
    consume_pair<2, 2>(lds, dout, q, candidates, lane);
    consume_pair<2, 3>(lds, dout, q, candidates, lane);
    consume_pair<2, 4>(lds, dout, q, candidates, lane);
    consume_pair<3, 0>(lds, dout, q, candidates, lane);
    consume_pair<3, 1>(lds, dout, q, candidates, lane);
    consume_pair<3, 2>(lds, dout, q, candidates, lane);
    consume_pair<3, 3>(lds, dout, q, candidates, lane);
    consume_pair<3, 4>(lds, dout, q, candidates, lane);

    ProbeF16x8 p_source{};
    ProbeF16x8 ds_source{};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        int read_lane = -1;
        int read_word = -1;
        const bool valid =
            source_slot_to_normal_read(lane, word, read_lane, read_word);
        const uint16_t slot = static_cast<uint16_t>(read_lane * 8 + read_word);
        p_source.scalar[word] = valid
                                    ? half_from_bits(
                                          static_cast<uint16_t>(0x2800 + slot))
                                    : static_cast<_Float16>(0.0f);
        ds_source.scalar[word] = valid
                                     ? half_from_bits(static_cast<uint16_t>(
                                           0x2c00 + slot))
                                     : static_cast<_Float16>(0.0f);
    }
    write_matrix<0, 0>(p_source, lds + kSourceScheduledPage);
    write_matrix<0, 0>(ds_source,
                       lds + kSourceScheduledPage + kPageElems);
    ins::wait_lgkm(0);
    const ProbeF16x8 p_scheduled =
        read_matrix<0>(lds + kSourceScheduledPage);
    const ProbeF16x8 ds_scheduled =
        read_matrix<0>(lds + kSourceScheduledPage + kPageElems);
    ins::wait_lgkm(0);
    const ProbeF32x4 scheduled_dv = run_pair_mmac(p_scheduled, dout);
    const ProbeF32x4 scheduled_dk = run_pair_mmac(ds_scheduled, q);
#pragma unroll
    for (int i = 0; i < kFloatsPerOutput; ++i) {
        source_scheduled[(0 * kWaveSize + lane) * kFloatsPerOutput + i] =
            scheduled_dv.scalar[i];
        source_scheduled[(1 * kWaveSize + lane) * kFloatsPerOutput + i] =
            scheduled_dk.scalar[i];
    }
#else
    (void)direct;
    (void)candidates;
    (void)normal_map;
    (void)source_scheduled;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(2);
    }
}

const char* candidate_name(int candidate) {
    static const char* names[kCandidates] = {
        "write_n_a0_read_n_a0", "write_n_a0_read_n_a1",
        "write_n_a0_read_t32_a0", "write_n_a0_read_t16_a0",
        "write_n_a0_read_t16_a1", "write_n_a1_read_n_a0",
        "write_n_a1_read_n_a1", "write_n_a1_read_t32_a0",
        "write_n_a1_read_t16_a0", "write_n_a1_read_t16_a1",
        "write_t_a0_read_n_a0", "write_t_a0_read_n_a1",
        "write_t_a0_read_t32_a0", "write_t_a0_read_t16_a0",
        "write_t_a0_read_t16_a1", "write_t_a1_read_n_a0",
        "write_t_a1_read_n_a1", "write_t_a1_read_t32_a0",
        "write_t_a1_read_t16_a0", "write_t_a1_read_t16_a1",
    };
    return names[candidate];
}

}  // namespace

int main() {
    constexpr int kDirectFloats =
        kOutputsPerCandidate * kWaveSize * kFloatsPerOutput;
    constexpr int kCandidateFloats = kCandidates * kDirectFloats;
    float* direct_dev = nullptr;
    float* candidates_dev = nullptr;
    uint16_t* normal_map_dev = nullptr;
    float* source_scheduled_dev = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&direct_dev),
                        kDirectFloats * sizeof(float)),
              "hipMalloc direct");
    check_hip(hipMalloc(reinterpret_cast<void**>(&candidates_dev),
                        kCandidateFloats * sizeof(float)),
              "hipMalloc candidates");
    check_hip(hipMalloc(reinterpret_cast<void**>(&normal_map_dev),
                        kWaveSize * 8 * sizeof(uint16_t)),
              "hipMalloc normal_map");
    check_hip(hipMalloc(reinterpret_cast<void**>(&source_scheduled_dev),
                        kDirectFloats * sizeof(float)),
              "hipMalloc source_scheduled");

    hipLaunchKernelGGL(dkv_pds_handoff_operand_probe_kernel, dim3(1),
                       dim3(kWaveSize), 0, 0, direct_dev, candidates_dev,
                       normal_map_dev, source_scheduled_dev);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    std::vector<float> direct(kDirectFloats);
    std::vector<float> candidates(kCandidateFloats);
    std::vector<uint16_t> normal_map(kWaveSize * 8);
    std::vector<float> source_scheduled(kDirectFloats);
    check_hip(hipMemcpy(direct.data(), direct_dev,
                        direct.size() * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy direct");
    check_hip(hipMemcpy(candidates.data(), candidates_dev,
                        candidates.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy candidates");
    check_hip(hipMemcpy(normal_map.data(), normal_map_dev,
                        normal_map.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy normal_map");
    check_hip(hipMemcpy(source_scheduled.data(), source_scheduled_dev,
                        source_scheduled.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy source_scheduled");
    check_hip(hipFree(source_scheduled_dev), "hipFree source_scheduled");
    check_hip(hipFree(normal_map_dev), "hipFree normal_map");
    check_hip(hipFree(candidates_dev), "hipFree candidates");
    check_hip(hipFree(direct_dev), "hipFree direct");

    bool any_pass = false;
    for (int candidate = 0; candidate < kCandidates; ++candidate) {
        int errors = 0;
        float max_abs = 0.0f;
        for (int i = 0; i < kDirectFloats; ++i) {
            const float diff = std::fabs(
                candidates[candidate * kDirectFloats + i] - direct[i]);
            max_abs = std::max(max_abs, diff);
            errors += diff > kTolerance ? 1 : 0;
        }
        const bool pass = errors == 0;
        any_pass |= pass;
        std::printf("pds_handoff candidate=%s errors=%d max_abs=%g pass=%d\n",
                    candidate_name(candidate), errors, max_abs,
                    pass ? 1 : 0);
    }
    std::printf("pds_handoff_operand_probe any_native_pair=%d\n",
                any_pass ? 1 : 0);
    int scheduled_errors = 0;
    float scheduled_max_abs = 0.0f;
    for (int i = 0; i < kDirectFloats; ++i) {
        const float diff = std::fabs(source_scheduled[i] - direct[i]);
        scheduled_max_abs = std::max(scheduled_max_abs, diff);
        scheduled_errors += diff > kTolerance ? 1 : 0;
    }
    const bool source_scheduled_pass = scheduled_errors == 0;
    std::printf(
        "pds_source_scheduled_normal errors=%d max_abs=%g pass=%d\n",
        scheduled_errors, scheduled_max_abs,
        source_scheduled_pass ? 1 : 0);
    for (int read_lane = 0; read_lane < kWaveSize; ++read_lane) {
        for (int read_word = 0; read_word < 8; ++read_word) {
            const int slot = static_cast<int>(
                normal_map[read_lane * 8 + read_word]) - 0x2800;
            std::printf(
                "pds_normal_map read_lane=%d read_word=%d src_lane=%d "
                "src_word=%d\n",
                read_lane, read_word, slot >> 3, slot & 7);
        }
    }
    return source_scheduled_pass ? 0 : 3;
}
