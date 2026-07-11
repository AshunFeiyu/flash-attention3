#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kThreads = 64;
constexpr int kWaveSize = 64;
constexpr int kFragWords = 8;
constexpr int kFragElems = kWaveSize * kFragWords;
constexpr int kRows = 32;
constexpr int kCols = 64;
constexpr int kProbeK = 7;
constexpr int kKTagBase = 1024;
constexpr int kMmacFloatsPerLane = 4;
constexpr int kDsPageWords = 0;
constexpr int kKPageBytes = 4096;
constexpr int kLdsWords = 4096;

union ProbeF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[kFragWords];
    uint16_t u16[kFragWords];
};

union ProbeF32x4 {
    ins::Vec4F32 f32;
    float scalar[kMmacFloatsPerLane];
};

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(2);
    }
}

uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

int decode_half_tag(uint16_t bits) {
    union {
        uint16_t u;
        _Float16 h;
    } value{bits};
    const float as_float = static_cast<float>(value.h);
    const int tag = static_cast<int>(std::lround(as_float));
    if (tag < 1 || tag > kFragElems) {
        return -1;
    }
    if (std::fabs(as_float - static_cast<float>(tag)) > 0.125f) {
        return -1;
    }
    return tag;
}

float q_scale_for_decode(int q) {
    return 1.0f + static_cast<float>(q) * (7.0f / 512.0f);
}

__device__ __forceinline__ ProbeF32x4 mmac_pair_lit(const ProbeF16x8& lhs,
                                                    const ProbeF16x8& rhs,
                                                    const ins::F16x8& zero) {
    ProbeF32x4 out{};
#if defined(__gfx946__) || defined(__gfx938__)
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[0], rhs.f16x4[0], zero.f32);
    out.f32 = ins::mmac_f16_lit(lhs.f16x4[1], rhs.f16x4[1], out.f32);
#else
    (void)lhs;
    (void)rhs;
    (void)zero;
#endif
    return out;
}

__device__ __forceinline__ void store_acc(float* out,
                                          int lane,
                                          const ProbeF32x4& acc) {
#pragma unroll
    for (int i = 0; i < kMmacFloatsPerLane; ++i) {
        out[lane * kMmacFloatsPerLane + i] = acc.scalar[i];
    }
}

__global__ void __launch_bounds__(kThreads, 1)
    slotmap_identity_kernel(uint16_t* __restrict__ read_dump) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) uint16_t lds[kLdsWords];
    const int lane = static_cast<int>(threadIdx.x & 63);

    ProbeF16x8 producer{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        producer.u16[word] =
            static_cast<uint16_t>(0x3c00 + lane * kFragWords + word + 1);
        producer.scalar[word] =
            static_cast<_Float16>(lane * kFragWords + word + 1);
    }

    __builtin_hcu_ds_write_matrix_format_f16(
        producer.f16x8, reinterpret_cast<_Float16*>(lds + kDsPageWords), 16,
        2, 1, 0, 0);
    ins::wait_lgkm(0);

    ProbeF16x8 read_frag{};
    read_frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        reinterpret_cast<_Float16*>(lds + kDsPageWords), 16, 2, 1, 0);
    ins::wait_lgkm(0);

#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        read_dump[lane * kFragWords + word] = read_frag.u16[word];
    }
#else
    (void)read_dump;
#endif
}

__global__ void __launch_bounds__(kThreads, 1)
    slotmap_consume_kernel(const uint16_t* __restrict__ ds_src_values,
                           const __half* __restrict__ k_input,
                           uint16_t* __restrict__ read_dump,
                           float* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) uint16_t lds[kLdsWords];
    const int lane = static_cast<int>(threadIdx.x & 63);
    constexpr int kMlsBarrier = 0;

    for (int i = threadIdx.x; i < kLdsWords; i += blockDim.x) {
        lds[i] = 0;
    }
    __syncthreads();

    if (lane == 0) {
        __builtin_hcu_s_abarrier_init(kMlsBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    ins::Vec4U32 k_src = ins::prepare_matrix_src(k_input, kCols);
    ins::abarrier_seq<false>(kMlsBarrier);
    ins::matrix_load_32x16_b16_bps_lds(reinterpret_cast<__half*>(lds), k_src,
                                       kKPageBytes);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(kMlsBarrier, 1);
    int phase = 0;
    ins::abarrier_try_wait<false>(kMlsBarrier, phase);

    ProbeF16x8 producer{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        producer.u16[word] = ds_src_values[lane * kFragWords + word];
    }
    __builtin_hcu_ds_write_matrix_format_f16(
        producer.f16x8, reinterpret_cast<_Float16*>(lds + kDsPageWords), 16,
        2, 1, 0, 0);
    ins::wait_lgkm(0);

    ProbeF16x8 ds_frag{};
    ds_frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        reinterpret_cast<_Float16*>(lds + kDsPageWords), 16, 2, 1, 0);
    ProbeF16x8 k_frag{};
    ins::ds_read_matrix_32x16_normal(reinterpret_cast<__half*>(lds),
                                    kKPageBytes, k_frag.f16x8);
    ins::wait_lgkm(0);

#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        read_dump[lane * kFragWords + word] = ds_frag.u16[word];
    }

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    store_acc(out, lane, mmac_pair_lit(ds_frag, k_frag, zero));

    __syncthreads();
    if (lane == 0) {
        __builtin_hcu_s_abarrier_inv(kMlsBarrier);
    }
#else
    (void)ds_src_values;
    (void)k_input;
    (void)read_dump;
    (void)out;
#endif
}

std::vector<int> build_src_index_for_dst(const std::vector<uint16_t>& dump) {
    std::vector<int> src_for_dst(kFragElems, -1);
    for (int dst = 0; dst < kFragElems; ++dst) {
        const int tag = decode_half_tag(dump[dst]);
        if (tag > 0) {
            src_for_dst[dst] = tag - 1;
        }
    }
    return src_for_dst;
}

std::vector<uint16_t> make_ds_src_for_probe(
    const std::vector<int>& src_for_dst,
    const int slot_k[4][kFragWords]) {
    std::vector<uint16_t> src(kFragElems, half_bits(0.0f));
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int q = lane & 15;
        const int group = lane >> 4;
        for (int word = 0; word < kFragWords; ++word) {
            if (slot_k[group][word] != kProbeK) {
                continue;
            }
            const int dst = lane * kFragWords + word;
            const int src_idx = src_for_dst[dst];
            if (src_idx >= 0) {
                src[src_idx] = half_bits(q_scale_for_decode(q));
            }
        }
    }
    return src;
}

std::vector<uint16_t> make_ds_src_for_single_dst(
    const std::vector<int>& src_for_dst,
    int dst_slot,
    int q) {
    std::vector<uint16_t> src(kFragElems, half_bits(0.0f));
    if (dst_slot < 0 || dst_slot >= kFragElems) {
        return src;
    }
    const int src_idx = src_for_dst[dst_slot];
    if (src_idx >= 0) {
        src[src_idx] = half_bits(q_scale_for_decode(q));
    }
    return src;
}

bool decode_dq_value(float value, int& q, int& krow, int& d) {
    if (!std::isfinite(value) || std::fabs(value) < 0.5f) {
        return false;
    }
    float best_err = 1.0e30f;
    int best_q = -1;
    int best_krow = -1;
    int best_d = -1;
    for (int q_candidate = 0; q_candidate < 16; ++q_candidate) {
        const float q_scale = q_scale_for_decode(q_candidate);
        const float raw_tag =
            value / q_scale - static_cast<float>(kKTagBase);
        const int tag_candidate = static_cast<int>(std::lround(raw_tag));
        if (tag_candidate < 0 || tag_candidate >= 16 * kCols) {
            continue;
        }
        const int krow_candidate = tag_candidate / kCols;
        const int d_candidate = tag_candidate % kCols;
        if (d_candidate >= 32) {
            continue;
        }
        const float expected =
            q_scale * static_cast<float>(kKTagBase + tag_candidate);
        const float err = std::fabs(value - expected);
        if (err < best_err) {
            best_err = err;
            best_q = q_candidate;
            best_krow = krow_candidate;
            best_d = d_candidate;
        }
    }
    if (best_err > 0.75f) {
        return false;
    }
    q = best_q;
    krow = best_krow;
    d = best_d;
    return true;
}

void summarize_mapping(const std::vector<int>& src_for_dst) {
    int mapped = 0;
    for (int src : src_for_dst) {
        mapped += src >= 0 ? 1 : 0;
    }
    std::printf("slotmap_reverse_mapping mapped=%d expected=%d\n", mapped,
                kFragElems);
    for (int group = 0; group < 4; ++group) {
        for (int q = 0; q < 4; ++q) {
            const int lane = group * 16 + q;
            for (int word = 0; word < kFragWords; ++word) {
                const int dst = lane * kFragWords + word;
                const int src = src_for_dst[dst];
                std::printf(
                    "slotmap_reverse_dst group=%d q=%d word=%d dst=%d "
                    "src_lane=%d src_word=%d\n",
                    group, q, word, dst, src >= 0 ? src / kFragWords : -1,
                    src >= 0 ? src % kFragWords : -1);
            }
        }
    }
    std::printf(
        "slotmap_reverse_formula abandoned; using inferred slot_k table "
        "from single-dst MMAC probes; probe_k=%d\n",
        kProbeK);
}

bool summarize_output(const std::vector<float>& out) {
    bool seen_output[kWaveSize * kMmacFloatsPerLane] = {};
    bool seen_q[16] = {};
    bool seen_krow[16] = {};
    bool seen_d[32] = {};
    int raw_nonzero = 0;
    int decoded = 0;
    int nan = 0;
    int printed = 0;
    int anomaly_printed = 0;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
            const float value = out[lane * kMmacFloatsPerLane + vec];
            if (!std::isfinite(value)) {
                ++nan;
                continue;
            }
            if (std::fabs(value) > 0.5f) {
                ++raw_nonzero;
            }
            int q = -1;
            int krow = -1;
            int d = -1;
            if (!decode_dq_value(value, q, krow, d)) {
                continue;
            }
            seen_output[lane * kMmacFloatsPerLane + vec] = true;
            seen_q[q] = true;
            seen_krow[krow] = true;
            seen_d[d] = true;
            ++decoded;
            if (krow != kProbeK && anomaly_printed < 12) {
                std::printf(
                    "slotmap_reverse_anomaly lane=%d vec=%d q=%d "
                    "krow=%d d=%d value=%g\n",
                    lane, vec, q, krow, d, value);
                ++anomaly_printed;
            }
            if (printed < 12) {
                std::printf(
                    "slotmap_reverse_sample lane=%d vec=%d q=%d krow=%d "
                    "d=%d value=%g\n",
                    lane, vec, q, krow, d, value);
                ++printed;
            }
        }
    }

    int unique_output = 0;
    int unique_q = 0;
    int unique_krow = 0;
    int unique_d = 0;
    for (bool v : seen_output) unique_output += v ? 1 : 0;
    for (bool v : seen_q) unique_q += v ? 1 : 0;
    for (bool v : seen_krow) unique_krow += v ? 1 : 0;
    for (bool v : seen_d) unique_d += v ? 1 : 0;
    std::printf("slotmap_reverse_seen_krow");
    for (int i = 0; i < 16; ++i) {
        if (seen_krow[i]) {
            std::printf(" %d", i);
        }
    }
    std::printf("\n");
    std::printf("slotmap_reverse_seen_d");
    for (int i = 0; i < 32; ++i) {
        if (seen_d[i]) {
            std::printf(" %d", i);
        }
    }
    std::printf("\n");
    const bool pass =
        unique_output == 256 && unique_q == 16 && unique_krow == 1 &&
        seen_krow[kProbeK] && unique_d >= 16;
    std::printf(
        "slotmap_reverse_summary raw_nonzero=%d nan=%d decoded=%d "
        "unique_output=%d unique_q=%d unique_krow=%d probe_k_seen=%d "
        "unique_d=%d pass=%d\n",
        raw_nonzero, nan, decoded, unique_output, unique_q, unique_krow,
        seen_krow[kProbeK] ? 1 : 0, unique_d, pass ? 1 : 0);
    return pass;
}

struct SlotInference {
    int raw_nonzero = 0;
    int decoded = 0;
    int unique_q = 0;
    int q = -1;
    int unique_krow = 0;
    int krow = -1;
    int unique_d = 0;
    bool ok = false;
};

SlotInference infer_single_slot(const std::vector<float>& out,
                                int group,
                                int word,
                                int expected_q) {
    bool seen_q[16] = {};
    bool seen_krow[16] = {};
    bool seen_d[32] = {};
    SlotInference summary{};
    for (int lane = 0; lane < kWaveSize; ++lane) {
        for (int vec = 0; vec < kMmacFloatsPerLane; ++vec) {
            const float value = out[lane * kMmacFloatsPerLane + vec];
            if (std::isfinite(value) && std::fabs(value) > 0.5f) {
                ++summary.raw_nonzero;
            }
            int q = -1;
            int krow = -1;
            int d = -1;
            if (!decode_dq_value(value, q, krow, d)) {
                continue;
            }
            seen_q[q] = true;
            seen_krow[krow] = true;
            seen_d[d] = true;
            summary.q = q;
            summary.krow = krow;
            ++summary.decoded;
        }
    }
    for (int i = 0; i < 16; ++i) {
        summary.unique_q += seen_q[i] ? 1 : 0;
        summary.unique_krow += seen_krow[i] ? 1 : 0;
    }
    for (int i = 0; i < 32; ++i) {
        summary.unique_d += seen_d[i] ? 1 : 0;
    }
    summary.ok = summary.decoded > 0 && summary.unique_q == 1 &&
                 summary.q == expected_q && summary.unique_krow == 1 &&
                 summary.unique_d > 0;
    std::printf(
        "slotmap_reverse_infer group=%d word=%d raw_nonzero=%d decoded=%d "
        "unique_q=%d q=%d unique_krow=%d krow=%d unique_d=%d ok=%d\n",
        group, word, summary.raw_nonzero, summary.decoded, summary.unique_q,
        summary.q, summary.unique_krow, summary.krow, summary.unique_d,
        summary.ok ? 1 : 0);
    return summary;
}

void run_consume(uint16_t* ds_src_dev,
                 __half* k_dev,
                 uint16_t* read_dump_dev,
                 float* out_dev,
                 const std::vector<uint16_t>& ds_src,
                 std::vector<uint16_t>& read_dump,
                 std::vector<float>& out) {
    check_hip(hipMemcpy(ds_src_dev, ds_src.data(),
                        ds_src.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy ds_src");
    check_hip(hipMemset(read_dump_dev, 0, kFragElems * sizeof(uint16_t)),
              "hipMemset read_dump");
    check_hip(hipMemset(out_dev, 0,
                        kWaveSize * kMmacFloatsPerLane * sizeof(float)),
              "hipMemset out");
    hipLaunchKernelGGL(slotmap_consume_kernel, dim3(1), dim3(kThreads), 0, 0,
                       ds_src_dev, k_dev, read_dump_dev, out_dev);
    check_hip(hipDeviceSynchronize(), "slotmap_consume_kernel");
    check_hip(hipMemcpy(read_dump.data(), read_dump_dev,
                        read_dump.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy read_dump");
    check_hip(hipMemcpy(out.data(), out_dev, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy out");
}

}  // namespace

int main() {
    uint16_t* identity_dump_dev = nullptr;
    uint16_t* ds_src_dev = nullptr;
    uint16_t* read_dump_dev = nullptr;
    __half* k_dev = nullptr;
    float* out_dev = nullptr;

    check_hip(hipMalloc(reinterpret_cast<void**>(&identity_dump_dev),
                        kFragElems * sizeof(uint16_t)),
              "hipMalloc identity_dump");
    check_hip(hipMalloc(reinterpret_cast<void**>(&ds_src_dev),
                        kFragElems * sizeof(uint16_t)),
              "hipMalloc ds_src");
    check_hip(hipMalloc(reinterpret_cast<void**>(&read_dump_dev),
                        kFragElems * sizeof(uint16_t)),
              "hipMalloc read_dump");
    check_hip(hipMalloc(reinterpret_cast<void**>(&k_dev),
                        kRows * kCols * sizeof(__half)),
              "hipMalloc k");
    check_hip(hipMalloc(reinterpret_cast<void**>(&out_dev),
                        kWaveSize * kMmacFloatsPerLane * sizeof(float)),
              "hipMalloc out");

    std::vector<uint16_t> identity_dump(kFragElems);
    check_hip(hipMemset(identity_dump_dev, 0,
                        kFragElems * sizeof(uint16_t)),
              "hipMemset identity_dump");
    hipLaunchKernelGGL(slotmap_identity_kernel, dim3(1), dim3(kThreads), 0, 0,
                       identity_dump_dev);
    check_hip(hipDeviceSynchronize(), "slotmap_identity_kernel");
    check_hip(hipMemcpy(identity_dump.data(), identity_dump_dev,
                        identity_dump.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy identity_dump");

    const std::vector<int> src_for_dst =
        build_src_index_for_dst(identity_dump);
    summarize_mapping(src_for_dst);

    std::vector<__half> k(kRows * kCols, __float2half(0.0f));
    for (int row = 0; row < 16; ++row) {
        for (int d = 0; d < 32; ++d) {
            k[row * kCols + d] = __float2half(
                static_cast<float>(kKTagBase + row * kCols + d));
        }
    }
    check_hip(hipMemcpy(k_dev, k.data(), k.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");

    std::vector<uint16_t> read_dump(kFragElems);
    std::vector<float> out(kWaveSize * kMmacFloatsPerLane);

    int slot_k[4][kFragWords];
    for (int group = 0; group < 4; ++group) {
        for (int word = 0; word < kFragWords; ++word) {
            slot_k[group][word] = -1;
            const int q = 0;
            const int lane = group * 16 + q;
            const int dst = lane * kFragWords + word;
            const std::vector<uint16_t> single_src =
                make_ds_src_for_single_dst(src_for_dst, dst, q);
            run_consume(ds_src_dev, k_dev, read_dump_dev, out_dev, single_src,
                        read_dump, out);
            const SlotInference inferred =
                infer_single_slot(out, group, word, q);
            if (inferred.ok) {
                slot_k[group][word] = inferred.krow;
            }
        }
        std::printf("slotmap_reverse_k_table group=%d", group);
        for (int word = 0; word < kFragWords; ++word) {
            std::printf(" w%d=%d", word, slot_k[group][word]);
        }
        std::printf("\n");
    }

    const std::vector<uint16_t> ds_src =
        make_ds_src_for_probe(src_for_dst, slot_k);
    run_consume(ds_src_dev, k_dev, read_dump_dev, out_dev, ds_src, read_dump,
                out);

    int nonzero_read_words = 0;
    for (uint16_t bits : read_dump) {
        nonzero_read_words += bits != half_bits(0.0f) ? 1 : 0;
    }
    std::printf("slotmap_reverse_ds_read_nonzero_words=%d\n",
                nonzero_read_words);
    const bool pass = summarize_output(out);
    std::printf("slotmap_reverse_final pass=%d\n", pass ? 1 : 0);

    (void)hipFree(identity_dump_dev);
    (void)hipFree(ds_src_dev);
    (void)hipFree(read_dump_dev);
    (void)hipFree(k_dev);
    (void)hipFree(out_dev);
    return pass ? 0 : 1;
}
