#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_probe_contract.h"
#include "shaobo_instr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dq = shaobo::fa3::bwd::dq;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kThreads = 64;
constexpr int kWaveSize = 64;
constexpr int kRows = 32;
constexpr int kDim = 128;
constexpr int kKBlocks = kDim / 32;
constexpr int kMatrixBlockBytes = 32 * 32 * 2;
constexpr int kMatrixBlockHalfs = kMatrixBlockBytes / sizeof(__half);
constexpr int kQBase = 0;
constexpr int kKBase = kQBase + kKBlocks * kMatrixBlockHalfs;
constexpr int kDsBase = kKBase + kKBlocks * kMatrixBlockHalfs;
constexpr int kModes = 16;
constexpr int kLdsHalfs = kDsBase + kModes * 1024;
constexpr int kWordsPerLane = 8;

union Frag8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 h[kWordsPerLane];
    uint16_t u16[kWordsPerLane];
};

union Acc4 {
    ins::Vec4F32 f32;
    float f[4];
};

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

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

__device__ __forceinline__ uint16_t half_bits(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } v{value};
    return v.u;
}

template <int Mode>
__device__ __forceinline__ void compute_mode(__half* __restrict__ lds,
                                             float* __restrict__ acc_dump,
                                             float* __restrict__ read_dump,
                                             uint16_t* __restrict__ read_bits,
                                             int lane) {
    constexpr int kReaderMode = Mode & 3;
    constexpr int kLit = (Mode >> 2) & 1;
    constexpr int kLts = (Mode >> 3) & 1;
    Frag8 q_reg[kKBlocks];
    Frag8 k_frag0[kKBlocks];
    Frag8 k_frag1[kKBlocks];
#pragma unroll
    for (int d_block = 0; d_block < kKBlocks; ++d_block) {
        const int lds_offset = d_block * kMatrixBlockBytes;
        if constexpr (kReaderMode == 0 || kReaderMode == 2) {
            ins::ds_read_matrix_32x16_trans(lds + kQBase, lds_offset,
                                            q_reg[d_block].f16x8);
        } else {
            ins::ds_read_matrix_32x16_normal(lds + kQBase, lds_offset,
                                             q_reg[d_block].f16x8);
        }

        if constexpr (kReaderMode == 0 || kReaderMode == 1) {
            ins::ds_read_matrix_trans_pair(lds + kKBase, lds_offset,
                                           k_frag0[d_block].f16x8,
                                           k_frag1[d_block].f16x8);
        } else {
            ins::ds_read_matrix_normal_pair(lds + kKBase, lds_offset,
                                            k_frag0[d_block].f16x8,
                                            k_frag1[d_block].f16x8);
        }
    }

    ins::F32x4 zero;
    ins::zero_vgpr2(zero.u64[0]);
    ins::zero_vgpr2(zero.u64[1]);

    Acc4 acc0{};
    Acc4 acc1{};
    ins::wait_lgkm(0);
    acc0.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[0],
                                     k_frag0[0].f16x4[0], zero.f32);
    acc1.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[0],
                                     k_frag1[0].f16x4[0], zero.f32);
#pragma unroll
    for (int k_half = 1; k_half < 2; ++k_half) {
        acc0.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[k_half],
                                         k_frag0[0].f16x4[k_half], acc0.f32);
        acc1.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[k_half],
                                         k_frag1[0].f16x4[k_half], acc1.f32);
    }
#pragma unroll
    for (int d_block = 1; d_block < kKBlocks; ++d_block) {
#pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            acc0.f32 = mmac_mode<kLit, kLts>(
                q_reg[d_block].f16x4[k_half],
                k_frag0[d_block].f16x4[k_half], acc0.f32);
            acc1.f32 = mmac_mode<kLit, kLts>(
                q_reg[d_block].f16x4[k_half],
                k_frag1[d_block].f16x4[k_half], acc1.f32);
        }
    }

    Frag8 natural{};
    const int dump_base = (Mode * kWaveSize + lane) * kWordsPerLane;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc_dump[dump_base + i] = acc0.f[i];
        acc_dump[dump_base + 4 + i] = acc1.f[i];
        natural.h[i] = static_cast<_Float16>(acc0.f[i]);
        natural.h[4 + i] = static_cast<_Float16>(acc1.f[i]);
    }

    ins::ds_write_matrix_32x16_f16(natural.f16x8,
                                   lds + kDsBase + Mode * 1024, 0);
    ins::wait_lgkm(0);
    Frag8 readback{};
    readback.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
        reinterpret_cast<_Float16*>(lds + kDsBase + Mode * 1024), 16, 2, 1,
        0);
    ins::wait_lgkm(0);
#pragma unroll
    for (int i = 0; i < kWordsPerLane; ++i) {
        read_dump[dump_base + i] = static_cast<float>(readback.h[i]);
        read_bits[dump_base + i] = half_bits(readback.h[i]);
    }
}

__global__ void __launch_bounds__(kThreads, 1)
dq_source_slot_coordinate_probe_kernel(const __half* __restrict__ q,
                                       const __half* __restrict__ k,
                                       float* __restrict__ acc_dump,
                                       float* __restrict__ read_dump,
                                       uint16_t* __restrict__ read_bits) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) __half lds[kLdsHalfs];
    const int lane = static_cast<int>(threadIdx.x & 63);
    constexpr int kMlsBarrier = 0;

    for (int i = threadIdx.x; i < kLdsHalfs; i += blockDim.x) {
        lds[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (lane == 0) {
        __builtin_hcu_s_abarrier_init(kMlsBarrier, 1);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    ins::abarrier_seq<false>(kMlsBarrier);
#pragma unroll
    for (int d_block = 0; d_block < kKBlocks; ++d_block) {
        const int d_base = d_block * 32;
        ins::Vec4U32 q_src =
            ins::prepare_matrix_src(q + d_base, kDim);
        ins::matrix_load_32x32_b16_bps_lds(
            lds + kQBase, q_src, d_block * kMatrixBlockBytes, true);

        ins::Vec4U32 k_src =
            ins::prepare_matrix_src(k + d_base, kDim);
        ins::matrix_load_32x32_b16_bps_lds(
            lds + kKBase, k_src, d_block * kMatrixBlockBytes, true);
    }
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(kMlsBarrier, 1);
    int phase = 0;
    ins::abarrier_try_wait<false>(kMlsBarrier, phase);

    compute_mode<0>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<1>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<2>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<3>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<4>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<5>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<6>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<7>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<8>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<9>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<10>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<11>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<12>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<13>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<14>(lds, acc_dump, read_dump, read_bits, lane);
    compute_mode<15>(lds, acc_dump, read_dump, read_bits, lane);

    __syncthreads();
    if (lane == 0) {
        __builtin_hcu_s_abarrier_inv(kMlsBarrier);
    }
#else
    (void)q;
    (void)k;
    (void)acc_dump;
    (void)read_dump;
    (void)read_bits;
#endif
}

bool source_slot_to_dst(int src_lane,
                        int src_word,
                        int& dst_group,
                        int& dst_q,
                        int& dst_word) {
    const int low = src_word & 1;
    for (int carry = 0; carry < 2; ++carry) {
        const int q_hi_word = src_word - 2 * carry;
        if (q_hi_word < 0) {
            continue;
        }
        const int q_hi = q_hi_word >> 1;
        if (q_hi > 3) {
            continue;
        }
        const int raw_lane = src_lane + carry * dq::NativeDsSlotMap::kWaveSize;
        const int base = raw_lane - 4;
        if (base < 0 || base >= dq::NativeDsSlotMap::kWaveSize) {
            continue;
        }
        const int q_lo = base >> 4;
        const int rem = base & 15;
        const int word_hi = rem >> 3;
        const int rem2 = rem & 7;
        dst_group = rem2 >> 1;
        dst_q = 4 * q_hi + q_lo;
        dst_word = 4 * word_hi + 2 * (rem2 & 1) + low;
        if (dst_group < 4 && dst_q < 16 &&
            dst_word < dq::NativeDsSlotMap::kWordsPerFrag) {
            return true;
        }
    }
    return false;
}

int decode_code(float value) {
    const int code = static_cast<int>(std::lround(value));
    return std::fabs(value - static_cast<float>(code)) < 0.25f ? code : -1;
}

bool decode_qk(float value, int& q, int& k) {
    const int code = decode_code(value);
    if (code < 0) {
        q = -1;
        k = -1;
        return false;
    }
    q = code / 64;
    k = code - q * 64;
    return q >= 0 && q < 32 && k >= 0 && k < 64;
}

const char* mode_name(int mode) {
    static const char* names[kModes] = {
        "qT_kT_lit0_lts0", "qN_kT_lit0_lts0",
        "qT_kN_lit0_lts0", "qN_kN_lit0_lts0",
        "qT_kT_lit1_lts0", "qN_kT_lit1_lts0",
        "qT_kN_lit1_lts0", "qN_kN_lit1_lts0",
        "qT_kT_lit0_lts1", "qN_kT_lit0_lts1",
        "qT_kN_lit0_lts1", "qN_kN_lit0_lts1",
        "qT_kT_lit1_lts1", "qN_kT_lit1_lts1",
        "qT_kN_lit1_lts1", "qN_kN_lit1_lts1",
    };
    return mode >= 0 && mode < kModes ? names[mode] : "unknown";
}

bool summarize_acc(const std::vector<float>& acc_dump, int mode) {
    int identity_errors = 0;
    int source_slot_errors = 0;
    int source_slots = 0;
    int printed = 0;
    const int mode_base = mode * kWaveSize * kWordsPerLane;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int identity_q = lane & 15;
        const int group = lane >> 4;
        for (int word = 0; word < kWordsPerLane; ++word) {
            int q = -1;
            int k = -1;
            const int idx = mode_base + lane * kWordsPerLane + word;
            const bool ok = decode_qk(acc_dump[idx],
                                      q, k);
            const int identity_k = (word >= 4 ? 16 : 0) + group * 4 +
                                   (word & 3);
            if (!ok || q != identity_q || k != identity_k) {
                ++identity_errors;
            }

            int dst_group = -1;
            int dst_q = -1;
            int dst_word = -1;
            if (source_slot_to_dst(lane, word, dst_group, dst_q, dst_word)) {
                ++source_slots;
                const int expected_k =
                    (word >= 4 ? 16 : 0) +
                    dq::NativeDsSlotMap::slot_krow(dst_group, dst_word);
                if (!ok || q != dst_q || k != expected_k) {
                    ++source_slot_errors;
                    if (printed < 10) {
                        std::printf(
                            "source_slot_mismatch lane=%d word=%d got_q=%d "
                            "got_k=%d expected_q=%d expected_k=%d "
                            "dst_group=%d dst_word=%d value=%g\n",
                            lane, word, q, k, dst_q, expected_k, dst_group,
                            dst_word,
                            acc_dump[idx]);
                        ++printed;
                    }
                }
            }
        }
    }
    std::printf(
        "source_slot_coordinate_acc_summary mode=%d name=%s identity_errors=%d "
        "source_slot_errors=%d source_slots=%d identity_pass=%d "
        "source_slot_direct_pass=%d\n",
        mode, mode_name(mode), identity_errors, source_slot_errors, source_slots,
        identity_errors == 0 ? 1 : 0,
        source_slot_errors == 0 && source_slots != 0 ? 1 : 0);
    return source_slot_errors == 0 && source_slots != 0;
}

bool summarize_direct_read(const std::vector<float>& read_dump, int mode) {
    int read_identity_errors = 0;
    int printed = 0;
    const int mode_base = mode * kWaveSize * kWordsPerLane;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int q_expected = lane & 15;
        const int group = lane >> 4;
        for (int word = 0; word < kWordsPerLane; ++word) {
            int q = -1;
            int k = -1;
            const int idx = mode_base + lane * kWordsPerLane + word;
            const bool ok = decode_qk(read_dump[idx],
                                      q, k);
            const int k_expected = group * 4 + (word & 3);
            if (!ok || q != q_expected || k != k_expected) {
                ++read_identity_errors;
                if (printed < 10) {
                    std::printf(
                        "direct_read_mismatch lane=%d word=%d got_q=%d "
                        "got_k=%d expected_q=%d expected_k=%d value=%g\n",
                        lane, word, q, k, q_expected, k_expected,
                        read_dump[idx]);
                    ++printed;
                }
            }
        }
    }
    std::printf(
        "source_slot_coordinate_direct_read_summary mode=%d name=%s "
        "read_identity_errors=%d "
        "read_identity_pass=%d\n",
        mode, mode_name(mode), read_identity_errors,
        read_identity_errors == 0 ? 1 : 0);
    return read_identity_errors == 0;
}

}  // namespace

int main() {
    __half* q_dev = nullptr;
    __half* k_dev = nullptr;
    float* acc_dev = nullptr;
    float* read_dev = nullptr;
    uint16_t* bits_dev = nullptr;

    const size_t matrix_bytes = kRows * kDim * sizeof(__half);
    const size_t dump_elems = kModes * kWaveSize * kWordsPerLane;
    check_hip(hipMalloc(reinterpret_cast<void**>(&q_dev), matrix_bytes),
              "hipMalloc q");
    check_hip(hipMalloc(reinterpret_cast<void**>(&k_dev), matrix_bytes),
              "hipMalloc k");
    check_hip(hipMalloc(reinterpret_cast<void**>(&acc_dev),
                        dump_elems * sizeof(float)),
              "hipMalloc acc");
    check_hip(hipMalloc(reinterpret_cast<void**>(&read_dev),
                        dump_elems * sizeof(float)),
              "hipMalloc read");
    check_hip(hipMalloc(reinterpret_cast<void**>(&bits_dev),
                        dump_elems * sizeof(uint16_t)),
              "hipMalloc bits");

    std::vector<__half> q(kRows * kDim, __float2half(0.0f));
    std::vector<__half> k(kRows * kDim, __float2half(0.0f));
    for (int row = 0; row < kRows; ++row) {
        q[row * kDim + 0] = __float2half(static_cast<float>(row));
        q[row * kDim + 1] = __float2half(1.0f);
        k[row * kDim + 0] = __float2half(64.0f);
        k[row * kDim + 1] = __float2half(static_cast<float>(row));
    }

    check_hip(hipMemcpy(q_dev, q.data(), matrix_bytes, hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemcpy(k_dev, k.data(), matrix_bytes, hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemset(acc_dev, 0, dump_elems * sizeof(float)),
              "hipMemset acc");
    check_hip(hipMemset(read_dev, 0, dump_elems * sizeof(float)),
              "hipMemset read");
    check_hip(hipMemset(bits_dev, 0, dump_elems * sizeof(uint16_t)),
              "hipMemset bits");

    hipLaunchKernelGGL(dq_source_slot_coordinate_probe_kernel, dim3(1),
                       dim3(kThreads), 0, 0, q_dev, k_dev, acc_dev,
                       read_dev, bits_dev);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    std::vector<float> acc_dump(dump_elems);
    std::vector<float> read_dump(dump_elems);
    std::vector<uint16_t> read_bits(dump_elems);
    check_hip(hipMemcpy(acc_dump.data(), acc_dev, dump_elems * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy acc");
    check_hip(hipMemcpy(read_dump.data(), read_dev, dump_elems * sizeof(float),
                        hipMemcpyDeviceToHost),
              "hipMemcpy read");
    check_hip(hipMemcpy(read_bits.data(), bits_dev,
                        dump_elems * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy bits");

    bool any_source_slot_pass = false;
    bool any_direct_read_pass = false;
    for (int mode = 0; mode < kModes; ++mode) {
        any_source_slot_pass |= summarize_acc(acc_dump, mode);
        any_direct_read_pass |= summarize_direct_read(read_dump, mode);
    }
    std::printf(
        "source_slot_orientation_final any_source_slot_pass=%d "
        "any_direct_read_pass=%d\n",
        any_source_slot_pass ? 1 : 0, any_direct_read_pass ? 1 : 0);

    (void)hipFree(bits_dev);
    (void)hipFree(read_dev);
    (void)hipFree(acc_dev);
    (void)hipFree(k_dev);
    (void)hipFree(q_dev);
    return 0;
}
