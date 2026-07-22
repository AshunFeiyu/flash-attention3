#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
constexpr int kBaseModes = 16;
constexpr int kPairKinds = 2;
constexpr int kModes = kBaseModes * kPairKinds;
constexpr int kLdsHalfs = kDsBase + kBaseModes * 1024;
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
    Frag8 q_reg_m1[kKBlocks];
    Frag8 k_frag0[kKBlocks];
    Frag8 k_frag1[kKBlocks];
#pragma unroll
    for (int d_block = 0; d_block < kKBlocks; ++d_block) {
        const int lds_offset = d_block * kMatrixBlockBytes;
        if constexpr (kReaderMode == 0 || kReaderMode == 2) {
            ins::ds_read_matrix_32x16_trans(lds + kQBase, lds_offset,
                                            q_reg[d_block].f16x8);
            ins::ds_read_matrix_32x16_trans(lds + kQBase, lds_offset + 1024,
                                            q_reg_m1[d_block].f16x8);
        } else {
            ins::ds_read_matrix_32x16_normal(lds + kQBase, lds_offset,
                                             q_reg[d_block].f16x8);
            ins::ds_read_matrix_32x16_normal(lds + kQBase, lds_offset + 1024,
                                             q_reg_m1[d_block].f16x8);
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
    Acc4 acc_m1{};
    ins::wait_lgkm(0);
    acc0.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[0],
                                     k_frag0[0].f16x4[0], zero.f32);
    acc1.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[0],
                                     k_frag1[0].f16x4[0], zero.f32);
    acc_m1.f32 = mmac_mode<kLit, kLts>(q_reg_m1[0].f16x4[0],
                                       k_frag0[0].f16x4[0], zero.f32);
#pragma unroll
    for (int k_half = 1; k_half < 2; ++k_half) {
        acc0.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[k_half],
                                         k_frag0[0].f16x4[k_half], acc0.f32);
        acc1.f32 = mmac_mode<kLit, kLts>(q_reg[0].f16x4[k_half],
                                         k_frag1[0].f16x4[k_half], acc1.f32);
        acc_m1.f32 = mmac_mode<kLit, kLts>(
            q_reg_m1[0].f16x4[k_half], k_frag0[0].f16x4[k_half],
            acc_m1.f32);
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
            acc_m1.f32 = mmac_mode<kLit, kLts>(
                q_reg_m1[d_block].f16x4[k_half],
                k_frag0[d_block].f16x4[k_half], acc_m1.f32);
        }
    }

    Frag8 natural{};
    const int dump_base = (Mode * kWaveSize + lane) * kWordsPerLane;
    const int mpair_base =
        ((kBaseModes + Mode) * kWaveSize + lane) * kWordsPerLane;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc_dump[dump_base + i] = acc0.f[i];
        acc_dump[dump_base + 4 + i] = acc1.f[i];
        acc_dump[mpair_base + i] = acc0.f[i];
        acc_dump[mpair_base + 4 + i] = acc_m1.f[i];
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

int bit(int value, int index) {
    return (value >> index) & 1;
}

int pack6(int b0, int b1, int b2, int b3, int b4, int b5) {
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) |
           (b5 << 5);
}

int pack3(int b0, int b1, int b2) {
    return b0 | (b1 << 1) | (b2 << 2);
}

// Exact dst(reader) -> src(writer) register-slot permutations measured by
// ds_matrix_reg_roundtrip_probe. Writer alt does not change this register ABI;
// reader alt does. These six formulas reproduce all 6144 measured CSV rows.
void measured_writer_source_slot(int writer,
                                 int reader,
                                 int dst_lane,
                                 int dst_word,
                                 int& src_lane,
                                 int& src_word) {
    const int l0 = bit(dst_lane, 0);
    const int l1 = bit(dst_lane, 1);
    const int l2 = bit(dst_lane, 2);
    const int l3 = bit(dst_lane, 3);
    const int l4 = bit(dst_lane, 4);
    const int l5 = bit(dst_lane, 5);
    const int w0 = bit(dst_word, 0);
    const int w1 = bit(dst_word, 1);
    const int w2 = bit(dst_word, 2);
    const bool writer_trans = writer >= 2;

    if (!writer_trans && reader == 0) {
        src_lane = pack6(l1, l2, l3, w2, w0, w1);
        src_word = pack3(l0, l4, l5);
    } else if (!writer_trans && reader == 1) {
        src_lane = pack6(l0, l1, l2, l3, w0, w1);
        src_word = pack3(w2, l4, l5);
    } else if (!writer_trans) {
        src_lane = pack6(w1, l4, l5, w2, l0, l1);
        src_word = pack3(w0, l2, l3);
    } else if (reader == 0) {
        src_lane = pack6(w0, w1, l4, l5, l1, l2);
        src_word = pack3(l0, l3, w2);
    } else if (reader == 1) {
        src_lane = pack6(w0, w1, l4, l5, l0, l1);
        src_word = pack3(w2, l2, l3);
    } else {
        src_lane = pack6(l0, l1, l2, l3, w1, l4);
        src_word = pack3(w0, l5, w2);
    }
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
    static const char* names[kBaseModes] = {
        "qT_kT_lit0_lts0", "qN_kT_lit0_lts0",
        "qT_kN_lit0_lts0", "qN_kN_lit0_lts0",
        "qT_kT_lit1_lts0", "qN_kT_lit1_lts0",
        "qT_kN_lit1_lts0", "qN_kN_lit1_lts0",
        "qT_kT_lit0_lts1", "qN_kT_lit0_lts1",
        "qT_kN_lit0_lts1", "qN_kN_lit0_lts1",
        "qT_kT_lit1_lts1", "qN_kT_lit1_lts1",
        "qT_kN_lit1_lts1", "qN_kN_lit1_lts1",
    };
    const int base_mode = mode % kBaseModes;
    return mode >= 0 && mode < kModes ? names[base_mode] : "unknown";
}

const char* pair_name(int mode) {
    return mode < kBaseModes ? "N_pair" : "M_pair";
}

bool summarize_acc(const std::vector<float>& acc_dump, int mode) {
    int identity_errors = 0;
    const int mode_base = mode * kWaveSize * kWordsPerLane;
    for (int lane = 0; lane < kWaveSize; ++lane) {
        const int group = lane >> 4;
        for (int word = 0; word < kWordsPerLane; ++word) {
            int q = -1;
            int k = -1;
            const int idx = mode_base + lane * kWordsPerLane + word;
            const bool ok = decode_qk(acc_dump[idx],
                                      q, k);
            const bool m_pair = mode >= kBaseModes;
            const int identity_q =
                (m_pair && word >= 4 ? 16 : 0) + (lane & 15);
            const int identity_k =
                (!m_pair && word >= 4 ? 16 : 0) + group * 4 +
                (word & 3);
            if (!ok || q != identity_q || k != identity_k) {
                ++identity_errors;
            }
        }
    }
    std::printf(
        "source_slot_coordinate_acc_summary mode=%d pair=%s name=%s "
        "identity_errors=%d identity_pass=%d\n",
        mode, pair_name(mode), mode_name(mode), identity_errors,
        identity_errors == 0 ? 1 : 0);
    return identity_errors == 0;
}

bool summarize_exact_writer_abi(const std::vector<float>& acc_dump, int mode) {
    static const char* writer_names[4] = {
        "normal_alt0", "normal_alt1", "trans_alt0", "trans_alt1"};
    static const char* reader_names[3] = {
        "normal_m32_alt0", "normal_m32_alt1", "trans_m32_alt0"};
    const int mode_base = mode * kWaveSize * kWordsPerLane;
    int best_errors = kWaveSize * kWordsPerLane + 1;
    int best_writer = -1;
    int best_reader = -1;
    bool exact = false;

    for (int writer = 0; writer < 4; ++writer) {
        for (int reader = 0; reader < 3; ++reader) {
            int errors = 0;
            for (int dst_lane = 0; dst_lane < kWaveSize; ++dst_lane) {
                const int group = dst_lane >> 4;
                for (int dst_word = 0; dst_word < kWordsPerLane; ++dst_word) {
                    int src_lane = -1;
                    int src_word = -1;
                    measured_writer_source_slot(writer, reader, dst_lane,
                                                dst_word, src_lane, src_word);
                    int q = -1;
                    int k = -1;
                    const int src_index =
                        mode_base + src_lane * kWordsPerLane + src_word;
                    const bool ok = decode_qk(acc_dump[src_index], q, k);
                    const bool m_pair = mode >= kBaseModes;
                    const int expected_q =
                        (m_pair && dst_word >= 4 ? 16 : 0) +
                        (dst_lane & 15);
                    const int expected_k =
                        (!m_pair && dst_word >= 4 ? 16 : 0) + group * 4 +
                        (dst_word & 3);
                    errors += !ok || q != expected_q || k != expected_k;
                }
            }
            if (errors < best_errors) {
                best_errors = errors;
                best_writer = writer;
                best_reader = reader;
            }
            if (errors == 0) {
                exact = true;
                std::printf(
                    "source_slot_exact_match mode=%d pair=%s name=%s writer=%s "
                    "reader=%s source_slot_exact_pass=1\n",
                    mode, pair_name(mode), mode_name(mode), writer_names[writer],
                    reader_names[reader]);
            }
        }
    }
    std::printf(
        "source_slot_exact_summary mode=%d pair=%s name=%s best_errors=%d "
        "best_writer=%s best_reader=%s source_slot_exact_pass=%d\n",
        mode, pair_name(mode), mode_name(mode), best_errors,
        writer_names[best_writer],
        reader_names[best_reader], exact ? 1 : 0);
    return exact;
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

    bool any_identity_pass = false;
    bool any_exact_source_slot_pass = false;
    bool any_direct_read_pass = false;
    for (int mode = 0; mode < kModes; ++mode) {
        any_identity_pass |= summarize_acc(acc_dump, mode);
        any_exact_source_slot_pass |=
            summarize_exact_writer_abi(acc_dump, mode);
        if (mode < kBaseModes) {
            any_direct_read_pass |= summarize_direct_read(read_dump, mode);
        }
    }
    std::printf(
        "source_slot_orientation_final any_identity_pass=%d "
        "any_exact_source_slot_pass=%d any_direct_read_pass=%d\n",
        any_identity_pass ? 1 : 0, any_exact_source_slot_pass ? 1 : 0,
        any_direct_read_pass ? 1 : 0);

    (void)hipFree(bits_dev);
    (void)hipFree(read_dev);
    (void)hipFree(acc_dev);
    (void)hipFree(k_dev);
    (void)hipFree(q_dev);
    return 0;
}
