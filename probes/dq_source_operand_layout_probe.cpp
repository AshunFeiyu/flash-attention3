#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_contract.h"
#include "shaobo_instr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dq = shaobo::fa3::bwd::dq;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kThreads = 64;
constexpr int kWaveSize = 64;
constexpr int kWords = 8;
constexpr int kRows = 32;
constexpr int kCols = 64;
constexpr int kReadModes = 5;
constexpr int kLoadModes = 2;
constexpr int kModes = kReadModes * kLoadModes;
constexpr int kMatrix32Bytes = 32 * 32 * 2;
constexpr int kLdsHalfs = 4096;

union Frag {
    ins::Vec8F16 f16x8;
    _Float16 scalar[kWords];
    uint16_t u16[kWords];
};

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(2);
    }
}

__device__ __forceinline__ Frag read_mode(const __half* lds, int mode) {
    Frag frag{};
#if defined(__gfx946__) || defined(__gfx92a__)
    auto* ptr = reinterpret_cast<_Float16*>(const_cast<__half*>(lds));
    const int read_mode = mode % kReadModes;
    if (read_mode == 0) {
        frag.f16x8 =
            __builtin_hcu_ds_read_matrix_trans_format_f16(ptr, 16, 2, 1, 0);
    } else if (read_mode == 1) {
        frag.f16x8 =
            __builtin_hcu_ds_read_matrix_format_f16(ptr, 16, 2, 1, 0);
    } else if (read_mode == 2) {
        frag.f16x8 =
            __builtin_hcu_ds_read_matrix_trans_format_f16(ptr, 16, 1, 2, 0);
    } else if (read_mode == 3) {
        frag.f16x8 =
            __builtin_hcu_ds_read_matrix_trans_format_f16(ptr, 16, 1, 2, 1);
    } else {
        frag.f16x8 =
            __builtin_hcu_ds_read_matrix_format_f16(ptr, 16, 2, 1, 1);
    }
#else
    (void)lds;
    (void)mode;
#endif
    return frag;
}

const char* read_mode_name(int read_mode) {
    switch (read_mode) {
        case 0:
            return "trans_32x16_alt0";
        case 1:
            return "normal_32x16_alt0";
        case 2:
            return "trans_16x32_alt0";
        case 3:
            return "trans_16x32_alt1";
        case 4:
            return "normal_32x16_alt1";
        default:
            return "unknown";
    }
}

__global__ void __launch_bounds__(kThreads, 1)
    operand_layout_kernel(const __half* __restrict__ q_input,
                          uint16_t* __restrict__ dump) {
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

    ins::Vec4U32 q_src = ins::prepare_matrix_src(q_input, kCols);
    ins::abarrier_seq<false>(kMlsBarrier);
    ins::matrix_load_32x32_b16_bps_lds(lds, q_src, 0, false);
    ins::matrix_load_32x32_b16_bps_lds(
        lds + kMatrix32Bytes / sizeof(__half), q_src, 0, true);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(kMlsBarrier, 1);
    int phase = 0;
    ins::abarrier_try_wait<false>(kMlsBarrier, phase);

#pragma unroll
    for (int mode = 0; mode < kModes; ++mode) {
        const int load_mode = mode / kReadModes;
        const __half* page =
            lds + load_mode * (kMatrix32Bytes / sizeof(__half));
        Frag frag = read_mode(page, mode);
        ins::wait_lgkm(0);
#pragma unroll
        for (int word = 0; word < kWords; ++word) {
            dump[(mode * kWaveSize + lane) * kWords + word] = frag.u16[word];
        }
    }

    __syncthreads();
    if (lane == 0) {
        __builtin_hcu_s_abarrier_inv(kMlsBarrier);
    }
#else
    (void)q_input;
    (void)dump;
#endif
}

int decode_row(uint16_t bits) {
    union {
        uint16_t u;
        _Float16 h;
    } value{bits};
    const float f = static_cast<float>(value.h);
    const int row = static_cast<int>(std::lround(f)) - 1;
    if (row < 0 || row >= kRows ||
        std::fabs(f - static_cast<float>(row + 1)) > 0.125f) {
        return -1;
    }
    return row;
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
        const int group = rem2 >> 1;
        const int word_mid = rem2 & 1;
        const int word = 4 * word_hi + 2 * word_mid + low;
        const int q = 4 * q_hi + q_lo;
        if (group < 4 && q < 16 && word < dq::NativeDsSlotMap::kWordsPerFrag) {
            dst_group = group;
            dst_q = q;
            dst_word = word;
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    __half* q_dev = nullptr;
    uint16_t* dump_dev = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&q_dev),
                        kRows * kCols * sizeof(__half)),
              "hipMalloc q");
    check_hip(hipMalloc(reinterpret_cast<void**>(&dump_dev),
                        kModes * kWaveSize * kWords * sizeof(uint16_t)),
              "hipMalloc dump");

    std::vector<__half> q(kRows * kCols, __float2half(0.0f));
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            q[row * kCols + col] = __float2half(static_cast<float>(row + 1));
        }
    }
    std::vector<uint16_t> dump(kModes * kWaveSize * kWords);
    check_hip(hipMemcpy(q_dev, q.data(), q.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "hipMemcpy q");
    check_hip(hipMemset(dump_dev, 0, dump.size() * sizeof(uint16_t)),
              "hipMemset dump");

    hipLaunchKernelGGL(operand_layout_kernel, dim3(1), dim3(kThreads), 0, 0,
                       q_dev, dump_dev);
    check_hip(hipDeviceSynchronize(), "operand_layout_kernel");
    check_hip(hipMemcpy(dump.data(), dump_dev,
                        dump.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy dump");

    bool any_full = false;
    for (int mode = 0; mode < kModes; ++mode) {
        int mapped = 0;
        int q_match = 0;
        int decoded = 0;
        int printed = 0;
        for (int lane = 0; lane < kWaveSize; ++lane) {
            for (int word = 0; word < kWords; ++word) {
                int group = -1;
                int q_expected = -1;
                int dst_word = -1;
                if (!source_slot_to_dst(lane, word, group, q_expected,
                                        dst_word)) {
                    continue;
                }
                ++mapped;
                const int row =
                    decode_row(dump[(mode * kWaveSize + lane) * kWords + word]);
                decoded += row >= 0 ? 1 : 0;
                if (row == q_expected) {
                    ++q_match;
                } else if (printed < 8) {
                    std::printf(
                        "operand_layout_mismatch mode=%d lane=%d word=%d "
                        "expected_q=%d got_row=%d group=%d dst_word=%d\n",
                        mode, lane, word, q_expected, row, group, dst_word);
                    ++printed;
                }
            }
        }
        const bool full = mapped == 504 && q_match == mapped;
        any_full = any_full || full;
        std::printf(
            "operand_layout_summary mode=%d load=%d read=%d mapped=%d "
            "read_name=%s decoded=%d q_match=%d full_match=%d\n",
            mode, mode / kReadModes, mode % kReadModes,
            mapped, read_mode_name(mode % kReadModes), decoded, q_match,
            full ? 1 : 0);
    }
    std::printf("operand_layout_final any_full_match=%d\n",
                any_full ? 1 : 0);

    (void)hipFree(dump_dev);
    (void)hipFree(q_dev);
    return any_full ? 0 : 1;
}
