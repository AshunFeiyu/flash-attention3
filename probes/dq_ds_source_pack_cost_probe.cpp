#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "dq_contract.h"
#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dq = shaobo::fa3::bwd::dq;
namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kThreads = kWaveSize;
constexpr int kFragWords = 8;
constexpr int kLdsHalfs = 4096;
constexpr int kDsPageBytes = 0;
constexpr int kNaturalPageWords = 2048;
constexpr int kDefaultIters = 256;
constexpr int kPathCount = 4;

enum PackPath {
    kNativeSlot = 0,
    kBpermutePack = 1,
    kLdsGatherPack = 2,
    kNaturalWrong = 3,
};

union FragF16x8 {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[kFragWords];
    uint16_t u16[kFragWords];
    uint32_t u32[kFragWords / 2];
};

union AccF32x4 {
    ins::Vec4F32 f32;
    float scalar[4];
};

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(2);
    }
}

__device__ __forceinline__ bool source_slot_to_dst(int src_lane,
                                                   int src_word,
                                                   int& dst_group,
                                                   int& dst_q,
                                                   int& dst_word) {
    const int low = src_word & 1;
#pragma unroll
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
        if (group < 4 && q < 16 && word < dq::NativeDsSlotMap::kWordsPerFrag &&
            dq::NativeDsSlotMap::dst_source_lane(group, q, word) ==
                src_lane &&
            dq::NativeDsSlotMap::dst_source_word(
                q, word,
                dq::NativeDsSlotMap::dst_raw_source_lane(group, q, word)) ==
                src_word) {
            dst_group = group;
            dst_q = q;
            dst_word = word;
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ uint16_t half_bits(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } bits{value};
    return bits.u;
}

__device__ __forceinline__ _Float16 bits_half(uint16_t bits) {
    union {
        uint16_t u;
        _Float16 h;
    } value{bits};
    return value.h;
}

__device__ __forceinline__ uint32_t ds_read_b32_lds(const uint16_t* lds,
                                                    int byte_offset) {
    uint32_t out = 0;
#if defined(__gfx946__) || defined(__gfx92a__)
    const uint32_t addr =
        static_cast<uint32_t>(reinterpret_cast<size_t>(lds)) +
        static_cast<uint32_t>(byte_offset);
    asm volatile("ds_read_b32 %0, %1 offset:0\n"
                 : "=v"(out)
                 : "v"(addr)
                 : "memory");
#else
    (void)lds;
    (void)byte_offset;
#endif
    return out;
}

__device__ __forceinline__ uint32_t bpermute_b32(int src_lane,
                                                 uint32_t src_value) {
#if defined(__gfx946__) || defined(__gfx92a__)
    const int index = src_lane << 2;
    return static_cast<uint32_t>(
        __builtin_amdgcn_ds_bpermute(index, static_cast<int>(src_value)));
#else
    (void)src_lane;
    return src_value;
#endif
}

__device__ __forceinline__ _Float16 natural_value(int lane, int word) {
    const int group = lane >> 4;
    const int q = lane & 15;
    const int krow = dq::NativeDsSlotMap::slot_krow(group, word);
    return static_cast<_Float16>(1.0f + 0.125f * static_cast<float>(q) +
                                 0.015625f * static_cast<float>(krow));
}

__device__ __forceinline__ FragF16x8 make_natural_frag(int lane) {
    FragF16x8 frag{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        frag.scalar[word] = natural_value(lane, word);
    }
    return frag;
}

__device__ __forceinline__ FragF16x8 make_direct_source_frag(int lane) {
    FragF16x8 frag{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        int group = -1;
        int q = -1;
        int dst_word = -1;
        if (source_slot_to_dst(lane, word, group, q, dst_word)) {
            frag.scalar[word] =
                natural_value(group * 16 + q, dst_word);
        } else {
            frag.scalar[word] = static_cast<_Float16>(0.0f);
        }
    }
    return frag;
}

template <int Word>
__device__ __forceinline__ uint32_t fetch_bpermute_pair(
    int lane,
    const FragF16x8& natural) {
    int group = -1;
    int q = -1;
    int dst_word = -1;
    if (!source_slot_to_dst(lane, Word, group, q, dst_word)) {
        return static_cast<uint32_t>(0);
    }
    const int dst_lane = group * 16 + q;
    return bpermute_b32(dst_lane, natural.u32[dst_word >> 1]);
}

template <int Word>
__device__ __forceinline__ uint32_t fetch_lds_pair(const uint16_t* lds) {
    const int lane = static_cast<int>(threadIdx.x & 63);
    int group = -1;
    int q = -1;
    int dst_word = -1;
    if (!source_slot_to_dst(lane, Word, group, q, dst_word)) {
        return static_cast<uint32_t>(0);
    }
    const int dst_lane = group * 16 + q;
    const int pair_word = dst_word & ~1;
    const int byte_offset =
        (kNaturalPageWords + dst_lane * kFragWords + pair_word) *
        static_cast<int>(sizeof(uint16_t));
    return ds_read_b32_lds(lds, byte_offset);
}

__device__ __forceinline__ FragF16x8 make_bpermute_source_frag_noarray(
    int lane,
    const FragF16x8& natural) {
    const uint32_t p0 = fetch_bpermute_pair<0>(lane, natural);
    const uint32_t p1 = fetch_bpermute_pair<2>(lane, natural);
    const uint32_t p2 = fetch_bpermute_pair<4>(lane, natural);
    const uint32_t p3 = fetch_bpermute_pair<6>(lane, natural);
    ins::wait_lgkm(0);
    FragF16x8 frag{};
    frag.u32[0] = p0;
    frag.u32[1] = p1;
    frag.u32[2] = p2;
    frag.u32[3] = p3;
    return frag;
}

__device__ __forceinline__ FragF16x8 make_lds_gather_source_frag_noarray(
    const uint16_t* lds) {
    const uint32_t p0 = fetch_lds_pair<0>(lds);
    const uint32_t p1 = fetch_lds_pair<2>(lds);
    const uint32_t p2 = fetch_lds_pair<4>(lds);
    const uint32_t p3 = fetch_lds_pair<6>(lds);
    ins::wait_lgkm(0);
    FragF16x8 frag{};
    frag.u32[0] = p0;
    frag.u32[1] = p1;
    frag.u32[2] = p2;
    frag.u32[3] = p3;
    return frag;
}

template <int Path>
__device__ __forceinline__ void run_pack_path(float* __restrict__ sums,
                                              int* __restrict__ errors,
                                              int iters) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) uint16_t lds[kLdsHalfs];
    const int lane = static_cast<int>(threadIdx.x & 63);

    for (int i = threadIdx.x; i < kLdsHalfs; i += blockDim.x) {
        lds[i] = 0;
    }
    __syncthreads();

    FragF16x8 natural = make_natural_frag(lane);
    if constexpr (Path == kLdsGatherPack) {
#pragma unroll
        for (int word = 0; word < kFragWords; ++word) {
            lds[kNaturalPageWords + lane * kFragWords + word] =
                natural.u16[word];
        }
        __syncthreads();
    }

    FragF16x8 rhs{};
#pragma unroll
    for (int word = 0; word < kFragWords; ++word) {
        rhs.scalar[word] =
            static_cast<_Float16>(0.5f + 0.03125f * static_cast<float>(word));
    }

    float checksum = 0.0f;
    int local_errors = 0;
    for (int iter = 0; iter < iters; ++iter) {
        const FragF16x8 expected = make_direct_source_frag(lane);
        FragF16x8 producer{};
        if constexpr (Path == kNativeSlot) {
            producer = expected;
        } else if constexpr (Path == kBpermutePack) {
            producer = make_bpermute_source_frag_noarray(lane, natural);
        } else if constexpr (Path == kLdsGatherPack) {
            producer = make_lds_gather_source_frag_noarray(lds);
        } else {
            producer = natural;
        }

        if constexpr (Path != kNaturalWrong) {
            if (iter == 0) {
#pragma unroll
                for (int word = 0; word < kFragWords; ++word) {
                    local_errors +=
                        producer.u16[word] == expected.u16[word] ? 0 : 1;
                }
            }
        }

        ins::ds_write_matrix_32x16_f16(
            producer.f16x8, reinterpret_cast<__half*>(lds), kDsPageBytes);
        ins::wait_lgkm(0);

        FragF16x8 ds_frag{};
        ds_frag.f16x8 = __builtin_hcu_ds_read_matrix_trans_format_f16(
            reinterpret_cast<_Float16*>(lds), 16, 2, 1, 0);
        ins::wait_lgkm(0);

        ins::F32x4 acc{};
        acc.f32 = ins::mmac_f16_lit(ds_frag.f16x4[0], rhs.f16x4[0], acc.f32);
        acc.f32 = ins::mmac_f16_lit(ds_frag.f16x4[1], rhs.f16x4[1], acc.f32);
        ins::keep_accumulator_live(acc);
        checksum += acc.scalar[0] + acc.scalar[1] + acc.scalar[2] +
                    acc.scalar[3];
    }

    sums[Path * kWaveSize + lane] = checksum;
    if (local_errors != 0) {
        atomicAdd(errors + Path, local_errors);
    }
#else
    (void)sums;
    (void)errors;
    (void)iters;
#endif
}

__global__ void __launch_bounds__(kThreads, 1)
    native_slot_pack_cost_kernel(float* __restrict__ sums,
                                 int* __restrict__ errors,
                                 int iters) {
    run_pack_path<kNativeSlot>(sums, errors, iters);
}

__global__ void __launch_bounds__(kThreads, 1)
    bpermute_pack_cost_kernel(float* __restrict__ sums,
                              int* __restrict__ errors,
                              int iters) {
    run_pack_path<kBpermutePack>(sums, errors, iters);
}

__global__ void __launch_bounds__(kThreads, 1)
    lds_gather_pack_cost_kernel(float* __restrict__ sums,
                                int* __restrict__ errors,
                                int iters) {
    run_pack_path<kLdsGatherPack>(sums, errors, iters);
}

__global__ void __launch_bounds__(kThreads, 1)
    natural_wrong_pack_cost_kernel(float* __restrict__ sums,
                                   int* __restrict__ errors,
                                   int iters) {
    run_pack_path<kNaturalWrong>(sums, errors, iters);
}

int parse_iters(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0) {
            return std::max(1, std::atoi(argv[i + 1]));
        }
    }
    return kDefaultIters;
}

int parse_path(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--path") != 0) {
            continue;
        }
        if (std::strcmp(argv[i + 1], "native_slot") == 0) {
            return kNativeSlot;
        }
        if (std::strcmp(argv[i + 1], "bpermute_pack") == 0) {
            return kBpermutePack;
        }
        if (std::strcmp(argv[i + 1], "lds_gather_pack") == 0) {
            return kLdsGatherPack;
        }
        if (std::strcmp(argv[i + 1], "natural_wrong") == 0) {
            return kNaturalWrong;
        }
        if (std::strcmp(argv[i + 1], "all") == 0) {
            return -1;
        }
        std::fprintf(stderr, "unknown --path %s\n", argv[i + 1]);
        std::exit(2);
    }
    return kNativeSlot;
}

const char* path_name(int path) {
    switch (path) {
        case kNativeSlot:
            return "native_slot";
        case kBpermutePack:
            return "bpermute_pack";
        case kLdsGatherPack:
            return "lds_gather_pack";
        case kNaturalWrong:
            return "natural_wrong";
        default:
            return "unknown";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = parse_iters(argc, argv);
    const int selected_path = parse_path(argc, argv);
    float* d_sums = nullptr;
    int* d_errors = nullptr;
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_sums),
                        kPathCount * kWaveSize * sizeof(float)),
              "hipMalloc sums");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_errors),
                        kPathCount * sizeof(int)),
              "hipMalloc errors");
    check_hip(hipMemset(d_sums, 0, kPathCount * kWaveSize * sizeof(float)),
              "hipMemset sums");
    check_hip(hipMemset(d_errors, 0, kPathCount * sizeof(int)),
              "hipMemset errors");

    if (selected_path < 0 || selected_path == kNativeSlot) {
        hipLaunchKernelGGL(native_slot_pack_cost_kernel, dim3(1),
                           dim3(kThreads), 0, 0, d_sums, d_errors, iters);
        check_hip(hipGetLastError(), "launch native_slot_pack_cost_kernel");
    }
    if (selected_path < 0 || selected_path == kBpermutePack) {
        hipLaunchKernelGGL(bpermute_pack_cost_kernel, dim3(1), dim3(kThreads),
                           0, 0, d_sums, d_errors, iters);
        check_hip(hipGetLastError(), "launch bpermute_pack_cost_kernel");
    }
    if (selected_path < 0 || selected_path == kLdsGatherPack) {
        hipLaunchKernelGGL(lds_gather_pack_cost_kernel, dim3(1),
                           dim3(kThreads), 0, 0, d_sums, d_errors, iters);
        check_hip(hipGetLastError(), "launch lds_gather_pack_cost_kernel");
    }
    if (selected_path < 0 || selected_path == kNaturalWrong) {
        hipLaunchKernelGGL(natural_wrong_pack_cost_kernel, dim3(1),
                           dim3(kThreads), 0, 0, d_sums, d_errors, iters);
        check_hip(hipGetLastError(), "launch natural_wrong_pack_cost_kernel");
    }
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::vector<float> sums(kPathCount * kWaveSize);
    int errors[kPathCount] = {};
    check_hip(hipMemcpy(sums.data(), d_sums,
                        sums.size() * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy sums");
    check_hip(hipMemcpy(errors, d_errors, sizeof(errors),
                        hipMemcpyDeviceToHost),
              "hipMemcpy errors");
    check_hip(hipFree(d_sums), "hipFree sums");
    check_hip(hipFree(d_errors), "hipFree errors");

    bool pass = true;
    for (int path = 0; path < kPathCount; ++path) {
        if (selected_path >= 0 && selected_path != path) {
            continue;
        }
        double checksum = 0.0;
        for (int lane = 0; lane < kWaveSize; ++lane) {
            checksum += static_cast<double>(sums[path * kWaveSize + lane]);
        }
        std::printf(
            "ds_source_pack_cost path=%s dispatch=%d iters=%d errors=%d "
            "checksum=%.9g\n",
            path_name(path), path, iters, errors[path], checksum);
        const bool ignores_correctness = path == kNaturalWrong;
        pass = pass &&
               (ignores_correctness || errors[path] == 0) &&
               std::isfinite(checksum);
    }
    std::printf("ds_source_pack_cost_pass=%d\n", pass ? 1 : 0);
    return pass ? 0 : 1;
}
