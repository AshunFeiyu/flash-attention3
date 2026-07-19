#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kGroups = 4;
constexpr int kWavesPerGroup = 4;
constexpr int kWaves = kGroups * kWavesPerGroup;
constexpr int kThreads = kWaves * kWaveSize;
constexpr int kMq = 64;
constexpr int kNk = 256;
constexpr int kHeadDim = 128;
constexpr int kMRowsPerGroup = 16;
constexpr int kNRowsPerGroup = 64;
constexpr int kDColsPerBlock = 32;
constexpr int kDBlocks = kHeadDim / kDColsPerBlock;
constexpr int kGenerations = 3;

constexpr int kKvBlockBytes = 32 * 32 * sizeof(__half);
constexpr int kKvBlocksPerTensor = (kNk / 32) * kDBlocks;
constexpr int kKBase = 0;
constexpr int kVBase = kKBase + kKvBlocksPerTensor * kKvBlockBytes;
constexpr int kKvBytes = kVBase + kKvBlocksPerTensor * kKvBlockBytes;

constexpr int kRawBlockBytes = 16 * 32 * sizeof(__half);
constexpr int kRawBlocksPerTensor = (kMq / 16) * kDBlocks;
constexpr int kRawTensorBytes = kRawBlocksPerTensor * kRawBlockBytes;
constexpr int kSidecarFieldBytes = kMq * sizeof(float);
constexpr int kSidecarBytes = 3 * kSidecarFieldBytes;
constexpr int kRawPageBytes = 2 * kRawTensorBytes + kSidecarBytes;
constexpr int kRawPages = 2;
constexpr int kRawBytes = kRawPages * kRawPageBytes;
constexpr int kLdsBytes = kKvBytes > kRawBytes ? kKvBytes : kRawBytes;
constexpr int kCapturedFieldsPerGeneration = 5;
constexpr int kSinkValues =
    kWaves * (1 + kGenerations * kCapturedFieldsPerGeneration);

static_assert(kKvBytes == 128 * 1024, "K/V startup must fill 128KB");
static_assert(kRawPageBytes == 33536, "one raw page budget");
static_assert(kRawBytes == 67072, "two raw pages budget");
static_assert(kLdsBytes == 128 * 1024, "lifecycle overlay budget");

struct Bar {
    static constexpr int kResidentFilled = 0;
    static constexpr int kResidentUsed = 1;
    static constexpr int kPage0Filled = 2;
    static constexpr int kPage0Used = 3;
    static constexpr int kPage1Filled = 4;
    static constexpr int kPage1Used = 5;
    static constexpr int kAllDone = 6;
};

struct KvFragments {
    ins::F16x8 k[kDBlocks];
    ins::F16x8 v[kDBlocks];
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x % kWaveSize);
}

__host__ __device__ constexpr float k_value(int owner_nblock) {
    return static_cast<float>(owner_nblock + 1) / 64.0f;
}

__host__ __device__ constexpr float v_value(int owner_nblock) {
    return static_cast<float>(owner_nblock + 17) / 64.0f;
}

__host__ __device__ constexpr float q_value(int generation, int m_block) {
    return static_cast<float>(generation * 8 + m_block + 1) / 32.0f;
}

__host__ __device__ constexpr float dout_value(
    int generation, int m_block) {
    return static_cast<float>(generation * 8 + m_block + 17) / 32.0f;
}

__host__ __device__ constexpr float sidecar_value(
    int generation, int field, int m_block) {
    return static_cast<float>(generation * 100 + field * 16 + m_block + 1);
}

__device__ __forceinline__ int kv_block_offset(int row_block, int d_block) {
    return (row_block * kDBlocks + d_block) * kKvBlockBytes;
}

template <int Page>
__device__ __forceinline__ constexpr int raw_page_base() {
    static_assert(Page == 0 || Page == 1, "raw page range");
    return Page * kRawPageBytes;
}

template <int Page>
__device__ __forceinline__ float* sidecar_ptr(__half* lds) {
    constexpr int kOffset =
        raw_page_base<Page>() + 2 * kRawTensorBytes;
    return reinterpret_cast<float*>(lds + kOffset / sizeof(__half));
}

template <int Group>
__device__ __forceinline__ void publish_kv_group(
    __half* lds,
    const __half* k,
    const __half* v) {
    static_assert(Group >= 0 && Group < kGroups, "group range");
#pragma clang loop unroll(disable)
    for (int row32 = 0; row32 < 2; ++row32) {
        const int row_block = Group * 2 + row32;
#pragma clang loop unroll(disable)
        for (int d_block = 0; d_block < kDBlocks; ++d_block) {
            const int row = row_block * 32;
            const int col = d_block * kDColsPerBlock;
            const __half* k_src = k + row * kHeadDim + col;
            const __half* v_src = v + row * kHeadDim + col;
            const int offset = kv_block_offset(row_block, d_block);
            ins::matrix_load_32x32_b16_bps_lds(
                lds, ins::prepare_matrix_src(k_src, kHeadDim),
                kKBase + offset, true);
            ins::matrix_load_32x32_b16_bps_lds(
                lds, ins::prepare_matrix_src(v_src, kHeadDim),
                kVBase + offset, true);
        }
    }
}

template <int Group>
__device__ __forceinline__ void latch_kv(
    const __half* lds,
    int wave_local,
    KvFragments& kv) {
    // Each group covers two N32 blocks; wave_local selects one N16 owner.
    const int owner_row_block = Group * 2 + (wave_local >> 1);
    const int half = wave_local & 1;
#pragma unroll
    for (int d_block = 0; d_block < kDBlocks; ++d_block) {
        const int offset = kv_block_offset(owner_row_block, d_block) +
            half * kRawBlockBytes;
        ins::ds_read_matrix_32x16_trans(
            lds, kKBase + offset, kv.k[d_block].f16x8);
        ins::ds_read_matrix_32x16_trans(
            lds, kVBase + offset, kv.v[d_block].f16x8);
        ins::wait_lgkm(0);
    }
}

__device__ __forceinline__ void keep_kv_live(KvFragments& kv) {
#pragma unroll
    for (int d_block = 0; d_block < kDBlocks; ++d_block) {
        asm volatile("" : "+v"(kv.k[d_block].f16x8),
                         "+v"(kv.v[d_block].f16x8)
                     :
                     : "memory");
    }
}

template <int Generation, int Page, int Group>
__device__ __forceinline__ void publish_raw_group(
    __half* lds,
    const __half* q,
    const __half* dout,
    const float* sidecar,
    int lane) {
    static_assert(Generation >= 0 && Generation < kGenerations,
                  "generation range");
    const __half* q_gen = q + Generation * kMq * kHeadDim;
    const __half* dout_gen = dout + Generation * kMq * kHeadDim;
#pragma unroll
    for (int d_block = 0; d_block < kDBlocks; ++d_block) {
        const int row = Group * kMRowsPerGroup;
        const int col = d_block * kDColsPerBlock;
        const int q_offset = raw_page_base<Page>() +
            (Group * kDBlocks + d_block) * kRawBlockBytes;
        const int dout_offset = q_offset + kRawTensorBytes;
        ins::matrix_load_32x16_b16_bps_lds(
            lds,
            ins::prepare_matrix_src(q_gen + row * kHeadDim + col, kHeadDim),
            q_offset);
        ins::matrix_load_32x16_b16_bps_lds(
            lds,
            ins::prepare_matrix_src(
                dout_gen + row * kHeadDim + col, kHeadDim),
            dout_offset);
    }
    if (lane < kMRowsPerGroup) {
        const int row = Group * kMRowsPerGroup + lane;
        const float* src = sidecar +
            (Generation * kMq + row) * 3;
        float* dst = sidecar_ptr<Page>(lds);
        dst[row] = src[0];
        dst[kMq + row] = src[1];
        dst[2 * kMq + row] = src[2];
    }
}

template <int Generation, int Page>
__device__ __forceinline__ void capture_raw_page(
    const __half* lds,
    int check_group,
    int check_dblock,
    int lane,
    uint32_t wave_id,
    float* sink) {
    ins::F16x8 q_frag;
    ins::F16x8 dout_frag;
    const int q_offset = raw_page_base<Page>() +
        (check_group * kDBlocks + check_dblock) * kRawBlockBytes;
    ins::ds_read_matrix_32x16_normal(
        lds, q_offset, q_frag.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, q_offset + kRawTensorBytes, dout_frag.f16x8);
    ins::wait_lgkm(0);

    const int row = check_group * kMRowsPerGroup + lane % kMRowsPerGroup;
    const float* sidecar = sidecar_ptr<Page>(
        const_cast<__half*>(lds));
    const float sidecar0 = sidecar[row];
    const float sidecar1 = sidecar[kMq + row];
    const float sidecar2 = sidecar[2 * kMq + row];
    if (lane == 0) {
        const int wave = static_cast<int>(wave_id);
        const int base = kWaves +
            Generation * kCapturedFieldsPerGeneration * kWaves;
        sink[base + 0 * kWaves + wave] =
            static_cast<float>(q_frag.f16x8[0]);
        sink[base + 1 * kWaves + wave] =
            static_cast<float>(dout_frag.f16x8[0]);
        sink[base + 2 * kWaves + wave] = sidecar0;
        sink[base + 3 * kWaves + wave] = sidecar1;
        sink[base + 4 * kWaves + wave] = sidecar2;
    }
}

template <int Generation, int Page, int Group, int FilledBarrier,
          int UsedBarrier>
__device__ __forceinline__ void run_generation(
    __half* lds,
    const __half* q,
    const __half* dout,
    const float* sidecar,
    int wave_local,
    int lane,
    uint32_t wave_id,
    float* sink,
    int& filled_phase,
    int& used_phase) {
    constexpr int kPublisherWaveLocal = Group;
    if constexpr (Generation >= kRawPages) {
        if (wave_local == kPublisherWaveLocal) {
            ins::abarrier_try_wait<true>(UsedBarrier, used_phase);
        }
    }
    if (wave_local == kPublisherWaveLocal) {
        ins::abarrier_seq<false>(FilledBarrier);
        publish_raw_group<Generation, Page, Group>(
            lds, q, dout, sidecar, lane);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(FilledBarrier, 1);
    }

    ins::abarrier_try_wait<true>(FilledBarrier, filled_phase);
    capture_raw_page<Generation, Page>(
        lds, Group, wave_local, lane, wave_id, sink);
    ins::abarrier_arrive_cnt<false>(UsedBarrier, 1);
}

template <int Group>
__device__ __forceinline__ void run_group(
    __half* lds,
    const __half* k,
    const __half* v,
    const __half* q,
    const __half* dout,
    const float* sidecar,
    float* sink,
    int* status,
    int wave_local,
    int lane,
    uint32_t wave_id) {
    constexpr int kPublisherWaveLocal = Group;
    if (wave_local == kPublisherWaveLocal) {
        ins::abarrier_seq<false>(Bar::kResidentFilled);
        publish_kv_group<Group>(lds, k, v);
        ins::maybe_wait_bps_vbcnt_before_arrive();
        ins::abarrier_arrive_cnt<false>(Bar::kResidentFilled, 1);
    }

    int resident_filled_phase = 0;
    ins::abarrier_try_wait<true>(
        Bar::kResidentFilled, resident_filled_phase);
    KvFragments kv;
    latch_kv<Group>(lds, wave_local, kv);
    keep_kv_live(kv);
    ins::abarrier_arrive_cnt<false>(Bar::kResidentUsed, 1);
    int resident_used_phase = 0;
    ins::abarrier_try_wait<true>(Bar::kResidentUsed, resident_used_phase);

    int page0_filled_phase = 0;
    int page0_used_phase = 0;
    int page1_filled_phase = 0;
    int page1_used_phase = 0;
    run_generation<0, 0, Group, Bar::kPage0Filled, Bar::kPage0Used>(
        lds, q, dout, sidecar, wave_local, lane,
        wave_id, sink,
        page0_filled_phase, page0_used_phase);
    run_generation<1, 1, Group, Bar::kPage1Filled, Bar::kPage1Used>(
        lds, q, dout, sidecar, wave_local, lane,
        wave_id, sink,
        page1_filled_phase, page1_used_phase);
    run_generation<2, 0, Group, Bar::kPage0Filled, Bar::kPage0Used>(
        lds, q, dout, sidecar, wave_local, lane,
        wave_id, sink,
        page0_filled_phase, page0_used_phase);
    keep_kv_live(kv);
    float checksum = 0.0f;
#pragma unroll
    for (int d_block = 0; d_block < kDBlocks; ++d_block) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            checksum += static_cast<float>(kv.k[d_block].f16x8[i]);
            checksum += static_cast<float>(kv.v[d_block].f16x8[i]);
        }
    }
    if (lane == 0) {
        sink[static_cast<int>(wave_id)] = checksum;
        status[static_cast<int>(wave_id)] = 1;
    }
    ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
}

__global__ void __launch_bounds__(kThreads, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dkv_owner16_4c_lifecycle_probe_kernel(
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ q,
    const __half* __restrict__ dout,
    const float* __restrict__ sidecar,
    float* __restrict__ sink,
    int* __restrict__ status) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(128, 128, 128, 128);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kResidentUsed, 16);
        __builtin_hcu_s_abarrier_init(Bar::kPage0Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage0Used, 16);
        __builtin_hcu_s_abarrier_init(Bar::kPage1Filled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kPage1Used, 16);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(128);
        const int wave_local = static_cast<int>(wave_id);
        const int lane = lane_id();
        run_group<0>(
            lds, k, v, q, dout, sidecar, sink, status,
            wave_local, lane, wave_id);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(128);
        const int wave_local = static_cast<int>(wave_id - 4);
        const int lane = lane_id();
        run_group<1>(
            lds, k, v, q, dout, sidecar, sink, status,
            wave_local, lane, wave_id);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(128);
        const int wave_local = static_cast<int>(wave_id - 8);
        const int lane = lane_id();
        run_group<2>(
            lds, k, v, q, dout, sidecar, sink, status,
            wave_local, lane, wave_id);
    } else {
        __builtin_hcu_s_set_vgpr_size(128);
        const int wave_local = static_cast<int>(wave_id - 12);
        const int lane = lane_id();
        run_group<3>(
            lds, k, v, q, dout, sidecar, sink, status,
            wave_local, lane, wave_id);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kResidentFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kResidentUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kPage0Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage0Used);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1Filled);
        __builtin_hcu_s_abarrier_inv(Bar::kPage1Used);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)k;
    (void)v;
    (void)q;
    (void)dout;
    (void)sidecar;
    (void)sink;
    (void)status;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace

int main() {
    std::vector<__half> k(kNk * kHeadDim);
    std::vector<__half> v(kNk * kHeadDim);
    std::vector<__half> q(kGenerations * kMq * kHeadDim);
    std::vector<__half> dout(kGenerations * kMq * kHeadDim);
    std::vector<float> sidecar(kGenerations * kMq * 3);
    std::vector<float> sink(kSinkValues, 0.0f);

    for (int row = 0; row < kNk; ++row) {
        const int owner = row / 16;
        for (int d = 0; d < kHeadDim; ++d) {
            k[row * kHeadDim + d] = __float2half(k_value(owner));
            v[row * kHeadDim + d] = __float2half(v_value(owner));
        }
    }
    for (int generation = 0; generation < kGenerations; ++generation) {
        for (int row = 0; row < kMq; ++row) {
            const int m_block = row / kMRowsPerGroup;
            for (int d = 0; d < kHeadDim; ++d) {
                const int index =
                    (generation * kMq + row) * kHeadDim + d;
                q[index] = __float2half(q_value(generation, m_block));
                dout[index] =
                    __float2half(dout_value(generation, m_block));
            }
            for (int field = 0; field < 3; ++field) {
                sidecar[(generation * kMq + row) * 3 + field] =
                    sidecar_value(generation, field, m_block);
            }
        }
    }

    __half* d_k = nullptr;
    __half* d_v = nullptr;
    __half* d_q = nullptr;
    __half* d_dout = nullptr;
    float* d_sidecar = nullptr;
    float* d_sink = nullptr;
    std::vector<int> status(kWaves, 0);
    int* d_status = nullptr;
    check_hip(hipMalloc(&d_k, k.size() * sizeof(__half)), "hipMalloc k");
    check_hip(hipMalloc(&d_v, v.size() * sizeof(__half)), "hipMalloc v");
    check_hip(hipMalloc(&d_q, q.size() * sizeof(__half)), "hipMalloc q");
    check_hip(
        hipMalloc(&d_dout, dout.size() * sizeof(__half)), "hipMalloc dout");
    check_hip(
        hipMalloc(&d_sidecar, sidecar.size() * sizeof(float)),
        "hipMalloc sidecar");
    check_hip(hipMalloc(&d_sink, sink.size() * sizeof(float)), "hipMalloc sink");
    check_hip(
        hipMalloc(&d_status, status.size() * sizeof(int)),
        "hipMalloc status");

    check_hip(
        hipMemcpy(d_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice),
        "hipMemcpy k");
    check_hip(
        hipMemcpy(d_v, v.data(), v.size() * sizeof(__half), hipMemcpyHostToDevice),
        "hipMemcpy v");
    check_hip(
        hipMemcpy(d_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice),
        "hipMemcpy q");
    check_hip(
        hipMemcpy(
            d_dout, dout.data(), dout.size() * sizeof(__half),
            hipMemcpyHostToDevice),
        "hipMemcpy dout");
    check_hip(
        hipMemcpy(
            d_sidecar, sidecar.data(), sidecar.size() * sizeof(float),
            hipMemcpyHostToDevice),
        "hipMemcpy sidecar");
    check_hip(hipMemset(d_sink, 0, sink.size() * sizeof(float)),
              "hipMemset sink");
    check_hip(
        hipMemset(d_status, 0, status.size() * sizeof(int)),
        "hipMemset status");

    hipLaunchKernelGGL(
        dkv_owner16_4c_lifecycle_probe_kernel, dim3(1), dim3(kThreads), 0, 0,
        d_k, d_v, d_q, d_dout, d_sidecar, d_sink, d_status);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    check_hip(
        hipMemcpy(
            sink.data(), d_sink, sink.size() * sizeof(float),
            hipMemcpyDeviceToHost),
        "hipMemcpy sink");
    check_hip(
        hipMemcpy(
            status.data(), d_status, status.size() * sizeof(int),
            hipMemcpyDeviceToHost),
        "hipMemcpy status");
    int bad = 0;
    for (int wave = 0; wave < kWaves; ++wave) {
        bad += status[wave] != 1;
        const int owner_nblock = wave;
        const float expected_checksum =
            kDBlocks * 8.0f *
            (k_value(owner_nblock) + v_value(owner_nblock));
        bad += sink[wave] != expected_checksum;
        const int group = wave / kWavesPerGroup;
        for (int generation = 0; generation < kGenerations; ++generation) {
            const int base = kWaves +
                generation * kCapturedFieldsPerGeneration * kWaves;
            bad += sink[base + 0 * kWaves + wave] !=
                   q_value(generation, group);
            bad += sink[base + 1 * kWaves + wave] !=
                   dout_value(generation, group);
            bad += sink[base + 2 * kWaves + wave] !=
                   sidecar_value(generation, 0, group);
            bad += sink[base + 3 * kWaves + wave] !=
                   sidecar_value(generation, 1, group);
            bad += sink[base + 4 * kWaves + wave] !=
                   sidecar_value(generation, 2, group);
        }
    }
    if (bad != 0) {
        for (int wave = 0; wave < kWaves; ++wave) {
            std::printf(
                "lifecycle wave=%d heartbeat=%d checksum=%g\n",
                wave, status[wave], sink[wave]);
            for (int generation = 0; generation < kGenerations;
                 ++generation) {
                const int base = kWaves +
                    generation * kCapturedFieldsPerGeneration * kWaves;
                std::printf(
                    "  gen=%d q=%g dout=%g sidecar=%g/%g/%g\n",
                    generation,
                    sink[base + 0 * kWaves + wave],
                    sink[base + 1 * kWaves + wave],
                    sink[base + 2 * kWaves + wave],
                    sink[base + 3 * kWaves + wave],
                    sink[base + 4 * kWaves + wave]);
            }
        }
    }
    const bool pass = bad == 0;
    std::printf(
        "dkv_owner16_4c_lifecycle_probe bad=%d lds=%d "
        "resident=131072 raw2=67072 generations=3 roles=128/128/128/128 "
        "pass=%d\n",
        bad, kLdsBytes, pass ? 1 : 0);

    check_hip(hipFree(d_k), "hipFree k");
    check_hip(hipFree(d_v), "hipFree v");
    check_hip(hipFree(d_q), "hipFree q");
    check_hip(hipFree(d_dout), "hipFree dout");
    check_hip(hipFree(d_sidecar), "hipFree sidecar");
    check_hip(hipFree(d_sink), "hipFree sink");
    check_hip(hipFree(d_status), "hipFree status");
    return pass ? 0 : 2;
}
