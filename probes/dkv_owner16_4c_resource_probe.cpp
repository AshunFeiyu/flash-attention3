#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kGroups = 4;
constexpr int kWavesPerGroup = 4;
constexpr int kWaves = kGroups * kWavesPerGroup;
constexpr int kThreads = kWaves * kWaveSize;
constexpr int kMq = 64;
constexpr int kHeadDim = 128;
constexpr int kMRowsPerGroup = 16;
constexpr int kDColsPerWave = 32;
constexpr int kDBlocks = kHeadDim / kDColsPerWave;
constexpr int kRawBlockBytes =
    kMRowsPerGroup * kDColsPerWave * sizeof(__half);
constexpr int kTensorBytes = kMq * kHeadDim * sizeof(__half);
constexpr int kQBase = 0;
constexpr int kDoutBase = kQBase + kTensorBytes;
constexpr int kSidecarBase = kDoutBase + kTensorBytes;
constexpr int kSidecarFieldBytes = kMq * sizeof(float);
constexpr int kLdsBytes = kSidecarBase + 3 * kSidecarFieldBytes;
constexpr int kAccVectors = 8;
constexpr int kSinkValues = kWaves * kWaveSize;

static_assert(kLdsBytes == 33536, "one M64 raw page plus sidecar");

struct Bar {
    static constexpr int kFilled = 0;
    static constexpr int kUsed = 1;
    static constexpr int kAllDone = 2;
};

struct KvFragments {
    ins::F16x8 k[kDBlocks];
    ins::F16x8 v[kDBlocks];
};

__device__ __forceinline__ int lane_id() {
    return static_cast<int>(threadIdx.x % kWaveSize);
}

template <int Group, int DBlock>
__device__ __forceinline__ constexpr int raw_offset() {
    static_assert(Group >= 0 && Group < kGroups, "group range");
    static_assert(DBlock >= 0 && DBlock < kDBlocks, "D block range");
    return (Group * kDBlocks + DBlock) * kRawBlockBytes;
}

template <int Group>
__device__ __forceinline__ void publish_group_raw(
    __half* lds,
    const __half* q,
    const __half* dout,
    const float* packed_sidecar,
    int wave_local,
    int lane) {
    const int row_base = Group * kMRowsPerGroup;
    const int d_base = wave_local * kDColsPerWave;
    const __half* q_src =
        q + static_cast<int64_t>(row_base) * kHeadDim + d_base;
    const __half* dout_src =
        dout + static_cast<int64_t>(row_base) * kHeadDim + d_base;
    ins::Vec4U32 q_desc = ins::prepare_matrix_src(q_src, kHeadDim);
    ins::Vec4U32 dout_desc = ins::prepare_matrix_src(dout_src, kHeadDim);
    const int lds_offset =
        (Group * kDBlocks + wave_local) * kRawBlockBytes;
    ins::matrix_load_32x16_b16_bps_lds(
        lds, q_desc, kQBase + lds_offset);
    ins::matrix_load_32x16_b16_bps_lds(
        lds, dout_desc, kDoutBase + lds_offset);

    if (wave_local == 0 && lane < kMRowsPerGroup) {
        const int row = row_base + lane;
        float* sidecar = reinterpret_cast<float*>(
            lds + kSidecarBase / static_cast<int>(sizeof(__half)));
        const float* src = packed_sidecar + row * 3;
        sidecar[row] = src[0];
        sidecar[kMq + row] = src[1];
        sidecar[2 * kMq + row] = src[2];
    }
}

template <int Group, int DBlock>
__device__ __forceinline__ void latch_kv_dblock(
    const __half* lds,
    KvFragments& kv) {
    ins::ds_read_matrix_32x16_trans(
        lds, kQBase + raw_offset<Group, DBlock>(),
        kv.k[DBlock].f16x8);
    ins::ds_read_matrix_32x16_trans(
        lds, kDoutBase + raw_offset<Group, DBlock>(),
        kv.v[DBlock].f16x8);
}

template <int Group>
__device__ __forceinline__ void latch_kv(
    const __half* lds,
    KvFragments& kv) {
    latch_kv_dblock<Group, 0>(lds, kv);
    latch_kv_dblock<Group, 1>(lds, kv);
    latch_kv_dblock<Group, 2>(lds, kv);
    latch_kv_dblock<Group, 3>(lds, kv);
    ins::wait_lgkm(0);
}

__device__ __forceinline__ void keep_output_live(
    ins::F32x4 (&dv)[kAccVectors],
    ins::F32x4 (&dk)[kAccVectors]) {
#pragma unroll
    for (int i = 0; i < kAccVectors; ++i) {
        ins::keep_accumulator_live(dv[i]);
        ins::keep_accumulator_live(dk[i]);
    }
}

template <int DBlock, int KHalf>
__device__ __forceinline__ void score_dp_half(
    const KvFragments& kv,
    const ins::F16x8& q_frag,
    const ins::F16x8& dout_frag,
    const ins::F16x8& zero,
    ins::F32x4& score,
    ins::F32x4& dp) {
    static_assert(KHalf == 0 || KHalf == 1, "K half range");
    constexpr bool kFirst = DBlock == 0 && KHalf == 0;
    score.f32 = ins::mmac_f16_lit(
        kv.k[DBlock].f16x4[KHalf], q_frag.f16x4[KHalf],
        kFirst ? zero.f32 : score.f32);
    dp.f32 = ins::mmac_f16_lit(
        kv.v[DBlock].f16x4[KHalf], dout_frag.f16x4[KHalf],
        kFirst ? zero.f32 : dp.f32);
}

template <int Group, int DBlock>
__device__ __forceinline__ void score_dp_dblock(
    const __half* lds,
    const KvFragments& kv,
    const ins::F16x8& zero,
    ins::F32x4& score,
    ins::F32x4& dp) {
    ins::F16x8 q_frag;
    ins::F16x8 dout_frag;
    ins::ds_read_matrix_32x16_trans(
        lds, kQBase + raw_offset<Group, DBlock>(), q_frag.f16x8);
    ins::ds_read_matrix_32x16_trans(
        lds, kDoutBase + raw_offset<Group, DBlock>(), dout_frag.f16x8);
    ins::wait_lgkm(0);
    score_dp_half<DBlock, 0>(kv, q_frag, dout_frag, zero, score, dp);
    score_dp_half<DBlock, 1>(kv, q_frag, dout_frag, zero, score, dp);
}

template <int DBlock, int DHalf>
__device__ __forceinline__ void update_output_half(
    const ins::Vec4F16& p,
    const ins::Vec4F16& ds,
    const ins::F16x8& q_frag,
    const ins::F16x8& dout_frag,
    ins::F32x4 (&dv)[kAccVectors],
    ins::F32x4 (&dk)[kAccVectors]) {
    static_assert(DHalf == 0 || DHalf == 1, "D half range");
    constexpr int kOut = DBlock * 2 + DHalf;
    dv[kOut].f32 = ins::mmac_f16_lit(
        p, dout_frag.f16x4[DHalf], dv[kOut].f32);
    dk[kOut].f32 = ins::mmac_f16_lit(
        ds, q_frag.f16x4[DHalf], dk[kOut].f32);
}

template <int Group, int DBlock>
__device__ __forceinline__ void update_output_dblock(
    const __half* lds,
    const ins::Vec4F16& p,
    const ins::Vec4F16& ds,
    ins::F32x4 (&dv)[kAccVectors],
    ins::F32x4 (&dk)[kAccVectors]) {
    ins::F16x8 q_frag;
    ins::F16x8 dout_frag;
    ins::ds_read_matrix_32x16_normal(
        lds, kQBase + raw_offset<Group, DBlock>(), q_frag.f16x8);
    ins::ds_read_matrix_32x16_normal(
        lds, kDoutBase + raw_offset<Group, DBlock>(), dout_frag.f16x8);
    ins::wait_lgkm(0);
    update_output_half<DBlock, 0>(p, ds, q_frag, dout_frag, dv, dk);
    update_output_half<DBlock, 1>(p, ds, q_frag, dout_frag, dv, dk);
}

template <int Group>
__device__ __forceinline__ void run_group(
    const __half* q,
    const __half* dout,
    const float* packed_sidecar,
    float* sink,
    int* status,
    __half* lds,
    int wave_local,
    int lane,
    uint32_t wave_id) {
    ins::abarrier_seq<false>(Bar::kFilled);
    publish_group_raw<Group>(
        lds, q, dout, packed_sidecar, wave_local, lane);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    ins::abarrier_arrive_cnt<false>(Bar::kFilled, 1);

    int filled_phase = 0;
    ins::abarrier_try_wait<true>(Bar::kFilled, filled_phase);
    KvFragments kv;
    latch_kv<Group>(lds, kv);

    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    ins::F32x4 dv[kAccVectors];
    ins::F32x4 dk[kAccVectors];
#pragma unroll
    for (int i = 0; i < kAccVectors; ++i) {
        dv[i].f32 = zero.f32;
        dk[i].f32 = zero.f32;
    }
    keep_output_live(dv, dk);

    ins::F32x4 score;
    ins::F32x4 dp;
    score_dp_dblock<Group, 0>(lds, kv, zero, score, dp);
    score_dp_dblock<Group, 1>(lds, kv, zero, score, dp);
    score_dp_dblock<Group, 2>(lds, kv, zero, score, dp);
    score_dp_dblock<Group, 3>(lds, kv, zero, score, dp);
    keep_output_live(dv, dk);

    const int lane_col_group = lane >> 4;
    const int local_row = Group * kMRowsPerGroup + lane_col_group * 4;
    const float* sidecar = reinterpret_cast<const float*>(
        lds + kSidecarBase / static_cast<int>(sizeof(__half)));
    ins::Vec4F32 row_max;
    ins::Vec4F32 row_inv_sum;
    ins::Vec4F32 row_delta;
    ins::ds_read_b128_lds_imm3<kSidecarFieldBytes,
                               2 * kSidecarFieldBytes>(
        sidecar + local_row, row_max, row_inv_sum, row_delta);
    ins::wait_lgkm(0);

    ins::Vec4F16 p;
    ins::Vec4F16 ds;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float p_value =
            exp2f(score.scalar[i] - row_max[i]) * row_inv_sum[i];
        p[i] = static_cast<_Float16>(p_value);
        ds[i] = static_cast<_Float16>(
            p_value * (dp.scalar[i] - row_delta[i]));
    }

    update_output_dblock<Group, 0>(lds, p, ds, dv, dk);
    update_output_dblock<Group, 1>(lds, p, ds, dv, dk);
    update_output_dblock<Group, 2>(lds, p, ds, dv, dk);
    update_output_dblock<Group, 3>(lds, p, ds, dv, dk);

    float checksum = 0.0f;
#pragma unroll
    for (int i = 0; i < kAccVectors; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            checksum += dv[i].scalar[j] + dk[i].scalar[j];
        }
    }
    sink[static_cast<int>(wave_id) * kWaveSize + lane] = checksum;
    if (!isfinite(checksum)) {
        atomicAdd(status, 1);
    }

    ins::abarrier_arrive_cnt<false>(Bar::kUsed, 1);
    int used_phase = 0;
    ins::abarrier_try_wait<true>(Bar::kUsed, used_phase);
    ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
}

__global__ void __launch_bounds__(kThreads, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dkv_owner16_4c_resource_probe_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ dout,
    const float* __restrict__ packed_sidecar,
    float* __restrict__ sink,
    int* __restrict__ status) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(128, 128, 128, 128);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();

    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kFilled, 16);
        __builtin_hcu_s_abarrier_init(Bar::kUsed, 16);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, 16);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(128);
        const int lane = lane_id();
        run_group<0>(
            q, dout, packed_sidecar, sink, status, lds,
            static_cast<int>(wave_id), lane, wave_id);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(128);
        const int lane = lane_id();
        run_group<1>(
            q, dout, packed_sidecar, sink, status, lds,
            static_cast<int>(wave_id - 4), lane, wave_id);
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(128);
        const int lane = lane_id();
        run_group<2>(
            q, dout, packed_sidecar, sink, status, lds,
            static_cast<int>(wave_id - 8), lane, wave_id);
    } else {
        __builtin_hcu_s_set_vgpr_size(128);
        const int lane = lane_id();
        run_group<3>(
            q, dout, packed_sidecar, sink, status, lds,
            static_cast<int>(wave_id - 12), lane, wave_id);
    }

    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        __builtin_hcu_s_abarrier_inv(Bar::kFilled);
        __builtin_hcu_s_abarrier_inv(Bar::kUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)dout;
    (void)packed_sidecar;
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
    std::vector<__half> q(kMq * kHeadDim, __float2half(0.125f));
    std::vector<__half> dout(kMq * kHeadDim, __float2half(0.25f));
    std::vector<float> sidecar(kMq * 3, 0.0f);
    std::vector<float> sink(kSinkValues, 0.0f);
    for (int row = 0; row < kMq; ++row) {
        sidecar[row * 3 + 0] = 0.0f;
        sidecar[row * 3 + 1] = 1.0f;
        sidecar[row * 3 + 2] = 0.0f;
    }

    __half* d_q = nullptr;
    __half* d_dout = nullptr;
    float* d_sidecar = nullptr;
    float* d_sink = nullptr;
    int* d_status = nullptr;
    check_hip(hipMalloc(&d_q, q.size() * sizeof(__half)), "hipMalloc q");
    check_hip(
        hipMalloc(&d_dout, dout.size() * sizeof(__half)), "hipMalloc dout");
    check_hip(
        hipMalloc(&d_sidecar, sidecar.size() * sizeof(float)),
        "hipMalloc sidecar");
    check_hip(
        hipMalloc(&d_sink, sink.size() * sizeof(float)), "hipMalloc sink");
    check_hip(hipMalloc(&d_status, sizeof(int)), "hipMalloc status");
    check_hip(
        hipMemcpy(
            d_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice),
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
    check_hip(hipMemset(d_status, 0, sizeof(int)), "hipMemset status");

    hipLaunchKernelGGL(
        dkv_owner16_4c_resource_probe_kernel, dim3(1), dim3(kThreads), 0, 0,
        d_q, d_dout, d_sidecar, d_sink, d_status);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");

    int status = 0;
    check_hip(
        hipMemcpy(
            sink.data(), d_sink, sink.size() * sizeof(float),
            hipMemcpyDeviceToHost),
        "hipMemcpy sink");
    check_hip(
        hipMemcpy(&status, d_status, sizeof(int), hipMemcpyDeviceToHost),
        "hipMemcpy status");

    int bad = status;
    for (float value : sink) {
        bad += !std::isfinite(value) || value == 0.0f;
    }
    const bool pass = bad == 0;
    std::printf(
        "dkv_owner16_4c_resource_probe bad=%d lds=%d roles=128/128/128/128 "
        "pass=%d\n",
        bad, kLdsBytes, pass ? 1 : 0);

    check_hip(hipFree(d_q), "hipFree q");
    check_hip(hipFree(d_dout), "hipFree dout");
    check_hip(hipFree(d_sidecar), "hipFree sidecar");
    check_hip(hipFree(d_sink), "hipFree sink");
    check_hip(hipFree(d_status), "hipFree status");
    return pass ? 0 : 2;
}
