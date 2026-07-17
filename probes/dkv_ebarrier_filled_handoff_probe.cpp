#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kWaves = 16;
constexpr int kThreads = kWaveSize * kWaves;
constexpr int kProducerWaves = 8;
constexpr int kConsumerWaves = 8;
constexpr int kRows = 16;
constexpr int kDim = 128;
constexpr int kDimPerProducerWave = 32;
constexpr int kGenerations = 16;
constexpr int kTileElements = kRows * kDim;
constexpr int kWaveBlockBytes =
    kRows * kDimPerProducerWave * static_cast<int>(sizeof(__half));
constexpr int kQBase = 0;
constexpr int kDoutBase = kTileElements * static_cast<int>(sizeof(__half));
constexpr int kLdsBytes = 2 * kTileElements * static_cast<int>(sizeof(__half));
constexpr int kFragValues = 16;

struct Bar {
    static constexpr int kFilled = 0;
    static constexpr int kUsed = 1;
    static constexpr int kAllDone = 2;
};

constexpr int kFilledEBarrier = 1;

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

__device__ __forceinline__ void publish_m16_d128(
    __half* lds,
    int lds_base,
    const __half* src,
    int generation,
    int wave_local) {
    const __half* src_tile =
        src + generation * kTileElements + wave_local * kDimPerProducerWave;
    const ins::Vec4U32 srsrc = ins::prepare_matrix_src(src_tile, kDim);
    ins::matrix_load_32x16_b16_bps_lds(
        lds, srsrc, lds_base + wave_local * kWaveBlockBytes);
}

__device__ __forceinline__ void store_fragments(
    __half* out,
    int generation,
    int consumer_wave,
    int lane,
    const ins::F16x8& q_frag,
    const ins::F16x8& dout_frag) {
    const int base =
        ((generation * kConsumerWaves + consumer_wave) * kWaveSize + lane) *
        kFragValues;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[base + i] = static_cast<__half>(q_frag.f16x8[i]);
        out[base + 8 + i] = static_cast<__half>(dout_frag.f16x8[i]);
    }
}

template <bool UseEBarrierFilled>
__device__ __forceinline__ void run_producer(
    const __half* q,
    const __half* dout,
    __half* lds,
    int wave_local,
    bool publish_q) {
        int used_phase = 0;
#pragma clang loop unroll(disable)
        for (int generation = 0; generation < kGenerations; ++generation) {
            if (generation > 0) {
                ins::abarrier_try_wait<true>(Bar::kUsed, used_phase);
            }
            if constexpr (!UseEBarrierFilled) {
                ins::abarrier_seq<false>(Bar::kFilled);
            }
            publish_m16_d128(
                lds, publish_q ? kQBase : kDoutBase,
                publish_q ? q : dout, generation, wave_local);
            ins::maybe_wait_bps_vbcnt_before_arrive();
            if constexpr (UseEBarrierFilled) {
                __builtin_hcu_s_ebarrier_arrive_cnt(
                    kFilledEBarrier, kWaves);
            } else {
                ins::abarrier_arrive_cnt<false>(Bar::kFilled, 1);
            }
        }
}

template <bool UseEBarrierFilled>
__device__ __forceinline__ void run_consumer(
    const __half* lds,
    __half* out,
    int consumer_wave,
    int lane) {
        int filled_phase = 0;
#pragma clang loop unroll(disable)
        for (int generation = 0; generation < kGenerations; ++generation) {
            if constexpr (UseEBarrierFilled) {
                __builtin_hcu_s_ebarrier_sync_cnt(
                    kFilledEBarrier, kWaves);
            } else {
                ins::abarrier_try_wait<true>(Bar::kFilled, filled_phase);
            }
            ins::F16x8 q_frag;
            ins::F16x8 dout_frag;
            ins::ds_read_matrix_32x16_trans(
                lds, kQBase, q_frag.f16x8);
            ins::ds_read_matrix_32x16_trans(
                lds, kDoutBase, dout_frag.f16x8);
            ins::wait_lgkm(0);
            store_fragments(
                out, generation, consumer_wave, lane, q_frag, dout_frag);
            ins::abarrier_arrive_cnt<false>(Bar::kUsed, 1);
        }
}

template <bool UseEBarrierFilled>
__global__ void __launch_bounds__(kThreads, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
dkv_filled_handoff_probe_kernel(const __half* __restrict__ q,
                                const __half* __restrict__ dout,
                                __half* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds[kLdsBytes / sizeof(__half)];
    const uint32_t wave_id = __builtin_hcu_get_wave_id();
    const int wave_local = static_cast<int>(wave_id & 3u);

    if (wave_id == 0) {
        if constexpr (!UseEBarrierFilled) {
            __builtin_hcu_s_abarrier_init(Bar::kFilled, kProducerWaves);
        }
        __builtin_hcu_s_abarrier_init(Bar::kUsed, kConsumerWaves);
        __builtin_hcu_s_abarrier_init(Bar::kAllDone, kWaves);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave_id < 4) {
        __builtin_hcu_s_set_vgpr_size(8);
        run_producer<UseEBarrierFilled>(
            q, dout, lds, wave_local, true);
    } else if (wave_id < 8) {
        __builtin_hcu_s_set_vgpr_size(40);
        run_consumer<UseEBarrierFilled>(
            lds, out, static_cast<int>(wave_id - 4),
            static_cast<int>(threadIdx.x % kWaveSize));
    } else if (wave_id < 12) {
        __builtin_hcu_s_set_vgpr_size(40);
        run_consumer<UseEBarrierFilled>(
            lds, out, static_cast<int>(wave_id - 4),
            static_cast<int>(threadIdx.x % kWaveSize));
    } else {
        __builtin_hcu_s_set_vgpr_size(8);
        run_producer<UseEBarrierFilled>(
            q, dout, lds, wave_local, false);
    }

    ins::abarrier_arrive_cnt<false>(Bar::kAllDone, 1);
    int done_phase = 0;
    ins::abarrier_try_wait<false>(Bar::kAllDone, done_phase);
    __builtin_hcu_s_ebarrier_sync(0);
    if (wave_id == 0) {
        if constexpr (!UseEBarrierFilled) {
            __builtin_hcu_s_abarrier_inv(Bar::kFilled);
        }
        __builtin_hcu_s_abarrier_inv(Bar::kUsed);
        __builtin_hcu_s_abarrier_inv(Bar::kAllDone);
    }
#else
    (void)q;
    (void)dout;
    (void)out;
#endif
}

std::vector<__half> make_input(int salt) {
    std::vector<__half> values(kGenerations * kTileElements);
    for (int generation = 0; generation < kGenerations; ++generation) {
        for (int row = 0; row < kRows; ++row) {
            for (int d = 0; d < kDim; ++d) {
                const int raw =
                    ((generation * 19 + row * 13 + d * 7 + salt) % 61) - 30;
                values[generation * kTileElements + row * kDim + d] =
                    __float2half(raw * 0.03125f);
            }
        }
    }
    return values;
}

uint16_t half_bits(__half value) {
    union {
        __half h;
        uint16_t u;
    } bits{value};
    return bits.u;
}

}  // namespace

int main() {
    const std::vector<__half> q = make_input(3);
    const std::vector<__half> dout = make_input(17);
    const size_t output_count =
        static_cast<size_t>(kGenerations) * kConsumerWaves * kWaveSize *
        kFragValues;
    std::vector<__half> abarrier_out(output_count);
    std::vector<__half> ebarrier_out(output_count);

    __half *d_q = nullptr, *d_dout = nullptr;
    __half *d_abarrier_out = nullptr, *d_ebarrier_out = nullptr;
    check_hip(hipMalloc(&d_q, q.size() * sizeof(__half)), "hipMalloc q");
    check_hip(
        hipMalloc(&d_dout, dout.size() * sizeof(__half)), "hipMalloc dO");
    check_hip(
        hipMalloc(&d_abarrier_out, output_count * sizeof(__half)),
        "hipMalloc ABarrier output");
    check_hip(
        hipMalloc(&d_ebarrier_out, output_count * sizeof(__half)),
        "hipMalloc EBarrier output");
    check_hip(
        hipMemcpy(
            d_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice),
        "copy q");
    check_hip(
        hipMemcpy(
            d_dout, dout.data(), dout.size() * sizeof(__half),
            hipMemcpyHostToDevice),
        "copy dO");

    hipLaunchKernelGGL(
        (dkv_filled_handoff_probe_kernel<false>), dim3(1), dim3(kThreads), 0,
        0, d_q, d_dout, d_abarrier_out);
    check_hip(hipGetLastError(), "launch ABarrier reference");
    check_hip(hipDeviceSynchronize(), "sync ABarrier reference");
    hipLaunchKernelGGL(
        (dkv_filled_handoff_probe_kernel<true>), dim3(1), dim3(kThreads), 0,
        0, d_q, d_dout, d_ebarrier_out);
    check_hip(hipGetLastError(), "launch EBarrier candidate");
    check_hip(hipDeviceSynchronize(), "sync EBarrier candidate");

    check_hip(
        hipMemcpy(
            abarrier_out.data(), d_abarrier_out,
            output_count * sizeof(__half), hipMemcpyDeviceToHost),
        "copy ABarrier output");
    check_hip(
        hipMemcpy(
            ebarrier_out.data(), d_ebarrier_out,
            output_count * sizeof(__half), hipMemcpyDeviceToHost),
        "copy EBarrier output");

    int errors = 0;
    int nonzero = 0;
    int generation_changes = 0;
    for (size_t i = 0; i < output_count; ++i) {
        const uint16_t reference = half_bits(abarrier_out[i]);
        const uint16_t candidate = half_bits(ebarrier_out[i]);
        errors += reference != candidate;
        nonzero += candidate != 0;
        if (i >= output_count / kGenerations) {
            const size_t prior = i - output_count / kGenerations;
            generation_changes +=
                candidate != half_bits(ebarrier_out[prior]);
        }
    }
    const int pass = errors == 0 && nonzero > 0 && generation_changes > 0;
    std::printf(
        "dkv_ebarrier_filled_handoff_probe generations=%d errors=%d "
        "nonzero=%d generation_changes=%d pass=%d\n",
        kGenerations, errors, nonzero, generation_changes, pass);

    check_hip(hipFree(d_q), "hipFree q");
    check_hip(hipFree(d_dout), "hipFree dO");
    check_hip(hipFree(d_abarrier_out), "hipFree ABarrier output");
    check_hip(hipFree(d_ebarrier_out), "hipFree EBarrier output");
    return pass ? 0 : 1;
}
