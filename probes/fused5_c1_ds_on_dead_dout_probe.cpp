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
constexpr int kWaves = 16;
constexpr int kProducerEnd = 4;
constexpr int kConsumer0End = 8;
constexpr int kConsumer1End = 12;
constexpr int kGenerations = 3;
constexpr int kOwners = 4;
constexpr int kPanels = 4;
constexpr int kMatrixElements = 32 * 32;
constexpr int kMatrixBytes = kMatrixElements * sizeof(__half);
constexpr int kRawRegionMatrices = 8;
constexpr int kRawRegionBytes = kRawRegionMatrices * kMatrixBytes;
constexpr int kRawPageBytes = 2 * kRawRegionBytes;
constexpr int kRaw0Base = 0;
constexpr int kC0Ds0Base = 32 * 1024;
constexpr int kC0Ds1Base = 48 * 1024;
constexpr int kRaw1Base = 96 * 1024;
constexpr int kLdsBytes = 128 * 1024;
constexpr int kWriterStrideBytes = 1024;
constexpr int kDsPanelBytes = kOwners * kWriterStrideBytes;
constexpr int kDsPageBytes = kPanels * kDsPanelBytes;

constexpr int kRawOutputValues =
    kGenerations * 2 * kOwners * 2 * kWaveSize * 4;
constexpr int kNormalOutputValues =
    kGenerations * 2 * kPanels * kOwners * kWaveSize * 4;
constexpr int kTransOutputValues =
    kGenerations * 2 * kPanels * kOwners * kWaveSize * 4;
constexpr int kSourceValues =
    kGenerations * kRawRegionMatrices * kMatrixElements;

static_assert(kWaves * kWaveSize == 1024, "probe must use one 16-wave CTA");
static_assert(kRawRegionBytes == 16 * 1024,
              "Q and dO must each occupy 16KiB");
static_assert(kRawPageBytes == 32 * 1024,
              "each raw page must occupy 32KiB");
static_assert(kDsPageBytes == kRawRegionBytes,
              "one complete dS page must exactly alias dead dO");
static_assert(kRaw1Base + kRawPageBytes == kLdsBytes,
              "raw page1 must end at the 128KiB LDS boundary");

struct Bar {
    static constexpr int kResidentFilled = 0;
    static constexpr int kVSidecarReady = 1;
    static constexpr int kRawFilled0 = 2;
    static constexpr int kRawUsed0 = 3;
    static constexpr int kRawFilled1 = 4;
    static constexpr int kRawUsed1 = 5;
    static constexpr int kC0Filled0 = 6;
    static constexpr int kC1Filled0 = 7;
    static constexpr int kC0Done0 = 8;
    static constexpr int kC1Filled1 = 9;
    static constexpr int kC0Filled1 = 10;
    static constexpr int kC0Done1 = 11;
    static constexpr int kDoutDead0 = 12;
    static constexpr int kDoutDead1 = 13;
    static constexpr int kCount = 14;
};

union Fragment {
    ins::Vec8F16 f16x8;
    ins::Vec4F16 f16x4[2];
    _Float16 scalar[8];
};

union Accumulator {
    ins::Vec4F32 f32;
    float scalar[4];
};

__host__ __device__ constexpr int raw_page_base(int page) {
    return page == 0 ? kRaw0Base : kRaw1Base;
}

__host__ __device__ constexpr int c0_ds_base(int page) {
    return page == 0 ? kC0Ds0Base : kC0Ds1Base;
}

__host__ __device__ constexpr int c1_ds_base(int page) {
    return raw_page_base(page) + kRawRegionBytes;
}

__host__ __device__ __forceinline__ float ds_value(int generation,
                                                    int group,
                                                    int panel,
                                                    int owner) {
    return 1.0f + generation * 0.5f + group * 0.25f +
           panel * 0.0625f + owner * 0.015625f;
}

__device__ __forceinline__ ins::Vec4F16 make_rhs_one() {
    ins::Vec4F16 rhs{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rhs[i] = static_cast<_Float16>(1.0f);
    }
    return rhs;
}

template <int BarrierId>
__device__ __forceinline__ void wait_token(int& phase) {
    ins::abarrier_try_wait<true>(BarrierId, phase);
}

template <int BarrierId>
__device__ __forceinline__ void seq_token() {
    ins::abarrier_seq<false>(BarrierId);
}

template <int BarrierId>
__device__ __forceinline__ void arrive_token() {
    ins::abarrier_arrive_cnt<false>(BarrierId, 1);
}

template <int Page, int RegionOffset, int Generation>
__device__ __forceinline__ void load_raw_region(
    const __half* source, __half* lds, int producer_wave) {
#pragma unroll
    for (int row_block = 0; row_block < 2; ++row_block) {
        const int matrix = row_block * 4 + producer_wave;
        const int source_matrix = Generation * kRawRegionMatrices + matrix;
        ins::matrix_load_32x32_b16_bps_lds(
            lds,
            ins::prepare_matrix_src(source + source_matrix * kMatrixElements,
                                    32),
            raw_page_base(Page) + RegionOffset + matrix * kMatrixBytes,
            true);
    }
}

template <int Page, int RegionOffset>
__device__ __forceinline__ void read_raw_region(
    const __half* lds, Fragment (&fragment)[kRawRegionMatrices]) {
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + raw_page_base(Page) +
        RegionOffset);
    ins::ds_read_matrix_32x16_trans_imm4<0, 2048, 4096, 6144>(
        base, fragment[0].f16x8, fragment[1].f16x8,
        fragment[2].f16x8, fragment[3].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<8192, 10240, 12288, 14336>(
        base, fragment[4].f16x8, fragment[5].f16x8,
        fragment[6].f16x8, fragment[7].f16x8);
}

__device__ __forceinline__ Accumulator mmac_packet(
    const Fragment (&fragment)[kRawRegionMatrices],
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    Accumulator acc{};
    acc.f32 = zero.f32;
#pragma unroll
    for (int i = 0; i < kRawRegionMatrices; ++i) {
        acc.f32 = ins::mmac_f16_lit(fragment[i].f16x4[0], rhs, acc.f32);
    }
    return acc;
}

__device__ __forceinline__ Fragment make_ds_fragment(
    int generation, int group, int panel, int owner) {
    Fragment fragment{};
    const _Float16 value =
        static_cast<_Float16>(ds_value(generation, group, panel, owner));
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        fragment.scalar[word] = value;
    }
    return fragment;
}

template <int Group, int Page>
__device__ __forceinline__ int ds_offset(int panel, int owner) {
    static_assert(Group == 0 || Group == 1);
    const int base = Group == 0 ? c0_ds_base(Page) : c1_ds_base(Page);
    return base + panel * kDsPanelBytes + owner * kWriterStrideBytes;
}

template <int Group, int Page, int Generation>
__device__ __forceinline__ void publish_ds(__half* lds, int owner) {
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        const Fragment fragment =
            make_ds_fragment(Generation, Group, panel, owner);
        ins::ds_write_matrix_32x16_trans_f16(
            fragment.f16x8, lds, ds_offset<Group, Page>(panel, owner));
    }
    ins::wait_lgkm(0);
}

template <int Group, int Page, int Generation>
__device__ __forceinline__ void read_ds_normal(
    const __half* lds, float* normal_output, int owner,
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    Fragment fragment[kPanels];
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        ins::ds_read_matrix_32x16_normal(
            lds, ds_offset<Group, Page>(panel, owner),
            fragment[panel].f16x8);
    }
    ins::wait_lgkm(0);
    const int lane = threadIdx.x % kWaveSize;
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        Accumulator acc{};
        acc.f32 = ins::mmac_f16_lit(fragment[panel].f16x4[0], rhs,
                                    zero.f32);
        const int index =
            (((((Generation * 2 + Group) * kPanels + panel) * kOwners +
                owner) *
                   kWaveSize +
               lane) *
              4);
        *reinterpret_cast<ins::Vec4F32*>(normal_output + index) = acc.f32;
    }
}

template <int Group, int Page, int Generation>
__device__ __forceinline__ void read_ds_trans(
    const __half* lds, float* trans_output, int writer,
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    const int lane = threadIdx.x % kWaveSize;
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        Fragment fragment[kOwners];
        const auto* base = reinterpret_cast<const __half*>(
            reinterpret_cast<const char*>(lds) +
            ds_offset<Group, Page>(panel, 0));
        ins::ds_read_matrix_32x16_trans_imm4<0, 1024, 2048, 3072>(
            base, fragment[0].f16x8, fragment[1].f16x8,
            fragment[2].f16x8, fragment[3].f16x8);
        ins::wait_lgkm(0);
        Accumulator acc{};
        acc.f32 = zero.f32;
#pragma unroll
        for (int owner = 0; owner < kOwners; ++owner) {
            acc.f32 = ins::mmac_f16_lit(fragment[owner].f16x4[0], rhs,
                                        acc.f32);
        }
        const int index =
            (((((Generation * 2 + Group) * kPanels + panel) * kOwners +
                writer) *
                   kWaveSize +
               lane) *
              4);
        *reinterpret_cast<ins::Vec4F32*>(trans_output + index) = acc.f32;
    }
}

template <int Page, int Generation>
__device__ __forceinline__ void store_raw_mmac(
    float* raw_output, int group, int owner, int kind,
    const Accumulator& acc) {
    const int lane = threadIdx.x % kWaveSize;
    const int index =
        (((((Generation * 2 + group) * kOwners + owner) * 2 + kind) *
               kWaveSize +
           lane) *
          4);
    *reinterpret_cast<ins::Vec4F32*>(raw_output + index) = acc.f32;
}

template <int Page, int Generation>
__device__ __forceinline__ void producer_generation(
    const __half* q_source, const __half* dout_source, __half* lds,
    int producer_wave, int& raw_used_phase) {
    if constexpr (Generation >= 2) {
        if constexpr (Page == 0) {
            wait_token<Bar::kRawUsed0>(raw_used_phase);
        } else {
            wait_token<Bar::kRawUsed1>(raw_used_phase);
        }
    }
    if constexpr (Page == 0) {
        seq_token<Bar::kRawFilled0>();
    } else {
        seq_token<Bar::kRawFilled1>();
    }
    load_raw_region<Page, 0, Generation>(q_source, lds, producer_wave);
    load_raw_region<Page, kRawRegionBytes, Generation>(
        dout_source, lds, producer_wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    if constexpr (Page == 0) {
        arrive_token<Bar::kRawFilled0>();
    } else {
        arrive_token<Bar::kRawFilled1>();
    }
}

template <int Page, int Generation>
__device__ __forceinline__ void consumer0_generation(
    const __half* lds, __half* mutable_lds, float* raw_output,
    float* normal_output, int owner, int& raw_filled_phase,
    int& c0_done_phase, const ins::Vec4F16& rhs,
    const ins::F16x8& zero) {
    if constexpr (Page == 0) {
        wait_token<Bar::kRawFilled0>(raw_filled_phase);
    } else {
        wait_token<Bar::kRawFilled1>(raw_filled_phase);
    }
    if constexpr (Generation >= 2) {
        if constexpr (Page == 0) {
            wait_token<Bar::kC0Done0>(c0_done_phase);
        } else {
            wait_token<Bar::kC0Done1>(c0_done_phase);
        }
    }

    Fragment dout[kRawRegionMatrices];
    read_raw_region<Page, kRawRegionBytes>(lds, dout);
    ins::wait_lgkm(0);
    const Accumulator dv = mmac_packet(dout, rhs, zero);
    store_raw_mmac<Page, Generation>(raw_output, 0, owner, 1, dv);

    if constexpr (Page == 0) {
        seq_token<Bar::kDoutDead0>();
    } else {
        seq_token<Bar::kDoutDead1>();
    }
    if constexpr (Page == 0) {
        arrive_token<Bar::kDoutDead0>();
        seq_token<Bar::kC0Filled0>();
    } else {
        arrive_token<Bar::kDoutDead1>();
        seq_token<Bar::kC0Filled1>();
    }
    publish_ds<0, Page, Generation>(mutable_lds, owner);
    if constexpr (Page == 0) {
        arrive_token<Bar::kC0Filled0>();
    } else {
        arrive_token<Bar::kC0Filled1>();
    }

    Fragment q[kRawRegionMatrices];
    read_raw_region<Page, 0>(lds, q);
    ins::wait_lgkm(0);
    const Accumulator dk_q = mmac_packet(q, rhs, zero);
    store_raw_mmac<Page, Generation>(raw_output, 0, owner, 0, dk_q);
    read_ds_normal<0, Page, Generation>(
        lds, normal_output, owner, rhs, zero);
    if constexpr (Page == 0) {
        arrive_token<Bar::kC0Done0>();
        arrive_token<Bar::kRawUsed0>();
    } else {
        arrive_token<Bar::kC0Done1>();
        arrive_token<Bar::kRawUsed1>();
    }
}

template <int Page, int Generation>
__device__ __forceinline__ void consumer1_generation(
    const __half* lds, __half* mutable_lds, float* raw_output,
    float* normal_output, int owner, int& raw_filled_phase,
    int& dout_dead_phase, const ins::Vec4F16& rhs,
    const ins::F16x8& zero) {
    if constexpr (Page == 0) {
        wait_token<Bar::kRawFilled0>(raw_filled_phase);
    } else {
        wait_token<Bar::kRawFilled1>(raw_filled_phase);
    }
    Fragment dout[kRawRegionMatrices];
    read_raw_region<Page, kRawRegionBytes>(lds, dout);
    ins::wait_lgkm(0);
    const Accumulator dv = mmac_packet(dout, rhs, zero);
    store_raw_mmac<Page, Generation>(raw_output, 1, owner, 1, dv);

    if constexpr (Page == 0) {
        wait_token<Bar::kDoutDead0>(dout_dead_phase);
        seq_token<Bar::kC1Filled0>();
    } else {
        wait_token<Bar::kDoutDead1>(dout_dead_phase);
        seq_token<Bar::kC1Filled1>();
    }
    publish_ds<1, Page, Generation>(mutable_lds, owner);
    if constexpr (Page == 0) {
        arrive_token<Bar::kC1Filled0>();
    } else {
        arrive_token<Bar::kC1Filled1>();
    }

    Fragment q[kRawRegionMatrices];
    read_raw_region<Page, 0>(lds, q);
    ins::wait_lgkm(0);
    const Accumulator dk_q = mmac_packet(q, rhs, zero);
    store_raw_mmac<Page, Generation>(raw_output, 1, owner, 0, dk_q);
    read_ds_normal<1, Page, Generation>(
        lds, normal_output, owner, rhs, zero);
    if constexpr (Page == 0) {
        arrive_token<Bar::kRawUsed0>();
    } else {
        arrive_token<Bar::kRawUsed1>();
    }
}

template <int Page, int Generation>
__device__ __forceinline__ void writer_generation(
    const __half* lds, float* trans_output, int writer,
    int& c0_filled_phase, int& c1_filled_phase,
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    if constexpr (Page == 0) {
        wait_token<Bar::kC0Filled0>(c0_filled_phase);
    } else {
        wait_token<Bar::kC0Filled1>(c0_filled_phase);
    }
    read_ds_trans<0, Page, Generation>(
        lds, trans_output, writer, rhs, zero);
    if constexpr (Page == 0) {
        arrive_token<Bar::kC0Done0>();
        wait_token<Bar::kC1Filled0>(c1_filled_phase);
    } else {
        arrive_token<Bar::kC0Done1>();
        wait_token<Bar::kC1Filled1>(c1_filled_phase);
    }
    read_ds_trans<1, Page, Generation>(
        lds, trans_output, writer, rhs, zero);
    if constexpr (Page == 0) {
        arrive_token<Bar::kRawUsed0>();
    } else {
        arrive_token<Bar::kRawUsed1>();
    }
}

__device__ __forceinline__ void run_producer(
    const __half* q_source, const __half* dout_source, __half* lds,
    int producer_wave) {
    int sidecar_phase = 0;
    int raw_used0_phase = 0;
    int raw_used1_phase = 0;
    seq_token<Bar::kResidentFilled>();
    arrive_token<Bar::kResidentFilled>();
    wait_token<Bar::kVSidecarReady>(sidecar_phase);
    producer_generation<0, 0>(q_source, dout_source, lds, producer_wave,
                              raw_used0_phase);
    producer_generation<1, 1>(q_source, dout_source, lds, producer_wave,
                              raw_used1_phase);
    producer_generation<0, 2>(q_source, dout_source, lds, producer_wave,
                              raw_used0_phase);
    wait_token<Bar::kRawUsed0>(raw_used0_phase);
    wait_token<Bar::kRawUsed1>(raw_used1_phase);
}

__device__ __forceinline__ void run_consumer0(
    const __half* lds, __half* mutable_lds, float* raw_output,
    float* normal_output, int owner) {
    int resident_phase = 0;
    int raw_filled0_phase = 0;
    int raw_filled1_phase = 0;
    int c0_done0_phase = 0;
    int c0_done1_phase = 0;
    wait_token<Bar::kResidentFilled>(resident_phase);
    seq_token<Bar::kVSidecarReady>();
    arrive_token<Bar::kVSidecarReady>();
    const ins::Vec4F16 rhs = make_rhs_one();
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    consumer0_generation<0, 0>(
        lds, mutable_lds, raw_output, normal_output, owner,
        raw_filled0_phase, c0_done0_phase, rhs, zero);
    consumer0_generation<1, 1>(
        lds, mutable_lds, raw_output, normal_output, owner,
        raw_filled1_phase, c0_done1_phase, rhs, zero);
    consumer0_generation<0, 2>(
        lds, mutable_lds, raw_output, normal_output, owner,
        raw_filled0_phase, c0_done0_phase, rhs, zero);
}

__device__ __forceinline__ void run_consumer1(
    const __half* lds, __half* mutable_lds, float* raw_output,
    float* normal_output, int owner) {
    int resident_phase = 0;
    int raw_filled0_phase = 0;
    int raw_filled1_phase = 0;
    int dout_dead0_phase = 0;
    int dout_dead1_phase = 0;
    wait_token<Bar::kResidentFilled>(resident_phase);
    const ins::Vec4F16 rhs = make_rhs_one();
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    consumer1_generation<0, 0>(
        lds, mutable_lds, raw_output, normal_output, owner,
        raw_filled0_phase, dout_dead0_phase, rhs, zero);
    consumer1_generation<1, 1>(
        lds, mutable_lds, raw_output, normal_output, owner,
        raw_filled1_phase, dout_dead1_phase, rhs, zero);
    consumer1_generation<0, 2>(
        lds, mutable_lds, raw_output, normal_output, owner,
        raw_filled0_phase, dout_dead0_phase, rhs, zero);
}

__device__ __forceinline__ void run_writer(
    const __half* lds, float* trans_output, int writer) {
    int resident_phase = 0;
    int c0_filled0_phase = 0;
    int c0_filled1_phase = 0;
    int c1_filled0_phase = 0;
    int c1_filled1_phase = 0;
    wait_token<Bar::kResidentFilled>(resident_phase);
    const ins::Vec4F16 rhs = make_rhs_one();
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    writer_generation<0, 0>(lds, trans_output, writer,
                            c0_filled0_phase, c1_filled0_phase, rhs, zero);
    writer_generation<1, 1>(lds, trans_output, writer,
                            c0_filled1_phase, c1_filled1_phase, rhs, zero);
    writer_generation<0, 2>(lds, trans_output, writer,
                            c0_filled0_phase, c1_filled0_phase, rhs, zero);
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_c1_ds_on_dead_dout_probe_kernel(
    const __half* __restrict__ q_source,
    const __half* __restrict__ dout_source,
    float* __restrict__ raw_output,
    float* __restrict__ normal_output,
    float* __restrict__ trans_output) {
#if defined(SHAOBO_EXPLICIT_WDRA_INIT)
    __builtin_hcu_wdra_init(16, 204, 204, 88);
#endif
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) __half lds[kLdsBytes / sizeof(__half)];
    const int wave = static_cast<int>(__builtin_hcu_get_wave_id());
    if (wave == 0) {
        __builtin_hcu_s_abarrier_init(Bar::kResidentFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kVSidecarReady, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed0, 12);
        __builtin_hcu_s_abarrier_init(Bar::kRawFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kRawUsed1, 12);
        __builtin_hcu_s_abarrier_init(Bar::kC0Filled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kC1Filled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kC0Done0, 8);
        __builtin_hcu_s_abarrier_init(Bar::kC1Filled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kC0Filled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kC0Done1, 8);
        __builtin_hcu_s_abarrier_init(Bar::kDoutDead0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDoutDead1, 4);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < kProducerEnd) {
        __builtin_hcu_s_set_vgpr_size(16);
        run_producer(q_source, dout_source, lds, wave);
    } else if (wave < kConsumer0End) {
        __builtin_hcu_s_set_vgpr_size(204);
        run_consumer0(lds, lds, raw_output, normal_output,
                      wave - kProducerEnd);
    } else if (wave < kConsumer1End) {
        __builtin_hcu_s_set_vgpr_size(204);
        run_consumer1(lds, lds, raw_output, normal_output,
                      wave - kConsumer0End);
    } else {
        __builtin_hcu_s_set_vgpr_size(88);
        run_writer(lds, trans_output, wave - kConsumer1End);
    }

    __builtin_hcu_s_ebarrier_sync(0);
    if (wave == 0) {
#pragma unroll
        for (int barrier = 0; barrier < Bar::kCount; ++barrier) {
            __builtin_hcu_s_abarrier_inv(barrier);
        }
    }
#else
    (void)q_source;
    (void)dout_source;
    (void)raw_output;
    (void)normal_output;
    (void)trans_output;
#endif
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

template <typename T>
T* allocate_device(std::size_t count, const char* what) {
    T* pointer = nullptr;
    check_hip(hipMalloc(&pointer, count * sizeof(T)), what);
    return pointer;
}

}  // namespace

int main() {
    std::vector<__half> q_source(kSourceValues);
    std::vector<__half> dout_source(kSourceValues);
    for (int generation = 0; generation < kGenerations; ++generation) {
        const __half q_value = static_cast<__half>(1.0f + generation);
        const __half dout_value = static_cast<__half>(2.0f + generation);
        for (int i = 0; i < kRawRegionMatrices * kMatrixElements; ++i) {
            const int index = generation * kRawRegionMatrices *
                                  kMatrixElements +
                              i;
            q_source[index] = q_value;
            dout_source[index] = dout_value;
        }
    }

    std::vector<float> raw_output(kRawOutputValues);
    std::vector<float> normal_output(kNormalOutputValues);
    std::vector<float> trans_output(kTransOutputValues);
    __half* d_q = allocate_device<__half>(kSourceValues, "alloc q");
    __half* d_dout =
        allocate_device<__half>(kSourceValues, "alloc dout");
    float* d_raw =
        allocate_device<float>(kRawOutputValues, "alloc raw output");
    float* d_normal = allocate_device<float>(kNormalOutputValues,
                                              "alloc normal output");
    float* d_trans = allocate_device<float>(kTransOutputValues,
                                             "alloc trans output");
    check_hip(hipMemcpy(d_q, q_source.data(), q_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(d_dout, dout_source.data(),
                        dout_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy dout");
    check_hip(hipMemset(d_raw, 0, raw_output.size() * sizeof(float)),
              "clear raw output");
    check_hip(hipMemset(d_normal, 0,
                        normal_output.size() * sizeof(float)),
              "clear normal output");
    check_hip(hipMemset(d_trans, 0,
                        trans_output.size() * sizeof(float)),
              "clear trans output");

    hipLaunchKernelGGL(fused5_c1_ds_on_dead_dout_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, d_q, d_dout,
                       d_raw, d_normal, d_trans);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(raw_output.data(), d_raw,
                        raw_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy raw output");
    check_hip(hipMemcpy(normal_output.data(), d_normal,
                        normal_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy normal output");
    check_hip(hipMemcpy(trans_output.data(), d_trans,
                        trans_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy trans output");

    check_hip(hipFree(d_q), "free q");
    check_hip(hipFree(d_dout), "free dout");
    check_hip(hipFree(d_raw), "free raw output");
    check_hip(hipFree(d_normal), "free normal output");
    check_hip(hipFree(d_trans), "free trans output");

    int q_mismatches = 0;
    int dout_mismatches = 0;
    int normal_mismatches = 0;
    int trans_mismatches = 0;
    for (int generation = 0; generation < kGenerations; ++generation) {
        for (int group = 0; group < 2; ++group) {
            for (int owner = 0; owner < kOwners; ++owner) {
                for (int kind = 0; kind < 2; ++kind) {
                    const float expected =
                        128.0f * (kind == 0 ? 1.0f + generation
                                            : 2.0f + generation);
                    for (int lane = 0; lane < kWaveSize; ++lane) {
                        for (int word = 0; word < 4; ++word) {
                            const int index =
                                (((((generation * 2 + group) * kOwners +
                                    owner) *
                                       2 +
                                   kind) *
                                      kWaveSize +
                                  lane) *
                                     4 +
                                 word);
                            const bool mismatch =
                                std::fabs(raw_output[index] - expected) >
                                1.0e-4f;
                            if (kind == 0) {
                                q_mismatches += mismatch;
                            } else {
                                dout_mismatches += mismatch;
                            }
                        }
                    }
                }
            }
            for (int panel = 0; panel < kPanels; ++panel) {
                float trans_expected = 0.0f;
                for (int owner = 0; owner < kOwners; ++owner) {
                    trans_expected +=
                        16.0f * ds_value(generation, group, panel, owner);
                    const float normal_expected =
                        16.0f * ds_value(generation, group, panel, owner);
                    for (int lane = 0; lane < kWaveSize; ++lane) {
                        for (int word = 0; word < 4; ++word) {
                            const int index =
                                (((((generation * 2 + group) * kPanels +
                                    panel) *
                                       kOwners +
                                   owner) *
                                      kWaveSize +
                                  lane) *
                                     4 +
                                 word);
                            normal_mismatches +=
                                std::fabs(normal_output[index] -
                                          normal_expected) > 1.0e-4f;
                        }
                    }
                }
                for (int writer = 0; writer < kOwners; ++writer) {
                    for (int lane = 0; lane < kWaveSize; ++lane) {
                        for (int word = 0; word < 4; ++word) {
                            const int index =
                                (((((generation * 2 + group) * kPanels +
                                    panel) *
                                       kOwners +
                                   writer) *
                                      kWaveSize +
                                  lane) *
                                     4 +
                                 word);
                            trans_mismatches +=
                                std::fabs(trans_output[index] -
                                          trans_expected) > 1.0e-4f;
                        }
                    }
                }
            }
        }
    }

    const bool pass = q_mismatches == 0 && dout_mismatches == 0 &&
                      normal_mismatches == 0 && trans_mismatches == 0;
    std::printf(
        "fused5_c1_ds_dead_dout config waves=16 pages=2 generations=3 "
        "lds_bytes=131072 barriers=14 raw_used=12 roles=16/204/204/88\n");
    std::printf(
        "fused5_c1_ds_dead_dout q_mismatches=%d dout_mismatches=%d "
        "normal_mismatches=%d trans_mismatches=%d pass=%d\n",
        q_mismatches, dout_mismatches, normal_mismatches,
        trans_mismatches, pass ? 1 : 0);
    return pass ? 0 : 3;
}
