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
constexpr int kPanels = 8;
constexpr int kMatrixElements = 32 * 32;
constexpr int kMatrixBytes = kMatrixElements * sizeof(__half);
constexpr int kQRegionMatrices = 16;
constexpr int kQPageBytes = kQRegionMatrices * kMatrixBytes;
constexpr int kResidentMatrices = 32;
constexpr int kResidentBytes = kResidentMatrices * kMatrixBytes;
constexpr int kQ0Base = 0;
constexpr int kQ1Base = kQPageBytes;
constexpr int kDoutDsBase = 2 * kQPageBytes;
constexpr int kLdsBytes = kDoutDsBase + kResidentBytes;
constexpr int kWriterStrideBytes = 1024;
constexpr int kDsPanelBytes = kOwners * kWriterStrideBytes;
constexpr int kDsGroupBytes = kPanels * kDsPanelBytes;
constexpr int kRawKinds = 3;
constexpr int kRawOutputValues =
    kGenerations * 2 * kOwners * kRawKinds * kWaveSize * 4;
constexpr int kNormalOutputValues =
    kGenerations * 2 * kPanels * kOwners * kWaveSize * 4;
constexpr int kTransOutputValues = kNormalOutputValues;
constexpr int kQSourceValues =
    kGenerations * kQRegionMatrices * kMatrixElements;
constexpr int kDoutSourceValues = kQSourceValues;
constexpr int kResidentSourceValues = kResidentMatrices * kMatrixElements;

static_assert(kWaves * kWaveSize == 1024, "probe must use 16 waves");
static_assert(kQPageBytes == 32 * 1024, "each M128 Q page is 32KiB");
static_assert(kResidentBytes == 64 * 1024, "resident K/V is 64KiB");
static_assert(2 * kDsGroupBytes == kResidentBytes,
              "the M128 dS page must exactly alias dead resident/dO LDS");
static_assert(kLdsBytes == 128 * 1024, "probe must fit the Shaobo LDS limit");

struct Bar {
    static constexpr int kResidentFilled = 0;
    static constexpr int kResidentUsed = 1;
    static constexpr int kQFilled0 = 2;
    static constexpr int kQUsed0 = 3;
    static constexpr int kQFilled1 = 4;
    static constexpr int kQUsed1 = 5;
    static constexpr int kDoutFilled = 6;
    static constexpr int kDoutDead = 7;
    static constexpr int kDsFilled0 = 8;
    static constexpr int kDsFilled1 = 9;
    static constexpr int kEpochDone = 10;
    static constexpr int kCount = 11;
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

__host__ __device__ constexpr int q_page_base(int page) {
    return page == 0 ? kQ0Base : kQ1Base;
}

__host__ __device__ constexpr int ds_group_base(int group) {
    return kDoutDsBase + group * kDsGroupBytes;
}

__host__ __device__ __forceinline__ float ds_value(
    int generation, int group, int panel, int owner) {
    return 1.0f + generation * 0.5f + group * 0.25f +
           panel * 0.03125f + owner * 0.0078125f;
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

__device__ __forceinline__ ins::Vec4F16 make_rhs_one() {
    ins::Vec4F16 rhs{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rhs[i] = static_cast<_Float16>(1.0f);
    }
    return rhs;
}

template <int Matrices>
__device__ __forceinline__ Accumulator mmac_packet(
    const Fragment (&fragment)[Matrices], const ins::Vec4F16& rhs,
    const ins::F16x8& zero) {
    Accumulator acc{};
    acc.f32 = zero.f32;
#pragma unroll
    for (int i = 0; i < Matrices; ++i) {
        acc.f32 = ins::mmac_f16_lit(fragment[i].f16x4[0], rhs, acc.f32);
    }
    return acc;
}

template <int Generation, int Page>
__device__ __forceinline__ void load_q_page(
    const __half* source, __half* lds, int producer_wave) {
#pragma unroll
    for (int row_block = 0; row_block < 4; ++row_block) {
        const int matrix = row_block * 4 + producer_wave;
        ins::matrix_load_32x32_b16_bps_lds(
            lds,
            ins::prepare_matrix_src(
                source + (Generation * kQRegionMatrices + matrix) *
                             kMatrixElements,
                32),
            q_page_base(Page) + matrix * kMatrixBytes, true);
    }
}

template <int Generation>
__device__ __forceinline__ void load_dout(
    const __half* source, __half* lds, int producer_wave) {
#pragma unroll
    for (int row_block = 0; row_block < 4; ++row_block) {
        const int matrix = row_block * 4 + producer_wave;
        ins::matrix_load_32x32_b16_bps_lds(
            lds,
            ins::prepare_matrix_src(
                source + (Generation * kQRegionMatrices + matrix) *
                             kMatrixElements,
                32),
            kDoutDsBase + matrix * kMatrixBytes, true);
    }
}

__device__ __forceinline__ void load_resident(
    const __half* source, __half* lds, int producer_wave) {
#pragma unroll
    for (int row_block = 0; row_block < 8; ++row_block) {
        const int matrix = row_block * 4 + producer_wave;
        ins::matrix_load_32x32_b16_bps_lds(
            lds,
            ins::prepare_matrix_src(source + matrix * kMatrixElements, 32),
            kDoutDsBase + matrix * kMatrixBytes, true);
    }
}

template <int Page>
__device__ __forceinline__ void read_q(
    const __half* lds, Fragment (&fragment)[kQRegionMatrices]) {
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + q_page_base(Page));
    ins::ds_read_matrix_32x16_trans_imm4<0, 2048, 4096, 6144>(
        base, fragment[0].f16x8, fragment[1].f16x8,
        fragment[2].f16x8, fragment[3].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<8192, 10240, 12288, 14336>(
        base, fragment[4].f16x8, fragment[5].f16x8,
        fragment[6].f16x8, fragment[7].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<16384, 18432, 20480, 22528>(
        base, fragment[8].f16x8, fragment[9].f16x8,
        fragment[10].f16x8, fragment[11].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<24576, 26624, 28672, 30720>(
        base, fragment[12].f16x8, fragment[13].f16x8,
        fragment[14].f16x8, fragment[15].f16x8);
}

__device__ __forceinline__ void read_dout(
    const __half* lds, Fragment (&fragment)[kQRegionMatrices]) {
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + kDoutDsBase);
    ins::ds_read_matrix_32x16_trans_imm4<0, 2048, 4096, 6144>(
        base, fragment[0].f16x8, fragment[1].f16x8,
        fragment[2].f16x8, fragment[3].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<8192, 10240, 12288, 14336>(
        base, fragment[4].f16x8, fragment[5].f16x8,
        fragment[6].f16x8, fragment[7].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<16384, 18432, 20480, 22528>(
        base, fragment[8].f16x8, fragment[9].f16x8,
        fragment[10].f16x8, fragment[11].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<24576, 26624, 28672, 30720>(
        base, fragment[12].f16x8, fragment[13].f16x8,
        fragment[14].f16x8, fragment[15].f16x8);
}

template <int Role>
__device__ __forceinline__ Accumulator read_resident(
    const __half* lds, const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    static_assert(Role >= 0 && Role < 3);
    Fragment fragment[8];
    const auto* base = reinterpret_cast<const __half*>(
        reinterpret_cast<const char*>(lds) + kDoutDsBase +
        Role * 8 * kMatrixBytes);
    ins::ds_read_matrix_32x16_trans_imm4<0, 2048, 4096, 6144>(
        base, fragment[0].f16x8, fragment[1].f16x8,
        fragment[2].f16x8, fragment[3].f16x8);
    ins::ds_read_matrix_32x16_trans_imm4<8192, 10240, 12288, 14336>(
        base, fragment[4].f16x8, fragment[5].f16x8,
        fragment[6].f16x8, fragment[7].f16x8);
    ins::wait_lgkm(0);
    return mmac_packet(fragment, rhs, zero);
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

template <int Group>
__device__ __forceinline__ int ds_offset(int panel, int owner) {
    static_assert(Group == 0 || Group == 1);
    return ds_group_base(Group) + panel * kDsPanelBytes +
           owner * kWriterStrideBytes;
}

template <int Group, int Generation>
__device__ __forceinline__ void publish_ds(__half* lds, int owner) {
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        const Fragment fragment =
            make_ds_fragment(Generation, Group, panel, owner);
        ins::ds_write_matrix_32x16_trans_f16(
            fragment.f16x8, lds, ds_offset<Group>(panel, owner));
    }
    ins::wait_lgkm(0);
}

template <int Group, int Generation>
__device__ __forceinline__ void read_ds_normal(
    const __half* lds, float* output, int owner,
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    Fragment fragment[kPanels];
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        ins::ds_read_matrix_32x16_normal(
            lds, ds_offset<Group>(panel, owner), fragment[panel].f16x8);
    }
    ins::wait_lgkm(0);
    const int lane = threadIdx.x % kWaveSize;
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        const Accumulator acc =
            {ins::mmac_f16_lit(fragment[panel].f16x4[0], rhs, zero.f32)};
        const int index =
            (((((Generation * 2 + Group) * kPanels + panel) * kOwners +
                owner) *
                   kWaveSize +
               lane) *
              4);
        *reinterpret_cast<ins::Vec4F32*>(output + index) = acc.f32;
    }
}

template <int Group, int Generation>
__device__ __forceinline__ void read_ds_trans(
    const __half* lds, float* output, int writer,
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    const int lane = threadIdx.x % kWaveSize;
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        Fragment fragment[kOwners];
        const auto* base = reinterpret_cast<const __half*>(
            reinterpret_cast<const char*>(lds) +
            ds_offset<Group>(panel, 0));
        ins::ds_read_matrix_32x16_trans_imm4<0, 1024, 2048, 3072>(
            base, fragment[0].f16x8, fragment[1].f16x8,
            fragment[2].f16x8, fragment[3].f16x8);
        ins::wait_lgkm(0);
        Accumulator acc{};
        acc.f32 = zero.f32;
#pragma unroll
        for (int owner = 0; owner < kOwners; ++owner) {
            acc.f32 = ins::mmac_f16_lit(
                fragment[owner].f16x4[0], rhs, acc.f32);
        }
        const int index =
            (((((Generation * 2 + Group) * kPanels + panel) * kOwners +
                writer) *
                   kWaveSize +
               lane) *
              4);
        *reinterpret_cast<ins::Vec4F32*>(output + index) = acc.f32;
    }
}

template <int Generation>
__device__ __forceinline__ void store_raw(
    float* output, int group, int owner, int kind,
    const Accumulator& acc) {
    const int lane = threadIdx.x % kWaveSize;
    const int index =
        (((((Generation * 2 + group) * kOwners + owner) * kRawKinds +
            kind) *
               kWaveSize +
           lane) *
          4);
    *reinterpret_cast<ins::Vec4F32*>(output + index) = acc.f32;
}

template <int Page, int Generation>
__device__ __forceinline__ void consumer_generation(
    const __half* lds, __half* mutable_lds, float* raw_output,
    float* normal_output, int group, int owner, int& q_filled_phase,
    int& dout_filled_phase, int& dout_dead_phase,
    int& ds_filled_phase, const Accumulator& resident,
    const ins::Vec4F16& rhs, const ins::F16x8& zero) {
    if constexpr (Page == 0) {
        wait_token<Bar::kQFilled0>(q_filled_phase);
    } else {
        wait_token<Bar::kQFilled1>(q_filled_phase);
    }
    wait_token<Bar::kDoutFilled>(dout_filled_phase);

    Accumulator q_before{};
    Accumulator dout_acc{};
    {
        Fragment q[kQRegionMatrices];
        Fragment dout[kQRegionMatrices];
        read_q<Page>(lds, q);
        read_dout(lds, dout);
        ins::wait_lgkm(0);
        q_before = mmac_packet(q, rhs, zero);
        dout_acc = mmac_packet(dout, rhs, zero);
    }
    store_raw<Generation>(raw_output, group, owner, 0, q_before);
    store_raw<Generation>(raw_output, group, owner, 1, dout_acc);

    arrive_token<Bar::kDoutDead>();
    wait_token<Bar::kDoutDead>(dout_dead_phase);

    if (group == 0) {
        seq_token<Bar::kDsFilled0>();
        publish_ds<0, Generation>(mutable_lds, owner);
        arrive_token<Bar::kDsFilled0>();
        wait_token<Bar::kDsFilled0>(ds_filled_phase);
        read_ds_normal<0, Generation>(lds, normal_output, owner, rhs, zero);
    } else {
        seq_token<Bar::kDsFilled1>();
        publish_ds<1, Generation>(mutable_lds, owner);
        arrive_token<Bar::kDsFilled1>();
        wait_token<Bar::kDsFilled1>(ds_filled_phase);
        read_ds_normal<1, Generation>(lds, normal_output, owner, rhs, zero);
    }

    Accumulator q_after{};
    {
        Fragment q[kQRegionMatrices];
        read_q<Page>(lds, q);
        ins::wait_lgkm(0);
        q_after = mmac_packet(q, rhs, zero);
    }
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        q_after.scalar[word] += resident.scalar[word];
    }
    store_raw<Generation>(raw_output, group, owner, 2, q_after);

    if constexpr (Page == 0) {
        arrive_token<Bar::kQUsed0>();
    } else {
        arrive_token<Bar::kQUsed1>();
    }
    arrive_token<Bar::kEpochDone>();
}

__device__ __forceinline__ void run_producer(
    const __half* q_source, const __half* dout_source,
    const __half* resident_source, __half* lds, int wave) {
    int resident_used_phase = 0;
    int q_used0_phase = 0;
    int q_used1_phase = 0;
    int epoch_done_phase = 0;

    seq_token<Bar::kResidentFilled>();
    seq_token<Bar::kQFilled0>();
    load_resident(resident_source, lds, wave);
    load_q_page<0, 0>(q_source, lds, wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    arrive_token<Bar::kResidentFilled>();
    arrive_token<Bar::kQFilled0>();

    seq_token<Bar::kQFilled1>();
    load_q_page<1, 1>(q_source, lds, wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    arrive_token<Bar::kQFilled1>();

    wait_token<Bar::kResidentUsed>(resident_used_phase);
    seq_token<Bar::kDoutFilled>();
    load_dout<0>(dout_source, lds, wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    arrive_token<Bar::kDoutFilled>();

    wait_token<Bar::kQUsed0>(q_used0_phase);
    seq_token<Bar::kQFilled0>();
    load_q_page<2, 0>(q_source, lds, wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    arrive_token<Bar::kQFilled0>();

    wait_token<Bar::kEpochDone>(epoch_done_phase);
    seq_token<Bar::kDoutFilled>();
    load_dout<1>(dout_source, lds, wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    arrive_token<Bar::kDoutFilled>();

    wait_token<Bar::kQUsed1>(q_used1_phase);
    wait_token<Bar::kEpochDone>(epoch_done_phase);
    seq_token<Bar::kDoutFilled>();
    load_dout<2>(dout_source, lds, wave);
    ins::maybe_wait_bps_vbcnt_before_arrive();
    arrive_token<Bar::kDoutFilled>();

    wait_token<Bar::kQUsed0>(q_used0_phase);
    wait_token<Bar::kEpochDone>(epoch_done_phase);
}

template <int Group>
__device__ __forceinline__ void run_consumer(
    const __half* lds, __half* mutable_lds, float* raw_output,
    float* normal_output, int owner) {
    int resident_phase = 0;
    int q_filled0_phase = 0;
    int q_filled1_phase = 0;
    int dout_filled_phase = 0;
    int dout_dead_phase = 0;
    int ds_filled_phase = 0;
    const ins::Vec4F16 rhs = make_rhs_one();
    ins::F16x8 zero;
    ins::zero_f16x8(zero);

    wait_token<Bar::kResidentFilled>(resident_phase);
    const Accumulator resident = read_resident<Group>(lds, rhs, zero);
    arrive_token<Bar::kResidentUsed>();

    consumer_generation<0, 0>(
        lds, mutable_lds, raw_output, normal_output, Group, owner,
        q_filled0_phase, dout_filled_phase, dout_dead_phase,
        ds_filled_phase, resident, rhs, zero);
    consumer_generation<1, 1>(
        lds, mutable_lds, raw_output, normal_output, Group, owner,
        q_filled1_phase, dout_filled_phase, dout_dead_phase,
        ds_filled_phase, resident, rhs, zero);
    consumer_generation<0, 2>(
        lds, mutable_lds, raw_output, normal_output, Group, owner,
        q_filled0_phase, dout_filled_phase, dout_dead_phase,
        ds_filled_phase, resident, rhs, zero);
}

__device__ __forceinline__ void run_writer(
    const __half* lds, float* trans_output, int writer) {
    int resident_phase = 0;
    int filled0_phase = 0;
    int filled1_phase = 0;
    const ins::Vec4F16 rhs = make_rhs_one();
    ins::F16x8 zero;
    ins::zero_f16x8(zero);
    wait_token<Bar::kResidentFilled>(resident_phase);
    (void)read_resident<2>(lds, rhs, zero);
    arrive_token<Bar::kResidentUsed>();

    wait_token<Bar::kDsFilled1>(filled1_phase);
    read_ds_trans<1, 0>(lds, trans_output, writer, rhs, zero);
    wait_token<Bar::kDsFilled0>(filled0_phase);
    read_ds_trans<0, 0>(lds, trans_output, writer, rhs, zero);
    arrive_token<Bar::kEpochDone>();

    wait_token<Bar::kDsFilled1>(filled1_phase);
    read_ds_trans<1, 1>(lds, trans_output, writer, rhs, zero);
    wait_token<Bar::kDsFilled0>(filled0_phase);
    read_ds_trans<0, 1>(lds, trans_output, writer, rhs, zero);
    arrive_token<Bar::kEpochDone>();

    wait_token<Bar::kDsFilled1>(filled1_phase);
    read_ds_trans<1, 2>(lds, trans_output, writer, rhs, zero);
    wait_token<Bar::kDsFilled0>(filled0_phase);
    read_ds_trans<0, 2>(lds, trans_output, writer, rhs, zero);
    arrive_token<Bar::kEpochDone>();
}

extern "C" __global__ void __launch_bounds__(kWaves * kWaveSize, 1)
    __attribute__((hcu_wdra_waves_per_tg(kWaves)))
fused5_m128_qdouble_dout_single_probe_kernel(
    const __half* __restrict__ q_source,
    const __half* __restrict__ dout_source,
    const __half* __restrict__ resident_source,
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
        __builtin_hcu_s_abarrier_init(Bar::kResidentUsed, 12);
        __builtin_hcu_s_abarrier_init(Bar::kQFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kQUsed0, 8);
        __builtin_hcu_s_abarrier_init(Bar::kQFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kQUsed1, 8);
        __builtin_hcu_s_abarrier_init(Bar::kDoutFilled, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDoutDead, 8);
        __builtin_hcu_s_abarrier_init(Bar::kDsFilled0, 4);
        __builtin_hcu_s_abarrier_init(Bar::kDsFilled1, 4);
        __builtin_hcu_s_abarrier_init(Bar::kEpochDone, 12);
    }
    __builtin_hcu_s_ebarrier_sync(0);

    if (wave < kProducerEnd) {
        __builtin_hcu_s_set_vgpr_size(16);
        run_producer(q_source, dout_source, resident_source, lds, wave);
    } else if (wave < kConsumer0End) {
        __builtin_hcu_s_set_vgpr_size(204);
        run_consumer<0>(lds, lds, raw_output, normal_output,
                        wave - kProducerEnd);
    } else if (wave < kConsumer1End) {
        __builtin_hcu_s_set_vgpr_size(204);
        run_consumer<1>(lds, lds, raw_output, normal_output,
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
    (void)resident_source;
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
    std::vector<__half> q_source(kQSourceValues);
    std::vector<__half> dout_source(kDoutSourceValues);
    std::vector<__half> resident_source(kResidentSourceValues,
                                        static_cast<__half>(0.5f));
    for (int generation = 0; generation < kGenerations; ++generation) {
        const __half q_value = static_cast<__half>(1.0f + generation);
        const __half dout_value = static_cast<__half>(2.0f + generation);
        for (int i = 0; i < kQRegionMatrices * kMatrixElements; ++i) {
            const int index =
                generation * kQRegionMatrices * kMatrixElements + i;
            q_source[index] = q_value;
            dout_source[index] = dout_value;
        }
    }

    std::vector<float> raw_output(kRawOutputValues);
    std::vector<float> normal_output(kNormalOutputValues);
    std::vector<float> trans_output(kTransOutputValues);
    __half* d_q = allocate_device<__half>(kQSourceValues, "alloc q");
    __half* d_dout =
        allocate_device<__half>(kDoutSourceValues, "alloc dout");
    __half* d_resident = allocate_device<__half>(
        kResidentSourceValues, "alloc resident");
    float* d_raw =
        allocate_device<float>(kRawOutputValues, "alloc raw output");
    float* d_normal = allocate_device<float>(
        kNormalOutputValues, "alloc normal output");
    float* d_trans =
        allocate_device<float>(kTransOutputValues, "alloc trans output");
    check_hip(hipMemcpy(d_q, q_source.data(), q_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy q");
    check_hip(hipMemcpy(d_dout, dout_source.data(),
                        dout_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy dout");
    check_hip(hipMemcpy(d_resident, resident_source.data(),
                        resident_source.size() * sizeof(__half),
                        hipMemcpyHostToDevice),
              "copy resident");
    check_hip(hipMemset(d_raw, 0, raw_output.size() * sizeof(float)),
              "clear raw");
    check_hip(hipMemset(d_normal, 0,
                        normal_output.size() * sizeof(float)),
              "clear normal");
    check_hip(hipMemset(d_trans, 0,
                        trans_output.size() * sizeof(float)),
              "clear trans");

    hipLaunchKernelGGL(fused5_m128_qdouble_dout_single_probe_kernel, dim3(1),
                       dim3(kWaves * kWaveSize), 0, 0, d_q, d_dout,
                       d_resident, d_raw, d_normal, d_trans);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(raw_output.data(), d_raw,
                        raw_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy raw");
    check_hip(hipMemcpy(normal_output.data(), d_normal,
                        normal_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy normal");
    check_hip(hipMemcpy(trans_output.data(), d_trans,
                        trans_output.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "copy trans");

    check_hip(hipFree(d_q), "free q");
    check_hip(hipFree(d_dout), "free dout");
    check_hip(hipFree(d_resident), "free resident");
    check_hip(hipFree(d_raw), "free raw");
    check_hip(hipFree(d_normal), "free normal");
    check_hip(hipFree(d_trans), "free trans");

    int q_before_mismatches = 0;
    int dout_mismatches = 0;
    int q_after_mismatches = 0;
    int normal_mismatches = 0;
    int trans_mismatches = 0;
    for (int generation = 0; generation < kGenerations; ++generation) {
        for (int group = 0; group < 2; ++group) {
            for (int owner = 0; owner < kOwners; ++owner) {
                for (int kind = 0; kind < kRawKinds; ++kind) {
                    float expected = 0.0f;
                    if (kind == 0) {
                        expected = 256.0f * (1.0f + generation);
                    } else if (kind == 1) {
                        expected = 256.0f * (2.0f + generation);
                    } else {
                        expected = 256.0f * (1.0f + generation) + 64.0f;
                    }
                    for (int lane = 0; lane < kWaveSize; ++lane) {
                        for (int word = 0; word < 4; ++word) {
                            const int index =
                                (((((generation * 2 + group) * kOwners +
                                    owner) *
                                       kRawKinds +
                                   kind) *
                                      kWaveSize +
                                  lane) *
                                     4 +
                                 word);
                            const bool mismatch =
                                std::fabs(raw_output[index] - expected) >
                                1.0e-4f;
                            if (kind == 0) {
                                q_before_mismatches += mismatch;
                            } else if (kind == 1) {
                                dout_mismatches += mismatch;
                            } else {
                                q_after_mismatches += mismatch;
                            }
                        }
                    }
                }
            }
            for (int panel = 0; panel < kPanels; ++panel) {
                float trans_expected = 0.0f;
                for (int owner = 0; owner < kOwners; ++owner) {
                    const float normal_expected =
                        16.0f * ds_value(generation, group, panel, owner);
                    trans_expected += normal_expected;
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

    const bool pass = q_before_mismatches == 0 && dout_mismatches == 0 &&
                      q_after_mismatches == 0 && normal_mismatches == 0 &&
                      trans_mismatches == 0;
    std::printf(
        "fused5_m128_qdouble_dout_single config waves=16 generations=3 "
        "lds_bytes=131072 barriers=11 q_pages=2 dout_pages=1 "
        "resident_used=12 q_used=8 dout_dead=8 epoch_done=12 "
        "roles=16/204/204/88\n");
    std::printf(
        "fused5_m128_qdouble_dout_single q_before_mismatches=%d "
        "dout_mismatches=%d q_after_mismatches=%d normal_mismatches=%d "
        "trans_mismatches=%d pass=%d\n",
        q_before_mismatches, dout_mismatches, q_after_mismatches,
        normal_mismatches, trans_mismatches, pass ? 1 : 0);
    return pass ? 0 : 3;
}
