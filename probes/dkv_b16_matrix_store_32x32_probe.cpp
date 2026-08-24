#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::dkv::instr;

namespace {

constexpr int kWaveSize = 64;
constexpr int kRows = 32;
constexpr int kCols = 32;
constexpr int kElems = kRows * kCols;
constexpr int kWriterCount = 4;
constexpr int kStoreCount = 4;
constexpr int kCandidateCount = kWriterCount * kStoreCount;
constexpr int kDispatchCount = kStoreCount + kCandidateCount;
constexpr int kLdsHalfs = 4096;
constexpr uint16_t kPatternBase = 0x3000;
constexpr uint16_t kPoison = 0xfefe;

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

__host__ __device__ _Float16 half_from_bits(uint16_t bits) {
    HalfBits value{};
    value.bits = bits;
    return value.value;
}

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(2);
    }
}

__device__ __forceinline__ void write_fragment(
    int writer, ins::Vec8F16 fragment, _Float16* page, int byte_offset) {
    _Float16* const ptr = reinterpret_cast<_Float16*>(
        reinterpret_cast<char*>(page) + byte_offset);
    switch (writer) {
        case 0:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, ptr, 0, 2, 1, 0, 0);
            break;
        case 1:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, ptr, 0, 2, 1, 1, 0);
            break;
        case 2:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, ptr, 0, 2, 1, 0, 1);
            break;
        default:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, ptr, 0, 2, 1, 1, 1);
            break;
    }
}

__device__ __forceinline__ void matrix_store_32x32(
    int store, _Float16* page, _Float16* output) {
    const ins::Vec4U32 dst = ins::prepare_matrix_src(
        reinterpret_cast<const __half*>(output), kCols);
    switch (store) {
        case 0:
            __builtin_hcu_matrix_store_32x32_b16(
                dst, reinterpret_cast<short*>(page), 0,
                false, false, false, false);
            break;
        case 1:
            __builtin_hcu_matrix_store_32x32_b16(
                dst, reinterpret_cast<short*>(page), 0,
                true, false, false, false);
            break;
        case 2:
            __builtin_hcu_matrix_store_32x32_b16(
                dst, reinterpret_cast<short*>(page), 0,
                false, true, false, false);
            break;
        default:
            __builtin_hcu_matrix_store_32x32_b16(
                dst, reinterpret_cast<short*>(page), 0,
                true, true, false, false);
            break;
    }
    ins::wait_vmem_lgkm();
}

__global__ void __launch_bounds__(kWaveSize, 1)
matrix_store_32x32_transport_kernel(
    const _Float16* input,
    const _Float16* register_input,
    _Float16* output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(2048) _Float16 page[kLdsHalfs];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int path = static_cast<int>(blockIdx.x);

    for (int index = lane; index < kLdsHalfs; index += kWaveSize) {
        page[index] = half_from_bits(kPoison);
    }
    ins::wait_lgkm(0);

    _Float16* const path_output = output + path * kElems;
    if (path < kStoreCount) {
        ins::matrix_load_32x32_b16_bps_lds(
            reinterpret_cast<__half*>(page),
            ins::prepare_matrix_src(
                reinterpret_cast<const __half*>(input), kCols),
            0, false);
        ins::wait_vbcnt0();
        ins::wait_lgkm(0);
        matrix_store_32x32(path, page, path_output);
        return;
    }

    const int candidate = path - kStoreCount;
    const int writer = candidate / kStoreCount;
    const int store = candidate % kStoreCount;
    const _Float16* const source = register_input + candidate * kElems;
    ins::Vec8F16 first{};
    ins::Vec8F16 second{};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        first[word] = source[lane * 8 + word];
        second[word] = source[512 + lane * 8 + word];
    }
    write_fragment(writer, first, page, 0);
    write_fragment(writer, second, page, 1024);
    ins::wait_lgkm(0);
    matrix_store_32x32(store, page, path_output);
#else
    (void)input;
    (void)register_input;
    (void)output;
#endif
}

}  // namespace

namespace {

void run_probe(
    _Float16* device_input,
    _Float16* device_register_input,
    _Float16* device_output,
    const std::vector<_Float16>& register_input,
    std::vector<uint16_t>& output) {
    const size_t register_bytes =
        register_input.size() * sizeof(_Float16);
    const size_t output_bytes = output.size() * sizeof(uint16_t);
    std::vector<uint16_t> poison(output.size(), kPoison);
    check_hip(hipMemcpy(device_register_input, register_input.data(),
                        register_bytes, hipMemcpyHostToDevice),
              "copy register input");
    check_hip(hipMemcpy(device_output, poison.data(), output_bytes,
                        hipMemcpyHostToDevice),
              "poison output");
    hipLaunchKernelGGL(matrix_store_32x32_transport_kernel,
                       dim3(kDispatchCount), dim3(kWaveSize), 0, 0,
                       device_input, device_register_input, device_output);
    check_hip(hipGetLastError(), "kernel launch");
    check_hip(hipDeviceSynchronize(), "kernel sync");
    check_hip(hipMemcpy(output.data(), device_output, output_bytes,
                        hipMemcpyDeviceToHost),
              "copy output");
}

int direct_mismatches(const std::vector<uint16_t>& output) {
    int mismatches = 0;
    for (int index = 0; index < kElems; ++index) {
        mismatches += output[index] != kPatternBase + index;
    }
    return mismatches;
}

}  // namespace

int main() {
    const char* writer_names[kWriterCount] = {
        "normal_alt0", "normal_alt1", "trans_alt0", "trans_alt1"};
    const char* store_names[kStoreCount] = {
        "t0r0", "t1r0", "t0r1", "t1r1"};
    std::vector<_Float16> input(kElems);
    std::vector<_Float16> calibration_input(kCandidateCount * kElems);
    std::vector<_Float16> replay_input(
        kCandidateCount * kElems, half_from_bits(kPoison));
    for (int index = 0; index < kElems; ++index) {
        input[index] = half_from_bits(
            static_cast<uint16_t>(kPatternBase + index));
        for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
            calibration_input[candidate * kElems + index] = input[index];
        }
    }

    _Float16* device_input = nullptr;
    _Float16* device_register_input = nullptr;
    _Float16* device_output = nullptr;
    const size_t input_bytes = kElems * sizeof(_Float16);
    const size_t register_bytes =
        calibration_input.size() * sizeof(_Float16);
    const size_t output_bytes =
        kDispatchCount * kElems * sizeof(uint16_t);
    check_hip(hipMalloc(&device_input, input_bytes), "hipMalloc input");
    check_hip(hipMalloc(&device_register_input, register_bytes),
              "hipMalloc register input");
    check_hip(hipMalloc(&device_output, output_bytes), "hipMalloc output");
    check_hip(hipMemcpy(device_input, input.data(), input_bytes,
                        hipMemcpyHostToDevice),
              "copy input");

    std::vector<uint16_t> calibration_output(kDispatchCount * kElems);
    run_probe(device_input, device_register_input, device_output,
              calibration_input, calibration_output);
    int direct_exact = 0;
    for (int store = 0; store < kStoreCount; ++store) {
        std::vector<uint16_t> direct_output(
            calibration_output.begin() + store * kElems,
            calibration_output.begin() + (store + 1) * kElems);
        const int mismatches = direct_mismatches(direct_output);
        direct_exact += mismatches == 0;
        std::printf(
            "matrix_store_32x32_direct store=%s mismatches=%d exact=%d\n",
            store_names[store], mismatches, mismatches == 0 ? 1 : 0);
    }

    std::FILE* map = std::fopen("matrix_store_32x32_slot_map.csv", "w");
    if (map == nullptr) {
        std::fprintf(stderr, "cannot create slot map\n");
        return 2;
    }
    std::fprintf(
        map,
        "writer,store,dst_row,dst_col,source_half,"
        "source_lane,source_word\n");

    std::vector<int> complete(kCandidateCount, 0);
    int slot_identity_exact = 0;
    int fwd_pack_exact = 0;
    int complete_candidates = 0;
    for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
        const uint16_t* result =
            calibration_output.data() +
            (kStoreCount + candidate) * kElems;
        std::vector<int> counts(kElems, 0);
        int slot_identity_mismatches = 0;
        int fwd_pack_mismatches = 0;
        int poison_count = 0;
        int unexpected = 0;
        int duplicate = 0;
        for (int dst = 0; dst < kElems; ++dst) {
            const uint16_t value = result[dst];
            slot_identity_mismatches += value != kPatternBase + dst;
            poison_count += value == kPoison;
            if (value >= kPatternBase && value < kPatternBase + kElems) {
                const int source = value - kPatternBase;
                duplicate += ++counts[source] > 1;
                const int source_half = source / 512;
                const int source_lane = (source % 512) / 8;
                const int source_word = source % 8;
                const int fwd_row =
                    source_half * 16 + (source_lane & 15);
                const int fwd_col =
                    (source_lane >> 4) * 8 + source_word;
                fwd_pack_mismatches +=
                    fwd_row * kCols + fwd_col != dst;
                replay_input[candidate * kElems + source] =
                    half_from_bits(static_cast<uint16_t>(
                        kPatternBase + dst));
                std::fprintf(
                    map, "%s,%s,%d,%d,%d,%d,%d\n",
                    writer_names[candidate / kStoreCount],
                    store_names[candidate % kStoreCount], dst / kCols,
                    dst % kCols, source_half, source_lane, source_word);
            } else if (value != kPoison) {
                ++unexpected;
            }
        }
        int missing = 0;
        for (int count : counts) missing += count == 0;
        complete[candidate] =
            poison_count == 0 && unexpected == 0 && duplicate == 0 &&
            missing == 0;
        slot_identity_exact += slot_identity_mismatches == 0;
        fwd_pack_exact += fwd_pack_mismatches == 0;
        complete_candidates += complete[candidate];
        std::printf(
            "matrix_store_32x32 writer=%s store=%s "
            "slot_identity_mismatches=%d fwd_pack_mismatches=%d "
            "poison=%d unexpected=%d duplicate=%d missing=%d "
            "complete=%d fwd_pack_exact=%d\n",
            writer_names[candidate / kStoreCount],
            store_names[candidate % kStoreCount],
            slot_identity_mismatches, fwd_pack_mismatches, poison_count,
            unexpected, duplicate, missing, complete[candidate],
            fwd_pack_mismatches == 0 ? 1 : 0);
    }
    std::fclose(map);

    std::vector<uint16_t> replay_output(kDispatchCount * kElems);
    run_probe(device_input, device_register_input, device_output,
              replay_input, replay_output);
    int replay_exact = 0;
    for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
        if (!complete[candidate]) continue;
        const uint16_t* result =
            replay_output.data() +
            (kStoreCount + candidate) * kElems;
        int mismatches = 0;
        for (int index = 0; index < kElems; ++index) {
            mismatches += result[index] != kPatternBase + index;
        }
        replay_exact += mismatches == 0;
        std::printf(
            "matrix_store_32x32_replay writer=%s store=%s "
            "mismatches=%d exact=%d\n",
            writer_names[candidate / kStoreCount],
            store_names[candidate % kStoreCount], mismatches,
            mismatches == 0 ? 1 : 0);
    }

    check_hip(hipFree(device_output), "hipFree output");
    check_hip(hipFree(device_register_input), "hipFree register input");
    check_hip(hipFree(device_input), "hipFree input");

    const bool pass =
        direct_exact > 0 && complete_candidates > 0 &&
        replay_exact == complete_candidates;
    std::printf(
        "matrix_store_32x32_transport_status=%s direct_exact=%d "
        "slot_identity_exact=%d fwd_pack_exact=%d complete=%d "
        "replay_exact=%d\n",
        pass ? "PASS" : "FAIL", direct_exact, slot_identity_exact,
        fwd_pack_exact, complete_candidates, replay_exact);
    return pass ? 0 : 1;
}
