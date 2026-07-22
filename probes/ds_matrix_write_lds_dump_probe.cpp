#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

using Vec8F16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using Vec4U32 = ins::Vec4U32;

constexpr int kWaveSize = 64;
constexpr int kFragmentWordsPerLane = 8;
constexpr int kPageWordsPerLane = 16;
constexpr int kMatrixElems = kWaveSize * kFragmentWordsPerLane;
constexpr int kPageElems = kWaveSize * kPageWordsPerLane;
constexpr int kWriterCount = 4;
constexpr uint16_t kPatternBase = 0x3000;
constexpr uint16_t kPoison = 0xfefe;

union HalfBits {
    _Float16 value;
    uint16_t bits;
};

__device__ __forceinline__ _Float16 half_from_bits(uint16_t bits) {
    HalfBits value{};
    value.bits = bits;
    return value.value;
}

__device__ __forceinline__ void poison_page(_Float16* page, int lane) {
    Vec4U32 poison{};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        poison[word] = 0xfefefefeU;
    }
    const uint32_t addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(page + lane * kPageWordsPerLane));
    asm volatile("ds_write_b128 %0, %1 offset:0\n\t"
                 "ds_write_b128 %0, %1 offset:16"
                 :
                 : "v"(addr), "v"(poison)
                 : "memory");
}

__device__ __forceinline__ void dump_page(
    _Float16* page, uint16_t* output, int lane) {
    const uint32_t addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(page + lane * kPageWordsPerLane));
    Vec4U32 words0{};
    Vec4U32 words1{};
    asm volatile("ds_read_b128 %0, %2 offset:0\n\t"
                 "ds_read_b128 %1, %2 offset:16"
                 : "=v"(words0), "=v"(words1)
                 : "v"(addr)
                 : "memory");
    ins::wait_lgkm(0);
    *reinterpret_cast<Vec4U32*>(
        output + lane * kPageWordsPerLane) = words0;
    *reinterpret_cast<Vec4U32*>(
        output + lane * kPageWordsPerLane + kFragmentWordsPerLane) = words1;
}

__device__ __forceinline__ void write_fragment(
    int writer, const Vec8F16& fragment, _Float16* page) {
    switch (writer) {
        case 0:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, page, 0, 2, 1, 0, 0);
            break;
        case 1:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, page, 0, 2, 1, 1, 0);
            break;
        case 2:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, page, 0, 2, 1, 0, 1);
            break;
        default:
            __builtin_hcu_ds_write_matrix_format_f16(
                fragment, page, 0, 2, 1, 1, 1);
            break;
    }
}

// One block tests one writer mode. Ordinary DS reads dump the complete 2 KiB
// physical page, and ordinary global stores return it to the CPU oracle.
__global__ void __launch_bounds__(kWaveSize, 1)
    ds_matrix_writer_dump_probe_kernel(uint16_t* __restrict__ output) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __align__(256) _Float16 page[kPageElems];
    const int lane = static_cast<int>(threadIdx.x % kWaveSize);
    const int writer = static_cast<int>(blockIdx.x);

    poison_page(page, lane);
    ins::wait_lgkm(0);

    Vec8F16 fragment{};
#pragma unroll
    for (int word = 0; word < kFragmentWordsPerLane; ++word) {
        fragment[word] = half_from_bits(static_cast<uint16_t>(
            kPatternBase + lane * kFragmentWordsPerLane + word));
    }
    write_fragment(writer, fragment, page);
    ins::wait_lgkm(0);
    dump_page(page, output + writer * kPageElems, lane);
#else
    (void)output;
#endif
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     hipGetErrorString(error));
        std::exit(1);
    }
}

const char* kWriterNames[kWriterCount] = {
    "normal_alt0", "normal_alt1", "trans_alt0", "trans_alt1"};

}  // namespace

int main() {
    const size_t total_words =
        static_cast<size_t>(kWriterCount) * kPageElems;
    uint16_t* device_output = nullptr;
    std::vector<uint16_t> output(total_words);

    check_hip(hipMalloc(&device_output, total_words * sizeof(uint16_t)),
              "hipMalloc output");
    hipLaunchKernelGGL(ds_matrix_writer_dump_probe_kernel,
                       dim3(kWriterCount), dim3(kWaveSize), 0, 0,
                       device_output);
    check_hip(hipGetLastError(), "launch writer dump");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(output.data(), device_output,
                        total_words * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy output");
    check_hip(hipFree(device_output), "hipFree output");

    int complete_writers = 0;
    for (int writer = 0; writer < kWriterCount; ++writer) {
        std::vector<int> counts(kMatrixElems, 0);
        int poison = 0;
        int unexpected = 0;
        int duplicate = 0;
        int missing = 0;
        const uint16_t* page = output.data() + writer * kPageElems;
        for (int index = 0; index < kPageElems; ++index) {
            const uint16_t value = page[index];
            if (value == kPoison) {
                ++poison;
            } else if (value >= kPatternBase &&
                       value < kPatternBase + kMatrixElems) {
                const int source = value - kPatternBase;
                duplicate += ++counts[source] > 1;
            } else {
                ++unexpected;
            }
        }
        for (int count : counts) {
            missing += count == 0;
        }
        const bool complete =
            poison == kPageElems - kMatrixElems && unexpected == 0 &&
            duplicate == 0 && missing == 0;
        complete_writers += complete ? 1 : 0;
        std::printf(
            "writer_dump writer=%s poison=%d unexpected=%d duplicate=%d "
            "missing=%d complete=%d\n",
            kWriterNames[writer], poison, unexpected, duplicate, missing,
            complete ? 1 : 0);
    }
    std::printf(
        "writer_dump_complete=1 complete_writers=%d total_writers=%d\n",
        complete_writers, kWriterCount);
    return complete_writers == kWriterCount ? 0 : 1;
}
