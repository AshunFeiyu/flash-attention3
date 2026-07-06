#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "shaobo_instr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ins = shaobo::fa3::bwd::instr;

namespace {

constexpr int kRows = 64;
constexpr int kDim = 128;
constexpr int kMatrixBlockBytes = 32 * 32 * 2;
constexpr int kLoadKinds = 2;
constexpr int kReadKinds = 2;
constexpr int kDTiles = 4;
constexpr int kNTiles = 2;
constexpr int kHalves = 2;
constexpr int kLanes = 64;
constexpr int kVec = 8;
constexpr uint16_t kBitBase = 0x1000;

__host__ __device__ __forceinline__ int out_index(int load_kind,
                                                  int read_kind,
                                                  int d_tile,
                                                  int n_tile,
                                                  int half,
                                                  int lane,
                                                  int vec) {
    int idx = load_kind;
    idx = idx * kReadKinds + read_kind;
    idx = idx * kDTiles + d_tile;
    idx = idx * kNTiles + n_tile;
    idx = idx * kHalves + half;
    idx = idx * kLanes + lane;
    idx = idx * kVec + vec;
    return idx;
}

__global__ void dq_k_fragment_probe_kernel(const __half* __restrict__ k,
                                           uint16_t* __restrict__ out) {
#if defined(__gfx946__) || defined(__gfx92a__)
    __shared__ __half lds32x32[kRows * kDim];
    __shared__ __half lds32x16[kRows * kDim];
    const int lane = static_cast<int>(threadIdx.x & 63);

    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
        for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
            const int n_base = n_tile * 32;
            const int d_base = d_tile * 32;
            const int off = (n_tile * kDTiles + d_tile) * kMatrixBlockBytes;
            ins::Vec4U32 src32 =
                ins::prepare_matrix_src(k + n_base * kDim + d_base, kDim);
            ins::matrix_load_32x32_b16_bps_lds(
                lds32x32, src32, off, true);

            ins::Vec4U32 src16_lo =
                ins::prepare_matrix_src(k + n_base * kDim + d_base, kDim);
            ins::matrix_load_32x16_b16_bps_lds(
                lds32x16, src16_lo, off);
            ins::Vec4U32 src16_hi =
                ins::prepare_matrix_src(k + n_base * kDim + d_base + 16, kDim);
            ins::matrix_load_32x16_b16_bps_lds(
                lds32x16, src16_hi, off + 1024);
        }
    }
    ins::wait_lgkm(0);
    __syncthreads();

    for (int load_kind = 0; load_kind < kLoadKinds; ++load_kind) {
        const __half* lds = load_kind == 0 ? lds32x32 : lds32x16;
        for (int read_kind = 0; read_kind < kReadKinds; ++read_kind) {
            for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
                for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
                    for (int half = 0; half < kHalves; ++half) {
                        const int off =
                            (n_tile * kDTiles + d_tile) * kMatrixBlockBytes +
                            half * 1024;
                        ins::F16x8 frag;
                        if (read_kind == 0) {
                            ins::ds_read_matrix_32x16_normal(
                                lds, off, frag.f16x8);
                        } else {
                            ins::ds_read_matrix_32x16_trans(
                                lds, off, frag.f16x8);
                        }
                        ins::wait_lgkm(0);
                        const uint16_t* bits =
                            reinterpret_cast<const uint16_t*>(&frag.f16x8);
                        for (int vec = 0; vec < kVec; ++vec) {
                            out[out_index(load_kind, read_kind, d_tile, n_tile,
                                          half, lane, vec)] = bits[vec];
                        }
                    }
                }
            }
        }
    }
#else
    (void)k;
    (void)out;
#endif
}

void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

const char* load_name(int load_kind) {
    return load_kind == 0 ? "mls32x32_t" : "mls32x16_pair";
}

const char* read_name(int read_kind) {
    return read_kind == 0 ? "normal" : "trans";
}

}  // namespace

int main() {
    std::vector<uint16_t> h_k(kRows * kDim);
    for (int r = 0; r < kRows; ++r) {
        for (int d = 0; d < kDim; ++d) {
            h_k[r * kDim + d] =
                static_cast<uint16_t>(kBitBase + r * kDim + d);
        }
    }

    const size_t out_elems = static_cast<size_t>(kLoadKinds) * kReadKinds *
                             kDTiles * kNTiles * kHalves * kLanes * kVec;
    std::vector<uint16_t> h_out(out_elems, 0);

    __half* d_k = nullptr;
    uint16_t* d_out = nullptr;
    check_hip(hipMalloc(&d_k, h_k.size() * sizeof(uint16_t)), "hipMalloc k");
    check_hip(hipMalloc(&d_out, h_out.size() * sizeof(uint16_t)),
              "hipMalloc out");
    check_hip(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              "hipMemcpy k");
    check_hip(hipMemset(d_out, 0, h_out.size() * sizeof(uint16_t)),
              "hipMemset out");

    hipLaunchKernelGGL(dq_k_fragment_probe_kernel, dim3(1), dim3(64), 0, 0,
                       d_k, d_out);
    check_hip(hipGetLastError(), "launch");
    check_hip(hipDeviceSynchronize(), "sync");
    check_hip(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy out");

    std::puts(
        "load_kind,read_kind,d_tile,n_tile,half,lane,vec,bits,k_row,k_d");
    for (int load_kind = 0; load_kind < kLoadKinds; ++load_kind) {
        for (int read_kind = 0; read_kind < kReadKinds; ++read_kind) {
            for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
                for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
                    for (int half = 0; half < kHalves; ++half) {
                        for (int lane = 0; lane < kLanes; ++lane) {
                            for (int vec = 0; vec < kVec; ++vec) {
                                const uint16_t bits = h_out[out_index(
                                    load_kind, read_kind, d_tile, n_tile,
                                    half, lane, vec)];
                                int row = -1;
                                int d = -1;
                                if (bits >= kBitBase &&
                                    bits < kBitBase + kRows * kDim) {
                                    const int coord = bits - kBitBase;
                                    row = coord / kDim;
                                    d = coord - row * kDim;
                                }
                                std::printf(
                                    "%s,%s,%d,%d,%d,%d,%d,0x%04x,%d,%d\n",
                                    load_name(load_kind), read_name(read_kind),
                                    d_tile, n_tile, half, lane, vec, bits, row,
                                    d);
                            }
                        }
                    }
                }
            }
        }
    }

    hipFree(d_out);
    hipFree(d_k);
    return 0;
}
