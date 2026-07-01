#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TARGET_GFX="${TARGET_GFX:-946}"
SRC="${SRC:-src/dkv_kernel.cpp}"
BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BIN:-${BUILD_DIR}/fa3_bwd_wasp_clean}"
ASM="${ASM:-${BUILD_DIR}/fa3_bwd_wasp_clean.asm}"

mkdir -p "${BUILD_DIR}"

COMMON_FLAGS=(
  -std=c++17
  -O3
  -g
  -Iinclude
  --offload-arch="gfx${TARGET_GFX}"
  -ffast-math
  -fno-finite-math-only
  -fno-strict-return
  -mcode-object-version=5
  -D__HIP_PLATFORM_AMD__
  -DTARGET="${TARGET_GFX}"
  -DSHAOBO_FA3_BWD_WASP_CLEAN=1
)

SHAOBO_FLAGS=(
  -mllvm -support-768-vgprs=true
  -mllvm -vgpr-greedy-alloc-mode=local-wave
  -mllvm -disallow-uniform-vmed3-combine=true
  -mllvm -stream-unfolded-args-in-metadata
  -mllvm -disable-machine-sink
)

echo "target gfx${TARGET_GFX}"
echo "building ${BIN}"
hipcc "${COMMON_FLAGS[@]}" "${SHAOBO_FLAGS[@]}" "${SRC}" -o "${BIN}"

if [[ "${BUILD_ASM:-1}" == "1" ]]; then
  echo "building ${ASM}"
  /opt/rocm/llvm/bin/clang++ "${COMMON_FLAGS[@]}" "${SHAOBO_FLAGS[@]}" \
    --cuda-device-only -x hip -S "${SRC}" -o "${ASM}"
fi

chmod +x "${BIN}"
echo "build complete: ${BIN}"
