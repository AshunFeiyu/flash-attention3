#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TARGET_GFX="${TARGET_GFX:-946}"
SRC="${SRC:-src/dkv_kernel.cpp}"
BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BIN:-${BUILD_DIR}/fa3_bwd_wasp_clean}"
ASM="${ASM:-${BUILD_DIR}/fa3_bwd_wasp_clean.asm}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
DEFAULT_OVERLAY_BIN="/home/zhangyushun/toolchains/zwj_liuchang_llvm_7940/bin"

if [[ -z "${CLANGXX:-}" ]]; then
  if [[ -x "${DEFAULT_OVERLAY_BIN}/clang++" ]]; then
    export HIP_CLANG_PATH="${HIP_CLANG_PATH:-${DEFAULT_OVERLAY_BIN}}"
    export PATH="${HIP_CLANG_PATH}:${PATH}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    CLANGXX="${HIP_CLANG_PATH}/clang++"
    TOOLCHAIN_NOTE="zwj_liuchang_llvm_7940 overlay"
  else
    CLANGXX="${ROCM_PATH}/llvm/bin/clang++"
    TOOLCHAIN_NOTE="${ROCM_PATH}/llvm default"
  fi
else
  TOOLCHAIN_NOTE="CLANGXX override"
fi

HIPCC="${HIPCC:-hipcc}"

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

EXTRA_FLAGS=()
if [[ -n "${EXTRA_CXXFLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FLAGS=(${EXTRA_CXXFLAGS})
fi

SHAOBO_FLAGS=(
  -mllvm -support-768-vgprs=true
  -mllvm -vgpr-greedy-alloc-mode=local-wave
  -mllvm -disallow-uniform-vmed3-combine=true
  -mllvm -stream-unfolded-args-in-metadata
  -mllvm -disable-machine-sink
)

echo "target gfx${TARGET_GFX}"
echo "toolchain ${TOOLCHAIN_NOTE}"
echo "clangxx ${CLANGXX}"
echo "hipcc ${HIPCC}"
if [[ -n "${HIP_CLANG_PATH:-}" ]]; then
  echo "HIP_CLANG_PATH ${HIP_CLANG_PATH}"
fi
echo "building ${BIN}"
"${HIPCC}" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" "${SHAOBO_FLAGS[@]}" "${SRC}" -o "${BIN}"

if [[ "${BUILD_ASM:-1}" == "1" ]]; then
  echo "building ${ASM}"
  "${CLANGXX}" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" "${SHAOBO_FLAGS[@]}" \
    --cuda-device-only -x hip -S "${SRC}" -o "${ASM}"
fi

chmod +x "${BIN}"
echo "build complete: ${BIN}"
