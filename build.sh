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

if [[ -n "${SHAOBO_COMPILER_ROOT:-}" ]]; then
  compiler_rocm="${SHAOBO_COMPILER_ROOT}"
  if [[ ! -x "${compiler_rocm}/llvm/bin/clang++" ]]; then
    compiler_rocm="${SHAOBO_COMPILER_ROOT}/opt/rocm-6.3.3"
  fi
  if [[ ! -x "${compiler_rocm}/llvm/bin/clang++" ]]; then
    echo "invalid SHAOBO_COMPILER_ROOT: ${SHAOBO_COMPILER_ROOT}" >&2
    exit 1
  fi
  export HIP_CLANG_PATH="${compiler_rocm}/llvm/bin"
  export PATH="${compiler_rocm}/bin:${HIP_CLANG_PATH}:${PATH}"
  export LD_LIBRARY_PATH="${compiler_rocm}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
  CLANGXX="${compiler_rocm}/llvm/bin/clang++"
  HIPCC="${compiler_rocm}/bin/hipcc"
  TOOLCHAIN_NOTE="SHAOBO_COMPILER_ROOT=${compiler_rocm}"
elif [[ -z "${CLANGXX:-}" ]]; then
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

if [[ "${SHAOBO_RUN_ON_MODEL:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(
    -DSHAOBO_EXPLICIT_WDRA_INIT=1
    -mllvm -run-on-model=true
  )
fi

SHAOBO_FLAGS=(
  -mllvm -disallow-uniform-vmed3-combine=true
  -mllvm -stream-unfolded-args-in-metadata
  -mllvm -disable-machine-sink
)
if [[ "${SHAOBO_DISABLE_WDRA_FLAGS:-0}" != "1" ]]; then
  SHAOBO_FLAGS=(
    -mllvm -support-768-vgprs=true
    -mllvm -vgpr-greedy-alloc-mode=local-wave
    "${SHAOBO_FLAGS[@]}"
  )
fi

echo "target gfx${TARGET_GFX}"
echo "toolchain ${TOOLCHAIN_NOTE}"
echo "clangxx ${CLANGXX}"
echo "hipcc ${HIPCC}"
"${CLANGXX}" --version | sed -n '1p'
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${CLANGXX}"
else
  shasum -a 256 "${CLANGXX}"
fi
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
