#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source scripts/toolchain_lock.sh

TARGET_GFX="${TARGET_GFX:-946}"
SRC="${SRC:-src/dkv_kernel.cpp}"
LINK_OBJECTS="${LINK_OBJECTS:-}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"
BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BIN:-${BUILD_DIR}/fa3_bwd_wasp_clean}"
ASM="${ASM:-${BUILD_DIR}/fa3_bwd_wasp_clean.asm}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.3}"

if [[ -z "${SHAOBO_COMPILER_ROOT:-}" ]]; then
  echo "latest compiler lock cannot resolve ${SHAOBO_LATEST_COMPILER_ROOT}" >&2
  exit 1
fi

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
# The rolling package root does not include libamdhip64. Use the installed HIP
# runtime wrapper, but prove below that it dispatches the locked compiler.
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
TOOLCHAIN_NOTE="compiler=${compiler_rocm}; runtime=${ROCM_PATH}"
[[ -x "${HIPCC}" ]] || {
  echo "invalid HIPCC runtime wrapper: ${HIPCC}" >&2
  exit 1
}

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

if [[ "${SHAOBO_EXPLICIT_WDRA_INIT}" == "1" ]]; then
  EXTRA_FLAGS+=(-DSHAOBO_EXPLICIT_WDRA_INIT=1)
fi

SHAOBO_FLAGS=(
  -mllvm "-turn-off-wdra-trap-handler=${SHAOBO_WDRA_TRAP_HANDLER_MODE}"
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
shaobo_verify_latest_compiler "${CLANGXX}"
shaobo_verify_hipcc_uses_latest_compiler "${HIPCC}"
{
  printf 'compiler_root=%s\n' "${SHAOBO_COMPILER_ROOT:-}"
  printf 'compiler=%s\n' "${CLANGXX}"
  printf 'compiler_llvm_commit=%s\n' "${SHAOBO_LATEST_COMPILER_LLVM_COMMIT}"
  printf 'compiler_sha256=%s\n' "$(shaobo_sha256 "${CLANGXX}")"
  printf 'compiler_index_sha256=%s\n' "${SHAOBO_LATEST_COMPILER_INDEX_SHA256}"
  printf 'compiler_deb_sha256=%s\n' "${SHAOBO_LATEST_COMPILER_DEB_SHA256}"
  printf 'compiler_index_last_modified=%s\n' "${SHAOBO_LATEST_COMPILER_INDEX_LAST_MODIFIED}"
  printf 'hipcc=%s\n' "${HIPCC}"
  printf 'hipcc_sha256=%s\n' "$(shaobo_sha256 "${HIPCC}")"
  printf 'hipcc_compiler_llvm_commit=%s\n' "${SHAOBO_LATEST_COMPILER_LLVM_COMMIT}"
  printf 'pmd_root=%s\n' "${SHAOBO_PMD_ROOT:-}"
  printf 'pmd_config_seed=%s\n' "${PMD_CONFIG_SEED:-}"
  if [[ -n "${PMD_CONFIG_SEED:-}" && -s "${PMD_CONFIG_SEED}" ]]; then
    printf 'pmd_config_seed_sha256=%s\n' "$(shaobo_sha256 "${PMD_CONFIG_SEED}")"
  fi
  printf 'target_gfx=%s\n' "${TARGET_GFX}"
  printf 'wdra_init=%s\n' "${SHAOBO_EXPLICIT_WDRA_INIT}"
  printf 'wdra_trap_handler=%s\n' "${SHAOBO_WDRA_TRAP_HANDLER_MODE}"
} >"${BUILD_DIR}/toolchain_fingerprint.txt"
echo "building ${BIN}"
if [[ -n "${LINK_OBJECTS}" ]]; then
  # shellcheck disable=SC2206
  OBJECT_INPUTS=(${LINK_OBJECTS})
  "${HIPCC}" "${COMMON_FLAGS[@]}" "${OBJECT_INPUTS[@]}" -o "${BIN}"
elif [[ "${COMPILE_ONLY}" == "1" ]]; then
  "${HIPCC}" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
    "${SHAOBO_FLAGS[@]}" -c "${SRC}" -o "${BIN}"
else
  "${HIPCC}" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
    "${SHAOBO_FLAGS[@]}" "${SRC}" -o "${BIN}"
fi

if [[ "${BUILD_ASM:-1}" == "1" && -z "${LINK_OBJECTS}" ]]; then
  echo "building ${ASM}"
  "${CLANGXX}" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" "${SHAOBO_FLAGS[@]}" \
    --cuda-device-only -x hip -S "${SRC}" -o "${ASM}"
fi

if [[ "${COMPILE_ONLY}" != "1" ]]; then
  chmod +x "${BIN}"
fi
echo "build complete: ${BIN}"
