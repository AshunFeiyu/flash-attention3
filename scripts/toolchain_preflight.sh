#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${repo_dir}"

: "${SHAOBO_REQUIRE_TOOLCHAIN_LOCK:=1}"
export SHAOBO_REQUIRE_TOOLCHAIN_LOCK
source scripts/env.sh

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

run_py="${PMD_PATH}/scripts/run.py"
[[ -f "${run_py}" ]] || run_py="${PMD_PATH}/core/scripts/run.py"
[[ -f "${run_py}" ]] || {
  echo "PMD run.py not found under ${PMD_PATH}" >&2
  exit 1
}

gem5="${PMD_PATH}/gem5.opt"
[[ -x "${gem5}" ]] || gem5="${PMD_PATH}/build/GCN3_X86/gem5.opt"
[[ -x "${gem5}" ]] || {
  echo "PMD gem5.opt not found under ${PMD_PATH}" >&2
  exit 1
}

pmd_lib="${SHAOBO_PMD_ROOT}/lib/libgem5.so"
[[ -f "${pmd_lib}" ]] || pmd_lib="${PMD_PATH}/libgem5_opt.so"
[[ -f "${pmd_lib}" ]] || {
  echo "PMD libgem5 not found under ${SHAOBO_PMD_ROOT}" >&2
  exit 1
}

compiler="${CLANGXX:-}"
if [[ -z "${compiler}" && -n "${SHAOBO_COMPILER_ROOT:-}" ]]; then
  compiler="${SHAOBO_COMPILER_ROOT}/llvm/bin/clang++"
  [[ -x "${compiler}" ]] || \
    compiler="${SHAOBO_COMPILER_ROOT}/opt/rocm-6.3.3/llvm/bin/clang++"
fi
[[ -n "${compiler}" && -x "${compiler}" ]] || {
  echo "set CLANGXX or SHAOBO_COMPILER_ROOT to the audited compiler" >&2
  exit 1
}

echo "GPU_CHIP=${GPU_CHIP}"
echo "GPU_ARGS=${GPU_ARGS}"
echo "ROCM_PATH=${ROCM_PATH}"
echo "SHAOBO_PMD_ROOT=${SHAOBO_PMD_ROOT}"
echo "PMD_PATH=${PMD_PATH}"
echo "SOC_PATH=${SOC_PATH}"
echo "PMD_CONFIG_SEED=${PMD_CONFIG_SEED}"
echo "PMD_RUN_PY=${run_py}"
echo "COMPILER=${compiler}"
"${compiler}" --version | sed -n '1,3p'
hash_file "${compiler}"
shaobo_verify_latest_compiler "${compiler}"
hash_file "${gem5}"
hash_file "${pmd_lib}"
hash_file "${PMD_CONFIG_SEED}"
echo "toolchain_preflight_status=PASS"
