#!/usr/bin/env bash

export SHAOBO_LATEST_COMPILER_ROOT="${SHAOBO_LATEST_COMPILER_ROOT:-/zys/shaobo/toolchains/compiler_perf_model_latest_20260721_root}"
export SHAOBO_LATEST_COMPILER_LLVM_COMMIT="47a7d59a80a4313d0c33d4667c3c8573604d0dbc"
export SHAOBO_LATEST_COMPILER_SHA256="fddad9d6b6a0bc2264d815e97bbc7679fba9268e8f0b71d145acfa466da3b395"
export SHAOBO_REQUIRE_LATEST_COMPILER="${SHAOBO_REQUIRE_LATEST_COMPILER:-1}"
export SHAOBO_LATEST_PMD_ROOT="${SHAOBO_LATEST_PMD_ROOT:-/zys/shaobo/toolchains/pmd_20260717}"
export SHAOBO_LATEST_PMD_CORE_SHA256="4748d40d99414c7be6ab3d2b62bca1f134d3454edec711a6321bdafa237be1e9"
export SHAOBO_LATEST_PMD_LIB_SHA256="29fa2020e6bfb399225e206cf7c589ba838ad56b891cb07c97e88029e954bfa5"
export SHAOBO_LATEST_PMD_SOC_SHA256="d0c03538753a4b91c2aa3e110cb12f1302b66c891c3ab2d446c85de99fe24524"
export SHAOBO_REQUIRE_LATEST_PMD="${SHAOBO_REQUIRE_LATEST_PMD:-1}"
export SHAOBO_EXPLICIT_WDRA_INIT="${SHAOBO_EXPLICIT_WDRA_INIT:-1}"
export SHAOBO_WDRA_TRAP_HANDLER_MODE="${SHAOBO_WDRA_TRAP_HANDLER_MODE:-no-pad}"

if [[ -z "${SHAOBO_COMPILER_ROOT:-}" && \
      -x "${SHAOBO_LATEST_COMPILER_ROOT}/opt/rocm-6.3.3/llvm/bin/clang++" ]]; then
  export SHAOBO_COMPILER_ROOT="${SHAOBO_LATEST_COMPILER_ROOT}"
fi

if [[ -z "${SHAOBO_PMD_ROOT:-}" && \
      -f "${SHAOBO_LATEST_PMD_ROOT}/core/scripts/run.py" ]]; then
  export SHAOBO_PMD_ROOT="${SHAOBO_LATEST_PMD_ROOT}"
fi

shaobo_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

shaobo_verify_latest_compiler() {
  local compiler="$1"
  local actual

  [[ "${SHAOBO_REQUIRE_LATEST_COMPILER}" == "1" ]] || return 0
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${compiler}")"
  else
    actual="$(shasum -a 256 "${compiler}")"
  fi
  actual="${actual%% *}"
  if [[ "${actual}" != "${SHAOBO_LATEST_COMPILER_SHA256}" ]]; then
    echo "compiler lock mismatch: expected ${SHAOBO_LATEST_COMPILER_SHA256}, got ${actual}" >&2
    return 1
  fi
}

shaobo_verify_latest_pmd() {
  local core_gem5="$1"
  local core_lib="$2"
  local soc_gem5="$3"
  local actual_core actual_lib actual_soc

  [[ "${SHAOBO_REQUIRE_LATEST_PMD}" == "1" ]] || return 0
  actual_core="$(shaobo_sha256 "${core_gem5}")"
  actual_lib="$(shaobo_sha256 "${core_lib}")"
  actual_soc="$(shaobo_sha256 "${soc_gem5}")"
  if [[ "${actual_core}" != "${SHAOBO_LATEST_PMD_CORE_SHA256}" ||
        "${actual_lib}" != "${SHAOBO_LATEST_PMD_LIB_SHA256}" ||
        "${actual_soc}" != "${SHAOBO_LATEST_PMD_SOC_SHA256}" ]]; then
    echo "PMD lock mismatch: expected audited HEAD1694 core/lib/soc hashes" >&2
    echo "  core=${actual_core}" >&2
    echo "  lib=${actual_lib}" >&2
    echo "  soc=${actual_soc}" >&2
    return 1
  fi
}
