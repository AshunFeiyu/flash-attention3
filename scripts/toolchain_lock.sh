#!/usr/bin/env bash

export SHAOBO_LATEST_COMPILER_ROOT="${SHAOBO_LATEST_COMPILER_ROOT:-/zys/shaobo/toolchains/compiler_perf_model_latest_20260721_root}"
export SHAOBO_LATEST_COMPILER_LLVM_COMMIT="47a7d59a80a4313d0c33d4667c3c8573604d0dbc"
export SHAOBO_LATEST_COMPILER_SHA256="fddad9d6b6a0bc2264d815e97bbc7679fba9268e8f0b71d145acfa466da3b395"
export SHAOBO_REQUIRE_LATEST_COMPILER="${SHAOBO_REQUIRE_LATEST_COMPILER:-1}"
export SHAOBO_EXPLICIT_WDRA_INIT="${SHAOBO_EXPLICIT_WDRA_INIT:-1}"
export SHAOBO_WDRA_TRAP_HANDLER_MODE="${SHAOBO_WDRA_TRAP_HANDLER_MODE:-no-pad}"

if [[ -z "${SHAOBO_COMPILER_ROOT:-}" && \
      -x "${SHAOBO_LATEST_COMPILER_ROOT}/opt/rocm-6.3.3/llvm/bin/clang++" ]]; then
  export SHAOBO_COMPILER_ROOT="${SHAOBO_LATEST_COMPILER_ROOT}"
fi

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
