#!/usr/bin/env bash
set -euo pipefail

export GPU_CHIP="${GPU_CHIP:-sb}"
export GPU_ARGS="${GPU_ARGS:-['--SQCIPfLines=7']}"
export SHAOBO_RUN_ROOT="${SHAOBO_RUN_ROOT:-/zys/shaobo_runs/fa3_bwd_wasp_clean}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.3}"
if [[ -z "${SHAOBO_PMD_ROOT:-}" ]]; then
  if [[ -f "${ROCM_PATH}/core/scripts/run.py" ]]; then
    export SHAOBO_PMD_ROOT="${ROCM_PATH}"
  else
    export SHAOBO_PMD_ROOT="${ROCM_PATH}/pmd"
  fi
else
  export SHAOBO_PMD_ROOT
fi
if [[ -z "${PMD_PATH:-}" ]]; then
  if [[ -f "${SHAOBO_PMD_ROOT}/core/scripts/run.py" ]]; then
    export PMD_PATH="${SHAOBO_PMD_ROOT}/core"
  else
    export PMD_PATH="${SHAOBO_PMD_ROOT}"
  fi
else
  export PMD_PATH
fi
if [[ -z "${SOC_PATH:-}" ]]; then
  if [[ -d "${SHAOBO_PMD_ROOT}/soc" ]]; then
    export SOC_PATH="${SHAOBO_PMD_ROOT}/soc"
  else
    export SOC_PATH="${SHAOBO_PMD_ROOT}"
  fi
else
  export SOC_PATH
fi
export LD_LIBRARY_PATH="${SOC_PATH}/libs:${SHAOBO_PMD_ROOT}/lib:${PMD_PATH}:${LD_LIBRARY_PATH:-}"
export RPY_LIB_PATH="${ROCM_PATH}/lib:${SHAOBO_PMD_ROOT}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs"
export XCU_SIDECAR_ROOT="${XCU_SIDECAR_ROOT:-${SHAOBO_RUN_ROOT}/tools/xcompute-light}"
export XCU_DEB="${XCU_DEB:-}"

if [[ "${SHAOBO_REQUIRE_TOOLCHAIN_LOCK:-0}" == "1" ]]; then
  [[ "${GPU_CHIP}" == "sb" ]] || {
    echo "toolchain lock requires GPU_CHIP=sb" >&2
    exit 1
  }
  [[ "${GPU_ARGS}" == "['--SQCIPfLines=7']" ]] || {
    echo "toolchain lock requires GPU_ARGS=['--SQCIPfLines=7']" >&2
    exit 1
  }
  [[ -n "${PMD_CONFIG_SEED:-}" && -s "${PMD_CONFIG_SEED}" ]] || {
    echo "toolchain lock requires a non-empty PMD_CONFIG_SEED" >&2
    exit 1
  }
fi

mkdir -p "${SHAOBO_RUN_ROOT}"
export SHAOBO_RUNPY_STUB_ROOT="${SHAOBO_RUNPY_STUB_ROOT:-/tmp/shaobo_runpy_stub_${BASHPID:-$$}}"
mkdir -p "${SHAOBO_RUNPY_STUB_ROOT}"
printf 'def setproctitle(title):\n    return None\n\ndef getproctitle():\n    return "pmd"\n' \
  >"${SHAOBO_RUNPY_STUB_ROOT}/setproctitle.py"
export PYTHONPATH="${SHAOBO_RUNPY_STUB_ROOT}:${PYTHONPATH:-}"
