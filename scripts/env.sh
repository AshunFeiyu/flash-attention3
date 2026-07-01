#!/usr/bin/env bash
set -euo pipefail

export GPU_CHIP="${GPU_CHIP:-sb}"
export GPU_ARGS="${GPU_ARGS:-['--SQCIPfLines=7']}"
export SHAOBO_RUN_ROOT="${SHAOBO_RUN_ROOT:-/zys/shaobo_runs/fa3_bwd_wasp_clean}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.3}"
if [[ -z "${PMD_PATH:-}" ]]; then
  if [[ -f "${ROCM_PATH}/pmd/core/scripts/run.py" ]]; then
    export PMD_PATH="${ROCM_PATH}/pmd/core"
  else
    export PMD_PATH="${ROCM_PATH}/pmd"
  fi
else
  export PMD_PATH
fi
if [[ -z "${SOC_PATH:-}" ]]; then
  if [[ -d "${ROCM_PATH}/pmd/soc" ]]; then
    export SOC_PATH="${ROCM_PATH}/pmd/soc"
  else
    export SOC_PATH="${ROCM_PATH}/pmd"
  fi
else
  export SOC_PATH
fi
export LD_LIBRARY_PATH="${SOC_PATH}/libs:${LD_LIBRARY_PATH:-}"
export RPY_LIB_PATH="${ROCM_PATH}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs"
export XCU_SIDECAR_ROOT="${XCU_SIDECAR_ROOT:-${SHAOBO_RUN_ROOT}/tools/xcompute-light}"
export XCU_DEB="${XCU_DEB:-}"

mkdir -p "${SHAOBO_RUN_ROOT}"
mkdir -p /tmp/codex_runpy_stub
printf 'def setproctitle(title):\n    return None\n' >/tmp/codex_runpy_stub/setproctitle.py
export PYTHONPATH="/tmp/codex_runpy_stub:${PYTHONPATH:-}"
