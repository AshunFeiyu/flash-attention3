#!/usr/bin/env bash
set -euo pipefail

export GPU_CHIP="${GPU_CHIP:-sb}"
export GPU_ARGS="${GPU_ARGS:-['--SQCIPfLines=7']}"
export SHAOBO_RUN_ROOT="${SHAOBO_RUN_ROOT:-/zys/shaobo_runs/fa3_bwd_wasp_fwdstyle_clean}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.3}"
export PMD_PATH="${PMD_PATH:-${ROCM_PATH}/pmd}"
export SOC_PATH="${SOC_PATH:-${PMD_PATH}/soc}"
export LD_LIBRARY_PATH="${SOC_PATH}/libs:${LD_LIBRARY_PATH:-}"
export RPY_LIB_PATH="${ROCM_PATH}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs"
export XCU_SIDECAR_ROOT="${XCU_SIDECAR_ROOT:-${SHAOBO_RUN_ROOT}/tools/xcompute-light}"
export XCU_DEB="${XCU_DEB:-}"

mkdir -p "${SHAOBO_RUN_ROOT}"
