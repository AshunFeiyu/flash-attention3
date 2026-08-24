#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"

export GPU_CHIP="${GPU_CHIP:-sb}"
export GPU_ARGS="${GPU_ARGS:-['--SQCIPfLines=7']}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.3}"
export SHAOBO_PMD_ROOT="${SHAOBO_PMD_ROOT:-/zys/shaobo/toolchains/pmd_20260824}"
export PMD_PATH="${PMD_PATH:-${SHAOBO_PMD_ROOT}/core}"
export SOC_PATH="${SOC_PATH:-${SHAOBO_PMD_ROOT}/soc}"
export PATH="${SHAOBO_PMD_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SOC_PATH}/libs:${SHAOBO_PMD_ROOT}/lib:${PMD_PATH}:${LD_LIBRARY_PATH:-}"
export RPY_LIB_PATH="${ROCM_PATH}/lib:${SHAOBO_PMD_ROOT}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs"

BUILD_DIR="${MATRIX_STORE_COMPLETION_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/matrix_store_32x16_completion_probe"
ASM="${BUILD_DIR}/matrix_store_32x16_completion_probe.asm"
RUN_ROOT="${MATRIX_STORE_COMPLETION_RUN_ROOT:-/zys/sb/matrix_store_32x16_completion_probe}"
RUN_DIR="${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S)"

TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/matrix_store_32x16_completion_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" --symbol-regex matrix_store_32x16_completion_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

mkdir -p "${RUN_DIR}/m5out"
cd "${RUN_DIR}"
stub="/tmp/shaobo_runpy_stub_${BASHPID:-$$}"
mkdir -p "${stub}"
printf 'def setproctitle(title):\n    return None\n\ndef getproctitle():\n    return "pmd"\n' \
  >"${stub}/setproctitle.py"
export PYTHONPATH="${stub}:${PYTHONPATH:-}"

timeout --kill-after=5 600 \
  python3 "${PMD_PATH}/scripts/run.py" -c "${GPU_CHIP}" -m m5out \
  -e "$(realpath "${ROOT}/${BIN}")" >pmd_stdout.log 2>&1
cat pmd_stdout.log

grep '^matrix_store_32x16_completion' pmd_stdout.log | tee result.txt
grep -q 'matrix_store_32x16_completion_status=PASS' result.txt
! grep -Eiq 'panic:|fatal:|not init or has been freed' pmd_stdout.log
printf 'matrix_store_32x16_completion_probe_status=PASS run=%s\n' \
  "${RUN_DIR}" | tee -a result.txt
