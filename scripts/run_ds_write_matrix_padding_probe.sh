#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${PADDING_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/ds_write_matrix_padding_probe"
ASM="${BUILD_DIR}/ds_write_matrix_padding_probe.asm"
RUN_ROOT="${PADDING_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/ds_write_matrix_padding_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${PADDING_PMD_TIMEOUT:-180}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/ds_write_matrix_padding_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex ds_write_matrix_padding_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

writer_count="$(awk '/^[[:space:]]*ds_write_matrix_format/ { count += 1 } END { print count + 0 }' "${ASM_ABS}")"
if [[ "${writer_count}" -lt 1 ]]; then
  echo "padding probe missing ds_write_matrix_format" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"
cd "${RUN_DIR}"

run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi

set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${run_py}" -c "${GPU_CHIP}" -m m5out \
  -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
semantic_pass="$(grep -c 'ds_write_matrix_padding_probe_status=PASS' pmd_stdout.log || true)"
if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${semantic_pass}" == "0" ]]; then
  echo "ds_write_matrix_padding_probe_result=FAIL run=${RUN_DIR}" >&2
  exit 1
fi

grep -E '^ds_write_padding|^touched_range_bytes|^ds_write_matrix_padding_probe_status' \
  pmd_stdout.log | tee result.txt
printf 'ds_write_matrix_padding_probe_result=PASS writer=%s run=%s\n' \
  "${writer_count}" "${RUN_DIR}" | tee -a result.txt
