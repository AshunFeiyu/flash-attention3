#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${MATRIX_STORE_COMPLETION_BUILD_DIR:-build/matrix_store_completion}"
BIN="${BUILD_DIR}/matrix_store_completion_probe"
ASM="${BUILD_DIR}/matrix_store_completion_probe.asm"
RUN_ROOT="${MATRIX_STORE_COMPLETION_RUN_ROOT:-/zys/sb/matrix_store_completion}"
PMD_TIMEOUT="${MATRIX_STORE_COMPLETION_PMD_TIMEOUT:-180}"
BUILD_ROCM_PATH="${MATRIX_STORE_COMPLETION_BUILD_ROCM_PATH:-${ROCM_PATH}}"
MODES="${MATRIX_STORE_COMPLETION_MODES:-0 2 4 6 7 8 9}"
ENABLE_VWCNT="${MATRIX_STORE_COMPLETION_ENABLE_VWCNT:-0}"
CONFIG_SEED="${MATRIX_STORE_COMPLETION_CONFIG_SEED:-${PMD_CONFIG_SEED:-}}"

mkdir -p "${RUN_ROOT}" "${BUILD_DIR}"
ROCM_PATH="${BUILD_ROCM_PATH}" TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/matrix_store_completion_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:--fno-strict-aliasing} -DSHAOBO_PROBE_ENABLE_VWCNT=${ENABLE_VWCNT}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_RUN_ON_MODEL=0 \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex matrix_store_completion_probe_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0 --csv \
  | tee "${RUN_ROOT}/metadata.csv"

awk '
  /matrix_store_32x16_b16/ { store += 1 }
  /matrix_store_32x16_b16/ && /lds/ { lds += 1 }
  /matrix_store_32x16_b16/ && /v\[/ { vgpr += 1 }
  /^[[:space:]]*s_waitcnt_vwcnt/ { vw += 1 }
  /^[[:space:]]*s_trap/ { trap += 1 }
  END {
    printf("asm_gate matrix_store=%d lds_source=%d vgpr_form=%d " \
           "vwcnt=%d trap=%d\n", store, lds, vgpr, vw, trap)
    if (store < 2 || lds < 2 || vgpr < 1 || trap != 0) exit 1
  }
' "${ASM}" | tee "${RUN_ROOT}/asm_gate.txt"

printf 'mode,status,mismatches,poison,first_row,first_col,pass,run_dir\n' \
  >"${RUN_ROOT}/summary.csv"
for mode in ${MODES}; do
  run_dir="${RUN_ROOT}/mode${mode}"
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}/m5out"
  if [[ -n "${CONFIG_SEED}" ]]; then
    cp "${CONFIG_SEED}" "${run_dir}/m5out/config.ini"
  fi
  set +e
  export MATRIX_STORE_COMPLETION_MODE="${mode}"
  timeout --kill-after=5 "${PMD_TIMEOUT}" \
    python3 "${PMD_PATH}/scripts/run.py" -c sb -m "${run_dir}/m5out" \
    -e "${ROOT}/${BIN}" >"${run_dir}/pmd_stdout.log" 2>&1
  status="$?"
  set -e
  line="$(grep 'matrix_store_completion mode=' \
    "${run_dir}/pmd_stdout.log" | tail -n 1 || true)"
  if [[ -n "${line}" ]]; then
    mismatches="$(sed -n 's/.*mismatches=\([0-9]*\).*/\1/p' <<<"${line}")"
    poison="$(sed -n 's/.*poison=\([0-9]*\).*/\1/p' <<<"${line}")"
    first_row="$(sed -n 's/.*first_row=\(-\{0,1\}[0-9]*\).*/\1/p' <<<"${line}")"
    first_col="$(sed -n 's/.*first_col=\(-\{0,1\}[0-9]*\).*/\1/p' <<<"${line}")"
    pass="$(sed -n 's/.*pass=\([01]\).*/\1/p' <<<"${line}")"
  else
    mismatches=""; poison=""; first_row=""; first_col=""; pass=""
    status="${status}:NO_RESULT"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${mode}" "${status}" "${mismatches}" "${poison}" \
    "${first_row}" "${first_col}" "${pass}" "${run_dir}" \
    | tee -a "${RUN_ROOT}/summary.csv"
done

cat "${RUN_ROOT}/summary.csv"
