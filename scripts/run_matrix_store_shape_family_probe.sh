#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${MATRIX_STORE_SHAPE_BUILD_DIR:-build/matrix_store_shape_family}"
BIN="${BUILD_DIR}/matrix_store_shape_family_probe"
ASM="${BUILD_DIR}/matrix_store_shape_family_probe.asm"
RUN_ROOT="${MATRIX_STORE_SHAPE_RUN_ROOT:-/zys/sb/matrix_store_shape_family}"
PMD_TIMEOUT="${MATRIX_STORE_SHAPE_PMD_TIMEOUT:-180}"
CONFIG_SEED="${MATRIX_STORE_SHAPE_CONFIG_SEED:-${PMD_CONFIG_SEED:-}}"

mkdir -p "${RUN_ROOT}" "${BUILD_DIR}"
TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/matrix_store_shape_family_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:--fno-strict-aliasing}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_RUN_ON_MODEL=0 \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" --symbol-regex matrix_store_shape_family_probe_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0 --csv \
  | tee "${RUN_ROOT}/metadata.csv"

for op in 32x16 64x16 32x32; do
  grep -q "matrix_load_${op}_b16" "${ASM}" || {
    echo "missing matrix_load_${op}_b16" >&2; exit 1;
  }
  grep -q "matrix_store_${op}_b16" "${ASM}" || {
    echo "missing matrix_store_${op}_b16" >&2; exit 1;
  }
done

printf 'mode,status,shape,mismatches,poison,first_row,first_col,pass,run_dir\n' \
  >"${RUN_ROOT}/summary.csv"
for mode in 0 1 2; do
  run_dir="${RUN_ROOT}/mode${mode}"
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}/m5out"
  if [[ -n "${CONFIG_SEED}" ]]; then
    cp "${CONFIG_SEED}" "${run_dir}/m5out/config.ini"
  fi
  set +e
  MATRIX_STORE_SHAPE_MODE="${mode}" timeout --kill-after=5 "${PMD_TIMEOUT}" \
    python3 "${PMD_PATH}/scripts/run.py" -c sb -m "${run_dir}/m5out" \
    -e "${ROOT}/${BIN}" >"${run_dir}/pmd_stdout.log" 2>&1
  status="$?"
  set -e
  line="$(grep 'matrix_store_shape shape=' \
    "${run_dir}/pmd_stdout.log" | tail -n 1 || true)"
  if [[ -n "${line}" ]]; then
    shape="$(sed -n 's/.*shape=\([0-9]*x[0-9]*\).*/\1/p' <<<"${line}")"
    mismatches="$(sed -n 's/.*mismatches=\([0-9]*\).*/\1/p' <<<"${line}")"
    poison="$(sed -n 's/.*poison=\([0-9]*\).*/\1/p' <<<"${line}")"
    first_row="$(sed -n 's/.*first_row=\(-\{0,1\}[0-9]*\).*/\1/p' <<<"${line}")"
    first_col="$(sed -n 's/.*first_col=\(-\{0,1\}[0-9]*\).*/\1/p' <<<"${line}")"
    pass="$(sed -n 's/.*pass=\([01]\).*/\1/p' <<<"${line}")"
  else
    shape=""; mismatches=""; poison=""; first_row=""; first_col=""; pass=""
    status="${status}:NO_RESULT"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${mode}" "${status}" "${shape}" "${mismatches}" "${poison}" \
    "${first_row}" "${first_col}" "${pass}" "${run_dir}" \
    | tee -a "${RUN_ROOT}/summary.csv"
done

cat "${RUN_ROOT}/summary.csv"
