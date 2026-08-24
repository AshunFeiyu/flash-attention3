#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${DQ_B32_STORE_BUILD_DIR:-build/dq_b32_matrix_store}"
BIN="${BUILD_DIR}/dq_b32_matrix_store_probe"
ASM="${BUILD_DIR}/dq_b32_matrix_store_probe.asm"
RUN_ROOT="${DQ_B32_STORE_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/dq_b32_matrix_store_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DQ_B32_STORE_PMD_TIMEOUT:-180}"

SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dq_b32_matrix_store_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex dq_b32_matrix_store_probe_kernel

awk '
  /^[[:space:]]*v_mmac_f32_16x16x16_f16/ { mmac += 1 }
  /^[[:space:]]*matrix_store_16x16_b32/ { store += 1 }
  /^[[:space:]]*matrix_store_16x16_b32/ && /lds/ { lds_store += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_write/ { ds_write += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  END {
    printf("asm_gate mmac=%d matrix_store_b32=%d lds_store=%d scalar_read=%d ds_write=%d permute=%d\n",
           mmac, store, lds_store, scalar_read, ds_write, permute)
    if (mmac == 0 || store == 0 || lds_store != 0 || scalar_read != 0 ||
        ds_write != 0 || permute != 0) exit 1
  }
' "${ASM_ABS}"

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"
cd "${RUN_DIR}"

set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c "${GPU_CHIP}" -m m5out \
  -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' \
  pmd_stdout.log || true)"
pass_lines="$(grep -cE 'dq_b32_vgpr_matrix_store passing=[1-9][0-9]*/48' \
  pmd_stdout.log || true)"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
if [[ "${#stats_files[@]}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" || \
      "${pass_lines}" != "1" || "${bank_conflicts}" != "0" ]]; then
  printf 'dq_b32_matrix_store_status=FAIL pmd=%s panic=%s pass=%s bank=%s run=%s\n' \
    "${pmd_status}" "${panic_lines}" "${pass_lines}" \
    "${bank_conflicts}" "${RUN_DIR}"
  exit 1
fi

grep 'dq_b32_vgpr_matrix_store' pmd_stdout.log | tee result.txt
printf 'dq_b32_matrix_store_status=PASS bank=%s run=%s\n' \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
