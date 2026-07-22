#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${DS_WRITE_DUMP_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/ds_matrix_write_lds_dump_probe"
ASM="${BUILD_DIR}/ds_matrix_write_lds_dump_probe.asm"
RUN_ROOT="${DS_WRITE_DUMP_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/ds_matrix_write_lds_dump_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DS_WRITE_DUMP_PMD_TIMEOUT:-600}"

TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/ds_matrix_write_lds_dump_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex 'ds_matrix_writer_dump_probe_kernel' \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

awk '
  /^[[:space:]]*matrix_load_/ { matrix_load += 1 }
  /^[[:space:]]*ds_read_matrix/ { matrix_read += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { matrix_write += 1 }
  /^[[:space:]]*matrix_store_/ { matrix_store += 1 }
  /^[[:space:]]*ds_read_b128/ { dump_read += 1 }
  /^[[:space:]]*ds_write_b128/ { poison_write += 1 }
  /^[[:space:]]*(global|flat)_store_dwordx4/ { global_store += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate matrix_write=%d dump_read=%d poison_write=%d global_store=%d matrix_load=%d matrix_read=%d matrix_store=%d mmac=%d permute=%d permlane=%d\n",
           matrix_write, dump_read, poison_write, global_store, matrix_load,
           matrix_read, matrix_store, mmac, permute, permlane)
    if (matrix_write != 4 || dump_read != 2 || poison_write != 2 ||
        global_store != 2 || matrix_load != 0 || matrix_read != 0 ||
        matrix_store != 0 || mmac != 0 || permute != 0 || permlane != 0)
      exit 1
  }
' "${ASM_ABS}"

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"
cd "${RUN_DIR}"

run_py="${PMD_PATH}/scripts/run.py"
[[ -f "${run_py}" ]] || run_py="${PMD_PATH}/core/scripts/run.py"
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${run_py}" -c "${GPU_CHIP}" -m m5out -e "${BIN_ABS}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
complete_writers="$(sed -n 's/.*complete_writers=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
complete_writers="${complete_writers:-0}"
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
bank_conflicts=0
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${panic_lines}" == "0" &&
      "${complete_writers}" == "4" && "${stats_found}" -gt 0 ]]; then
  status=PASS
else
  status=FAIL
fi

grep -E '^writer_dump' pmd_stdout.log | tee result.txt || true
printf 'ds_matrix_write_dump status=%s complete_writers=%s pmd_status=%s panic=%s stats=%s raw_dump_bank=%s run=%s\n' \
  "${status}" "${complete_writers}" "${pmd_status}" "${panic_lines}" \
  "${stats_found}" "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${status}" == "PASS" ]]
