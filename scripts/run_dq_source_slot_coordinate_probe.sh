#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${DQ_SLOT_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/dq_source_slot_coordinate_probe"
ASM="${BUILD_DIR}/dq_source_slot_coordinate_probe.asm"
RUN_ROOT="${DQ_SLOT_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/dq_source_slot_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DQ_SLOT_PMD_TIMEOUT:-300}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dq_source_slot_coordinate_probe.cpp \
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
  --symbol-regex dq_source_slot_coordinate_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*matrix_load_32x32_b16/ && /bps/ && /lds/ { mls += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { write += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { read_t += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate mls=%d writer=%d read_t=%d mmac=%d scalar_read=%d permute=%d permlane=%d\n",
           mls, write, read_t, mmac, scalar_read, permute, permlane)
    if (mls < 2 || write < 1 || read_t < 1 || mmac < 16 ||
        scalar_read != 0 || permute != 0 || permlane != 0) exit 1
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

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
source_pass="$(grep -c 'source_slot_direct_pass=1' pmd_stdout.log || true)"
bank_conflicts=0
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

if [[ "${pmd_status}" == "0" && "${panic_lines}" == "0" &&
      "${bank_conflicts}" == "0" ]]; then
  transport=PASS
else
  transport=FAIL
fi
if [[ "${source_pass}" -gt 0 ]]; then
  semantic=NATIVE_SOURCE_SLOT_PASS
else
  semantic=NO_MATCH
fi

grep -E 'source_slot_coordinate_(acc_summary|direct_read_summary)|source_slot_orientation_final' \
  pmd_stdout.log | tee result.txt || true
printf 'dq_source_slot_probe transport=%s semantic=%s pmd_status=%s panic=%s bank=%s run=%s\n' \
  "${transport}" "${semantic}" "${pmd_status}" "${panic_lines}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${transport}" == "PASS" ]]
