#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${DS_ROUNDTRIP_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/ds_matrix_reg_roundtrip_probe"
ASM="${BUILD_DIR}/ds_matrix_reg_roundtrip_probe.asm"
RUN_ROOT="${DS_ROUNDTRIP_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/ds_matrix_reg_roundtrip_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DS_ROUNDTRIP_PMD_TIMEOUT:-300}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/ds_matrix_reg_roundtrip_probe.cpp \
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
  --symbol-regex ds_matrix_reg_roundtrip_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*ds_write_matrix_format/ { writer += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { reader_n += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { reader_t += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate writer=%d reader_n=%d reader_t=%d mmac=%d scalar_read=%d permute=%d permlane=%d\n",
           writer, reader_n, reader_t, mmac, scalar_read, permute, permlane)
    if (writer != 4 || reader_n != 2 || reader_t != 1 || mmac != 0 ||
        scalar_read != 0 || permute != 0 || permlane != 0) exit 1
  }
' "${ASM_ABS}"

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
complete="$(grep -c 'roundtrip_probe_complete=1' pmd_stdout.log || true)"
identity_pairs="$(sed -n 's/.*roundtrip_probe_complete=1 identity_pairs=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
permutation_pairs="$(sed -n 's/.*permutation_pairs=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
replay_identity_pairs="$(sed -n 's/.*replay_identity_pairs=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
identity_pairs="${identity_pairs:-0}"
permutation_pairs="${permutation_pairs:-0}"
replay_identity_pairs="${replay_identity_pairs:-0}"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${panic_lines}" == "0" &&
      "${complete}" -gt 0 && "${stats_found}" -gt 0 &&
      "${bank_conflicts}" == "0" ]]; then
  transport=PASS
else
  transport=FAIL
fi
if [[ "${permutation_pairs}" == "12" &&
      "${replay_identity_pairs}" == "12" ]]; then
  semantic=CALIBRATED_ABI_IDENTITY_PASS
elif [[ "${permutation_pairs}" -gt 0 ]]; then
  semantic=PERMUTATION_ONLY
else
  semantic=NO_BIJECTION
fi

grep -E '^(calibration_summary|inverse_replay_summary|roundtrip_probe_complete)' pmd_stdout.log \
  | tee result.txt || true
[[ -s ds_matrix_slot_map.csv ]]
printf 'ds_matrix_reg_roundtrip transport=%s semantic=%s identity_pairs=%s permutation_pairs=%s replay_identity_pairs=%s pmd_status=%s panic=%s stats=%s bank=%s run=%s\n' \
  "${transport}" "${semantic}" "${identity_pairs}" "${permutation_pairs}" \
  "${replay_identity_pairs}" "${pmd_status}" "${panic_lines}" \
  "${stats_found}" "${bank_conflicts}" \
  "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt
sha256sum ds_matrix_slot_map.csv | tee slot_map_sha256.txt

[[ "${transport}" == "PASS" ]]
