#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${DQ_MPAIR_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/dq_native_ds_write_mmac_mpair_probe"
ASM="${BUILD_DIR}/dq_native_ds_write_mmac_mpair_probe.asm"
RUN_ROOT="${DQ_MPAIR_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/dq_mpair_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DQ_MPAIR_PMD_TIMEOUT:-300}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dq_native_ds_write_mmac_mpair_probe.cpp \
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
  --symbol-regex dq_native_ds_write_mmac_mpair_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*ds_write_matrix_format/ { write += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { read_n += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { read_t += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate writer=%d read_n=%d read_t=%d mmac=%d scalar_read=%d permute=%d permlane=%d\n",
           write, read_n, read_t, mmac, scalar_read, permute, permlane)
    if (write != 4 || read_n != 2 || read_t != 3 || mmac < 8 ||
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
no_pack_pass="$(grep -c 'native_ds_write_mmac_mpair_no_pack_pass=1' pmd_stdout.log || true)"
any_pass="$(grep -c 'native_ds_write_mmac_mpair_any_pass=1' pmd_stdout.log || true)"
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
if [[ "${no_pack_pass}" == "1" ]]; then
  semantic=NATIVE_NO_PACK_PASS
elif [[ "${any_pass}" == "1" ]]; then
  semantic=PACK_ONLY_PASS
else
  semantic=NO_MATCH
fi

grep -E '^(PASS|BEST|native_ds_write_mmac_mpair_)' pmd_stdout.log \
  | tee result.txt || true
printf 'dq_mpair_probe transport=%s semantic=%s pmd_status=%s panic=%s bank=%s run=%s\n' \
  "${transport}" "${semantic}" "${pmd_status}" "${panic_lines}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${transport}" == "PASS" ]]
