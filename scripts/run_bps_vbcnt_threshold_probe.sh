#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${BPS_VBCNT_BUILD_DIR:-build/bps_vbcnt_threshold}"
BIN="${BUILD_DIR}/bps_vbcnt_threshold_probe"
ASM="${BUILD_DIR}/bps_vbcnt_threshold_probe.asm"
RUN_ROOT="${BPS_VBCNT_RUN_ROOT:-${SHAOBO_RUN_ROOT}/bps_vbcnt_threshold}"
RUN_DIR="${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${BPS_VBCNT_PMD_TIMEOUT:-240}"
PARTIAL_WAIT="${BPS_VBCNT_PARTIAL_WAIT:-1}"
if [[ "${PARTIAL_WAIT}" != "0" && "${PARTIAL_WAIT}" != "1" &&
      "${PARTIAL_WAIT}" != "2" ]]; then
  echo "BPS_VBCNT_PARTIAL_WAIT must be 0, 1, or 2" >&2
  exit 2
fi

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/bps_vbcnt_threshold_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
EXTRA_CXXFLAGS="-DBPS_VBCNT_PARTIAL_WAIT=${PARTIAL_WAIT}" \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex bps_vbcnt_threshold_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

expected_wait4=0
expected_wait0=1
if [[ "${PARTIAL_WAIT}" == "1" ]]; then
  expected_wait4=1
elif [[ "${PARTIAL_WAIT}" == "2" ]]; then
  expected_wait0=2
fi
awk -v expected_wait4="${expected_wait4}" -v expected_wait0="${expected_wait0}" '
  /^[[:space:]]*matrix_load_32x16_b16/ && /bps/ && /lds/ { bps += 1 }
  /^[[:space:]]*s_waitcnt_vbcnt 4/ { wait4 += 1 }
  /^[[:space:]]*s_waitcnt_vbcnt 0/ { wait0 += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { read += 1 }
  /^[[:space:]]*s_set_vgpr_size/ { resize += 1 }
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  END {
    printf("asm_gate bps=%d wait4=%d wait0=%d matrix_read=%d resize=%d trap=%d\n",
           bps, wait4, wait0, read, resize, trap)
    if (bps != 32 || wait4 != expected_wait4 || wait0 != expected_wait0 ||
        resize != 0 || trap != 0) exit 1
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
pass_lines="$(grep -c 'bps_vbcnt_threshold_probe .* pass=1' pmd_stdout.log || true)"
bank_conflicts=0
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt
grep 'bps_vbcnt_threshold_probe' pmd_stdout.log | tee result.txt || true
if [[ "${pmd_status}" == "0" && "${pass_lines}" == "1" &&
      "${panic_lines}" == "0" && "${bank_conflicts}" == "0" ]]; then
  verdict=PASS
else
  verdict=FAIL
fi
printf 'bps_vbcnt_threshold_status=%s pmd_status=%s pass=%s panic=%s bank=%s run=%s\n' \
  "${verdict}" "${pmd_status}" "${pass_lines}" "${panic_lines}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
printf 'partial_wait=%s\n' "${PARTIAL_WAIT}" | tee -a result.txt

[[ "${verdict}" == "PASS" ]]
