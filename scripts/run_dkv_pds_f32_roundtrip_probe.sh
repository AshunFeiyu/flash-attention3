#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${PDS_F32_BUILD_DIR:-build/pds_f32_roundtrip}"
BIN="${BUILD_DIR}/dkv_pds_f32_roundtrip_probe"
ASM="${BUILD_DIR}/dkv_pds_f32_roundtrip_probe.asm"
RUN_ROOT="${PDS_RUN_ROOT:-/zys/shaobo/runs}"
RUN_DIR="${RUN_ROOT}/dkv_pds_f32_roundtrip_probe_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${PDS_F32_PMD_TIMEOUT:-120}"

cleanup_probe_processes() {
  # PMD may detach the simulated executable from timeout's process group.
  pkill -TERM -f -x "${ROOT}/${BIN}" 2>/dev/null || true
  pkill -TERM -f "${PMD_PATH}/scripts/run.py.*${ROOT}/${BIN}" \
    2>/dev/null || true
  sleep 1
  pkill -KILL -f -x "${ROOT}/${BIN}" 2>/dev/null || true
  pkill -KILL -f "${PMD_PATH}/scripts/run.py.*${ROOT}/${BIN}" \
    2>/dev/null || true
}

trap cleanup_probe_processes EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dkv_pds_f32_roundtrip_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex dkv_pds_f32_roundtrip_probe_kernel

awk '
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  /^[[:space:]]*ds_write_matrix/ { write += 1 }
  /^[[:space:]]*ds_read_matrix/ { read += 1 }
  /^[[:space:]]*v_mmac_f32_16x16x16_f16/ { mmac += 1 }
  END {
    printf("asm trap=%d ds_write=%d ds_read=%d mmac=%d\n",
           trap, write, read, mmac)
    if (trap != 0 || write == 0 || read == 0 || mmac == 0) exit 1
  }
' "${ASM}"

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out -e "${ROOT}/${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
cleanup_probe_processes
trap - EXIT INT TERM
cat pmd_stdout.log
set -e

panic_lines="$(grep -ciE 'panic:|fatal:' pmd_stdout.log || true)"
invalid_opcode_lines="$(grep -ci 'Invalid opcode encountered:' \
  pmd_stdout.log || true)"
result_lines="$(grep -c 'f32_pds_roundtrip candidate=' pmd_stdout.log || true)"
semantic_pass_lines="$(grep -c 'f32_pds_roundtrip any_semantic_pair=1' \
  pmd_stdout.log || true)"
bank_conflicts="0"
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${invalid_opcode_lines}" != "0" ||
      "${result_lines}" != "4" ||
      "${semantic_pass_lines}" != "1" ||
      "${bank_conflicts}" != "0" ]]; then
  printf 'pds_f32_roundtrip_status=FAIL pmd_status=%s result_lines=%s semantic_pass=%s panic=%s invalid_opcode=%s bank=%s\n' \
    "${pmd_status}" "${result_lines}" "${semantic_pass_lines}" \
    "${panic_lines}" "${invalid_opcode_lines}" "${bank_conflicts}" | \
    tee result.txt
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'f32_pds_roundtrip' pmd_stdout.log | tee result.txt
printf 'pds_f32_roundtrip_status=PASS run=%s\n' "${RUN_DIR}" | \
  tee -a result.txt
