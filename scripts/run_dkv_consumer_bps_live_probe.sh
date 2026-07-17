#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${CONSUMER_BPS_BUILD_DIR:-build/consumer_bps_live}"
BIN="${BUILD_DIR}/dkv_consumer_bps_live_probe"
ASM="${BUILD_DIR}/dkv_consumer_bps_live_probe.asm"
RUN_ROOT="${CONSUMER_BPS_RUN_ROOT:-/zys/shaobo_runs}"
RUN_DIR="${RUN_ROOT}/dkv_consumer_bps_live_probe_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${CONSUMER_BPS_PMD_TIMEOUT:-180}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dkv_consumer_bps_live_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex dkv_consumer_bps_live_probe_kernel

awk '
  /matrix_load_32x16_b16/ && /bps/ && /lds/ { bps += 1 }
  /ds_read_matrix_format/ { read += 1 }
  /s_set_vgpr_size/ { resize += 1 }
  /s_abarrier_try_wait/ { wait += 1 }
  /s_trap/ { trap += 1 }
  END {
    printf("asm bps=%d ds_read_matrix=%d resize=%d abarrier_wait=%d trap=%d\n",
           bps, read, resize, wait, trap)
    if (bps < 2 || read < 4 || resize < 4 || wait < 3 || trap != 0) exit 1
  }
' "${ASM}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out -e "${ROOT}/${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
pass_lines="$(grep -c 'dkv_consumer_bps_live_probe .* pass=1' pmd_stdout.log || true)"
bank_conflicts="0"
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${pass_lines}" != "1" || "${bank_conflicts}" != "0" ]]; then
  printf 'consumer_bps_live_probe_status=FAIL pmd_status=%s pass_lines=%s panic=%s bank=%s run=%s\n' \
    "${pmd_status}" "${pass_lines}" "${panic_lines}" \
    "${bank_conflicts}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'dkv_consumer_bps_live_probe' pmd_stdout.log | tee result.txt
printf 'consumer_bps_live_probe_status=PASS bank=%s run=%s\n' \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
