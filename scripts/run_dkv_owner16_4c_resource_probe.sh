#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${DKV_4C_BUILD_DIR:-build/dkv_owner16_4c_resource}"
BIN="${BUILD_DIR}/dkv_owner16_4c_resource_probe"
ASM="${BUILD_DIR}/dkv_owner16_4c_resource_probe.asm"
RUN_ROOT="${DKV_4C_RUN_ROOT:-/zys/shaobo_runs}"
RUN_DIR="${RUN_ROOT}/dkv_owner16_4c_resource_probe_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DKV_4C_PMD_TIMEOUT:-240}"
PMD_RUNTIME_ROOT="${PMD_RUNTIME_ROOT:-${ROCM_PATH}}"
if [[ "$(basename "${PMD_PATH}")" == "core" &&
      -d "$(dirname "${PMD_PATH}")/lib" ]]; then
  PMD_RUNTIME_ROOT="${PMD_RUNTIME_ROOT_OVERRIDE:-$(dirname "${PMD_PATH}")}"
fi

TARGET_GFX=946 \
SHAOBO_RUN_ON_MODEL=1 \
BUILD_ASM=1 \
SRC=probes/dkv_owner16_4c_resource_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex dkv_owner16_4c_resource_probe_kernel

awk '
  /matrix_load_32x16_b16/ && /bps/ && /lds/ { bps += 1 }
  /ds_read_matrix_format/ || /ds_read_matrix_trans_format/ { read += 1 }
  /v_mmac/ { mmac += 1 }
  /s_set_vgpr_size 128/ { resize128 += 1 }
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  END {
    printf("asm bps=%d ds_read_matrix=%d mmac=%d resize128=%d trap=%d\n",
           bps, read, mmac, resize128, trap)
    if (bps < 8 || read < 16 || mmac < 16 || resize128 != 4 || trap != 0) {
      exit 1
    }
  }
' "${ASM}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
if [[ -n "${PMD_CONFIG_SEED:-}" ]]; then
  mkdir -p m5out
  cp "${PMD_CONFIG_SEED}" m5out/config.ini
fi
run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi
set +e
ROCM_PATH="${PMD_RUNTIME_ROOT}" \
LD_LIBRARY_PATH="${SOC_PATH}/libs" \
RPY_LIB_PATH="${PMD_RUNTIME_ROOT}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs" \
GPU_ARGS="['--SQCIPfLines=7']" \
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${run_py}" -c sb -m m5out -e "${ROOT}/${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
pass_lines="$(grep -c 'dkv_owner16_4c_resource_probe .* pass=1' pmd_stdout.log || true)"
bank_conflicts="0"
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${pass_lines}" != "1" || "${bank_conflicts}" != "0" ]]; then
  printf 'dkv_owner16_4c_resource_probe_status=FAIL pmd_status=%s pass_lines=%s panic=%s bank=%s run=%s\n' \
    "${pmd_status}" "${pass_lines}" "${panic_lines}" \
    "${bank_conflicts}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'dkv_owner16_4c_resource_probe' pmd_stdout.log | tee result.txt
printf 'dkv_owner16_4c_resource_probe_status=PASS bank=%s run=%s\n' \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
