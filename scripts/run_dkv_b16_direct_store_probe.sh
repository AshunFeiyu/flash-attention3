#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${DKV_B16_DIRECT_BUILD_DIR:-build/dkv_b16_direct_store}"
BIN="${BUILD_DIR}/dkv_b16_direct_store_probe"
ASM="${BUILD_DIR}/dkv_b16_direct_store_probe.asm"
RUN_ROOT="${DKV_B16_DIRECT_RUN_ROOT:-/zys/shaobo_runs/dkv_b16_direct_store_probe}"
RUN_DIR="${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DKV_B16_DIRECT_PMD_TIMEOUT:-180}"
PMD_CONFIG_SEED="${DKV_B16_DIRECT_CONFIG_SEED:-}"
BUILD_ROCM_PATH="${DKV_B16_DIRECT_BUILD_ROCM_PATH:-${ROCM_PATH}}"

mkdir -p "${RUN_DIR}"
if [[ -n "${PMD_CONFIG_SEED}" ]]; then
  mkdir -p "${RUN_DIR}/m5out"
  cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
fi

set +e
ROCM_PATH="${BUILD_ROCM_PATH}" TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/dkv_b16_direct_store_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:--fno-strict-aliasing}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_RUN_ON_MODEL=0 \
./build.sh >"${RUN_DIR}/build.log" 2>&1
build_status="$?"
set -e
cat "${RUN_DIR}/build.log"
if [[ "${build_status}" != "0" ]]; then
  printf 'dkv_b16_direct_store_status=BUILD_FAILURE build_status=%s run=%s\n' \
    "${build_status}" "${RUN_DIR}" | tee "${RUN_DIR}/result.txt"
  exit 3
fi

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex dkv_b16_direct_store_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0 \
  --csv | tee "${RUN_DIR}/metadata.csv"

awk '
  /global_store_dwordx2/ { store_x2 += 1 }
  /global_store_dwordx4/ { store_x4 += 1 }
  /v_cvt.*f16.*f32/ { cvt_f16 += 1 }
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  END {
    printf("asm_gate direct_store_x2=%d direct_store_x4=%d " \
           "f32_to_f16=%d trap=%d\n", store_x2, store_x4, cvt_f16, trap)
    if (store_x2 == 0 || cvt_f16 == 0 || trap != 0) exit 1
  }
' "${ASM}" | tee "${RUN_DIR}/asm_gate.txt"

cd "${RUN_DIR}"
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out \
  -e "${ROOT}/${BIN}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
pass_lines="$(grep -c 'dkv_b16_direct_store .* pass=1' pmd_stdout.log || true)"
bank_conflicts=0
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${pass_lines}" != "1" || "${bank_conflicts}" != "0" ]]; then
  grep 'dkv_b16_direct_store' pmd_stdout.log | tee result.txt || true
  printf 'dkv_b16_direct_store_status=FAIL pmd_status=%s pass=%s panic=%s bank=%s run=%s\n' \
    "${pmd_status}" "${pass_lines}" "${panic_lines}" \
    "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'dkv_b16_direct_store' pmd_stdout.log | tee result.txt
printf 'dkv_b16_direct_store_status=PASS bank=%s run=%s\n' \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
printf 'comparison_contract=C1_fp32_vgpr_to_packed_b16_direct_global; canonical_perf_pending=1\n' \
  | tee -a result.txt
