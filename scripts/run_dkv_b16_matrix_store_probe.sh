#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${DKV_B16_MATRIX_STORE_BUILD_DIR:-build/dkv_b16_matrix_store}"
BIN="${BUILD_DIR}/dkv_b16_matrix_store_probe"
ASM="${BUILD_DIR}/dkv_b16_matrix_store_probe.asm"
RUN_ROOT="${DKV_B16_MATRIX_STORE_RUN_ROOT:-/zys/shaobo_runs/dkv_b16_matrix_store_probe}"
RUN_DIR="${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DKV_B16_MATRIX_STORE_PMD_TIMEOUT:-180}"
PMD_CONFIG_SEED="${DKV_B16_MATRIX_STORE_CONFIG_SEED:-}"
BUILD_ROCM_PATH="${DKV_B16_MATRIX_STORE_BUILD_ROCM_PATH:-${ROCM_PATH}}"

mkdir -p "${RUN_DIR}"
if [[ -n "${PMD_CONFIG_SEED}" ]]; then
  mkdir -p "${RUN_DIR}/m5out"
  cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
fi

set +e
ROCM_PATH="${BUILD_ROCM_PATH}" TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/dkv_b16_matrix_store_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:--fno-strict-aliasing}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_RUN_ON_MODEL=0 \
./build.sh >"${RUN_DIR}/build.log" 2>&1
build_status="$?"
set -e
cat "${RUN_DIR}/build.log"
if [[ "${build_status}" != "0" ]]; then
  if grep -Eiq 'error:.*matrix_store|unknown builtin|use of undeclared identifier.*matrix_store|too (few|many) arguments' \
      "${RUN_DIR}/build.log"; then
    compile_status=BLOCKED_BUILTIN_SIGNATURE
  else
    compile_status=BUILD_OR_TOOLCHAIN_FAILURE
  fi
  printf 'b16_matrix_store_compile_status=%s build_status=%s run=%s\n' \
    "${compile_status}" "${build_status}" "${RUN_DIR}" \
    | tee "${RUN_DIR}/result.txt"
  printf 'matrix_store_contract=official_hcu_builtin; candidate=t/r=10; canonical_integration_requires_signature_and_layout_pass\n' \
    | tee -a "${RUN_DIR}/result.txt"
  exit 3
fi

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex dkv_b16_matrix_store_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0 \
  --csv | tee "${RUN_DIR}/metadata.csv"

awk '
  /ds_write_matrix_format/ && /element:(0x)?2/ { writer += 1 }
  /matrix_store_32x16_b16/ { store += 1 }
  /^[[:space:]]*s_abarrier_init([[:space:]]|$)/ { abarrier_init += 1 }
  /^[[:space:]]*s_abarrier_seq([[:space:]]|$)/ { abarrier_seq += 1 }
  /^[[:space:]]*s_abarrier_arrive([[:space:]]|$)/ { abarrier_arrive += 1 }
  /^[[:space:]]*s_abarrier_try_wait([[:space:]]|$)/ { abarrier_wait += 1 }
  /^[[:space:]]*s_abarrier_inv([[:space:]]|$)/ { abarrier_inv += 1 }
  /^[[:space:]]*s_ebarrier_sync([[:space:]]|$)/ { ebarrier_sync += 1 }
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  END {
    printf("asm_gate writer_f16=%d matrix_store_b16=%d abarrier_init=%d " \
           "abarrier_seq=%d abarrier_arrive=%d abarrier_wait=%d " \
           "abarrier_inv=%d ebarrier_sync=%d trap=%d\n", writer, store,
           abarrier_init, abarrier_seq, abarrier_arrive, abarrier_wait,
           abarrier_inv, ebarrier_sync, trap)
    if (writer == 0 || store != 2 || abarrier_init != 1 ||
        abarrier_seq != 2 || abarrier_arrive != 2 || abarrier_wait != 2 ||
        abarrier_inv != 1 || ebarrier_sync < 2 || trap != 0) exit 1
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
matrix_control_pass="$(grep -c 'b16_matrix_store_probe matrix_control_pass=1' pmd_stdout.log || true)"
writer_chain_pass="$(grep -c 'b16_matrix_store_probe matrix_control_pass=1 writer_chain_pass=1' pmd_stdout.log || true)"
candidate_lines="$(grep -c 'b16_matrix_store path=' pmd_stdout.log || true)"
bank_conflicts=0
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    m5out/0/*/stats.txt)"
fi

if [[ "${panic_lines}" != "0" || "${candidate_lines}" != "2" ||
      "${bank_conflicts}" != "0" ]]; then
  printf 'b16_matrix_store_probe_status=FAIL_INFRA contract=C2 pmd_status=%s candidates=%s panic=%s bank=%s run=%s\n' \
    "${pmd_status}" "${candidate_lines}" \
    "${panic_lines}" "${bank_conflicts}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

if [[ "${matrix_control_pass}" != "1" ]]; then
  printf 'b16_matrix_store_probe_status=FAIL_MATRIX_STORE_CONTROL pmd_status=%s bank=%s run=%s\n' \
    "${pmd_status}" "${bank_conflicts}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

if [[ "${writer_chain_pass}" != "1" ]]; then
  grep 'b16_matrix_store' pmd_stdout.log | tee result.txt
  printf 'b16_matrix_store_probe_status=OBSERVE_WRITER_LAYOUT_MISMATCH bank=%s run=%s\n' \
    "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
  exit 0
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'b16_matrix_store_probe' pmd_stdout.log | tee result.txt
printf 'b16_matrix_store_probe_status=PASS bank=%s run=%s\n' \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
printf 'comparison_contract=C2_only; C0_fp32_oracle=pending; C1_packed_b16_direct_global=pending\n' \
  | tee -a result.txt
printf 'promotion_gate=C0_fp32_oracle_and_C1_packed_b16_direct_global_required=1\n' \
  | tee -a result.txt
