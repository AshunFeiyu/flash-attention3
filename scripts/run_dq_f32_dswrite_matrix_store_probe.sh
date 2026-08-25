#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${DQ_F32_DSSTORE_BUILD_DIR:-build/dq_f32_dswrite_store}"
BIN="${BUILD_DIR}/dq_f32_dswrite_matrix_store_probe"
ASM="${BUILD_DIR}/dq_f32_dswrite_matrix_store_probe.asm"
RUN_ROOT="${DQ_F32_DSSTORE_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/dq_f32_dswrite_store_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DQ_F32_DSSTORE_PMD_TIMEOUT:-180}"

SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dq_f32_dswrite_matrix_store_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex dq_f32_dswrite_matrix_store_probe_kernel

awk '
  /^[[:space:]]*matrix_load_32x32_b16/ { load_b16 += 1 }
  /^[[:space:]]*matrix_load_16x16_b32/ { load_b32 += 1 }
  /^[[:space:]]*ds_read_matrix/ { read += 1 }
  /^[[:space:]]*v_mmac_f32_16x16x16_f16/ { mmac += 1 }
  /^[[:space:]]*ds_write_matrix/ { write += 1 }
  /^[[:space:]]*matrix_store_16x16_b32/ { store += 1 }
  /^[[:space:]]*matrix_store_16x16_b32/ && /lds/ { lds_store += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /^[[:space:]]*s_waitcnt_vbcnt[[:space:]]+0/ { vbcnt += 1 }
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  END {
    printf("asm_gate load_b16=%d load_b32=%d read=%d mmac=%d write=%d store=%d lds_store=%d scalar_read=%d permute=%d vbcnt=%d trap=%d\n",
           load_b16, load_b32, read, mmac, write, store, lds_store,
           scalar_read, permute, vbcnt, trap)
    if (load_b16 == 0 || load_b32 == 0 || read == 0 || mmac == 0 ||
        write == 0 || store == 0 || lds_store == 0 || scalar_read != 0 ||
        permute != 0 || vbcnt == 0 || trap != 0)
      exit 1
  }
' "${ASM_ABS}"

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"
"${ROOT}/scripts/extract_device_isa.sh" "${BIN_ABS}" \
  "${RUN_DIR}/device_isa"
cd "${RUN_DIR}"

set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c "${GPU_CHIP}" -m m5out \
  -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' \
  pmd_stdout.log || true)"
invalid_opcode_lines="$(grep -ci 'Invalid opcode encountered:' \
  pmd_stdout.log || true)"
control_pass_lines="$(grep -cE \
  'dq_f32_dswrite_store source=mls_control .*pass=1' \
  pmd_stdout.log || true)"
target_pass_lines="$(grep -cE \
  'dq_f32_dswrite_store source=mmac_lit0_lts0_write_t1 .*pass=1' \
  pmd_stdout.log || true)"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
if [[ "${#stats_files[@]}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${invalid_opcode_lines}" != "0" ]]; then
  : >invalid_opcode_map.txt
  while read -r reported_word; do
    printf 'reported_word=%s\n' "${reported_word}" | \
      tee -a invalid_opcode_map.txt
    grep -i "${reported_word#0x}" device_isa/device_isa_raw.txt | \
      tee -a invalid_opcode_map.txt || true
  done < <(grep -ioE 'Invalid opcode encountered: 0x[0-9a-f]+' \
    pmd_stdout.log | awk '{print $NF}' | sort -u)
fi

grep 'dq_f32_dswrite_store' pmd_stdout.log | tee result.txt || true
printf 'dq_f32_dswrite_store_status=%s pmd=%s panic=%s invalid_opcode=%s control_pass=%s target_pass=%s bank=%s run=%s\n' \
  "$([[ "${pmd_status}" == "0" && "${panic_lines}" == "0" && \
       "${invalid_opcode_lines}" == "0" && \
       "${control_pass_lines}" == "1" && "${target_pass_lines}" == "1" && \
       "${bank_conflicts}" == "0" ]] && echo PASS || echo FAIL)" \
  "${pmd_status}" "${panic_lines}" "${invalid_opcode_lines}" \
  "${control_pass_lines}" "${target_pass_lines}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt

[[ "${pmd_status}" == "0" && "${panic_lines}" == "0" && \
   "${invalid_opcode_lines}" == "0" && \
   "${control_pass_lines}" == "1" && "${target_pass_lines}" == "1" && \
   "${bank_conflicts}" == "0" ]]
