#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${FUSED5_DS_CONVEYOR_BUILD_DIR:-build/probes/fused5_ds_conveyor}"
BIN="${BUILD_DIR}/fused5_ds_conveyor_probe"
ASM="${BUILD_DIR}/fused5_ds_conveyor_probe.asm"
RUN_ROOT="${FUSED5_DS_CONVEYOR_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/fused5_ds_conveyor_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${FUSED5_DS_CONVEYOR_PMD_TIMEOUT:-300}"

grep -Fq 'dense_value' probes/fused5_ds_conveyor_probe.cpp
grep -Fq 'source_slot_qk' probes/fused5_ds_conveyor_probe.cpp
grep -Fq 'view_mismatches' probes/fused5_ds_conveyor_probe.cpp
grep -Fq 'output_mismatches' probes/fused5_ds_conveyor_probe.cpp
! grep -Fq 'kExpectedReader' probes/fused5_ds_conveyor_probe.cpp

mkdir -p "${BUILD_DIR}"
set +e
TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/fused5_ds_conveyor_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
SHAOBO_EXPLICIT_WDRA_INIT=1 \
./build.sh >"${BUILD_DIR}.build.log" 2>&1
build_status="$?"
set -e
cat "${BUILD_DIR}.build.log"
[[ "${build_status}" == "0" ]]

if grep -qiE '(^|[^[:alnum:]_])warning:' "${BUILD_DIR}.build.log"; then
  echo "compiler warning gate failed" >&2
  exit 1
fi

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex fused5_ds_conveyor_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  # The assembler spells the verified t1/alt0 tuple as a trailing bare "t";
  # alt0 is omitted rather than printed as alt:0x0.
  /^[[:space:]]*ds_write_matrix_format/ && /element:2/ && /row:2/ && /col:1/ && / t[[:space:]]*$/ { write_t1_alt0 += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { read_trans += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*s_set_vgpr_size/ { resize += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate trap=%d write_t1_alt0=%d read_trans=%d mmac=%d resize=%d scalar_read=%d permute=%d permlane=%d\n",
           trap, write_t1_alt0, read_trans, mmac, resize, scalar_read, permute, permlane)
    if (trap != 0 || write_t1_alt0 < 1 || read_trans < 2 || mmac < 8 ||
        resize != 3 || scalar_read != 0 || permute != 0 || permlane != 0) exit 1
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
  python3 "${run_py}" -c "${GPU_CHIP}" -m m5out -e "${BIN_ABS}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

semantic_pass="$(grep -c 'fused5_ds_conveyor view_mismatches=0 output_mismatches=0 pressure_mismatches=0 pass=1' pmd_stdout.log || true)"
dense_pass="$(grep -c 'fused5_ds_conveyor config .*dense=1 ' pmd_stdout.log || true)"
panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning_lines="$(grep -ciE 'warn:.*read vgpr.*before writing' pmd_stdout.log || true)"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${semantic_pass}" == "1" &&
      "${dense_pass}" == "1" &&
      "${panic_lines}" == "0" && "${vgpr_warning_lines}" == "0" &&
      "${stats_found}" -gt 0 && "${bank_conflicts}" == "0" ]]; then
  status=PASS
else
  status=FAIL
fi

grep -E '^fused5_ds_conveyor ' pmd_stdout.log | tee result.txt || true
printf 'fused5_ds_conveyor_probe_status=%s pmd_status=%s semantic=%s dense=%s panic=%s vgpr_warnings=%s stats=%s bank=%s run=%s\n' \
  "${status}" "${pmd_status}" "${semantic_pass}" "${dense_pass}" "${panic_lines}" \
  "${vgpr_warning_lines}" "${stats_found}" "${bank_conflicts}" "${RUN_DIR}" \
  | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${status}" == "PASS" ]]
