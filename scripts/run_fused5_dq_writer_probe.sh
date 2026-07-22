#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh
: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked config.ini}"

BUILD_DIR="${FUSED5_DQ_WRITER_BUILD_DIR:-build/probes/fused5_dq_writer}"
BIN="${BUILD_DIR}/fused5_dq_writer_probe"
ASM="${BUILD_DIR}/fused5_dq_writer_probe.asm"
RUN_ROOT="${FUSED5_DQ_WRITER_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/fused5_dq_writer_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BUILD_DIR}"
TARGET_GFX=946 BUILD_ASM=1 SRC=probes/fused5_dq_writer_probe.cpp \
  BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
  SHAOBO_EXPLICIT_WDRA_INIT=1 ./build.sh
python3 scripts/check_symbol_metadata_gate.py --asm "${ASM}" \
  --symbol-regex fused5_dq_writer_probe_kernel --max-private-segment 0 \
  --max-sgpr-spill 0 --max-vgpr-spill 0

read_count="$(grep -Ec '^[[:space:]]*ds_read_b(128|64)' "${ASM}" || true)"
write_count="$(grep -Ec '^[[:space:]]*ds_write_b(128|64)' "${ASM}" || true)"
scalar_count="$(grep -Ec '^[[:space:]]*ds_(read|write)_b32' "${ASM}" || true)"
printf 'asm_gate vector_read=%s vector_write=%s scalar=%s\n' \
  "${read_count}" "${write_count}" "${scalar_count}"
[[ "${read_count}" -ge 2 && "${write_count}" -ge 2 && "${scalar_count}" == 0 ]]

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
BIN_ABS="$(realpath "${BIN}")"
run_py="${PMD_PATH}/scripts/run.py"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 300 python3 "${run_py}" -c "${GPU_CHIP}" -m m5out \
  -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e

semantic="$(grep -c 'fused5_dq_writer mismatches=0 pass=1' pmd_stdout.log || true)"
panic="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning="$(grep -ciE 'read vgpr.*before writing' pmd_stdout.log || true)"
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
bank=0
if [[ "${#stats_files[@]}" -gt 0 ]]; then
  bank="$(awk '/ldsBankConflict/ {sum += $2} END {print sum + 0}' "${stats_files[@]}")"
fi
grep '^fused5_dq_writer ' pmd_stdout.log | tee result.txt || true
printf 'fused5_dq_writer_status pmd=%s semantic=%s panic=%s vgpr_warning=%s stats=%s bank=%s run=%s\n' \
  "${pmd_status}" "${semantic}" "${panic}" "${vgpr_warning}" "${#stats_files[@]}" "${bank}" \
  "${RUN_DIR}" | tee -a result.txt
[[ "${pmd_status}" == 0 && "${semantic}" == 1 && "${panic}" == 0 &&
   "${vgpr_warning}" == 0 &&
   "${#stats_files[@]}" -eq 1 && "${bank}" == 0 ]]
