#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${FUSED5_1P3C_PROBE_BUILD_DIR:-build/probes/fused5_1p3c_native_handoff}"
BIN="${BUILD_DIR}/fused5_1p3c_native_handoff_probe"
ASM="${BUILD_DIR}/fused5_1p3c_native_handoff_probe.asm"
RUN_ROOT="${FUSED5_1P3C_PROBE_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/fused5_1p3c_native_handoff_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${FUSED5_1P3C_PROBE_TIMEOUT:-300}"

mkdir -p "${BUILD_DIR}"
TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/fused5_1p3c_native_handoff_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
SHAOBO_EXPLICIT_WDRA_INIT=1 ./build.sh | tee "${BUILD_DIR}.build.log"

if grep -qiE '(^|[^[:alnum:]_])warning:' "${BUILD_DIR}.build.log"; then
  echo "compiler warning gate failed" >&2
  exit 1
fi

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex fused5_1p3c_native_handoff_probe_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

awk '
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { write_matrix += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { read_trans += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { read_normal += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*global_atomic_add_f32/ { atomic += 1 }
  /^[[:space:]]*s_set_vgpr_size/ { resize += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate trap=%d write=%d trans=%d normal=%d mmac=%d atomic=%d resize=%d scalar_read=%d permute=%d permlane=%d\n",
           trap, write_matrix, read_trans, read_normal, mmac, atomic, resize,
           scalar_read, permute, permlane)
    if (trap != 0 || write_matrix < 6 || read_trans < 24 ||
        read_normal < 6 || mmac < 96 || atomic < 6 || resize != 4 ||
        scalar_read != 0 || permute != 0 || permlane != 0) exit 1
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

semantic="$(grep -c 'fused5_1p3c_native_handoff config waves=16 roles=32/160/160/160 groups=3 lds_bytes=49152 dq_mismatches=0 dk_mismatches=0 pressure_mismatches=0 pass=1' pmd_stdout.log || true)"
panic="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning="$(grep -ciE 'warn:.*read vgpr.*before writing' pmd_stdout.log || true)"
bank=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
if [[ "${#stats_files[@]}" -gt 0 ]]; then
  bank="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi

status=FAIL
if [[ "${pmd_status}" == 0 && "${semantic}" == 1 && "${panic}" == 0 &&
      "${vgpr_warning}" == 0 && "${#stats_files[@]}" -gt 0 &&
      "${bank}" == 0 ]]; then
  status=PASS
fi

grep '^fused5_1p3c_native_handoff ' pmd_stdout.log | tee result.txt || true
printf 'fused5_1p3c_native_handoff_status=%s pmd=%s semantic=%s panic=%s vgpr_warning=%s stats=%s bank=%s run=%s\n' \
  "${status}" "${pmd_status}" "${semantic}" "${panic}" \
  "${vgpr_warning}" "${#stats_files[@]}" "${bank}" "${RUN_DIR}" \
  | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${status}" == PASS ]]
