#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"
BUILD_DIR="${FUSED5_RAW_Q_DOUT_SPLIT_BUILD_DIR:-build/probes/fused5_raw_q_dout_split}"
BIN="${BUILD_DIR}/fused5_raw_q_dout_split_lifetime_probe"
ASM="${BUILD_DIR}/fused5_raw_q_dout_split_lifetime_probe.asm"
RUN_ROOT="${FUSED5_RAW_Q_DOUT_SPLIT_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/fused5_raw_q_dout_split_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${FUSED5_RAW_Q_DOUT_SPLIT_PMD_TIMEOUT:-600}"

mkdir -p "${BUILD_DIR}"
TARGET_GFX=946 BUILD_ASM=1 SRC=probes/fused5_raw_q_dout_split_lifetime_probe.cpp \
  BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
  SHAOBO_EXPLICIT_WDRA_INIT=1 ./build.sh | tee "${BUILD_DIR}.build.log"

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex fused5_raw_q_dout_split_lifetime_probe_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

awk '
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  /^[[:space:]]*matrix_load_32x32_b16/ { mls += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { ds += 1 }
  /^[[:space:]]*s_abarrier_init/ { init += 1 }
  /^[[:space:]]*s_abarrier_seq/ { seq += 1 }
  /^[[:space:]]*s_abarrier_arrive/ { arrive += 1 }
  /^[[:space:]]*s_abarrier_try_wait/ { wait += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate trap=%d mls=%d ds_trans=%d init=%d seq=%d arrive=%d wait=%d scalar_read=%d permute=%d permlane=%d\n",
           trap, mls, ds, init, seq, arrive, wait, scalar_read, permute, permlane)
    if (trap != 0 || mls < 6 || ds < 8 || init != 9 || seq < 4 ||
        arrive < 9 || wait < 8 || scalar_read != 0 || permute != 0 ||
        permlane != 0) exit 1
  }
' "${ASM_ABS}"

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"
cd "${RUN_DIR}"
run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then run_py="${PMD_PATH}/core/scripts/run.py"; fi
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" python3 "${run_py}" \
  -c "${GPU_CHIP}" -m m5out -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

semantic="$(grep -c 'fused5_raw_q_dout_split q_errors=0 dout_errors=0 pass=1' \
  pmd_stdout.log || true)"
panic="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning="$(grep -ciE 'warn:.*read vgpr.*before writing' pmd_stdout.log || true)"
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
bank=0
if [[ "${#stats_files[@]}" -gt 0 ]]; then
  bank="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == 0 && "${semantic}" == 1 && "${panic}" == 0 &&
      "${vgpr_warning}" == 0 && "${#stats_files[@]}" -gt 0 &&
      "${bank}" == 0 ]]; then
  status=PASS
else
  status=FAIL
fi
printf 'fused5_raw_q_dout_split_status=%s pmd=%s semantic=%s panic=%s vgpr_warning=%s stats=%s bank=%s run=%s\n' \
  "${status}" "${pmd_status}" "${semantic}" "${panic}" \
  "${vgpr_warning}" "${#stats_files[@]}" "${bank}" "${RUN_DIR}" | tee result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt
[[ "${status}" == PASS ]]
