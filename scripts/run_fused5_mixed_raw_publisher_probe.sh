#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${FUSED5_MIXED_PUBLISHER_BUILD_DIR:-build/probes/fused5_mixed_raw_publisher}"
BIN="${BUILD_DIR}/fused5_mixed_raw_publisher_probe"
ASM="${BUILD_DIR}/fused5_mixed_raw_publisher_probe.asm"
RUN_ROOT="${FUSED5_MIXED_PUBLISHER_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/fused5_mixed_raw_publisher_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${FUSED5_MIXED_PUBLISHER_PMD_TIMEOUT:-300}"

grep -Fq 'publishers=4+4 consumers=8' probes/fused5_mixed_raw_publisher_probe.cpp || true
mkdir -p "${BUILD_DIR}"
set +e
TARGET_GFX=946 BUILD_ASM=1 \
  SRC=probes/fused5_mixed_raw_publisher_probe.cpp \
  BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
  SHAOBO_EXPLICIT_WDRA_INIT=1 ./build.sh >"${BUILD_DIR}.build.log" 2>&1
build_status="$?"
set -e
cat "${BUILD_DIR}.build.log"
[[ "${build_status}" == "0" ]]
if grep -qE ':[0-9]+:[0-9]+: warning:' "${BUILD_DIR}.build.log"; then
  echo "compiler warning gate failed" >&2
  exit 1
fi

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex fused5_mixed_raw_publisher_probe_kernel \
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
    if (trap != 0 || mls < 4 || ds < 8 || init != 3 || seq < 1 ||
        arrive < 3 || wait < 2 || scalar_read != 0 || permute != 0 ||
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
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${run_py}" -c "${GPU_CHIP}" -m m5out -e "${BIN_ABS}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

semantic_pass="$(grep -c 'fused5_mixed_raw_publisher q_errors=0 dout_errors=0 producer_done=8 dout_publisher_done=8 consumer_done=16 stats_pass=1 pass=1' pmd_stdout.log || true)"
panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning_lines="$(grep -ciE 'warn:.*read vgpr.*before writing' pmd_stdout.log || true)"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${semantic_pass}" == "1" &&
      "${panic_lines}" == "0" && "${vgpr_warning_lines}" == "0" &&
      "${stats_found}" -gt 0 && "${bank_conflicts}" == "0" ]]; then
  status=PASS
else
  status=FAIL
fi
grep -E '^fused5_mixed_raw_publisher ' pmd_stdout.log | tee result.txt || true
printf 'fused5_mixed_raw_publisher_probe_status=%s pmd_status=%s semantic=%s panic=%s vgpr_warnings=%s stats=%s bank=%s run=%s\n' \
  "${status}" "${pmd_status}" "${semantic_pass}" "${panic_lines}" \
  "${vgpr_warning_lines}" "${stats_found}" "${bank_conflicts}" "${RUN_DIR}" \
  | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt
[[ "${status}" == "PASS" ]]
