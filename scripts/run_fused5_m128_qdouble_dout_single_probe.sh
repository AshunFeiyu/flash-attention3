#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${FUSED5_M128_QDOUBLE_BUILD_DIR:-build/probes/fused5_m128_qdouble_dout_single}"
BIN="${BUILD_DIR}/fused5_m128_qdouble_dout_single_probe"
ASM="${BUILD_DIR}/fused5_m128_qdouble_dout_single_probe.asm"
RUN_ROOT="${FUSED5_M128_QDOUBLE_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/fused5_m128_qdouble_dout_single_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${FUSED5_M128_QDOUBLE_PMD_TIMEOUT:-600}"

grep -Fq '__builtin_hcu_wdra_init(16, 204, 204, 88)' \
  probes/fused5_m128_qdouble_dout_single_probe.cpp
grep -Fq 'kLdsBytes == 128 * 1024' \
  probes/fused5_m128_qdouble_dout_single_probe.cpp
grep -Fq 'Bar::kResidentUsed, 12' \
  probes/fused5_m128_qdouble_dout_single_probe.cpp
grep -Fq 'Bar::kDoutDead, 8' \
  probes/fused5_m128_qdouble_dout_single_probe.cpp
grep -Fq 'Bar::kEpochDone, 12' \
  probes/fused5_m128_qdouble_dout_single_probe.cpp

mkdir -p "${BUILD_DIR}"
set +e
TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/fused5_m128_qdouble_dout_single_probe.cpp \
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
  --symbol-regex fused5_m128_qdouble_dout_single_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*s_trap([[:space:]]|$)/ { trap += 1 }
  /^[[:space:]]*matrix_load_32x32_b16/ && /bps lds/ { mls += 1 }
  /^[[:space:]]*ds_write_matrix_format/ && /element:2/ && /row:2/ && /col:1/ && / t[[:space:]]*$/ { write_t1_alt0 += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { read_trans += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { read_normal += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*s_set_vgpr_size/ { resize += 1 }
  /^[[:space:]]*s_abarrier_init/ { abar_init += 1 }
  /^[[:space:]]*s_abarrier_inv/ { abar_inv += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate trap=%d mls=%d write_t1_alt0=%d read_trans=%d read_normal=%d mmac=%d resize=%d abar_init=%d abar_inv=%d scalar_read=%d permute=%d permlane=%d\n",
           trap, mls, write_t1_alt0, read_trans, read_normal, mmac, resize,
           abar_init, abar_inv, scalar_read, permute, permlane)
    if (trap != 0 || mls < 24 || write_t1_alt0 < 16 || read_trans < 48 ||
        read_normal < 16 || mmac < 128 || resize != 4 || abar_init != 11 ||
        abar_inv != 11 || scalar_read != 0 || permute != 0 ||
        permlane != 0) exit 1
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

semantic_pass="$(grep -c 'fused5_m128_qdouble_dout_single q_before_mismatches=0 dout_mismatches=0 q_after_mismatches=0 normal_mismatches=0 trans_mismatches=0 pass=1' pmd_stdout.log || true)"
config_pass="$(grep -c 'fused5_m128_qdouble_dout_single config waves=16 generations=3 lds_bytes=131072 barriers=11 q_pages=2 dout_pages=1 resident_used=12 q_used=8 dout_dead=8 epoch_done=12 roles=16/204/204/88' pmd_stdout.log || true)"
panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning_lines="$(grep -ciE 'warn:.*read vgpr.*before writing' pmd_stdout.log || true)"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${semantic_pass}" == "1" &&
      "${config_pass}" == "1" && "${panic_lines}" == "0" &&
      "${vgpr_warning_lines}" == "0" && "${stats_found}" -gt 0 &&
      "${bank_conflicts}" == "0" ]]; then
  status=PASS
else
  status=FAIL
fi

grep -E '^fused5_m128_qdouble_dout_single ' pmd_stdout.log | tee result.txt || true
printf 'fused5_m128_qdouble_dout_single_probe_status=%s pmd_status=%s semantic=%s config=%s panic=%s vgpr_warnings=%s stats=%s bank=%s run=%s\n' \
  "${status}" "${pmd_status}" "${semantic_pass}" "${config_pass}" \
  "${panic_lines}" "${vgpr_warning_lines}" "${stats_found}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${status}" == "PASS" ]]
