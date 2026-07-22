#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${DQ_DENSE_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/dq_native_ds_write_dense_probe"
ASM="${BUILD_DIR}/dq_native_ds_write_dense_probe.asm"
RUN_ROOT="${DQ_DENSE_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/dq_native_ds_dense_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${DQ_DENSE_PMD_TIMEOUT:-300}"
NATIVE_F16_SCORE="${DQ_DENSE_NATIVE_F16_SCORE:-0}"
NATIVE_F16_LTS="${DQ_DENSE_NATIVE_F16_LTS:-0}"
REQUIRE_SEMANTIC_PASS="${DQ_DENSE_REQUIRE_SEMANTIC_PASS:-1}"

case "${NATIVE_F16_SCORE}:${NATIVE_F16_LTS}:${REQUIRE_SEMANTIC_PASS}" in
  0:0:0|0:0:1|1:0:0|1:0:1|1:1:0|1:1:1) ;;
  *) echo "invalid native score/lts/require-pass combination" >&2; exit 2 ;;
esac

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dq_native_ds_write_dense_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
EXTRA_CXXFLAGS="-DSHAOBO_DENSE_NATIVE_F16_SCORE=${NATIVE_F16_SCORE} -DSHAOBO_DENSE_NATIVE_F16_LTS=${NATIVE_F16_LTS}" \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex dq_native_ds_write_dense_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*matrix_load_/ && /bps/ && /lds/ { mls += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { writer += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { reader_n += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { reader_t += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*v_mmac_16x16x16_f16/ { mmac_f16_out += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate mls=%d writer=%d reader_n=%d reader_t=%d mmac=%d mmac_f16_out=%d scalar_read=%d permute=%d permlane=%d\n",
           mls, writer, reader_n, reader_t, mmac, mmac_f16_out, scalar_read, permute, permlane)
    if (mls < 4 || writer < 1 || reader_n < 1 || reader_t < 1 || mmac < 12 ||
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
  python3 "${run_py}" -c "${GPU_CHIP}" -m m5out \
  -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
semantic_pass="$(grep -c 'dense_native_ds_final .* pass=1' pmd_stdout.log || true)"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${panic_lines}" == "0" &&
      "${stats_found}" -gt 0 && "${bank_conflicts}" == "0" ]]; then
  transport=PASS
else
  transport=FAIL
fi
semantic=FAIL
[[ "${semantic_pass}" -gt 0 ]] && semantic=PASS
source_name=f32_ds_downcast
[[ "${NATIVE_F16_SCORE}" == "1" ]] && source_name=f16_mmac_score

grep -E '^dense_native_ds |^dense_native_ds_final' pmd_stdout.log | tee result.txt || true
printf 'dq_native_ds_dense transport=%s semantic=%s source=%s lts=%s pmd_status=%s panic=%s stats=%s bank=%s run=%s\n' \
  "${transport}" "${semantic}" "${source_name}" "${NATIVE_F16_LTS}" \
  "${pmd_status}" "${panic_lines}" "${stats_found}" "${bank_conflicts}" \
  "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${transport}" == "PASS" ]]
if [[ "${REQUIRE_SEMANTIC_PASS}" == "1" ]]; then
  [[ "${semantic}" == "PASS" ]]
fi
