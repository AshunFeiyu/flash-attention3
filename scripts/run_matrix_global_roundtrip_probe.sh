#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

BUILD_DIR="${GLOBAL_ROUNDTRIP_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/matrix_global_roundtrip_probe"
ASM="${BUILD_DIR}/matrix_global_roundtrip_probe.asm"
RUN_ROOT="${GLOBAL_ROUNDTRIP_RUN_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
RUN_DIR="${RUN_ROOT}/matrix_global_roundtrip_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${GLOBAL_ROUNDTRIP_PMD_TIMEOUT:-600}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/matrix_global_roundtrip_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 \
SHAOBO_EXPLICIT_WDRA_INIT=0 \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex matrix_global_roundtrip_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*matrix_load_32x16_b16/ && /bps/ && /lds/ { mls += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { reader_n += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { reader_t += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { writer += 1 }
  /^[[:space:]]*matrix_store_32x16_b16/ { store += 1 }
  /^[[:space:]]*v_mmac_/ { mmac += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate mls=%d reader_n=%d reader_t=%d writer=%d store=%d mmac=%d scalar_read=%d permute=%d permlane=%d\n",
           mls, reader_n, reader_t, writer, store, mmac, scalar_read, permute,
           permlane)
    if (mls != 4 || reader_n != 2 || reader_t != 1 || writer != 4 ||
        store != 8 || mmac != 0 || scalar_read != 0 || permute != 0 ||
        permlane != 0) exit 1
  }
' "${ASM_ABS}"

group_segment="$(awk '/\.group_segment_fixed_size:/ { print $2; exit }' "${ASM_ABS}")"
group_segment="${group_segment:-0}"
printf 'lds_gate group_segment=%s max=131072\n' "${group_segment}"
[[ "${group_segment}" -le 131072 ]]

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
complete="$(grep -c 'global_roundtrip_complete=1' pmd_stdout.log || true)"
exact_chain="$(sed -n 's/.*exact_chain_pairs=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
exact_direct="$(sed -n 's/.*exact_direct_pairs=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
permutation_chain="$(sed -n 's/.*permutation_chain_pairs=\([0-9][0-9]*\).*/\1/p' pmd_stdout.log | tail -1)"
exact_chain="${exact_chain:-0}"
exact_direct="${exact_direct:-0}"
permutation_chain="${permutation_chain:-0}"
bank_conflicts=0
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${panic_lines}" == "0" &&
      "${complete}" -gt 0 && "${stats_found}" -gt 0 &&
      "${bank_conflicts}" == "0" ]]; then
  transport=PASS
else
  transport=FAIL
fi
if [[ "${exact_chain}" -gt 0 ]]; then
  semantic=EXACT_CHAIN_PASS
elif [[ "${exact_direct}" -gt 0 ]]; then
  semantic=DIRECT_ONLY_PASS
elif [[ "${permutation_chain}" -gt 0 ]]; then
  semantic=CHAIN_PERMUTATION_ONLY
else
  semantic=NO_COMPLETE_CHAIN
fi

grep -E '^global_roundtrip_(chain|direct|complete)' pmd_stdout.log \
  | tee result.txt || true
printf 'matrix_global_roundtrip transport=%s semantic=%s exact_chain=%s exact_direct=%s permutation_chain=%s pmd_status=%s panic=%s stats=%s bank=%s run=%s\n' \
  "${transport}" "${semantic}" "${exact_chain}" "${exact_direct}" \
  "${permutation_chain}" "${pmd_status}" "${panic_lines}" \
  "${stats_found}" "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${transport}" == "PASS" ]]
