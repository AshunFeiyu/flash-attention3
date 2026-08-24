#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"

export GPU_CHIP="${GPU_CHIP:-sb}"
export GPU_ARGS="${GPU_ARGS:-['--SQCIPfLines=7']}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-6.3.3}"
export SHAOBO_PMD_ROOT="${SHAOBO_PMD_ROOT:-/zys/shaobo/toolchains/pmd_20260824}"
export PMD_PATH="${PMD_PATH:-${SHAOBO_PMD_ROOT}/core}"
export SOC_PATH="${SOC_PATH:-${SHAOBO_PMD_ROOT}/soc}"
export PATH="${SHAOBO_PMD_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SOC_PATH}/libs:${SHAOBO_PMD_ROOT}/lib:${PMD_PATH}:${LD_LIBRARY_PATH:-}"
export RPY_LIB_PATH="${ROCM_PATH}/lib:${SHAOBO_PMD_ROOT}/lib:${PMD_PATH}:${SOC_PATH}:${SOC_PATH}/libs"

BUILD_DIR="${FWD_NATIVE_WRITER_BUILD_DIR:-build/probes}"
BIN="${BUILD_DIR}/fwd_native_mmac_writer_chain_probe"
ASM="${BUILD_DIR}/fwd_native_mmac_writer_chain_probe.asm"
RUN_ROOT="${FWD_NATIVE_WRITER_RUN_ROOT:-/zys/sb/fwd_native_mmac_writer_chain_probe}"
RUN_DIR="${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
PMD_TIMEOUT="${FWD_NATIVE_WRITER_PMD_TIMEOUT:-900}"

TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/fwd_native_mmac_writer_chain_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 \
./build.sh

BIN_ABS="$(realpath "${BIN}")"
ASM_ABS="$(realpath "${ASM}")"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM_ABS}" \
  --symbol-regex fwd_native_mmac_writer_chain_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

awk '
  /^[[:space:]]*matrix_load_32x32_b16/ && /bps/ && /lds/ { mls += 1 }
  /^[[:space:]]*ds_read_matrix/ { read += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { writer += 1 }
  /^[[:space:]]*matrix_store_32x16_b16/ { store += 1 }
  /^[[:space:]]*v_mmac_f32_16x16x16_f16/ { mmac += 1 }
  /^[[:space:]]*v_cvt_pk_f16_f32/ { cvt_pk += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate mls=%d read=%d writer=%d store=%d mmac=%d cvt_pk=%d scalar_read=%d permute=%d permlane=%d\n",
           mls, read, writer, store, mmac, cvt_pk, scalar_read, permute, permlane)
    if (mls < 3 || read < 5 || writer < 2 || store < 4 ||
        mmac < 8 || cvt_pk < 8 || scalar_read != 0 ||
        permute != 0 || permlane != 0) exit 1
  }
' "${ASM_ABS}"

mkdir -p "${RUN_DIR}/m5out"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"
cd "${RUN_DIR}"

stub_root="/tmp/shaobo_runpy_stub_${BASHPID:-$$}"
mkdir -p "${stub_root}"
printf 'def setproctitle(title):\n    return None\n\ndef getproctitle():\n    return "pmd"\n' \
  >"${stub_root}/setproctitle.py"
export PYTHONPATH="${stub_root}:${PYTHONPATH:-}"

set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c "${GPU_CHIP}" -m m5out \
  -e "${BIN_ABS}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
semantic_pass="$(grep -c 'fwd_native_mmac_writer_status=PASS' pmd_stdout.log || true)"
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c)
stats_found="${#stats_files[@]}"
bank_conflicts=0
if [[ "${stats_found}" -gt 0 ]]; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi

if [[ "${pmd_status}" == "0" && "${panic_lines}" == "0" &&
      "${semantic_pass}" -gt 0 && "${stats_found}" -gt 0 ]]; then
  status=PASS
else
  status=FAIL
fi

sed -E $'s/\\x1B\\[[0-9;]*[mK]//g' pmd_stdout.log |
  grep -E '^fwd_native_mmac_writer' | tee result.txt || true
printf 'fwd_native_mmac_writer_probe status=%s pmd_status=%s panic=%s stats=%s bank=%s run=%s\n' \
  "${status}" "${pmd_status}" "${panic_lines}" "${stats_found}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
sha256sum "${BIN_ABS}" "${ASM_ABS}" | tee artifact_sha256.txt

[[ "${status}" == "PASS" ]]
