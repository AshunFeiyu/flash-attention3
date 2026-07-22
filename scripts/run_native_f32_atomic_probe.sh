#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked config.ini}"
BUILD_DIR="${NATIVE_ATOMIC_BUILD_DIR:-build/native_f32_atomic_probe}"
RUN_DIR="${NATIVE_ATOMIC_RUN_DIR:-${SHAOBO_RUN_ROOT}/native_f32_atomic_$(date +%Y%m%d_%H%M%S)}"

TARGET_GFX=946 BUILD_ASM=1 SRC=probes/native_f32_atomic.cpp \
  BUILD_DIR="${BUILD_DIR}" BIN="${BUILD_DIR}/native_f32_atomic" \
  ASM="${BUILD_DIR}/native_f32_atomic.asm" SHAOBO_DISABLE_WDRA_FLAGS=1 \
  SHAOBO_EXPLICIT_WDRA_INIT=0 ./build.sh

atomic_count="$(grep -Ec '^[[:space:]]+global_atomic_add_f32[[:space:]]' \
  "${BUILD_DIR}/native_f32_atomic.asm" || true)"
cas_count="$(grep -Ec '^[[:space:]]+[^;]*atomic_cmpswap[^;]*$' \
  "${BUILD_DIR}/native_f32_atomic.asm" || true)"
[[ "${atomic_count}" -gt 0 && "${cas_count}" -eq 0 ]]

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"

run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi

cd "${RUN_DIR}"
export GPU_DFLAGS="['StatLog','MMUCheck']"
timeout --kill-after=5 300 python3 "${run_py}" -c "${GPU_CHIP}" \
  -m m5out -e "$(realpath "${ROOT}/${BUILD_DIR}/native_f32_atomic")" \
  >pmd_stdout.log 2>&1

grep 'native_f32_atomic values=' pmd_stdout.log | tee correctness.txt
grep -q 'mismatches=0 pass=1' correctness.txt
printf 'native_f32_atomic_status=PASS atomic=%s cmpswap=%s run=%s\n' \
  "${atomic_count}" "${cas_count}" "${RUN_DIR}" | tee result.txt
