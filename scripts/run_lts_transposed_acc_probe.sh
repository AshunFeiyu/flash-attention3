#!/usr/bin/env bash
# PR1: MMOP LTS bit probe runner (compile/ASM -> PMD dense oracle).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${BUILD_DIR:-build/lts_probe}"
BIN="${BIN:-${BUILD_DIR}/lts_transposed_acc_probe}"
ASM="${ASM:-${BUILD_DIR}/lts_transposed_acc_probe.asm}"
RUN_DIR="${RUN_DIR:-${SHAOBO_RUN_ROOT}/lts_probe_$(date +%Y%m%d_%H%M%S)}"
PMD_TIMEOUT="${PMD_TIMEOUT:-300}"

# Plain compile without build.sh's local-wave WDRA flags: the probe kernel
# has no WDRA shell, and local-wave makes the allocator emit s_set_vgpr_size
# which PMD rejects for non-WDRA launches.
CLANG="${SHAOBO_COMPILER_ROOT}/opt/rocm-6.3.3/llvm/bin/clang++"
mkdir -p "${BUILD_DIR}"
"${CLANG}" -x hip --offload-arch=gfx946 -Iinclude -std=c++17 -O2 \
  probes/lts_transposed_acc_probe.cpp -o "${BIN}"
"${CLANG}" -x hip --offload-arch=gfx946 -Iinclude -std=c++17 -O2 -S \
  probes/lts_transposed_acc_probe.cpp -o "${ASM}"

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex lts_probe_kernel

# Evidence layer 1: both v_mmac sites must exist and their encodings must
# differ (the LTS bit flips); no real s_trap may appear.
mmac_lines="$(grep -c 'v_mmac_f32_16x16x16_f16' "${ASM}" || true)"
trap_lines="$(grep -cE '^\s*s_trap' "${ASM}" || true)"
printf 'asm v_mmac=%d s_trap=%d\n' "${mmac_lines}" "${trap_lines}"
[[ "${mmac_lines}" -ge 2 ]]
[[ "${trap_lines}" == "0" ]]

mkdir -p "${RUN_DIR}/m5out"
# PMD HEAD1694 cannot generate a fresh config (ASTCA num_phase); the locked
# seed must be staged as m5out/config.ini before run.py starts.
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out -e "${ROOT}/${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|unimplemented' pmd_stdout.log || true)"
result_lines="$(grep -c 'lts_probe ' pmd_stdout.log || true)"
verdict="$(grep -oE 'verdict=[A-Z_]+' pmd_stdout.log | head -1 || true)"

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${result_lines}" != "1" ]]; then
  printf 'lts_probe_status=FAIL pmd=%s panic=%s result_lines=%s run=%s\n' \
    "${pmd_status}" "${panic_lines}" "${result_lines}" "${RUN_DIR}" \
    | tee result.txt
  exit 1
fi

printf 'lts_probe_status=PASS %s run=%s\n' "${verdict:-none}" "${RUN_DIR}" \
  | tee result.txt
