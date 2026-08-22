#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${BUILD_DIR:-build/p_direct_lit_probe}"
BIN="${BIN:-${BUILD_DIR}/fused5_p_direct_lit_probe}"
ASM="${ASM:-${BUILD_DIR}/fused5_p_direct_lit_probe.asm}"
RUN_DIR="${RUN_DIR:-${SHAOBO_RUN_ROOT}/p_direct_lit_probe_$(date +%Y%m%d_%H%M%S)}"
PMD_TIMEOUT="${PMD_TIMEOUT:-300}"
CLANG="${SHAOBO_COMPILER_ROOT}/opt/rocm-6.3.3/llvm/bin/clang++"

mkdir -p "${BUILD_DIR}"
"${CLANG}" -x hip --offload-arch=gfx946 -Iinclude -std=c++17 -O2 \
  probes/fused5_p_direct_lit_probe.cpp -o "${BIN}"
"${CLANG}" -x hip --offload-arch=gfx946 -Iinclude -std=c++17 -O2 -S \
  probes/fused5_p_direct_lit_probe.cpp -o "${ASM}"

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex fused5_p_direct_lit_probe_kernel \
  --max-private-segment 0 \
  --max-sgpr-spill 0 \
  --max-vgpr-spill 0

awk '
  /^[[:space:]]*matrix_load_/ && /bps/ && /lds/ { mls += 1 }
  /^[[:space:]]*ds_write_matrix_format/ { writer += 1 }
  /^[[:space:]]*ds_read_matrix_format/ { reader_n += 1 }
  /^[[:space:]]*ds_read_matrix_trans_format/ { reader_t += 1 }
  /^[[:space:]]*v_mmac_16x16x16_f16/ { mmac_f16_out += 1 }
  /^[[:space:]]*v_mmac_f32_16x16x16_f16/ { mmac_f32 += 1 }
  /^[[:space:]]*ds_read_b/ { scalar_read += 1 }
  /^[[:space:]]*ds_(m|b)permute/ { permute += 1 }
  /permlane/ { permlane += 1 }
  END {
    printf("asm_gate mls=%d writer=%d reader_n=%d reader_t=%d mmac_f16_out=%d mmac_f32=%d scalar_read=%d permute=%d permlane=%d\n",
           mls, writer, reader_n, reader_t, mmac_f16_out, mmac_f32,
           scalar_read, permute, permlane)
    if (mls < 3 || writer != 1 || reader_n < 2 || reader_t < 2 ||
        mmac_f16_out < 4 || mmac_f32 < 2 || scalar_read != 0 ||
        permute != 0 || permlane != 0) exit 1
  }
' "${ASM}"

mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 "${PMD_TIMEOUT}" \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out \
  -e "${ROOT}/${BIN}" >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
pass_lines="$(grep -c 'p_direct_lit .* pass=1' pmd_stdout.log || true)"
stats_file="$(find m5out -name stats.txt -print -quit)"
bank_conflict="$(grep -h 'ldsBankConflict' "${stats_file}" 2>/dev/null | awk '{sum += $2} END {print sum + 0}')"

if [[ "${pmd_status}" != "0" || "${panic_lines}" != "0" ||
      "${pass_lines}" != "1" || "${bank_conflict}" != "0" ]]; then
  printf 'p_direct_lit_probe_status=REJECT pmd=%s panic=%s pass=%s bank=%s run=%s\n' \
    "${pmd_status}" "${panic_lines}" "${pass_lines}" "${bank_conflict}" \
    "${RUN_DIR}" | tee result.txt
  exit 1
fi

printf 'p_direct_lit_probe_status=PASS bank=%s run=%s\n' \
  "${bank_conflict}" "${RUN_DIR}" | tee result.txt
