#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${MLS_PAGE_PROBE_BUILD_DIR:-build/mls_dual_view_page_offset}"
BIN="${BUILD_DIR}/dkv_mls_dual_view_page_offset_probe"
ASM="${BUILD_DIR}/dkv_mls_dual_view_page_offset_probe.asm"
RUN_ROOT="${MLS_PAGE_PROBE_RUN_ROOT:-/zys/sb/probes}"
RUN_DIR="${RUN_ROOT}/mls_page_imm_$(date +%Y%m%d_%H%M%S)"

TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/dkv_mls_dual_view_page_offset_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" ./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" --symbol-regex dual_view_page_offset_probe

awk '
  /matrix_load_32x16_b16/ && /bps/ && /lds/ { bps += 1 }
  /ds_read_matrix_trans_format/ { trans += 1 }
  /ds_read_matrix_format/ { normal += 1 }
  /s_trap/ { trap += 1 }
  END {
    printf("asm bps=%d trans_read=%d normal_read=%d trap=%d\n",
           bps, trans, normal, trap)
    if (bps < 4 || trans < 4 || normal < 4 || trap != 0) exit 1
  }
' "${ASM}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 180 \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out -e "${ROOT}/${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
pass_lines="$(grep -c 'dkv_mls_dual_view_page_offset .* pass=1' pmd_stdout.log || true)"
bank_conflicts=0
if compgen -G 'm5out/0/*/stats.txt' >/dev/null; then
  bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' m5out/0/*/stats.txt)"
fi

if [[ "${pmd_status}" != 0 || "${panic_lines}" != 0 ||
      "${pass_lines}" != 1 || "${bank_conflicts}" != 0 ]]; then
  printf 'mls_page_offset_probe_status=FAIL pmd=%s pass=%s panic=%s bank=%s run=%s\n' \
    "${pmd_status}" "${pass_lines}" "${panic_lines}" "${bank_conflicts}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'dkv_mls_dual_view_page_offset' pmd_stdout.log | tee result.txt
printf 'mls_page_offset_probe_status=PASS bank=%s run=%s\n' \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
