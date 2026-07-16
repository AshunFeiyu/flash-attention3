#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${PDS_BUILD_DIR:-build/pds_split64}"
BIN="${BUILD_DIR}/dkv_pds_split64_probe"
ASM="${BUILD_DIR}/dkv_pds_split64_probe.asm"
RUN_ROOT="${PDS_RUN_ROOT:-/zys/shaobo/runs}"
RUN_DIR="${RUN_ROOT}/dkv_pds_split64_probe_$(date +%Y%m%d_%H%M%S)"
DEFAULT_FLAGS="-DPDS_PROBE_HIGH_PRESSURE=1 -DPDS_PROBE_SPLIT_OUTPUT_READERS=1 -DPDS_PROBE_ACCUM_F32X4=16 -DPDS_PROBE_READER_VGPRS=160"
PROBE_FLAGS="${PDS_PROBE_FLAGS:-${DEFAULT_FLAGS}}"
EXPECT_TRANS_READ="${PDS_EXPECT_TRANS_READ:-1}"

TARGET_GFX=946 \
BUILD_ASM=1 \
SRC=probes/dkv_pds_cross_wave_probe.cpp \
BUILD_DIR="${BUILD_DIR}" \
BIN="${BIN}" \
ASM="${ASM}" \
EXTRA_CXXFLAGS="${PROBE_FLAGS}" \
./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex dkv_pds_cross_wave_probe_kernel

awk -v expect_trans_read="${EXPECT_TRANS_READ}" '
  /s_trap/ { trap += 1 }
  /ds_write_matrix/ { write += 1 }
  /ds_read_matrix/ { read += 1 }
  /s_set_vgpr_size/ { resize += 1 }
  END {
    printf("asm trap=%d ds_write=%d ds_read_trans=%d set_vgpr=%d\n",
           trap, write, read, resize)
    if (trap != 0 || write == 0 ||
        (expect_trans_read != 0 && read == 0) || resize != 4) exit 1
  }
' "${ASM}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out -e "${ROOT}/${BIN}" \
  2>&1 | tee pmd_stdout.log

pass_lines="$(grep -cE 'sync=(abarrier|cta) .*mismatches=0 .*pass=1' pmd_stdout.log)"
panic_lines="$(grep -ciE 'panic:|fatal:' pmd_stdout.log || true)"
vgpr_warning_lines="$(grep -ciE 'warn: read vgpr[0-9]+ before writing' pmd_stdout.log || true)"
bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
  m5out/0/*/stats.txt)"

if [[ "${pass_lines}" != "2" || "${panic_lines}" != "0" ||
      "${bank_conflicts}" != "0" ]]; then
  printf 'pds_split64_probe_status=FAIL pass_lines=%s panic=%s bank=%s\n' \
    "${pass_lines}" "${panic_lines}" "${bank_conflicts}"
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
printf 'pds_split64_probe_status=PASS pass_lines=%s panic=%s bank=%s vgpr_warnings=%s run=%s\n' \
  "${pass_lines}" "${panic_lines}" "${bank_conflicts}" \
  "${vgpr_warning_lines}" "${RUN_DIR}" | tee result.txt
