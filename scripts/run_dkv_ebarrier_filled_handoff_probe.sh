#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env.sh

BUILD_DIR="${EBARRIER_FILLED_PROBE_BUILD_DIR:-build/ebarrier_filled_handoff}"
BIN="${BUILD_DIR}/dkv_ebarrier_filled_handoff_probe"
ASM="${BUILD_DIR}/dkv_ebarrier_filled_handoff_probe.asm"
RUN_ROOT="${EBARRIER_FILLED_PROBE_RUN_ROOT:-/zys/sb/probes}"
RUN_DIR="${RUN_ROOT}/dkv_ebarrier_filled_$(date +%Y%m%d_%H%M%S)"

TARGET_GFX=946 BUILD_ASM=1 \
SRC=probes/dkv_ebarrier_filled_handoff_probe.cpp \
BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" ./build.sh

python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" --symbol-regex dkv_filled_handoff_probe_kernel

awk '
  /matrix_load_32x16_b16/ && /bps/ && /lds/ { bps += 1 }
  /ds_read_matrix/ { read += 1 }
  /s_ebarrier_arrive/ { earrive += 1 }
  /s_ebarrier_sync/ { esync += 1 }
  /s_abarrier_seq/ { aseq += 1 }
  /s_trap/ { trap += 1 }
  END {
    printf("asm bps=%d ds_read_matrix=%d ebarrier_arrive=%d ebarrier_sync=%d abarrier_seq=%d trap=%d\n",
           bps, read, earrive, esync, aseq, trap)
    if (bps < 4 || read < 4 || earrive < 1 || esync < 3 ||
        aseq < 1 || trap != 0) exit 1
  }
' "${ASM}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
set +e
timeout --kill-after=5 240 \
  python3 "${PMD_PATH}/scripts/run.py" -c sb -m m5out -e "${ROOT}/${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e
cat pmd_stdout.log

panic_lines="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
pass_lines="$(grep -c 'dkv_ebarrier_filled_handoff_probe .* pass=1' pmd_stdout.log || true)"
bank_conflicts="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
  m5out/0/*/stats.txt)"

mapfile -t stats_files < <(printf '%s\n' m5out/0/*/stats.txt | sort)
if [[ "${#stats_files[@]}" != 2 ]]; then
  printf 'ebarrier_filled_probe_status=FAIL dispatch_stats=%s run=%s\n' \
    "${#stats_files[@]}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

kernel_ticks() {
  awk '
    /firstWaveStartTick/ { first = $2 }
    /lastWaveEndTick/ { last = $2 }
    END { print last - first }
  ' "$1"
}

reference_ticks="$(kernel_ticks "${stats_files[0]}")"
candidate_ticks="$(kernel_ticks "${stats_files[1]}")"
speedup_pct="$(awk -v base="${reference_ticks}" -v cand="${candidate_ticks}" \
  'BEGIN { printf "%.4f", (base - cand) * 100.0 / base }')"

if [[ "${pmd_status}" != 0 || "${panic_lines}" != 0 ||
      "${pass_lines}" != 1 || "${bank_conflicts}" != 0 ]]; then
  printf 'ebarrier_filled_probe_status=FAIL pmd=%s pass=%s panic=%s bank=%s run=%s\n' \
    "${pmd_status}" "${pass_lines}" "${panic_lines}" \
    "${bank_conflicts}" "${RUN_DIR}" | tee result.txt
  exit 1
fi

sha256sum "${ROOT}/${BIN}" "${ROOT}/${ASM}" | tee artifact_sha256.txt
grep 'dkv_ebarrier_filled_handoff_probe' pmd_stdout.log | tee result.txt
printf 'ebarrier_filled_probe_status=PASS reference_ticks=%s candidate_ticks=%s improvement_pct=%s bank=%s run=%s\n' \
  "${reference_ticks}" "${candidate_ticks}" "${speedup_pct}" \
  "${bank_conflicts}" "${RUN_DIR}" | tee -a result.txt
