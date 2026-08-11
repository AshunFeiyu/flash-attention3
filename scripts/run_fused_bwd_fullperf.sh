#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked config.ini}"
BUILD_DIR="${FUSED5_BUILD_DIR:-build/fused5_owner_rewrite}"
BIN="$(realpath "${BUILD_DIR}/fused_bwd_correctness")"
S="${S:-1024}"
CAUSAL="${CAUSAL:-1}"
RUN_ROOT="${FUSED5_FULLPERF_ROOT:-${SHAOBO_RUN_ROOT}}"
RUN_DIR="${RUN_ROOT}/5gemm_owner_s${S}_c${CAUSAL}_fullperf_$(date +%Y%m%d_%H%M%S)"
HELPER="${HSA_TOOLS_LIB:-/opt/rocm-6.3.3/lib/xprofiler/libperf_gen_helper.so}"
TIMEOUT="${FUSED5_FULLPERF_TIMEOUT:-1800}"
EXPECTED_DISPATCHES="${FUSED5_EXPECT_DISPATCHES:-2}"

[[ -x "${BIN}" ]]
[[ -r "${HELPER}" ]]
mkdir -p "${RUN_DIR}/m5out"
cp "${PMD_CONFIG_SEED}" "${RUN_DIR}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${RUN_DIR}/"

run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi

cd "${RUN_DIR}"
export GPU_DFLAGS="['StatLog','SQAbar','SQEbar','MMUCheck','TT','Perf']"
export HSA_TOOLS_LIB="${HELPER}"
set +e
S="${S}" CAUSAL="${CAUSAL}" timeout --kill-after=5 "${TIMEOUT}" \
  python3 "${run_py}" -c "${GPU_CHIP}" -m m5out -e "${BIN}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e

grep -E 'fused5_correctness|fused5_correctness_final' pmd_stdout.log \
  | tee correctness.txt || true
semantic="$(grep -c "fused5_correctness_final S=${S} D=128 causal=${CAUSAL} pass=1" pmd_stdout.log || true)"
panic="$(grep -ciE 'panic:|fatal:|not init or has been freed' pmd_stdout.log || true)"
vgpr_warning="$(grep -ciE 'read vgpr.*before writing' pmd_stdout.log || true)"
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c | sort -V)
mapfile -t perf_files < <(find . -type f -name '*.perf' -size +0c)
bank=0

if [[ "${#stats_files[@]}" -eq "${EXPECTED_DISPATCHES}" ]]; then
  : > stats_summary.txt
  for index in "${!stats_files[@]}"; do
    printf 'dispatch=%s stats=%s\n' "$((index + 1))" "${stats_files[index]}" \
      | tee -a stats_summary.txt
    python3 "${ROOT}/scripts/parse_fused_bwd_stats.py" "${stats_files[index]}" \
      | tee -a stats_summary.txt
  done
  bank="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' "${stats_files[@]}")"
fi
printf 'fullperf_status pmd=%s semantic=%s panic=%s vgpr_warning=%s stats=%s bank=%s perf=%s run=%s\n' \
  "${pmd_status}" "${semantic}" "${panic}" "${vgpr_warning}" "${#stats_files[@]}" \
  "${bank}" "${#perf_files[@]}" "${RUN_DIR}" | tee result.txt
printf '%s\n' "${perf_files[@]}" | tee perf_files.txt

[[ "${pmd_status}" == 0 && "${semantic}" == 1 && "${panic}" == 0 &&
   "${vgpr_warning}" == 0 &&
   "${#stats_files[@]}" == "${EXPECTED_DISPATCHES}" &&
   "${bank}" == 0 &&
   "${#perf_files[@]}" -ge "${EXPECTED_DISPATCHES}" ]]
